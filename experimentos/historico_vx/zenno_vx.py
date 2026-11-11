# Copia independiente de zenno_v9.py (R3), creada el 2026-09-06.
# VX-R1: contrato experimental definido en readme_vx.md.
"""
ZennoVX - Entorno SAC incremental para manipulador móvil.

Es una copia funcional de ``zenno_v5.py``: conserva dinámica, observaciones,
PID, Pure Pursuit, IK, gripper automático y criterios de transición/éxito. Los
únicos cambios son de recompensa:

  - shaping potencial denso en place, sin premio estacionario acumulable;
  - hitos de place pagados una sola vez;
  - penalización terminal cuando una fase no se completa;
  - mayor penalización temporal durante la fase 2.

V9 conserva la física y la observación 78D de V8, pero usa un único contrato
7D normalizado: los cinco canales del brazo son incrementales en P0, P1 y P2;
los dos canales restantes son correcciones residuales acotadas del PID en P1.
No carga ni importa el ambiente V8.
"""

import json
from pathlib import Path
from typing import List, Tuple, Set, Optional, Dict, Sequence

import numpy as np
import mujoco

from gymnasium import spaces, utils
from gymnasium.spaces import Box
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils import EzPickle

PKG_DIR = Path(__file__).resolve().parent
ASSET_ROOT = PKG_DIR / "assets"
MODEL_DIR = ASSET_ROOT / "Zenno_Model_v3"
MAPS_DIR_DEFAULT = ASSET_ROOT / "maps"
SCENE_XML = MODEL_DIR / "scene.xml"


# ======================= HELPERS PUROS =======================
def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    return np.array([cy, 0.0, 0.0, sy], dtype=np.float64)


def one_hot(idx: int, n: int) -> np.ndarray:
    v = np.zeros(int(n), dtype=np.float32)
    i = int(idx)
    if 0 <= i < int(n):
        v[i] = 1.0
    return v


DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}


class _ZennoReference(MujocoEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 100,
    }

    # ========================================================================
    # STATIC HELPERS
    # ========================================================================
    @staticmethod
    def _exp_shift_norm(d: float, e0: float, k: float) -> float:
        ed = float(np.exp(-float(k) * float(d)))
        denom = float(max(1e-9, 1.0 - float(e0)))
        r = (ed - float(e0)) / denom
        return float(np.clip(r, 0.0, 1.0))

    @staticmethod
    def _as_int(x) -> int:
        return int(np.asarray(x).item())

    # ========================================================================
    # INIT
    # ========================================================================
    def __init__(self, **kwargs):
        model_path = "/home/claudio/Documentos/Zenno_Model_v5/scene.xml"
        frame_skip = 5

        # Observación extendida: 78 dims
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(78,), dtype=np.float32
        )

        utils.EzPickle.__init__(self, **kwargs)
        MujocoEnv.__init__(
            self,
            model_path,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        # Inicialización modular
        self._init_action_space_and_arm()
        self._init_arm_lock_config()
        self._init_arm_initial_postures()
        self._init_offsets_and_gripper()
        self._init_base_residuals()
        self._init_reward_weights()
        self._init_phase0_params()
        self._init_phase1_params()
        self._init_phase2_params()
        self._init_episode_state()
        self._init_wheels()
        self._init_grasp_contacts()
        self._init_pedestal_collision_sets()
        self._init_gripper_collision_sets()
        self._init_base_dynamics_and_pid()
        self._init_nav_state()
        self._init_maps_and_obstacles()
        self._init_cube_mass_sampling()
        self._init_peak_tracking()
        self._init_debug()

        # OMPL opcional
        try:
            from ompl import base as ob  # noqa: F401
            from ompl import geometric as og  # noqa: F401
            self._ompl_ok = True
        except Exception:
            self._ompl_ok = False
            print("[WARN] OMPL no disponible: RRT* usará línea recta como fallback.")

        # Cargar rutas cacheadas si existen
        if self.LOAD_ROUTES_ON_START:
            self._load_map_routes_from_folder(self.MAPS_DIR, map_ids=[1, 2, 3, 4, 5])

        self._reset_debug_all_phases()

    # ------------------------------------------------------------------------
    # Sub-inicializadores
    # ------------------------------------------------------------------------
    def _init_action_space_and_arm(self):
        self.arm_act_names = (
            "motor_j1", "motor_j2", "motor_j3",
            "motor_j4", "motor_j5", "motor_j6",
        )
        self.arm_act_ids: List[int] = []
        for nm in self.arm_act_names:
            try:
                self.arm_act_ids.append(self.model.actuator(nm).id)
            except Exception as e:
                raise RuntimeError(f"No se encontró el actuador '{nm}' en el XML.") from e

        self.arm_joint_names = [
            "robot0:base_joint",
            "robot0:arm1_joint",
            "robot0:arm2_joint",
            "robot0:arm3_joint",
            "robot0:eef_joint",
        ]

        self.ARM_POS_LOW = np.array([-1.57, -1.23, -0.85, -1.5, -1.57], dtype=np.float32)
        self.ARM_POS_HIGH = np.array([1.57, 1.23, 1.00, 1.5, 1.57], dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.array([-1.57, -1.23, -0.85, -1.5, -1.57, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.57, 1.23, 1.00, 1.5, 1.57, 1.0, 1.0], dtype=np.float32),
            shape=(7,),
            dtype=np.float32,
        )
        self.arm_target = np.zeros(5, dtype=np.float32)

    def _init_arm_lock_config(self):
        self.LOCK_J1_J5_AFTER_LIFT = True
        self.LOCK_J1_VALUE = 0.0
        self.LOCK_J5_VALUE = 0.0
        self.J1J5_LOCK_MAX_DELTA = np.array([0.04, 0.04], dtype=np.float32)

        self.ARM_IDX_J1 = 0
        self.ARM_IDX_J5 = 4

        self.LIFT_LOCK_DELTA_Z = 0.02
        self._cube_z_grasp_ref: Optional[float] = None
        self._j1_j5_locked = False

    def _init_arm_initial_postures(self):
        self.init_arm_positions_stage0 = {
            "robot0:base_joint": 0.0,
            "robot0:arm1_joint": -0.90,
            "robot0:arm2_joint": -0.85,
            "robot0:arm3_joint": -1.5,
            "robot0:eef_joint": 0.0,
        }
        self.init_arm_positions_staged = self.init_arm_positions_stage0

        self.q_arm_init = self._arm_pose_dict_to_array(self.init_arm_positions_stage0)
        self.q_arm_home = self.q_arm_init.copy()
        self.q_arm_carry_ref = self.q_arm_init.copy()

    def _init_offsets_and_gripper(self):
        self.REACH_Z_OFFSET = 0.015

        self.grip_open = 0.0
        self.grip_close = 255.0
        self.grip_aid = self.arm_act_ids[5]
        self.dist_threshold = 0.028

        self.P2_AUTO_RELEASE = True
        self.P2_RETURN_HOME_AFTER_RELEASE = True

        self.PEN_OPEN_OUTSIDE = -6.0
        self.PEN_EARLY_RELEASE = -10.0

    def _init_base_residuals(self):
        # Residual de base reducido: el agente ya no puede dominar el PID.
        # La navegación base queda gobernada por PID/RRT*, y el residual actúa
        # solo como corrección fina.
        self.RES_TAU_LIN_MAX = 0.05
        self.RES_TAU_ANG_MAX = 0.10

        # ================================================================
        # Protección fase 1: base PID dominante
        # ================================================================
        # El PID puro ya completa la navegación; por tanto, durante fase 1
        # el residual de base del agente se desactiva por defecto para evitar
        # que una corrección aprendida saque a la base del rumbo y la deje
        # girando buscando el siguiente waypoint.
        #
        # Modos:
        #   "off"      : fase 1 usa PID puro en la base.
        #   "guarded"  : residual pequeño solo si no contradice al PID. Recomendado.
        #   "original" : comportamiento antiguo, residual completo dentro del corredor.
        self.P1_BASE_RESIDUAL_MODE = "guarded"

        # Límites si se usa modo "guarded".
        # En fase 1 el residual se limita todavía más. Con estos valores:
        #   lin: 0.15 * 0.20 = 3.0% del torque disponible
        #   ang: 0.25 * 0.12 = 3.0% del torque disponible
        # y se bloquea si contradice el giro pedido por el PID.
        self.P1_RES_U_LIN_MAX = 0.20
        self.P1_RES_U_ANG_MAX = 0.12
        self.P1_RES_SIGN_GUARD = True
        self.P1_RES_MIN_OMEGA_FOR_SIGN = 0.08

        # Diagnóstico del residual aplicado.
        self._last_residual_safe_corridor = False
        self._last_residual_block_reason = "init"
        self._last_u_lin_raw = 0.0
        self._last_u_ang_raw = 0.0
        self._last_u_lin_applied = 0.0
        self._last_u_ang_applied = 0.0

    def _init_reward_weights(self):
        # Costos comunes
        self.W_TIME = 0.0001
        self.W_EN_ARM = 0.0005
        self.W_EN_BASE = 0.0002
        self.W_TIME_P2 = 0.0002
        self.W_EN_ARM_P2 = 0.0012
        self.W_EN_BASE_P2 = 0.0000

        # Drop
        self.drop_height = 0.35
        self.drop_penalty = -20.0

        # Peak-tracking: "bonus" distribuido al completar cada fase
        # (reemplaza los antiguos complete_bonus_* discretos)
        self.W_PEAK_P0 = 50.0
        self.W_PEAK_P1 = 50.0
        self.W_PEAK_P2 = 60.0

        # Pequeña señal terminal residual, útil para cortar ambigüedad
        # en el último paso. Se mantiene acotada y << W_PEAK_*.
        self.BONUS_FINAL_P0 = 50.0
        self.BONUS_FINAL_P1 = 50.0
        self.BONUS_FINAL_P2 = 100.0

        # Reward instantáneo por fase (placeholder, se actualiza por fase)
        self.reward = 0.0

        # Subrewards (exposición para debug)
        self.r_reach = 0.0
        self.r_grip = 0.0
        self.r_lift = 0.0
        self.r_yaw = 0.0
        self.r_pos = 0.0
        self.r_track = 0.0

    def _init_phase0_params(self):
        # Pesos internos para componer Φ_0 (suman 1)
        self.p0_w_reach = 0.25
        self.p0_w_grip = 0.15
        self.p0_w_lift = 0.60

        # Dense shaping local (se suma a peak-tracking con peso bajo)
        self.W_DENSE_P0 = 0.50

        self.K_REACH_DIST = 2.0
        self.K_LIFT_DIST = 6.0

        # Baselines se calculan en reset_model
        self.d0_ee_cube: Optional[float] = None
        self._e0_ee_cube: Optional[float] = None
        self.d0_cube_sp: Optional[float] = None
        self._e0_cube_sp: Optional[float] = None

        # Umbral de éxito fase 0
        self.PHASE0_SUCCESS_DIST_SP = 0.05

    def _init_phase1_params(self):
        # Pesos internos para Φ_1
        self.p1_w_yaw = 0.10
        self.p1_w_pos = 0.10
        self.p1_w_track = 0.20
        self.p1_w_hold_sp = 0.60

        self.W_DENSE_P1 = 0.50

        self.HOLD_SP_SIGMA = 0.05
        self.SIGMA_POS = 0.05
        self.SIGMA_YAW = 0.25
        self.SIGMA_TRACK = 0.25

        # Postura en transporte
        self.W_POSTURE_P1 = 0.12
        self.K_POSTURE_P1 = 3.0
        self.POSTURE_W_JOINTS_P1 = np.array(
            [0.10, 0.30, 0.30, 0.25, 0.05], dtype=np.float32
        )
        self.W_POSTURE_VEL_P1 = 0.01

    def _init_phase2_params(self):
        # Geometría física
        self.PED_RADIUS = 0.10
        self.CUBE_HALF = 0.0225
        self.Z_PED_TOP = 0.50
        self.Z_CUBE_REST = 0.5225

        self.HOME_POS_TOL = 0.05
        self.HOME_VEL_TOL = 0.10

        # Criterio de éxito / estabilidad
        # Se mantiene el criterio original para no degradar el modelo que ya
        # completa fase 0 y fase 1. La robustez nueva se introduce como shaping
        # suave, no como restricciones duras adicionales.
        self.PLACE_STABLE_COUNT_MIN = 20

        # Release realista: se conservan las tolerancias del mejor modelo.
        # El fix anterior endurecía demasiado el release y terminó reduciendo
        # la tasa de éxito. Aquí solo se agrega reward/penalización suave.
        self.RELEASE_DIST_TOL_XY = 0.025
        self.RELEASE_ERR_Z_MAX = 0.035
        self.RELEASE_V_MAX = 0.060
        self.RELEASE_HYSTERESIS_STEPS = 3
        self._release_ready_count = 0

        # Shaping P2 conservador.
        # Se refuerza XY, pero sin cambiar la transición desde fase 1 ni
        # bloquear el release. El objetivo es reducir FAIL_P2_XY_NOT_CENTERED
        # sin perder lo que ya funciona en fase 0/fase 1.
        self.K_PLACE_XY = 14.0
        self.K_PLACE_Z = 15.0
        self.K_APPROACH = 8.0
        self.K_PLACE_STABILITY = 14.0

        self.APPROACH_TOP_Z_MIN = 0.050
        self.APPROACH_TOP_Z_MAX = 0.150

        # Pesos internos para Φ_2 (suman 1)
        self.p2_w_xy = 0.42
        self.p2_w_z = 0.18
        self.p2_w_approach = 0.20
        self.p2_w_contact = 0.20

        self.W_DENSE_P2 = 0.60

        self.W_HOLD_PLACED = 0.20  # reward constante por step mientras cubo está colocado

        # Shaping adicional suave para fase 2. Estos términos NO cambian el
        # criterio de éxito; solo orientan al agente hacia release más centrado
        # y colocación más estable.
        self.W_RELEASE_READY_DENSE = 0.16
        self.W_PLACE_STABILITY_P2 = 0.06

        # Penalización suave por intentar bajar cuando el cubo aún no está
        # suficientemente cerca del centro del pedestal. Es moderada para no
        # bloquear el aprendizaje ni inducir overfitting a un mapa.
        self.P2_SAFE_DESCENT_XY_TOL = 0.030   # 1.2x la tolerancia de release (0.025 m)
        self.P2_XY_GATE_ZERO = 0.070
        self.P2_XY_SCALE_FAR = 0.15
        self.P2_XY_SCALE_NEAR = 0.025
        self.P2_LOW_DESCENT_HEIGHT = 0.030
        self.W_LOW_DESCENT_P2 = 0.05

        # Penalizaciones de colisión con pedestal, incrementos moderados.
        self.W_PED_P2 = 0.12
        self.W_GRIPPER_PED = 0.28

        # Penalización por arrastre: cubo toca pedestal pero lejos del centro.
        self.W_DRAG_P2 = 0.55

        # Recompensa por evento de soltura correcta (puntual)
        self.W_RELEASE_EVENT = 0.30

        # Rate limit brazo en fase 2
        self.ARM_MAX_DELTA_P2 = np.array(
            [0.025, 0.025, 0.025, 0.025, 0.025], dtype=np.float32
        )

    def _init_episode_state(self):
        self.phase_state = 0
        self.MAX_STEPS_0 = 500
        # El mapa 3 contiene la ruta válida más larga. Este margen permite
        # completar su cierre estricto de 20 mm sin relajar la tolerancia.
        self.MAX_STEPS_1 = 2650
        self.MAX_STEPS_2 = 1000

        self._steps = 0
        self._phase_steps = 0

        self.cube_dropped = False
        self._printed_drop = False
        self._drop_penalty_applied = False

        self.base_phase = 0
        self._last_action_base = np.zeros(2, dtype=np.float32)
        self._printed_phase = {0: False, 1: False, 2: False}

        self._carry_latch = False
        self._released = False
        self._placed_success = False
        self._place_stable_count = 0
        self._prev_gripper_open_cmd = False
        self._p2_returning_home = False

        # Grasp
        self.gripping = False
        self._grasp_count = 0
        self.GRASP_COUNT_MIN = 3
        self.GRASP_FN_MIN = 0.8
        self.GRASP_REQUIRE_BOTH_SIDES = True

    def _init_wheels(self):
        self.wheel_left_names = ("tau_fl", "tau_cl", "tau_rl")
        self.wheel_right_names = ("tau_fr", "tau_cr", "tau_rr")
        self.wheel_left_ids: List[int] = []
        self.wheel_right_ids: List[int] = []

        for nm in self.wheel_left_names:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)
            if aid >= 0:
                self.wheel_left_ids.append(int(aid))
        for nm in self.wheel_right_names:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)
            if aid >= 0:
                self.wheel_right_ids.append(int(aid))

    def _init_grasp_contacts(self):
        cube_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        if cube_bid < 0:
            raise RuntimeError("No se encontró body 'cube' en el XML.")

        self._cube_geom_ids: Set[int] = set(
            gi for gi in range(self.model.ngeom)
            if int(self.model.geom_bodyid[gi]) == int(cube_bid)
        )
        if not self._cube_geom_ids:
            raise RuntimeError("No se encontraron geoms asociados al body 'cube'.")

        self._padL_geom_ids: Set[int] = set()
        self._padR_geom_ids: Set[int] = set()

        for nm in ("left_pad1", "left_pad2"):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, nm)
            if gid >= 0:
                self._padL_geom_ids.add(int(gid))
        for nm in ("right_pad1", "right_pad2"):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, nm)
            if gid >= 0:
                self._padR_geom_ids.add(int(gid))

        if (not self._padL_geom_ids) or (not self._padR_geom_ids):
            raise RuntimeError(
                f"No se pudieron identificar pads: L={self._padL_geom_ids}, "
                f"R={self._padR_geom_ids}."
            )

    def _init_base_dynamics_and_pid(self):
        self.M_BASE = 158.0
        self.J_BASE = 1.5
        self.C_BASE = 0.5
        self.B_BASE = 0.5
        self.R_WHEEL = 0.1
        self.W_TRACK = 0.699

        # ================================================================
        # PID BASE @ 10 ms — FAST_10MS_V2
        # ================================================================
        # Con timestep XML = 0.002 s y frame_skip = 5:
        # Ts_control = 0.010 s.
        #
        # Estos parámetros corresponden al perfil estable usado en:
        # baseline_visual_pid_fast10ms_tracking_plots_yaw_controlref.py.
        # No se modifica la lógica de recompensa; solo se actualiza el
        # controlador de base y su seguimiento geométrico con CTE.
        self.BASE_CTRL_DT_TARGET = 0.010

        self.Kp_vel, self.Ki_vel, self.Kd_vel = 55.0, 4.5, 0.05
        self.Kp_omega, self.Ki_omega, self.Kd_omega = 720.0, 28.0, 0.16

        # Anti-windup del PID incremental.
        self.ACC_LIN_MAX = 35.0
        self.ACC_ANG_MAX = 330.0

        # Deadband solo para integración, no para el error proporcional.
        self.VEL_ERR_DEADBAND = 0.003
        self.OMEGA_ERR_DEADBAND = 0.004

        # Rate limit de referencias para control discreto a 10 ms.
        self.USE_REF_RATE_LIMIT = True
        self.V_DES_RATE_MAX = 4.0
        self.OMEGA_DES_RATE_MAX = 14.0

        # Descarga del acumulador angular cuando cambia el sentido de giro.
        self.RESET_ANG_ACC_ON_SIGN_FLIP = True
        self.OMEGA_SIGN_RESET_THRESH = 0.10

        self._v_des_f = 0.0
        self._omega_des_f = 0.0
        self._omega_des_prev = 0.0

        self.V_MAX = 0.68
        self.OMEGA_MAX = 0.95
        self.V_GAIN = 2.8
        self.OMEGA_GAIN = 3.6
        self.HEADING_K = 1.0

        # ================================================================
        # Seguimiento de ruta suave hacia el goal
        # ================================================================
        # La ruta ya no se genera hacia un candidato orbital aleatorio. Se
        # calcula un punto de parada sobre la línea base->goal, exactamente a
        # FINAL_GOAL_RADIUS_STOP del goal, y el RRT* solo se usa para sortear
        # obstáculos antes de suavizar la polilínea resultante.
        self.USE_DIRECT_GOAL_PLANNER = True
        self.USE_INITIAL_ALIGN = True

        # Separación y lookahead mayores para evitar que la base intente
        # reorientarse contra cada waypoint individual.
        self.WAYPOINT_TOL = 0.14
        self.LOOKAHEAD_DIST = 0.72
        self.WAYPOINT_SPACING = 0.08

        # Suavizado geométrico post-RRT*: shortcut + Chaikin + remuestreo.
        # PATH_MIN_TURN_SEG_LEN elimina microsegmentos que producen cambios
        # bruscos de referencia angular.
        self.PATH_SHORTCUT_ITERS = 120
        self.PATH_CHAIKIN_ITERS = 3
        self.PATH_MIN_TURN_SEG_LEN = 0.08
        self.PATH_SMOOTH_COLLISION_CHECK = True
        self.PATH_USE_SPLINE = True
        self.PATH_SPLINE_SAMPLES_PER_SEG = 10
        self.PATH_TERMINAL_TANGENT_TOL = 0.12


        # Corrección lateral CTE respecto a la polilínea RRT*/órbita.
        self.CTE_GAIN = 1.80
        self.CTE_CORR_MAX = 0.24

        self.YAW_ALIGN_EXIT = 0.16
        self.OMEGA_STOP_THRESH = 0.05
        self.MIN_OMEGA_ALIGN = 0.14

        self.V_CURV = 0.72
        self.K_CURV = 1.5

        # Reducción de avance durante tracking si la base apunta mal.
        self.YAW_SLOW_1 = 0.35
        self.YAW_SLOW_2 = 0.65
        self.YAW_STOP = 1.10
        self.YAW_SLOW_GAIN_1 = 0.85
        self.YAW_SLOW_GAIN_2 = 0.55

        # Atenuación de velocidad por error lateral.
        self.CTE_SLOW_1 = 0.04
        self.CTE_SLOW_2 = 0.065
        self.CTE_SLOW_GAIN_1 = 0.78
        self.CTE_SLOW_GAIN_2 = 0.58

        # Rotación rápida exclusiva para alineaciones inicial/final.
        # No altera el seguimiento normal de trayectoria.
        self.INIT_ALIGN_OMEGA_MAX = 1.10
        self.INIT_ALIGN_OMEGA_GAIN = 4.50
        self.INIT_ALIGN_OMEGA_RATE_MAX = 18.0
        self.INIT_ALIGN_ACC_ANG_MAX = 380.0
        self.INIT_ALIGN_MIN_OMEGA = 0.18

        # Alineación final rápida con hard stop al llegar al umbral.
        # Estrategia: gira rápido hasta alcanzar la tolerancia angular y,
        # al entrar en el umbral, corta la velocidad con hard stop para pasar
        # inmediatamente a la aproximación radial.
        self.FINAL_ALIGN_OMEGA_MAX = 1.45
        self.FINAL_ALIGN_OMEGA_GAIN = 6.50
        self.FINAL_ALIGN_OMEGA_RATE_MAX = 40.0
        self.FINAL_ALIGN_ACC_ANG_MAX = 520.0

        # Sin velocidad mínima forzada cerca del objetivo. La captura se hace
        # por umbral + hard stop, evitando esperas por velocidad angular baja.
        self.FINAL_ALIGN_MIN_OMEGA = 0.0

        # 0.035 rad ≈ 2.0 grados. Si necesitas más precisión: 0.025 rad.
        self.FINAL_ALIGN_HEADING_TOL = 0.1

        # Parámetros mantenidos por compatibilidad, pero el modo hard stop no
        # espera omega_z baja ni varios pasos estables.
        self.FINAL_ALIGN_OMEGA_STOP_TOL = 999.0
        self.FINAL_ALIGN_STABLE_STEPS = 1
        self.FINAL_ALIGN_DECEL = 999.0

        # Filtro muy corto para evitar salto visual de referencia sin hacer
        # lenta la alineación.
        self.FINAL_ALIGN_CMD_TAU = 0.015
        self._final_align_stable_count = 0
        self._final_align_omega_cmd_f = 0.0

        # Distancia deseada base-goal al terminar fase 1.
        self.FINAL_GOAL_RADIUS_STOP = 0.70
        self.FINAL_RADIUS_TOL = 0.06
        self.FINAL_APPROACH_VMAX = 0.26
        self.FINAL_APPROACH_KV = 2.4
        self.FINAL_OMEGA_MIN = 0.50

        self.OBSTACLE_INFLATION = 0.0
        self.TRACKING_MARGIN = 0.03
        self.ROBOT_RADIUS_GEOM = 0.58
        self.ROBOT_RADIUS_PLAN = float(self.ROBOT_RADIUS_GEOM + self.TRACKING_MARGIN)

        self.USE_FIXED_RRT_BOUNDS = True
        self.RRT_FIXED_BOUNDS = (-5.0, 5.0, -5.0, 5.0)
        self.RRT_MAX_ATTEMPTS = 5
        self.RRT_TIMEOUT_BASE_S = 1.5
        self.RRT_TIMEOUT_STEP_S = 0.8
        self.RRT_STEPRANGE_FIXED = 0.2
        self.RRT_MIN_WP_POINTS = 10
        self.TERMINAL_ALIGN_DIST = 0.45
        self.RRT_STOP_CANDIDATE_LIMIT = 4
        self.RRT_START_REPAIR_MAX_DIST = 0.35
        self.RRT_START_REPAIR_STEP = 0.025
        self.RRT_PLANNER_VERSION = "direct_goal_stop_ped1_ranked_ring_v9_r070"
        
        # Parámetros heredados del planificador orbital. Se mantienen por
        # compatibilidad, pero el flujo nuevo usa _plan_direct_goal_stop().
        self.ORBIT_SAMPLES_N = 24
        self.ORBIT_TRY_TOPK = 8
        self.ORBIT_RANDOM_ROTATE = True

        # Si el punto radial directo cae dentro de un obstáculo, se explora todo
        # el anillo de parada. El score final sigue escogiendo la ruta más corta.
        self.DIRECT_STOP_FALLBACK_ANGLES_DEG = [0]
        for _a in range(8, 181, 8):
            self.DIRECT_STOP_FALLBACK_ANGLES_DEG.extend([_a, -_a])


        # Estados PID
        self._Accel_lin_des = 0.0
        self._Accel_ang_des = 0.0
        self._e_vel1 = self._e_vel2 = 0.0
        self._e_omega1 = self._e_omega2 = 0.0

        # Diagnóstico básico del último comando de base.
        self._last_base_mode = "idle"
        self._last_v_des_cmd = 0.0
        self._last_omega_des_cmd = 0.0
        self._last_cte_signed = 0.0
        self._last_cte_corr = 0.0
        self._last_track_err = 0.0
        self._last_yaw_track_err = 0.0

        # Retroceso (fase 1, entre fase0 y navegación)
        self._reverse_active = False
        self._reverse_goal_xy: Optional[np.ndarray] = None
        self.RETREAT_DIST = 0.35
        self.RETREAT_TOL = 0.01
        self.RETREAT_V_MAX = 0.20
        self.REV_SLOW_DIST = 0.20

        # Criterio parada
        self.STOP_V_MAX = 0.03
        self.STOP_OMEGA_MAX = 0.04
        self.STOP_COUNT_MIN = 5
        self._stop_count = 0

        # Corredor seguro del residual. En esta versión base se conserva la
        # lógica original: el residual solo se aplica dentro del corredor.
        # Corredor seguro más estricto para permitir residual en fase 1.
        # Si la base ya está saliendo del rumbo, el agente no interviene.
        self.TRACK_CORRIDOR = 0.06
        self.YAW_CORRIDOR = 0.20


    def _init_nav_state(self):
        self._waypoints: List[Tuple[float, float]] = []
        self._wp_idx = 0
        self._plan_ready = False
        self._initial_aligning = True
        self._final_aligning = False
        self._final_radial_approach = False
        self._reached_last_wp = False
        self._nav_done = False
        self._final_align_stable_count = 0
        self._final_align_omega_cmd_f = 0.0

        # Diagnóstico/caché de rutas TRACK_RRT:
        #   0 = sin ruta lista
        #   1 = ruta reutilizada desde caché/disco
        #   2 = ruta generada durante el episodio
        self._route_reused = False
        self._route_source_code = 0
        self._route_loaded_from_disk = False

    def _init_maps_and_obstacles(self):
        self.PREDEFINED_MAPS = {
            1: [(-2.0, 1.5), (-0.8, 0.8), (-0.8, 3.0)],
            2: [(-2.0, 1.7), (0.0, 0.2), (-1.0, 2.8)],
            3: [(-1.1, 2.0), (-1.0, 0.8), (1.0, 1.8)],
            4: [(-2.0, 1.5), (-1.0, 1.8), (1.0, 1.75)],
        }
        self.INITIALS_BY_MAP = {
            1: {
                "base_qpos7": [0.69905732, -0.7008061, 0.4, 0.70710678, 0.0, 0.0, 0.70710678],
                "cube_qpos7": [0.699057321, -0.000806096824, 0.5222, 0.882061601, 0.0, 0.0, -0.471134092],
            },
            2: {
                "base_qpos7": [0.94714452, -0.63903625, 0.4, 0.57357644, 0.0, 0.0, 0.81915204],
                "cube_qpos7": [0.70773042, 0.01874858, 0.5222, 0.99996507, 0.0, 0.0, -0.0083576],
            },
            3: {
                "base_qpos7": [1.41028180, 0.0000568799682, 0.4, 6.123234e-17, 0.0, 0.0, 1.0],
                "cube_qpos7": [0.710281796, 0.0000568799682, 0.5222, 0.843043949, 0.0, 0.0, -0.537844681],
            },
            4: {
                "base_qpos7": [0.36241759, -0.53709581, 0.4, 0.88701083, 0.0, 0.0, 0.46174861],
                "cube_qpos7": [0.76392109, 0.03631062, 0.5222, 0.94469484, 0.0, 0.0, -0.32795068],
            },
        }

        self.USE_FIXED_INITIALS = True
        self.CUBE_XY_NOISE_RANGE = 0.0
        self.CUBE_YAW_NOISE_RANGE_RAD = 0.0

        self.MAP_SELECT_MODE = "cycle"
        # Lista usada por _pick_next_map_id(). Se mantiene [1,2,3,4]
        # porque este experimento vuelve a trabajar con todos los mapas.
        self.TRAIN_MAP_IDS = [1, 2, 3, 4]
        self.FIXED_MAP_ID: Optional[int] = None
        self._map_cycle_idx = 0
        self.map_id = 1

        self._OBS_BODY_NAMES = ("obstaculo1", "obstaculo2", "obstaculo3")
        self._OBS_BODY_IDS: List[int] = []
        for nm in self._OBS_BODY_NAMES:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, nm)
            if bid < 0:
                raise RuntimeError(f"No se encontró body obstáculo: {nm}")
            self._OBS_BODY_IDS.append(int(bid))

        # Carpeta nueva para rutas directas suavizadas. Al usar una carpeta
        # distinta evitamos reutilizar rutas orbitales antiguas.
        self.MAPS_DIR = str(PKG_DIR / "RUTAS_SAC")
        self.LOAD_ROUTES_ON_START = True
        self.SAVE_ROUTES_ON_PLAN = True
        self.SAVE_ROUTE_PLOTS = True

        # Reutilización robusta para entrenamiento vectorizado:
        # cada instancia del ambiente carga rutas al iniciar, pero además
        # puede recargar perezosamente desde disco justo antes de planificar.
        # Esto permite que una ruta generada por un sub-entorno quede
        # disponible para los demás sin reiniciar el proceso.
        self.RELOAD_ROUTE_FROM_DISK_ON_REUSE = True
        self.STRICT_ROUTE_VERSION_CHECK = True
        self.ROUTE_GOAL_TOL = 1e-4
        self.ROUTE_OBSTACLE_TOL = 1e-4

        self.MAP_CACHE: Dict[int, dict] = {}

    def _init_cube_mass_sampling(self):
        self.MASS_SET = [0.50, 0.75, 1.00, 1.25, 1.50]
        self.eval_mode = False
        self.eval_mass = 1.50
        self._cube_mass: Optional[float] = None
        self._cube_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")

    def _init_peak_tracking(self):
        # Φ_peak[p] almacena el máximo histórico alcanzado durante el episodio
        self._phi_peak = {0: 0.0, 1: 0.0, 2: 0.0}

    def _init_debug(self):
        self.DEBUG_REWARD_PHASES = True
        self._phase_debug_accums: Dict[int, dict] = {}
        self._phase_success_flags = {0: False, 1: False, 2: False}

    # ========================================================================
    # DEBUG ACCUMULATORS
    # ========================================================================
    def _reset_phase_debug_accum(self, phase_id: int):
        self._phase_debug_accums[int(phase_id)] = {
            "steps": 0,
            "reward_total": 0.0,
            "phi_dense": 0.0,
            "phi_peak_gain": 0.0,
            "pen_time": 0.0,
            "pen_energy": 0.0,
            "pen_drop": 0.0,
            "pen_ped": 0.0,
            "pen_gripper_ped": 0.0,
            "pen_open_outside": 0.0,
            "early_release_pen": 0.0,
            "pen_drag": 0.0,
            "pen_posture_err": 0.0,
            "pen_posture_vel": 0.0,
            "r_release_event": 0.0,
            "r_release_ready_dense": 0.0,
            "r_place_stability": 0.0,
            # Métricas exclusivas del reward P2 V6. Deben existir también
            # cuando el episodio termina en P0/P1 porque SB3 Monitor accede
            # directamente a cada info_keyword al cerrar el episodio.
            "r_place_progress": 0.0,
            "r_place_peak_gain": 0.0,
            "r_center_event": 0.0,
            "r_descent_event": 0.0,
            "r_home_progress": 0.0,
            "r_home_event": 0.0,
            "pen_low_descent": 0.0,
            "bonus_final": 0.0,
        }

    def _reset_debug_all_phases(self):
        self._phase_success_flags = {0: False, 1: False, 2: False}
        for p in (0, 1, 2):
            self._reset_phase_debug_accum(p)

    def _accum_phase_debug(self, phase_id: int, reward_total: float, **parts):
        acc = self._phase_debug_accums[int(phase_id)]
        acc["steps"] += 1
        acc["reward_total"] += float(reward_total)
        for k, v in parts.items():
            if k in acc:
                acc[k] += float(v)

    def _print_phase_debug_summary(self, phase_id: int, success: bool, reason: str):
        if not self.DEBUG_REWARD_PHASES:
            return
        acc = self._phase_debug_accums[int(phase_id)]
        print(f"\n[EPISODE-END][PHASE-{int(phase_id)}]")
        print(f"success={bool(success)} | reason={reason} | steps={acc['steps']}")
        print(f"reward_total_acc={acc['reward_total']:.6f}")
        print(f"phi_dense_acc={acc['phi_dense']:.6f}")
        print(f"phi_peak_gain_acc={acc['phi_peak_gain']:.6f}")
        print(f"bonus_final_acc={acc['bonus_final']:.6f}")
        print(f"r_release_event_acc={acc['r_release_event']:.6f}")
        print(f"pen_time_acc={acc['pen_time']:.6f}")
        print(f"pen_energy_acc={acc['pen_energy']:.6f}")
        print(f"pen_drop_acc={acc['pen_drop']:.6f}")
        if int(phase_id) == 1:
            print(f"pen_posture_err_acc={acc['pen_posture_err']:.6f}")
            print(f"pen_posture_vel_acc={acc['pen_posture_vel']:.6f}")
        if int(phase_id) == 2:
            print(f"pen_ped_acc={acc['pen_ped']:.6f}")
            print(f"pen_gripper_ped_acc={acc['pen_gripper_ped']:.6f}")
            print(f"pen_open_outside_acc={acc['pen_open_outside']:.6f}")
            print(f"early_release_pen_acc={acc['early_release_pen']:.6f}")
            print(f"pen_drag_acc={acc['pen_drag']:.6f}")
            print(f"pen_low_descent_acc={acc['pen_low_descent']:.6f}")
            print(f"r_release_ready_dense_acc={acc['r_release_ready_dense']:.6f}")
            print(f"r_place_stability_acc={acc['r_place_stability']:.6f}")
            print(f"r_place_progress_acc={acc['r_place_progress']:.6f}")
            print(f"r_descent_progress_acc={acc.get('r_descent_progress', 0.0):.6f}")
            print(f"r_place_peak_gain_acc={acc['r_place_peak_gain']:.6f}")
            print(f"r_center_event_acc={acc['r_center_event']:.6f}")
            print(f"r_descent_event_acc={acc['r_descent_event']:.6f}")
            print(f"r_home_progress_acc={acc['r_home_progress']:.6f}")
            print(f"r_home_event_acc={acc['r_home_event']:.6f}")

    # ========================================================================
    # MAPAS / CACHÉ
    # ========================================================================
    def _ensure_maps_dir(self, folder: str) -> Path:
        p = Path(folder)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _map_route_filepath(self, map_id: int, folder: str) -> Path:
        return self._ensure_maps_dir(folder) / f"map_{int(map_id)}.json"


    def _load_map_routes_from_folder(self, folder: str, map_ids=None):
        mids = map_ids if map_ids is not None else [1, 2, 3, 4]
        self._ensure_maps_dir(folder)
        loaded = 0

        for mid in mids:
            if self._load_single_map_route_from_disk(int(mid), folder=folder, verbose=False):
                loaded += 1

        if loaded > 0:
            print(f"[MAPS] Rutas cargadas desde '{folder}': {loaded} mapas")

    def _load_single_map_route_from_disk(
        self,
        map_id: int,
        folder: Optional[str] = None,
        verbose: bool = True,
    ) -> bool:
        """Carga un archivo TRACK_RRT/map_X.json al MAP_CACHE.

        Esta función es deliberadamente tolerante: conserva campos antiguos
        cuando existen y acepta campos nuevos usados por el planificador
        directo. La validación estricta se realiza después en
        _cached_route_is_usable().
        """
        folder = str(folder if folder is not None else self.MAPS_DIR)
        fp = self._map_route_filepath(int(map_id), folder)
        if not fp.exists():
            return False

        try:
            with open(fp, "r", encoding="utf-8") as f:
                payload = json.load(f)

            entry = self.MAP_CACHE.get(int(map_id), {})

            obs = payload.get("obstacles_xy", None)
            if obs is None:
                entry["obstacles_xy"] = [
                    tuple(p) for p in self.PREDEFINED_MAPS.get(int(map_id), [])
                ]
            else:
                entry["obstacles_xy"] = [(float(p[0]), float(p[1])) for p in obs]

            for src_key, dst_key in (
                ("base_qpos7", "base_qpos7"),
                ("cube_qpos7", "cube_qpos7"),
                ("cube_qpos7_nominal", "cube_qpos7_nominal"),
            ):
                value = payload.get(src_key, None)
                if value is not None:
                    entry[dst_key] = np.array(value, dtype=np.float64)

            for src_key, dst_key in (
                ("path", "path"),
                ("raw_path", "raw_path"),
            ):
                value = payload.get(src_key, None)
                entry[dst_key] = (
                    [(float(p[0]), float(p[1])) for p in value]
                    if value else None
                )

            for src_key, dst_key in (
                ("plan_goal_xy", "plan_goal_xy"),
                ("plan_pregoal_xy", "plan_pregoal_xy"),
                ("plan_stop_xy", "plan_stop_xy"),
            ):
                value = payload.get(src_key, None)
                entry[dst_key] = (
                    (float(value[0]), float(value[1]))
                    if value is not None else None
                )

            entry["route_type"] = str(payload.get("route_type", "unknown"))
            entry["stop_radius"] = float(payload.get("stop_radius", -1.0))
            entry["planner_version"] = str(payload.get("planner_version", "unknown"))
            entry["terminal_tangent_goal_err_rad"] = float(
                payload.get("terminal_tangent_goal_err_rad", np.nan)
            )
            entry["waypoint_spacing"] = float(payload.get("waypoint_spacing", np.nan))
            entry["lookahead_dist"] = float(payload.get("lookahead_dist", np.nan))
            entry["robot_radius_plan"] = float(payload.get("robot_radius_plan", np.nan))
            entry["obstacle_inflation"] = float(payload.get("obstacle_inflation", np.nan))
            entry["plot_xy_png"] = payload.get("plot_xy_png", None)
            entry["_loaded_from_disk"] = True
            entry["_route_file"] = str(fp)

            self.MAP_CACHE[int(map_id)] = entry

            if verbose:
                print(f"[MAPS] Ruta TRACK_RRT cargada: {fp}")
            return True

        except Exception as e:
            print(f"[MAPS] No se pudo cargar {fp}: {e}")
            return False

    def _cached_route_is_usable(self, cached: Optional[dict]) -> bool:
        if not cached:
            return False

        path = cached.get("path", None)
        if not path or len(path) < 2:
            return False

        route_type = str(cached.get("route_type", "unknown"))
        if route_type != "direct_goal_stop":
            return False

        stop_radius = float(cached.get("stop_radius", -1.0))
        if abs(stop_radius - float(self.FINAL_GOAL_RADIUS_STOP)) > 1e-6:
            return False

        if bool(getattr(self, "STRICT_ROUTE_VERSION_CHECK", True)):
            planner_version = str(cached.get("planner_version", "unknown"))
            if planner_version != str(getattr(self, "RRT_PLANNER_VERSION", "")):
                return False

        goal_xy = self._goal_xy()
        cached_goal = cached.get("plan_goal_xy", None)
        if cached_goal is not None:
            cached_goal = np.asarray(cached_goal, dtype=np.float64).reshape(2)
            if float(np.linalg.norm(cached_goal - goal_xy)) > float(getattr(self, "ROUTE_GOAL_TOL", 1e-4)):
                return False

        # Verificación geométrica mínima: el último waypoint debe estar en el
        # anillo de parada actual respecto del goal del XML/mapa cargado.
        last = np.asarray(path[-1], dtype=np.float64).reshape(2)
        d_last_goal = float(np.linalg.norm(last - goal_xy))
        if abs(d_last_goal - float(self.FINAL_GOAL_RADIUS_STOP)) > max(float(self.FINAL_RADIUS_TOL), 0.08):
            return False

        # Si el JSON trae obstáculos, comprobar que corresponde al mapa actual.
        cached_obs = cached.get("obstacles_xy", None)
        expected_obs = self.PREDEFINED_MAPS.get(int(self.map_id), None)
        if cached_obs is not None and expected_obs is not None and len(cached_obs) == len(expected_obs):
            c = np.asarray(cached_obs, dtype=np.float64)
            e = np.asarray(expected_obs, dtype=np.float64)
            if c.shape == e.shape:
                max_err = float(np.max(np.abs(c - e)))
                if max_err > float(getattr(self, "ROUTE_OBSTACLE_TOL", 1e-4)):
                    return False

        return True

    def _reload_current_map_route_from_disk(self) -> bool:
        """Recarga perezosa de la ruta del mapa activo desde TRACK_RRT."""
        if not bool(getattr(self, "RELOAD_ROUTE_FROM_DISK_ON_REUSE", True)):
            return False
        return self._load_single_map_route_from_disk(
            int(self.map_id),
            folder=str(self.MAPS_DIR),
            verbose=True,
        )


    def _try_reuse_cached_route(self) -> bool:
        cached = self.MAP_CACHE.get(int(self.map_id), None)

        # En entrenamiento vectorizado, otro sub-entorno puede haber generado
        # TRACK_RRT/map_X.json después de que esta instancia se inicializó.
        # Por eso, si la caché local no sirve, se intenta recargar desde disco
        # justo antes de lanzar una nueva planificación.
        if not self._cached_route_is_usable(cached):
            if self._reload_current_map_route_from_disk():
                cached = self.MAP_CACHE.get(int(self.map_id), None)

        if not self._cached_route_is_usable(cached):
            return False

        path = cached.get("path", None)
        self._waypoints = [(float(p[0]), float(p[1])) for p in path]
        self._wp_idx = 0
        self._plan_ready = True
        self._initial_aligning = False if not bool(getattr(self, "USE_INITIAL_ALIGN", False)) else True
        self._final_aligning = False
        self._final_radial_approach = False
        self._reached_last_wp = False
        self._nav_done = False
        self._final_align_stable_count = 0
        self._final_align_omega_cmd_f = 0.0

        self._route_reused = True
        self._route_source_code = 1
        self._route_loaded_from_disk = bool(cached.get("_loaded_from_disk", False))

        goal_xy = self._goal_xy()
        last = np.asarray(path[-1], dtype=np.float64).reshape(2)
        d_last_goal = float(np.linalg.norm(last - goal_xy))

        src = "disco" if self._route_loaded_from_disk else "memoria"
        print(
            f"[RRT] Reusando ruta TRACK_RRT para mapa {int(self.map_id)} desde {src} "
            f"(wps={len(self._waypoints)}, radio={d_last_goal:.3f} m)."
        )
        return True

    def set_train_maps(self, map_ids):
        ids = [int(mid) for mid in map_ids]
        if not ids:
            raise ValueError("set_train_maps() recibió una lista vacía.")
        invalid = [mid for mid in ids if mid not in self.PREDEFINED_MAPS]
        if invalid:
            raise ValueError(f"Mapas inválidos en set_train_maps(): {invalid}")
        self.TRAIN_MAP_IDS = ids
        self._map_cycle_idx = 0

    def _pick_next_map_id(self) -> int:
        if self.FIXED_MAP_ID is not None:
            return int(self.FIXED_MAP_ID)

        seq = list(getattr(self, "TRAIN_MAP_IDS", [1, 2, 3, 4]))
        if not seq:
            raise RuntimeError("TRAIN_MAP_IDS está vacío.")

        if self.MAP_SELECT_MODE == "cycle":
            mid = seq[self._map_cycle_idx % len(seq)]
            self._map_cycle_idx += 1
            return int(mid)

        return int(self.np_random.choice(seq))

    def _safe_set_static_body_xy(self, body_id: int, x: float, y: float, keep_z: bool = True):
        z = float(self.model.body_pos[body_id, 2])
        self.model.body_pos[body_id, 0] = float(x)
        self.model.body_pos[body_id, 1] = float(y)
        self.model.body_pos[body_id, 2] = float(z if keep_z else self.model.body_pos[body_id, 2])

    def _set_obstacles_from_map(self, map_id: int):
        pts = self.PREDEFINED_MAPS[int(map_id)]
        for bid, (x, y) in zip(self._OBS_BODY_IDS, pts):
            self._safe_set_static_body_xy(bid, float(x), float(y), keep_z=True)

    # ========================================================================
    # CUBE MASS API
    # ========================================================================
    def _sample_cube_mass(self) -> float:
        return float(self.eval_mass) if self.eval_mode else float(self.np_random.choice(self.MASS_SET))

    def _sample_cube_reset_noise(self) -> Tuple[float, float, float]:
        """Muestrea la perturbación de pose del cubo usada en ``reset``.

        Se mantiene uniforme por defecto. Los currículos V8 pueden redefinir
        este único punto sin duplicar la lógica de inicialización de MuJoCo.
        """
        xy = float(self.CUBE_XY_NOISE_RANGE)
        yaw = float(self.CUBE_YAW_NOISE_RANGE_RAD)
        dx = float(self.np_random.uniform(-xy, xy)) if xy > 0.0 else 0.0
        dy = float(self.np_random.uniform(-xy, xy)) if xy > 0.0 else 0.0
        dyaw = float(self.np_random.uniform(-yaw, yaw)) if yaw > 0.0 else 0.0
        return dx, dy, dyaw

    def set_eval_mode(self, enabled: bool = True):
        self.eval_mode = bool(enabled)

    def set_eval_mass(self, mass: float):
        self.eval_mass = float(mass)

    def set_train_masses(self, masses):
        self.MASS_SET = [float(m) for m in masses]

    # ========================================================================
    # ARM STATE
    # ========================================================================

    def set_route_cache_dir(self, folder: str):
        """Permite que scripts externos definan la carpeta TRACK_RRT."""
        self.MAPS_DIR = str(folder)
        self._ensure_maps_dir(self.MAPS_DIR)
        if bool(getattr(self, "LOAD_ROUTES_ON_START", True)):
            self.reload_route_cache()

    def reload_route_cache(self, map_ids=None) -> bool:
        """Recarga rutas guardadas desde MAPS_DIR/TRACK_RRT."""
        if map_ids is None:
            map_ids = [1, 2, 3, 4]
        before = set(self.MAP_CACHE.keys())
        self._load_map_routes_from_folder(str(self.MAPS_DIR), map_ids=map_ids)
        after = set(self.MAP_CACHE.keys())
        return bool(after or before)

    def set_route_cache_options(
        self,
        load_on_start: Optional[bool] = None,
        save_on_plan: Optional[bool] = None,
        save_plots: Optional[bool] = None,
        reload_from_disk: Optional[bool] = None,
        strict_version: Optional[bool] = None,
    ):
        """Configura el comportamiento de caché de rutas desde scripts."""
        if load_on_start is not None:
            self.LOAD_ROUTES_ON_START = bool(load_on_start)
        if save_on_plan is not None:
            self.SAVE_ROUTES_ON_PLAN = bool(save_on_plan)
        if save_plots is not None:
            self.SAVE_ROUTE_PLOTS = bool(save_plots)
        if reload_from_disk is not None:
            self.RELOAD_ROUTE_FROM_DISK_ON_REUSE = bool(reload_from_disk)
        if strict_version is not None:
            self.STRICT_ROUTE_VERSION_CHECK = bool(strict_version)

    def _get_arm_qvel(self) -> np.ndarray:
        v = np.zeros(5, dtype=np.float32)
        for i, jn in enumerate(self.arm_joint_names):
            j = self.model.joint(jn)
            dofadr = self._as_int(j.dofadr[0])
            v[i] = float(self.data.qvel[dofadr])
        return v

    def _get_arm_qpos(self) -> np.ndarray:
        q = np.zeros(5, dtype=np.float32)
        for i, jn in enumerate(self.arm_joint_names):
            j = self.model.joint(jn)
            qadr = self._as_int(j.qposadr[0])
            q[i] = float(self.data.qpos[qadr])
        return q

    def _arm_pose_dict_to_array(self, arm_pose: dict) -> np.ndarray:
        return np.array([
            arm_pose["robot0:base_joint"],
            arm_pose["robot0:arm1_joint"],
            arm_pose["robot0:arm2_joint"],
            arm_pose["robot0:arm3_joint"],
            arm_pose["robot0:eef_joint"],
        ], dtype=np.float32)

    def _apply_arm_pose_to_qpos(self, qpos: np.ndarray, arm_pose: dict) -> np.ndarray:
        qpos = qpos.copy()
        for jname, jval in arm_pose.items():
            jadr = self._as_int(self.model.joint(jname).qposadr[0])
            qpos[jadr] = float(jval)
        return qpos

    def _arm_is_at_home(self) -> bool:
        """Verifica si el brazo está en posición home y quieto."""
        q_arm = self._get_arm_qpos()
        qd_arm = self._get_arm_qvel()

        pos_err = float(np.max(np.abs(q_arm - self.q_arm_home)))
        vel_max = float(np.max(np.abs(qd_arm)))

        return (pos_err <= float(self.HOME_POS_TOL)) and (vel_max <= float(self.HOME_VEL_TOL))

    # ------------------------------------------------------------------------
    # Lock J1/J5 tras levantar
    # ------------------------------------------------------------------------
    def _update_j1_j5_lock_state(self):
        if not bool(self.LOCK_J1_J5_AFTER_LIFT):
            self._j1_j5_locked = False
            self._cube_z_grasp_ref = None
            return

        if bool(self.cube_dropped):
            self._j1_j5_locked = False
            self._cube_z_grasp_ref = None
            return

        cube_z = float(self._cube_pos()[2])

        if bool(self.gripping):
            if self._cube_z_grasp_ref is None:
                self._cube_z_grasp_ref = cube_z
            dz = cube_z - float(self._cube_z_grasp_ref)
            if dz >= float(self.LIFT_LOCK_DELTA_Z):
                self._j1_j5_locked = True
        else:
            if not bool(self._carry_latch):
                self._cube_z_grasp_ref = None
                self._j1_j5_locked = False

    def _should_lock_j1_j5(self) -> bool:
        if not bool(self.LOCK_J1_J5_AFTER_LIFT):
            return False
        if bool(self.cube_dropped):
            return False
        if int(self.phase_state) == 0:
            return bool(self._j1_j5_locked)
        if int(self.phase_state) == 1:
            return bool(self._j1_j5_locked or self._carry_latch)
        return False

    def _apply_j1_j5_soft_lock_to_target(self, arm_target: np.ndarray) -> np.ndarray:
        arm_target = np.asarray(arm_target, dtype=np.float32).copy()
        if not self._should_lock_j1_j5():
            return arm_target

        current_target = np.asarray(self.arm_target, dtype=np.float32).copy()
        desired_j1 = float(self.LOCK_J1_VALUE)
        desired_j5 = float(self.LOCK_J5_VALUE)
        max_d_j1 = float(self.J1J5_LOCK_MAX_DELTA[0])
        max_d_j5 = float(self.J1J5_LOCK_MAX_DELTA[1])

        err_j1 = desired_j1 - float(current_target[self.ARM_IDX_J1])
        err_j5 = desired_j5 - float(current_target[self.ARM_IDX_J5])

        step_j1 = float(np.clip(err_j1, -max_d_j1, max_d_j1))
        step_j5 = float(np.clip(err_j5, -max_d_j5, max_d_j5))

        arm_target[self.ARM_IDX_J1] = float(current_target[self.ARM_IDX_J1] + step_j1)
        arm_target[self.ARM_IDX_J5] = float(current_target[self.ARM_IDX_J5] + step_j5)
        return arm_target.astype(np.float32)

    def _rate_limit_arm_target(self, desired_target: np.ndarray) -> np.ndarray:
        desired_target = np.asarray(desired_target, dtype=np.float32)
        desired_target = self._apply_j1_j5_soft_lock_to_target(desired_target)

        if int(self.phase_state) != 2:
            return np.clip(desired_target, self.ARM_POS_LOW, self.ARM_POS_HIGH).astype(np.float32)

        max_delta = self.ARM_MAX_DELTA_P2
        current_target = np.asarray(self.arm_target, dtype=np.float32)
        delta = np.clip(desired_target - current_target, -max_delta, max_delta)
        new_target = current_target + delta
        return np.clip(new_target, self.ARM_POS_LOW, self.ARM_POS_HIGH).astype(np.float32)

    # ========================================================================
    # BASE STATE HELPERS
    # ========================================================================
    def _stop_base(self):
        for aid in (self.wheel_left_ids + self.wheel_right_ids):
            self.data.ctrl[aid] = 0.0

    def _hard_stop_base(self):
        self._stop_base()
        try:
            self.data.qvel[0] = 0.0
            self.data.qvel[1] = 0.0
            self.data.qvel[5] = 0.0
        except Exception:
            pass

    def _zero_base_qvel_like_extract(self):
        try:
            self.data.qvel[0:6] = 0.0
            if hasattr(self.data, "qacc_warmstart") and (self.data.qacc_warmstart is not None):
                self.data.qacc_warmstart[0:6] = 0.0
        except Exception:
            pass

    def _base_is_stopped(self) -> bool:
        try:
            vx = float(self.data.qvel[0])
            vy = float(self.data.qvel[1])
            w = float(self.data.qvel[5])
        except Exception:
            return False
        v = float(np.hypot(vx, vy))
        return (v <= float(self.STOP_V_MAX)) and (abs(w) <= float(self.STOP_OMEGA_MAX))

    def _base_xy(self) -> np.ndarray:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "chassis_frame")
        if sid >= 0:
            p = self.data.site_xpos[sid]
            return np.array([p[0], p[1]], dtype=np.float64)
        return np.array([self.data.qpos[0], self.data.qpos[1]], dtype=np.float64)

    def _base_yaw(self) -> float:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "chassis_frame")
        if sid < 0:
            quat = self.data.qpos[3:7]
            w, x, y, z = quat
            return float(np.arctan2(2.0 * w * z, 1.0 - 2.0 * z * z))
        Rb = self.data.site_xmat[sid].reshape(3, 3)
        return float(np.arctan2(Rb[1, 0], Rb[0, 0]))

    def _goal_xy(self) -> np.ndarray:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "goal")
        if sid >= 0:
            p = self.data.site_xpos[sid]
            return np.array([p[0], p[1]], dtype=np.float64)
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pedestal_2")
        if bid >= 0:
            p = self.data.xpos[bid]
            return np.array([p[0], p[1]], dtype=np.float64)
        return np.zeros(2, dtype=np.float64)

    # ========================================================================
    # SITES / CUBE STATE
    # ========================================================================
    def _site_xpos(self, name: str) -> np.ndarray:
        return self.data.site(name).xpos.copy()

    def _cube_pos(self) -> np.ndarray:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target")
        if sid >= 0:
            return np.array(self.data.site_xpos[sid], dtype=np.float64)
        if self._cube_bid is not None and self._cube_bid >= 0:
            return np.array(self.data.xpos[self._cube_bid], dtype=np.float64)
        return self._site_xpos("target")

    def _cube_quat_wxyz(self) -> np.ndarray:
        if self._cube_bid is not None and self._cube_bid >= 0:
            q = np.array(self.data.xquat[self._cube_bid], dtype=np.float64)
            if np.linalg.norm(q) > 1e-9:
                return q
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def _cube_twist(self) -> np.ndarray:
        """Twist del cubo en frame global: [wx, wy, wz, vx, vy, vz].

        Compatible con MuJoCo 3.3.5 mediante mj_objectVelocity().  Se usa el
        body del cubo para que velocidad lineal y angular provengan del mismo
        estado físico y no dependan de la presencia de sensores auxiliares en
        el XML.
        """
        res = np.zeros(6, dtype=np.float64)
        if self._cube_bid is None or int(self._cube_bid) < 0:
            return res

        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            int(self._cube_bid),
            res,
            0,  # 0 = expresar el twist en el frame global
        )
        return res

    def _cube_vel_vec(self) -> np.ndarray:
        return self._cube_twist()[3:6].copy()

    def _cube_angular_vel(self) -> np.ndarray:
        return self._cube_twist()[0:3].copy()

    def _cube_speed(self) -> float:
        return float(np.linalg.norm(self._cube_vel_vec()))

    def _cube_reach_target_pos(self) -> np.ndarray:
        p = self._cube_pos().copy()
        p[2] += float(self.REACH_Z_OFFSET)
        return p

    def _goal_place_target_pos(self) -> np.ndarray:
        gxy = self._goal_xy()
        return np.array([gxy[0], gxy[1], self.Z_CUBE_REST], dtype=np.float64)

    def _cube_clearance_over_pedestal(self) -> float:
        """Holgura del fondo del cubo sobre la tapa del pedestal [m]."""
        return float(self._cube_pos()[2] - self.CUBE_HALF - self.Z_PED_TOP)

    # ========================================================================
    # ENERGÍA
    # ========================================================================
    def _actuator_power_abs(self, actuator_ids: List[int]) -> float:
        p = 0.0
        for aid in actuator_ids:
            try:
                tau = float(self.data.actuator_force[aid])
            except Exception:
                tau = 0.0
            jid = int(self.model.actuator_trnid[aid, 0])
            if 0 <= jid < self.model.njnt:
                dofadr = int(self.model.jnt_dofadr[jid])
                qd = float(self.data.qvel[dofadr])
                p += abs(tau * qd)
            else:
                p += abs(tau)
        return float(p)

    def _energy_cost(self, Ts: float) -> Tuple[float, float, float]:
        arm_ids = list(self.arm_act_ids[:5])
        base_ids = list(self.wheel_left_ids + self.wheel_right_ids)
        P_arm = self._actuator_power_abs(arm_ids)
        P_base = self._actuator_power_abs(base_ids)
        E_arm = float(P_arm) * float(Ts)
        E_base = float(P_base) * float(Ts)
        return float(E_arm + E_base), float(E_arm), float(E_base)

    # ========================================================================
    # GRASP / CONTACTOS
    # ========================================================================
    def _is_grasping_by_contacts(self, fN_min: float = 0.8, require_both_sides: bool = True) -> bool:
        if self.data.ncon <= 0:
            return False

        left_ok = False
        right_ok = False
        f = np.zeros(6, dtype=np.float64)

        for ci in range(self.data.ncon):
            con = self.data.contact[ci]
            g1 = int(con.geom1)
            g2 = int(con.geom2)

            cube_pad = (
                (g1 in self._cube_geom_ids and (g2 in self._padL_geom_ids or g2 in self._padR_geom_ids)) or
                (g2 in self._cube_geom_ids and (g1 in self._padL_geom_ids or g1 in self._padR_geom_ids))
            )
            if not cube_pad:
                continue

            mujoco.mj_contactForce(self.model, self.data, ci, f)
            fN = float(f[0])
            if fN < float(fN_min):
                continue

            pad_geom = g2 if (g1 in self._cube_geom_ids) else g1
            if pad_geom in self._padL_geom_ids:
                left_ok = True
            elif pad_geom in self._padR_geom_ids:
                right_ok = True

            if require_both_sides:
                if left_ok and right_ok:
                    return True
            else:
                return True

        return (left_ok and right_ok) if require_both_sides else False

    def _update_grasp_flag(self):
        raw = self._is_grasping_by_contacts(
            fN_min=self.GRASP_FN_MIN,
            require_both_sides=self.GRASP_REQUIRE_BOTH_SIDES,
        )
        if raw:
            self._grasp_count = min(self._grasp_count + 1, 1000)
        else:
            self._grasp_count = max(self._grasp_count - 1, 0)
        self.gripping = bool(self._grasp_count >= int(self.GRASP_COUNT_MIN))

    def _grasp_strength(self) -> float:
        """Intensidad continua del agarre ∈ [0,1], basada en histéresis."""
        return float(np.clip(self._grasp_count / float(max(self.GRASP_COUNT_MIN, 1)), 0.0, 1.0))

    # ========================================================================
    # COLLISION SETS
    # ========================================================================
    def _children_lists(self) -> List[List[int]]:
        ch = [[] for _ in range(self.model.nbody)]
        for b in range(1, self.model.nbody):
            p = int(self.model.body_parentid[b])
            if p >= 0:
                ch[p].append(b)
        return ch

    def _collect_subtree(self, root_body_id: int) -> Set[int]:
        ch = self._children_lists()
        stack = [int(root_body_id)]
        seen: Set[int] = set(stack)
        while stack:
            b = stack.pop()
            for c in ch[b]:
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        return seen

    def _init_pedestal_collision_sets(self):
        self._pedestal_geom_ids: Set[int] = set()
        for gi in range(self.model.ngeom):
            gname = self.model.geom(gi).name or ""
            if gname.startswith("pedestal_geom_"):
                self._pedestal_geom_ids.add(int(gi))
        if not self._pedestal_geom_ids:
            raise RuntimeError("No se encontraron geoms pedestal_geom_* en el modelo.")

        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "chassis_frame")
        if sid >= 0:
            root_bid = int(self.model.site_bodyid[sid])
        else:
            root_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
            if root_bid < 0:
                raise RuntimeError("No se encontró 'chassis_frame' ni 'base_link'.")

        robot_bodies = self._collect_subtree(root_bid)
        cube_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")

        self._robot_geom_ids: Set[int] = set()
        for gi in range(self.model.ngeom):
            bid = int(self.model.geom_bodyid[gi])
            if bid in robot_bodies and bid != int(cube_bid):
                self._robot_geom_ids.add(int(gi))

    def _init_gripper_collision_sets(self):
        self._gripper_geom_ids: Set[int] = set()
        for nm in ("left_pad1", "left_pad2", "right_pad1", "right_pad2"):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, nm)
            if gid >= 0:
                self._gripper_geom_ids.add(int(gid))
        if not self._gripper_geom_ids:
            raise RuntimeError("No se pudieron identificar geoms del gripper.")

    def _count_pair_contacts(self, set_a: Set[int], set_b: Set[int]) -> int:
        n = 0
        for ci in range(self.data.ncon):
            con = self.data.contact[ci]
            g1 = int(con.geom1)
            g2 = int(con.geom2)
            if (g1 in set_a and g2 in set_b) or (g2 in set_a and g1 in set_b):
                n += 1
        return n

    def _count_robot_pedestal_contacts(self) -> int:
        return self._count_pair_contacts(self._robot_geom_ids, self._pedestal_geom_ids)

    def _count_gripper_pedestal_contacts(self) -> int:
        return self._count_pair_contacts(self._gripper_geom_ids, self._pedestal_geom_ids)

    def _count_cube_pedestal_contacts(self) -> int:
        return self._count_pair_contacts(self._cube_geom_ids, self._pedestal_geom_ids)

    # ========================================================================
    # PLANIFICACIÓN (RRT* directo al goal + suavizado geométrico)
    # ========================================================================
    def _get_rrt_bounds(self) -> Tuple[float, float, float, float]:
        if self.USE_FIXED_RRT_BOUNDS:
            return self.RRT_FIXED_BOUNDS
        cx, cy = self.model.stat.center[0], self.model.stat.center[1]
        ext = self.model.stat.extent
        return float(cx - ext), float(cx + ext), float(cy - ext), float(cy + ext)

    def _clamp_point(self, p: Tuple[float, float], margin: float = 0.05) -> Tuple[float, float]:
        minx, maxx, miny, maxy = self._get_rrt_bounds()
        x = float(np.clip(p[0], minx + margin, maxx - margin))
        y = float(np.clip(p[1], miny + margin, maxy - margin))
        return x, y

    def _extract_obstacles_2d(self, exclude_body_names: Tuple[str, ...] = ()) -> List[Tuple[float, float, float]]:
        circles: List[Tuple[float, float, float]] = []
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "chassis_frame")
        base_body = int(self.model.site_bodyid[site_id]) if site_id != -1 else -1
        robot_bodies: Set[int] = (self._collect_subtree(base_body) if base_body != -1 else set())

        excl: Set[int] = set()
        for nm in exclude_body_names:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, nm)
            if bid != -1:
                excl.add(int(bid))

        GEOM_BOX = mujoco.mjtGeom.mjGEOM_BOX
        GEOM_CYL = mujoco.mjtGeom.mjGEOM_CYLINDER
        GEOM_SPH = mujoco.mjtGeom.mjGEOM_SPHERE
        GEOM_PLN = mujoco.mjtGeom.mjGEOM_PLANE

        for gi in range(self.model.ngeom):
            gtype = int(self.model.geom_type[gi])
            if gtype == GEOM_PLN:
                continue
            body_id = int(self.model.geom_bodyid[gi])
            if body_id in robot_bodies or body_id in excl:
                continue

            cx = float(self.data.geom_xpos[gi, 0])
            cy = float(self.data.geom_xpos[gi, 1])
            sx, sy, _sz = self.model.geom_size[gi]

            if gtype == GEOM_BOX:
                R = float(np.hypot(sx, sy))
            elif gtype in (GEOM_CYL, GEOM_SPH):
                R = float(sx)
            else:
                R = float(max(sx, sy))

            R = float(R + self.OBSTACLE_INFLATION)
            circles.append((cx, cy, R))
        return circles

    def _point_is_free_2d(
        self,
        p: Tuple[float, float],
        obstacles: Optional[List[Tuple[float, float, float]]] = None,
        *,
        margin: float = 0.05,
    ) -> bool:
        x, y = float(p[0]), float(p[1])
        minx, maxx, miny, maxy = self._get_rrt_bounds()
        if not (minx + margin <= x <= maxx - margin and miny + margin <= y <= maxy - margin):
            return False
        if obstacles is None:
            obstacles = self._extract_obstacles_2d(exclude_body_names=("cube", "pedestal_2"))
        for ox, oy, orad in obstacles:
            rr = float(orad + self.ROBOT_RADIUS_PLAN)
            if (x - ox) ** 2 + (y - oy) ** 2 <= rr * rr:
                return False
        return True

    def _segment_hits_obstacles(self, p0: Tuple[float, float], p1: Tuple[float, float], obstacles) -> bool:
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        vx = x1 - x0
        vy = y1 - y0
        vv = vx * vx + vy * vy

        for ox, oy, orad in obstacles:
            rr = float(orad + self.ROBOT_RADIUS_PLAN)
            if vv < 1e-12:
                d2 = (x0 - ox) ** 2 + (y0 - oy) ** 2
            else:
                t = ((ox - x0) * vx + (oy - y0) * vy) / vv
                t = float(np.clip(t, 0.0, 1.0))
                px = x0 + t * vx
                py = y0 + t * vy
                d2 = (px - ox) ** 2 + (py - oy) ** 2
            if d2 <= rr * rr:
                return True
        return False

    def _path_is_collision_free(
        self,
        pts: List[Tuple[float, float]],
        obstacles: Optional[List[Tuple[float, float, float]]] = None,
    ) -> bool:
        if len(pts) < 2:
            return False
        if obstacles is None:
            obstacles = self._extract_obstacles_2d(exclude_body_names=("cube", "pedestal_2"))
        for p in pts:
            if not self._point_is_free_2d(p, obstacles):
                return False
        for i in range(len(pts) - 1):
            if self._segment_hits_obstacles(pts[i], pts[i + 1], obstacles):
                return False
        return True

    def _densify_path(self, pts: List[Tuple[float, float]], spacing: float) -> List[Tuple[float, float]]:
        if len(pts) <= 1:
            return pts[:]
        out = [pts[0]]
        ds = float(max(spacing, 1e-3))
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            seg = float(np.hypot(x1 - x0, y1 - y0))
            if seg <= ds:
                out.append((float(x1), float(y1)))
                continue
            n = int(np.ceil(seg / ds))
            for k in range(1, n + 1):
                t = float(k) / float(n)
                x = x0 * (1.0 - t) + x1 * t
                y = y0 * (1.0 - t) + y1 * t
                out.append((float(x), float(y)))
        return out

    def _remove_short_segments(
        self,
        pts: List[Tuple[float, float]],
        min_len: Optional[float] = None,
    ) -> List[Tuple[float, float]]:
        if len(pts) <= 2:
            return [(float(x), float(y)) for x, y in pts]
        min_len = float(self.PATH_MIN_TURN_SEG_LEN if min_len is None else min_len)
        out = [tuple(map(float, pts[0]))]
        for p in pts[1:-1]:
            p = tuple(map(float, p))
            if np.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) >= min_len:
                out.append(p)
        out.append(tuple(map(float, pts[-1])))
        return out

    def _shortcut_path(
        self,
        pts: List[Tuple[float, float]],
        obstacles: List[Tuple[float, float, float]],
        iters: int,
    ) -> List[Tuple[float, float]]:
        pts = [(float(x), float(y)) for x, y in pts]
        if len(pts) <= 2:
            return pts

        # Primero una pasada determinista de visibilidad directa.
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j_best = i + 1
            for j in range(len(pts) - 1, i, -1):
                if not self._segment_hits_obstacles(pts[i], pts[j], obstacles):
                    j_best = j
                    break
            out.append(pts[j_best])
            i = j_best
        pts = out

        # Luego atajos aleatorios, usando el generador del ambiente.
        n_iter = int(max(0, iters))
        for _ in range(n_iter):
            if len(pts) <= 3:
                break
            i = int(self.np_random.integers(0, len(pts) - 2))
            j = int(self.np_random.integers(i + 2, len(pts)))
            if j <= i + 1:
                continue
            if not self._segment_hits_obstacles(pts[i], pts[j], obstacles):
                pts = pts[:i + 1] + pts[j:]
        return pts

    def _chaikin_path(
        self,
        pts: List[Tuple[float, float]],
        obstacles: List[Tuple[float, float, float]],
        iters: int,
    ) -> List[Tuple[float, float]]:
        pts = [(float(x), float(y)) for x, y in pts]
        for _ in range(int(max(0, iters))):
            if len(pts) <= 2:
                return pts
            cand = [pts[0]]
            for i in range(len(pts) - 1):
                p0 = np.asarray(pts[i], dtype=np.float64)
                p1 = np.asarray(pts[i + 1], dtype=np.float64)
                q = 0.75 * p0 + 0.25 * p1
                r = 0.25 * p0 + 0.75 * p1
                cand.append((float(q[0]), float(q[1])))
                cand.append((float(r[0]), float(r[1])))
            cand.append(pts[-1])

            # Solo aceptar el suavizado si mantiene holgura geométrica.
            if bool(self.PATH_SMOOTH_COLLISION_CHECK) and not self._path_is_collision_free(cand, obstacles):
                break
            pts = cand
        return pts

    def _catmull_rom_path(
        self,
        pts: List[Tuple[float, float]],
        obstacles: List[Tuple[float, float, float]],
    ) -> List[Tuple[float, float]]:
        pts = [(float(x), float(y)) for x, y in pts]
        if len(pts) <= 2 or not bool(getattr(self, "PATH_USE_SPLINE", True)):
            return pts

        P = [np.asarray(p, dtype=np.float64) for p in pts]
        samples = int(max(4, getattr(self, "PATH_SPLINE_SAMPLES_PER_SEG", 10)))
        out: List[Tuple[float, float]] = [pts[0]]

        for i in range(len(P) - 1):
            p0 = P[max(i - 1, 0)]
            p1 = P[i]
            p2 = P[i + 1]
            p3 = P[min(i + 2, len(P) - 1)]

            for k in range(1, samples + 1):
                t = float(k) / float(samples)
                t2 = t * t
                t3 = t2 * t
                q = 0.5 * (
                    (2.0 * p1)
                    + (-p0 + p2) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
                )
                out.append((float(q[0]), float(q[1])))

        if bool(self.PATH_SMOOTH_COLLISION_CHECK) and not self._path_is_collision_free(out, obstacles):
            return pts
        return out

    def _terminal_tangent_goal_error(
        self,
        pts: List[Tuple[float, float]],
        goal_xy: np.ndarray,
    ) -> float:
        if len(pts) < 2:
            return float(np.pi)
        p_prev = np.asarray(pts[-2], dtype=np.float64).reshape(2)
        p_last = np.asarray(pts[-1], dtype=np.float64).reshape(2)
        goal = np.asarray(goal_xy, dtype=np.float64).reshape(2)

        tangent = p_last - p_prev
        to_goal = goal - p_last
        nt = float(np.linalg.norm(tangent))
        ng = float(np.linalg.norm(to_goal))
        if nt < 1e-9 or ng < 1e-9:
            return float(np.pi)
        yaw_tangent = float(np.arctan2(tangent[1], tangent[0]))
        yaw_goal = float(np.arctan2(to_goal[1], to_goal[0]))
        return float(abs(wrap_angle(yaw_goal - yaw_tangent)))

    def _prepare_followable_path(
        self,
        raw_pts: List[Tuple[float, float]],
        obstacles: Optional[List[Tuple[float, float, float]]] = None,
    ) -> List[Tuple[float, float]]:
        if obstacles is None:
            obstacles = self._extract_obstacles_2d(exclude_body_names=("cube", "pedestal_2"))

        raw_pts = [(float(x), float(y)) for x, y in raw_pts]
        terminal_stop = raw_pts[-1] if len(raw_pts) >= 3 else None
        work_pts = raw_pts[:-1] if terminal_stop is not None else raw_pts

        pts = self._remove_short_segments(work_pts, min_len=0.02)
        pts = self._shortcut_path(pts, obstacles, int(self.PATH_SHORTCUT_ITERS))
        pts = self._remove_short_segments(pts)
        pts = self._chaikin_path(pts, obstacles, int(self.PATH_CHAIKIN_ITERS))
        pts = self._catmull_rom_path(pts, obstacles)
        pts = self._remove_short_segments(pts, min_len=0.02)
        pts = self._densify_path(pts, self.WAYPOINT_SPACING)

        if terminal_stop is not None and len(pts) >= 1:
            terminal = self._densify_path([pts[-1], terminal_stop], self.WAYPOINT_SPACING)
            pts = pts + terminal[1:]

        # El primer y último punto deben permanecer exactos: start y parada radial.
        if len(pts) >= 2:
            pts[0] = (float(raw_pts[0][0]), float(raw_pts[0][1]))
            pts[-1] = (float(raw_pts[-1][0]), float(raw_pts[-1][1]))
        return [(float(x), float(y)) for x, y in pts]

    def _plan_rrt_star_or_fallback(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        *,
        densify: bool = True,
    ):
        obstacles = self._extract_obstacles_2d(exclude_body_names=("cube", "pedestal_2"))

        start = (float(start[0]), float(start[1]))
        goal = (float(goal[0]), float(goal[1]))

        if not self._point_is_free_2d(start, obstacles):
            print(f"[RRT] Start no está libre para planificación: {start}")
            return None
        if not self._point_is_free_2d(goal, obstacles):
            print(f"[RRT] Goal de planificación no está libre: {goal}")
            return None

        if not self._segment_hits_obstacles(start, goal, obstacles):
            pts = [start, goal]
            return self._densify_path(pts, self.WAYPOINT_SPACING) if densify else pts

        if not self._ompl_ok:
            return None

        try:
            from ompl import base as ob
            from ompl import geometric as og

            minx, maxx, miny, maxy = self._get_rrt_bounds()
            space = ob.RealVectorStateSpace(2)
            bounds = ob.RealVectorBounds(2)
            bounds.setLow(0, minx)
            bounds.setHigh(0, maxx)
            bounds.setLow(1, miny)
            bounds.setHigh(1, maxy)
            space.setBounds(bounds)

            si = ob.SpaceInformation(space)

            def is_valid(state):
                x = float(state[0])
                y = float(state[1])
                for ox, oy, orad in obstacles:
                    rr = float(orad + self.ROBOT_RADIUS_PLAN)
                    if (x - ox) ** 2 + (y - oy) ** 2 <= rr * rr:
                        return False
                return True

            si.setStateValidityChecker(ob.StateValidityCheckerFn(is_valid))
            si.setup()

            pdef = ob.ProblemDefinition(si)
            pdef.setOptimizationObjective(ob.PathLengthOptimizationObjective(si))
            s = ob.State(space)
            s()[0] = float(start[0])
            s()[1] = float(start[1])
            g = ob.State(space)
            g()[0] = float(goal[0])
            g()[1] = float(goal[1])
            pdef.setStartAndGoalStates(s, g)

            planner = og.RRTstar(si)
            planner.setProblemDefinition(pdef)
            planner.setRange(float(self.RRT_STEPRANGE_FIXED))
            planner.setup()

            best_pts = None
            best_len = np.inf
            n_attempts = int(max(1, getattr(self, "RRT_MAX_ATTEMPTS", 1)))
            base_timeout = float(max(getattr(self, "RRT_TIMEOUT_BASE_S", 1.0), 1e-3))
            step_timeout = float(max(getattr(self, "RRT_TIMEOUT_STEP_S", 0.0), 0.0))

            for attempt in range(n_attempts):
                timeout = base_timeout if attempt == 0 else step_timeout
                if timeout <= 0.0:
                    continue
                solved = planner.solve(timeout)
                if not solved:
                    continue

                path = pdef.getSolutionPath()
                path.interpolate()
                states = path.getStates()
                pts = [(float(st[0]), float(st[1])) for st in states]
                if len(pts) < 2:
                    continue

                path_len = float(sum(
                    np.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
                    for i in range(len(pts) - 1)
                ))
                if path_len < best_len:
                    best_len = path_len
                    best_pts = pts

            if best_pts is None:
                return None
            return self._densify_path(best_pts, self.WAYPOINT_SPACING) if densify else best_pts
        except Exception as e:
            print(f"[RRT] Error OMPL: {e}")
            return None

    def _repair_planning_start(
        self,
        start: np.ndarray,
        goal_xy: np.ndarray,
        obstacles: List[Tuple[float, float, float]],
    ) -> Optional[np.ndarray]:
        start = np.asarray(start, dtype=np.float64).reshape(2)
        if self._point_is_free_2d(tuple(start), obstacles):
            return start

        push = np.zeros(2, dtype=np.float64)
        for ox, oy, orad in obstacles:
            c = np.array([float(ox), float(oy)], dtype=np.float64)
            v = start - c
            d = float(np.linalg.norm(v))
            rr = float(orad + self.ROBOT_RADIUS_PLAN)
            if d <= rr:
                if d < 1e-9:
                    to_goal = np.asarray(goal_xy, dtype=np.float64).reshape(2) - start
                    n_goal = float(np.linalg.norm(to_goal))
                    u = -to_goal / max(n_goal, 1e-9) if n_goal > 1e-9 else np.array([1.0, 0.0])
                else:
                    u = v / d
                push += u * float(rr - d + 0.02)

        n_push = float(np.linalg.norm(push))
        if n_push < 1e-9:
            to_goal = np.asarray(goal_xy, dtype=np.float64).reshape(2) - start
            n_goal = float(np.linalg.norm(to_goal))
            base_dir = to_goal / max(n_goal, 1e-9) if n_goal > 1e-9 else np.array([1.0, 0.0])
        else:
            base_dir = push / n_push

        max_dist = float(max(getattr(self, "RRT_START_REPAIR_MAX_DIST", 0.35), 0.0))
        step = float(max(getattr(self, "RRT_START_REPAIR_STEP", 0.025), 1e-3))
        angle_offsets = [0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90, 120, -120, 150, -150, 180]

        best = None
        best_dist = np.inf
        n_steps = int(np.ceil(max_dist / step))
        base_angle = float(np.arctan2(base_dir[1], base_dir[0]))
        for k in range(1, n_steps + 1):
            dist = float(k) * step
            for da_deg in angle_offsets:
                a = base_angle + np.deg2rad(float(da_deg))
                cand = start + dist * np.array([np.cos(a), np.sin(a)], dtype=np.float64)
                pc = np.asarray(self._clamp_point((float(cand[0]), float(cand[1]))), dtype=np.float64)
                if not self._point_is_free_2d(tuple(pc), obstacles):
                    continue
                d_start = float(np.linalg.norm(pc - start))
                if d_start < best_dist:
                    best_dist = d_start
                    best = pc
            if best is not None:
                break

        if best is not None:
            print(
                f"[RRT] Start ajustado a punto libre cercano: "
                f"({start[0]:.3f}, {start[1]:.3f}) -> ({best[0]:.3f}, {best[1]:.3f}), "
                f"delta={best_dist:.3f} m."
            )
            return best
        return None

    def _direct_stop_candidates(
        self,
        start: np.ndarray,
        goal_xy: np.ndarray,
        obstacles: List[Tuple[float, float, float]],
    ) -> List[Tuple[float, float]]:
        start = np.asarray(start, dtype=np.float64).reshape(2)
        goal_xy = np.asarray(goal_xy, dtype=np.float64).reshape(2)
        v = start - goal_xy
        d = float(np.linalg.norm(v))
        if d < 1e-9:
            # Caso degenerado: usar eje +X respecto del goal.
            base_angle = 0.0
        else:
            base_angle = float(np.arctan2(v[1], v[0]))

        radius = float(self.FINAL_GOAL_RADIUS_STOP)
        scored: List[Tuple[float, Tuple[float, float]]] = []
        seen = set()
        for da_deg in getattr(self, "DIRECT_STOP_FALLBACK_ANGLES_DEG", [0]):
            a = base_angle + np.deg2rad(float(da_deg))
            p = goal_xy + radius * np.array([np.cos(a), np.sin(a)], dtype=np.float64)
            pc = self._clamp_point((float(p[0]), float(p[1])))
            key = (round(pc[0], 4), round(pc[1], 4))
            if key in seen:
                continue
            seen.add(key)
            if not self._point_is_free_2d(pc, obstacles):
                continue

            approach_xy = self._terminal_alignment_point(pc, goal_xy)
            if approach_xy is None:
                continue
            if not self._point_is_free_2d(approach_xy, obstacles):
                continue
            if self._segment_hits_obstacles(approach_xy, pc, obstacles):
                continue

            heuristic = (
                float(np.linalg.norm(start - np.asarray(approach_xy, dtype=np.float64)))
                + float(np.linalg.norm(np.asarray(pc, dtype=np.float64) - np.asarray(approach_xy, dtype=np.float64)))
                + 0.001 * abs(float(da_deg))
            )
            scored.append((heuristic, pc))

        scored.sort(key=lambda item: item[0])
        limit = int(max(1, getattr(self, "RRT_STOP_CANDIDATE_LIMIT", 10)))
        return [pc for _score, pc in scored[:limit]]

    def _terminal_alignment_point(
        self,
        stop_xy: Tuple[float, float],
        goal_xy: np.ndarray,
    ) -> Optional[Tuple[float, float]]:
        stop = np.asarray(stop_xy, dtype=np.float64).reshape(2)
        goal = np.asarray(goal_xy, dtype=np.float64).reshape(2)
        to_goal = goal - stop
        norm = float(np.linalg.norm(to_goal))
        if norm < 1e-9:
            return None
        u_goal = to_goal / norm
        p = stop - float(self.TERMINAL_ALIGN_DIST) * u_goal
        return self._clamp_point((float(p[0]), float(p[1])))

    def _save_route_plot(
        self,
        path: List[Tuple[float, float]],
        raw_path: List[Tuple[float, float]],
        stop_xy: Tuple[float, float],
        obstacles: List[Tuple[float, float, float]],
        folder: str,
    ) -> Optional[Path]:
        """Guarda una gráfica XY auditable de la ruta generada.

        La figura se guarda dentro de TRACK_RRT y muestra:
          - ruta RRT* cruda;
          - ruta final suavizada que seguirá la base;
          - punto de inicio, goal y punto de parada radial;
          - anillo de parada a FINAL_GOAL_RADIUS_STOP;
          - obstáculos físicos y obstáculos inflados usados por el planificador.
        """
        if not bool(getattr(self, "SAVE_ROUTE_PLOTS", False)):
            return None

        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle

            folder_path = self._ensure_maps_dir(folder)
            fp = folder_path / f"map_{int(self.map_id)}_xy.png"

            fig, ax = plt.subplots(figsize=(8, 8))

            if raw_path and len(raw_path) >= 2:
                rp = np.asarray(raw_path, dtype=np.float64)
                ax.plot(
                    rp[:, 0], rp[:, 1],
                    "--", linewidth=1.0, alpha=0.75,
                    label="RRT* cruda",
                )

            pp = np.asarray(path, dtype=np.float64)
            if pp.ndim == 2 and pp.shape[0] >= 2:
                ax.plot(
                    pp[:, 0], pp[:, 1],
                    linewidth=2.2,
                    label="Ruta suavizada",
                )
                ax.scatter(pp[0, 0], pp[0, 1], marker="o", s=55, label="inicio")

            stop_xy = (float(stop_xy[0]), float(stop_xy[1]))
            ax.scatter(stop_xy[0], stop_xy[1], marker="x", s=70, label=f"parada {self.FINAL_GOAL_RADIUS_STOP:.2f} m")

            gxy = np.asarray(self._goal_xy(), dtype=np.float64).reshape(2)
            ax.scatter(gxy[0], gxy[1], marker="*", s=140, label="goal")
            ax.add_patch(
                Circle(
                    (gxy[0], gxy[1]),
                    float(self.FINAL_GOAL_RADIUS_STOP),
                    fill=False,
                    linestyle=":",
                    linewidth=1.8,
                    label="radio de parada",
                )
            )

            # Obstáculos: orad ya incluye OBSTACLE_INFLATION; el planificador
            # además suma ROBOT_RADIUS_PLAN para asegurar holgura de la base.
            first_geom = True
            first_inflated = True
            for ox, oy, orad in obstacles:
                ox, oy, orad = float(ox), float(oy), float(orad)
                inflated_r = float(orad + self.ROBOT_RADIUS_PLAN)

                ax.add_patch(
                    Circle(
                        (ox, oy), orad,
                        fill=False, linestyle="-", linewidth=1.0, alpha=0.65,
                        label="obstáculo físico" if first_geom else None,
                    )
                )
                ax.add_patch(
                    Circle(
                        (ox, oy), inflated_r,
                        fill=False, linestyle="--", linewidth=1.2, alpha=0.85,
                        label="obstáculo inflado" if first_inflated else None,
                    )
                )
                ax.scatter(ox, oy, marker=".", s=20)
                first_geom = False
                first_inflated = False

            # Dibujar límites usados por RRT* para verificar rápidamente si el
            # planificador está trabajando dentro del mapa esperado.
            minx, maxx, miny, maxy = self._get_rrt_bounds()
            ax.set_xlim(float(minx), float(maxx))
            ax.set_ylim(float(miny), float(maxy))

            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linewidth=0.5, alpha=0.45)
            ax.set_title(f"TRACK_RRT - mapa {int(self.map_id)}")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            fig.savefig(fp, dpi=160)
            plt.close(fig)
            return fp

        except Exception as e:
            print(f"[RRT] No se pudo guardar gráfico XY de ruta: {e}")
            return None

    def _save_route_to_cache_and_disk(
        self,
        path: List[Tuple[float, float]],
        raw_path: List[Tuple[float, float]],
        stop_xy: Tuple[float, float],
        obstacles: List[Tuple[float, float, float]],
    ):
        entry = self.MAP_CACHE.get(int(self.map_id), {})
        entry["path"] = [(float(x), float(y)) for x, y in path]
        entry["raw_path"] = [(float(x), float(y)) for x, y in raw_path]
        entry["plan_stop_xy"] = (float(stop_xy[0]), float(stop_xy[1]))
        entry["plan_goal_xy"] = tuple(map(float, self._goal_xy()))
        entry["route_type"] = "direct_goal_stop"
        entry["stop_radius"] = float(self.FINAL_GOAL_RADIUS_STOP)
        entry["planner_version"] = str(self.RRT_PLANNER_VERSION)
        entry["terminal_tangent_goal_err_rad"] = self._terminal_tangent_goal_error(path, self._goal_xy())
        entry["obstacles_xy"] = [tuple(map(float, p)) for p in self.PREDEFINED_MAPS.get(int(self.map_id), [])]
        entry["obstacles_plan_circles"] = [(float(x), float(y), float(r)) for x, y, r in obstacles]
        entry["obstacles_inflated_circles"] = [
            (float(x), float(y), float(r + self.ROBOT_RADIUS_PLAN))
            for x, y, r in obstacles
        ]
        self.MAP_CACHE[int(self.map_id)] = entry

        if not bool(getattr(self, "SAVE_ROUTES_ON_PLAN", True)):
            return

        folder = str(self.MAPS_DIR)
        fp = self._map_route_filepath(int(self.map_id), folder)
        plot_fp = self._save_route_plot(path, raw_path, stop_xy, obstacles, folder)

        def _json_list(value):
            if value is None:
                return None
            return np.asarray(value, dtype=np.float64).reshape(-1).tolist()

        payload = {
            "route_type": "direct_goal_stop",
            "planner_version": str(self.RRT_PLANNER_VERSION),
            "map_id": int(self.map_id),
            "stop_radius": float(self.FINAL_GOAL_RADIUS_STOP),
            "waypoint_spacing": float(self.WAYPOINT_SPACING),
            "lookahead_dist": float(self.LOOKAHEAD_DIST),
            "terminal_align_dist": float(self.TERMINAL_ALIGN_DIST),
            "terminal_tangent_goal_err_rad": float(self._terminal_tangent_goal_error(path, self._goal_xy())),
            "robot_radius_plan": float(self.ROBOT_RADIUS_PLAN),
            "obstacle_inflation": float(self.OBSTACLE_INFLATION),
            "path": [[float(x), float(y)] for x, y in path],
            "raw_path": [[float(x), float(y)] for x, y in raw_path],
            "plan_goal_xy": [float(v) for v in self._goal_xy()],
            "plan_stop_xy": [float(stop_xy[0]), float(stop_xy[1])],
            "obstacles_xy": [[float(x), float(y)] for x, y in self.PREDEFINED_MAPS.get(int(self.map_id), [])],
            "obstacles_plan_circles": [
                {"x": float(x), "y": float(y), "radius_model": float(r)}
                for x, y, r in obstacles
            ],
            "obstacles_inflated_circles": [
                {"x": float(x), "y": float(y), "radius_inflated": float(r + self.ROBOT_RADIUS_PLAN)}
                for x, y, r in obstacles
            ],
            "plot_xy_png": str(plot_fp) if plot_fp is not None else None,
            "base_qpos7": _json_list(entry.get("base_qpos7", None)),
            "cube_qpos7": _json_list(entry.get("cube_qpos7", None)),
            "cube_qpos7_nominal": _json_list(entry.get("cube_qpos7", None)),
        }
        try:
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"[RRT] Ruta directa guardada en: {fp}")
            if plot_fp is not None:
                print(f"[RRT] Gráfico XY guardado en: {plot_fp}")
        except Exception as e:
            print(f"[RRT] No se pudo guardar ruta en {fp}: {e}")

    def _plan_direct_goal_stop(self) -> bool:
        """Planifica hacia el goal y detiene la base a FINAL_GOAL_RADIUS_STOP.

        La meta del planificador no es un punto orbital arbitrario, sino el
        punto radial sobre la recta goal->base ubicado a 0.70 m del goal. Si
        esa meta directa no está libre, se prueban pequeñas desviaciones
        angulares como rescate; la trayectoria resultante se suaviza antes de
        entregarla al controlador de seguimiento.
        """
        start_real = np.asarray(self._base_xy(), dtype=np.float64).reshape(2)
        goal_xy = np.asarray(self._goal_xy(), dtype=np.float64).reshape(2)
        obstacles = self._extract_obstacles_2d(exclude_body_names=("cube", "pedestal_2"))

        start = self._repair_planning_start(start_real, goal_xy, obstacles)
        if start is None:
            print(f"[RRT] Start no se pudo ajustar a un punto libre: {tuple(start_real)}")
            self._plan_ready = False
            self._waypoints = []
            self._wp_idx = 0
            return False

        candidates = self._direct_stop_candidates(start, goal_xy, obstacles)
        if not candidates:
            print(
                "[RRT] No hay punto de parada radial libre a "
                f"{self.FINAL_GOAL_RADIUS_STOP:.2f} m del goal."
            )
            self._plan_ready = False
            self._waypoints = []
            self._wp_idx = 0
            return False

        best_path = None
        best_raw = None
        best_stop = None
        best_tangent_err = np.inf
        best_score = np.inf

        direct_stop = candidates[0]
        for stop_xy in candidates:
            approach_xy = self._terminal_alignment_point(stop_xy, goal_xy)
            if approach_xy is None:
                continue
            if not self._point_is_free_2d(approach_xy, obstacles):
                continue
            if self._segment_hits_obstacles(approach_xy, stop_xy, obstacles):
                continue

            raw_to_approach = self._plan_rrt_star_or_fallback(tuple(start), tuple(approach_xy), densify=False)
            if raw_to_approach is None or len(raw_to_approach) < 2:
                continue

            raw = list(raw_to_approach)
            if np.hypot(raw[-1][0] - approach_xy[0], raw[-1][1] - approach_xy[1]) > 1e-6:
                raw.append(tuple(approach_xy))
            raw.append(tuple(stop_xy))
            if raw is None or len(raw) < 2:
                continue

            path = self._prepare_followable_path(raw, obstacles)
            if len(path) < int(self.RRT_MIN_WP_POINTS):
                path = self._densify_path(path, self.WAYPOINT_SPACING)

            if not self._path_is_collision_free(path, obstacles):
                continue

            tangent_err = self._terminal_tangent_goal_error(path, goal_xy)
            if tangent_err > float(self.PATH_TERMINAL_TANGENT_TOL):
                continue

            length = float(sum(
                np.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
                for i in range(len(path) - 1)
            ))
            direct_dev = float(np.hypot(stop_xy[0] - direct_stop[0], stop_xy[1] - direct_stop[1]))
            score = length + 1e-3 * direct_dev + 0.25 * tangent_err
            if score < best_score:
                best_score = score
                best_path = path
                best_raw = raw
                best_stop = stop_xy
                best_tangent_err = tangent_err

        if best_path is None or best_raw is None or best_stop is None:
            self._plan_ready = False
            self._waypoints = []
            self._wp_idx = 0
            print("[RRT] Falló planificación directa hacia punto de parada del goal.")
            return False

        self._waypoints = [(float(px), float(py)) for px, py in best_path]
        self._wp_idx = 0
        self._plan_ready = True
        self._initial_aligning = False if not bool(getattr(self, "USE_INITIAL_ALIGN", False)) else True
        self._final_aligning = False
        self._final_radial_approach = False
        self._reached_last_wp = False
        self._nav_done = False
        self._final_align_stable_count = 0
        self._final_align_omega_cmd_f = 0.0
        self._route_reused = False
        self._route_source_code = 2
        self._route_loaded_from_disk = False

        d_stop_goal = float(np.linalg.norm(np.asarray(best_stop, dtype=np.float64) - goal_xy))
        print(
            f"[RRT] Ruta directa suavizada lista: mapa={int(self.map_id)}, "
            f"wps={len(self._waypoints)}, stop_goal={d_stop_goal:.3f} m, "
            f"tangent_err={best_tangent_err:.3f} rad."
        )

        self._save_route_to_cache_and_disk(self._waypoints, best_raw, best_stop, obstacles)
        return True

    def _plan_orbit_style(self) -> bool:
        # Wrapper de compatibilidad con scripts antiguos: el comportamiento real
        # ahora es planificación directa al goal con parada radial.
        return self._plan_direct_goal_stop()

    # ========================================================================
    # PID BASE
    # ========================================================================

    def _nearest_path_segment_projection(
        self,
        xy: np.ndarray,
        *,
        start_idx: Optional[int] = None,
        search_back: int = 3,
        search_ahead: int = 24,
    ) -> Tuple[float, float, float, int, float]:
        """Proyecta la base sobre el segmento cercano más próximo de la ruta.

        Retorna:
            cte_signed:
                Error lateral firmado [m]. Positivo si la base está a la
                izquierda de la dirección del segmento A→B.
            px, py:
                Punto proyectado sobre el segmento.
            seg_idx:
                Índice del segmento más cercano.
            seg_yaw:
                Orientación tangencial del segmento.
        """
        if (not self._waypoints) or len(self._waypoints) < 2:
            P = np.asarray(xy, dtype=np.float64).reshape(2)
            return 0.0, float(P[0]), float(P[1]), 0, 0.0

        P = np.asarray(xy, dtype=np.float64).reshape(2)
        idx = self._wp_idx if start_idx is None else int(start_idx)

        i0 = max(0, int(idx) - int(search_back))
        i1 = min(len(self._waypoints) - 2, int(idx) + int(search_ahead))

        best_dist2 = np.inf
        best = (0.0, float(P[0]), float(P[1]), i0, 0.0)

        for i in range(i0, i1 + 1):
            A = np.asarray(self._waypoints[i], dtype=np.float64).reshape(2)
            B = np.asarray(self._waypoints[i + 1], dtype=np.float64).reshape(2)
            AB = B - A
            L2 = float(np.dot(AB, AB))
            if L2 < 1e-12:
                continue

            t = float(np.clip(np.dot(P - A, AB) / L2, 0.0, 1.0))
            Q = A + t * AB
            err = P - Q
            dist2 = float(np.dot(err, err))

            if dist2 < best_dist2:
                L = float(np.sqrt(L2))
                cross = float(AB[0] * (P[1] - A[1]) - AB[1] * (P[0] - A[0]))
                cte_signed = cross / max(L, 1e-9)
                seg_yaw = float(np.arctan2(AB[1], AB[0]))
                best_dist2 = dist2
                best = (float(cte_signed), float(Q[0]), float(Q[1]), int(i), seg_yaw)

        return best
    def _signed_cross_track_error(self, xy: np.ndarray) -> float:
        cte_signed, _px, _py, _seg_idx, _seg_yaw = self._nearest_path_segment_projection(xy)
        return float(cte_signed)

    def _reset_pid_accums(self):
        self._Accel_lin_des = 0.0
        self._Accel_ang_des = 0.0
        self._e_vel1 = self._e_vel2 = 0.0
        self._e_omega1 = self._e_omega2 = 0.0
        self._v_des_f = 0.0
        self._omega_des_f = 0.0
        self._omega_des_prev = 0.0

    def _actuator_tau_limit(self, aid: int) -> float:
        fr_hi = float(abs(self.model.actuator_forcerange[aid, 1]))
        if fr_hi > 1e-9:
            return fr_hi
        cr_hi = float(abs(self.model.actuator_ctrlrange[aid, 1]))
        if cr_hi > 1e-9:
            return cr_hi
        return 0.0

    def _apply_wheel_torque_with_residual(self, tau_each_L: float, tau_each_R: float, u_lin: float, u_ang: float):
        u_lin = float(np.clip(u_lin, -1.0, 1.0))
        u_ang = float(np.clip(u_ang, -1.0, 1.0))

        for aid in self.wheel_left_ids:
            hi = self._actuator_tau_limit(aid)
            if hi <= 0.0:
                self.data.ctrl[aid] = 0.0
                continue
            d_lin = float(self.RES_TAU_LIN_MAX) * hi * u_lin
            d_ang = float(self.RES_TAU_ANG_MAX) * hi * u_ang
            raw = float(tau_each_L + (d_lin - d_ang))
            self.data.ctrl[aid] = float(np.clip(raw, -hi, hi))

        for aid in self.wheel_right_ids:
            hi = self._actuator_tau_limit(aid)
            if hi <= 0.0:
                self.data.ctrl[aid] = 0.0
                continue
            d_lin = float(self.RES_TAU_LIN_MAX) * hi * u_lin
            d_ang = float(self.RES_TAU_ANG_MAX) * hi * u_ang
            raw = float(tau_each_R + (d_lin + d_ang))
            self.data.ctrl[aid] = float(np.clip(raw, -hi, hi))

    def _control_base_vel(
        self,
        Ts: float,
        v_des: float,
        omega_des: float,
        u_lin: float = 0.0,
        u_ang: float = 0.0,
        *,
        omega_max_override: Optional[float] = None,
        omega_rate_max_override: Optional[float] = None,
        acc_ang_max_override: Optional[float] = None,
        mode: str = "tracking",
    ):
        """Control PID incremental de base calibrado para decisiones cada 10 ms.

        La salida nominal del PID se suma al residual del agente en torque.
        En esta versión base, la activación o bloqueo del residual se decide
        fuera de esta función; aquí solo se aplica el residual recibido.
        """
        Ts_ctrl = float(max(Ts, 1e-6))
        self._last_base_mode = str(mode)
        self._last_v_des_cmd = float(v_des)
        self._last_omega_des_cmd = float(omega_des)

        theta = self._base_yaw()

        vx_w = float(self.data.qvel[0])
        vy_w = float(self.data.qvel[1])
        v_body = float(np.cos(theta) * vx_w + np.sin(theta) * vy_w)
        omega_z = float(self.data.qvel[5])

        omega_max = float(self.OMEGA_MAX if omega_max_override is None else omega_max_override)
        v_des = float(np.clip(v_des, -float(self.V_MAX), float(self.V_MAX)))
        omega_des = float(np.clip(omega_des, -omega_max, omega_max))

        if bool(self.RESET_ANG_ACC_ON_SIGN_FLIP):
            prev = float(self._omega_des_prev)
            th = float(self.OMEGA_SIGN_RESET_THRESH)
            if abs(prev) > th and abs(omega_des) > th and np.sign(prev) != np.sign(omega_des):
                self._Accel_ang_des = 0.0
                self._e_omega1 = 0.0
                self._e_omega2 = 0.0
        self._omega_des_prev = float(omega_des)

        if bool(self.USE_REF_RATE_LIMIT):
            dv_max = float(self.V_DES_RATE_MAX) * Ts_ctrl
            omega_rate_max = float(
                self.OMEGA_DES_RATE_MAX
                if omega_rate_max_override is None
                else omega_rate_max_override
            )
            dw_max = omega_rate_max * Ts_ctrl

            self._v_des_f += float(np.clip(v_des - self._v_des_f, -dv_max, dv_max))
            self._omega_des_f += float(np.clip(omega_des - self._omega_des_f, -dw_max, dw_max))

            v_des_pid = float(self._v_des_f)
            omega_des_pid = float(self._omega_des_f)
        else:
            v_des_pid = float(v_des)
            omega_des_pid = float(omega_des)

        e_vel = float(v_des_pid) - v_body
        e_omega = float(omega_des_pid) - omega_z

        e_vel_i = 0.0 if abs(e_vel) < float(self.VEL_ERR_DEADBAND) else e_vel
        e_omega_i = 0.0 if abs(e_omega) < float(self.OMEGA_ERR_DEADBAND) else e_omega

        self._Accel_lin_des += (
            self.Kp_vel * (e_vel - self._e_vel1)
            + self.Ki_vel * e_vel_i * Ts_ctrl
            + self.Kd_vel * (e_vel - 2.0 * self._e_vel1 + self._e_vel2) / Ts_ctrl
        )
        self._Accel_ang_des += (
            self.Kp_omega * (e_omega - self._e_omega1)
            + self.Ki_omega * e_omega_i * Ts_ctrl
            + self.Kd_omega * (e_omega - 2.0 * self._e_omega1 + self._e_omega2) / Ts_ctrl
        )

        acc_ang_max = float(self.ACC_ANG_MAX if acc_ang_max_override is None else acc_ang_max_override)
        self._Accel_lin_des = float(np.clip(self._Accel_lin_des, -float(self.ACC_LIN_MAX), float(self.ACC_LIN_MAX)))
        self._Accel_ang_des = float(np.clip(self._Accel_ang_des, -acc_ang_max, acc_ang_max))

        self._e_vel2, self._e_vel1 = self._e_vel1, e_vel
        self._e_omega2, self._e_omega1 = self._e_omega1, e_omega

        m = float(self.M_BASE)
        J = float(self.J_BASE)
        c = float(self.C_BASE)
        b = float(self.B_BASE)
        r = float(self.R_WHEEL)
        Wt = float(self.W_TRACK)

        Torque_R = (
            (m * r / 2.0) * self._Accel_lin_des
            + (J * r / Wt) * self._Accel_ang_des
            + (c * r / 2.0) * v_body
            + (b * r / Wt) * omega_z
        )
        Torque_L = (
            (m * r / 2.0) * self._Accel_lin_des
            - (J * r / Wt) * self._Accel_ang_des
            + (c * r / 2.0) * v_body
            - (b * r / Wt) * omega_z
        )

        tau_each_L = Torque_L / max(len(self.wheel_left_ids), 1)
        tau_each_R = Torque_R / max(len(self.wheel_right_ids), 1)
        self._apply_wheel_torque_with_residual(tau_each_L, tau_each_R, u_lin=u_lin, u_ang=u_ang)

    def _advance_wp_if_reached(self, x: float, y: float) -> bool:
        if (not self._waypoints) or self._reached_last_wp:
            return False

        # Avance por proyección sobre segmento, no solo por cercanía al punto.
        # Esto evita que la base quede "mirando" waypoints ya superados.
        _cte, _px, _py, seg_idx, _seg_yaw = self._nearest_path_segment_projection(
            np.array([x, y], dtype=np.float64),
            start_idx=self._wp_idx,
            search_back=2,
            search_ahead=32,
        )
        if seg_idx > self._wp_idx:
            self._wp_idx = int(seg_idx)

        advanced = False
        while self._wp_idx < len(self._waypoints) - 1:
            gx, gy = self._waypoints[self._wp_idx]
            if np.hypot(gx - x, gy - y) > float(self.WAYPOINT_TOL):
                break
            self._wp_idx += 1
            advanced = True

        if self._wp_idx >= len(self._waypoints) - 1:
            gx, gy = self._waypoints[-1]
            if np.hypot(gx - x, gy - y) <= float(self.WAYPOINT_TOL):
                self._reached_last_wp = True
                advanced = True

        return bool(advanced)

    def _choose_lookahead(self, x, y, start_idx, lookahead):
        if (not self._waypoints) or len(self._waypoints) < 2:
            return x, y, start_idx

        P = np.array([x, y], dtype=np.float64)
        _cte, px, py, seg_idx, _seg_yaw = self._nearest_path_segment_projection(
            P,
            start_idx=start_idx,
            search_back=2,
            search_ahead=40,
        )
        seg_idx = int(np.clip(seg_idx, 0, len(self._waypoints) - 2))
        self._wp_idx = max(int(self._wp_idx), seg_idx)

        # Avanzar una distancia lookahead medida sobre la ruta desde la
        # proyección actual. Esto genera una referencia continua sobre la curva,
        # no una persecución punto-a-punto.
        curr = np.array([px, py], dtype=np.float64)
        remaining = float(max(lookahead, 1e-6))
        j = seg_idx

        while j < len(self._waypoints) - 1:
            B = np.asarray(self._waypoints[j + 1], dtype=np.float64)
            seg_vec = B - curr
            seg_len = float(np.linalg.norm(seg_vec))
            if seg_len >= remaining:
                t = remaining / max(seg_len, 1e-9)
                G = curr + t * seg_vec
                return float(G[0]), float(G[1]), int(j)
            remaining -= seg_len
            j += 1
            curr = np.asarray(self._waypoints[j], dtype=np.float64)

        return float(self._waypoints[-1][0]), float(self._waypoints[-1][1]), int(len(self._waypoints) - 1)

    def _track_error_to_path(self, xy: np.ndarray, window: int = 20) -> float:
        if not self._waypoints:
            return 0.0
        cte_signed, _px, _py, _seg_idx, _seg_yaw = self._nearest_path_segment_projection(
            np.asarray(xy, dtype=np.float64).reshape(2),
            start_idx=self._wp_idx,
            search_back=max(2, int(window // 4)),
            search_ahead=max(8, int(window)),
        )
        return float(abs(cte_signed))

    def _start_reverse(self, distance_m: float = 0.35):
        x0, y0 = self._base_xy()
        yaw0 = self._base_yaw()
        dir_vec = np.array([np.cos(yaw0), np.sin(yaw0)], dtype=np.float64)

        self._reverse_goal_xy = np.array([x0, y0], dtype=np.float64) - float(distance_m) * dir_vec
        self._reverse_active = True

        self._plan_ready = False
        self._waypoints = []
        self._wp_idx = 0
        self._initial_aligning = True
        self._final_aligning = False
        self._final_radial_approach = False
        self._reached_last_wp = False
        self._nav_done = False
        self._final_align_stable_count = 0
        self._final_align_omega_cmd_f = 0.0

        self._reset_pid_accums()
        self._hard_stop_base()
        print(f"[REV] Inicia retroceso PID: objetivo {distance_m:.2f} m atrás.")

    # ------------------------------------------------------------------------
    # Controlador principal de base (pure pursuit + alineamientos)
    # ------------------------------------------------------------------------
    def _control_base_step(self, Ts: float, v_res_u: float, omega_res_u: float):
        if (not self._plan_ready) or (not self._waypoints) or self._nav_done:
            self._hard_stop_base()
            return

        x, y = self._base_xy()
        theta = self._base_yaw()
        target_xy = self._goal_xy()
        tx, ty = float(target_xy[0]), float(target_xy[1])

        # 1) Alineamiento inicial: rotación rápida, sin residual.
        if (
            bool(getattr(self, "USE_INITIAL_ALIGN", False))
            and (not self._reached_last_wp)
            and (not self._final_aligning)
            and (not self._final_radial_approach)
        ):
            if self._initial_aligning:
                align_idx = min(max(self._wp_idx + 2, 1), len(self._waypoints) - 1)
                gx_align, gy_align = self._waypoints[align_idx]
                theta_des0 = float(np.arctan2(gy_align - y, gx_align - x))
                e_theta0 = float(wrap_angle(theta_des0 - theta))

                if abs(e_theta0) <= float(self.YAW_ALIGN_EXIT):
                    self._initial_aligning = False
                    self._wp_idx = min(align_idx, len(self._waypoints) - 1)
                    self._reset_pid_accums()
                    return

                raw_omega = float(np.clip(
                    self.INIT_ALIGN_OMEGA_GAIN * e_theta0,
                    -self.INIT_ALIGN_OMEGA_MAX,
                    self.INIT_ALIGN_OMEGA_MAX,
                ))
                omega_cmd = (
                    float(np.sign(e_theta0) * self.INIT_ALIGN_MIN_OMEGA)
                    if abs(raw_omega) < float(self.INIT_ALIGN_MIN_OMEGA)
                    else raw_omega
                )

                self._control_base_vel(
                    Ts,
                    v_des=0.0,
                    omega_des=omega_cmd,
                    u_lin=0.0,
                    u_ang=0.0,
                    omega_max_override=float(self.INIT_ALIGN_OMEGA_MAX),
                    omega_rate_max_override=float(self.INIT_ALIGN_OMEGA_RATE_MAX),
                    acc_ang_max_override=float(self.INIT_ALIGN_ACC_ANG_MAX),
                    mode="initial_align",
                )
                return

        # 2) Seguimiento de trayectoria: lookahead + CTE.
        if (not self._reached_last_wp) and (not self._final_aligning) and (not self._final_radial_approach):
            self._advance_wp_if_reached(x, y)

            # ------------------------------------------------------------
            # Transición a cierre de navegación
            # ------------------------------------------------------------
            # La ruta termina en el punto radial de parada (0.70 m). Por eso,
            # si la base ya llegó con distancia y yaw aceptables, se declara
            # navegación lista sin ejecutar una alineación final. La alineación
            # final solo se activa cuando el heading hacia el goal todavía no
            # cumple la tolerancia.
            d_goal = float(np.hypot(x - tx, y - ty))
            radial_err_abs = abs(d_goal - float(self.FINAL_GOAL_RADIUS_STOP))

            d_last = 1e9
            if self._waypoints:
                last_x, last_y = self._waypoints[-1]
                d_last = float(np.hypot(last_x - x, last_y - y))

            theta_goal_now = float(np.arctan2(ty - y, tx - x))
            e_theta_goal_now = float(wrap_angle(theta_goal_now - theta))
            abs_e_goal_now = abs(e_theta_goal_now)

            ready_for_closeout = bool(
                self._reached_last_wp
                or (self._wp_idx >= len(self._waypoints) - 1 and d_last <= 0.20)
                or (radial_err_abs <= 0.12)
            )

            if ready_for_closeout:
                self._reached_last_wp = True
                self._initial_aligning = False
                self._reset_pid_accums()
                self._final_align_stable_count = 0
                self._final_align_omega_cmd_f = 0.0

                if radial_err_abs <= float(self.FINAL_RADIUS_TOL) and abs_e_goal_now <= float(self.FINAL_ALIGN_HEADING_TOL):
                    # Caso ideal: la ruta suavizada ya entregó la base en el
                    # anillo correcto y mirando hacia el goal.
                    self._final_aligning = False
                    self._final_radial_approach = False
                    self._nav_done = True
                    self._hard_stop_base()
                    self._zero_base_qvel_like_extract()
                    return

                if abs_e_goal_now > float(self.FINAL_ALIGN_HEADING_TOL):
                    self._final_aligning = True
                    self._final_radial_approach = False
                    self._stop_base()
                    return

                # Yaw OK, pero falta corregir levemente el radio.
                self._final_aligning = False
                self._final_radial_approach = True
                self._stop_base()
                return

            gx, gy, _lookahead_seg_idx = self._choose_lookahead(
                x,
                y,
                self._wp_idx,
                self.LOOKAHEAD_DIST,
            )

            e_wp = float(np.hypot(gx - x, gy - y))

            cte_signed = self._signed_cross_track_error(
                np.array([x, y], dtype=np.float64)
            )
            cte_corr = float(np.clip(
                float(self.CTE_GAIN) * cte_signed,
                -float(self.CTE_CORR_MAX),
                float(self.CTE_CORR_MAX),
            ))

            theta_des = float(np.arctan2(gy - y, gx - x))
            theta_des_corr = float(wrap_angle(theta_des - cte_corr))
            e_theta = float(wrap_angle(theta_des_corr - theta))

            self._last_cte_signed = float(cte_signed)
            self._last_cte_corr = float(cte_corr)
            self._last_track_err = float(abs(cte_signed))
            self._last_yaw_track_err = float(e_theta)

            heading_scale = float(1.0 / (1.0 + float(self.HEADING_K) * abs(e_theta)))
            slow_near_wp = float(np.clip(
                e_wp / max(float(self.WAYPOINT_TOL), 1e-3),
                0.2,
                1.0,
            ))

            v_path = (
                float(np.clip(float(self.V_GAIN) * e_wp, 0.0, float(self.V_MAX)))
                * heading_scale
                * slow_near_wp
            )

            omega_des = float(np.clip(
                float(self.OMEGA_GAIN) * e_theta,
                -float(self.OMEGA_MAX),
                float(self.OMEGA_MAX),
            ))

            v_cap_curv = float(self.V_CURV / (1.0 + float(self.K_CURV) * abs(omega_des)))
            v_des = float(np.clip(min(v_path, v_cap_curv), 0.0, float(self.V_MAX)))

            # No avanzar fuerte si la orientación de seguimiento es deficiente.
            abs_e_theta = abs(e_theta)
            if abs_e_theta > float(self.YAW_STOP):
                v_des = 0.0
            elif abs_e_theta > float(self.YAW_SLOW_2):
                v_des *= float(self.YAW_SLOW_GAIN_2)
            elif abs_e_theta > float(self.YAW_SLOW_1):
                v_des *= float(self.YAW_SLOW_GAIN_1)

            # Atenuar velocidad cuando el error lateral crece.
            cte_abs = abs(float(cte_signed))
            if cte_abs > float(self.CTE_SLOW_2):
                v_des *= float(self.CTE_SLOW_GAIN_2)
            elif cte_abs > float(self.CTE_SLOW_1):
                v_des *= float(self.CTE_SLOW_GAIN_1)

            # Corredor seguro geométrico. Este solo indica si sería razonable
            # permitir residual; no obliga a aplicarlo.
            safe_corridor = (
                cte_abs <= float(self.TRACK_CORRIDOR)
                and abs_e_theta <= float(self.YAW_CORRIDOR)
            )

            u_lin_raw = float(np.clip(v_res_u, -1.0, 1.0))
            u_ang_raw = float(np.clip(omega_res_u, -1.0, 1.0))
            u_lin = 0.0
            u_ang = 0.0
            block_reason = "pid_pure_phase1"

            residual_mode = str(getattr(self, "P1_BASE_RESIDUAL_MODE", "off")).lower()

            if residual_mode == "original":
                # Comportamiento anterior: residual completo dentro del corredor.
                if safe_corridor:
                    u_lin = u_lin_raw
                    u_ang = u_ang_raw
                    block_reason = "original_safe_corridor"
                else:
                    block_reason = "outside_safe_corridor"

            elif residual_mode == "guarded":
                # Residual pequeño y protegido: no se permite que el agente
                # contradiga el sentido de giro pedido por el PID cuando el
                # PID ya tiene una referencia angular clara.
                if safe_corridor:
                    u_lin = float(np.clip(
                        u_lin_raw,
                        -float(self.P1_RES_U_LIN_MAX),
                        float(self.P1_RES_U_LIN_MAX),
                    ))
                    u_ang_candidate = float(np.clip(
                        u_ang_raw,
                        -float(self.P1_RES_U_ANG_MAX),
                        float(self.P1_RES_U_ANG_MAX),
                    ))

                    sign_guard = bool(getattr(self, "P1_RES_SIGN_GUARD", True))
                    min_w = float(getattr(self, "P1_RES_MIN_OMEGA_FOR_SIGN", 0.08))
                    if (
                        sign_guard
                        and abs(float(omega_des)) > min_w
                        and abs(u_ang_candidate) > 1e-6
                        and np.sign(u_ang_candidate) != np.sign(float(omega_des))
                    ):
                        u_ang = 0.0
                        block_reason = "guarded_angular_sign_block"
                    else:
                        u_ang = u_ang_candidate
                        block_reason = "guarded_safe_corridor"
                else:
                    block_reason = "outside_safe_corridor"

            else:
                # Modo recomendado para esta prueba: fase 1 = PID puro.
                u_lin = 0.0
                u_ang = 0.0
                block_reason = "pid_pure_phase1"

            self._last_residual_safe_corridor = bool(safe_corridor)
            self._last_residual_block_reason = str(block_reason)
            self._last_u_lin_raw = float(u_lin_raw)
            self._last_u_ang_raw = float(u_ang_raw)
            self._last_u_lin_applied = float(u_lin)
            self._last_u_ang_applied = float(u_ang)

            self._control_base_vel(
                Ts,
                v_des=v_des,
                omega_des=omega_des,
                u_lin=u_lin,
                u_ang=u_ang,
                mode="tracking",
            )
            return

        # 3) Alineamiento final al goal: rápido con hard stop al llegar al umbral.
        if self._final_aligning and (not self._final_radial_approach):
            theta_goal = float(np.arctan2(ty - y, tx - x))
            e_theta = float(wrap_angle(theta_goal - theta))
            abs_e = abs(e_theta)

            # ------------------------------------------------------------
            # Captura rápida:
            # Si ya está dentro del umbral angular, se corta el giro con
            # hard stop y se pasa inmediatamente a la aproximación radial.
            # No se espera velocidad angular baja ni pasos estables, porque
            # el hard stop elimina el giro residual.
            # ------------------------------------------------------------
            if abs_e <= float(self.FINAL_ALIGN_HEADING_TOL):
                self._final_aligning = False
                self._final_radial_approach = True
                self._reset_pid_accums()

                self._final_align_stable_count = 0
                self._final_align_omega_cmd_f = 0.0

                # Aquí sí se usa hard stop: ya estamos dentro del umbral
                # deseado y queremos evitar cualquier giro residual antes
                # de entrar a final_radial.
                self._hard_stop_base()
                return

            # ------------------------------------------------------------
            # Comando angular rápido.
            # No se usa frenado anticipado; el corte se hace al entrar al
            # umbral angular mediante hard stop.
            # ------------------------------------------------------------
            omega_cmd = float(self.FINAL_ALIGN_OMEGA_GAIN) * e_theta
            omega_cmd = float(np.clip(
                omega_cmd,
                -float(self.FINAL_ALIGN_OMEGA_MAX),
                float(self.FINAL_ALIGN_OMEGA_MAX),
            ))

            # Filtro muy leve de referencia. Tau bajo = respuesta rápida.
            Ts_ctrl = float(max(Ts, 1e-6))
            tau = float(max(self.FINAL_ALIGN_CMD_TAU, 1e-6))
            beta = 1.0 - float(np.exp(-Ts_ctrl / tau))
            self._final_align_omega_cmd_f += beta * (
                omega_cmd - float(self._final_align_omega_cmd_f)
            )
            omega_cmd_f = float(self._final_align_omega_cmd_f)

            self._control_base_vel(
                Ts,
                v_des=0.0,
                omega_des=omega_cmd_f,
                u_lin=0.0,
                u_ang=0.0,
                omega_max_override=float(self.FINAL_ALIGN_OMEGA_MAX),
                omega_rate_max_override=float(self.FINAL_ALIGN_OMEGA_RATE_MAX),
                acc_ang_max_override=float(self.FINAL_ALIGN_ACC_ANG_MAX),
                mode="final_align",
            )
            return

        # 4) Aproximación radial final, sin residual.
        if self._final_radial_approach:
            d_goal = float(np.hypot(x - tx, y - ty))
            radial_err = float(d_goal - float(self.FINAL_GOAL_RADIUS_STOP))

            theta_goal = float(np.arctan2(ty - y, tx - x))
            e_theta = float(wrap_angle(theta_goal - theta))

            if abs(radial_err) <= float(self.FINAL_RADIUS_TOL):
                self._final_radial_approach = False
                self._nav_done = True
                self._hard_stop_base()
                self._zero_base_qvel_like_extract()
                self._reset_pid_accums()
                return

            raw_omega = float(np.clip(
                float(self.OMEGA_GAIN) * e_theta,
                -float(self.OMEGA_MAX),
                float(self.OMEGA_MAX),
            ))

            if abs(raw_omega) < float(self.FINAL_OMEGA_MIN) and abs(e_theta) > 1e-4:
                omega_des = float(np.sign(e_theta) * float(self.FINAL_OMEGA_MIN))
            else:
                omega_des = raw_omega

            v_cmd = float(np.clip(
                float(self.FINAL_APPROACH_KV) * radial_err,
                -float(self.FINAL_APPROACH_VMAX),
                float(self.FINAL_APPROACH_VMAX),
            ))

            if abs(e_theta) > 0.35:
                v_cmd *= 0.25

            self._control_base_vel(
                Ts,
                v_des=float(v_cmd),
                omega_des=float(omega_des),
                u_lin=0.0,
                u_ang=0.0,
                mode="final_radial",
            )
            return

        self._hard_stop_base()

    # ========================================================================
    # STAGE SETUPS    # ========================================================================
    # STAGE SETUPS
    # ========================================================================
    def _stage_setup_phase0(self):
        self.phase_state = 0
        self._phase_steps = 0

    def _stage_setup_phase1(self):
        self.phase_state = 1
        self._phase_steps = 0

        self.base_phase = 0
        self._stop_count = 0

        self._plan_ready = False
        self._waypoints = []
        self._wp_idx = 0
        self._initial_aligning = True
        self._final_aligning = False
        self._final_radial_approach = False
        self._reached_last_wp = False
        self._nav_done = False
        self._final_align_stable_count = 0
        self._final_align_omega_cmd_f = 0.0
        self._route_reused = False
        self._route_source_code = 0
        self._route_loaded_from_disk = False

        self._carry_latch = True
        self._released = False
        self._release_ready_count = 0
        self._place_stable_count = 0
        self._placed_success = False
        self._prev_gripper_open_cmd = False
        self._p2_returning_home = False

        if bool(self._carry_latch):
            self._j1_j5_locked = True

    def _stage_setup_phase2(self):
        self.phase_state = 2
        self._phase_steps = 0
        self._release_ready_count = 0

        self._hard_stop_base()
        self._zero_base_qvel_like_extract()

        self.base_phase = 3
        self._plan_ready = False
        self._waypoints = []
        self._wp_idx = 0
        self._initial_aligning = False
        self._final_aligning = False
        self._final_radial_approach = False
        self._reached_last_wp = True
        self._nav_done = True
        self._stop_count = 0
        self._route_reused = False
        self._route_source_code = 0
        self._route_loaded_from_disk = False

        self._carry_latch = False
        self._released = False
        self._place_stable_count = 0
        self._placed_success = False
        self._prev_gripper_open_cmd = False
        self._p2_returning_home = False

    # ========================================================================
    # RESET
    # ========================================================================
    def reset_model(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0.0

        self._reset_debug_all_phases()
        self._stage_setup_phase0()

        self._steps = 0
        self._phase_steps = 0
        self.base_phase = 0
        self._last_action_base[:] = 0.0

        self._last_residual_safe_corridor = False
        self._last_residual_block_reason = "reset"
        self._last_u_lin_raw = 0.0
        self._last_u_ang_raw = 0.0
        self._last_u_lin_applied = 0.0
        self._last_u_ang_applied = 0.0

        self._plan_ready = False
        self._waypoints = []
        self._wp_idx = 0
        self._initial_aligning = True
        self._final_aligning = False
        self._final_radial_approach = False
        self._reached_last_wp = False
        self._nav_done = False
        self._final_align_stable_count = 0
        self._final_align_omega_cmd_f = 0.0
        self._route_reused = False
        self._route_source_code = 0
        self._route_loaded_from_disk = False

        self._reverse_active = False
        self._reverse_goal_xy = None
        self._reset_pid_accums()

        self.cube_dropped = False
        self._printed_drop = False
        self._drop_penalty_applied = False

        self.gripping = False
        self._grasp_count = 0

        self._place_stable_count = 0
        self._released = False
        self._stop_count = 0
        self._carry_latch = False
        self._placed_success = False
        self._printed_phase = {0: False, 1: False, 2: False}

        self._release_ready_count = 0
        self._prev_gripper_open_cmd = False
        self._p2_returning_home = False

        self._cube_z_grasp_ref = None
        self._j1_j5_locked = False

        # Peak tracking reset
        self._phi_peak = {0: 0.0, 1: 0.0, 2: 0.0}

        self._stop_base()

        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        self.map_id = self._pick_next_map_id()
        self._set_obstacles_from_map(self.map_id)

        self._cube_mass = self._sample_cube_mass()
        self.model.body_mass[self._cube_bid] = float(self._cube_mass)

        mujoco.mj_forward(self.model, self.data)

        if bool(self.USE_FIXED_INITIALS):
            mid = int(self.map_id)
            if mid not in self.INITIALS_BY_MAP:
                raise RuntimeError(f"Faltan INITIALS_BY_MAP para map_id={mid}")

            base_qpos7 = np.array(self.INITIALS_BY_MAP[mid]["base_qpos7"], dtype=np.float64)
            cube_qpos7 = np.array(self.INITIALS_BY_MAP[mid]["cube_qpos7"], dtype=np.float64)

            dx, dy, dyaw = self._sample_cube_reset_noise()
            if dx != 0.0 or dy != 0.0:
                cube_qpos7[0] += dx
                cube_qpos7[1] += dy

            if dyaw != 0.0:
                w, _xq, _yq, zq = cube_qpos7[3:7].astype(np.float64)
                yaw0 = float(np.arctan2(2.0 * w * zq, 1.0 - 2.0 * zq * zq))
                yawN = float(wrap_angle(yaw0 + dyaw))
                cube_qpos7[3:7] = yaw_to_quat_wxyz(yawN)

            qpos[:7] = base_qpos7
            qpos[-7:] = cube_qpos7

            entry = self.MAP_CACHE.get(int(self.map_id), {})
            entry["base_qpos7"] = base_qpos7.tolist()
            entry["cube_qpos7"] = cube_qpos7.tolist()
            self.MAP_CACHE[int(self.map_id)] = entry

        qpos = self._apply_arm_pose_to_qpos(qpos, self.init_arm_positions_stage0)
        qvel[:] = 0.0
        self.set_state(qpos, qvel)
        self.data.ctrl[:] = 0.0

        self.q_arm_init = self._arm_pose_dict_to_array(self.init_arm_positions_stage0)
        self.q_arm_home = self._arm_pose_dict_to_array(self.init_arm_positions_stage0)

        for i, jn in enumerate(self.arm_joint_names):
            jadr = self._as_int(self.model.joint(jn).qposadr[0])
            self.arm_target[i] = float(np.clip(
                self.data.qpos[jadr], self.ARM_POS_LOW[i], self.ARM_POS_HIGH[i]
            ))

        for i, aid in enumerate(self.arm_act_ids[:5]):
            self.data.ctrl[aid] = float(self.arm_target[i])
        self.data.ctrl[self.grip_aid] = float(self.grip_open)
        mujoco.mj_forward(self.model, self.data)

        # Baselines para shaping potencial
        cube_pos = self._cube_pos()
        cube_reach_pos = self._cube_reach_target_pos()
        ee_pos = self._site_xpos("gripper_site")
        sp_pos = self._site_xpos("setpoint")

        self.d0_ee_cube = float(np.linalg.norm(cube_reach_pos - ee_pos))
        self._e0_ee_cube = float(np.exp(-float(self.K_REACH_DIST) * self.d0_ee_cube))

        self.d0_cube_sp = float(np.linalg.norm(sp_pos - cube_pos))
        self._e0_cube_sp = float(np.exp(-float(self.K_LIFT_DIST) * self.d0_cube_sp))

        obs = self._get_obs()
        self._assert_finite("reset_obs", obs)
        return obs

    # ========================================================================
    # STEP (modular)
    # ========================================================================
    def step(self, action: np.ndarray):
        self._steps += 1
        self._phase_steps += 1

        phase_eval = int(self.phase_state)

        if not self._printed_phase.get(self.phase_state, False):
            budget = {0: self.MAX_STEPS_0, 1: self.MAX_STEPS_1, 2: self.MAX_STEPS_2}[phase_eval]
            print(f"[PHASE {phase_eval}] START (budget={budget})")
            self._printed_phase[self.phase_state] = True

        action = np.clip(action, self.action_space.low, self.action_space.high).astype(np.float32)
        arm_act = action[:5].copy()
        base_act = action[5:7].copy()
        self._last_action_base = base_act.astype(np.float32)

        Ts = float(self.model.opt.timestep) * float(self.frame_skip)

        # Pre-cálculos usados por el gripper (release)
        release_ready, r_release_ready = self._compute_release_readiness()

        # Aplicar acciones
        self._apply_arm_action(arm_act)
        self._apply_gripper_logic(release_ready)
        self._apply_base_action(base_act, Ts)

        # Simular
        self.do_simulation(self.data.ctrl, self.frame_skip)

        # Post-simulación: drop, grasp, locks
        self._post_simulation_updates()

        # Reward + transiciones
        reward, info_extra, advance_to_phase1, advance_to_phase2 = self._compute_phase_reward(
            phase_eval, Ts, r_release_ready
        )
        self.reward = float(reward)

        # Aplicar transiciones DESPUÉS de reward
        if advance_to_phase1:
            self._stage_setup_phase1()
            self._carry_latch = True
        if advance_to_phase2:
            self._stage_setup_phase2()

        # Termination / truncation
        terminated, truncated, term_reason, trunc_reason = self._check_termination(phase_eval)

        if terminated or truncated:
            final_success = {
                0: self._phase_success_flags[0],
                1: self._phase_success_flags[1],
                2: self._phase_success_flags[2] or self._placed_success,
            }[phase_eval]
            reason = term_reason if terminated else trunc_reason
            self._print_phase_debug_summary(phase_eval, bool(final_success), reason)

        if self.render_mode == "human":
            self.render()

        info = {
            "phase_state": int(self.phase_state),
            "phase_steps": int(self._phase_steps),
            "map_id": int(getattr(self, "map_id", 0)),
            "cube_mass": float(getattr(self, "_cube_mass", 0.0)),
            "base_phase": int(getattr(self, "base_phase", 0)),
            "plan_ready": bool(getattr(self, "_plan_ready", False)),
            "route_reused": bool(getattr(self, "_route_reused", False)),
            "route_source_code": int(getattr(self, "_route_source_code", 0)),
            "route_loaded_from_disk": bool(getattr(self, "_route_loaded_from_disk", False)),
            "wp_idx": int(getattr(self, "_wp_idx", 0)),
            "n_waypoints": int(len(getattr(self, "_waypoints", []))),
            "track_err": float(getattr(self, "_last_track_err", 0.0)),
            "cte_signed": float(getattr(self, "_last_cte_signed", 0.0)),
            "yaw_track_err": float(getattr(self, "_last_yaw_track_err", 0.0)),
            "v_des_cmd": float(getattr(self, "_last_v_des_cmd", 0.0)),
            "omega_des_cmd": float(getattr(self, "_last_omega_des_cmd", 0.0)),
            "u_lin_applied": float(getattr(self, "_last_u_lin_applied", 0.0)),
            "u_ang_applied": float(getattr(self, "_last_u_ang_applied", 0.0)),
            "phase0_success": bool(self._phase_success_flags[0]),
            "phase1_success": bool(self._phase_success_flags[1]),
            "phase2_success": bool(self._phase_success_flags[2]),
            "phi_peak_p0": float(self._phi_peak[0]),
            "phi_peak_p1": float(self._phi_peak[1]),
            "phi_peak_p2": float(self._phi_peak[2]),
            "gripping": bool(self.gripping),
            "grasp_strength": float(self._grasp_strength()),
            **info_extra,
        }

        # Stable-Baselines3 Monitor exige que TODAS las claves declaradas en
        # info_keywords existan en el info del último step del episodio. Como
        # phi_0/phi_1/phi_2 y varios sub-rewards solo existen en su fase, aquí
        # se completan con valores neutros para evitar KeyError cuando el
        # episodio termina en una fase distinta.
        monitor_default_info = {
            "phi_0": 0.0,
            "phi_1": 0.0,
            "phi_2": 0.0,
            "r_reach": 0.0,
            "r_grip": 0.0,
            "r_lift": 0.0,
            "r_yaw": 0.0,
            "r_pos": 0.0,
            "r_track": 0.0,
            "r_hold_sp": 0.0,
            "r_transport_posture": 0.0,
            "posture_err_w": 0.0,
            "posture_vel_cost": 0.0,
            "r_place_xy": 0.0,
            "r_place_z": 0.0,
            "r_approach_top": 0.0,
            "r_place_contact": 0.0,
            "r_release_ready": 0.0,
            "r_release_event": 0.0,
            "r_release_ready_dense": 0.0,
            "r_place_stability": 0.0,
            "pen_low_descent": 0.0,
            "cube_goal_xy_error": 0.0,
            "cube_z_abs_error": 0.0,
            "cube_speed": 0.0,
            "cube_ped_contacts": 0,
            "robot_ped_contacts": 0,
            "gripper_ped_contacts": 0,
            "place_stable_count": 0,
            "release_ready_count": 0,
        }
        for k, v in monitor_default_info.items():
            info.setdefault(k, v)

        obs = self._get_obs()
        self._assert_finite("step_obs", obs)
        return obs, float(self.reward), terminated, truncated, info

    # ------------------------------------------------------------------------
    # Sub-operaciones del step
    # ------------------------------------------------------------------------
    def _compute_release_readiness(self) -> Tuple[bool, float]:
        """Determina si el agarre está listo para soltar (pre-simulación)."""
        phase_eval = int(self.phase_state)
        if phase_eval != 2:
            return False, 0.0

        cube_pos = self._cube_pos()
        ped_xy = self._goal_xy()
        dist_xy = float(np.linalg.norm(cube_pos[:2] - ped_xy))
        # Usar error absoluto de altura. La versión anterior solo penalizaba
        # cubo por ENCIMA de Z_CUBE_REST, por lo que podía considerar listo
        # un release demasiado bajo; eso después aparecía como FAIL_P2_Z_NOT_READY
        # en la evaluación, que sí mide |Z cubo - Z_CUBE_REST|.
        err_z = float(abs(cube_pos[2] - self.Z_CUBE_REST))
        v_cube = self._cube_speed()

        p_xy = float(np.clip(1.0 - dist_xy / max(self.RELEASE_DIST_TOL_XY, 1e-6), 0.0, 1.0))
        p_z = float(np.clip(1.0 - err_z / max(self.RELEASE_ERR_Z_MAX, 1e-6), 0.0, 1.0))
        p_speed = float(np.clip(1.0 - v_cube / max(self.RELEASE_V_MAX, 1e-6), 0.0, 1.0))

        if not self._released:
            r_ready = float(0.50 * p_xy + 0.35 * p_z + 0.15 * p_speed)
        else:
            r_ready = 1.0

        ready = bool(
            dist_xy <= self.RELEASE_DIST_TOL_XY
            and err_z <= self.RELEASE_ERR_Z_MAX
            and v_cube <= self.RELEASE_V_MAX
        )
        return ready, r_ready

    def _apply_arm_action(self, arm_act: np.ndarray):
        phase_eval = int(self.phase_state)
        if phase_eval == 2 and self._p2_returning_home:
            desired = self.q_arm_home.copy().astype(np.float32)
        else:
            desired = np.clip(arm_act, self.ARM_POS_LOW, self.ARM_POS_HIGH).astype(np.float32)

        self.arm_target = self._rate_limit_arm_target(desired)
        for i, aid in enumerate(self.arm_act_ids[:5]):
            self.data.ctrl[aid] = float(self.arm_target[i])

    def _apply_gripper_logic(self, release_ready: bool):
        phase_eval = int(self.phase_state)

        cube_pos_pre = self._cube_pos()
        ee_pos_pre = self._site_xpos("gripper_site")
        dist_pre = float(np.linalg.norm(cube_pos_pre - ee_pos_pre))
        close_ok = bool(dist_pre < self.dist_threshold)

        if phase_eval == 2 and bool(self.P2_AUTO_RELEASE):
            if (not self._released) and release_ready:
                self._release_ready_count = min(
                    self._release_ready_count + 1, int(self.RELEASE_HYSTERESIS_STEPS)
                )
            else:
                self._release_ready_count = 0

            auto_open_now = bool(
                (not self._released)
                and (self._release_ready_count >= int(self.RELEASE_HYSTERESIS_STEPS))
            )
            if self._released or auto_open_now:
                self.data.ctrl[self.grip_aid] = float(self.grip_open)
                self._released = True
            else:
                self.data.ctrl[self.grip_aid] = float(self.grip_close)
        else:
            self._release_ready_count = 0
            self.data.ctrl[self.grip_aid] = float(self.grip_close if close_ok else self.grip_open)

        if phase_eval == 1 and self._carry_latch:
            self.data.ctrl[self.grip_aid] = float(self.grip_close)

        # Evento de apertura
        gripper_open_cmd_now = bool(self.data.ctrl[self.grip_aid] <= float(self.grip_open) + 1e-6)
        self._gripper_open_event = bool((not self._prev_gripper_open_cmd) and gripper_open_cmd_now)
        self._prev_gripper_open_cmd = gripper_open_cmd_now

        if self._gripper_open_event:
            self._released = True
            self._release_ready_count = 0
            if phase_eval == 2 and bool(self.P2_RETURN_HOME_AFTER_RELEASE):
                self._p2_returning_home = True

        if phase_eval != 2:
            self._p2_returning_home = False

    def _apply_base_action(self, base_act: np.ndarray, Ts: float):
        phase_eval = int(self.phase_state)

        if phase_eval == 0:
            self._hard_stop_base()
            return

        if phase_eval == 1:
            if self.base_phase == 0:
                self.base_phase = 1
                self._start_reverse(distance_m=float(self.RETREAT_DIST))

            if self.base_phase == 1:
                if (self._reverse_goal_xy is None) or (not self._reverse_active):
                    self._reverse_active = False
                    self.base_phase = 2
                    self._hard_stop_base()
                    self._zero_base_qvel_like_extract()
                    self._reset_pid_accums()
                else:
                    x, y = self._base_xy()
                    pos = np.array([x, y], dtype=np.float64)
                    rem = float(np.linalg.norm(self._reverse_goal_xy - pos))
                    slow_near = float(np.clip(rem / max(self.REV_SLOW_DIST, 1e-6), 0.2, 1.0))
                    v_des = -float(self.RETREAT_V_MAX) * slow_near
                    self._control_base_vel(Ts, v_des=v_des, omega_des=0.0)

                    if rem <= float(self.RETREAT_TOL):
                        self._hard_stop_base()
                        self._zero_base_qvel_like_extract()
                        self._reset_pid_accums()
                        self._reverse_active = False
                        self.base_phase = 2
                        print("[REV] Retroceso PID completado.")

            elif self.base_phase == 2:
                if self._try_reuse_cached_route():
                    self.base_phase = 3
                    self._reset_pid_accums()
                else:
                    ok = self._plan_direct_goal_stop()
                    self.base_phase = 3 if ok else 1
                    self._reset_pid_accums()

            elif self.base_phase >= 3:
                self._control_base_step(
                    Ts,
                    v_res_u=float(base_act[0]),
                    omega_res_u=float(base_act[1]),
                )
            return

        # Fase 2
        self._hard_stop_base()

    def _post_simulation_updates(self):
        cube_pos = self._cube_pos()
        cube_height = float(cube_pos[2])

        if (not self.cube_dropped) and (cube_height < self.drop_height):
            self.cube_dropped = True
            if not self._printed_drop:
                print(f"[DROP] step={self._steps} phase={self.phase_state} z={cube_height:.3f}")
                self._printed_drop = True

        if not self.cube_dropped:
            # En Fase 2, una vez que se liberó el cubo, no re-enganchar el grasp flag
            if int(self.phase_state) == 2 and self._released:
                self.gripping = False
                self._grasp_count = 0
            else:
                self._update_grasp_flag()
        else:
            self.gripping = False
            self._grasp_count = 0

        self._update_j1_j5_lock_state()

    # ------------------------------------------------------------------------
    # Recompensa por fase (con peak tracking)
    # ------------------------------------------------------------------------
    def _compute_phase_reward(self, phase_eval: int, Ts: float, r_release_ready: float):
        """
        Retorna:
            reward (float),
            info_extra (dict),
            advance_to_phase1 (bool),
            advance_to_phase2 (bool)
        """
        # Costos comunes
        pen_drop = 0.0
        if self.cube_dropped and (not self._drop_penalty_applied):
            pen_drop = float(self.drop_penalty)
            self._drop_penalty_applied = True

        E_tot, E_arm, E_base = self._energy_cost(Ts)
        if phase_eval == 2:
            pen_time = -float(self.W_TIME_P2)
            pen_energy = -float(self.W_EN_ARM_P2) * E_arm - float(self.W_EN_BASE_P2) * E_base
        else:
            pen_time = -float(self.W_TIME)
            pen_energy = -float(self.W_EN_ARM) * E_arm - float(self.W_EN_BASE) * E_base

        # Rama por fase
        if phase_eval == 0:
            r_total, info_extra, advance_p1 = self._reward_phase0(
                pen_time, pen_energy, pen_drop
            )
            return r_total, info_extra, advance_p1, False

        if phase_eval == 1:
            r_total, info_extra, advance_p2 = self._reward_phase1(
                pen_time, pen_energy, pen_drop
            )
            return r_total, info_extra, False, advance_p2

        # Fase 2
        r_total, info_extra = self._reward_phase2(
            pen_time, pen_energy, pen_drop, r_release_ready
        )
        return r_total, info_extra, False, False

    # ------------------------------------------------------------------------
    # Fase 0 reward
    # ------------------------------------------------------------------------
    def _reward_phase0(self, pen_time, pen_energy, pen_drop):
        cube_pos = self._cube_pos()
        cube_reach_pos = self._cube_reach_target_pos()
        ee_pos = self._site_xpos("gripper_site")
        sp_pos = self._site_xpos("setpoint")

        dist_ee_cube = float(np.linalg.norm(cube_reach_pos - ee_pos))
        dist_cube_sp = float(np.linalg.norm(sp_pos - cube_pos))

        self.r_reach = self._safe_exp_shift_norm(dist_ee_cube, self._e0_ee_cube, self.K_REACH_DIST)

        gs = self._grasp_strength()
        self.r_grip = gs
        self.r_lift = gs * self._safe_exp_shift_norm(dist_cube_sp, self._e0_cube_sp, self.K_LIFT_DIST)

        # Φ_0 ∈ [0,1]
        phi = (
            self.p0_w_reach * float(self.r_reach)
            + self.p0_w_grip * float(self.r_grip)
            + self.p0_w_lift * float(self.r_lift)
        )
        phi = float(np.clip(phi, 0.0, 1.0))

        peak_prev = float(self._phi_peak[0])
        peak_gain = max(0.0, phi - peak_prev)
        self._phi_peak[0] = max(peak_prev, phi)

        r_dense = float(self.W_DENSE_P0) * phi
        r_peak = float(self.W_PEAK_P0) * peak_gain

        # Éxito fase 0
        advance_to_phase1 = False
        bonus_final = 0.0
        if (dist_cube_sp <= self.PHASE0_SUCCESS_DIST_SP) and (not self.cube_dropped) and self.gripping:
            if not self._phase_success_flags[0]:
                bonus_final = float(self.BONUS_FINAL_P0)
            self._phase_success_flags[0] = True
            advance_to_phase1 = True

        reward = (
            r_dense + r_peak + bonus_final
            + pen_time + pen_energy + pen_drop
        )

        self._accum_phase_debug(
            0, reward,
            phi_dense=r_dense,
            phi_peak_gain=r_peak,
            bonus_final=bonus_final,
            pen_time=pen_time,
            pen_energy=pen_energy,
            pen_drop=pen_drop,
        )

        info_extra = {
            "phi_0": float(phi),
            "r_reach": float(self.r_reach),
            "r_grip": float(self.r_grip),
            "r_lift": float(self.r_lift),
        }
        return float(reward), info_extra, advance_to_phase1

    # ------------------------------------------------------------------------
    # Fase 1 reward
    # ------------------------------------------------------------------------
    def _reward_phase1(self, pen_time, pen_energy, pen_drop):
        bxy = self._base_xy()
        gxy = self._goal_xy()

        # r_pos: anillo orbital
        d_bg = float(np.linalg.norm(gxy - bxy))
        e_pos = abs(d_bg - float(self.FINAL_GOAL_RADIUS_STOP))
        self.r_pos = float(np.exp(-(e_pos / self.SIGMA_POS) ** 2))

        # r_yaw
        yaw_base = self._base_yaw()
        yaw_to_goal = float(np.arctan2(gxy[1] - bxy[1], gxy[0] - bxy[0]))
        e_yaw = abs(float(wrap_angle(yaw_to_goal - yaw_base)))
        self.r_yaw = float(np.exp(-(e_yaw / self.SIGMA_YAW) ** 2))

        # r_track
        e_track = self._track_error_to_path(bxy)
        self.r_track = float(np.exp(-(e_track / self.SIGMA_TRACK) ** 2))

        # r_hold_sp (cubo pegado al setpoint)
        cube_pos = self._cube_pos()
        sp_pos = self._site_xpos("setpoint")
        dist_cube_sp = float(np.linalg.norm(sp_pos - cube_pos))
        r_hold_sp = (
            float(np.exp(-(dist_cube_sp / self.HOLD_SP_SIGMA) ** 2))
            if (self.gripping and not self.cube_dropped)
            else 0.0
        )

        # Postura
        q_arm = self._get_arm_qpos().astype(np.float32)
        q_ref = self.q_arm_carry_ref.astype(np.float32)
        qd_arm = self._get_arm_qvel().astype(np.float32)
        dq = q_arm - q_ref
        posture_err_w = float(np.sum(self.POSTURE_W_JOINTS_P1 * (dq ** 2)))
        posture_vel_cost = float(np.sum(self.POSTURE_W_JOINTS_P1 * (qd_arm ** 2)))
        r_transport_posture = float(np.exp(-self.K_POSTURE_P1 * posture_err_w))

        pen_posture_err = float(self.W_POSTURE_P1) * (r_transport_posture - 1.0)  # ∈ [-W,0]
        # Clip de penalización por velocidad (evita explosiones)
        pen_posture_vel = -float(self.W_POSTURE_VEL_P1) * float(
            np.clip(posture_vel_cost, 0.0, 100.0)
        )

        # Φ_1 ∈ [0,1]
        phi = (
            self.p1_w_yaw * self.r_yaw
            + self.p1_w_pos * self.r_pos
            + self.p1_w_track * self.r_track
            + self.p1_w_hold_sp * r_hold_sp
        )
        phi = float(np.clip(phi, 0.0, 1.0))

        peak_prev = float(self._phi_peak[1])
        peak_gain = max(0.0, phi - peak_prev)
        self._phi_peak[1] = max(peak_prev, phi)

        r_dense = float(self.W_DENSE_P1) * phi
        r_peak = float(self.W_PEAK_P1) * peak_gain

        # Éxito fase 1: llegada geométrica + detenido + cubo vivo y sujeto
        arrived_geom = (e_pos <= float(self.FINAL_RADIUS_TOL))
        stopped_now = self._base_is_stopped()
        if stopped_now:
            self._stop_count = min(self._stop_count + 1, 10_000)
        else:
            self._stop_count = max(self._stop_count - 1, 0)

        arrived_stopped = (
            arrived_geom
            and (self._stop_count >= int(self.STOP_COUNT_MIN))
            and self.gripping
            and (not self.cube_dropped)
        )

        advance_to_phase2 = False
        bonus_final = 0.0
        if arrived_stopped:
            if not self._phase_success_flags[1]:
                bonus_final = float(self.BONUS_FINAL_P1)
            self._phase_success_flags[1] = True
            advance_to_phase2 = True

        reward = (
            r_dense + r_peak + bonus_final
            + pen_posture_err + pen_posture_vel
            + pen_time + pen_energy + pen_drop
        )

        self._accum_phase_debug(
            1, reward,
            phi_dense=r_dense,
            phi_peak_gain=r_peak,
            bonus_final=bonus_final,
            pen_time=pen_time,
            pen_energy=pen_energy,
            pen_drop=pen_drop,
            pen_posture_err=pen_posture_err,
            pen_posture_vel=pen_posture_vel,
        )

        info_extra = {
            "phi_1": float(phi),
            "r_yaw": float(self.r_yaw),
            "r_pos": float(self.r_pos),
            "r_track": float(self.r_track),
            "r_hold_sp": float(r_hold_sp),
            "r_transport_posture": float(r_transport_posture),
            "posture_err_w": float(posture_err_w),
            "posture_vel_cost": float(posture_vel_cost),
        }
        return float(reward), info_extra, advance_to_phase2

    # ------------------------------------------------------------------------
    # Fase 2 reward
    # ------------------------------------------------------------------------
    def _reward_phase2(self, pen_time, pen_energy, pen_drop, r_release_ready):
        cube_pos = self._cube_pos()
        ee_pos = self._site_xpos("gripper_site")
        ped_xy = self._goal_xy()

        cube_xy = cube_pos[:2]
        cube_z = float(cube_pos[2])
        dist_xy = float(np.linalg.norm(cube_xy - ped_xy))
        # Error absoluto de altura para alinear reward, readiness y evaluación.
        # Si se usa max(cube_z-rest, 0), el agente no recibe castigo por bajar
        # el cubo bajo la altura de reposo, generando releases prematuros.
        cube_z_abs_error = float(abs(cube_z - self.Z_CUBE_REST))
        err_z = cube_z_abs_error
        cube_speed = float(self._cube_speed())
        ee_height_above_ped = self._cube_clearance_over_pedestal()

        # ---- r_place_xy con gate de altura ----
        # Cuando el EEF está lejos del centro XY, exigir que esté alto.
        # Cuando ya está centrado, permitir descenso sin penalización.
        raw_place_xy = float(np.exp(-self.K_PLACE_XY * dist_xy))
        height_gate = float(np.clip(
            ee_height_above_ped / max(self.APPROACH_TOP_Z_MIN, 1e-6), 0.0, 1.0
        ))
        xy_far_ratio = float(np.clip(dist_xy / max(self.PED_RADIUS, 1e-6), 0.0, 1.0))
        # Lejos del centro → necesita altura (gate activo)
        # Centrado → gate = 1.0 (sin restricción)
        effective_gate = 1.0 - xy_far_ratio * (1.0 - height_gate)
        r_place_xy = raw_place_xy * effective_gate

        r_place_z = float(np.exp(-self.K_PLACE_Z * err_z))

        cube_ped_contact = bool(self._count_cube_pedestal_contacts() > 0)

        # ---- r_approach_top: activo SIEMPRE que no se haya soltado ----
        # (antes se desactivaba con contacto, premiando arrastre)
        if not self._released:
            ee_xy = ee_pos[:2]
            dz_ee_ped = ee_height_above_ped
            r_ap_xy = float(np.exp(-self.K_APPROACH * float(np.linalg.norm(ee_xy - ped_xy))))

            if dz_ee_ped <= self.APPROACH_TOP_Z_MIN:
                r_ap_z = float(max(dz_ee_ped / max(self.APPROACH_TOP_Z_MIN, 1e-6), 0.0))
            elif dz_ee_ped <= self.APPROACH_TOP_Z_MAX:
                r_ap_z = 1.0
            else:
                r_ap_z = float(np.exp(-self.K_APPROACH * (dz_ee_ped - self.APPROACH_TOP_Z_MAX)))

            r_approach_top = float(np.clip(r_ap_xy * r_ap_z, 0.0, 1.0))
        else:
            r_approach_top = 0.0

        if cube_ped_contact and self._released and (not self.gripping):
            r_place_contact = float(np.exp(-self.K_PLACE_XY * dist_xy))
        else:
            r_place_contact = 0.0

        # ---- Estabilidad suave ----
        # No es condición dura de éxito: solo da una señal positiva cuando el
        # cubo está quieto durante contacto/release. Ayuda a reducir
        # FAIL_P2_UNSTABLE_PLACEMENT sin bloquear comportamientos útiles.
        if cube_ped_contact or self._released:
            r_place_stability = float(np.exp(-self.K_PLACE_STABILITY * cube_speed))
        else:
            r_place_stability = 0.0

        r_release_ready_dense = float(self.W_RELEASE_READY_DENSE) * float(r_release_ready)
        r_stability_dense = float(self.W_PLACE_STABILITY_P2) * float(r_place_stability)

        # ---- Penalización por descenso prematuro ----
        # Si el gripper baja demasiado mientras el cubo aún está descentrado,
        # se penaliza de forma suave. No se bloquea el release.
        pen_low_descent = 0.0
        if (not self._released) and dist_xy > float(self.P2_SAFE_DESCENT_XY_TOL):
            low_excess = max(0.0, float(self.P2_LOW_DESCENT_HEIGHT) - ee_height_above_ped)
            if low_excess > 0.0:
                pen_low_descent = -float(self.W_LOW_DESCENT_P2) * float(
                    np.clip(low_excess / max(float(self.P2_LOW_DESCENT_HEIGHT), 1e-6), 0.0, 1.0)
                )

        # ---- Penalización por arrastre ----
        # Cubo toca pedestal pero lejos del centro y aún no soltado
        pen_drag = 0.0
        if cube_ped_contact and (not self._released) and dist_xy > self.PED_RADIUS:
            pen_drag = -float(self.W_DRAG_P2) * float(dist_xy / max(self.PED_RADIUS, 1e-6))

        # Evento de apertura
        r_release_event = 0.0
        pen_open_outside = 0.0
        early_release_pen = 0.0
        if getattr(self, "_gripper_open_event", False):
            ready_at_event, _ = self._compute_release_readiness()
            if ready_at_event or r_release_ready >= 0.95:
                r_release_event = float(self.W_RELEASE_EVENT)
            else:
                pen_open_outside = float(self.PEN_OPEN_OUTSIDE)
                if dist_xy > float(self.RELEASE_DIST_TOL_XY) * 1.5:
                    early_release_pen = float(self.PEN_EARLY_RELEASE)

        # Colisiones pedestal (solo fase 2)
        robot_ped_col = self._count_robot_pedestal_contacts() > 0
        gripper_ped_col = self._count_gripper_pedestal_contacts() > 0
        pen_ped = -float(self.W_PED_P2) if robot_ped_col else 0.0
        pen_gripper_ped = -float(self.W_GRIPPER_PED) if gripper_ped_col else 0.0

        # Φ_2 ∈ [0,1]
        phi = (
            self.p2_w_xy * r_place_xy
            + self.p2_w_z * r_place_z
            + self.p2_w_approach * r_approach_top
            + self.p2_w_contact * r_place_contact
        )
        phi = float(np.clip(phi, 0.0, 1.0))

        peak_prev = float(self._phi_peak[2])
        peak_gain = max(0.0, phi - peak_prev)
        self._phi_peak[2] = max(peak_prev, phi)

        r_dense = float(self.W_DENSE_P2) * phi
        r_peak = float(self.W_PEAK_P2) * peak_gain

        # Éxito estable
        placed_stable_now = bool(
            cube_ped_contact
            and self._released
            and (not self.gripping)
            and dist_xy <= self.RELEASE_DIST_TOL_XY * 2.0  # centrado
        )

        if placed_stable_now:
            self._place_stable_count += 1
        else:
            self._place_stable_count = 0

        bonus_final = 0.0
        if (self._place_stable_count >= int(self.PLACE_STABLE_COUNT_MIN)) and (not self._placed_success):
            self._placed_success = True
            self._phase_success_flags[2] = True
            bonus_final = float(self.BONUS_FINAL_P2)
            # Activar retorno a home
            self._p2_returning_home = True

        if self._placed_success:
            # Post-colocación: reward continuo, sin penalizaciones
            reward = (
                    r_dense + r_peak + bonus_final
                    + float(self.W_HOLD_PLACED)  # reward constante por mantener cubo colocado
                    + pen_drop  # solo drop penalty se mantiene
            )
        else:
            # Pre-colocación: lógica normal con todas las penalizaciones
            reward = (
                    r_dense + r_peak + bonus_final + r_release_event
                    + r_release_ready_dense + r_stability_dense
                    + pen_time + pen_energy + pen_drop
                    + pen_ped + pen_gripper_ped + pen_drag + pen_low_descent
                    + pen_open_outside + early_release_pen
            )

        self._accum_phase_debug(
            2, reward,
            phi_dense=r_dense,
            phi_peak_gain=r_peak,
            bonus_final=bonus_final,
            r_release_event=r_release_event,
            r_release_ready_dense=r_release_ready_dense,
            r_place_stability=r_stability_dense,
            pen_low_descent=pen_low_descent,
            pen_time=pen_time,
            pen_energy=pen_energy,
            pen_drop=pen_drop,
            pen_ped=pen_ped,
            pen_gripper_ped=pen_gripper_ped,
            pen_open_outside=pen_open_outside,
            early_release_pen=early_release_pen,
            pen_drag=pen_drag,
        )

        info_extra = {
            "phi_2": float(phi),
            "r_place_xy": float(r_place_xy),
            "r_place_z": float(r_place_z),
            "r_approach_top": float(r_approach_top),
            "r_place_contact": float(r_place_contact),
            "r_release_ready": float(r_release_ready),
            "r_release_event": float(r_release_event),
            "r_release_ready_dense": float(r_release_ready_dense),
            "r_place_stability": float(r_stability_dense),
            "pen_low_descent": float(pen_low_descent),
            "cube_goal_xy_error": float(dist_xy),
            "cube_z_abs_error": float(cube_z_abs_error),
            "cube_speed": float(cube_speed),
            "cube_ped_contacts": int(self._count_cube_pedestal_contacts()),
            "robot_ped_contacts": int(self._count_robot_pedestal_contacts()),
            "gripper_ped_contacts": int(self._count_gripper_pedestal_contacts()),
            "place_stable_count": int(self._place_stable_count),
            "release_ready_count": int(self._release_ready_count),
        }
        return float(reward), info_extra

    # ------------------------------------------------------------------------
    # Termination / truncation
    # ------------------------------------------------------------------------
    def _check_termination(self, phase_eval: int):
        terminated = False
        truncated = False
        term_reason = ""
        trunc_reason = ""

        if self.cube_dropped:
            terminated = True
            term_reason = "drop"

        # Terminación exitosa: cubo colocado Y brazo en home
        if (not terminated) and (phase_eval == 2) and self._placed_success:
            if self._arm_is_at_home():
                terminated = True
                term_reason = "place_success_arm_home"

        if not terminated:
            budget = {0: self.MAX_STEPS_0, 1: self.MAX_STEPS_1, 2: self.MAX_STEPS_2}[self.phase_state]
            if self._phase_steps >= int(budget):
                truncated = True
                trunc_reason = f"phase{self.phase_state}_budget"

        self._last_termination_reason = str(term_reason)
        self._last_truncation_reason = str(trunc_reason)
        return terminated, truncated, term_reason, trunc_reason

    # ========================================================================
    # OBS (62 dims)
    # ========================================================================
    def _get_obs(self):
        p0 = int(self.phase_state) == 0
        p1 = int(self.phase_state) == 1
        p2 = int(self.phase_state) == 2

        q_arm = self._get_arm_qpos()
        dq_arm = self._get_arm_qvel()
        delta_q_init = q_arm - self.q_arm_init

        pos_base = self._site_xpos("chassis_frame")
        pos_ee = self._site_xpos("gripper_site")
        pos_sp = self._site_xpos("setpoint")
        pos_cube = self._cube_pos()
        cube_vel = self._cube_vel_vec()
        cube_avel = self._cube_angular_vel()
        cube_quat = self._cube_quat_wxyz()

        goal_place_target = self._goal_place_target_pos()

        if p2:
            p_rel_ee_cube = np.zeros(3, dtype=np.float32)
            dist_ee_cube = 0.0
            dist_cube_sp = 0.0
            is_lifted = 0.0
        else:
            p_rel_ee_cube = (pos_cube - pos_ee).astype(np.float32)
            dist_ee_cube = float(np.linalg.norm(p_rel_ee_cube))
            dist_cube_sp = float(np.linalg.norm(pos_sp - pos_cube))
            is_lifted = 1.0 if float(pos_cube[2]) > 0.70 else 0.0

        p_cube_goal = (pos_cube - goal_place_target).astype(np.float32)
        dist_cube_goal = float(np.linalg.norm(p_cube_goal))
        p_rel_ee_goal = (goal_place_target - pos_ee).astype(np.float32)

        mat_base = self.data.site("chassis_frame").xmat.reshape(3, 3)
        yaw_base = float(np.arctan2(mat_base[1, 0], mat_base[0, 0]))
        vec_bc = pos_cube[:2] - pos_base[:2]
        yaw_cube = float(np.arctan2(vec_bc[1], vec_bc[0]))
        yaw_rel = float(wrap_angle(yaw_cube - yaw_base))

        # Navegación (sólo fase 1)
        if p1:
            plan_ready = 1.0 if self._plan_ready else 0.0
            reached_last_wp = 1.0 if self._reached_last_wp else 0.0
            nav_done = 1.0 if self._nav_done else 0.0
            n_wp = len(self._waypoints) if self._waypoints else 0
            denom = float(max(n_wp - 1, 1))
            wp_idx_norm = float(np.clip(self._wp_idx / denom, 0.0, 1.0))
            n_wp_norm = float(np.clip(n_wp / 200.0, 0.0, 1.0))
            if (n_wp > 0) and self._plan_ready:
                track_err = float(self._track_error_to_path(np.array(self._base_xy(), dtype=np.float64)))
            else:
                track_err = 0.0
            track_err_norm = float(np.clip(track_err / 0.5, 0.0, 1.0))
            stop_count_norm = float(np.clip(
                self._stop_count / float(max(self.STOP_COUNT_MIN, 1)), 0.0, 1.0
            ))
            last_action_base = self._last_action_base.astype(np.float32)
        else:
            plan_ready = 0.0
            reached_last_wp = 0.0
            nav_done = 0.0
            wp_idx_norm = 0.0
            n_wp_norm = 0.0
            track_err_norm = 0.0
            stop_count_norm = 0.0
            last_action_base = np.zeros(2, dtype=np.float32)

        # Flags fase 2
        if p2:
            released = 1.0 if self._released else 0.0
            place_cnt_norm = float(np.clip(
                self._place_stable_count / float(max(self.PLACE_STABLE_COUNT_MIN, 1)),
                0.0, 1.0,
            ))
            placed_success = 1.0 if self._placed_success else 0.0
            release_ready_norm = float(np.clip(
                self._release_ready_count / float(max(self.RELEASE_HYSTERESIS_STEPS, 1)),
                0.0, 1.0,
            ))
            cube_vel_obs = cube_vel.astype(np.float32)
            cube_avel_obs = cube_avel.astype(np.float32)
        else:
            released = 0.0
            place_cnt_norm = 0.0
            placed_success = 0.0
            release_ready_norm = 0.0
            cube_vel_obs = np.zeros(3, dtype=np.float32)
            cube_avel_obs = np.zeros(3, dtype=np.float32)

        carry_latch = 1.0 if (p1 and self._carry_latch) else 0.0
        grasp_strength = float(self._grasp_strength())

        arm_target_obs = np.asarray(self.arm_target, dtype=np.float32)
        arm_track_err = (arm_target_obs - q_arm.astype(np.float32)).astype(np.float32)

        m_cube = float(getattr(self, "_cube_mass", None) or 1.0)
        _ms = getattr(self, "MASS_SET", None) or [0.5, 1.5]
        m_lo, m_hi = float(min(_ms)), float(max(_ms))
        m_mid = 0.5 * (m_hi + m_lo)
        m_half = max(0.5 * (m_hi - m_lo), 1e-6)
        mass_norm = float((m_cube - m_mid) / m_half)          # ~[-1, 1]

        # Proximidad a la caja que dispara el auto-release (post-simulación).
        if p2:
            _dxy = float(np.linalg.norm(pos_cube[:2] - self._goal_xy()))
            _dz = float(abs(float(pos_cube[2]) - float(self.Z_CUBE_REST)))
            _dv = float(np.linalg.norm(cube_vel))
            rel_p_xy = float(np.clip(1.0 - _dxy / max(self.RELEASE_DIST_TOL_XY, 1e-6), 0.0, 1.0))
            rel_p_z  = float(np.clip(1.0 - _dz  / max(self.RELEASE_ERR_Z_MAX,   1e-6), 0.0, 1.0))
            rel_p_v  = float(np.clip(1.0 - _dv  / max(self.RELEASE_V_MAX,       1e-6), 0.0, 1.0))
        else:
            rel_p_xy = rel_p_z = rel_p_v = 0.0

        # One-hot fases (3 + 4 = 7 dims, antes eran 2 floats)
        ph_oh = one_hot(int(self.phase_state), 3)
        bp_oh = one_hot(int(self.base_phase), 4)

        obs = np.concatenate(
            [
                q_arm.astype(np.float32),                      # 5
                dq_arm.astype(np.float32),                     # 5  -> 10
                delta_q_init.astype(np.float32),               # 5  -> 15
                p_rel_ee_cube.astype(np.float32),              # 3  -> 18
                np.array([dist_ee_cube], np.float32),          # 1  -> 19
                p_cube_goal.astype(np.float32),                # 3  -> 22
                np.array([dist_cube_goal], np.float32),        # 1  -> 23
                np.array([float(pos_cube[2]), is_lifted], np.float32),  # 2 -> 25
                np.array([yaw_rel], np.float32),               # 1  -> 26
                np.array([float(self.gripping), float(self.cube_dropped),
                          grasp_strength], np.float32),        # 3  -> 29
                np.array([dist_cube_sp], np.float32),          # 1  -> 30
                last_action_base,                              # 2  -> 32
                p_rel_ee_goal.astype(np.float32),              # 3  -> 35
                cube_vel_obs,                                  # 3  -> 38
                cube_avel_obs,                                 # 3  -> 41
                cube_quat.astype(np.float32),                  # 4  -> 45
                np.array([
                    plan_ready,
                    wp_idx_norm,
                    n_wp_norm,
                    reached_last_wp,
                    nav_done,
                    track_err_norm,
                    stop_count_norm,
                    released,
                    place_cnt_norm,
                    placed_success,
                    carry_latch,
                    release_ready_norm,
                ], dtype=np.float32),                          # 12 -> 57
                arm_target_obs,                                # 5  -> 62
                arm_track_err,                                 # 5  -> 67
                np.array([
                    mass_norm, rel_p_xy, rel_p_z, rel_p_v,
                ], dtype=np.float32),                          # 4  -> 71
                ph_oh,                                         # 3  -> 74
                bp_oh,                                         # 4  -> 78                                    # 4  -> 62
            ],
            dtype=np.float32,
        )

        if obs.shape != (78,):
            raise RuntimeError(
                f"Observación ZennoV7 inválida: shape={obs.shape}, "
                "esperada=(78,)."
            )
        self._assert_finite("obs", obs)
        return obs

    # ========================================================================
    # UTILS
    # ========================================================================
    def _safe_exp_shift_norm(self, d: float, e0: Optional[float], k: float) -> float:
        if e0 is None or (not np.isfinite(e0)):
            return 0.0
        if d is None or (not np.isfinite(d)):
            return 0.0
        return self._exp_shift_norm(float(d), float(e0), float(k))

    def _assert_finite(self, name: str, arr):
        arr = np.asarray(arr, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            bad = arr[~np.isfinite(arr)]
            raise RuntimeError(f"[{name}] contiene NaN/Inf. Muestra inválida: {bad[:10]}")

# ============================================================================
# V7 ACTIVO: NAVEGACIÓN + REWARD PLACE + OBSERVACIÓN EXTENDIDA
# ============================================================================
HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "Zenno_Model" / "scene.xml"
ROUTES_PATH = HERE / "RUTAS_SAC"

ACTION_DIM = 7
OBS_DIM = 78
VALID_BASE_MODES = ("residual_pid",)
VALID_OBJECTIVES = ("balanced",)

class ZennoV7(_ZennoReference):
    """ZennoV7: P2 secuencial XY -> Z -> release -> home, obs=78."""

    PID_PROFILE_NAME = "PID_PSO_6D_MODEL_DYNAMICS_STRICT_20MM_3DEG"
    TRACKING_CONTROLLER = "pure_pursuit"
    LATERAL_ACCEPTANCE_LIMIT_M = 0.020
    ORIENTATION_ACCEPTANCE_LIMIT_RAD = float(np.deg2rad(3.0))

    def __init__(
        self,
        model_path: Optional[str] = None,
        frame_skip: int = 5,
        routes_path: Optional[str] = None,
        base_control_mode: str = "residual_pid",
        objective: str = "balanced",
        debug_rewards: bool = False,
        curriculum_enabled: bool = False,
        curriculum_start_probs: Sequence[float] = (1.0, 0.0, 0.0),
        **kwargs,
    ):
        if base_control_mode not in VALID_BASE_MODES:
            raise ValueError(
                "zenno_sac conserva el control residual de ZennoV3; "
                f"base_control_mode debe ser 'residual_pid', no {base_control_mode!r}."
            )
        if objective not in VALID_OBJECTIVES:
            raise ValueError(
                "Las recompensas deben permanecer idénticas a ZennoV3; "
                f"objective debe ser 'balanced', no {objective!r}."
            )
        if int(frame_skip) != 5:
            raise ValueError(
                "La sintonización fue hecha para timestep=2 ms y frame_skip=5 "
                "(periodo de control de 10 ms)."
            )

        probs = np.asarray(curriculum_start_probs, dtype=np.float64)
        if curriculum_enabled or probs.shape != (3,) or not np.allclose(
            probs, (1.0, 0.0, 0.0), atol=1e-12
        ):
            raise ValueError(
                "Este entorno reproduce el entrenamiento end-to-end de ZennoV3: "
                "cada episodio debe comenzar en la fase 0 y curriculum_enabled=False."
            )

        self.base_control_mode = base_control_mode
        self.objective = objective
        self.routes_path = (
            Path(routes_path).expanduser().resolve()
            if routes_path is not None
            else ROUTES_PATH
        )
        # La caché puede partir vacía. Se crea aquí y las rutas nuevas se
        # regeneran/guardan con metadatos compatibles con la versión actual.
        self.routes_path.mkdir(parents=True, exist_ok=True)
        self.curriculum_enabled = False
        self.curriculum_start_probs = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.curriculum_start_phase = 0
        self._last_energy_total_j = 0.0
        self._last_energy_arm_j = 0.0
        self._last_energy_base_j = 0.0

        observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )

        xml_path = Path(model_path).expanduser().resolve() if model_path else MODEL_PATH
        if not xml_path.is_file():
            raise FileNotFoundError(f"No se encontró el modelo MuJoCo: {xml_path}")

        # Replica deliberadamente la secuencia de inicialización de ZennoV3. El
        # constructor padre fija una ruta externa al XML, por eso se inicializa
        # aquí con el modelo local sin alterar ninguna de sus subsistemas.
        utils.EzPickle.__init__(
            self,
            model_path=str(xml_path),
            frame_skip=frame_skip,
            routes_path=str(self.routes_path),
            base_control_mode=base_control_mode,
            objective=objective,
            debug_rewards=debug_rewards,
            curriculum_enabled=False,
            curriculum_start_probs=(1.0, 0.0, 0.0),
            **kwargs,
        )
        MujocoEnv.__init__(
            self,
            str(xml_path),
            frame_skip,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        self._init_action_space_and_arm()
        self._init_arm_lock_config()
        self._init_arm_initial_postures()
        self._init_offsets_and_gripper()
        self._init_base_residuals()
        self._init_reward_weights()
        self._init_phase0_params()
        self._init_phase1_params()
        self._init_phase2_params()
        self._init_episode_state()
        self._init_wheels()
        self._init_grasp_contacts()
        self._init_pedestal_collision_sets()
        self._init_gripper_collision_sets()
        self._init_base_dynamics_and_pid()
        self._init_nav_state()
        self._init_maps_and_obstacles()
        self._init_cube_mass_sampling()
        self._init_peak_tracking()
        self._init_debug()

        try:
            from ompl import base as ob  # noqa: F401
            from ompl import geometric as og  # noqa: F401

            self._ompl_ok = True
        except Exception:
            self._ompl_ok = False

        if self.LOAD_ROUTES_ON_START:
            self._load_map_routes_from_folder(
                self.MAPS_DIR, map_ids=[1, 2, 3, 4]
            )
        self._reset_debug_all_phases()
        self.DEBUG_REWARD_PHASES = bool(debug_rewards)

    # ====================================================================
    # REWARD V6
    # ====================================================================
    def _init_reward_weights(self):
        """Conserva V5 salvo el costo temporal y fallos explícitos de fase."""
        super()._init_reward_weights()

        # V5 usaba 0.0002: durante 1000 steps el costo máximo era apenas 0.2.
        # V6 acumula hasta -5.0 en P2, suficiente para distinguir una
        # colocación rápida de otra que agota todo el presupuesto, sin dominar
        # el bonus terminal de 100.
        self.W_TIME_P2 = 0.005

        # Se aplican una sola vez cuando el episodio termina sin completar la
        # fase activa. La penalización crece con el avance de la tarea.
        self.PHASE_FAILURE_PENALTIES = {
            0: 0.0,
            1: 0.0,
            # P2 ya acumula costo temporal y penalizaciones físicas. Un -30
            # terminal hacía que muchos episodios "casi correctos" quedaran
            # dominados por la misma señal negativa. Se conserva un costo de
            # timeout, pero suficientemente pequeño para no borrar el shaping.
            2: 0.0,
        }

    def _init_phase2_params(self):
        """Shaping P2 secuencial: centrar XY -> descender -> estabilizar -> soltar."""
        super()._init_phase2_params()

        self.W_DENSE_P2 = 0.0
        self.W_HOLD_PLACED = 0.0
        self.W_RELEASE_READY_DENSE = 0.0
        self.W_PLACE_STABILITY_P2 = 0.0
        self.W_PEAK_P2 = 0.0

        self.P2_PROGRESS_GAMMA = 1.0
        self.W_P2_PROGRESS = 8.0

        self.W_P2_DESCENT_PROGRESS = 100.0
        self.P2_DESCENT_DELTA_CLIP_M = 0.010

        # El peak de la versión anterior reforzaba demasiado el primer máximo de XY.
        self.W_P2_PEAK_GAIN = 0.0

        self.W_P2_CENTER_EVENT = 2.0
        self.W_P2_DESCENT_EVENT = 6.0
        self.W_P2_RELEASE_EVENT = 12.0
        self.W_P2_HOME_PROGRESS = 12.0
        self.W_P2_HOME_EVENT = 10.0

        # Penalizaciones moderadas. El gripper se excluye de _robot_geom_ids para
        # no castigar dos veces un mismo contacto con el pedestal.
        self.W_PED_P2 = 0.08
        self.W_GRIPPER_PED = 0.12
        self.W_LOW_DESCENT_P2 = 0.03
        self.W_DRAG_P2 = 0.40

        self._reset_v6_place_reward_state()

    def _reset_v6_place_reward_state(self):
        """Reinicia memoria de transición usada exclusivamente por el reward P2."""
        self._v6_prev_place_potential = None
        self._v6_best_place_potential = 0.0
        self._v6_prev_z_error = None
        self._v6_prev_home_potential = None
        self._v6_prev_xy_gate = None
        self._v6_prev_descent_phi = None
        self._v6_center_event_done = False
        self._v6_descent_event_done = False
        self._v6_release_event_done = False
        self._v6_home_event_done = False
        self._v6_release_ready_pre_sim = False

    def _reset_phase_debug_accum(self, phase_id: int):
        """Extiende el debug heredado con métricas del descenso P2."""
        super()._reset_phase_debug_accum(phase_id)
        if int(phase_id) == 2:
            self._phase_debug_accums[int(phase_id)].setdefault(
                "r_descent_progress", 0.0
            )

    def _init_gripper_collision_sets(self):
        """Evita doble penalización del mismo contacto gripper-pedestal."""
        super()._init_gripper_collision_sets()
        if hasattr(self, "_robot_geom_ids"):
            self._robot_geom_ids.difference_update(self._gripper_geom_ids)

    def _apply_gripper_logic(self, release_ready: bool):
        """Guarda la condición pre-sim que realmente autorizó el auto-release."""
        self._v6_release_ready_pre_sim = bool(release_ready)
        return super()._apply_gripper_logic(release_ready)


    @staticmethod
    def _v6_cauchy_score(error: float, scale: float) -> float:
        """Score suave (0, 1] que conserva gradiente lejos del objetivo."""
        ratio = float(error) / max(float(scale), 1e-9)
        return float(1.0 / (1.0 + ratio * ratio))

    @staticmethod
    def _v6_smooth_gate_below(value: float, full_at: float, zero_at: float) -> float:
        """Gate C1: 1 por debajo de full_at y 0 por encima de zero_at."""
        full_at = float(full_at)
        zero_at = float(zero_at)
        if zero_at <= full_at:
            raise ValueError("zero_at debe ser mayor que full_at.")
        x = float(np.clip((zero_at - float(value)) / (zero_at - full_at), 0.0, 1.0))
        return float(x * x * (3.0 - 2.0 * x))

    def _v6_place_potential(self) -> Tuple[float, dict]:
        """Potencial P2 secuencial sin óptimo local de hover."""
        cube_pos = self._cube_pos()
        ee_pos = self._site_xpos("gripper_site")
        ped_xy = self._goal_xy()

        dist_xy = float(np.linalg.norm(cube_pos[:2] - ped_xy))
        z_error = float(abs(float(cube_pos[2]) - float(self.Z_CUBE_REST)))
        cube_speed = float(self._cube_speed())
        ee_xy_error = float(np.linalg.norm(ee_pos[:2] - ped_xy))
        ee_height = self._cube_clearance_over_pedestal()
        
        xy_score = float(
            0.5 * self._v6_cauchy_score(dist_xy, self.P2_XY_SCALE_FAR)
            + 0.5 * self._v6_cauchy_score(dist_xy, self.P2_XY_SCALE_NEAR)
        )
        z_score = self._v6_cauchy_score(z_error, 0.06)
        speed_score = self._v6_cauchy_score(cube_speed, 0.08)
        approach_xy = self._v6_cauchy_score(ee_xy_error, 0.18)

        if ee_height < float(self.APPROACH_TOP_Z_MIN):
            approach_z_error = float(self.APPROACH_TOP_Z_MIN) - ee_height
        elif ee_height > float(self.APPROACH_TOP_Z_MAX):
            approach_z_error = ee_height - float(self.APPROACH_TOP_Z_MAX)
        else:
            approach_z_error = 0.0
        approach_z = self._v6_cauchy_score(approach_z_error, 0.06)
        approach_raw = float(approach_xy * approach_z)

        xy_gate = self._v6_smooth_gate_below(
            dist_xy,
            full_at=float(self.P2_SAFE_DESCENT_XY_TOL),
            zero_at=float(self.P2_XY_GATE_ZERO),
        )
        z_gate = self._v6_smooth_gate_below(
            z_error,
            full_at=float(self.RELEASE_ERR_Z_MAX),
            zero_at=float(self.RELEASE_ERR_Z_MAX) * 2.0,
        )

        approach_score = float((1.0 - xy_gate) * approach_raw)

        cube_ped_contacts = int(self._count_cube_pedestal_contacts())
        contact_score = float(
            cube_ped_contacts > 0 and self._released and (not self.gripping)
        )

        if not self._released:
            phi = (
                0.30 * xy_score
                + 0.55 * xy_gate * z_score
                + 0.10 * approach_score
                + 0.05 * xy_gate * z_gate * speed_score
            )
        else:
            phi = (
                0.25 * xy_score
                + 0.25 * z_score
                + 0.20 * speed_score
                + 0.30 * contact_score
            )

        components = {
            "dist_xy": dist_xy,
            "z_error": z_error,
            "cube_speed": cube_speed,
            "xy_score": xy_score,
            "z_score": z_score,
            "speed_score": speed_score,
            "xy_gate": xy_gate,
            "z_gate": z_gate,
            "approach_raw": approach_raw,
            "approach_score": approach_score,
            "contact_score": contact_score,
            "cube_ped_contacts": cube_ped_contacts,
            "ee_height": ee_height,
        }
        return float(np.clip(phi, 0.0, 1.0)), components

    def _v6_home_potential(self) -> float:
        """Potencial denso para retirar el brazo después de soltar."""
        q_error = float(np.max(np.abs(self._get_arm_qpos() - self.q_arm_home)))
        q_speed = float(np.max(np.abs(self._get_arm_qvel())))
        pos_score = self._v6_cauchy_score(q_error, 0.20)
        vel_score = self._v6_cauchy_score(q_speed, 0.20)
        return float(0.80 * pos_score + 0.20 * vel_score)

    def _stage_setup_phase2(self):
        """Mantiene el setup físico e inicializa referencias del shaping P2."""
        super()._stage_setup_phase2()
        self._reset_v6_place_reward_state()
        phi, comp = self._v6_place_potential()
        self._v6_prev_place_potential = float(phi)
        self._v6_best_place_potential = float(phi)
        self._v6_prev_z_error = float(comp["z_error"])
        self._v6_prev_xy_gate = float(comp["xy_gate"])
        self._v6_prev_descent_phi = float(comp["xy_gate"] * comp["z_score"])

    def reset_model(self):
        observation = super().reset_model()
        self._reset_v6_place_reward_state()
        return observation

    def incremental_arm_target(
        self, arm_action: np.ndarray, phase: Optional[int] = None
    ) -> np.ndarray:
        """Convierte una acción normalizada en target relativo; nunca absoluto."""
        active_phase = int(self.phase_state if phase is None else phase)
        if active_phase not in (0, 1, 2):
            raise ValueError(f"Fase incremental inválida: {active_phase}")
        public_arm = np.clip(
            np.asarray(arm_action, dtype=np.float32).reshape(5), -1.0, 1.0
        )
        return np.clip(
            np.asarray(self.arm_target, dtype=np.float32)
            + self.ARM_DELTA_SCALE_BY_PHASE[active_phase] * public_arm,
            self.ARM_POS_LOW,
            self.ARM_POS_HIGH,
        ).astype(np.float32)

    def _reward_phase2(self, pen_time, pen_energy, pen_drop, r_release_ready):
        """Reward P2: XY -> descenso -> release -> estabilidad -> home."""
        phi, comp = self._v6_place_potential()
        dist_xy = float(comp["dist_xy"])
        z_error = float(comp["z_error"])
        cube_speed = float(comp["cube_speed"])
        xy_gate = float(comp["xy_gate"])
        z_gate = float(comp["z_gate"])
        cube_ped_contacts = int(comp["cube_ped_contacts"])
        cube_ped_contact = bool(cube_ped_contacts > 0)
        opening_event = bool(getattr(self, "_gripper_open_event", False))

        place_progress = 0.0
        if self._v6_prev_place_potential is not None and not opening_event:
            place_progress = float(self.W_P2_PROGRESS) * (
                float(self.P2_PROGRESS_GAMMA) * phi
                - float(self._v6_prev_place_potential)
            )
        self._v6_prev_place_potential = float(phi)

        # ============================================================
        # PROGRESO DE DESCENSO CON GATE CONSERVADOR ENTRE STEPS
        # ============================================================
        # Progreso de descenso como DIFERENCIA DE POTENCIAL.
        # Psi = xy_gate * z_score es una función de estado pura, así que la
        # suma sobre cualquier ciclo cerrado es exactamente 0: bajar centrado
        # y volver a subir descentrado ya no rinde nada. La versión anterior
        # (W * gate * (z_prev - z)) no telescopaba porque el gate cambiaba
        # entre pasos, y ese ciclo rendía +85 por episodio sin colocar el cubo.
        r_descent_progress = 0.0
        descent_delta_m = 0.0
        descent_phi = float(xy_gate * float(comp["z_score"]))

        if (
            not self._released
            and self._v6_prev_descent_phi is not None
            and not opening_event
        ):
            descent_delta_m = float(
                np.clip(
                    float(self._v6_prev_z_error) - z_error,
                    -float(self.P2_DESCENT_DELTA_CLIP_M),
                    float(self.P2_DESCENT_DELTA_CLIP_M),
                )
            ) if self._v6_prev_z_error is not None else 0.0   # solo telemetría

            r_descent_progress = float(self.W_P2_DESCENT_PROGRESS) * (
                descent_phi - float(self._v6_prev_descent_phi)
            )

        self._v6_prev_z_error = float(z_error)
        self._v6_prev_xy_gate = float(xy_gate)
        self._v6_prev_descent_phi = descent_phi

        if opening_event:
            place_peak_gain = 0.0
            self._v6_best_place_potential = float(phi)
        else:
            peak_gain_raw = max(0.0, phi - float(self._v6_best_place_potential))
            place_peak_gain = float(self.W_P2_PEAK_GAIN) * peak_gain_raw
            self._v6_best_place_potential = max(
                float(self._v6_best_place_potential), phi
            )
        self._phi_peak[2] = max(float(self._phi_peak[2]), phi)

        r_center_event = 0.0
        if (
            not self._v6_center_event_done
            and not self._released
            and dist_xy <= float(self.P2_SAFE_DESCENT_XY_TOL)
        ):
            r_center_event = float(self.W_P2_CENTER_EVENT)
            self._v6_center_event_done = True

        r_descent_event = 0.0
        if (
            not self._v6_descent_event_done
            and not self._released
            and dist_xy <= float(self.RELEASE_DIST_TOL_XY) * 1.5
            and z_error <= float(self.RELEASE_ERR_Z_MAX)
        ):
            r_descent_event = float(self.W_P2_DESCENT_EVENT)
            self._v6_descent_event_done = True

        r_release_event = 0.0
        pen_open_outside = 0.0
        early_release_pen = 0.0
        if opening_event:
            correct_release = bool(
                getattr(self, "_v6_release_ready_pre_sim", False)
            )
            if correct_release and not self._v6_release_event_done:
                r_release_event = float(self.W_P2_RELEASE_EVENT)
                self._v6_release_event_done = True
            elif not correct_release:
                pen_open_outside = float(self.PEN_OPEN_OUTSIDE)
                if dist_xy > float(self.RELEASE_DIST_TOL_XY) * 1.5:
                    early_release_pen = float(self.PEN_EARLY_RELEASE)

        ee_height = float(comp["ee_height"])
        pen_low_descent = 0.0
        if (not self._released) and dist_xy > float(self.P2_SAFE_DESCENT_XY_TOL):
            low_excess = max(
                0.0, float(self.P2_LOW_DESCENT_HEIGHT) - ee_height
            )
            if low_excess > 0.0:
                pen_low_descent = -float(self.W_LOW_DESCENT_P2) * float(
                    np.clip(
                        low_excess
                        / max(float(self.P2_LOW_DESCENT_HEIGHT), 1e-6),
                        0.0,
                        1.0,
                    )
                )

        pen_drag = 0.0
        if cube_ped_contact and (not self._released) and dist_xy > self.PED_RADIUS:
            pen_drag = -float(self.W_DRAG_P2) * float(
                dist_xy / max(float(self.PED_RADIUS), 1e-6)
            )

        robot_ped_contacts = int(self._count_robot_pedestal_contacts())
        gripper_ped_contacts = int(self._count_gripper_pedestal_contacts())
        pen_ped = -float(self.W_PED_P2) if robot_ped_contacts > 0 else 0.0
        pen_gripper_ped = (
            -float(self.W_GRIPPER_PED) if gripper_ped_contacts > 0 else 0.0
        )

        placed_stable_now = bool(
            cube_ped_contact
            and self._released
            and (not self.gripping)
            and dist_xy <= float(self.RELEASE_DIST_TOL_XY) * 2.0
        )
        self._place_stable_count = (
            self._place_stable_count + 1 if placed_stable_now else 0
        )

        bonus_final = 0.0
        if (
            self._place_stable_count >= int(self.PLACE_STABLE_COUNT_MIN)
            and not self._placed_success
        ):
            self._placed_success = True
            self._phase_success_flags[2] = True
            self._p2_returning_home = True
            bonus_final = float(self.BONUS_FINAL_P2)
            self._v6_prev_home_potential = self._v6_home_potential()

        home_progress = 0.0
        r_home_event = 0.0
        if self._placed_success:
            home_phi = self._v6_home_potential()
            if self._v6_prev_home_potential is not None:
                home_progress = float(self.W_P2_HOME_PROGRESS) * (
                    float(self.P2_PROGRESS_GAMMA) * home_phi
                    - float(self._v6_prev_home_potential)
                )
            self._v6_prev_home_potential = float(home_phi)

            if self._arm_is_at_home() and not self._v6_home_event_done:
                r_home_event = float(self.W_P2_HOME_EVENT)
                self._v6_home_event_done = True

            place_progress = 0.0
            place_peak_gain = 0.0
            r_descent_progress = 0.0

        reward = (
            place_progress
            + r_descent_progress
            + place_peak_gain
            + r_center_event
            + r_descent_event
            + r_release_event
            + bonus_final
            + home_progress
            + r_home_event
            + float(pen_time)
            + float(pen_energy)
            + float(pen_drop)
            + pen_ped
            + pen_gripper_ped
            + pen_drag
            + pen_low_descent
            + pen_open_outside
            + early_release_pen
        )

        self._accum_phase_debug(
            2,
            reward,
            r_place_progress=place_progress,
            r_place_peak_gain=place_peak_gain,
            r_center_event=r_center_event,
            r_descent_event=r_descent_event,
            r_release_event=r_release_event,
            r_descent_progress=r_descent_progress,
            r_home_progress=home_progress,
            r_home_event=r_home_event,
            bonus_final=bonus_final,
            pen_low_descent=pen_low_descent,
            pen_time=pen_time,
            pen_energy=pen_energy,
            pen_drop=pen_drop,
            pen_ped=pen_ped,
            pen_gripper_ped=pen_gripper_ped,
            pen_open_outside=pen_open_outside,
            early_release_pen=early_release_pen,
            pen_drag=pen_drag,
        )

        info_extra = {
            "phi_2": float(phi),
            "r_place_xy": float(comp["xy_score"]),
            "r_place_z": float(comp["z_score"]),
            "r_approach_top": float(comp["approach_score"]),
            "r_place_contact": float(comp["contact_score"]),
            "p2_xy_gate": float(xy_gate),
            "p2_z_gate": float(z_gate),
            "r_release_ready": float(r_release_ready),
            "r_release_event": float(r_release_event),
            "r_release_ready_dense": 0.0,
            "r_place_stability": 0.0,
            "r_place_progress": float(place_progress),
            "r_descent_progress": float(r_descent_progress),
            "p2_descent_delta_m": float(descent_delta_m),
            "r_place_peak_gain": float(place_peak_gain),
            "r_center_event": float(r_center_event),
            "r_descent_event": float(r_descent_event),
            "r_home_progress": float(home_progress),
            "r_home_event": float(r_home_event),
            "pen_low_descent": float(pen_low_descent),
            "cube_goal_xy_error": float(dist_xy),
            "cube_z_abs_error": float(z_error),
            "cube_speed": float(cube_speed),
            "cube_ped_contacts": cube_ped_contacts,
            "robot_ped_contacts": robot_ped_contacts,
            "gripper_ped_contacts": gripper_ped_contacts,
            "place_stable_count": int(self._place_stable_count),
            "release_ready_count": int(self._release_ready_count),
            "release_ready_pre_sim": bool(
                getattr(self, "_v6_release_ready_pre_sim", False)
            ),
        }
        return float(reward), info_extra

    def _init_base_dynamics_and_pid(self):
        """Configura el mismo PID/Pure Pursuit validado por el baseline."""
        super()._init_base_dynamics_and_pid()

        # Dinámica compuesta del subárbol móvil obtenida directamente del XML.
        self.M_BASE, self.J_BASE = self._mobile_subtree_planar_dynamics()

        # Ganancias PSO para Ts=10 ms; parada geométrica actual a 0.70 m.
        self.Kp_vel = 25.763229287504284
        self.Ki_vel = 3.0318334011181163
        self.Kd_vel = 0.04929680970337456
        self.Kp_omega = 1149.8272721326991
        self.Ki_omega = 36.98551603282882
        self.Kd_omega = 0.11001842680805748

        self.V_MAX = 0.60
        self.OMEGA_MAX = 0.95
        self.V_GAIN = 2.8
        self.OMEGA_GAIN = 3.6

        self.LOOKAHEAD_DIST = 0.30
        self.WAYPOINT_SPACING = 0.05
        self.WAYPOINT_TOL = 0.14
        self.PURE_PURSUIT_MIN_LD_M = 0.08
        self.PURE_PURSUIT_PATH_HEADING_GAIN = 2.5
        self.PURE_PURSUIT_CTE_OMEGA_GAIN = 7.0

        self.YAW_ALIGN_EXIT = float(np.deg2rad(2.0))
        self.YAW_SLOW_1 = float(np.deg2rad(3.0))
        self.YAW_SLOW_2 = float(np.deg2rad(7.0))
        self.YAW_STOP = float(np.deg2rad(25.0))
        self.YAW_SLOW_GAIN_1 = 0.80
        self.YAW_SLOW_GAIN_2 = 0.55
        self.CTE_SLOW_1 = 0.015
        self.CTE_SLOW_2 = 0.025
        self.CTE_SLOW_GAIN_1 = 0.78
        self.CTE_SLOW_GAIN_2 = 0.58

        self.INIT_ALIGN_OMEGA_MAX = 1.30
        self.INIT_ALIGN_OMEGA_GAIN = 8.0
        self.INIT_ALIGN_OMEGA_RATE_MAX = 40.0
        self.INIT_ALIGN_ACC_ANG_MAX = 520.0
        self.INIT_ALIGN_MIN_OMEGA = 0.80

        self.FINAL_GOAL_RADIUS_STOP = 0.70
        self.FINAL_RADIUS_TOL = 0.02
        self.FINAL_WAYPOINT_TOL = 0.020
        self.FINAL_ROUTE_HEADING_TOL = float(np.deg2rad(3.0))
        self.FINAL_ROUTE_STABLE_STEPS = 10
        self.PURE_PURSUIT_TERMINAL_POSE_DIST_M = 0.55
        self.PURE_PURSUIT_TERMINAL_V_GAIN = 2.4
        self.PURE_PURSUIT_TERMINAL_V_MAX = 0.32
        self.PURE_PURSUIT_TERMINAL_V_REVERSE_MAX = 0.10
        # Evita el equilibrio muerto v=0 cuando el error longitudinal ya es
        # casi nulo pero todavía queda error lateral respecto del endpoint.
        # El signo se elige hacia la orientación (avance/reversa) que apunta
        # más directamente al punto final y el propio cierre corrige después
        # cualquier pequeño sobrepaso longitudinal.
        self.PURE_PURSUIT_TERMINAL_V_CRAWL = 0.080
        self.PURE_PURSUIT_TERMINAL_PATH_HEADING_GAIN = 3.8
        self.PURE_PURSUIT_TERMINAL_CTE_OMEGA_GAIN = 10.0
        self.PURE_PURSUIT_TERMINAL_CURVATURE_FRACTION = 0.55
        self.PURE_PURSUIT_TERMINAL_MIN_SPEED_SCALE = 0.22

        self.RETREAT_TOL = 0.0005
        self.REV_MIN_SPEED_SCALE = 0.05

        self.PID_PROFILE_NAME = "PID_PSO_6D_MODEL_DYNAMICS_STRICT_20MM_3DEG"
        self.TRACKING_CONTROLLER = "pure_pursuit"

    def _mobile_subtree_planar_dynamics(self) -> Tuple[float, float]:
        """Retorna masa e inercia yaw del robot móvil según el XML."""
        mujoco.mj_forward(self.model, self.data)
        chassis_sid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "chassis_frame"
        )
        if chassis_sid < 0:
            raise RuntimeError("No se encontró el site chassis_frame.")
        chassis_bid = int(self.model.site_bodyid[chassis_sid])
        root_bid = int(self.model.body_rootid[chassis_bid])
        body_ids = np.flatnonzero(np.asarray(self.model.body_rootid) == root_bid)
        masses = np.asarray(self.model.body_mass[body_ids], dtype=np.float64)
        total_mass = float(np.sum(masses))
        if total_mass <= 0.0:
            raise RuntimeError("La masa del subárbol móvil no es positiva.")
        body_com = np.asarray(self.data.xipos[body_ids], dtype=np.float64)
        subtree_com = np.sum(masses[:, None] * body_com, axis=0) / total_mass
        yaw_inertia = 0.0
        for body_id, body_mass, center in zip(body_ids, masses, body_com):
            rotation = np.asarray(
                self.data.ximat[int(body_id)], dtype=np.float64
            ).reshape(3, 3)
            inertia_world = (
                rotation
                @ np.diag(self.model.body_inertia[int(body_id)])
                @ rotation.T
            )
            offset = center - subtree_com
            yaw_inertia += float(
                inertia_world[2, 2]
                + body_mass * (offset[0] ** 2 + offset[1] ** 2)
            )
        return total_mass, float(yaw_inertia)

    def _init_nav_state(self):
        super()._init_nav_state()
        self._path_reference_points = np.empty((0, 2), dtype=np.float64)
        self._path_reference_arc = np.empty(0, dtype=np.float64)
        self._path_reference_yaw = np.empty(0, dtype=np.float64)
        self._terminal_route_stable_count = 0

    def _ensure_path_reference_cache(self) -> None:
        points = np.asarray(self._waypoints, dtype=np.float64).reshape(-1, 2)
        if (
            points.shape == self._path_reference_points.shape
            and len(points) >= 2
            and np.array_equal(points, self._path_reference_points)
        ):
            return
        self._path_reference_points = points.copy()
        if len(points) < 2:
            self._path_reference_arc = np.empty(0, dtype=np.float64)
            self._path_reference_yaw = np.empty(0, dtype=np.float64)
            return
        segments = np.maximum(
            np.linalg.norm(np.diff(points, axis=0), axis=1), 1.0e-6
        )
        arc = np.concatenate(([0.0], np.cumsum(segments)))
        half_window = 0.16
        yaw = np.zeros(len(points), dtype=np.float64)
        for index in range(len(points)):
            low = int(np.searchsorted(arc, arc[index] - half_window, side="left"))
            high = int(
                np.searchsorted(arc, arc[index] + half_window, side="right") - 1
            )
            low = int(np.clip(low, 0, len(points) - 1))
            high = int(np.clip(high, 0, len(points) - 1))
            if high <= low:
                low = max(0, index - 1)
                high = min(len(points) - 1, index + 1)
            chord = points[high] - points[low]
            yaw[index] = float(np.arctan2(chord[1], chord[0]))
        self._path_reference_arc = arc
        self._path_reference_yaw = np.unwrap(yaw)

    def _continuous_path_reference(
        self,
        point: np.ndarray,
        *,
        start_idx: int = 0,
        global_search: bool = False,
    ) -> Tuple[float, float, float, int, float]:
        """Proyección y tangente continua usadas por control y métricas."""
        self._ensure_path_reference_cache()
        p = np.asarray(point, dtype=np.float64).reshape(2)
        if len(self._path_reference_points) < 2:
            return 0.0, float(p[0]), float(p[1]), 0, 0.0
        cte, px, py, segment_index, _ = self._nearest_path_segment_projection(
            p,
            start_idx=0 if global_search else int(start_idx),
            search_back=0 if global_search else 2,
            search_ahead=(
                max(len(self._waypoints), 1) if global_search else 40
            ),
        )
        segment_index = int(np.clip(
            segment_index, 0, len(self._path_reference_points) - 2
        ))
        start = self._path_reference_points[segment_index]
        end = self._path_reference_points[segment_index + 1]
        segment = end - start
        length2 = float(np.dot(segment, segment))
        projection = np.array([px, py], dtype=np.float64)
        ratio = 0.0 if length2 < 1.0e-12 else float(np.clip(
            np.dot(projection - start, segment) / length2, 0.0, 1.0
        ))
        s_projection = float(
            self._path_reference_arc[segment_index]
            + ratio
            * (
                self._path_reference_arc[segment_index + 1]
                - self._path_reference_arc[segment_index]
            )
        )
        path_yaw = float(np.interp(
            s_projection, self._path_reference_arc, self._path_reference_yaw
        ))
        path_yaw = float((path_yaw + np.pi) % (2.0 * np.pi) - np.pi)
        return float(cte), float(px), float(py), segment_index, path_yaw

    def _init_maps_and_obstacles(self):
        """Usa los cuatro mapas/rutas actuales sin tocar la lógica de episodios."""
        super()._init_maps_and_obstacles()
        self.MAPS_DIR = str(self.routes_path)
        self.LOAD_ROUTES_ON_START = True
        self.RELOAD_ROUTE_FROM_DISK_ON_REUSE = True
        self.STRICT_ROUTE_VERSION_CHECK = True
        self.SAVE_ROUTES_ON_PLAN = True
        self.SAVE_ROUTE_PLOTS = False

    def _load_single_map_route_from_disk(
        self,
        map_idx: int,
        folder: Optional[str] = None,
        verbose: bool = True,
    ):
        """Carga también la geometría asociada a cada ruta regenerada.

        ``ZennoV3`` ya carga la polilínea. Las rutas nuevas incluyen además los
        estados iniciales y obstáculos; sincronizarlos evita evaluar una ruta en
        la configuración histórica de otro mapa.
        """
        loaded = super()._load_single_map_route_from_disk(
            map_idx, folder=folder, verbose=verbose
        )
        if not loaded:
            return False

        route_folder = str(folder if folder is not None else self.MAPS_DIR)
        route_file = self._map_route_filepath(map_idx, route_folder)
        try:
            with route_file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError, TypeError):
            return loaded

        entry = self.MAP_CACHE.get(int(map_idx))
        if entry is None:
            return loaded

        base_qpos7 = np.asarray(payload.get("base_qpos7", []), dtype=np.float64)
        cube_qpos7 = np.asarray(
            payload.get("cube_qpos7_nominal", []), dtype=np.float64
        )
        if base_qpos7.shape == (7,) and cube_qpos7.shape == (7,):
            self.INITIALS_BY_MAP[int(map_idx)] = {
                "base_qpos7": base_qpos7.copy(),
                "cube_qpos7": cube_qpos7.copy(),
            }
            entry["base_qpos7"] = base_qpos7.copy()
            entry["cube_qpos7_nominal"] = cube_qpos7.copy()

        obstacles = payload.get("obstacles_xy")
        if isinstance(obstacles, list):
            parsed_obstacles = []
            for obstacle in obstacles:
                try:
                    x, y = obstacle
                    parsed_obstacles.append((float(x), float(y)))
                except (TypeError, ValueError):
                    continue
            self.PREDEFINED_MAPS[int(map_idx)] = parsed_obstacles
            entry["obstacles_xy"] = list(parsed_obstacles)

        return True

    def _guarded_tracking_residual(
        self,
        v_res_u: float,
        omega_res_u: float,
        *,
        cte_abs: float,
        yaw_abs: float,
        omega_des: float,
    ) -> Tuple[float, float]:
        """Aplica una corrección residual continua, filtrada y de baja autoridad.

        La autoridad cae suavemente a cero tanto cerca de la trayectoria (donde
        el PID no necesita ayuda) como en el borde del corredor (donde una acción
        aprendida ya no es segura). El filtro y el slew limiter evitan que el
        torque residual cambie de signo a 100 Hz.
        """
        raw_linear = float(np.clip(v_res_u, -1.0, 1.0))
        raw_angular = float(np.clip(omega_res_u, -1.0, 1.0))
        cte_ratio = float(cte_abs) / max(float(self.TRACK_CORRIDOR), 1.0e-9)
        yaw_ratio = float(yaw_abs) / max(float(self.YAW_CORRIDOR), 1.0e-9)
        safe = bool(cte_ratio < 1.0 and yaw_ratio < 1.0)
        geometry_authority = (
            max(0.0, 1.0 - cte_ratio * cte_ratio)
            * max(0.0, 1.0 - yaw_ratio * yaw_ratio)
        )
        need_ratio = max(
            float(cte_abs) / max(float(self.P1_RES_DEADBAND_CTE_M), 1.0e-9),
            float(yaw_abs) / max(float(self.P1_RES_DEADBAND_YAW_RAD), 1.0e-9),
        )
        need_authority = float(np.clip(need_ratio - 1.0, 0.0, 1.0))
        need_authority = need_authority * need_authority * (3.0 - 2.0 * need_authority)
        authority = float(geometry_authority * need_authority) if safe else 0.0

        target_linear = authority * float(np.clip(
            raw_linear, -float(self.P1_RES_U_LIN_MAX), float(self.P1_RES_U_LIN_MAX)
        ))
        target_angular = authority * float(np.clip(
            raw_angular, -float(self.P1_RES_U_ANG_MAX), float(self.P1_RES_U_ANG_MAX)
        ))
        omega_scale = float(np.clip(
            abs(float(omega_des))
            / max(float(getattr(self, "P1_RES_MIN_OMEGA_FOR_SIGN", 0.08)), 1.0e-9),
            0.0,
            1.0,
        ))
        target_angular *= omega_scale
        sign_conflict = bool(
            abs(float(omega_des)) > float(getattr(self, "P1_RES_MIN_OMEGA_FOR_SIGN", 0.08))
            and abs(target_angular) > 1.0e-8
            and np.sign(target_angular) != np.sign(float(omega_des))
        )
        if sign_conflict:
            target_angular = 0.0

        dt = float(self.model.opt.timestep * self.frame_skip)
        alpha = float(dt / (float(self.P1_RES_FILTER_TAU_S) + dt))
        filtered_linear = float(self._residual_lin_filtered + alpha * (
            target_linear - self._residual_lin_filtered
        ))
        filtered_angular = float(self._residual_ang_filtered + alpha * (
            target_angular - self._residual_ang_filtered
        ))
        max_lin_change = float(self.P1_RES_LIN_SLEW_PER_S) * dt
        max_ang_change = float(self.P1_RES_ANG_SLEW_PER_S) * dt
        applied_linear = float(self._residual_lin_filtered + np.clip(
            filtered_linear - self._residual_lin_filtered,
            -max_lin_change,
            max_lin_change,
        ))
        applied_angular = float(self._residual_ang_filtered + np.clip(
            filtered_angular - self._residual_ang_filtered,
            -max_ang_change,
            max_ang_change,
        ))
        self._residual_lin_filtered = applied_linear
        self._residual_ang_filtered = applied_angular
        self._last_residual_authority = authority
        reason = (
            "outside_safe_corridor" if not safe
            else "residual_deadband" if need_authority <= 0.0
            else "angular_sign_block" if sign_conflict
            else "filtered_continuous_authority"
        )

        self._last_residual_safe_corridor = safe
        self._last_residual_block_reason = reason
        self._last_u_lin_raw = raw_linear
        self._last_u_ang_raw = raw_angular
        self._last_u_lin_applied = applied_linear
        self._last_u_ang_applied = applied_angular
        return applied_linear, applied_angular

    def _control_base_step(self, Ts: float, v_res_u: float, omega_res_u: float):
        """Pure Pursuit actual con cierre continuo del endpoint 20 mm/3°."""
        if (not self._plan_ready) or len(self._waypoints) < 2 or self._nav_done:
            self._hard_stop_base()
            return

        position = np.asarray(self._base_xy(), dtype=np.float64)
        x, y = float(position[0]), float(position[1])
        theta = float(self._base_yaw())

        # La alineación usa la misma tangente continua de las métricas.
        if self._initial_aligning:
            _cte, _px, _py, segment_index, path_yaw = (
                self._continuous_path_reference(
                    position, start_idx=self._wp_idx, global_search=True
                )
            )
            yaw_error = float(wrap_angle(path_yaw - theta))
            if abs(yaw_error) <= float(self.YAW_ALIGN_EXIT):
                self._initial_aligning = False
                self._wp_idx = min(int(segment_index), len(self._waypoints) - 1)
                self._reset_pid_accums()
                self._hard_stop_base()
                return
            raw_omega = float(np.clip(
                float(self.INIT_ALIGN_OMEGA_GAIN) * yaw_error,
                -float(self.INIT_ALIGN_OMEGA_MAX),
                float(self.INIT_ALIGN_OMEGA_MAX),
            ))
            omega_cmd = (
                float(np.sign(yaw_error) * float(self.INIT_ALIGN_MIN_OMEGA))
                if abs(raw_omega) < float(self.INIT_ALIGN_MIN_OMEGA)
                else raw_omega
            )
            self._control_base_vel(
                Ts,
                v_des=0.0,
                omega_des=omega_cmd,
                u_lin=0.0,
                u_ang=0.0,
                omega_max_override=float(self.INIT_ALIGN_OMEGA_MAX),
                omega_rate_max_override=float(self.INIT_ALIGN_OMEGA_RATE_MAX),
                acc_ang_max_override=float(self.INIT_ALIGN_ACC_ANG_MAX),
                mode="initial_align",
            )
            return

        self._advance_wp_if_reached(x, y)
        last = np.asarray(self._waypoints[-1], dtype=np.float64)
        previous = np.asarray(self._waypoints[-2], dtype=np.float64)
        terminal_vector = last - previous
        terminal_yaw = float(np.arctan2(terminal_vector[1], terminal_vector[0]))
        endpoint_distance = float(np.linalg.norm(last - position))

        lookahead_x, lookahead_y, _lookahead_segment = self._choose_lookahead(
            x, y, self._wp_idx, self.LOOKAHEAD_DIST
        )
        lookahead_vector = np.array(
            [lookahead_x - x, lookahead_y - y], dtype=np.float64
        )
        lookahead_distance = max(
            float(np.linalg.norm(lookahead_vector)),
            float(self.PURE_PURSUIT_MIN_LD_M),
        )
        bearing_error = float(wrap_angle(
            float(np.arctan2(lookahead_vector[1], lookahead_vector[0])) - theta
        ))
        curvature = float(2.0 * np.sin(bearing_error) / lookahead_distance)
        cte, _px, _py, path_segment, path_yaw = self._continuous_path_reference(
            position, start_idx=self._wp_idx
        )
        self._wp_idx = max(int(self._wp_idx), int(path_segment))
        path_heading_error = float(wrap_angle(path_yaw - theta))

        terminal_heading_error = float(wrap_angle(terminal_yaw - theta))
        if (
            endpoint_distance <= float(self.FINAL_WAYPOINT_TOL)
            and abs(terminal_heading_error)
            <= float(self.FINAL_ROUTE_HEADING_TOL)
        ):
            self._terminal_route_stable_count += 1
        else:
            self._terminal_route_stable_count = 0
        if self._terminal_route_stable_count >= int(self.FINAL_ROUTE_STABLE_STEPS):
            self._reached_last_wp = True
            self._nav_done = True
            self._reset_pid_accums()
            self._hard_stop_base()
            self._zero_base_qvel_like_extract()
            return

        # Cierre terminal sobre la tangente; no existe realineación posterior.
        if endpoint_distance <= float(self.PURE_PURSUIT_TERMINAL_POSE_DIST_M):
            tangent = np.array(
                [np.cos(terminal_yaw), np.sin(terminal_yaw)], dtype=np.float64
            )
            endpoint_error = last - position
            along_error = float(np.dot(endpoint_error, tangent))
            lateral_error = float(
                tangent[0] * (y - float(last[1]))
                - tangent[1] * (x - float(last[0]))
            )
            fraction = float(np.clip(
                1.0
                - endpoint_distance
                / max(float(self.PURE_PURSUIT_TERMINAL_POSE_DIST_M), 1.0e-6),
                0.0,
                1.0,
            ))
            blend = fraction * fraction * (3.0 - 2.0 * fraction)
            terminal_path_yaw = float(wrap_angle(
                path_yaw + blend * wrap_angle(terminal_yaw - path_yaw)
            ))
            heading_error = float(wrap_angle(terminal_path_yaw - theta))
            velocity = float(np.clip(
                float(self.PURE_PURSUIT_TERMINAL_V_GAIN) * along_error,
                -float(self.PURE_PURSUIT_TERMINAL_V_REVERSE_MAX),
                float(self.PURE_PURSUIT_TERMINAL_V_MAX),
            ))
            geometry_scale = float(np.clip(
                abs(along_error)
                / max(abs(along_error) + 2.0 * abs(lateral_error), 1.0e-6),
                float(self.PURE_PURSUIT_TERMINAL_MIN_SPEED_SCALE),
                1.0,
            ))
            velocity *= geometry_scale / (1.0 + 1.5 * abs(heading_error))
            if (
                endpoint_distance > float(self.FINAL_WAYPOINT_TOL)
                and abs(velocity)
                < float(self.PURE_PURSUIT_TERMINAL_V_CRAWL)
            ):
                endpoint_bearing = float(np.arctan2(
                    endpoint_error[1], endpoint_error[0]
                ))
                forward_error = abs(float(wrap_angle(endpoint_bearing - theta)))
                reverse_error = abs(float(wrap_angle(
                    endpoint_bearing + np.pi - theta
                )))
                recovery_sign = 1.0 if forward_error <= reverse_error else -1.0
                velocity = (
                    recovery_sign
                    * float(self.PURE_PURSUIT_TERMINAL_V_CRAWL)
                )
            drive_sign = 1.0 if velocity >= 0.0 else -1.0
            omega = float(np.clip(
                (1.0 - blend)
                * float(self.PURE_PURSUIT_TERMINAL_CURVATURE_FRACTION)
                * velocity
                * curvature
                + float(self.PURE_PURSUIT_TERMINAL_PATH_HEADING_GAIN)
                * heading_error
                - drive_sign
                * float(self.PURE_PURSUIT_TERMINAL_CTE_OMEGA_GAIN)
                * lateral_error,
                -float(self.OMEGA_MAX),
                float(self.OMEGA_MAX),
            ))
            self._last_cte_signed = lateral_error
            self._last_cte_corr = float(
                -drive_sign
                * float(self.PURE_PURSUIT_TERMINAL_CTE_OMEGA_GAIN)
                * lateral_error
            )
            self._last_track_err = abs(lateral_error)
            self._last_yaw_track_err = heading_error
            self._last_u_lin_raw = float(np.clip(v_res_u, -1.0, 1.0))
            self._last_u_ang_raw = float(np.clip(omega_res_u, -1.0, 1.0))
            self._last_u_lin_applied = 0.0
            self._last_u_ang_applied = 0.0
            self._residual_lin_filtered = 0.0
            self._residual_ang_filtered = 0.0
            self._last_residual_authority = 0.0
            self._last_residual_block_reason = "terminal_pose_controller"
            self._control_base_vel(
                Ts,
                v_des=velocity,
                omega_des=omega,
                u_lin=0.0,
                u_ang=0.0,
                mode="tracking",
            )
            return

        lookahead_error = float(np.linalg.norm(lookahead_vector))
        heading_scale = float(1.0 / (1.0 + float(self.HEADING_K) * abs(bearing_error)))
        slow_near = float(np.clip(
            lookahead_error / max(float(self.WAYPOINT_TOL), 1.0e-3), 0.2, 1.0
        ))
        velocity = (
            float(np.clip(
                float(self.V_GAIN) * lookahead_error, 0.0, float(self.V_MAX)
            ))
            * heading_scale
            * slow_near
        )
        omega_feedback = float(
            float(self.PURE_PURSUIT_PATH_HEADING_GAIN) * path_heading_error
            - float(self.PURE_PURSUIT_CTE_OMEGA_GAIN) * cte
        )
        omega = float(np.clip(
            velocity * curvature + omega_feedback,
            -float(self.OMEGA_MAX),
            float(self.OMEGA_MAX),
        ))
        velocity = float(np.clip(
            min(
                velocity,
                float(self.V_CURV) / (1.0 + float(self.K_CURV) * abs(omega)),
            ),
            0.0,
            float(self.V_MAX),
        ))
        yaw_abs = abs(path_heading_error)
        if yaw_abs > float(self.YAW_STOP):
            velocity = 0.0
        elif yaw_abs > float(self.YAW_SLOW_2):
            velocity *= float(self.YAW_SLOW_GAIN_2)
        elif yaw_abs > float(self.YAW_SLOW_1):
            velocity *= float(self.YAW_SLOW_GAIN_1)
        cte_abs = abs(float(cte))
        if cte_abs > float(self.CTE_SLOW_2):
            velocity *= float(self.CTE_SLOW_GAIN_2)
        elif cte_abs > float(self.CTE_SLOW_1):
            velocity *= float(self.CTE_SLOW_GAIN_1)
        omega = float(np.clip(
            velocity * curvature + omega_feedback,
            -float(self.OMEGA_MAX),
            float(self.OMEGA_MAX),
        ))
        residual_linear, residual_angular = self._guarded_tracking_residual(
            v_res_u,
            omega_res_u,
            cte_abs=cte_abs,
            yaw_abs=yaw_abs,
            omega_des=omega,
        )
        self._last_cte_signed = float(cte)
        self._last_cte_corr = float(
            -float(self.PURE_PURSUIT_CTE_OMEGA_GAIN) * cte
        )
        self._last_track_err = cte_abs
        self._last_yaw_track_err = path_heading_error
        self._control_base_vel(
            Ts,
            v_des=velocity,
            omega_des=omega,
            u_lin=residual_linear,
            u_ang=residual_angular,
            mode="tracking",
        )

    def _apply_base_action(self, base_act: np.ndarray, Ts: float):
        """Replica la máquina de V3 con retroceso preciso de 0.5 mm."""
        phase = int(self.phase_state)
        if phase == 0:
            self._hard_stop_base()
            return
        if phase == 1:
            if self.base_phase == 0:
                self.base_phase = 1
                self._start_reverse(distance_m=float(self.RETREAT_DIST))
                self._terminal_route_stable_count = 0
            if self.base_phase == 1:
                if self._reverse_goal_xy is None or not self._reverse_active:
                    self._reverse_active = False
                    self.base_phase = 2
                    self._hard_stop_base()
                    self._zero_base_qvel_like_extract()
                    self._reset_pid_accums()
                else:
                    position = np.asarray(self._base_xy(), dtype=np.float64)
                    remaining = float(np.linalg.norm(
                        self._reverse_goal_xy - position
                    ))
                    slow = float(np.clip(
                        remaining / max(float(self.REV_SLOW_DIST), 1.0e-6),
                        float(self.REV_MIN_SPEED_SCALE),
                        1.0,
                    ))
                    self._control_base_vel(
                        Ts,
                        v_des=-float(self.RETREAT_V_MAX) * slow,
                        omega_des=0.0,
                        u_lin=0.0,
                        u_ang=0.0,
                        mode="reverse",
                    )
                    if remaining <= float(self.RETREAT_TOL):
                        self._hard_stop_base()
                        self._zero_base_qvel_like_extract()
                        self._reset_pid_accums()
                        self._reverse_active = False
                        self.base_phase = 2
            elif self.base_phase == 2:
                reused = self._try_reuse_cached_route()
                success = reused or self._plan_direct_goal_stop()
                self.base_phase = 3 if success else 1
                self._terminal_route_stable_count = 0
                self._reset_pid_accums()
            elif self.base_phase >= 3:
                self._control_base_step(
                    Ts,
                    v_res_u=float(base_act[0]),
                    omega_res_u=float(base_act[1]),
                )
            return
        self._hard_stop_base()

    def set_curriculum(self, enabled: bool):
        """API de compatibilidad: este entorno sólo admite end-to-end."""
        if bool(enabled):
            raise ValueError(
                "El curriculum está deshabilitado para conservar exactamente "
                "el esquema end-to-end de ZennoV3."
            )
        self.curriculum_enabled = False

    def set_curriculum_start_probs(self, probs: Sequence[float]):
        """API de compatibilidad; acepta únicamente inicio en fase 0."""
        values = np.asarray(probs, dtype=np.float64)
        if values.shape != (3,) or not np.allclose(
            values, (1.0, 0.0, 0.0), atol=1e-12
        ):
            raise ValueError(
                "El entrenamiento end-to-end debe usar probabilidades (1, 0, 0)."
            )
        self.curriculum_start_probs = np.array(
            [1.0, 0.0, 0.0], dtype=np.float64
        )

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Ejecuta V5 y añade la penalización terminal por fase incompleta."""
        phase_before_step = int(self.phase_state)
        observation, reward, terminated, truncated, info = super().step(action)

        episode_done = bool(terminated or truncated)
        strict_success = bool(
            episode_done
            and phase_before_step == 2
            and bool(self._placed_success)
            and bool(self._arm_is_at_home())
            and not bool(self.cube_dropped)
            and not bool(truncated)
        )
        phase_failure_penalty = 0.0
        if episode_done and not strict_success:
            phase_failure_penalty = float(
                self.PHASE_FAILURE_PENALTIES[phase_before_step]
            )
            reward = float(reward) + phase_failure_penalty
            self.reward = float(reward)

        dt = float(self.model.opt.timestep * self.frame_skip)
        energy_total, energy_arm, energy_base = self._energy_cost(dt)
        self._last_energy_total_j = float(energy_total)
        self._last_energy_arm_j = float(energy_arm)
        self._last_energy_base_j = float(energy_base)

        # Garantía adicional para cualquier wrapper Monitor externo.
        # Aunque el step de la clase base ya completa estas claves, se vuelven
        # a asegurar aquí porque ZennoV7 añade métricas P2 después de super().step().
        p2_monitor_defaults = {
            "r_place_progress": 0.0,
            "r_descent_progress": 0.0,
            "p2_descent_delta_m": 0.0,
            "r_place_peak_gain": 0.0,
            "p2_xy_gate": 0.0,
            "p2_z_gate": 0.0,
            "release_ready_pre_sim": False,
            "cube_goal_xy_error": 0.0,
            "cube_z_abs_error": 0.0,
            "place_stable_count": 0,
            "release_ready_count": 0,
        }
        for key, value in p2_monitor_defaults.items():
            info.setdefault(key, value)

        info.update(
            {
                "curriculum_start_phase": 0,
                "is_success": bool(strict_success),
                "strict_success": bool(strict_success),
                "terminal_phase": int(phase_before_step) if episode_done else -1,
                "phase_completed_at_end": bool(strict_success),
                "phase_failure_penalty": float(phase_failure_penalty),
                "reward_place_version": "v7_p2_xy_multiscale_gate_r070_obs78",
                "p2_time_penalty_per_step": -float(self.W_TIME_P2),
                "energy_step_total_j": self._last_energy_total_j,
                "energy_step_arm_j": self._last_energy_arm_j,
                "energy_step_base_j": self._last_energy_base_j,
                "base_control_mode": self.base_control_mode,
                "objective": self.objective,
                "pid_profile": self.PID_PROFILE_NAME,
                "tracking_controller": self.TRACKING_CONTROLLER,
                "routes_dir": str(self.MAPS_DIR),
                "mobile_mass_kg": float(self.M_BASE),
                "mobile_yaw_inertia_kg_m2": float(self.J_BASE),
                "terminal_endpoint_tolerance_m": float(self.FINAL_WAYPOINT_TOL),
                "terminal_heading_tolerance_rad": float(
                    self.FINAL_ROUTE_HEADING_TOL
                ),
            }
        )
        return observation, float(reward), terminated, truncated, info

class _VXReference(ZennoV7):
    """Un único ambiente end-to-end con acciones incrementales P0/P1/P2.

    Los primeros cinco canales nunca representan posiciones articulares. En
    cada decisión expresan un incremento normalizado respecto de
    ``arm_target``. Los dos últimos canales conservan el residual acotado del
    controlador PID de la base, que sólo tiene autoridad durante P1.
    """

    REVISION = "VX_R3_REFERENCE_CLONE_78D_7D"
    # P2 dispone de MAX_STEPS_2 exclusivamente para aprender la colocacion.
    # Una vez validada la colocacion, el retorno a home deja de consumir ese
    # presupuesto y usa este margen separado. El retorno lo ejecuta el
    # controlador interno, no una segunda politica.
    # Validacion nominal R3: 18 retornos correctos necesitaron 322--346 pasos.
    # 400 conserva 54 pasos de margen sobre el peor caso observado.
    P2_HOME_MAX_STEPS = 400
    ARM_DELTA_SCALE_BY_PHASE = np.asarray(
        (
            (0.025, 0.025, 0.020, 0.020, 0.025),
            (0.008, 0.008, 0.006, 0.006, 0.008),
            (0.008, 0.008, 0.008, 0.006, 0.010),
        ),
        dtype=np.float32,
    )
    EXPOSE_MAP_CONTEXT = False

    # Regularización de la política incremental. Penaliza cambios bruscos sin
    # castigar de forma relevante los pequeños incrementos necesarios.
    W_ARM_ACTION_MAG = 0.002
    W_ARM_ACTION_SLEW = 0.010
    W_BASE_ACTION_MAG = 0.002
    W_BASE_ACTION_SLEW = 0.010

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._v7_action_space = self.action_space
        self.action_space = spaces.Box(
            low=-np.ones(ACTION_DIM, dtype=np.float32),
            high=np.ones(ACTION_DIM, dtype=np.float32),
            dtype=np.float32,
        )
        self._last_public_action = np.zeros(ACTION_DIM, dtype=np.float32)

        # Residual P1 de baja autoridad con filtrado y limitación de slew. El
        # PID sigue siendo el controlador principal de navegación.
        self.P1_RES_U_LIN_MAX = 0.15
        self.P1_RES_U_ANG_MAX = 0.08
        self.P1_RES_FILTER_TAU_S = 0.10
        self.P1_RES_LIN_SLEW_PER_S = 1.0
        self.P1_RES_ANG_SLEW_PER_S = 0.8
        self.P1_RES_DEADBAND_CTE_M = 0.006
        self.P1_RES_DEADBAND_YAW_RAD = 0.025
        self._residual_lin_filtered = 0.0
        self._residual_ang_filtered = 0.0
        self._last_residual_authority = 0.0

    def reset_model(self):
        observation = super().reset_model()
        self._p2_placed_at_phase_step = None
        self._last_public_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._residual_lin_filtered = 0.0
        self._residual_ang_filtered = 0.0
        self._last_residual_authority = 0.0
        return observation

    def _stage_setup_phase2(self):
        super()._stage_setup_phase2()
        self._p2_placed_at_phase_step = None

    def _check_termination(self, phase_eval: int):
        """Separa colocacion aprendida y retorno determinista dentro de P2."""
        terminated = False
        truncated = False
        term_reason = ""
        trunc_reason = ""

        if self.cube_dropped:
            terminated = True
            term_reason = "drop"

        if (not terminated) and phase_eval == 2 and self._placed_success:
            if self._p2_placed_at_phase_step is None:
                self._p2_placed_at_phase_step = int(self._phase_steps)
            if self._arm_is_at_home():
                terminated = True
                term_reason = "place_success_arm_home"
            elif (
                int(self._phase_steps) - int(self._p2_placed_at_phase_step)
                >= int(self.P2_HOME_MAX_STEPS)
            ):
                truncated = True
                trunc_reason = "phase2_home_budget"

        if not terminated and not truncated:
            if phase_eval == 2:
                # MAX_STEPS_2 solo limita el tramo previo a colocar el cubo.
                if (
                    not self._placed_success
                    and self._phase_steps >= int(self.MAX_STEPS_2)
                ):
                    truncated = True
                    trunc_reason = "phase2_placement_budget"
            else:
                budget = {
                    0: self.MAX_STEPS_0,
                    1: self.MAX_STEPS_1,
                    2: self.MAX_STEPS_2,
                }[self.phase_state]
                if self._phase_steps >= int(budget):
                    truncated = True
                    trunc_reason = f"phase{self.phase_state}_budget"

        self._last_termination_reason = str(term_reason)
        self._last_truncation_reason = str(trunc_reason)
        return terminated, truncated, term_reason, trunc_reason

    def _get_obs(self):
        """Devuelve siempre la observación física real de 78 dimensiones."""
        observation = super()._get_obs()
        if int(self.phase_state) == 0 and bool(self.EXPOSE_MAP_CONTEXT):
            # En P0 estos cuatro canales de navegación son históricamente cero.
            # Se reutilizan sin cambiar OBS_DIM=78; el actor condicionado los
            # anula antes de alimentar las cabezas docentes congeladas.
            observation[45:49] = 0.0
            observation[45 + int(self.map_id) - 1] = 1.0
        return observation

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        public_action = np.clip(
            np.asarray(action, dtype=np.float32), -1.0, 1.0
        )
        if public_action.shape != (ACTION_DIM,):
            raise ValueError(
                f"ZennoVX espera accion ({ACTION_DIM},), recibio {public_action.shape}."
            )
        phase_before = int(self.phase_state)
        scale = self.ARM_DELTA_SCALE_BY_PHASE[phase_before]
        desired_arm = self.incremental_arm_target(
            public_action[:5], phase=phase_before
        )
        v7_action = np.concatenate([desired_arm, public_action[5:7]]).astype(np.float32)
        public_space = self.action_space
        self.action_space = self._v7_action_space
        try:
            observation, reward, terminated, truncated, info = super().step(v7_action)
        finally:
            self.action_space = public_space

        delta_action = public_action - self._last_public_action
        arm_mag_penalty = -float(self.W_ARM_ACTION_MAG) * float(
            np.mean(np.square(public_action[:5]))
        )
        arm_slew_penalty = -float(self.W_ARM_ACTION_SLEW) * float(
            np.mean(np.square(delta_action[:5]))
        )
        base_mag_penalty = -float(self.W_BASE_ACTION_MAG) * float(
            np.mean(np.square(public_action[5:7]))
        ) if phase_before == 1 else 0.0
        base_slew_penalty = -float(self.W_BASE_ACTION_SLEW) * float(
            np.mean(np.square(delta_action[5:7]))
        ) if phase_before == 1 else 0.0
        smoothness_penalty = (
            arm_mag_penalty + arm_slew_penalty
            + base_mag_penalty + base_slew_penalty
        )
        reward = float(reward) + float(smoothness_penalty)
        self.reward = float(reward)
        self._last_public_action = public_action.copy()
        info.update(
            {
                "environment_revision": self.REVISION,
                "student_phase": int(info.get("phase_state", self.phase_state)),
                "action_dim": int(ACTION_DIM),
                "action_semantics": "incremental_normalized_all_phases",
                "arm_delta_scale": scale.astype(float).tolist(),
                "arm_action_magnitude_penalty": float(arm_mag_penalty),
                "arm_action_slew_penalty": float(arm_slew_penalty),
                "base_action_magnitude_penalty": float(base_mag_penalty),
                "base_action_slew_penalty": float(base_slew_penalty),
                "smoothness_penalty": float(smoothness_penalty),
                "residual_authority": float(self._last_residual_authority),
                "p2_placement_budget_steps": int(self.MAX_STEPS_2),
                "p2_home_budget_steps": int(self.P2_HOME_MAX_STEPS),
                "p2_placed_at_phase_step": (
                    None
                    if self._p2_placed_at_phase_step is None
                    else int(self._p2_placed_at_phase_step)
                ),
                "p2_home_steps": (
                    0
                    if self._p2_placed_at_phase_step is None
                    else max(
                        0,
                        int(self._phase_steps)
                        - int(self._p2_placed_at_phase_step),
                    )
                ),
                "termination_reason": str(
                    getattr(self, "_last_termination_reason", "")
                ),
                "truncation_reason": str(
                    getattr(self, "_last_truncation_reason", "")
                ),
                "arm_home_position_error": float(
                    np.max(np.abs(self._get_arm_qpos() - self.q_arm_home))
                ),
                "arm_home_velocity_max": float(
                    np.max(np.abs(self._get_arm_qvel()))
                ),
            }
        )
        return observation, reward, terminated, truncated, info


# VX conserva los primeros 78 canales y agrega señales de significado fijo.
VX_EXTRA_NAMES = (
    'cte_m', 'path_yaw_sin', 'path_yaw_cos', 'route_valid',
    'v_body', 'omega_body', 'v_reference', 'omega_reference',
    'v_reference_filtered', 'omega_reference_filtered',
    'pid_acc_lin', 'pid_acc_ang', 'pid_e_v1', 'pid_e_v2',
    'pid_e_w1', 'pid_e_w2', 'omega_reference_previous',
    'nominal_left', 'nominal_right', 'delta_left', 'delta_right',
    'residual_filter_lin', 'residual_filter_ang', 'residual_active',
    'route_progress', 'route_remaining_m', 'preview_x_body', 'preview_y_body',
    'terminal_distance_m', 'terminal_yaw_sin', 'terminal_yaw_cos',
    'carry_x', 'carry_y', 'carry_z', 'carry_vx', 'carry_vy', 'carry_vz',
    'cube_wx', 'cube_wy', 'cube_wz', 'phase_remaining', 'previous_arm_0',
    'previous_arm_1', 'previous_arm_2', 'previous_arm_3', 'previous_arm_4',
)
VX_OBS_DIM = 78 + len(VX_EXTRA_NAMES)
OBS_DIM = VX_OBS_DIM


class ZennoVX(_VXReference):
    """VX-R1: observación 124D, residual de par bidireccional y P1 geométrica."""

    REVISION = 'VX_R1_ROUTE_REWARD_124D_7D_DELTA008_UNIFORM_COSTS'
    TIME_COST_PER_SECOND = 0.03
    ENERGY_COST_PER_JOULE = 0.0005
    # Incremento máximo de la referencia por decisión, igual en P0/P1/P2.
    ARM_DELTA_SCALE_BY_PHASE = np.full((3, 5), 0.008, dtype=np.float32)
    ACTION_SEMANTICS = 'incremental_arm_bidirectional_torque_residual'
    RESIDUAL_FRACTION = 0.20
    RESIDUAL_TAU_S = 0.15
    RESIDUAL_SLEW_PER_S = 1.0
    W_BASE_ACTION_MAG = 0.0
    W_BASE_ACTION_SLEW = 0.0

    def __init__(self, *, residual_enabled=True, **kwargs):
        self.residual_enabled = bool(residual_enabled)
        self._reset_vx()
        super().__init__(**kwargs)
        self.MAX_STEPS_0, self.MAX_STEPS_1, self.MAX_STEPS_2 = 1000, 3000, 1000
        self._set_resource_cost_weights(self.dt)
        self.observation_space = spaces.Box(-np.inf, np.inf, (VX_OBS_DIM,), np.float32)

    def _set_resource_cost_weights(self, dt):
        # Keep inherited reward branches and telemetry on the same physical units.
        self.W_TIME = self.W_TIME_P2 = self.TIME_COST_PER_SECOND * float(dt)
        self.W_EN_ARM = self.W_EN_BASE = self.ENERGY_COST_PER_JOULE
        self.W_EN_ARM_P2 = self.W_EN_BASE_P2 = self.ENERGY_COST_PER_JOULE

    def _compute_phase_reward(self, phase_eval, Ts, r_release_ready):
        self._set_resource_cost_weights(Ts)
        return super()._compute_phase_reward(phase_eval, Ts, r_release_ready)

    def _reset_vx(self):
        self._vx_filter = np.zeros(2)
        self._vx_nominal = np.zeros(2)
        self._vx_delta = np.zeros(2)
        self._vx_saturation = 0.0
        self._vx_active = False
        self._vx_prev_sp = None
        self._vx_relative_velocity = np.zeros(3)
        self._last_track_err = self._last_cte_signed = self._last_yaw_track_err = 0.0
        self._last_v_des_cmd = self._last_omega_des_cmd = 0.0

    def reset_model(self):
        self._reset_vx()
        super().reset_model()
        self._vx_prev_sp = self._site_xpos('setpoint').copy()
        return self._get_obs()

    def _route_state(self):
        if not self._plan_ready or len(self._waypoints) < 2:
            return np.zeros(2), 0.0, 0.0, 0.0, np.zeros(2), 0.0
        xy = np.asarray(self._base_xy())
        points = np.asarray(self._waypoints)[:, :2]
        cte, px, py, segment, yaw = self._continuous_path_reference(xy, start_idx=self._wp_idx)
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        s = float(arc[segment] + np.linalg.norm(np.array([px, py]) - points[segment]))
        preview_s = min(s + self.LOOKAHEAD_DIST, arc[-1])
        preview = np.array([np.interp(preview_s, arc, points[:, i]) for i in (0, 1)]) - xy
        theta = self._base_yaw()
        rotation = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        return np.array([cte, wrap_angle(yaw-theta)]), s, float(arc[-1]), float(np.linalg.norm(points[-1]-xy)), rotation @ preview, 1.0

    def _guarded_tracking_residual(self, v_res_u, omega_res_u, **kwargs):
        self._vx_active = bool(self.residual_enabled)
        if not self._vx_active:
            self._vx_filter[:] = 0.0
            return 0.0, 0.0
        requested = np.clip([v_res_u, omega_res_u], -1, 1)
        change = self.dt / (self.RESIDUAL_TAU_S + self.dt) * (requested-self._vx_filter)
        self._vx_filter += np.clip(change, -self.dt*self.RESIDUAL_SLEW_PER_S, self.dt*self.RESIDUAL_SLEW_PER_S)
        self._last_residual_authority = self.RESIDUAL_FRACTION
        return tuple(self._vx_filter)

    def _apply_wheel_torque_with_residual(self, tau_each_L, tau_each_R, u_lin, u_ang):
        # L1 allocation: independent symmetric/differential controls, each wheel
        # remains within RESIDUAL_FRACTION even when both channels are maximal.
        mix = np.array([u_lin-u_ang, u_lin+u_ang], dtype=float)
        mix /= max(1.0, float(np.max(np.abs(mix))))
        deltas, nominal, saturation = [], [], []
        for side, ids in enumerate((self.wheel_left_ids, self.wheel_right_ids)):
            nd, dd = [], []
            for aid in ids:
                limit = self._actuator_tau_limit(aid)
                if limit <= 0:
                    self.data.ctrl[aid] = 0.0
                    continue
                raw = (tau_each_L, tau_each_R)[side]
                base = float(np.clip(raw, -limit, limit))
                correction = self.RESIDUAL_FRACTION * limit * mix[side]
                commanded = float(np.clip(base + correction, -limit, limit))
                self.data.ctrl[aid] = commanded
                nd.append(base/limit)
                dd.append((commanded-base)/limit)
                saturation.append(((base+correction-commanded)/limit)**2)
            nominal.append(float(np.mean(nd)) if nd else 0.0)
            deltas.append(float(np.mean(dd)) if dd else 0.0)
        self._vx_nominal = np.asarray(nominal)
        self._vx_delta = np.asarray(deltas)
        self._vx_saturation = float(np.mean(saturation)) if saturation else 0.0

    def _get_obs(self):
        original = super()._get_obs()
        geom, s, length, endpoint, preview, valid = self._route_state()
        theta = self._base_yaw()
        velocity = float(np.cos(theta)*self.data.qvel[0]+np.sin(theta)*self.data.qvel[1])
        goal = self._goal_xy()-self._base_xy()
        terminal_yaw = wrap_angle(np.arctan2(goal[1], goal[0])-theta)
        carry = self._cube_pos()-self._site_xpos('setpoint')
        budget = (1000, 3000, 1000)[int(self.phase_state)]
        extras = [
            geom[0], np.sin(geom[1])*valid, np.cos(geom[1])*valid, valid,
            velocity, self.data.qvel[5], self._last_v_des_cmd, self._last_omega_des_cmd,
            getattr(self, '_v_des_f', 0), getattr(self, '_omega_des_f', 0),
            *[getattr(self, n, 0) for n in ('_Accel_lin_des', '_Accel_ang_des', '_e_vel1', '_e_vel2', '_e_omega1', '_e_omega2', '_omega_des_prev')],
            *self._vx_nominal, *self._vx_delta, *self._vx_filter, float(self._vx_active),
            s/max(length, 1e-6), max(0, length-s), *preview, endpoint,
            np.sin(terminal_yaw), np.cos(terminal_yaw), *carry,
            *self._vx_relative_velocity, *self._cube_angular_vel(),
            max(0, 1-self._phase_steps/budget),
            *getattr(self, '_last_public_action', np.zeros(7))[:5],
        ]
        result = np.r_[original, extras].astype(np.float32)
        if result.shape != (VX_OBS_DIM,) or not np.all(np.isfinite(result)):
            raise RuntimeError(f'Observación VX inválida: {result.shape}')
        return result

    def _reward_phase1(self, pen_time, pen_energy, pen_drop):
        geom, s, length, endpoint, preview, valid = self._route_state()
        sp = self._site_xpos('setpoint')
        if self._vx_prev_sp is not None:
            self._vx_relative_velocity = self._cube_vel_vec() - (sp-self._vx_prev_sp)/self.dt
        self._vx_prev_sp = sp.copy()
        carry = float(np.linalg.norm(self._cube_pos()-sp))
        terminal = bool(valid and not self._initial_aligning and endpoint <= self.PURE_PURSUIT_TERMINAL_POSE_DIST_M)
        tracking = bool(valid and not terminal and getattr(self, '_last_base_mode', '') == 'tracking')
        previous = getattr(self, '_vx_before_route', (np.zeros(2),0,0,0,np.zeros(2),0))
        progress = 4.0*(s-previous[1]) if tracking and previous[5] else 0.0
        terminal_progress = 4.0*(previous[3]-endpoint) if terminal and previous[5] else 0.0
        capped = lambda x: min(float(x)**2, 25.0)
        track_cost = self.dt*(0.30*capped(geom[0]/0.05) + 0.10*capped(geom[1]/0.15)) if tracking else 0.0
        hold_cost = self.dt*(0.60*capped(carry/0.05) + 0.08*capped(np.linalg.norm(self._vx_relative_velocity)/0.10) + 0.03*capped(np.linalg.norm(self._cube_angular_vel())/0.5))
        goal_vec = self._goal_xy()-self._base_xy()
        heading = wrap_angle(np.arctan2(goal_vec[1],goal_vec[0])-self._base_yaw())
        terminal_cost = self.dt*0.10*capped(heading/0.15) if terminal else 0.0
        stopped = self._base_is_stopped()
        self._stop_count = self._stop_count+1 if stopped else 0
        # Handoff must be the reached route endpoint, not any point on the ring.
        ready = bool(valid and self._nav_done and endpoint <= self.FINAL_WAYPOINT_TOL
                     and abs(heading) <= self.FINAL_ROUTE_HEADING_TOL
                     and self._stop_count >= self.STOP_COUNT_MIN and self.gripping
                     and not self.cube_dropped)
        bonus = float(self.BONUS_FINAL_P1) if ready and not self._phase_success_flags[1] else 0.0
        if ready:
            self._phase_success_flags[1] = True
        terms = dict(vx_progress=progress, vx_terminal_progress=terminal_progress,
                     vx_tracking_cost=-track_cost, vx_hold_cost=-hold_cost,
                     vx_terminal_cost=-terminal_cost, vx_time_cost=float(pen_time),
                     vx_energy_cost=float(pen_energy), vx_drop=float(pen_drop), vx_arrival_bonus=bonus)
        self._last_cte_signed, self._last_yaw_track_err = geom
        self._last_track_err = abs(geom[0])
        return float(sum(terms.values())), {**terms, 'vx_carry_error_m':carry,
                'vx_relative_speed_m_s':float(np.linalg.norm(self._vx_relative_velocity)),
                'vx_tracking_active':tracking, 'vx_cte_m':float(geom[0]),
                'vx_yaw_rad':float(geom[1])}, ready

    def _check_termination(self, phase_eval):
        result = super()._check_termination(phase_eval)
        if phase_eval == 2 and not result[0] and self._phase_steps >= 1000:
            self._last_truncation_reason = 'phase2_total_budget'
            return False, True, '', 'phase2_total_budget'
        return result

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            raise ValueError('VX requiere 7 acciones finitas.')
        action = np.clip(action, -1, 1)
        phase = int(self.phase_state)
        self._vx_prev_sp = self._site_xpos('setpoint').copy()
        self._vx_before_route = self._route_state()
        previous_delta = self._vx_delta.copy()
        self._vx_active = False
        self._vx_delta[:] = 0.0
        self._vx_nominal[:] = 0.0
        self._vx_saturation = 0.0
        observation, reward, terminated, truncated, info = super().step(action)
        self._vx_relative_velocity = self._cube_vel_vec() - (self._site_xpos('setpoint')-self._vx_prev_sp)/self.dt if phase != 1 else self._vx_relative_velocity
        if not self._vx_active:
            self._vx_filter[:] = 0.0
        penalty = 0.0
        if phase == 1:
            # Costs per second; actual torque fraction, actual slew, saturation,
            # plus quartic requests (small useful actions are inexpensive).
            penalty = -self.dt*(2.0*np.mean(self._vx_delta**2)
                + 0.02*np.mean(((self._vx_delta-previous_delta)/self.dt)**2)
                + 4.0*self._vx_saturation
                + (0.05*np.mean(np.asarray(action)[5:7]**4) if self._vx_active else 0.0))
        reward += float(penalty)
        self.reward = float(reward)
        info.update(action_semantics=self.ACTION_SEMANTICS, vx_residual_cost=float(penalty),
                    vx_delta_torque_fraction=self._vx_delta.tolist(),
                    vx_nominal_torque_fraction=self._vx_nominal.tolist(),
                    vx_residual_active=self._vx_active, vx_saturation=self._vx_saturation,
                    vx_budgets=[1000,3000,1000], p2_home_budget_steps=0,
                    p2_total_budget_steps=1000, p2_home_included_in_total=True)
        return self._get_obs(), float(reward), terminated, truncated, info


# API pública V9. No se exportan aliases V8: este módulo es deliberadamente
# independiente y evita que un script cargue accidentalmente el ambiente viejo.
ZennoVXEnv = ZennoVX

__all__ = [
    "ACTION_DIM",
    "OBS_DIM",
    "VALID_BASE_MODES",
    "VALID_OBJECTIVES",
    "ZennoVX",
    "ZennoVXEnv",
    "wrap_angle",
    "yaw_to_quat_wxyz",
    "one_hot",
]


if __name__ == "__main__":
    env = ZennoVX(render_mode=None)
    try:
        observation, reset_info = env.reset(seed=7)
        print(
            "ZennoVX listo:",
            f"obs={observation.shape}",
            f"action={env.action_space.shape}",
            f"map={getattr(env, 'map_id', None)}",
            f"Ts={env.model.opt.timestep * env.frame_skip:.3f}s",
            f"routes={env.MAPS_DIR}",
        )
        observation, reward, terminated, truncated, info = env.step(
            np.zeros(ACTION_DIM, dtype=np.float32)
        )
        print(
            "Primer paso:",
            f"reward={reward:.6f}",
            f"phase={info.get('phase_state')}",
            f"terminated={terminated}",
            f"truncated={truncated}",
        )
    finally:
        env.close()
