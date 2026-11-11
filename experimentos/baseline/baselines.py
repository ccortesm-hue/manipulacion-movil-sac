#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "glfw" if "--viewer" in sys.argv else "egl")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vx_compact_matplotlib")
RESOURCE_ROOT = Path(os.environ.get("VX_RESOURCE_ROOT", str(Path(__file__).resolve().parent.parent))).resolve()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Optional, Dict, Any
from pathlib import Path
import json

# ============================================================
# RENDER GPU / OPENGL BACKEND
# ============================================================
# Debe definirse ANTES de importar mujoco / mujoco_viewer.
# Para viewer visible en Ubuntu usamos GLFW. En laptops híbridas
# Intel/NVIDIA, las variables PRIME fuerzan que OpenGL use NVIDIA
# cuando el sistema lo permite.
os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/zenno_new_mm_visual_matplotlib_cache")

import numpy as np
import mujoco

# Matplotlib se usa solo para guardar las gráficas de error de seguimiento.
# Backend Agg evita conflictos con el viewer GLFW.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


try:
    import mujoco_viewer
except ImportError:
    mujoco_viewer = None

# ====================== CONFIG ======================
SCRIPT_DIR = RESOURCE_ROOT.parent
XML = str(Path(os.environ.get("VX_BASELINE_XML", str(RESOURCE_ROOT / "Zenno_Model" / "scene.xml"))))
SHOW_LOG  = False
HEADLESS  = False  # False = ventana del viewer abierta (modo demo)

# Si True, sincroniza la simulación con el reloj de pared.
# La sincronización se hace en los frames renderizados, no en cada paso físico,
# para evitar tirones por sleeps de 2 ms.
REAL_TIME = True

# --- CONTROL / VISUALIZACIÓN ---
# El controlador fue diseñado para actualizarse cada paso físico de MuJoCo.
FRAMESKIP_CONTROL = 5

# Renderizar cada decisión de control equivale a ~100 FPS simulados si Ts = 0.002 s y frame_skip=5.
# El control corre a 100 Hz con FRAMESKIP_CONTROL = 5.
RENDER_EVERY = 1

# Resolución del viewer. Bajar resolución ayuda bastante si la visualización va lenta.
VIEWER_WIDTH = 854
VIEWER_HEIGHT = 480
VIEWER_HIDE_MENUS = True

# Límite en pasos de decisión. Con frame_skip=5 y Ts=0.002, cada decisión son 10 ms.
MAX_DECISION_STEPS = 20000  # ≈ 200 s con dt_control=0.010

# ====================== EVALUACIÓN / LOGGING ======================
# Para grabar demos cortos puedes bajar estos valores.
# Con BLOCK_EPISODES=1, cada episodio usa una masa distinta.
EPISODES_PER_MAP      = 5       # 5 masas × 1 episodio c/u → 5 episodios por mapa
BLOCK_EPISODES        = 1       # cambia masa cada episodio
AUTO_STOP_AFTER_EVAL  = True

CUBE_MASS_BY_BLOCK_KG = [0.50, 0.75, 1.00, 1.25, 1.50]
GOAL_SUCCESS_DIST_TOL = 0.050

# Resultados estructurados de la última ejecución. Los ejecutores de prueba
# pueden persistirlos sin duplicar esta máquina de estados ni parsear stdout.
EPISODE_RESULTS: List[Dict[str, Any]] = []
# Trazas ligeras de la última ejecución. Permiten que MM_TEST.py genere sus
# propias gráficas sin duplicar el lazo de control ni leer datos desde stdout.
EPISODE_TRACES: List[Dict[str, Any]] = []

# ====================== RUTAS GUARDADAS (por mapa) ======================
# El paquete de comparación reutiliza las rutas verificadas, sin sobrescribirlas.
MAPS_DIR = str(RESOURCE_ROOT / "RUTAS_SAC")
ROUTE_PLOTS_DIR = str(RESOURCE_ROOT / "data_vx" / "route_diagnostics")
SAVE_ROUTES_PER_MAP   = False
LOAD_ROUTES_ON_START  = True
SAVE_GENERATED_ROUTE_PLOTS = False

# La evaluación conserva la misma ruta para todas las masas y repeticiones.
GENERATE_NEW_RRT_ROUTE_EACH_EPISODE = False

# Controlador de seguimiento. "frenet_lqr" es la opción 2 propuesta;
# "pure_pursuit" conserva el baseline anterior para comparaciones A/B.
TRACKING_CONTROLLER = "pure_pursuit"

# ====================== GRÁFICAS DE ERROR DE SEGUIMIENTO ======================
# Se guarda una figura PNG por episodio con:
#   1) error euclidiano de la base respecto a la polilínea RRT*/órbita;
#   2) error angular geométrico de la base respecto a la tangente Frenet:
#        e_yaw_ctrl = wrap(psi_ref - yaw_base)
#
# Importante:
#   La tangente local de una polilínea puede cambiar a saltos entre segmentos
#   y producir un "serrucho" artificial. Por eso, para evaluar el control,
#   se grafica por defecto el error respecto a la tangente Frenet suavizada.
SAVE_TRACKING_ERROR_PLOTS = True
TRACKING_PLOTS_DIR        = f"plots_tracking_{TRACKING_CONTROLLER}_pid_only_tuned"

# Para diagnosticar el seguimiento real, se excluyen alineación inicial/final
# y aproximación radial. De este modo el error angular no queda contaminado
# por maniobras que no son seguimiento de la ruta.
PLOT_ONLY_PURE_TRACKING   = True

# Si True, agrega en la gráfica una curva auxiliar discontinua con el error
# respecto a la tangente local de la polilínea. Esa curva es diagnóstica;
# no debe usarse como métrica principal de orientación del controlador.
PLOT_GEOMETRIC_YAW_DIAGNOSTIC = True

# ====================== RUIDO DEL CUBO (por episodio) ======================
CUBE_XY_NOISE_RANGE      = 0.01
CUBE_YAW_NOISE_RANGE_RAD = 0.10
_cube_noise_rng          = np.random.default_rng()

# --------- Brazo / IK ---------
JOINTS = [
    "robot0:base_joint",
    "robot0:arm1_joint",
    "robot0:arm2_joint",
    "robot0:arm3_joint",
    "robot0:eef_joint",
]
EEF_SITE_NAME      = "gripper_site"
TARGET_SITE_NAME   = "target"
SETPOINT_SITE_NAME = "setpoint"

ACTS_ARM      = ["motor_j1","motor_j2","motor_j3","motor_j4","motor_j5"]
GRIP_ACT_NAME = "motor_j6"

CUBE_BODY_NAME = "cube"
PEDESTAL2_BODY = "pedestal_2"
PEDESTAL2_SITE = "goal"

OBSTACLE_INFLATION = 0.0

# ====================== RETURN HOME ======================
RETURN_HOME_TOL          = 0.06   # rad (joints dentro de esta tolerancia → home)
RETURN_HOME_MAX_TIME_S   = 8.0    # preservar duración original: 800 pasos × 0.010 s

# ====================== SETTLE TRAS DROP ======================
DROP_SETTLE_TIME_S = 0.50  # preservar duración original: 50 pasos × 0.010 s

# ======================= MAPAS PREDEFINIDOS (4 mapas) =======================
PREDEFINED_MAPS: Dict[int, List[Tuple[float, float]]] = {
    1: [(-2.0, 1.5), (-0.8, 0.8), (-0.8, 3.0)],
    2: [(-1.0, 1.7), (-0.5, 0.2), (-0.8, 2.5)],
    3: [(-2.0, 1.7), ( 0.0, 0.2), (-1.0, 2.8)],
    4: [(-1.1, 2.0), (-1.0, 0.8), ( 1.0, 1.8)],
}

MAPS_LIST = sorted(PREDEFINED_MAPS.keys())  # [1, 2, 3, 4]

MAP_SELECT_MODE = "cycle"
FIXED_MAP_ID: Optional[int] = None

_current_map_id = 1
_map_cycle_idx  = 0

MAP_CACHE: Dict[int, Dict[str, Any]] = {}

# ======================= INICIALES FIJAS (sin mapa 5) =======================
INITIALS_BY_MAP: Dict[int, Dict[str, List[float]]] = {
    1: {
        "base_qpos7": [0.69905732, -0.7008061, 0.4, 0.70710678, 0.0, 0.0, 0.70710678],
        "cube_qpos7": [0.699057321, -0.000806096824, 0.5222, 0.882061601, 0.0, 0.0, -0.471134092],
    },
    2: {
        "base_qpos7": [1.08083411, -0.5541963, 0.4, 0.46174861, 0.0, 0.0, 0.88701083],
        "cube_qpos7": [0.6793306, 0.01921013, 0.5222, 0.92530831, 0.0, 0.0, 0.37921568],
    },
    3: {
        "base_qpos7": [0.94714452, -0.63903625, 0.4, 0.57357644, 0.0, 0.0, 0.81915204],
        "cube_qpos7": [0.70773042, 0.01874858, 0.5222, 0.99996507, 0.0, 0.0, -0.0083576],
    },
    4: {
        "base_qpos7": [1.41028180, 0.0000568799682, 0.4, 6.123234e-17, 0.0, 0.0, 1.0],
        "cube_qpos7": [0.710281796, 0.0000568799682, 0.5222, 0.843043949, 0.0, 0.0, -0.537844681],
    },
}

# ======================= RESET EXACTO COMO TU RL =======================
PED1_CENTER_XY = np.array([0.7, 0.0], dtype=np.float64)
PED1_RADIUS    = 0.10
SAFE_R         = float(PED1_RADIUS - 0.021)

CUBE_HALF      = 0.0225
CUBE_Z_ON_PED1 = float(0.5 + CUBE_HALF - 0.0003)

BASE_ORBIT_R   = 0.70
BASE_REF       = np.array([0.0, 0.0], dtype=np.float64)
BASE_Z         = 0.4

GRASP_OFFSET_Z = 0.01
LIFT_AFTER     = 0.10
HOVER_UP       = 0.15

# ===================================================================
# POSE DEL BRAZO UNIFICADA CON EL ENTORNO RL (ZennoV3)
# -------------------------------------------------------------------
# En ZennoV3:
#   init_arm_positions_stage0 = {
#       "robot0:base_joint":  0.0,
#       "robot0:arm1_joint": -0.90,
#       "robot0:arm2_joint": -0.85,
#       "robot0:arm3_joint": -1.5,
#       "robot0:eef_joint":   0.0,
#   }
#   q_arm_init = q_arm_home = q_arm_carry_ref = [0.0, -0.90, -0.85, -1.5, 0.0]
# ===================================================================
INIT_Q  = np.array([0.0, -0.90, -0.85, -1.5, 0.0], dtype=float)
CARRY_Q = INIT_Q.copy()  # transporte = misma pose que init/home

XY_TOL_HOVER   = 0.005
XY_FINE_TOL    = 0.006
GRASP_DIST_TOL = 0.010

# El residuo estacionario observado con el cubo perturbado alcanza ~2.3 cm en
# algunos mapas. 2.5 cm sigue siendo menor que la tolerancia de colocación y
# evita bloquear indefinidamente una toma y elevación ya exitosas.
SETPOINT_DIST_TOL       = 0.025
SETPOINT_STABLE_STEPS   = 25
GRASP_SETTLE_STEPS      = 12
POST_CLOSE_SETTLE_STEPS = 8

# --------------------- Base / PID + RRT ---------------------
WHEEL_NAMES_L = ["tau_fl", "tau_cl", "tau_rl"]
WHEEL_NAMES_R = ["tau_fr", "tau_cr", "tau_rr"]

# ============================================================
# PID BASE @ 10 ms
# ============================================================
# Con XML Ts=0.002 s y FRAMESKIP_CONTROL=5, el control se actualiza a 0.010 s.
# Estas ganancias provienen de la calibración del entorno ZennoV3 para frame_skip=5.
BASE_CTRL_DT_TARGET = 0.010

# ============================================================
# PERFIL PID_ONLY_TUNED_P1p20_I1p00_D0p50
# ============================================================
# Resultado de PID_ONLY.py sobre cinco waypoints aleatorios desde (-5, -5):
#   lookahead=0.35 m, CTE RMS=0.016 m, saturación=4.9 %, tiempo=13.46 s.
# Escalas ganadoras respecto de FAST_10MS_V2: P=1.20, I=1.00, D=0.50.
PID_PROFILE_NAME = "PID_ONLY_TUNED_P1p20_I1p00_D0p50"
Kp_vel, Ki_vel, Kd_vel       = 66.0, 4.5, 0.025
Kp_omega, Ki_omega, Kd_omega = 864.0, 28.0, 0.080

# Anti-windup del PID incremental. La escala angular real observada trabaja en cientos.
ACC_LIN_MAX = 35.0
ACC_ANG_MAX = 330.0

# Deadband solo para integración.
VEL_ERR_DEADBAND = 0.003
OMEGA_ERR_DEADBAND = 0.004

# Rate limit de referencias para control discreto a 10 ms.
USE_REF_RATE_LIMIT = True
V_DES_RATE_MAX = 4.0       # [m/s^2]  perfil rápido @10 ms
OMEGA_DES_RATE_MAX = 14.0  # [rad/s^2] perfil rápido @10 ms

# Descarga del acumulador angular cuando cambia el sentido de giro.
RESET_ANG_ACC_ON_SIGN_FLIP = True
OMEGA_SIGN_RESET_THRESH = 0.10

V_MAX      = 0.60
OMEGA_MAX  = 0.95
V_GAIN     = 2.8
OMEGA_GAIN = 3.6
HEADING_K  = 1.0

V_REV_MAX     = 0.20
REV_SLOW_DIST = 0.20
REV_MIN_SPEED_SCALE = 0.05
REV_STOP_TOL  = 0.0005

MAX_RRT_ATTEMPTS   = 4
RRT_TIMEOUT_BASE_S = 1.8
RRT_TIMEOUT_STEP_S = 0.8
RRT_MIN_WP_POINTS  = 12

# Parámetros del planificador adaptados a Pure Pursuit con Ld=0.35 m.
# Se prueban varios puntos de llegada y ya no gana simplemente la ruta más
# corta: también se penalizan curvatura, giros locales y poco clearance.
PLANNER_ORBIT_SAMPLES = 40
PLANNER_GOAL_CANDIDATES = 10
PLANNER_SMOOTH_ITERATIONS = 2
PLANNER_COLLISION_CHECK_STEP = 0.025
PLANNER_MIN_CLEARANCE_HARD_M = 0.01
PLANNER_MIN_CLEARANCE_TARGET_M = 0.06
PLANNER_MAX_LOCAL_TURN_RAD = 0.60
# ωmax/Vmax = 1.583 1/m. Se reserva ~24 % de la autoridad angular para
# corregir CTE/yaw; por eso el percentil 95 geométrico se limita a 1.20 1/m.
PLANNER_P95_CURVATURE_MAX_1PM = 1.20
ROUTE_COST_CURVATURE_INTEGRAL_W = 0.20
ROUTE_COST_P95_CURVATURE_W = 0.35
ROUTE_COST_MAX_CURVATURE_W = 0.10
ROUTE_COST_CLEARANCE_W = 8.0

# El punto auxiliar usado por RRT* para imponer la tangente terminal no es un
# estado de navegacion ni un punto de parada. Al desactivar PREGOAL, queda
# integrado en la polilinea y la base sigue de forma continua hasta el endpoint.
USE_PREGOAL_FOR_PLANNING = True
TERMINAL_TANGENT_CONSTRAINT_M = 0.35

approach_dist    = 0.67
PREALIGN_DIST    = 0.55
USE_INITIAL_ALIGN = True
WAYPOINT_TOL     = 0.14
LOOKAHEAD_DIST   = 0.35
WAYPOINT_SPACING = 0.05

# Criterio de aceptación de tesis. Se aplica sobre la pose real de
# ``chassis_frame`` y no sobre el waypoint/lookahead interno del controlador.
# 20 mm y 3 grados son límites duros de posición terminal y orientación.
FINAL_WAYPOINT_TOL = 0.020
FINAL_ROUTE_HEADING_TOL = float(np.deg2rad(3.0))
FINAL_ROUTE_STABLE_STEPS = 10
FINAL_HEADING_BLEND_DIST = 0.20

# Seguimiento alternativo waypoint-a-waypoint. No utiliza lookahead ni la
# corrección lateral de Pure Pursuit; solo genera referencias v/omega hacia el
# siguiente waypoint para que el PID interno las regule.
WAYPOINT_PID_REACHED_TOL = 0.080
WAYPOINT_PID_V_GAIN = 5.0
WAYPOINT_PID_OMEGA_GAIN = 4.5
WAYPOINT_PID_V_MAX = 0.30

# Geometría Frenet. La tangente y la curvatura se precalculan una vez por ruta;
# el lazo de 100 Hz solo hace una búsqueda local e interpolaciones escalares.
FRENET_TANGENT_WINDOW_M = 0.32
FRENET_CURV_SMOOTH_WINDOW_M = 0.40
FRENET_HEADING_PREVIEW_M = 0.08
FRENET_CURVATURE_PREVIEW_M = 0.20
FRENET_KAPPA_MAX = 2.50

# LQR discreto para x=[e_y, e_psi], con e_psi=theta-psi_ref y
# u=omega-v*kappa. El control aplicado equivale a
# omega = v*kappa - K_y*e_y + K_psi*(psi_ref-theta).
# La DARE se resuelve una sola vez al arrancar, nunca dentro del control.
FRENET_LQR_V_NOMINAL = 0.50
FRENET_LQR_Q_Y = 8.0
FRENET_LQR_Q_PSI = 4.0
FRENET_LQR_R = 1.0

# Reserva una fracción del límite angular para que la realimentación LQR
# todavía pueda corregir mientras el feedforward anticipa una curva.
FRENET_FF_OMEGA_FRACTION = 0.72

CTE_GAIN = 1.80
CTE_CORR_MAX = 0.24

# Pure Pursuit geometrico. La curvatura se calcula como
# kappa=2*sin(alpha)/Ld y la referencia angular como omega=v*kappa. El limite
# inferior solo evita una singularidad numerica cuando el lookahead colapsa en
# el ultimo punto; no altera el lookahead nominal de 0.30 m usado por MM_TEST.
PURE_PURSUIT_MIN_LD_M = 0.08
PURE_PURSUIT_PATH_HEADING_GAIN = 2.5
PURE_PURSUIT_CTE_OMEGA_GAIN = 7.0
# En el tramo final no se persigue directamente el endpoint: hacerlo amplifica
# cualquier CTE pequeño como un gran error de bearing y obliga después a girar
# sobre el sitio. El cierre nuevo sigue la tangente continua de la ruta y
# corrige simultáneamente avance, CTE y yaw durante los últimos 0.55 m.
PURE_PURSUIT_TERMINAL_POSE_DIST_M = 0.55
PURE_PURSUIT_TERMINAL_V_GAIN = 2.4
PURE_PURSUIT_TERMINAL_V_MAX = 0.32
PURE_PURSUIT_TERMINAL_V_REVERSE_MAX = 0.10
PURE_PURSUIT_TERMINAL_PATH_HEADING_GAIN = 3.8
PURE_PURSUIT_TERMINAL_CTE_OMEGA_GAIN = 10.0
PURE_PURSUIT_TERMINAL_CURVATURE_FRACTION = 0.55
PURE_PURSUIT_TERMINAL_MIN_SPEED_SCALE = 0.22

# El tracking comienza sólo después de quedar por debajo de 2 grados. Así el
# primer transitorio medido también puede satisfacer el límite de 3 grados.
YAW_ALIGN_EXIT    = float(np.deg2rad(2.0))
OMEGA_STOP_THRESH = 0.05
MIN_OMEGA_ALIGN   = 0.14

A_BRAKE_MAX = 0.90
V_CURV      = 0.72
K_CURV      = 1.5
V_STOP_GAIN = 1.2

# Reducción de avance respecto a la tangente real de la ruta. Los umbrales se
# ubican alrededor del requisito de 3 grados, no en 20/37 grados como antes.
YAW_SLOW_1 = float(np.deg2rad(3.0))
YAW_SLOW_2 = float(np.deg2rad(7.0))
YAW_STOP = float(np.deg2rad(25.0))
YAW_SLOW_GAIN_1 = 0.80
YAW_SLOW_GAIN_2 = 0.55

# Atenuación antes de alcanzar el límite lateral de 20 mm.
CTE_SLOW_1 = 0.015
CTE_SLOW_2 = 0.025
CTE_SLOW_GAIN_1 = 0.78
CTE_SLOW_GAIN_2 = 0.58

# Rotación rápida exclusiva para alineación inicial/final.
# No modifica los parámetros de seguimiento.
INIT_ALIGN_OMEGA_MAX = 1.30
INIT_ALIGN_OMEGA_GAIN = 8.0
INIT_ALIGN_OMEGA_RATE_MAX = 40.0
INIT_ALIGN_ACC_ANG_MAX = 520.0
# La base de seis ruedas necesita superar el rozamiento lateral para girar
# sobre el sitio; una referencia pequeña de 0.18 rad/s la dejaba casi inmóvil.
INIT_ALIGN_MIN_OMEGA = 0.80

FINAL_ALIGN_OMEGA_MAX = 1.45
FINAL_ALIGN_OMEGA_GAIN = 6.50
FINAL_ALIGN_OMEGA_RATE_MAX = 40.0
FINAL_ALIGN_ACC_ANG_MAX = 520.0
FINAL_ALIGN_MIN_OMEGA = 0.0

# Valores de respaldo. Después de cargar el XML se sustituyen por la masa y la
# inercia compuestas del subárbol móvil, calculadas directamente desde MuJoCo.
m, J = 158.0, 1.5
c, b = 0.5, 0.5
r, W = 0.1, 0.699
RIGHT_SIGN = 1.0
USE_MODEL_IDENTIFIED_BASE_DYNAMICS = True
MODEL_YAW_INERTIA_SCALE = 1.0

ROBOT_RADIUS_GEOM = 0.58
TRACKING_MARGIN   = 0.03
ROBOT_RADIUS_PLAN = ROBOT_RADIUS_GEOM + TRACKING_MARGIN

RRT_STEPRANGE_FIXED  = 0.16
USE_FIXED_RRT_BOUNDS = True
RRT_FIXED_BOUNDS     = (-5.0, 5.0, -5.0, 5.0)

FINAL_GOAL_RADIUS_STOP   = 0.67
FINAL_RADIUS_TOL         = 0.06
FINAL_APPROACH_VMAX      = 0.26
FINAL_APPROACH_KV        = 2.4
FINAL_ALIGN_HEADING_TOL  = 0.10
FINAL_ALIGN_OMEGA_STOP_TOL = 999.0
FINAL_ALIGN_STABLE_STEPS = 1
FINAL_ALIGN_DECEL = 999.0
FINAL_ALIGN_CMD_TAU = 0.015
FINAL_OMEGA_MIN          = 0.5

def log(*a, **k):
    if SHOW_LOG:
        print(*a, **k)

# ======================= UTILIDADES =======================
def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi

def yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    cy = np.cos(yaw * 0.5); sy = np.sin(yaw * 0.5)
    return np.array([cy, 0.0, 0.0, sy], dtype=np.float64)

def quat2euler(quat: np.ndarray) -> np.ndarray:
    Rmat = np.zeros((9,), dtype=np.float64)
    mujoco.mju_quat2Mat(Rmat, quat.astype(np.float64))
    R = Rmat.reshape(3, 3)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 1] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0
    return np.array([x, y, z], dtype=np.float64)

def quat_wxyz_to_yaw(quat_wxyz: np.ndarray) -> float:
    Rmat = np.zeros((9,), dtype=np.float64)
    mujoco.mju_quat2Mat(Rmat, quat_wxyz.astype(np.float64))
    R = Rmat.reshape(3, 3)
    return float(np.arctan2(R[1, 0], R[0, 0]))


def base_yaw_rad(model, data) -> float:
    """Yaw real de la base móvil en radianes.

    Se prioriza la orientación del site `chassis_frame`, ya que es el mismo
    marco usado para medir la posición XY de la base. Si el site no existe,
    se usa el quaternion wxyz de qpos[3:7] como respaldo.
    """
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "chassis_frame")
    if sid != -1:
        R = np.asarray(data.site_xmat[sid], dtype=np.float64).reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    try:
        return quat_wxyz_to_yaw(np.asarray(data.qpos[3:7], dtype=np.float64))
    except Exception:
        return float(quat2euler(data.qpos[3:7])[2])

def clamp_cube_xy_to_pedestal(xy: np.ndarray) -> np.ndarray:
    v = xy - PED1_CENTER_XY
    n = float(np.linalg.norm(v))
    if n <= SAFE_R:
        return xy
    if n < 1e-12:
        return PED1_CENTER_XY.copy()
    return PED1_CENTER_XY + (SAFE_R / n) * v

def _name2id(model, mtype, name):
    jid = mujoco.mj_name2id(model, mtype, name)
    if jid < 0:
        raise RuntimeError(f"No se encontró '{name}'")
    return jid

def site_xpos(model, data, name: str) -> np.ndarray:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise RuntimeError(f"Site no encontrado: {name}")
    return data.site_xpos[sid].copy()

def base_xy(model, data) -> Tuple[float, float]:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "chassis_frame")
    if sid != -1:
        p = data.site_xpos[sid]
        return float(p[0]), float(p[1])
    return float(data.qpos[0]), float(data.qpos[1])

def pedestal2_xy(model, data) -> Tuple[float,float]:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, PEDESTAL2_SITE)
    if sid >= 0:
        p = data.site_xpos[sid]
        return float(p[0]), float(p[1])
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PEDESTAL2_BODY)
    p = model.body_pos[bid]
    return float(p[0]), float(p[1])

def get_rrt_bounds(model) -> Tuple[float, float, float, float]:
    return RRT_FIXED_BOUNDS if USE_FIXED_RRT_BOUNDS else (
        float(model.stat.center[0] - model.stat.extent),
        float(model.stat.center[0] + model.stat.extent),
        float(model.stat.center[1] - model.stat.extent),
        float(model.stat.center[1] + model.stat.extent),
    )

def clamp_point_to_bounds(p, bounds, margin=0.05):
    minx, maxx, miny, maxy = bounds
    x = float(np.clip(p[0], minx + margin, maxx - margin))
    y = float(np.clip(p[1], miny + margin, maxy - margin))
    return (x, y)

# ======================= GUARDAR/CARGAR RUTAS POR MAPA =======================
def ensure_maps_dir(folder: str = MAPS_DIR) -> Path:
    p = Path(folder); p.mkdir(parents=True, exist_ok=True); return p

def map_route_filepath(map_id: int, folder: str = MAPS_DIR) -> Path:
    return ensure_maps_dir(folder) / f"map_{int(map_id)}.json"

def save_map_route_to_folder(map_id: int, folder: str = MAPS_DIR):
    if map_id not in MAP_CACHE:
        return
    dct = MAP_CACHE[map_id]
    payload = {
        "map_id": int(map_id),
        "obstacles_xy": [(float(x), float(y)) for (x, y) in dct.get("obstacles_xy", [])],
        "base_qpos7": dct.get("base_qpos7", None).tolist() if dct.get("base_qpos7") is not None else None,
        "cube_qpos7_nominal": dct.get("cube_qpos7_nominal", None).tolist() if dct.get("cube_qpos7_nominal") is not None else None,
        "path": [(float(x), float(y)) for (x, y) in (dct.get("path") or [])] if dct.get("path") else None,
        "plan_goal_xy": list(dct.get("plan_goal_xy")) if dct.get("plan_goal_xy") is not None else None,
        "terminal_tangent_constraint_m": float(TERMINAL_TANGENT_CONSTRAINT_M),
        "tracking_controller": TRACKING_CONTROLLER,
        "lookahead_dist": float(LOOKAHEAD_DIST),
        "pid_profile": PID_PROFILE_NAME,
        "route_cost": dct.get("route_cost", None),
        "route_metrics": dct.get("route_metrics", None),
        "planner_parameters": {
            "rrt_range": float(RRT_STEPRANGE_FIXED),
            "waypoint_spacing": float(WAYPOINT_SPACING),
            "orbit_samples": int(PLANNER_ORBIT_SAMPLES),
            "goal_candidates": int(PLANNER_GOAL_CANDIDATES),
            "smooth_iterations": int(PLANNER_SMOOTH_ITERATIONS),
            "robot_radius_plan": float(ROBOT_RADIUS_PLAN),
            "p95_curvature_max_1pm": float(PLANNER_P95_CURVATURE_MAX_1PM),
            "min_clearance_hard_m": float(PLANNER_MIN_CLEARANCE_HARD_M),
        },
    }
    if dct.get("plan_pregoal_xy") is not None:
        payload["plan_pregoal_xy"] = list(dct["plan_pregoal_xy"])
    fp = map_route_filepath(map_id, folder)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[MAPS] Ruta guardada: {fp}")

def load_map_routes_from_folder(folder: str = MAPS_DIR, map_ids: Optional[List[int]] = None):
    mids = map_ids if map_ids is not None else MAPS_LIST
    ensure_maps_dir(folder)
    loaded = 0
    for mid in mids:
        fp = map_route_filepath(mid, folder)
        if not fp.exists():
            continue
        try:
            with open(fp, "r", encoding="utf-8") as f:
                payload = json.load(f)
            entry = MAP_CACHE.get(mid, {})
            entry["obstacles_xy"] = [tuple(p) for p in (payload.get("obstacles_xy") or PREDEFINED_MAPS.get(mid, []))]
            if payload.get("base_qpos7") is not None:
                entry["base_qpos7"] = np.array(payload["base_qpos7"], dtype=np.float64)
            if payload.get("cube_qpos7_nominal") is not None:
                entry["cube_qpos7_nominal"] = np.array(payload["cube_qpos7_nominal"], dtype=np.float64)
            entry["path"] = [tuple(p) for p in (payload.get("path") or [])] if payload.get("path") else None
            entry["plan_goal_xy"]    = tuple(payload["plan_goal_xy"])    if payload.get("plan_goal_xy")    else None
            entry["plan_pregoal_xy"] = tuple(payload["plan_pregoal_xy"]) if payload.get("plan_pregoal_xy") else None
            MAP_CACHE[mid] = entry
            loaded += 1
        except Exception as e:
            print(f"[MAPS] No se pudo cargar {fp}: {e}")
    if loaded > 0:
        print(f"[MAPS] Rutas cargadas desde '{folder}': {loaded} mapas")

def pick_next_map_id() -> int:
    global _map_cycle_idx
    if FIXED_MAP_ID is not None:
        return int(FIXED_MAP_ID)
    if MAP_SELECT_MODE == "cycle":
        mid = MAPS_LIST[_map_cycle_idx % len(MAPS_LIST)]
        _map_cycle_idx += 1
        return int(mid)
    return int(np.random.choice(MAPS_LIST))

# ======================= CARGA MODELO =======================
model = mujoco.MjModel.from_xml_path(XML)
data  = mujoco.MjData(model)
mujoco.mj_forward(model, data)
Ts = float(model.opt.timestep)
DT_CONTROL = Ts * FRAMESKIP_CONTROL
DT_RENDER = Ts * RENDER_EVERY


def _mobile_subtree_planar_dynamics() -> Tuple[float, float]:
    """Calcula masa e inercia yaw compuestas del robot desde el XML.

    ``body_inertia`` está expresada en los ejes principales de cada cuerpo. La
    suma usa el teorema de ejes paralelos respecto al centro de masa del
    subárbol libre que contiene ``chassis_frame``. Obstáculos, pedestales y el
    cubo libre quedan fuera porque pertenecen a otros árboles cinemáticos.
    """
    chassis_sid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "chassis_frame"
    )
    if chassis_sid < 0:
        raise RuntimeError("No se encontró el site chassis_frame para identificar la base.")
    chassis_bid = int(model.site_bodyid[chassis_sid])
    root_bid = int(model.body_rootid[chassis_bid])
    body_ids = np.flatnonzero(np.asarray(model.body_rootid) == root_bid)
    masses = np.asarray(model.body_mass[body_ids], dtype=np.float64)
    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        raise RuntimeError("La masa compuesta del subárbol móvil no es positiva.")

    body_com_world = np.asarray(data.xipos[body_ids], dtype=np.float64)
    subtree_com = np.sum(masses[:, None] * body_com_world, axis=0) / total_mass
    yaw_inertia = 0.0
    for bid, body_mass, body_com in zip(body_ids, masses, body_com_world):
        rotation = np.asarray(data.ximat[int(bid)], dtype=np.float64).reshape(3, 3)
        inertia_world = rotation @ np.diag(model.body_inertia[int(bid)]) @ rotation.T
        offset = body_com - subtree_com
        yaw_inertia += float(
            inertia_world[2, 2]
            + body_mass * (offset[0] * offset[0] + offset[1] * offset[1])
        )
    return total_mass, float(yaw_inertia)


MODEL_BASE_MASS_KG, MODEL_BASE_YAW_INERTIA_KGM2 = _mobile_subtree_planar_dynamics()
if USE_MODEL_IDENTIFIED_BASE_DYNAMICS:
    m = float(MODEL_BASE_MASS_KG)
    J = float(MODEL_BASE_YAW_INERTIA_KGM2 * MODEL_YAW_INERTIA_SCALE)

# Duraciones discretizadas en pasos de control.
# Estas cantidades se discretizan en pasos de control; con frame_skip=5
# cada paso de control equivale a 10 ms.
DROP_SETTLE_CONTROL_STEPS = max(1, int(round(DROP_SETTLE_TIME_S / DT_CONTROL)))
MAX_RETURN_HOME_CONTROL_STEPS = max(1, int(round(RETURN_HOME_MAX_TIME_S / DT_CONTROL)))

print(
    f"[CFG] Ts físico = {Ts:.6f} s | "
    f"FRAMESKIP_CONTROL = {FRAMESKIP_CONTROL} | "
    f"dt control = {DT_CONTROL:.6f} s | "
    f"RENDER_EVERY = {RENDER_EVERY} | "
    f"dt render = {DT_RENDER:.6f} s | "
    f"viewer = {VIEWER_WIDTH}x{VIEWER_HEIGHT} | "
    f"MUJOCO_GL = {os.environ.get('MUJOCO_GL')} | "
    f"m_base = {m:.3f} kg | J_yaw = {J:.3f} kg m2 | "
    f"drop_settle_steps = {DROP_SETTLE_CONTROL_STEPS} | "
    f"return_home_steps = {MAX_RETURN_HOME_CONTROL_STEPS}"
)
if abs(DT_CONTROL - BASE_CTRL_DT_TARGET) > 1e-9:
    print(
        f"[WARN][PID] DT_CONTROL={DT_CONTROL:.6f} s, pero el PID fue calibrado para "
        f"{BASE_CTRL_DT_TARGET:.6f} s. Revisa Ts o FRAMESKIP_CONTROL."
    )
_CUBE_BID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CUBE_BODY_NAME)
if _CUBE_BID < 0:
    raise RuntimeError(f"No se encontró body: {CUBE_BODY_NAME}")

# ======================= Obstáculos =======================
_OBS_BODY_NAMES = ("obstaculo1", "obstaculo2", "obstaculo3")
_OBS_BODY_IDS: List[int] = []
for nm in _OBS_BODY_NAMES:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
    if bid < 0:
        raise RuntimeError(f"No se encontró body obstáculo: {nm}")
    _OBS_BODY_IDS.append(int(bid))

def _safe_set_static_body_xy(model, body_id, x, y, keep_z=True):
    z = float(model.body_pos[body_id, 2])
    model.body_pos[body_id, 0] = float(x)
    model.body_pos[body_id, 1] = float(y)
    model.body_pos[body_id, 2] = float(z)

def set_obstacles_from_map(map_id):
    if map_id not in PREDEFINED_MAPS:
        raise ValueError(f"Mapa inválido: {map_id}")
    pts = PREDEFINED_MAPS[map_id]
    for bid, (x, y) in zip(_OBS_BODY_IDS, pts):
        _safe_set_static_body_xy(model, bid, float(x), float(y), keep_z=True)

# ======================= IDs actuadores =======================
def aid(name: str) -> int:
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if idx < 0:
        raise RuntimeError(f"Actuador no encontrado: {name}")
    return idx

wheel_ids_L = [aid(n) for n in WHEEL_NAMES_L]
wheel_ids_R = [aid(n) for n in WHEEL_NAMES_R]
arm_act_ids = [aid(n) for n in ACTS_ARM]
grip_act_id = aid(GRIP_ACT_NAME)

def actuator_tau_max(act_id):
    gear = abs(model.actuator_gear[act_id, 0])
    ctrl_hi = float(abs(model.actuator_ctrlrange[act_id, 1]) or 1.0)
    force_hi = float(abs(model.actuator_forcerange[act_id, 1]))
    return float(min(gear * ctrl_hi, force_hi)) if force_hi > 0 else float(gear * ctrl_hi)

tau_max_L = np.array([actuator_tau_max(a) for a in wheel_ids_L], dtype=float)
tau_max_R = np.array([actuator_tau_max(a) for a in wheel_ids_R], dtype=float)

# ======================= Obstáculos (planning) =======================
def _children_lists(model):
    ch = [[] for _ in range(model.nbody)]
    for b_ in range(1, model.nbody):
        p = model.body_parentid[b_]
        if p >= 0:
            ch[p].append(b_)
    return ch

def _collect_subtree(model, root_body_id):
    ch = _children_lists(model)
    stack = [root_body_id]; seen = set(stack)
    while stack:
        b_ = stack.pop()
        for c_ in ch[b_]:
            if c_ not in seen:
                seen.add(c_); stack.append(c_)
    return seen

def extract_obstacles_2d(model, data, exclude_body_names=()):
    circles = []
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "chassis_frame")
    base_body = model.site_bodyid[site_id] if site_id != -1 else -1
    robot_bodies = _collect_subtree(model, base_body) if base_body != -1 else set()
    excl = set()
    for nm in exclude_body_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
        if bid != -1:
            excl.add(bid)
    GEOM_BOX = mujoco.mjtGeom.mjGEOM_BOX
    GEOM_CYL = mujoco.mjtGeom.mjGEOM_CYLINDER
    GEOM_SPH = mujoco.mjtGeom.mjGEOM_SPHERE
    GEOM_PLN = mujoco.mjtGeom.mjGEOM_PLANE
    for gi in range(model.ngeom):
        gtype = model.geom_type[gi]
        if gtype == GEOM_PLN:
            continue
        body_id = int(model.geom_bodyid[gi])
        if body_id in robot_bodies or body_id in excl:
            continue
        cx, cy = float(data.geom_xpos[gi, 0]), float(data.geom_xpos[gi, 1])
        sx, sy, _sz = model.geom_size[gi]
        if gtype == GEOM_BOX:
            R_ = float(np.hypot(sx, sy))
        elif gtype in (GEOM_CYL, GEOM_SPH):
            R_ = float(sx)
        else:
            R_ = float(max(sx, sy))
        R_ = float(R_ + OBSTACLE_INFLATION)
        circles.append((cx, cy, R_))
    return circles

# ======================= RRT*: OMPL =======================
try:
    from ompl import base as ob
    from ompl import geometric as og
except Exception as e:
    raise ImportError("No se pudo importar OMPL.") from e

def is_xy_valid_for_planner(xy, bounds, obstacles, robot_radius):
    minx, maxx, miny, maxy = bounds
    x, y = float(xy[0]), float(xy[1])
    if not (minx <= x <= maxx and miny <= y <= maxy):
        return False
    for (cx, cy, R_) in obstacles:
        if (x - cx)**2 + (y - cy)**2 <= (R_ + robot_radius)**2:
            return False
    return True

def path_collision_free(points, obstacles, robot_radius, step=0.05):
    if len(points) < 2: return True
    P = [np.array(p, float) for p in points]
    for i in range(len(P)-1):
        a, b_ = P[i], P[i+1]
        seg = b_ - a; L = float(np.linalg.norm(seg))
        if L < 1e-9: continue
        n = max(1, int(np.ceil(L/step)))
        for k in range(n+1):
            p = a + (k/n) * seg
            for (cx, cy, R_) in obstacles:
                if (p[0]-cx)**2 + (p[1]-cy)**2 <= (R_ + robot_radius)**2:
                    return False
    return True

def resample_polyline(points, spacing, keep_end=True):
    if not points or len(points) == 1:
        return points[:]
    spacing = float(max(spacing, 1e-6))
    out = [points[0]]
    prev = np.array(points[0], dtype=float)
    dist_next = spacing
    for i in range(1, len(points)):
        cur = np.array(points[i], dtype=float)
        seg = float(np.linalg.norm(cur - prev))
        if seg < 1e-12:
            prev = cur; continue
        while seg >= dist_next:
            t = dist_next / seg
            p = prev + t * (cur - prev)
            out.append((float(p[0]), float(p[1])))
            seg -= dist_next; prev = p; dist_next = spacing
        dist_next -= seg
        prev = cur
    if keep_end and out[-1] != points[-1]:
        out.append(points[-1])
    return out


def smooth_path_for_pure_pursuit(points, obstacles, robot_radius, iterations=2):
    """Suavizado de Chaikin conservador con validación de colisiones.

    Conserva exactamente inicio y final. Cada iteración solo se acepta si toda
    la curva continúa siendo válida para el radio inflado del robot.
    """
    if points is None or len(points) < 3:
        return list(points or [])
    current = [tuple(map(float, p)) for p in points]
    for _ in range(max(0, int(iterations))):
        source = np.asarray(current, dtype=np.float64).reshape(-1, 2)
        candidate = [tuple(source[0])]
        for a, b_ in zip(source[:-1], source[1:]):
            q = 0.75 * a + 0.25 * b_
            r_ = 0.25 * a + 0.75 * b_
            candidate.extend(((float(q[0]), float(q[1])), (float(r_[0]), float(r_[1]))))
        candidate.append(tuple(source[-1]))
        if not path_collision_free(
            candidate,
            obstacles,
            robot_radius,
            step=PLANNER_COLLISION_CHECK_STEP,
        ):
            break
        current = candidate
    return current


def route_geometry_metrics(points, obstacles, robot_radius) -> Dict[str, float]:
    """Métricas geométricas relacionadas con seguimiento Pure Pursuit."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return {
            "length_m": 0.0,
            "max_turn_rad": 0.0,
            "p95_curvature_1pm": 0.0,
            "max_curvature_1pm": 0.0,
            "curvature_integral": 0.0,
            "min_clearance_m": 0.0,
        }

    delta = np.diff(pts, axis=0)
    ds = np.maximum(np.linalg.norm(delta, axis=1), 1e-9)
    headings = np.arctan2(delta[:, 1], delta[:, 0])
    turns = np.abs(np.arctan2(np.sin(np.diff(headings)), np.cos(np.diff(headings))))
    local_ds = 0.5 * (ds[:-1] + ds[1:])
    curvature = turns / np.maximum(local_ds, 1e-6)

    min_clearance = np.inf
    for p in pts:
        for cx, cy, radius in obstacles:
            clearance = float(np.hypot(p[0] - cx, p[1] - cy) - (radius + robot_radius))
            min_clearance = min(min_clearance, clearance)
    if not np.isfinite(min_clearance):
        min_clearance = 999.0

    return {
        "length_m": float(np.sum(ds)),
        "max_turn_rad": float(np.max(turns)) if turns.size else 0.0,
        "p95_curvature_1pm": float(np.percentile(curvature, 95.0)) if curvature.size else 0.0,
        "max_curvature_1pm": float(np.max(curvature)) if curvature.size else 0.0,
        "curvature_integral": (
            float(np.sum(np.square(curvature) * local_ds)) if curvature.size else 0.0
        ),
        "min_clearance_m": float(min_clearance),
    }


def pure_pursuit_route_cost(metrics: Dict[str, float]) -> float:
    clearance_deficit = max(
        0.0,
        float(PLANNER_MIN_CLEARANCE_TARGET_M) - float(metrics["min_clearance_m"]),
    )
    return float(
        metrics["length_m"]
        + ROUTE_COST_CURVATURE_INTEGRAL_W * metrics["curvature_integral"]
        + ROUTE_COST_P95_CURVATURE_W * metrics["p95_curvature_1pm"]
        + ROUTE_COST_MAX_CURVATURE_W * metrics["max_curvature_1pm"]
        + ROUTE_COST_CLEARANCE_W * clearance_deficit
    )


def save_generated_route_plot(
    *,
    map_id: int,
    path,
    obstacles,
    start_xy,
    pregoal_xy,
    orbit_xy,
    goal_center_xy,
    metrics: Dict[str, float],
    route_cost: float,
    output_dir: str = ROUTE_PLOTS_DIR,
) -> Optional[Path]:
    """Plot XY de ruta, obstáculos reales y envolvente usada al planificar."""
    if not SAVE_GENERATED_ROUTE_PLOTS:
        return None
    if plt is None:
        print("[WARN][PLOT] matplotlib no está disponible; no se graficará la ruta.")
        return None

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f"map_{int(map_id)}_pure_pursuit_route.png"
    pts = np.asarray(path, dtype=np.float64).reshape(-1, 2)

    fig, ax = plt.subplots(figsize=(9.0, 8.0), constrained_layout=True)
    for idx, (cx, cy, radius) in enumerate(obstacles):
        ax.add_patch(plt.Circle(
            (cx, cy), radius + ROBOT_RADIUS_PLAN,
            fill=False, color="tab:red", linestyle="--", linewidth=1.0,
            alpha=0.65,
            label=(
                f"espacio de configuración (huella={ROBOT_RADIUS_PLAN:.2f} m)"
                if idx == 0 else None
            ),
        ))
        ax.add_patch(plt.Circle(
            (cx, cy), radius,
            color="0.35", alpha=0.55,
            label="obstáculo físico" if idx == 0 else None,
        ))

    ax.plot(pts[:, 0], pts[:, 1], color="tab:blue", linewidth=2.0, label="ruta optimizada")
    stride = max(1, len(pts) // 18)
    arrow_pts = pts[::stride]
    if len(arrow_pts) >= 2:
        arrow_delta = np.diff(arrow_pts, axis=0)
        ax.quiver(
            arrow_pts[:-1, 0], arrow_pts[:-1, 1],
            arrow_delta[:, 0], arrow_delta[:, 1],
            angles="xy", scale_units="xy", scale=1.0,
            width=0.003, color="tab:blue", alpha=0.55,
        )

    ax.scatter(*start_xy, s=80, color="tab:cyan", edgecolor="black", zorder=5, label="inicio")
    if pregoal_xy is not None:
        ax.scatter(
            *pregoal_xy, s=70, color="tab:orange", edgecolor="black",
            zorder=5, label="pre-goal",
        )
    ax.scatter(*orbit_xy, s=80, color="tab:green", edgecolor="black", zorder=5, label="fin ruta")
    ax.scatter(*goal_center_xy, s=130, marker="*", color="gold", edgecolor="black", zorder=6, label="goal")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(
        f"Mapa {map_id} | Pure Pursuit Ld={LOOKAHEAD_DIST:.2f} m | "
        f"L={metrics['length_m']:.2f} m | costo={route_cost:.2f}\n"
        f"κ95={metrics['p95_curvature_1pm']:.2f} 1/m | "
        f"κmax={metrics['max_curvature_1pm']:.2f} 1/m | "
        f"clearance min={metrics['min_clearance_m']:.3f} m"
    )
    fig.savefig(fig_path, dpi=170)
    plt.close(fig)
    return fig_path


def save_all_maps_route_plot(map_cache: Dict[int, Dict[str, Any]]) -> Optional[Path]:
    """Crea un resumen 2x2 cuando ya existen rutas de los cuatro mapas."""
    if not SAVE_GENERATED_ROUTE_PLOTS or plt is None:
        return None
    if any(not (map_cache.get(mid) or {}).get("path") for mid in MAPS_LIST):
        return None

    out_dir = Path(ROUTE_PLOTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "all_maps_pure_pursuit_routes.png"
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)

    for ax, mid in zip(axes.flat, MAPS_LIST):
        entry = map_cache[mid]
        pts = np.asarray(entry["path"], dtype=np.float64).reshape(-1, 2)
        for cx, cy, radius in entry.get("planner_obstacles", []):
            ax.add_patch(plt.Circle(
                (cx, cy), radius + ROBOT_RADIUS_PLAN,
                fill=False, color="tab:red", linestyle="--", linewidth=0.8, alpha=0.55,
            ))
            ax.add_patch(plt.Circle((cx, cy), radius, color="0.4", alpha=0.45))
        ax.plot(pts[:, 0], pts[:, 1], color="tab:blue", linewidth=1.8)
        ax.scatter(pts[0, 0], pts[0, 1], color="tab:cyan", edgecolor="black", zorder=5)
        ax.scatter(pts[-1, 0], pts[-1, 1], color="tab:green", edgecolor="black", zorder=5)
        goal_xy = entry.get("goal_center_xy")
        if goal_xy is not None:
            ax.scatter(*goal_xy, marker="*", s=100, color="gold", edgecolor="black", zorder=6)
        metrics = entry.get("route_metrics", {})
        ax.set_title(
            f"Mapa {mid} | L={metrics.get('length_m', 0.0):.2f} m | "
            f"κ95={metrics.get('p95_curvature_1pm', 0.0):.2f} 1/m"
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"Rutas optimizadas para Pure Pursuit | lookahead={LOOKAHEAD_DIST:.2f} m | {PID_PROFILE_NAME}",
        fontsize=14,
    )
    fig.savefig(fig_path, dpi=170)
    plt.close(fig)
    return fig_path

def plan_rrtstar(start_xy, goal_xy, bounds, obstacles,
                 robot_radius=ROBOT_RADIUS_PLAN, timeout=2.0, range_step=None):
    space = ob.RealVectorStateSpace(2)
    minx, maxx, miny, maxy = bounds
    sb = ob.RealVectorBounds(2)
    sb.setLow(0, minx); sb.setHigh(0, maxx)
    sb.setLow(1, miny); sb.setHigh(1, maxy)
    space.setBounds(sb)
    si = ob.SpaceInformation(space)

    def is_state_valid(state):
        x = float(state[0]); y = float(state[1])
        if not (minx <= x <= maxx and miny <= y <= maxy):
            return False
        for (cx, cy, R_) in obstacles:
            if (x - cx)**2 + (y - cy)**2 <= (R_ + robot_radius)**2:
                return False
        return True

    si.setStateValidityChecker(ob.StateValidityCheckerFn(is_state_valid))
    si.setStateValidityCheckingResolution(0.002)
    start_xy = clamp_point_to_bounds(start_xy, bounds, margin=0.02)
    goal_xy  = clamp_point_to_bounds(goal_xy,  bounds, margin=0.02)
    start = ob.State(space); start[0], start[1] = float(start_xy[0]), float(start_xy[1])
    goal  = ob.State(space); goal[0],  goal[1]  = float(goal_xy[0]),  float(goal_xy[1])
    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start, goal, 0.10)
    pdef.setOptimizationObjective(ob.PathLengthOptimizationObjective(si))
    planner = og.RRTstar(si)
    if range_step is not None:
        planner.setRange(float(range_step))
    planner.setProblemDefinition(pdef)
    planner.setup()
    status = planner.solve(float(timeout))
    try:
        if hasattr(pdef, "hasExactSolution"):
            if not pdef.hasExactSolution():
                return []
        else:
            if not status:
                return []
            if hasattr(status, "asString"):
                s = status.asString().lower()
                if "exact" not in s:
                    return []
    except Exception:
        return []
    path = pdef.getSolutionPath()
    ss = og.PathSimplifier(si)
    if hasattr(ss, "reduceVertices"):
        try: ss.reduceVertices(path)
        except Exception: pass
    for fn in ("smoothBSpline", "bsplineSmooth"):
        if hasattr(ss, fn):
            try: getattr(ss, fn)(path)
            except Exception: pass
    try: path.interpolate(200)
    except Exception: pass
    states = path.getStates()
    pts = [(float(s[0]), float(s[1])) for s in states]
    if not path_collision_free(pts, obstacles, robot_radius, step=0.05):
        return []
    return pts

# ======================= PLANIFICACIÓN/ESTADOS =======================
waypoints: List[Tuple[float, float]] = []
_wp_idx                = 0
_plan_ready            = False
_reached_final         = False
_final_aligning        = False
_initial_aligning      = False
_final_radial_approach = False
_final_align_omega_cmd_f = 0.0

_last_plan: Dict[str, Any] = {
    "goal": {"path": None, "obstacles": None, "start": None, "pregoal": None, "orbit": None, "goal_center": None},
}

_reverse_active = False
_reverse_done   = False
_reverse_goal   = np.array([0.0, 0.0], dtype=float)

def sample_orbit_goals(center_xy, radius, bounds, obstacles, robot_radius, n=24, margin=0.05, random_rotate=True):
    cx, cy = float(center_xy[0]), float(center_xy[1])
    phi0 = float(np.random.uniform(0.0, 2.0*np.pi)) if random_rotate else 0.0
    out = []
    for k in range(int(n)):
        a = phi0 + (2.0*np.pi * k / float(n))
        gx = cx + radius * float(np.cos(a))
        gy = cy + radius * float(np.sin(a))
        gx, gy = clamp_point_to_bounds((gx, gy), bounds, margin=margin)
        if is_xy_valid_for_planner((gx, gy), bounds, obstacles, robot_radius):
            out.append((gx, gy))
    return out

def _make_orbit_and_pregoal(center_xy, orbit_xy, prealign_dist, bounds):
    G = np.array(center_xy, dtype=float); P = np.array(orbit_xy, dtype=float)
    v = (G - P); n = float(np.linalg.norm(v))
    u = (v / n) if n >= 1e-9 else np.array([1.0, 0.0], dtype=float)
    P_pre = P - float(prealign_dist) * u
    P_pre = np.array(clamp_point_to_bounds((float(P_pre[0]), float(P_pre[1])), bounds, margin=0.02), dtype=float)
    return (float(P_pre[0]), float(P_pre[1])), (float(P[0]), float(P[1]))

def _enforce_last_segment_colinear_to_goal(path, goal_center_xy, orbit_xy, spacing):
    if not path: return path
    G = np.array(goal_center_xy, dtype=float); P_last = np.array(orbit_xy, dtype=float)
    v = G - P_last; n = float(np.linalg.norm(v))
    if n < 1e-9: return path
    u = v / n
    if len(path) == 1: return [orbit_xy]
    P_prev = P_last - float(max(spacing, 1e-3)) * u
    new = path[:-2] + [(float(P_prev[0]), float(P_prev[1])), (float(P_last[0]), float(P_last[1]))]
    return new

def replan_path_to_goal_for_current_map():
    global waypoints, _wp_idx, _plan_ready, _reached_final, _initial_aligning, _final_aligning, _final_radial_approach
    global _last_plan

    cached = (
        None
        if GENERATE_NEW_RRT_ROUTE_EACH_EPISODE
        else MAP_CACHE.get(_current_map_id, None)
    )
    if cached is not None and cached.get("path", None):
        waypoints[:] = cached["path"][:]
        cached_length = sum(
            float(np.hypot(
                waypoints[i + 1][0] - waypoints[i][0],
                waypoints[i + 1][1] - waypoints[i][1],
            ))
            for i in range(len(waypoints) - 1)
        )
        _build_frenet_path_cache()
        _wp_idx = 0
        _plan_ready = True; _reached_final = False
        _initial_aligning = True; _final_aligning = False; _final_radial_approach = False
        obstacles = extract_obstacles_2d(model, data, exclude_body_names=("cube", "pedestal_2"))
        gc = pedestal2_xy(model, data)
        _last_plan["goal"] = {
            "path": waypoints[:], "obstacles": obstacles,
            "start": base_xy(model, data),
            "pregoal": cached.get("plan_pregoal_xy"),
            "orbit": cached.get("plan_goal_xy"),
            "goal_center": gc,
        }
        log(f"[RRT] Reusando ruta cacheada mapa {_current_map_id}")
        print(
            f"[RRT][CARGADA] mapa={_current_map_id} | puntos={len(waypoints)} | "
            f"longitud={cached_length:.3f} m | fuente={map_route_filepath(_current_map_id, MAPS_DIR)}"
        )
        if SAVE_ROUTES_PER_MAP:
            fp = map_route_filepath(_current_map_id, MAPS_DIR)
            if not fp.exists():
                save_map_route_to_folder(_current_map_id, MAPS_DIR)
        return True

    bxy = base_xy(model, data)
    goal_center_xy = pedestal2_xy(model, data)
    obstacles = extract_obstacles_2d(model, data, exclude_body_names=("cube", "pedestal_2"))
    bounds = get_rrt_bounds(model)
    bxy_c = clamp_point_to_bounds(bxy, bounds, margin=0.02)

    candidates_orbit = sample_orbit_goals(
        center_xy=goal_center_xy, radius=approach_dist,
        bounds=bounds, obstacles=obstacles,
        robot_radius=ROBOT_RADIUS_PLAN,
        n=PLANNER_ORBIT_SAMPLES,
        margin=0.05,
        random_rotate=True,
    )
    if not candidates_orbit:
        print("[RRT] Sin puntos válidos en la órbita.")
        waypoints[:] = []; _clear_frenet_path_cache(); _plan_ready = False
        return False

    candidates_orbit.sort(key=lambda g: (g[0]-bxy_c[0])**2 + (g[1]-bxy_c[1])**2)

    best_path = []
    best_cost = np.inf
    best_metrics: Dict[str, float] = {}
    best_orbit = None
    best_pregoal = None

    for attempt in range(1, MAX_RRT_ATTEMPTS + 1):
        timeout_s = float(RRT_TIMEOUT_BASE_S + (attempt - 1) * RRT_TIMEOUT_STEP_S)
        K = min(int(PLANNER_GOAL_CANDIDATES), len(candidates_orbit))
        for gi in range(K):
            orbit_xy = clamp_point_to_bounds(candidates_orbit[gi], bounds, margin=0.02)
            terminal_constraint_m = (
                PREALIGN_DIST
                if USE_PREGOAL_FOR_PLANNING
                else TERMINAL_TANGENT_CONSTRAINT_M
            )
            terminal_entry_xy, orbit_xy = _make_orbit_and_pregoal(
                goal_center_xy, orbit_xy, terminal_constraint_m, bounds
            )
            if not is_xy_valid_for_planner(terminal_entry_xy, bounds, obstacles, ROBOT_RADIUS_PLAN):
                continue
            if not is_xy_valid_for_planner(orbit_xy, bounds, obstacles, ROBOT_RADIUS_PLAN):
                continue
            if not path_collision_free([terminal_entry_xy, orbit_xy], obstacles, ROBOT_RADIUS_PLAN, step=0.03):
                continue
            pts_try = plan_rrtstar(bxy_c, terminal_entry_xy, bounds, obstacles,
                                   robot_radius=ROBOT_RADIUS_PLAN, timeout=timeout_s,
                                   range_step=RRT_STEPRANGE_FIXED)
            if not pts_try: continue
            if pts_try[-1] != terminal_entry_xy: pts_try.append(terminal_entry_xy)
            # Se agrega la órbita ANTES de suavizar. Así Chaikin redondea la
            # unión RRT→pre-goal que concentraba los picos de curvatura. Como
            # conserva extremos y el último vector, mantiene la tangente radial.
            pts_try.append(orbit_xy)
            pts_try = smooth_path_for_pure_pursuit(
                pts_try,
                obstacles,
                ROBOT_RADIUS_PLAN,
                iterations=PLANNER_SMOOTH_ITERATIONS,
            )
            pts_try = resample_polyline(pts_try, WAYPOINT_SPACING, keep_end=True)
            if len(pts_try) < int(RRT_MIN_WP_POINTS): continue
            if not path_collision_free(
                pts_try,
                obstacles,
                ROBOT_RADIUS_PLAN,
                step=PLANNER_COLLISION_CHECK_STEP,
            ):
                continue
            metrics = route_geometry_metrics(pts_try, obstacles, ROBOT_RADIUS_PLAN)
            if metrics["max_turn_rad"] > float(PLANNER_MAX_LOCAL_TURN_RAD):
                continue
            if metrics["p95_curvature_1pm"] > float(PLANNER_P95_CURVATURE_MAX_1PM):
                continue
            if metrics["min_clearance_m"] < float(PLANNER_MIN_CLEARANCE_HARD_M):
                continue
            route_cost = pure_pursuit_route_cost(metrics)
            if route_cost < best_cost:
                best_cost = route_cost
                best_metrics = metrics
                best_path = pts_try
                best_orbit = orbit_xy
                best_pregoal = (
                    terminal_entry_xy if USE_PREGOAL_FOR_PLANNING else None
                )
        if best_path:
            break

    if not best_path:
        print(f"[RRT] FALLÓ tras {MAX_RRT_ATTEMPTS} intentos.")
        waypoints[:] = []; _clear_frenet_path_cache(); _wp_idx = 0; _plan_ready = False; _reached_final = False
        _initial_aligning = False; _final_aligning = False
        return False

    waypoints[:] = best_path
    _build_frenet_path_cache()
    _wp_idx = 0; _plan_ready = True; _reached_final = False
    _initial_aligning = True; _final_aligning = False

    _last_plan["goal"] = {
        "path": best_path[:], "obstacles": obstacles[:],
        "start": bxy_c, "pregoal": best_pregoal, "orbit": best_orbit, "goal_center": goal_center_xy,
    }
    log(f"[RRT] OK. waypoints={len(best_path)}")
    print(
        f"[RRT][NUEVA] mapa={_current_map_id} | puntos={len(best_path)} | "
        f"longitud={best_metrics['length_m']:.3f} m | costo={best_cost:.3f} | "
        f"kappa95={best_metrics['p95_curvature_1pm']:.3f} 1/m | "
        f"kappa_max={best_metrics['max_curvature_1pm']:.3f} 1/m | "
        f"clearance={best_metrics['min_clearance_m']:.3f} m"
    )

    route_plot = save_generated_route_plot(
        map_id=_current_map_id,
        path=best_path,
        obstacles=obstacles,
        start_xy=bxy_c,
        pregoal_xy=best_pregoal,
        orbit_xy=best_orbit,
        goal_center_xy=goal_center_xy,
        metrics=best_metrics,
        route_cost=float(best_cost),
    )
    if route_plot is not None:
        print(f"[RRT][PLOT] {route_plot}")

    if _current_map_id not in MAP_CACHE:
        MAP_CACHE[_current_map_id] = {"obstacles_xy": PREDEFINED_MAPS[_current_map_id][:]}
    MAP_CACHE[_current_map_id]["path"] = best_path[:]
    MAP_CACHE[_current_map_id]["plan_goal_xy"] = best_orbit
    MAP_CACHE[_current_map_id]["plan_pregoal_xy"] = best_pregoal
    MAP_CACHE[_current_map_id]["route_cost"] = float(best_cost)
    MAP_CACHE[_current_map_id]["route_metrics"] = dict(best_metrics)
    MAP_CACHE[_current_map_id]["route_plot"] = str(route_plot) if route_plot is not None else None
    MAP_CACHE[_current_map_id]["planner_obstacles"] = [tuple(map(float, obs)) for obs in obstacles]
    MAP_CACHE[_current_map_id]["goal_center_xy"] = tuple(map(float, goal_center_xy))

    if SAVE_ROUTES_PER_MAP:
        save_map_route_to_folder(_current_map_id, MAPS_DIR)
    all_maps_plot = save_all_maps_route_plot(MAP_CACHE)
    if all_maps_plot is not None:
        print(f"[RRT][PLOT RESUMEN] {all_maps_plot}")
    return True

# ======================= SEGUIMIENTO / GEOMETRÍA =======================
def _advance_wp_if_reached(x, y):
    global _wp_idx
    if not waypoints:
        return False

    _cte, _px, _py, seg_idx, _seg_yaw = nearest_path_segment_projection(
        np.array([x, y], dtype=np.float64), waypoints,
        start_idx=_wp_idx, search_back=2, search_ahead=32,
    )
    if seg_idx > _wp_idx:
        _wp_idx = int(seg_idx)

    advanced = False
    while _wp_idx < len(waypoints) - 1:
        gx, gy = waypoints[_wp_idx]
        if np.hypot(gx - x, gy - y) > float(WAYPOINT_TOL):
            break
        _wp_idx += 1
        advanced = True

    return bool(advanced)


def _advance_waypoint_pid_if_reached(x: float, y: float) -> bool:
    """Avanza secuencialmente al alcanzar o rebasar el waypoint actual."""
    global _wp_idx, _reached_final
    if not waypoints or _reached_final:
        return False
    position = np.array([x, y], dtype=np.float64)
    advanced = False
    while _wp_idx < len(waypoints) - 1:
        target = np.asarray(waypoints[_wp_idx], dtype=np.float64)
        distance = float(np.linalg.norm(target - position))
        passed = False
        if _wp_idx > 0:
            previous = np.asarray(waypoints[_wp_idx - 1], dtype=np.float64)
            direction = target - previous
            passed = bool(float(np.dot(position - target, direction)) >= 0.0)
        if distance > float(WAYPOINT_PID_REACHED_TOL) and not passed:
            break
        _wp_idx += 1
        advanced = True

    if _wp_idx >= len(waypoints) - 1:
        target = np.asarray(waypoints[-1], dtype=np.float64)
        distance = float(np.linalg.norm(target - position))
        passed = False
        if len(waypoints) > 1:
            previous = np.asarray(waypoints[-2], dtype=np.float64)
            direction = target - previous
            passed = bool(float(np.dot(position - target, direction)) >= 0.0)
        if distance <= float(WAYPOINT_PID_REACHED_TOL) or passed:
            _reached_final = True
            advanced = True
    return advanced

def _choose_lookahead_target(x, y, start_idx, lookahead):
    """Lookahead continuo medido sobre la ruta desde la proyección actual."""
    global _wp_idx
    if (not waypoints) or len(waypoints) < 2:
        return float(x), float(y), int(start_idx)

    _cte, px, py, seg_idx, _seg_yaw = nearest_path_segment_projection(
        np.array([x, y], dtype=np.float64), waypoints,
        start_idx=int(start_idx), search_back=2, search_ahead=40,
    )
    seg_idx = int(np.clip(seg_idx, 0, len(waypoints) - 2))
    _wp_idx = max(int(_wp_idx), seg_idx)

    curr = np.array([px, py], dtype=np.float64)
    remaining = float(max(lookahead, 1e-6))
    j = seg_idx
    while j < len(waypoints) - 1:
        endpoint = np.asarray(waypoints[j + 1], dtype=np.float64)
        seg_vec = endpoint - curr
        seg_len = float(np.linalg.norm(seg_vec))
        if seg_len >= remaining:
            target = curr + (remaining / max(seg_len, 1e-9)) * seg_vec
            return float(target[0]), float(target[1]), int(j)
        remaining -= seg_len
        j += 1
        curr = np.asarray(waypoints[j], dtype=np.float64)

    return float(waypoints[-1][0]), float(waypoints[-1][1]), len(waypoints) - 1


def nearest_path_segment_projection(
    p_xy: np.ndarray,
    pts: List[Tuple[float, float]],
    *,
    start_idx: int = 0,
    search_back: int = 3,
    search_ahead: int = 24,
) -> Tuple[float, float, float, int, float]:
    """Proyección firmada del punto sobre el segmento más cercano de la ruta.

    Retorna:
        cte_signed:
            error lateral firmado [m]. Positivo si la base está a la izquierda
            de la dirección del segmento A→B.
        px, py:
            punto proyectado sobre el segmento.
        seg_idx:
            índice del segmento más cercano.
        seg_yaw:
            orientación tangencial del segmento.
    """
    if pts is None or len(pts) < 2:
        return 0.0, float(p_xy[0]), float(p_xy[1]), 0, 0.0

    P = np.asarray(p_xy, dtype=np.float64).reshape(2)

    i0 = max(0, int(start_idx) - int(search_back))
    i1 = min(len(pts) - 2, int(start_idx) + int(search_ahead))

    best_dist2 = np.inf
    best = (0.0, float(P[0]), float(P[1]), i0, 0.0)

    for i in range(i0, i1 + 1):
        A = np.asarray(pts[i], dtype=np.float64).reshape(2)
        B = np.asarray(pts[i + 1], dtype=np.float64).reshape(2)
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


def signed_cross_track_error(
    p_xy: np.ndarray,
    pts: List[Tuple[float, float]],
    *,
    start_idx: int = 0,
) -> float:
    """Error lateral firmado de la base respecto a la ruta."""
    cte_signed, _px, _py, _seg_idx, _seg_yaw = nearest_path_segment_projection(
        p_xy,
        pts,
        start_idx=start_idx,
    )
    return float(cte_signed)


# ======================= REFERENCIA FRENET + LQR =======================
_frenet_pts = np.empty((0, 2), dtype=np.float64)
_frenet_s = np.empty((0,), dtype=np.float64)
_frenet_yaw_unwrapped = np.empty((0,), dtype=np.float64)
_frenet_kappa = np.empty((0,), dtype=np.float64)


def _solve_frenet_dlqr_gains() -> Tuple[float, float]:
    """Resuelve una DARE 2x1 una sola vez para el periodo real de control."""
    dt = float(max(DT_CONTROL, 1e-6))
    v0 = float(max(FRENET_LQR_V_NOMINAL, 1e-3))

    # x=[e_y, e_psi], e_psi=theta-psi_ref:
    # x[k+1] = A*x[k] + B*(omega-v*kappa).
    A = np.array([[1.0, v0 * dt], [0.0, 1.0]], dtype=np.float64)
    B = np.array([[0.0], [dt]], dtype=np.float64)
    Q = np.diag([float(FRENET_LQR_Q_Y), float(FRENET_LQR_Q_PSI)])
    R_lqr = np.array([[float(FRENET_LQR_R)]], dtype=np.float64)

    P = Q.copy()
    converged = False
    for _ in range(10000):
        S = R_lqr + B.T @ P @ B
        P_next = A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(S, B.T @ P @ A) + Q
        if float(np.max(np.abs(P_next - P))) < 1e-11:
            P = P_next
            converged = True
            break
        P = P_next

    if not converged:
        raise RuntimeError("La DARE del controlador Frenet-LQR no convergió.")

    K = np.linalg.solve(R_lqr + B.T @ P @ B, B.T @ P @ A).reshape(-1)
    if K.size != 2 or not np.all(np.isfinite(K)):
        raise RuntimeError("Ganancias Frenet-LQR inválidas.")
    return float(K[0]), float(K[1])


FRENET_K_Y, FRENET_K_PSI = _solve_frenet_dlqr_gains()


def _clear_frenet_path_cache():
    global _frenet_pts, _frenet_s, _frenet_yaw_unwrapped, _frenet_kappa
    _frenet_pts = np.empty((0, 2), dtype=np.float64)
    _frenet_s = np.empty((0,), dtype=np.float64)
    _frenet_yaw_unwrapped = np.empty((0,), dtype=np.float64)
    _frenet_kappa = np.empty((0,), dtype=np.float64)


def _build_frenet_path_cache():
    """Precalcula arco, tangente continua y curvatura suavizada de la ruta."""
    global _frenet_pts, _frenet_s, _frenet_yaw_unwrapped, _frenet_kappa

    if len(waypoints) < 2:
        _clear_frenet_path_cache()
        return

    pts = np.asarray(waypoints, dtype=np.float64).reshape(-1, 2).copy()
    ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    # La ruta resampleada no debería contener duplicados; el mínimo evita que
    # una coincidencia numérica rompa la derivada espacial.
    ds_safe = np.maximum(ds, 1e-6)
    s = np.concatenate(([0.0], np.cumsum(ds_safe)))

    half_tangent = 0.5 * float(max(FRENET_TANGENT_WINDOW_M, WAYPOINT_SPACING))
    yaw = np.zeros(len(pts), dtype=np.float64)
    for i in range(len(pts)):
        i0 = int(np.searchsorted(s, s[i] - half_tangent, side="left"))
        i1 = int(np.searchsorted(s, s[i] + half_tangent, side="right") - 1)
        i0 = int(np.clip(i0, 0, len(pts) - 1))
        i1 = int(np.clip(i1, 0, len(pts) - 1))
        if i1 <= i0:
            i0 = max(0, i - 1)
            i1 = min(len(pts) - 1, i + 1)
        chord = pts[i1] - pts[i0]
        if float(np.linalg.norm(chord)) < 1e-9:
            j = min(i, len(pts) - 2)
            chord = pts[j + 1] - pts[j]
        yaw[i] = float(np.arctan2(chord[1], chord[0]))

    yaw_u = np.unwrap(yaw)
    kappa_raw = np.gradient(yaw_u, s, edge_order=1)

    half_smooth = 0.5 * float(max(FRENET_CURV_SMOOTH_WINDOW_M, WAYPOINT_SPACING))
    kappa = np.zeros_like(kappa_raw)
    for i in range(len(pts)):
        j0 = int(np.searchsorted(s, s[i] - half_smooth, side="left"))
        j1 = int(np.searchsorted(s, s[i] + half_smooth, side="right"))
        kappa[i] = float(np.mean(kappa_raw[j0:max(j1, j0 + 1)]))
    kappa = np.clip(kappa, -float(FRENET_KAPPA_MAX), float(FRENET_KAPPA_MAX))

    _frenet_pts = pts
    _frenet_s = s
    _frenet_yaw_unwrapped = yaw_u
    _frenet_kappa = kappa


def _frenet_reference_at_pose(
    p_xy: np.ndarray,
    *,
    start_idx: int,
) -> Tuple[float, float, float, int, float, float, float]:
    """Retorna e_y, proyección, segmento, psi_ref, kappa_preview y s restante."""
    if len(_frenet_pts) != len(waypoints) or len(_frenet_s) < 2:
        _build_frenet_path_cache()
    if len(_frenet_s) < 2:
        p = np.asarray(p_xy, dtype=np.float64).reshape(2)
        return 0.0, float(p[0]), float(p[1]), 0, 0.0, 0.0, 0.0

    p = np.asarray(p_xy, dtype=np.float64).reshape(2)
    cte, px, py, seg_idx, _seg_yaw = nearest_path_segment_projection(
        p,
        waypoints,
        start_idx=int(start_idx),
        search_back=2,
        search_ahead=40,
    )
    seg_idx = int(np.clip(seg_idx, 0, len(_frenet_pts) - 2))

    A = _frenet_pts[seg_idx]
    Bp = _frenet_pts[seg_idx + 1]
    AB = Bp - A
    L2 = float(np.dot(AB, AB))
    proj = np.array([px, py], dtype=np.float64)
    t_seg = 0.0 if L2 < 1e-12 else float(np.clip(np.dot(proj - A, AB) / L2, 0.0, 1.0))
    s_proj = float(_frenet_s[seg_idx] + t_seg * (_frenet_s[seg_idx + 1] - _frenet_s[seg_idx]))

    s_heading = min(s_proj + float(FRENET_HEADING_PREVIEW_M), float(_frenet_s[-1]))
    s_curv = min(s_proj + float(FRENET_CURVATURE_PREVIEW_M), float(_frenet_s[-1]))
    psi_ref = wrap_angle(float(np.interp(s_heading, _frenet_s, _frenet_yaw_unwrapped)))
    kappa_ref = float(np.interp(s_curv, _frenet_s, _frenet_kappa))
    remaining_s = float(max(_frenet_s[-1] - s_proj, 0.0))

    return float(cte), float(px), float(py), seg_idx, float(psi_ref), kappa_ref, remaining_s


def _continuous_path_reference_at_pose(
    p_xy: np.ndarray,
    *,
    start_idx: int = 0,
    global_search: bool = False,
) -> Tuple[float, float, float, int, float]:
    """Proyeccion y tangente continua de la ruta en el arco proyectado.

    Esta referencia es la usada para evaluar el error de orientacion de ruta.
    No contiene preview ni filtra el error medido: interpola en longitud de
    arco la tangente geometrica precalculada de la ruta.
    """
    if len(_frenet_pts) != len(waypoints) or len(_frenet_s) < 2:
        _build_frenet_path_cache()
    p = np.asarray(p_xy, dtype=np.float64).reshape(2)
    if len(_frenet_s) < 2:
        return 0.0, float(p[0]), float(p[1]), 0, 0.0

    cte, px, py, seg_idx, _seg_yaw = nearest_path_segment_projection(
        p,
        waypoints,
        start_idx=0 if global_search else int(start_idx),
        search_back=0 if global_search else 2,
        search_ahead=max(len(waypoints), 1) if global_search else 40,
    )
    seg_idx = int(np.clip(seg_idx, 0, len(_frenet_pts) - 2))
    A = _frenet_pts[seg_idx]
    Bp = _frenet_pts[seg_idx + 1]
    AB = Bp - A
    L2 = float(np.dot(AB, AB))
    projection = np.array([px, py], dtype=np.float64)
    ratio = 0.0 if L2 < 1e-12 else float(
        np.clip(np.dot(projection - A, AB) / L2, 0.0, 1.0)
    )
    s_projection = float(
        _frenet_s[seg_idx]
        + ratio * (_frenet_s[seg_idx + 1] - _frenet_s[seg_idx])
    )
    path_heading = wrap_angle(
        float(np.interp(s_projection, _frenet_s, _frenet_yaw_unwrapped))
    )
    return float(cte), float(px), float(py), seg_idx, float(path_heading)

# ======================= PID/CONTROL BASE =======================
_e_vel1 = _e_vel2 = 0.0
_e_omega1 = _e_omega2 = 0.0
Accel_lin_des = 0.0
Accel_ang_des = 0.0
_v_des_f = 0.0
_omega_des_f = 0.0
_omega_des_prev = 0.0

# Diagnóstico simple del último modo/comando de base.
_base_control_mode = "idle"
_base_v_des_cmd = 0.0
_base_omega_des_cmd = 0.0

# Señales capturadas en el mismo instante de decisión del controlador lateral.
_last_tracking_active = False
_last_cte_signed = _last_cte_corr = _last_track_err = 0.0
_last_yaw_track_err = _last_theta_des_raw = _last_theta_des_corr = 0.0
_last_tracking_base_yaw = 0.0
_last_tracking_wp_idx = -1
_last_tracking_lookahead_xy = (0.0, 0.0)
_terminal_route_stable_count = 0

def reset_tracking_control_diag():
    global _last_tracking_active, _last_cte_signed, _last_cte_corr, _last_track_err
    global _last_yaw_track_err, _last_theta_des_raw, _last_theta_des_corr
    global _last_tracking_base_yaw, _last_tracking_wp_idx, _last_tracking_lookahead_xy
    _last_tracking_active = False
    _last_cte_signed = _last_cte_corr = _last_track_err = 0.0
    _last_yaw_track_err = _last_theta_des_raw = _last_theta_des_corr = 0.0
    _last_tracking_base_yaw = 0.0
    _last_tracking_wp_idx = -1
    _last_tracking_lookahead_xy = (0.0, 0.0)

def stop_wheels():
    for act in (wheel_ids_L + wheel_ids_R):
        data.ctrl[act] = 0.0

def hard_stop_base():
    """Corte inmediato equivalente a ZennoV3._hard_stop_base()."""
    stop_wheels()
    data.qvel[0] = 0.0
    data.qvel[1] = 0.0
    data.qvel[5] = 0.0

def reset_pid_accums():
    global Accel_lin_des, Accel_ang_des, _e_vel1, _e_vel2, _e_omega1, _e_omega2
    global _v_des_f, _omega_des_f, _omega_des_prev
    Accel_lin_des = 0.0
    Accel_ang_des = 0.0
    _e_vel1 = _e_vel2 = 0.0
    _e_omega1 = _e_omega2 = 0.0
    _v_des_f = 0.0
    _omega_des_f = 0.0
    _omega_des_prev = 0.0

def zero_base_qvel():
    data.qvel[0:6] = 0.0
    if hasattr(data, "qacc_warmstart") and data.qacc_warmstart is not None:
        data.qacc_warmstart[0:6] = 0.0

def start_reverse(distance_m=0.35):
    global _reverse_active, _reverse_done, _reverse_goal, _plan_ready, _reached_final
    x0, y0 = base_xy(model, data)
    yaw0 = base_yaw_rad(model, data)
    dir_vec = np.array([np.cos(yaw0), np.sin(yaw0)], dtype=float)
    _reverse_goal = np.array([x0, y0]) - distance_m * dir_vec
    _reverse_active = True
    _reverse_done = False
    _plan_ready = False
    _reached_final = False
    reset_pid_accums()
    stop_wheels()
    log(f"[REV] Inicia retroceso: {distance_m:.2f} m atrás.")

def _apply_wheel_ctrl_with_clip(tau_each_L, tau_each_R):
    for i, act in enumerate(wheel_ids_L):
        raw = float(tau_each_L)
        clipped = float(np.clip(raw, -tau_max_L[i], tau_max_L[i]))
        data.ctrl[act] = clipped
    for i, act in enumerate(wheel_ids_R):
        raw = float(tau_each_R * RIGHT_SIGN)
        clipped = float(np.clip(raw, -tau_max_R[i], tau_max_R[i]))
        data.ctrl[act] = clipped

def control_base_vel(
    v_des: float,
    omega_des: float,
    *,
    omega_max_override: Optional[float] = None,
    omega_rate_max_override: Optional[float] = None,
    acc_ang_max_override: Optional[float] = None,
    mode: str = "tracking",
):
    """Control PID incremental de base calibrado para decisiones cada DT_CONTROL.

    Los overrides permiten acelerar solo alineación inicial/final sin alterar
    el seguimiento normal de trayectoria.
    """
    global Accel_lin_des, Accel_ang_des
    global _e_vel1, _e_vel2, _e_omega1, _e_omega2
    global _v_des_f, _omega_des_f, _omega_des_prev
    global _base_control_mode, _base_v_des_cmd, _base_omega_des_cmd

    Ts_ctrl = float(max(DT_CONTROL, 1e-6))
    _base_control_mode = str(mode)
    _base_v_des_cmd = float(v_des)
    _base_omega_des_cmd = float(omega_des)

    theta = base_yaw_rad(model, data)
    vx_w = float(data.qvel[0])
    vy_w = float(data.qvel[1])
    v_body = float(np.cos(theta) * vx_w + np.sin(theta) * vy_w)
    omega_z = float(data.qvel[5])

    omega_max = float(OMEGA_MAX if omega_max_override is None else omega_max_override)
    v_des = float(np.clip(v_des, -V_MAX, V_MAX))
    omega_des = float(np.clip(omega_des, -omega_max, omega_max))

    if RESET_ANG_ACC_ON_SIGN_FLIP:
        prev = float(_omega_des_prev)
        th = float(OMEGA_SIGN_RESET_THRESH)
        if abs(prev) > th and abs(omega_des) > th and np.sign(prev) != np.sign(omega_des):
            Accel_ang_des = 0.0
            _e_omega1 = 0.0
            _e_omega2 = 0.0
    _omega_des_prev = float(omega_des)

    if USE_REF_RATE_LIMIT:
        dv_max = float(V_DES_RATE_MAX) * Ts_ctrl
        omega_rate_max = float(OMEGA_DES_RATE_MAX if omega_rate_max_override is None else omega_rate_max_override)
        dw_max = omega_rate_max * Ts_ctrl
        _v_des_f += float(np.clip(v_des - _v_des_f, -dv_max, dv_max))
        _omega_des_f += float(np.clip(omega_des - _omega_des_f, -dw_max, dw_max))
        v_des_pid = float(_v_des_f)
        omega_des_pid = float(_omega_des_f)
    else:
        v_des_pid = float(v_des)
        omega_des_pid = float(omega_des)

    e_vel = float(v_des_pid) - v_body
    e_omega = float(omega_des_pid) - omega_z

    e_vel_i = 0.0 if abs(e_vel) < float(VEL_ERR_DEADBAND) else e_vel
    e_omega_i = 0.0 if abs(e_omega) < float(OMEGA_ERR_DEADBAND) else e_omega

    Accel_lin_des += (
        Kp_vel * (e_vel - _e_vel1)
        + Ki_vel * e_vel_i * Ts_ctrl
        + Kd_vel * (e_vel - 2.0 * _e_vel1 + _e_vel2) / Ts_ctrl
    )
    Accel_ang_des += (
        Kp_omega * (e_omega - _e_omega1)
        + Ki_omega * e_omega_i * Ts_ctrl
        + Kd_omega * (e_omega - 2.0 * _e_omega1 + _e_omega2) / Ts_ctrl
    )

    acc_ang_max = float(ACC_ANG_MAX if acc_ang_max_override is None else acc_ang_max_override)
    Accel_lin_des = float(np.clip(Accel_lin_des, -ACC_LIN_MAX, ACC_LIN_MAX))
    Accel_ang_des = float(np.clip(Accel_ang_des, -acc_ang_max, acc_ang_max))

    _e_vel2, _e_vel1 = _e_vel1, e_vel
    _e_omega2, _e_omega1 = _e_omega1, e_omega

    Torque_R = (m * r / 2.0) * Accel_lin_des + (J * r / W) * Accel_ang_des + (c * r / 2.0) * v_body + (b * r / W) * omega_z
    Torque_L = (m * r / 2.0) * Accel_lin_des - (J * r / W) * Accel_ang_des + (c * r / 2.0) * v_body - (b * r / W) * omega_z

    _apply_wheel_ctrl_with_clip(Torque_L / max(len(wheel_ids_L), 1), Torque_R / max(len(wheel_ids_R), 1))

def control_base_step_nav_to_goal():
    global _plan_ready, _wp_idx, _reached_final, _initial_aligning, _final_aligning
    global _reverse_active, _reverse_done, _final_radial_approach
    global _final_align_omega_cmd_f
    global _last_tracking_active, _last_cte_signed, _last_cte_corr, _last_track_err
    global _last_yaw_track_err, _last_theta_des_raw, _last_theta_des_corr
    global _last_tracking_base_yaw, _last_tracking_wp_idx, _last_tracking_lookahead_xy
    global _terminal_route_stable_count

    reset_tracking_control_diag()

    if _reverse_active:
        x, y = base_xy(model, data)
        pos = np.array([x, y], dtype=float)
        rem = float(np.linalg.norm(_reverse_goal - pos))
        slow_near = float(np.clip(
            rem / REV_SLOW_DIST, REV_MIN_SPEED_SCALE, 1.0
        ))

        control_base_vel(
            v_des=-V_REV_MAX * slow_near,
            omega_des=0.0,
            mode="reverse",
        )

        if rem <= REV_STOP_TOL:
            stop_wheels()
            reset_pid_accums()
            zero_base_qvel()
            _reverse_active = False
            _reverse_done = True
            log("[REV] Retroceso completado.")
        return

    if not _plan_ready or len(waypoints) == 0:
        stop_wheels()
        return

    x, y = base_xy(model, data)
    theta = base_yaw_rad(model, data)

    # 1) Alineamiento inicial: rotación rápida, sin modificar tracking.
    if (not _reached_final) and (not _final_aligning) and (not _final_radial_approach):
        if _initial_aligning:
            if TRACKING_CONTROLLER == "frenet_lqr" and len(waypoints) >= 2:
                (
                    _cte0, _px0, _py0, align_seg_idx,
                    theta_des0, _kappa0, _remaining0,
                ) = _frenet_reference_at_pose(
                    np.array([x, y], dtype=np.float64),
                    start_idx=_wp_idx,
                )
                align_idx = int(align_seg_idx)
            elif TRACKING_CONTROLLER == "pure_pursuit" and len(waypoints) >= 2:
                # Debe usarse exactamente la misma tangente continua con la que
                # se calcula el error experimental. Alinear hacia wp+2 podía
                # liberar el tracking con 8--11 grados de error real de ruta.
                (
                    _cte0, _px0, _py0, align_seg_idx, theta_des0,
                ) = _continuous_path_reference_at_pose(
                    np.array([x, y], dtype=np.float64),
                    start_idx=_wp_idx,
                    global_search=True,
                )
                align_idx = int(align_seg_idx)
            elif TRACKING_CONTROLLER == "waypoint_pid":
                # El primer punto coincide con la posición inicial de la base.
                # Se alinea al siguiente waypoint, sin saltar puntos de la ruta.
                align_idx = min(max(_wp_idx + 1, 1), len(waypoints) - 1)
                gx_align, gy_align = waypoints[align_idx]
                theta_des0 = float(np.arctan2(gy_align - y, gx_align - x))
            else:
                align_idx = min(max(_wp_idx + 2, 1), len(waypoints) - 1)
                gx_align, gy_align = waypoints[align_idx]
                theta_des0 = float(np.arctan2(gy_align - y, gx_align - x))
            e_theta0 = wrap_angle(theta_des0 - theta)

            if abs(e_theta0) <= YAW_ALIGN_EXIT:
                _initial_aligning = False
                _wp_idx = min(align_idx, len(waypoints) - 1)
                reset_pid_accums()
                hard_stop_base()
                return

            raw_omega = float(np.clip(
                INIT_ALIGN_OMEGA_GAIN * e_theta0,
                -INIT_ALIGN_OMEGA_MAX,
                INIT_ALIGN_OMEGA_MAX,
            ))

            omega_cmd = (
                float(np.sign(e_theta0) * INIT_ALIGN_MIN_OMEGA)
                if abs(raw_omega) < INIT_ALIGN_MIN_OMEGA
                else raw_omega
            )

            control_base_vel(
                v_des=0.0,
                omega_des=omega_cmd,
                omega_max_override=INIT_ALIGN_OMEGA_MAX,
                omega_rate_max_override=INIT_ALIGN_OMEGA_RATE_MAX,
                acc_ang_max_override=INIT_ALIGN_ACC_ANG_MAX,
                mode="initial_align",
            )
            return

    # 2) Seguimiento de trayectoria. El selector permite comparar el nuevo
    #    Frenet-LQR contra el baseline Pure Pursuit sin cambiar el PID interno.
    if (not _reached_final) and (not _final_aligning) and (not _final_radial_approach):
        if TRACKING_CONTROLLER == "waypoint_pid":
            _advance_waypoint_pid_if_reached(x, y)
        else:
            _advance_wp_if_reached(x, y)

        last_x, last_y = waypoints[-1]
        d_last = float(np.hypot(float(last_x) - x, float(last_y) - y))
        terminal_dx = float(waypoints[-1][0] - waypoints[-2][0])
        terminal_dy = float(waypoints[-1][1] - waypoints[-2][1])
        terminal_yaw = float(np.arctan2(terminal_dy, terminal_dx))

        # El modo waypoint historico conserva su alineacion/aproximacion final.
        # Pure Pursuit y Frenet deben completar directamente el endpoint.
        if TRACKING_CONTROLLER == "waypoint_pid" and _reached_final:
            _reached_final = False
            _final_aligning = True
            _final_radial_approach = False
            _final_align_omega_cmd_f = 0.0
            reset_pid_accums()
            stop_wheels()
            log(
                "[BASE] Entra a alineación final | "
                f"reason=last_wp | wp_idx={_wp_idx}/{len(waypoints)-1} | "
                f"d_last={d_last:.4f} m"
            )
            return

        if TRACKING_CONTROLLER == "waypoint_pid":
            gx, gy = waypoints[min(_wp_idx, len(waypoints) - 1)]
            lookahead_seg_idx = int(_wp_idx)
        else:
            gx, gy, lookahead_seg_idx = _choose_lookahead_target(
                x,
                y,
                _wp_idx,
                LOOKAHEAD_DIST,
            )

        e_wp = float(np.hypot(gx - x, gy - y))

        if TRACKING_CONTROLLER == "frenet_lqr":
            (
                cte_signed, _proj_x, _proj_y, frenet_seg_idx,
                psi_ref, kappa_ref, _remaining_s,
            ) = _frenet_reference_at_pose(
                np.array([x, y], dtype=np.float64),
                start_idx=_wp_idx,
            )
            _wp_idx = max(int(_wp_idx), int(frenet_seg_idx))

            # e_theta > 0 significa que la tangente deseada queda a la
            # izquierda del rumbo actual. cte_signed > 0 significa que la
            # base está a la izquierda de la ruta y requiere giro a derecha.
            theta_des = float(psi_ref)
            theta_des_corr = float(psi_ref)
            e_theta = wrap_angle(float(psi_ref) - theta)
            cte_corr = float(-FRENET_K_Y * cte_signed)  # componente de omega [rad/s]
        elif TRACKING_CONTROLLER == "pure_pursuit":
            cte_signed = signed_cross_track_error(
                np.array([x, y], dtype=np.float64),
                waypoints,
                start_idx=_wp_idx,
            )
            # Pure Pursuit canonico: el error lateral ya queda incorporado en
            # la geometria del punto lookahead; no se resta otra correccion de
            # CTE al rumbo porque eso duplica la realimentacion lateral.
            cte_corr = 0.0
            theta_des = float(np.arctan2(gy - y, gx - x))
            theta_des_corr = float(theta_des)
            e_theta = wrap_angle(theta_des_corr - theta)
            (
                _cte_path_ctrl, _px_path_ctrl, _py_path_ctrl,
                _seg_path_ctrl, psi_path_ctrl,
            ) = _continuous_path_reference_at_pose(
                np.array([x, y], dtype=np.float64),
                start_idx=_wp_idx,
            )
            path_heading_error = wrap_angle(float(psi_path_ctrl) - theta)
            lookahead_actual = max(
                float(np.hypot(gx - x, gy - y)),
                float(PURE_PURSUIT_MIN_LD_M),
            )
            kappa_ref = float(2.0 * np.sin(e_theta) / lookahead_actual)
        elif TRACKING_CONTROLLER == "waypoint_pid":
            cte_signed = signed_cross_track_error(
                np.array([x, y], dtype=np.float64),
                waypoints,
                start_idx=max(_wp_idx - 1, 0),
            )
            cte_corr = 0.0
            theta_des = float(np.arctan2(gy - y, gx - x))
            theta_des_corr = float(theta_des)
            e_theta = wrap_angle(theta_des - theta)
            kappa_ref = 0.0
            path_heading_error = float(e_theta)
        else:
            raise ValueError(
                f"TRACKING_CONTROLLER={TRACKING_CONTROLLER!r} no válido; "
                "usa 'frenet_lqr', 'pure_pursuit' o 'waypoint_pid'."
            )

        _last_tracking_active = True
        _last_cte_signed = float(cte_signed)
        _last_cte_corr = float(cte_corr)
        _last_track_err = float(abs(cte_signed))
        _last_yaw_track_err = float(e_theta)
        _last_theta_des_raw = float(theta_des)
        _last_theta_des_corr = float(theta_des_corr)
        _last_tracking_base_yaw = float(theta)
        _last_tracking_wp_idx = int(_wp_idx)
        _last_tracking_lookahead_xy = (float(gx), float(gy))

        # Cierre terminal: posición y orientación deben permanecer dentro de
        # 20 mm / 3 grados durante varias decisiones consecutivas.
        if TRACKING_CONTROLLER in ("pure_pursuit", "frenet_lqr"):
            terminal_heading_error = wrap_angle(terminal_yaw - theta)
            if (
                d_last <= float(FINAL_WAYPOINT_TOL)
                and abs(terminal_heading_error) <= float(FINAL_ROUTE_HEADING_TOL)
            ):
                _terminal_route_stable_count += 1
            else:
                _terminal_route_stable_count = 0
            if _terminal_route_stable_count >= int(FINAL_ROUTE_STABLE_STEPS):
                _reached_final = True
                reset_pid_accums()
                hard_stop_base()
                log(
                    "[BASE] Ruta completa en endpoint | "
                    f"d_last={d_last:.4f} m | "
                    f"yaw_err={terminal_heading_error:.4f} rad | "
                    f"stable={_terminal_route_stable_count}"
                )
                return

        # Regulador terminal continuo sobre la tangente de la ruta. A diferencia
        # del regulador polar anterior, no apunta al endpoint ni detiene la
        # traslación para realinear. El error longitudinal regula v; el CTE y el
        # yaw se corrigen durante el movimiento. Un sobrepaso pequeño se recupera
        # mediante retroceso limitado, conservando la orientación de la tangente.
        if (
            TRACKING_CONTROLLER == "pure_pursuit"
            and d_last <= float(PURE_PURSUIT_TERMINAL_POSE_DIST_M)
        ):
            terminal_tangent = np.array(
                [np.cos(terminal_yaw), np.sin(terminal_yaw)], dtype=np.float64
            )
            endpoint_error = np.array([last_x - x, last_y - y], dtype=np.float64)
            along_track_error = float(np.dot(endpoint_error, terminal_tangent))
            lateral_error = float(
                terminal_tangent[0] * (y - last_y)
                - terminal_tangent[1] * (x - last_x)
            )

            terminal_fraction = float(np.clip(
                1.0 - d_last / max(PURE_PURSUIT_TERMINAL_POSE_DIST_M, 1e-6),
                0.0,
                1.0,
            ))
            terminal_blend = terminal_fraction * terminal_fraction * (
                3.0 - 2.0 * terminal_fraction
            )
            terminal_path_yaw = float(wrap_angle(
                float(psi_path_ctrl)
                + terminal_blend * wrap_angle(terminal_yaw - float(psi_path_ctrl))
            ))
            terminal_heading_error = float(wrap_angle(terminal_path_yaw - theta))

            terminal_v_raw = float(PURE_PURSUIT_TERMINAL_V_GAIN * along_track_error)
            terminal_v = float(np.clip(
                terminal_v_raw,
                -PURE_PURSUIT_TERMINAL_V_REVERSE_MAX,
                PURE_PURSUIT_TERMINAL_V_MAX,
            ))
            geometry_scale = float(np.clip(
                abs(along_track_error)
                / max(abs(along_track_error) + 2.0 * abs(lateral_error), 1e-6),
                PURE_PURSUIT_TERMINAL_MIN_SPEED_SCALE,
                1.0,
            ))
            heading_scale_terminal = float(
                1.0 / (1.0 + 1.5 * abs(terminal_heading_error))
            )
            terminal_v *= geometry_scale * heading_scale_terminal

            drive_sign = 1.0 if terminal_v >= 0.0 else -1.0
            terminal_omega = float(np.clip(
                (1.0 - terminal_blend)
                * PURE_PURSUIT_TERMINAL_CURVATURE_FRACTION
                * terminal_v
                * kappa_ref
                + PURE_PURSUIT_TERMINAL_PATH_HEADING_GAIN * terminal_heading_error
                - drive_sign * PURE_PURSUIT_TERMINAL_CTE_OMEGA_GAIN * lateral_error,
                -OMEGA_MAX,
                OMEGA_MAX,
            ))
            _last_cte_signed = float(lateral_error)
            _last_cte_corr = float(-drive_sign * PURE_PURSUIT_TERMINAL_CTE_OMEGA_GAIN * lateral_error)
            _last_track_err = float(abs(lateral_error))
            _last_yaw_track_err = float(terminal_heading_error)
            _last_theta_des_raw = float(terminal_path_yaw)
            _last_theta_des_corr = float(terminal_path_yaw)
            control_base_vel(
                v_des=terminal_v,
                omega_des=terminal_omega,
                mode="tracking",
            )
            return

        heading_scale = float(1.0 / (1.0 + HEADING_K * abs(e_theta)))
        slow_near_wp = float(np.clip(e_wp / max(WAYPOINT_TOL, 1e-3), 0.2, 1.0))

        if TRACKING_CONTROLLER == "waypoint_pid":
            Vel_lin_by_path = (
                float(np.clip(WAYPOINT_PID_V_GAIN * e_wp, 0.0, WAYPOINT_PID_V_MAX))
                * heading_scale
            )
        else:
            Vel_lin_by_path = (
                float(np.clip(V_GAIN * e_wp, 0.0, V_MAX))
                * heading_scale
                * slow_near_wp
            )

        if TRACKING_CONTROLLER == "frenet_lqr":
            # Limita solo cuando la curvatura consumiría casi toda la autoridad
            # angular; así queda margen para corregir e_y y e_psi.
            if abs(kappa_ref) > 1e-6:
                v_ff_cap = float(FRENET_FF_OMEGA_FRACTION * OMEGA_MAX / abs(kappa_ref))
                Vel_lin_by_path = min(Vel_lin_by_path, v_ff_cap)
            omega_fb = float(-FRENET_K_Y * cte_signed + FRENET_K_PSI * e_theta)
            Vel_ang_des = float(np.clip(
                Vel_lin_by_path * kappa_ref + omega_fb,
                -OMEGA_MAX,
                OMEGA_MAX,
            ))
        elif TRACKING_CONTROLLER == "pure_pursuit":
            omega_path_feedback = float(
                PURE_PURSUIT_PATH_HEADING_GAIN * path_heading_error
                - PURE_PURSUIT_CTE_OMEGA_GAIN * cte_signed
            )
            Vel_ang_des = float(np.clip(
                Vel_lin_by_path * kappa_ref + omega_path_feedback,
                -OMEGA_MAX,
                OMEGA_MAX,
            ))
        else:  # waypoint_pid
            Vel_ang_des = float(np.clip(
                WAYPOINT_PID_OMEGA_GAIN * e_theta,
                -OMEGA_MAX,
                OMEGA_MAX,
            ))

        v_cap_curv = float(V_CURV / (1.0 + K_CURV * abs(Vel_ang_des)))
        Vel_lin_des = float(np.clip(min(Vel_lin_by_path, v_cap_curv), 0.0, V_MAX))

        # Seguridad específica para tracking a 10 ms:
        # no avanzar fuerte si apunta demasiado fuera de la tangente corregida.
        speed_heading_error = (
            path_heading_error
            if TRACKING_CONTROLLER == "pure_pursuit"
            else e_theta
        )
        abs_e_theta = abs(float(speed_heading_error))
        if abs_e_theta > YAW_STOP:
            Vel_lin_des = 0.0
        elif abs_e_theta > YAW_SLOW_2:
            Vel_lin_des *= YAW_SLOW_GAIN_2
        elif abs_e_theta > YAW_SLOW_1:
            Vel_lin_des *= YAW_SLOW_GAIN_1

        # Cierre lateral de trayectoria:
        # si el error lateral ya está creciendo, se reduce la velocidad
        # antes de seguir abriendo la curva. Esto ataca el pico de ~8 cm
        # observado en Mapa 1 sin frenar toda la ruta.
        cte_abs = abs(float(cte_signed))
        if cte_abs > CTE_SLOW_2:
            Vel_lin_des *= CTE_SLOW_GAIN_2
        elif cte_abs > CTE_SLOW_1:
            Vel_lin_des *= CTE_SLOW_GAIN_1

        if TRACKING_CONTROLLER == "frenet_lqr":
            # El feedforward depende de la velocidad que realmente se solicita;
            # se recalcula tras todas las protecciones para no sobregirar al frenar.
            omega_ff = float(Vel_lin_des * kappa_ref)
            omega_fb = float(-FRENET_K_Y * cte_signed + FRENET_K_PSI * e_theta)
            Vel_ang_des = float(np.clip(omega_ff + omega_fb, -OMEGA_MAX, OMEGA_MAX))
        elif TRACKING_CONTROLLER == "pure_pursuit":
            # La expresion canonica depende de la velocidad finalmente
            # solicitada, despues de todas las protecciones de tracking.
            omega_path_feedback = float(
                PURE_PURSUIT_PATH_HEADING_GAIN * path_heading_error
                - PURE_PURSUIT_CTE_OMEGA_GAIN * cte_signed
            )
            Vel_ang_des = float(np.clip(
                Vel_lin_des * kappa_ref + omega_path_feedback,
                -OMEGA_MAX,
                OMEGA_MAX,
            ))

        control_base_vel(
            v_des=Vel_lin_des,
            omega_des=Vel_ang_des,
            mode="tracking",
        )
        return

    # 3) Alineamiento final: rotación rápida, sin modificar tracking.
    if _final_aligning and (not _final_radial_approach):
        goal_cx, goal_cy = pedestal2_xy(model, data)
        theta_goal = float(np.arctan2(goal_cy - y, goal_cx - x))
        e_theta = wrap_angle(theta_goal - theta)

        if abs(e_theta) <= FINAL_ALIGN_HEADING_TOL:
            _final_aligning = False
            _final_radial_approach = True
            reset_pid_accums()
            _final_align_omega_cmd_f = 0.0
            hard_stop_base()
            log("[BASE] Alineación final OK.")
            return

        omega_cmd = float(np.clip(
            FINAL_ALIGN_OMEGA_GAIN * e_theta,
            -FINAL_ALIGN_OMEGA_MAX,
            FINAL_ALIGN_OMEGA_MAX,
        ))
        tau = float(max(FINAL_ALIGN_CMD_TAU, 1e-6))
        beta = 1.0 - float(np.exp(-DT_CONTROL / tau))
        _final_align_omega_cmd_f += beta * (omega_cmd - _final_align_omega_cmd_f)

        control_base_vel(
            v_des=0.0,
            omega_des=float(_final_align_omega_cmd_f),
            omega_max_override=FINAL_ALIGN_OMEGA_MAX,
            omega_rate_max_override=FINAL_ALIGN_OMEGA_RATE_MAX,
            acc_ang_max_override=FINAL_ALIGN_ACC_ANG_MAX,
            mode="final_align",
        )
        return

    # 4) Aproximación radial final: parámetros normales del seguimiento.
    if _final_radial_approach:
        goal_cx, goal_cy = pedestal2_xy(model, data)

        d_goal = float(np.hypot(x - goal_cx, y - goal_cy))
        radial_err = d_goal - FINAL_GOAL_RADIUS_STOP

        theta_goal = float(np.arctan2(goal_cy - y, goal_cx - x))
        e_theta = wrap_angle(theta_goal - theta)

        if abs(radial_err) <= FINAL_RADIUS_TOL:
            _final_radial_approach = False
            _reached_final = True
            reset_pid_accums()
            stop_wheels()
            zero_base_qvel()
            log(f"[BASE] Detención final OK: d_goal={d_goal:.4f} m")
            return

        raw_omega = float(np.clip(
            OMEGA_GAIN * e_theta,
            -OMEGA_MAX,
            OMEGA_MAX,
        ))

        if abs(raw_omega) < FINAL_OMEGA_MIN and abs(e_theta) > 1e-4:
            Vel_ang_des = float(np.sign(e_theta) * FINAL_OMEGA_MIN)
        else:
            Vel_ang_des = raw_omega

        v_cmd = float(np.clip(
            FINAL_APPROACH_KV * radial_err,
            -FINAL_APPROACH_VMAX,
            FINAL_APPROACH_VMAX,
        ))

        if abs(e_theta) > 0.35:
            v_cmd *= 0.25

        control_base_vel(
            v_des=v_cmd,
            omega_des=Vel_ang_des,
            mode="final_radial",
        )
        return

    stop_wheels()

# ======================= IK (DLS) =======================
@dataclass
class IKParams:
    weight_xy:float=3.0; weight_z:float=1.0
    K_max:float=14.0; K_min:float=3.0; distance_threshold:float=0.01
    alpha:float=0.45; max_delta:float=0.25
    nullspace_K:float=0.02
    q_bias:np.ndarray=field(default_factory=lambda:np.array([0.0,-1.0,-1.0,-1.4,0.0]))
    z_only_xy_weight:float=0.05; z_only_z_weight:float=6.0
    dls_lambda0:float=3e-3; dls_lambda_max:float=8e-2; dls_gain_scale:float=10.0
    d_slow:float=0.05; alpha_min:float=0.25
    z_err_cap_down:float=0.008; z_err_cap_up:float=0.020

class ArmIK:
    def __init__(self,model,data,joint_names,eef_site_name,actuator_names=None,params=None):
        self.model,self.data,self.params=model,data,params or IKParams()
        self.ARM_DOF_IDX,self.ARM_QPOS_ADR=self._arm_indices_from_names(model,joint_names)
        self.sid_eef=_name2id(model,mujoco.mjtObj.mjOBJ_SITE,eef_site_name)
        self.act_ids=self._resolve_actuators(model,actuator_names or [])
        self._Jp=np.zeros((3,model.nv)); self._Jo=np.zeros((3,model.nv))

    def _arm_indices_from_names(self,model,names):
        dof,qpos=[],[]
        for n in names:
            j=_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)
            dof.append(int(model.jnt_dofadr[j])); qpos.append(int(model.jnt_qposadr[j]))
        return np.array(dof),np.array(qpos)

    def _resolve_actuators(self,model,names):
        out=[]
        for n in names:
            try: out.append(_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,n))
            except: out.append(-1)
        return np.array(out)

    def get_eef_pos(self): return self.data.site_xpos[self.sid_eef].copy()
    def _clip(self,act,q):
        if act<0: return q
        lo,hi=self.model.actuator_ctrlrange[act]
        return float(np.clip(q,lo,hi))

    def step(self,target,*,lock_xy=None,z_only=False):
        p=self.params
        mujoco.mj_jacSite(self.model,self.data,self._Jp,self._Jo,self.sid_eef)
        J=self._Jp[:,self.ARM_DOF_IDX]; p_now=self.get_eef_pos()
        err=target-p_now
        if lock_xy is not None: err[0:2]=lock_xy-p_now[0:2]
        if z_only:
            cap=p.z_err_cap_down if err[2]<0 else p.z_err_cap_up
            err[2]=np.sign(err[2])*min(abs(err[2]),cap)
        dist=np.linalg.norm(err)
        gain=p.K_max if dist>p.distance_threshold else p.K_min+(p.K_max-p.K_min)*(dist/max(p.distance_threshold,1e-9))
        alpha_eff=p.alpha*max(p.alpha_min,dist/max(p.d_slow,1e-9))
        wxy,wz=(p.z_only_xy_weight,p.z_only_z_weight) if z_only else (p.weight_xy,p.weight_z)
        e_w=np.array([wxy*err[0],wxy*err[1],wz*err[2]])
        lam=np.clip(p.dls_lambda0*(1+p.dls_gain_scale*dist),p.dls_lambda0,p.dls_lambda_max)
        dq_ts=J.T@np.linalg.solve(J@J.T+lam**2*np.eye(3),gain*e_w)
        q_now=self.data.qpos[self.ARM_QPOS_ADR].copy()
        dq_ns=p.nullspace_K*(p.q_bias-q_now); dq=dq_ts+dq_ns
        step=np.clip(alpha_eff*dq,-p.max_delta,p.max_delta)
        q_des=q_now+step
        for i,a in enumerate(self.act_ids):
            if a>=0: self.data.ctrl[a]=self._clip(a,q_des[i])

# Permite que un ejecutor de evaluación inyecte parámetros sintonizados sin
# duplicar la implementación de ArmIK. None conserva los valores históricos.
IK_PARAMS_OVERRIDE: Optional[IKParams] = None

def _hold_grip_closed(model,data):
    hi=float(model.actuator_ctrlrange[grip_act_id,1]); data.ctrl[grip_act_id]=hi

def _open_grip(model,data):
    lo=float(model.actuator_ctrlrange[grip_act_id,0]); data.ctrl[grip_act_id]=lo

def _zero_arm_vel_acc(ik):
    dofs=ik.ARM_DOF_IDX; data.qvel[dofs]=0.0
    if hasattr(data,"qacc_warmstart") and data.qacc_warmstart is not None:
        data.qacc_warmstart[dofs]=0.0

def _hold_carry_posture(model,data):
    """Mantiene la pose de transporte CARRY_Q (= INIT_Q = pose home del entorno RL).
    Esto asegura que durante el transporte el brazo esté en la misma postura
    que ZennoV3 mantiene en su shaping de fase 1 (q_arm_carry_ref)."""
    _hold_grip_closed(model,data)
    for i,act in enumerate(arm_act_ids):
        lo,hi=model.actuator_ctrlrange[act]
        data.ctrl[act]=float(np.clip(CARRY_Q[i],lo,hi))

def drive_arm_to_home(model, data):
    """Manda comandos para llevar el brazo a INIT_Q. Abre el gripper."""
    _open_grip(model, data)
    for i, act in enumerate(arm_act_ids):
        lo, hi = model.actuator_ctrlrange[act]
        data.ctrl[act] = float(np.clip(INIT_Q[i], lo, hi))

def is_arm_at_home(data, qpos_adr, tol=RETURN_HOME_TOL):
    q_now = data.qpos[qpos_adr]
    return float(np.max(np.abs(q_now - INIT_Q))) < tol

# ======================= PICK / CARRY =======================
_arm_phase=0; _lock_xy=None; _cube_z0=None; _grasp_successful=False
_carry_done=False; _setpoint_stable_count=0; _grasp_settle_count=0
_post_close_settle_count=0; _carry_offset=None

def _set_home_pose(model,data,ik):
    global _cube_z0
    data.qpos[ik.ARM_QPOS_ADR]=INIT_Q.copy()
    mujoco.mj_forward(model,data)
    bid=_name2id(model,mujoco.mjtObj.mjOBJ_BODY,CUBE_BODY_NAME)
    _cube_z0=float(data.xpos[bid][2])

def cube_setpoint_dist():
    c=site_xpos(model,data,TARGET_SITE_NAME); s=site_xpos(model,data,SETPOINT_SITE_NAME)
    return float(np.linalg.norm(c-s))

def run_arm_pick_and_carry_to_setpoint(model,data,ik):
    global _arm_phase,_lock_xy,_cube_z0,_grasp_successful
    global _carry_done,_setpoint_stable_count,_carry_offset
    global _grasp_settle_count,_post_close_settle_count

    sid_tgt=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,TARGET_SITE_NAME)
    tgt_xyz=data.site_xpos[sid_tgt].copy()
    grasp_xyz=tgt_xyz.copy(); grasp_xyz[2]+=GRASP_OFFSET_Z
    sid_eef=_name2id(model,mujoco.mjtObj.mjOBJ_SITE,EEF_SITE_NAME)
    p_eef=data.site_xpos[sid_eef].copy()
    bid_cube=_name2id(model,mujoco.mjtObj.mjOBJ_BODY,CUBE_BODY_NAME)
    cube_z=float(data.xpos[bid_cube][2])

    if _arm_phase==0:
        _arm_phase=1; _grasp_settle_count=0; _post_close_settle_count=0
        _setpoint_stable_count=0; _carry_done=False; _carry_offset=None; return
    if _arm_phase==1:
        hover=np.array([grasp_xyz[0],grasp_xyz[1],grasp_xyz[2]+HOVER_UP],dtype=float)
        ik.step(hover)
        if np.linalg.norm(p_eef[:2]-grasp_xyz[:2])<XY_TOL_HOVER:
            _lock_xy=grasp_xyz[:2].copy(); _arm_phase=2
        return
    if _arm_phase==2:
        err_xy=np.linalg.norm(p_eef[:2]-grasp_xyz[:2])
        if err_xy>XY_FINE_TOL: ik.step(grasp_xyz)
        else: ik.step(np.array([_lock_xy[0],_lock_xy[1],grasp_xyz[2]]),lock_xy=_lock_xy,z_only=True)
        if np.linalg.norm(p_eef-grasp_xyz)<=GRASP_DIST_TOL:
            _arm_phase=21; _grasp_settle_count=GRASP_SETTLE_STEPS
        return
    if _arm_phase==21:
        ik.step(np.array([_lock_xy[0],_lock_xy[1],grasp_xyz[2]]),lock_xy=_lock_xy,z_only=True)
        _zero_arm_vel_acc(ik); _grasp_settle_count-=1
        if _grasp_settle_count<=0: _arm_phase=3
        return
    if _arm_phase==3:
        _hold_grip_closed(model,data); _zero_arm_vel_acc(ik)
        cube_pos=site_xpos(model,data,TARGET_SITE_NAME)
        eef_pos=site_xpos(model,data,EEF_SITE_NAME)
        _carry_offset=(eef_pos-cube_pos)
        _arm_phase=31; _post_close_settle_count=POST_CLOSE_SETTLE_STEPS; return
    if _arm_phase==31:
        _hold_grip_closed(model,data)
        ik.step(np.array([_lock_xy[0],_lock_xy[1],grasp_xyz[2]]),lock_xy=_lock_xy,z_only=True)
        _zero_arm_vel_acc(ik); _post_close_settle_count-=1
        if _post_close_settle_count<=0: _arm_phase=4
        return
    if _arm_phase==4:
        _hold_grip_closed(model,data)
        up=np.array([grasp_xyz[0],grasp_xyz[1],grasp_xyz[2]+LIFT_AFTER],dtype=float)
        ik.step(up,lock_xy=grasp_xyz[:2],z_only=True)
        if (_cube_z0 is not None) and ((cube_z-_cube_z0)>=(LIFT_AFTER-0.005)):
            _grasp_successful=True; _arm_phase=5
        return
    if _arm_phase==5:
        _hold_grip_closed(model,data)
        sp=site_xpos(model,data,SETPOINT_SITE_NAME)
        if _carry_offset is None:
            cube_pos=site_xpos(model,data,TARGET_SITE_NAME)
            eef_pos=site_xpos(model,data,EEF_SITE_NAME)
            _carry_offset=(eef_pos-cube_pos)
        desired_eef=sp+_carry_offset; ik.step(desired_eef)
        dcs=cube_setpoint_dist()
        if dcs<SETPOINT_DIST_TOL: _setpoint_stable_count+=1
        else: _setpoint_stable_count=0
        if _setpoint_stable_count>=SETPOINT_STABLE_STEPS:
            _carry_done=True; _arm_phase=6
            log(f"[ARM] Cubo en setpoint estable: dist={dcs:.4f} m")
        return
    if _arm_phase==6:
        _hold_carry_posture(model,data); return

# ======================= PLACE =======================
_place_phase=0; _place_done=False; _drop_latched=False; _drop_step_counter=0

def _reset_place_task():
    global _place_phase,_place_done,_drop_latched,_drop_step_counter
    _place_phase=0; _place_done=False; _drop_latched=False; _drop_step_counter=0

def run_arm_place_on_goal(model,data,ik):
    global _place_phase,_place_done,_drop_latched,_drop_step_counter
    sid_set=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,PEDESTAL2_SITE)
    if sid_set<0: return
    set_xyz=data.site_xpos[sid_set].copy()
    place_xyz=set_xyz.copy(); hover_xyz=place_xyz.copy(); hover_xyz[2]+=HOVER_UP
    sid_eef=_name2id(model,mujoco.mjtObj.mjOBJ_SITE,EEF_SITE_NAME)
    p_eef=data.site_xpos[sid_eef].copy()

    if _place_phase==0:
        _hold_grip_closed(model,data); ik.step(hover_xyz)
        if np.linalg.norm(p_eef[:2]-place_xyz[:2])<XY_TOL_HOVER: _place_phase=1
    elif _place_phase==1:
        _hold_grip_closed(model,data)
        ik.step(np.array([place_xyz[0],place_xyz[1],place_xyz[2]]),lock_xy=place_xyz[:2],z_only=True)
        if np.linalg.norm(p_eef-place_xyz)<=0.05: _place_phase=2
    elif _place_phase==2:
        _open_grip(model,data)
        if not _drop_latched:
            _drop_latched=True; _drop_step_counter=0
            log("[DROP] Cubo soltado.")
        _place_phase=3
    elif _place_phase==3:
        ik.step(hover_xyz,lock_xy=place_xyz[:2],z_only=True)
        _place_done=True

# ======================= RESET =======================
def _as_int(x): return int(x)

def reset_world_like_rl_for_map(map_id, cube_mass_kg=None):
    global waypoints,_wp_idx,_plan_ready,_reached_final,_final_aligning,_initial_aligning,_final_radial_approach
    global _reverse_active,_reverse_done
    global _arm_phase,_lock_xy,_cube_z0,_grasp_successful
    global _carry_done,_setpoint_stable_count,_carry_offset
    global _grasp_settle_count,_post_close_settle_count
    global _place_phase,_place_done
    global Accel_lin_des,Accel_ang_des,_e_vel1,_e_vel2,_e_omega1,_e_omega2,_v_des_f,_omega_des_f,_omega_des_prev
    global _last_plan,_drop_latched,_drop_step_counter
    global _terminal_route_stable_count

    mujoco.mj_resetData(model,data); data.ctrl[:]=0.0
    qpos=data.qpos.copy(); qvel=data.qvel.copy()
    set_obstacles_from_map(map_id); mujoco.mj_forward(model,data)

    if map_id not in INITIALS_BY_MAP:
        raise RuntimeError(f"Faltan INITIALS_BY_MAP para map_id={map_id}")

    base_qpos7=np.array(INITIALS_BY_MAP[map_id]["base_qpos7"],dtype=np.float64)
    qpos[:7]=base_qpos7

    cube_nominal_qpos7=np.array(INITIALS_BY_MAP[map_id]["cube_qpos7"],dtype=np.float64)
    noisy=cube_nominal_qpos7.copy()
    dx=float(_cube_noise_rng.uniform(-CUBE_XY_NOISE_RANGE,CUBE_XY_NOISE_RANGE))
    dy=float(_cube_noise_rng.uniform(-CUBE_XY_NOISE_RANGE,CUBE_XY_NOISE_RANGE))
    xy=np.array([noisy[0]+dx,noisy[1]+dy],dtype=np.float64)
    xy=clamp_cube_xy_to_pedestal(xy); noisy[0],noisy[1]=float(xy[0]),float(xy[1])
    yaw0=quat_wxyz_to_yaw(cube_nominal_qpos7[3:7])
    dyaw=float(_cube_noise_rng.uniform(-CUBE_YAW_NOISE_RANGE_RAD,CUBE_YAW_NOISE_RANGE_RAD))
    yawN=wrap_angle(yaw0+dyaw); noisy[3:7]=yaw_to_quat_wxyz(yawN)
    qpos[-7:]=noisy

    if cube_mass_kg is not None:
        model.body_mass[_CUBE_BID]=float(cube_mass_kg); mujoco.mj_forward(model,data)

    if map_id not in MAP_CACHE:
        MAP_CACHE[map_id]={"obstacles_xy":PREDEFINED_MAPS[map_id][:],"path":None,"plan_goal_xy":None,"plan_pregoal_xy":None}
    MAP_CACHE[map_id]["base_qpos7"]=base_qpos7.copy()
    MAP_CACHE[map_id]["cube_qpos7_nominal"]=cube_nominal_qpos7.copy()

    # ---- Pose inicial del brazo: misma que ZennoV3.init_arm_positions_stage0 ----
    # ZennoV3 usa: arm1=-0.90, arm2=-0.85, arm3=-1.5  (q_arm_init = q_arm_home = q_arm_carry_ref)
    init_arm_positions = {
        "robot0:base_joint":  0.0,
        "robot0:arm1_joint": -0.90,
        "robot0:arm2_joint": -0.85,
        "robot0:arm3_joint": -1.5,
        "robot0:eef_joint":   0.0,
    }
    for jname,jval in init_arm_positions.items():
        jadr=_as_int(model.joint(jname).qposadr[0]); qpos[jadr]=float(jval)

    data.qpos[:]=qpos; data.qvel[:]=qvel; data.ctrl[:]=0.0
    _open_grip(model,data); mujoco.mj_forward(model,data)

    waypoints[:]=[]
    _clear_frenet_path_cache()
    _wp_idx=0; _plan_ready=False; _reached_final=False; _initial_aligning=False
    _final_aligning=False; _final_radial_approach=False
    _reverse_active=False; _reverse_done=False
    _arm_phase=0; _lock_xy=None; _cube_z0=None; _grasp_successful=False
    _carry_done=False; _setpoint_stable_count=0; _carry_offset=None
    _grasp_settle_count=0; _post_close_settle_count=0
    _place_phase=0; _place_done=False; _drop_latched=False; _drop_step_counter=0
    _terminal_route_stable_count=0
    reset_pid_accums()
    _last_plan={"goal":{"path":None,"obstacles":None,"start":None,"pregoal":None,"orbit":None,"goal_center":None}}
    stop_wheels(); zero_base_qvel()

# ======================= ENERGÍA =======================
_prev_act_len = None

def actuator_power(act_id):
    """Potencia instantánea de un actuador (W). f * v."""
    f = float(data.actuator_force[act_id]) if act_id < data.actuator_force.shape[0] else 0.0
    if hasattr(data, "actuator_velocity") and data.actuator_velocity is not None:
        v = float(data.actuator_velocity[act_id])
        return f * v
    global _prev_act_len
    if _prev_act_len is None:
        _prev_act_len = data.actuator_length.copy()
        return 0.0
    v = float((data.actuator_length[act_id] - _prev_act_len[act_id]) / Ts)
    _prev_act_len[act_id] = data.actuator_length[act_id]
    return f * v

def compute_arm_base_power():
    """Devuelve (P_arm_W, P_base_W) en valor absoluto (energía consumida)."""
    p_base_abs = 0.0
    for a in wheel_ids_L + wheel_ids_R:
        p_base_abs += abs(actuator_power(a))
    p_arm_abs = 0.0
    for a in arm_act_ids + [grip_act_id]:
        p_arm_abs += abs(actuator_power(a))
    return p_arm_abs, p_base_abs

def point_to_polyline_dist(p, pts):
    if not pts: return 0.0
    if len(pts) == 1: return float(np.linalg.norm(p - np.array(pts[0], dtype=float)))
    best = 1e9; P = p.astype(float)
    for i in range(len(pts)-1):
        A = np.array(pts[i], dtype=float); B = np.array(pts[i+1], dtype=float)
        AB = B - A; denom = float(np.dot(AB, AB))
        if denom < 1e-12:
            d = float(np.linalg.norm(P - A))
        else:
            t = float(np.clip(np.dot(P - A, AB) / denom, 0.0, 1.0))
            Q = A + t * AB
            d = float(np.linalg.norm(P - Q))
        if d < best: best = d
    return float(best)


def tracking_error_distance_and_orientation(
    p_xy: np.ndarray,
    yaw_base: float,
    pts: List[Tuple[float, float]],
    *,
    start_idx: int = 0,
) -> Tuple[float, float, float, int, float, float, float, float]:
    """Error de seguimiento de la base respecto a la trayectoria.

    Retorna:
        dist_err:
            Distancia euclidiana [m] desde la base al segmento más cercano.
            Para una proyección sobre polilínea corresponde a abs(cte_signed).
        yaw_err_ctrl:
            Error angular firmado [rad] respecto a la referencia geométrica
            empleada por el controlador lateral.
        cte_signed:
            Error lateral firmado [m].
        seg_idx:
            Índice del segmento usado para la proyección.
        seg_yaw:
            Orientación tangencial del segmento proyectado [rad].
        theta_des_corr:
            Referencia geométrica del controlador [rad].
        theta_des_raw:
            Referencia geométrica antes de la corrección del baseline [rad].
        yaw_err_geom:
            Error angular auxiliar [rad] respecto a la tangente local:
                wrap(seg_yaw - yaw_base)
            Esta señal puede presentar serrucho por discretización de la polilínea,
            por lo que se usa solo como diagnóstico.
    """
    p = np.asarray(p_xy, dtype=np.float64).reshape(2)
    x, y = float(p[0]), float(p[1])

    cte_signed, _px, _py, seg_idx, seg_yaw = nearest_path_segment_projection(
        p,
        pts,
        start_idx=int(start_idx),
    )
    dist_err = float(abs(cte_signed))

    if TRACKING_CONTROLLER == "frenet_lqr" and pts is waypoints:
        (
            cte_signed, _px, _py, seg_idx,
            psi_ref, _kappa_ref, _remaining_s,
        ) = _frenet_reference_at_pose(p, start_idx=int(start_idx))
        theta_des_raw = float(psi_ref)
        theta_des_corr = float(psi_ref)
        seg_yaw = float(psi_ref)
    elif TRACKING_CONTROLLER == "pure_pursuit":
        # Pure Pursuit canonico: direccion geometrica al lookahead. El CTE ya
        # modifica esa direccion y no requiere una segunda correccion angular.
        gx, gy, _lookahead_seg_idx = _choose_lookahead_target(
            x,
            y,
            int(start_idx),
            LOOKAHEAD_DIST,
        )
        theta_des_raw = float(np.arctan2(float(gy) - y, float(gx) - x))
        theta_des_corr = float(theta_des_raw)
    else:
        # PID waypoint por waypoint: la referencia angular es la dirección
        # directa al waypoint activo, sin lookahead ni corrección Pure Pursuit.
        target_idx = min(max(int(_wp_idx), 0), len(pts) - 1)
        gx, gy = pts[target_idx]
        theta_des_raw = float(np.arctan2(float(gy) - y, float(gx) - x))
        theta_des_corr = float(theta_des_raw)

    # Error de control: referencia - medición. Este signo coincide con e_theta
    # dentro del controlador.
    yaw_err_ctrl = float(wrap_angle(theta_des_corr - float(yaw_base)))

    # Error geométrico auxiliar: tangente local - medición.
    # Puede tener saltos al cambiar de segmento de la polilínea.
    yaw_err_geom = float(wrap_angle(float(seg_yaw) - float(yaw_base)))

    return (
        dist_err,
        yaw_err_ctrl,
        float(cte_signed),
        int(seg_idx),
        float(seg_yaw),
        float(theta_des_corr),
        float(theta_des_raw),
        float(yaw_err_geom),
    )


def save_tracking_error_plot(
    *,
    ep_idx: int,
    map_id: int,
    cube_mass_kg: float,
    times_s: List[float],
    dist_err_m: List[float],
    yaw_err_rad: List[float],
    yaw_geom_err_rad: Optional[List[float]] = None,
    output_dir: Optional[str] = None,
) -> Optional[Path]:
    """Guarda una figura PNG con error de distancia y orientación por episodio."""
    if not bool(SAVE_TRACKING_ERROR_PLOTS):
        return None
    if len(times_s) < 2 or len(dist_err_m) < 2 or len(yaw_err_rad) < 2:
        return None
    if plt is None:
        print("[WARN][PLOT] matplotlib no está disponible; no se guardará la gráfica.")
        return None

    t = np.asarray(times_s, dtype=np.float64)
    e_d = np.asarray(dist_err_m, dtype=np.float64)
    e_yaw = np.asarray(yaw_err_rad, dtype=np.float64)

    e_yaw_geom = None
    if yaw_geom_err_rad is not None and len(yaw_geom_err_rad) == len(times_s):
        e_yaw_geom = np.asarray(yaw_geom_err_rad, dtype=np.float64)

    # Resolver el directorio al ejecutar (no al importar el módulo) permite
    # que PLAY_MM_VISUAL.py mantenga sus gráficas separadas.
    out_dir = Path(TRACKING_PLOTS_DIR if output_dir is None else output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mass_tag = f"{float(cube_mass_kg):.2f}".replace(".", "p")
    fig_path = out_dir / f"ep{int(ep_idx):03d}_map{int(map_id)}_mass{mass_tag}_tracking_errors.png"

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)

    axes[0].plot(t, e_d, linewidth=1.6)
    axes[0].set_ylabel("Error distancia [m]")
    axes[0].set_title(
        f"Error de seguimiento | EP {int(ep_idx)} | Mapa {int(map_id)} | masa {float(cube_mass_kg):.2f} kg"
    )
    axes[0].grid(True, alpha=0.35)

    yaw_label = (
        "error Frenet: $\\psi_{ref}-\\theta_{base}$"
        if TRACKING_CONTROLLER == "frenet_lqr"
        else "ref. baseline: $\\theta_{des,corr}-\\theta_{base}$"
    )
    axes[1].plot(t, e_yaw, linewidth=1.6, label=yaw_label)
    if bool(PLOT_GEOMETRIC_YAW_DIAGNOSTIC) and e_yaw_geom is not None:
        axes[1].plot(
            t,
            e_yaw_geom,
            linewidth=1.0,
            linestyle="--",
            alpha=0.65,
            label="diag. tangente local: $\\theta_{seg}-\\theta_{base}$",
        )
    axes[1].axhline(0.0, linewidth=0.8)
    axes[1].set_xlabel("Tiempo relativo de seguimiento [s]")
    axes[1].set_ylabel("Error orientación [rad]")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.35)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    return fig_path



# ======================= TIMING / DIAGNÓSTICO =======================
PHASE_TIMING_ORDER = (
    "ARM_PICK",
    "REVERSE",
    "PLAN_GOAL",
    "NAV_TO_GOAL",
    "ARM_PLACE",
    "DROP_SETTLE",
    "RETURN_HOME",
)

NAV_MODE_TIMING_ORDER = (
    "initial_align",
    "tracking",
    "final_align",
    "final_radial",
    "reverse",
    "idle",
)


def _empty_timing_counter(keys):
    return {str(k): 0 for k in keys}


def _timing_steps(counter: Dict[str, int], *names: str) -> int:
    return int(sum(int(counter.get(str(n), 0)) for n in names))


def _print_phase_timing_summary(
    *,
    phase_steps: Dict[str, int],
    nav_mode_steps: Dict[str, int],
    dt_control: float,
):
    """Imprime tiempos discretos por fase y por submodo de navegación."""
    print("     tiempos/steps por fase lógica:")
    p0_steps = _timing_steps(phase_steps, "ARM_PICK")
    p1_steps = _timing_steps(phase_steps, "REVERSE", "PLAN_GOAL", "NAV_TO_GOAL")
    p2_steps = _timing_steps(phase_steps, "ARM_PLACE", "DROP_SETTLE", "RETURN_HOME")

    print(f"       fase 0 pick/grasp       : {p0_steps:5d} steps | {p0_steps * dt_control:7.2f} s")
    print(f"       fase 1 transporte/base  : {p1_steps:5d} steps | {p1_steps * dt_control:7.2f} s")
    print(f"       fase 2 place/return     : {p2_steps:5d} steps | {p2_steps * dt_control:7.2f} s")

    print("     detalle por estado interno:")
    for name in PHASE_TIMING_ORDER:
        s = int(phase_steps.get(name, 0))
        if s > 0:
            print(f"       {name:<12}: {s:5d} steps | {s * dt_control:7.2f} s")

    nav_total = _timing_steps(nav_mode_steps, *NAV_MODE_TIMING_ORDER)
    if nav_total > 0:
        print("     detalle NAV_TO_GOAL por modo de control:")
        for name in NAV_MODE_TIMING_ORDER:
            s = int(nav_mode_steps.get(name, 0))
            if s > 0:
                pct = 100.0 * float(s) / float(max(nav_total, 1))
                print(f"       {name:<13}: {s:5d} steps | {s * dt_control:7.2f} s | {pct:5.1f}%")


# ======================= MAIN =======================
def main():
    global _drop_latched, _drop_step_counter, _plan_ready, _current_map_id
    EPISODE_RESULTS.clear()
    EPISODE_TRACES.clear()
    viewer = None
    if (not HEADLESS) and (mujoco_viewer is not None):
        viewer = mujoco_viewer.MujocoViewer(
            model=model,
            data=data,
            width=VIEWER_WIDTH,
            height=VIEWER_HEIGHT,
            hide_menus=VIEWER_HIDE_MENUS,
            title="Zenno — NEW MM VISUAL — PID sintonizado"
        )

    if LOAD_ROUTES_ON_START:
        load_map_routes_from_folder(MAPS_DIR, MAPS_LIST)

    if SAVE_TRACKING_ERROR_PLOTS:
        print(
            "[DEMO] Modo visual: se guardan gráficas PNG de tracking en "
            + str(TRACKING_PLOTS_DIR)
            + "."
        )
    else:
        print("[DEMO] Gráficas internas de tracking desactivadas por el ejecutor.")
    if LOAD_ROUTES_ON_START and not GENERATE_NEW_RRT_ROUTE_EACH_EPISODE:
        route_mode_label = "rutas_guardadas"
    elif GENERATE_NEW_RRT_ROUTE_EACH_EPISODE:
        route_mode_label = "ruta_nueva_cada_episodio"
    else:
        route_mode_label = "rutas_nuevas_por_mapa_optimizadas_PP"
    if TRACKING_CONTROLLER == "waypoint_pid":
        tracking_config = (
            f"wp_tol={WAYPOINT_PID_REACHED_TOL:.3f} m | "
            f"wp_v_max={WAYPOINT_PID_V_MAX:.2f} m/s | "
            f"wp_heading_gain={WAYPOINT_PID_OMEGA_GAIN:.2f} 1/s"
        )
    else:
        tracking_config = f"lookahead={LOOKAHEAD_DIST:.2f} m | V_MAX={V_MAX:.2f} m/s"
    print(
        f"[DEMO] PID={PID_PROFILE_NAME} | tracking={TRACKING_CONTROLLER} | "
        f"{tracking_config} | rutas={route_mode_label} | "
        f"tracking_phase_only={PLOT_ONLY_PURE_TRACKING}"
    )

    ep_idx = 0
    ik = ArmIK(
        model,
        data,
        JOINTS,
        EEF_SITE_NAME,
        ACTS_ARM,
        params=IK_PARAMS_OVERRIDE,
    )

    maps = [int(FIXED_MAP_ID)] if (FIXED_MAP_ID is not None) else MAPS_LIST[:]
    map_ep_counts = {mid: 0 for mid in maps}
    map_order_idx = 0
    _current_map_id = maps[map_order_idx]

    # ===== Buffers ligeros por episodio =====
    # tracking_errs: distancia base-trayectoria [m]
    # tracking_yaw_errs: tangente continua de ruta - yaw_base [rad]
    # tracking_guidance_errs: rumbo interno del controlador - yaw_base [rad]
    # tracking_yaw_geom_errs: theta_tangente_segmento - yaw_base [rad], solo diagnóstico
    # tracking_yaw_smooth_errs: tangente suavizada de ruta - yaw_base [rad]
    tracking_times_s: List[float] = []
    tracking_errs: List[float] = []
    tracking_yaw_errs: List[float] = []
    tracking_guidance_errs: List[float] = []
    tracking_yaw_geom_errs: List[float] = []
    tracking_yaw_smooth_errs: List[float] = []
    tracking_cte_signed: List[float] = []
    tracking_guidance_yaws: List[float] = []
    tracking_chassis_x: List[float] = []
    tracking_chassis_y: List[float] = []
    tracking_waypoint_ref_x: List[float] = []
    tracking_waypoint_ref_y: List[float] = []
    tracking_waypoint_yaws: List[float] = []
    tracking_base_yaws: List[float] = []
    tracking_path_yaws: List[float] = []
    tracking_omega_requested: List[float] = []
    tracking_omega_pid_refs: List[float] = []
    tracking_omega_measured: List[float] = []
    tracking_wheel_saturation: List[float] = []
    tracking_control_us: List[float] = []
    cube_sp_errs:  List[float] = []
    cube_sp_times_s: List[float] = []
    tracking_start_time_s: Optional[float] = None
    # Series comunes con DATA_SAC.py: una muestra por intervalo de control.
    energy_times_s: List[float] = []
    energy_phase: List[int] = []
    energy_total_steps_j: List[float] = []
    energy_arm_steps_j: List[float] = []
    energy_base_steps_j: List[float] = []
    power_total_w: List[float] = []
    trajectory_times_s: List[float] = []
    trajectory_base_x: List[float] = []
    trajectory_base_y: List[float] = []
    trajectory_cube_x: List[float] = []
    trajectory_cube_y: List[float] = []
    trajectory_cube_z: List[float] = []

    # ===== Acumuladores de episodio =====
    ep_decision_steps   = 0
    ep_timeout          = False
    total_e_arm         = 0.0
    total_e_base        = 0.0
    cube_mass_kg        = float(CUBE_MASS_BY_BLOCK_KG[0])
    drop_settle_counter = 0
    return_home_steps   = 0
    success_return_home = False

    # ===== Contadores discretos de tiempo por fase =====
    # Se acumulan en pasos de decisión de control; cada step equivale a DT_CONTROL.
    phase_steps = _empty_timing_counter(PHASE_TIMING_ORDER)
    nav_mode_steps = _empty_timing_counter(NAV_MODE_TIMING_ORDER)

    # success per phase (se asignan al cierre)
    def ep_state_init():
        return "RESET"

    ep_state = ep_state_init()

    # Para pacing en tiempo real. Se reinicia al inicio de cada episodio para
    # sincronizar tiempo simulado vs reloj de pared sin dormir cada 2 ms.
    episode_wall_t0 = time.perf_counter()

    try:
        while True:
            # ========================= RESET DE EPISODIO =========================
            if ep_state == "RESET":
                ep_idx += 1
                ep_decision_steps = 0
                ep_timeout = False
                total_e_arm = 0.0
                total_e_base = 0.0
                drop_settle_counter = 0
                return_home_steps = 0
                success_return_home = False
                phase_steps = _empty_timing_counter(PHASE_TIMING_ORDER)
                nav_mode_steps = _empty_timing_counter(NAV_MODE_TIMING_ORDER)

                tracking_times_s.clear()
                tracking_errs.clear()
                tracking_yaw_errs.clear()
                tracking_guidance_errs.clear()
                tracking_yaw_geom_errs.clear()
                tracking_yaw_smooth_errs.clear()
                tracking_cte_signed.clear()
                tracking_guidance_yaws.clear()
                tracking_chassis_x.clear()
                tracking_chassis_y.clear()
                tracking_waypoint_ref_x.clear()
                tracking_waypoint_ref_y.clear()
                tracking_waypoint_yaws.clear()
                tracking_base_yaws.clear()
                tracking_path_yaws.clear()
                tracking_omega_requested.clear()
                tracking_omega_pid_refs.clear()
                tracking_omega_measured.clear()
                tracking_wheel_saturation.clear()
                tracking_control_us.clear()
                tracking_start_time_s = None
                cube_sp_errs.clear()
                cube_sp_times_s.clear()
                energy_times_s.clear()
                energy_phase.clear()
                energy_total_steps_j.clear()
                energy_arm_steps_j.clear()
                energy_base_steps_j.clear()
                power_total_w.clear()
                trajectory_times_s.clear()
                trajectory_base_x.clear()
                trajectory_base_y.clear()
                trajectory_cube_x.clear()
                trajectory_cube_y.clear()
                trajectory_cube_z.clear()

                # Seleccionar mapa actual
                while (map_order_idx < len(maps)) and (
                        map_ep_counts.get(maps[map_order_idx], 0) >= EPISODES_PER_MAP):
                    map_order_idx += 1
                if map_order_idx >= len(maps):
                    print("\n================= EVALUACIÓN COMPLETA =================")
                    break

                _current_map_id = maps[map_order_idx]
                map_ep_idx_next = map_ep_counts.get(_current_map_id, 0) + 1
                block_idx = (map_ep_idx_next - 1) // BLOCK_EPISODES
                cube_mass_kg = float(CUBE_MASS_BY_BLOCK_KG[block_idx % len(CUBE_MASS_BY_BLOCK_KG)])

                reset_world_like_rl_for_map(_current_map_id, cube_mass_kg=cube_mass_kg)
                _set_home_pose(model, data, ik)

                ep_state = "ARM_PICK"
                # La planificación/carga entre episodios no debe contar como atraso visual.
                episode_wall_t0 = time.perf_counter()
                print(f"\n=== EP {ep_idx} | MAPA {_current_map_id} | "
                      f"ep_mapa {map_ep_idx_next}/{EPISODES_PER_MAP} | m={cube_mass_kg:.2f} kg ===")

            # ========================= COMPUTAR CONTROL =========================
            # (1) BASE
            if ep_state in ("REVERSE", "NAV_TO_GOAL"):
                ctrl_t0_ns = time.perf_counter_ns()
                control_base_step_nav_to_goal()
                ctrl_elapsed_us = (time.perf_counter_ns() - ctrl_t0_ns) * 1e-3
                if ep_state == "NAV_TO_GOAL" and _base_control_mode == "tracking":
                    tracking_control_us.append(float(ctrl_elapsed_us))
            else:
                stop_wheels()

            # (2) BRAZO durante transporte (mantener cubo)
            if ep_state in ("ARM_PICK", "REVERSE", "PLAN_GOAL", "NAV_TO_GOAL") and (_grasp_successful or _carry_done):
                if _carry_done:
                    _hold_carry_posture(model, data)
                else:
                    _hold_grip_closed(model, data)

            # (3) MÁQUINA DE ESTADOS
            if ep_state == "ARM_PICK":
                run_arm_pick_and_carry_to_setpoint(model, data, ik)
                if _carry_done:
                    ep_state = "REVERSE"
                    start_reverse(distance_m=0.35)

            elif ep_state == "REVERSE":
                if _reverse_done:
                    ep_state = "PLAN_GOAL"

            elif ep_state == "PLAN_GOAL":
                _reset_place_task()
                ok = replan_path_to_goal_for_current_map()
                if not ok:
                    # No se pudo planear: terminar como fallo
                    ep_state = "DONE"
                else:
                    stop_wheels(); reset_pid_accums(); zero_base_qvel()
                    ep_state = "NAV_TO_GOAL"

            elif ep_state == "NAV_TO_GOAL":
                if _reached_final:
                    ep_state = "ARM_PLACE"

            elif ep_state == "ARM_PLACE":
                run_arm_place_on_goal(model, data, ik)
                if _place_done and _drop_latched:
                    ep_state = "DROP_SETTLE"
                    drop_settle_counter = 0

            elif ep_state == "DROP_SETTLE":
                # Mantener postura mientras el cubo asienta
                _hold_carry_posture(model, data)  # con gripper abierto sigue abierto
                _open_grip(model, data)
                drop_settle_counter += 1
                if drop_settle_counter >= DROP_SETTLE_CONTROL_STEPS:
                    ep_state = "RETURN_HOME"
                    return_home_steps = 0

            elif ep_state == "RETURN_HOME":
                drive_arm_to_home(model, data)
                return_home_steps += 1
                if is_arm_at_home(data, ik.ARM_QPOS_ADR, tol=RETURN_HOME_TOL):
                    success_return_home = True
                    ep_state = "DONE"
                elif return_home_steps >= MAX_RETURN_HOME_CONTROL_STEPS:
                    success_return_home = False
                    ep_state = "DONE"

            elif ep_state == "DONE":
                # ===== CALCULAR MÉTRICAS (solo para imprimir en consola) =====
                total_time_s = ep_decision_steps * DT_CONTROL

                if tracking_errs:
                    errs = np.array(tracking_errs, dtype=float)
                    track_rms  = float(np.sqrt(np.mean(np.square(errs))))
                    track_mean = float(np.mean(errs))
                    track_max  = float(np.max(errs))
                    track_p95  = float(np.percentile(errs, 95.0))
                else:
                    track_rms = track_mean = track_max = track_p95 = 0.0

                if tracking_control_us:
                    ctrl_p99_us = float(np.percentile(np.asarray(tracking_control_us), 99.0))
                else:
                    ctrl_p99_us = 0.0

                if tracking_yaw_errs:
                    yaw_errs = np.array(tracking_yaw_errs, dtype=float)
                    track_yaw_rms = float(np.sqrt(np.mean(np.square(yaw_errs))))
                    track_yaw_mean_abs = float(np.mean(np.abs(yaw_errs)))
                    track_yaw_max_abs = float(np.max(np.abs(yaw_errs)))
                else:
                    track_yaw_rms = track_yaw_mean_abs = track_yaw_max_abs = 0.0

                if cube_sp_errs:
                    errs2 = np.array(cube_sp_errs, dtype=float)
                    cube_sp_rms = float(np.sqrt(np.mean(np.square(errs2))))
                    cube_sp_max = float(np.max(errs2))
                else:
                    cube_sp_rms = cube_sp_max = 0.0

                # Distancia final cubo↔goal en XY
                cube_xyz = site_xpos(model, data, TARGET_SITE_NAME)
                goal_xyz = site_xpos(model, data, PEDESTAL2_SITE)
                cube_goal_dist_xy = float(np.linalg.norm(cube_xyz[:2] - goal_xyz[:2]))

                # Flags de éxito por fase
                success_pick      = bool(_grasp_successful and _carry_done)
                success_transport = bool(success_pick and _reached_final)
                success_place     = bool(_place_done and (cube_goal_dist_xy <= GOAL_SUCCESS_DIST_TOL))
                success_overall   = bool(
                    (not ep_timeout) and success_pick and success_transport
                    and success_place and success_return_home
                )

                total_energy = total_e_arm + total_e_base

                base_x, base_y = base_xy(model, data)
                goal_x, goal_y = pedestal2_xy(model, data)
                base_goal_distance_m = float(np.hypot(goal_x - base_x, goal_y - base_y))
                goal_heading_rad = float(np.arctan2(goal_y - base_y, goal_x - base_x))
                base_goal_heading_error_rad = abs(
                    float(wrap_angle(goal_heading_rad - base_yaw_rad(model, data)))
                )

                episode_result = {
                    "episode": int(ep_idx),
                    "map_id": int(_current_map_id),
                    "tracking_controller": str(TRACKING_CONTROLLER),
                    "cube_mass_kg": float(cube_mass_kg),
                    "success_overall": bool(success_overall),
                    "success_pick": bool(success_pick),
                    "success_transport": bool(success_transport),
                    "success_place": bool(success_place),
                    "success_return_home": bool(success_return_home),
                    "timeout": bool(ep_timeout),
                    "total_time_s": float(total_time_s),
                    "energy_total_j": float(total_energy),
                    "energy_arm_j": float(total_e_arm),
                    "energy_base_j": float(total_e_base),
                    "cube_goal_distance_xy_m": float(cube_goal_dist_xy),
                    "base_goal_distance_m": float(base_goal_distance_m),
                    "stop_distance_error_m": float(
                        abs(base_goal_distance_m - FINAL_GOAL_RADIUS_STOP)
                    ),
                    "base_goal_heading_error_rad": float(base_goal_heading_error_rad),
                    "tracking_samples": int(len(tracking_errs)),
                    "tracking_rms_m": float(track_rms),
                    "tracking_mean_m": float(track_mean),
                    "tracking_p95_m": float(track_p95),
                    "tracking_max_m": float(track_max),
                    "tracking_yaw_rms_rad": float(track_yaw_rms),
                    "tracking_yaw_mean_abs_rad": float(track_yaw_mean_abs),
                    "tracking_yaw_max_abs_rad": float(track_yaw_max_abs),
                    "cube_setpoint_rms_m": float(cube_sp_rms),
                    "cube_setpoint_max_m": float(cube_sp_max),
                    "controller_compute_p99_us": float(ctrl_p99_us),
                    "phase_steps": {k: int(v) for k, v in phase_steps.items()},
                    "nav_mode_steps": {k: int(v) for k, v in nav_mode_steps.items()},
                    "final_alignment_time_s": float(
                        nav_mode_steps.get("final_align", 0) * DT_CONTROL
                    ),
                    "final_radial_time_s": float(
                        nav_mode_steps.get("final_radial", 0) * DT_CONTROL
                    ),
                }
                EPISODE_RESULTS.append(episode_result)
                EPISODE_TRACES.append({
                    "trace_schema_version": 7,
                    "episode": int(ep_idx),
                    "map_id": int(_current_map_id),
                    "cube_mass_kg": float(cube_mass_kg),
                    "tracking_time_s": [float(v) for v in tracking_times_s],
                    "tracking_error_m": [float(v) for v in tracking_errs],
                    "signed_tracking_error_m": [
                        float(v) for v in tracking_cte_signed
                    ],
                    "chassis_x_m": [float(v) for v in tracking_chassis_x],
                    "chassis_y_m": [float(v) for v in tracking_chassis_y],
                    "waypoint_reference_x_m": [
                        float(v) for v in tracking_waypoint_ref_x
                    ],
                    "waypoint_reference_y_m": [
                        float(v) for v in tracking_waypoint_ref_y
                    ],
                    "waypoint_heading_rad": [
                        float(v) for v in tracking_waypoint_yaws
                    ],
                    "waypoint_orientation_error_rad": [
                        float(v) for v in tracking_yaw_geom_errs
                    ],
                    "orientation_error_rad": [float(v) for v in tracking_yaw_errs],
                    "path_heading_error_rad": [float(v) for v in tracking_yaw_errs],
                    "path_heading_rad": [float(v) for v in tracking_path_yaws],
                    "guidance_orientation_error_rad": [
                        float(v) for v in tracking_guidance_errs
                    ],
                    "guidance_heading_rad": [
                        float(v) for v in tracking_guidance_yaws
                    ],
                    "geometric_orientation_error_rad": [
                        float(v) for v in tracking_yaw_geom_errs
                    ],
                    "smooth_route_orientation_error_rad": [
                        float(v) for v in tracking_yaw_smooth_errs
                    ],
                    "base_yaw_rad": [float(v) for v in tracking_base_yaws],
                    "smooth_path_yaw_rad": [float(v) for v in tracking_path_yaws],
                    "omega_requested_rad_s": [
                        float(v) for v in tracking_omega_requested
                    ],
                    "omega_pid_reference_rad_s": [
                        float(v) for v in tracking_omega_pid_refs
                    ],
                    "omega_measured_rad_s": [
                        float(v) for v in tracking_omega_measured
                    ],
                    "wheel_saturation_fraction": [
                        float(v) for v in tracking_wheel_saturation
                    ],
                    "energy_time_s": [float(v) for v in energy_times_s],
                    "energy_phase": [int(v) for v in energy_phase],
                    "energy_total_step_j": [float(v) for v in energy_total_steps_j],
                    "energy_arm_step_j": [float(v) for v in energy_arm_steps_j],
                    "energy_base_step_j": [float(v) for v in energy_base_steps_j],
                    "power_total_w": [float(v) for v in power_total_w],
                    "trajectory_time_s": [float(v) for v in trajectory_times_s],
                    "base_x_m": [float(v) for v in trajectory_base_x],
                    "base_y_m": [float(v) for v in trajectory_base_y],
                    "cube_x_m": [float(v) for v in trajectory_cube_x],
                    "cube_y_m": [float(v) for v in trajectory_cube_y],
                    "cube_z_m": [float(v) for v in trajectory_cube_z],
                    "planned_path_xy": [
                        [float(point[0]), float(point[1])] for point in waypoints
                    ],
                    "cube_setpoint_time_s": [float(v) for v in cube_sp_times_s],
                    "cube_setpoint_error_m": [float(v) for v in cube_sp_errs],
                    "tracking_controller": str(TRACKING_CONTROLLER),
                })

                # ===== IMPRIMIR (no se guarda nada en disco) =====
                print(f"  → success_overall={int(success_overall)} "
                      f"[pick={int(success_pick)}, transport={int(success_transport)}, "
                      f"place={int(success_place)}, return_home={int(success_return_home)}]")
                print(f"     time={total_time_s:.2f}s | E_total={total_energy:.2f} J "
                      f"(arm={total_e_arm:.2f} J | base={total_e_base:.2f} J)")
                print(f"     dist_cubo-goal_XY={cube_goal_dist_xy:.4f} m | "
                      f"track_rms={track_rms:.4f} m | "
                      f"track_p95={track_p95:.4f} m | "
                      f"track_yaw_rms={track_yaw_rms:.4f} rad | "
                      f"cube_sp_rms={cube_sp_rms:.4f} m")
                print(f"     controller_compute_p99={ctrl_p99_us:.1f} us")

                _print_phase_timing_summary(
                    phase_steps=phase_steps,
                    nav_mode_steps=nav_mode_steps,
                    dt_control=DT_CONTROL,
                )

                plot_path = save_tracking_error_plot(
                    ep_idx=ep_idx,
                    map_id=_current_map_id,
                    cube_mass_kg=cube_mass_kg,
                    times_s=tracking_times_s,
                    dist_err_m=tracking_errs,
                    yaw_err_rad=tracking_yaw_errs,
                    yaw_geom_err_rad=tracking_yaw_geom_errs,
                )
                if plot_path is not None:
                    print(f"     grafica_tracking={plot_path}")

                # ===== CONTADORES =====
                map_ep_counts[_current_map_id] = map_ep_counts.get(_current_map_id, 0) + 1

                # Auto-stop
                if AUTO_STOP_AFTER_EVAL and all(
                        map_ep_counts.get(mid, 0) >= EPISODES_PER_MAP for mid in maps):
                    print("\n================= DEMO COMPLETA =================")
                    break

                ep_state = "RESET"
                continue  # saltar el bloque de step para reiniciar limpiamente

            # Si la máquina de estados terminó el episodio en este ciclo, no avanzamos
            # una muestra física extra. En la siguiente iteración se imprime el resumen.
            if ep_state == "DONE":
                continue

            # ========================= STEP DE SIMULACIÓN =========================
            # El controlador ya calculó una acción para esta decisión.
            # La acción se mantiene durante FRAMESKIP_CONTROL substeps físicos de MuJoCo.
            energy_arm_step = 0.0
            energy_base_step = 0.0

            for _ in range(FRAMESKIP_CONTROL):
                mujoco.mj_step(model, data)

                p_arm_now, p_base_now = compute_arm_base_power()
                energy_arm_step += p_arm_now * Ts
                energy_base_step += p_base_now * Ts

            ep_decision_steps += 1
            t_now = ep_decision_steps * DT_CONTROL

            # ===== TIMING POR FASE =====
            # Se cuenta la fase activa durante este paso de control.
            active_phase_for_timing = str(ep_state)
            if active_phase_for_timing in phase_steps:
                phase_steps[active_phase_for_timing] += 1

            if active_phase_for_timing == "NAV_TO_GOAL":
                mode_name = str(_base_control_mode)
                if mode_name not in nav_mode_steps:
                    nav_mode_steps[mode_name] = 0
                nav_mode_steps[mode_name] += 1

            # Render desacoplado del control.
            # El control corre a 100 Hz si Ts = 0.002 s y FRAMESKIP_CONTROL = 5.
            # El viewer se actualiza cada RENDER_EVERY decisiones de control.
            rendered_this_step = False
            if viewer is not None and (ep_decision_steps % RENDER_EVERY == 0):
                viewer.render()
                rendered_this_step = True

            # Acumular energía total de episodio (solo para imprimir al final)
            total_e_arm  += energy_arm_step
            total_e_base += energy_base_step

            # Persistencia por intervalo para comparación directa con SAC.
            if active_phase_for_timing == "ARM_PICK":
                phase_id = 0
            elif active_phase_for_timing in ("REVERSE", "PLAN_GOAL", "NAV_TO_GOAL"):
                phase_id = 1
            else:
                phase_id = 2
            bx_sample, by_sample = base_xy(model, data)
            cube_sample = site_xpos(model, data, TARGET_SITE_NAME)
            energy_times_s.append(float(t_now))
            energy_phase.append(int(phase_id))
            energy_total_steps_j.append(float(energy_arm_step + energy_base_step))
            energy_arm_steps_j.append(float(energy_arm_step))
            energy_base_steps_j.append(float(energy_base_step))
            power_total_w.append(float((energy_arm_step + energy_base_step) / DT_CONTROL))
            trajectory_times_s.append(float(t_now))
            trajectory_base_x.append(float(bx_sample))
            trajectory_base_y.append(float(by_sample))
            trajectory_cube_x.append(float(cube_sample[0]))
            trajectory_cube_y.append(float(cube_sample[1]))
            trajectory_cube_z.append(float(cube_sample[2]))

            # ===== PACING EN TIEMPO REAL =====
            # En modo visual se sincroniza solo cuando se renderiza un frame.
            # Con frame_skip=5, DT_CONTROL≈0.010 s.
            if REAL_TIME and (rendered_this_step or viewer is None):
                sim_elapsed = ep_decision_steps * DT_CONTROL
                wall_elapsed = time.perf_counter() - episode_wall_t0
                sleep_s = sim_elapsed - wall_elapsed
                if sleep_s > 0.0:
                    time.sleep(sleep_s)

            # Timeout
            if ep_decision_steps >= MAX_DECISION_STEPS and ep_state not in ("DONE", "RESET"):
                ep_timeout = True
                print(f"  [TIMEOUT] decision_steps={ep_decision_steps}")
                ep_state = "DONE"
                continue

            # ===== POSICIÓN BASE (para el cálculo de tracking error) =====
            bx, by = base_xy(model, data)

            # ===== TRACKING ERROR (solo durante NAV_TO_GOAL, post-reversa) =====
            if (
                ep_state == "NAV_TO_GOAL"
                and _plan_ready
                and _last_tracking_active
                and len(waypoints) >= 2
                and ((not PLOT_ONLY_PURE_TRACKING) or (_base_control_mode == "tracking"))
            ):
                # Pose medida DESPUES de los cinco substeps. Posicion, yaw,
                # proyeccion y errores quedan sincronizados en el mismo instante
                # fisico y se obtienen del site real ``chassis_frame``.
                chassis_yaw = base_yaw_rad(model, data)
                # Error de orientacion de guiado: referencia que genero la
                # accion menos yaw real medido tras los substeps.
                e_yaw_ctrl = wrap_angle(
                    float(_last_theta_des_corr) - float(chassis_yaw)
                )
                _cte, _px, _py, _seg_idx, _seg_yaw = nearest_path_segment_projection(
                    np.array([bx, by], dtype=np.float64),
                    waypoints,
                    # Evaluacion geometrica independiente del estado interno
                    # del controlador. Una ventana anclada a _wp_idx producia
                    # saltos artificiales cada vez que avanzaba el waypoint.
                    start_idx=0,
                    search_back=0,
                    search_ahead=max(len(waypoints), 1),
                )
                # Error lateral segun la definicion experimental: distancia
                # perpendicular firmada al segmento correspondiente.
                e_dist = float(abs(_cte))
                e_yaw_geom = wrap_angle(_seg_yaw - chassis_yaw)
                (
                    _cte_path, _px_path, _py_path, _seg_idx_path, psi_path,
                ) = _continuous_path_reference_at_pose(
                    np.array([bx, by], dtype=np.float64),
                    start_idx=max(int(_last_tracking_wp_idx) - 1, 0),
                    global_search=True,
                )
                e_yaw_smooth = wrap_angle(float(psi_path) - chassis_yaw)
                wheel_ctrl_abs = np.abs(np.asarray([
                    float(data.ctrl[a]) for a in (wheel_ids_L + wheel_ids_R)
                ], dtype=np.float64))
                wheel_limits = np.concatenate((tau_max_L, tau_max_R))
                wheel_sat_fraction = float(np.mean(
                    wheel_ctrl_abs >= 0.995 * np.maximum(wheel_limits, 1e-9)
                ))
                if tracking_start_time_s is None:
                    tracking_start_time_s = float(t_now)
                tracking_times_s.append(float(t_now - tracking_start_time_s))
                tracking_errs.append(float(e_dist))
                tracking_yaw_errs.append(float(e_yaw_smooth))
                tracking_guidance_errs.append(float(e_yaw_ctrl))
                tracking_guidance_yaws.append(float(_last_theta_des_corr))
                tracking_yaw_geom_errs.append(float(e_yaw_geom))
                tracking_yaw_smooth_errs.append(float(e_yaw_smooth))
                tracking_cte_signed.append(float(_cte))
                tracking_chassis_x.append(float(bx))
                tracking_chassis_y.append(float(by))
                tracking_waypoint_ref_x.append(float(_px))
                tracking_waypoint_ref_y.append(float(_py))
                tracking_waypoint_yaws.append(float(_seg_yaw))
                tracking_base_yaws.append(float(chassis_yaw))
                tracking_path_yaws.append(float(psi_path))
                tracking_omega_requested.append(float(_base_omega_des_cmd))
                tracking_omega_pid_refs.append(float(_omega_des_f))
                tracking_omega_measured.append(float(data.qvel[5]))
                tracking_wheel_saturation.append(float(wheel_sat_fraction))

            # ===== ERROR CUBO ↔ SETPOINT durante transporte =====
            if _carry_done and (ep_state in ("REVERSE", "PLAN_GOAL", "NAV_TO_GOAL")) \
               and (not _reached_final) and (not _drop_latched):
                cube_sp_times_s.append(float(t_now))
                cube_sp_errs.append(cube_setpoint_dist())

    finally:
        if viewer is not None:
            viewer.close()
        if SAVE_TRACKING_ERROR_PLOTS:
            print(
                "\n[DEMO] Sesión terminada. Las gráficas PNG quedaron en: "
                + str(TRACKING_PLOTS_DIR)
            )
        else:
            print("\n[DEMO] Sesión terminada.")

import math
from typing import Sequence
mm = sys.modules[__name__]
PID_PROFILE = "PID_PSO_6D_GAINS_REUSED_STOP_0p70"
PID_GAINS_TUNED = {
    "kp_vel": 25.763229287504284,
    "ki_vel": 3.0318334011181163,
    "kd_vel": 0.04929680970337456,
    "kp_omega": 1149.8272721326991,
    "ki_omega": 36.98551603282882,
    "kd_omega": 0.11001842680805748,
}
LOOKAHEAD_M = 0.30
STOP_DISTANCE_M = 0.70
STOP_TOLERANCE_M = 0.005

# ============================================================================
# IK DLS SINTONIZADA POR PSO PARA dt_control=10 ms (frameskip=5, Ts=2 ms)
# ============================================================================
IK_PROFILE = "IK_DLS_PSO_10MS"
IK_PARAMS_TUNED = {
    "weight_xy": 3.0,
    "weight_z": 1.0,
    "K_max": 14.0,
    "K_min": 2.7759411546936597,
    "distance_threshold": 0.01,
    "alpha": 0.0970902810377246,
    "max_delta": 0.2155388636078064,
    "nullspace_K": 0.02,
    "q_bias": [0.0, -1.0, -1.0, -1.4, 0.0],
    "z_only_xy_weight": 0.05,
    "z_only_z_weight": 6.0,
    "dls_lambda0": 0.02598053858310157,
    "dls_lambda_max": 0.08,
    "dls_gain_scale": 17.94838795314729,
    "d_slow": 0.05,
    "alpha_min": 0.4921464755152155,
    "z_err_cap_down": 0.008,
    "z_err_cap_up": 0.02,
}
IK_SETPOINT_TOLERANCE_M = 0.025
IK_SETPOINT_STABLE_STEPS = 25


def configure_tuned_controllers() -> None:
    if not math.isclose(float(mm.DT_CONTROL), 0.010, abs_tol=1e-12):
        raise RuntimeError(
            f"Los parámetros fueron sintonizados para dt_control=0.010 s; "
            f"el modelo actual usa {mm.DT_CONTROL:.6f} s."
        )

    mm.PID_PROFILE_NAME = PID_PROFILE
    mm.Kp_vel = PID_GAINS_TUNED["kp_vel"]
    mm.Ki_vel = PID_GAINS_TUNED["ki_vel"]
    mm.Kd_vel = PID_GAINS_TUNED["kd_vel"]
    mm.Kp_omega = PID_GAINS_TUNED["kp_omega"]
    mm.Ki_omega = PID_GAINS_TUNED["ki_omega"]
    mm.Kd_omega = PID_GAINS_TUNED["kd_omega"]
    mm.LOOKAHEAD_DIST = LOOKAHEAD_M
    mm.MODEL_YAW_INERTIA_SCALE = 1.0
    mm.J = float(mm.MODEL_BASE_YAW_INERTIA_KGM2 * mm.MODEL_YAW_INERTIA_SCALE)
    mm.FINAL_WAYPOINT_TOL = 0.020
    mm.FINAL_ROUTE_HEADING_TOL = float(np.deg2rad(3.0))
    mm.approach_dist = STOP_DISTANCE_M
    mm.FINAL_GOAL_RADIUS_STOP = STOP_DISTANCE_M
    mm.FINAL_RADIUS_TOL = STOP_TOLERANCE_M

    ik_values = dict(IK_PARAMS_TUNED)
    ik_values["q_bias"] = np.asarray(ik_values["q_bias"], dtype=float)
    mm.IK_PARAMS_OVERRIDE = mm.IKParams(**ik_values)
    mm.SETPOINT_DIST_TOL = IK_SETPOINT_TOLERANCE_M
    mm.SETPOINT_STABLE_STEPS = IK_SETPOINT_STABLE_STEPS

RUTAS_SAC = RESOURCE_ROOT / "RUTAS_SAC"
V9_STOP_DISTANCE_M = .70
V9_LOOKAHEAD_M = .30
def load_and_validate_v9_routes(
    routes_dir: Path, maps: Sequence[int]
) -> dict[int, dict]:
    """Carga los JSON RRT* V9 y comprueba su parada geométrica a 0,70 m."""
    routes_dir = routes_dir.expanduser().resolve()
    expected_root = RUTAS_SAC.resolve()
    if routes_dir != expected_root:
        raise ValueError(
            f"V9 debe usar exclusivamente RUTAS_SAC: {expected_root}; "
            f"recibido={routes_dir}"
        )

    payloads: dict[int, dict] = {}
    for map_id in maps:
        route_file = routes_dir / f"map_{int(map_id)}.json"
        if not route_file.is_file():
            raise FileNotFoundError(f"Falta la ruta RRT* V9 del mapa {map_id}: {route_file}")

        payload = json.loads(route_file.read_text(encoding="utf-8"))
        if int(payload.get("map_id", -1)) != int(map_id):
            raise ValueError(f"map_id incompatible en {route_file}")

        arrays = {
            "path": (None, 2),
            "base_qpos7": (7,),
            "cube_qpos7_nominal": (7,),
            "obstacles_xy": (None, 2),
            "goal_center_xy": (2,),
        }
        parsed: dict[str, np.ndarray] = {}
        for key, shape in arrays.items():
            value = np.asarray(payload.get(key, []), dtype=np.float64)
            valid = (
                value.ndim == 2
                and value.shape[1] == shape[1]
                and value.shape[0] >= 2
                if shape[0] is None
                else value.shape == shape
            )
            if not valid or not np.all(np.isfinite(value)):
                raise ValueError(f"{key} inválido en {route_file}: shape={value.shape}")
            parsed[key] = value

        if payload.get("route_type") != "direct_goal_stop":
            raise ValueError(
                f"route_type incompatible en {route_file}: {payload.get('route_type')!r}"
            )
        for key in ("stop_radius", "stop_distance_m"):
            value = float(payload.get(key, np.nan))
            if not math.isclose(value, V9_STOP_DISTANCE_M, abs_tol=1e-9):
                raise ValueError(
                    f"{key} incompatible en {route_file}: {value}; "
                    f"esperado={V9_STOP_DISTANCE_M}"
                )
        lookahead = float(payload.get("lookahead_dist", np.nan))
        if not math.isclose(lookahead, V9_LOOKAHEAD_M, abs_tol=1e-9):
            raise ValueError(
                f"lookahead_dist incompatible en {route_file}: {lookahead}"
            )

        endpoint_distance = float(
            np.linalg.norm(parsed["path"][-1] - parsed["goal_center_xy"])
        )
        if not math.isclose(endpoint_distance, V9_STOP_DISTANCE_M, abs_tol=1e-6):
            raise ValueError(
                f"El último waypoint de {route_file} queda a {endpoint_distance:.9f} m "
                f"del goal; se requieren {V9_STOP_DISTANCE_M:.2f} m"
            )
        payloads[int(map_id)] = payload
    return payloads

def validate_routes(routes_dir, maps, **kwargs):
    payloads = load_and_validate_v9_routes(Path(routes_dir), maps)
    for map_id, payload in payloads.items():
        INITIALS_BY_MAP[map_id] = {
            'base_qpos7': np.asarray(payload['base_qpos7'], dtype=np.float64),
            'cube_qpos7': np.asarray(payload['cube_qpos7_nominal'], dtype=np.float64)}
        PREDEFINED_MAPS[map_id] = [tuple(map(float, xy)) for xy in payload['obstacles_xy']]

def run_simulation():
    global MAPS_DIR, HEADLESS, REAL_TIME, AUTO_STOP_AFTER_EVAL
    configure_tuned_controllers()
    validate_routes(RUTAS_SAC, [1, 2, 3, 4])
    MAPS_DIR = str(RUTAS_SAC)
    HEADLESS = '--viewer' not in sys.argv
    REAL_TIME = not HEADLESS
    AUTO_STOP_AFTER_EVAL = True
    main()

if __name__ == '__main__':
    run_simulation()
