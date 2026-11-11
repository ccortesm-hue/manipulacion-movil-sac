#!/usr/bin/env python3
"""Profesor analítico end-to-end para ZennoVXR1.

No carga políticas, normalizadores ni bancos de estados aprendidos. Produce
exclusivamente acciones V9 incrementales:

* P0: IK cartesiana para aproximación, agarre automático y elevación.
* P1: Pure Pursuit + PID internos del ambiente; residual docente igual a cero.
* P2: IK cartesiana por etapas para alinear, descender, soltar y volver a home.

El profesor consulta siempre el estado físico alcanzado. Por ello también se
puede reutilizar posteriormente como oráculo DAgger.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mujoco
import numpy as np
from scipy.optimize import least_squares

from zenno_vx_r1 import ACTION_DIM, ZennoVXR1


@dataclass
class TeacherDiagnostics:
    phase: int = 0
    stage: str = "reset"
    ik_error_m: float = np.inf
    ik_solves: int = 0
    p2_cube_target_error_m: float = np.inf


class VXR1AnalyticTeacher:
    """Oráculo determinista IK/PID para episodios completos de V9."""

    # El primer punto P2 conserva la altura de transporte. Los siguientes
    # niveles fuerzan una aproximación desde arriba antes del contacto.
    # Dos metas de descenso bastan porque las acciones articulares siguen
    # siendo incrementales y el rate limiter del ambiente suaviza el trayecto.
    # Se utilizan dos metas de descenso para limitar el consumo del presupuesto de P2.
    P2_DESCENT_OFFSETS_M = (0.10, 0.0)
    P2_ALIGN_XY_TOL_M = 0.018
    P2_STAGE_POSITION_TOL_M = 0.018

    def __init__(
        self,
        env: ZennoVXR1,
        *,
        position_weight: float = 25.0,
        posture_weight: float = 0.010,
        max_ik_evaluations: int = 400,
    ):
        if not isinstance(env, ZennoVXR1):
            raise TypeError(f"Se esperaba ZennoVXR1, no {type(env)!r}.")
        self.env = env
        self.position_weight = float(position_weight)
        self.posture_weight = float(posture_weight)
        self.max_ik_evaluations = int(max_ik_evaluations)
        self.site_id = int(env.model.site("gripper_site").id)
        self.qpos_indices = np.asarray(
            [int(env.model.joint(name).qposadr[0]) for name in env.arm_joint_names],
            dtype=np.int64,
        )
        self.scratch = mujoco.MjData(env.model)
        self.diagnostics = TeacherDiagnostics()
        self._last_phase: Optional[int] = None
        self._joint_target: Optional[np.ndarray] = None
        self._p0_stage = "approach"
        self._p2_stage_index = -1
        self._p2_safe_cube_z = np.nan
        self._p2_carry_offset = np.zeros(3, dtype=np.float64)
        self._p2_cube_target: Optional[np.ndarray] = None

    @property
    def stage(self) -> str:
        return str(self.diagnostics.stage)

    def reset(self) -> None:
        self.diagnostics = TeacherDiagnostics(
            phase=int(self.env.phase_state), stage="reset"
        )
        self._last_phase = None
        self._joint_target = None
        self._p0_stage = "approach"
        self._p2_stage_index = -1
        self._p2_safe_cube_z = np.nan
        self._p2_carry_offset[:] = 0.0
        self._p2_cube_target = None

    def _copy_state_to_scratch(self) -> None:
        self.scratch.qpos[:] = self.env.data.qpos
        self.scratch.qvel[:] = self.env.data.qvel
        self.scratch.ctrl[:] = self.env.data.ctrl

    def _solve_ik(self, target_ee: np.ndarray) -> np.ndarray:
        target_ee = np.asarray(target_ee, dtype=np.float64).reshape(3)
        self._copy_state_to_scratch()
        low = np.asarray(self.env.ARM_POS_LOW, dtype=np.float64)
        high = np.asarray(self.env.ARM_POS_HIGH, dtype=np.float64)
        q_start = np.clip(
            np.asarray(self.env._get_arm_qpos(), dtype=np.float64),
            low + 1.0e-6,
            high - 1.0e-6,
        )

        def residual(q_arm: np.ndarray) -> np.ndarray:
            self.scratch.qpos[self.qpos_indices] = q_arm
            mujoco.mj_forward(self.env.model, self.scratch)
            site = np.asarray(
                self.scratch.site_xpos[self.site_id], dtype=np.float64
            )
            return np.concatenate((
                self.position_weight * (site - target_ee),
                self.posture_weight * (q_arm - q_start),
            ))

        result = least_squares(
            residual,
            q_start,
            bounds=(low, high),
            max_nfev=self.max_ik_evaluations,
            xtol=1.0e-10,
            ftol=1.0e-10,
            gtol=1.0e-10,
        )
        target = np.asarray(result.x, dtype=np.float64)
        self.scratch.qpos[self.qpos_indices] = target
        mujoco.mj_forward(self.env.model, self.scratch)
        solved = np.asarray(
            self.scratch.site_xpos[self.site_id], dtype=np.float64
        )
        self.diagnostics.ik_error_m = float(np.linalg.norm(solved - target_ee))
        self.diagnostics.ik_solves += 1
        return target

    def _incremental_action(
        self, joint_target: np.ndarray, phase: int
    ) -> np.ndarray:
        scale = np.asarray(
            self.env.ARM_DELTA_SCALE_BY_PHASE[int(phase)], dtype=np.float64
        )
        current_target = np.asarray(self.env.arm_target, dtype=np.float64)
        normalized = (
            np.asarray(joint_target, dtype=np.float64) - current_target
        ) / np.maximum(scale, 1.0e-9)
        return np.concatenate((
            np.asarray(np.clip(normalized, -1.0, 1.0), dtype=np.float32),
            np.zeros(2, dtype=np.float32),
        ))

    def _enter_phase(self, phase: int) -> None:
        self._last_phase = int(phase)
        self._joint_target = None
        self.diagnostics.phase = int(phase)
        if phase == 0:
            self._p0_stage = "lift" if bool(self.env.gripping) else "approach"
            self.diagnostics.stage = f"p0_{self._p0_stage}"
        elif phase == 1:
            self.diagnostics.stage = "p1_pid_zero_residual"
        else:
            cube = np.asarray(self.env._cube_pos(), dtype=np.float64)
            ee = np.asarray(
                self.env._site_xpos("gripper_site"), dtype=np.float64
            )
            self._p2_carry_offset = cube - ee
            self._p2_safe_cube_z = float(max(
                cube[2],
                float(self.env.Z_CUBE_REST)
                + max(self.P2_DESCENT_OFFSETS_M),
            ))
            self._p2_stage_index = -1
            self._set_p2_target(
                np.array([cube[0], cube[1], self._p2_safe_cube_z])
            )
            self.diagnostics.stage = "p2_raise"

    def _p0_target_ee(self, stage: str) -> np.ndarray:
        if stage == "approach":
            return np.asarray(
                self.env._cube_reach_target_pos(), dtype=np.float64
            )
        cube = np.asarray(self.env._cube_pos(), dtype=np.float64)
        ee = np.asarray(
            self.env._site_xpos("gripper_site"), dtype=np.float64
        )
        setpoint = np.asarray(
            self.env._site_xpos("setpoint"), dtype=np.float64
        )
        return setpoint - (cube - ee)

    def _action_phase0(self) -> np.ndarray:
        stage = "lift" if bool(self.env.gripping) else "approach"
        if self._joint_target is None or stage != self._p0_stage:
            self._p0_stage = stage
            self.diagnostics.stage = f"p0_{stage}"
            self._joint_target = self._solve_ik(self._p0_target_ee(stage))
        return self._incremental_action(self._joint_target, phase=0)

    def _set_p2_target(self, cube_target: np.ndarray) -> None:
        self._p2_cube_target = np.asarray(
            cube_target, dtype=np.float64
        ).reshape(3)
        target_ee = self._p2_cube_target - self._p2_carry_offset
        self._joint_target = self._solve_ik(target_ee)

    def _p2_target_error(self) -> float:
        if self._p2_cube_target is None:
            return np.inf
        error = float(np.linalg.norm(
            np.asarray(self.env._cube_pos(), dtype=np.float64)
            - self._p2_cube_target
        ))
        self.diagnostics.p2_cube_target_error_m = error
        return error

    def _advance_p2_stage(self) -> None:
        goal = np.asarray(
            self.env._goal_place_target_pos(), dtype=np.float64
        )
        self._p2_stage_index += 1
        if self._p2_stage_index == 0:
            target = np.array([goal[0], goal[1], self._p2_safe_cube_z])
            stage = "p2_align_xy"
        else:
            offset_index = self._p2_stage_index - 1
            offset_index = int(np.clip(
                offset_index, 0, len(self.P2_DESCENT_OFFSETS_M) - 1
            ))
            offset = float(self.P2_DESCENT_OFFSETS_M[offset_index])
            target = goal.copy()
            target[2] = float(self.env.Z_CUBE_REST + offset)
            stage = (
                "p2_place"
                if offset <= 0.0
                else f"p2_descend_{offset:.3f}m"
            )
        self.diagnostics.stage = stage
        self._set_p2_target(target)

    def _action_phase2(self) -> np.ndarray:
        if bool(self.env._released) or bool(self.env._p2_returning_home):
            self.diagnostics.stage = "p2_release_and_home"
            return np.zeros(ACTION_DIM, dtype=np.float32)

        cube = np.asarray(self.env._cube_pos(), dtype=np.float64)
        if self.diagnostics.stage == "p2_raise":
            if abs(float(cube[2]) - self._p2_safe_cube_z) <= 0.020:
                self._advance_p2_stage()
        elif self._p2_stage_index == 0:
            goal_xy = np.asarray(self.env._goal_xy(), dtype=np.float64)
            xy_error = float(np.linalg.norm(cube[:2] - goal_xy))
            z_error = abs(float(cube[2]) - self._p2_safe_cube_z)
            if xy_error <= self.P2_ALIGN_XY_TOL_M and z_error <= 0.025:
                self._advance_p2_stage()
        elif self._p2_cube_target is not None:
            # El cubo está unido al gripper mediante una restricción física:
            # durante el descenso su offset vertical puede variar varios cm.
            # Exigir distancia 3D simétrica deja al profesor bloqueado cuando
            # el cubo queda *más bajo* que el setpoint intermedio. Para avanzar
            # basta conservar el centrado XY y haber alcanzado o rebasado de
            # forma segura la altura objetivo.
            xy_error = float(np.linalg.norm(
                cube[:2] - self._p2_cube_target[:2]
            ))
            z_reached = bool(
                float(cube[2])
                <= float(self._p2_cube_target[2]) + 0.020
            )
            self._p2_target_error()
            if (
                xy_error <= 0.030
                and z_reached
                and self._p2_stage_index < len(self.P2_DESCENT_OFFSETS_M)
            ):
                self._advance_p2_stage()

        if self._joint_target is None:
            return np.zeros(ACTION_DIM, dtype=np.float32)
        return self._incremental_action(self._joint_target, phase=2)

    def action(self) -> np.ndarray:
        phase = int(self.env.phase_state)
        if self._last_phase != phase:
            self._enter_phase(phase)
        if phase == 0:
            action = self._action_phase0()
        elif phase == 1:
            action = np.zeros(ACTION_DIM, dtype=np.float32)
            # Nominal transport needs no motion near the carry target. Outside
            # its small corridor, provide a state-dependent recovery label.
            error = self.env._site_xpos('setpoint')-self.env._cube_pos()
            if np.linalg.norm(error) > .03:
                jacp = np.zeros((3,self.env.model.nv))
                jacr = np.zeros_like(jacp)
                mujoco.mj_jacSite(self.env.model,self.env.data,jacp,jacr,self.site_id)
                dofs = [int(self.env.model.joint(n).dofadr[0]) for n in self.env.arm_joint_names]
                jac = jacp[:,dofs]
                task = np.clip(.15*error, -.003, .003)
                delta = jac.T @ np.linalg.solve(jac @ jac.T + .035**2*np.eye(3),task)
                target = self.env._get_arm_qpos()+delta
                action = self._incremental_action(target, phase=1)
                self.diagnostics.stage = 'p1_carry_recovery'
            else:
                self.diagnostics.stage = 'p1_carry_hold'
        else:
            action = self._action_phase2()
        if action.shape != (ACTION_DIM,) or not np.all(np.isfinite(action)):
            raise RuntimeError(
                f"Acción analítica inválida en {self.stage}: {action!r}"
            )
        return np.clip(action, -1.0, 1.0).astype(np.float32)


__all__ = ["TeacherDiagnostics", "VXR1AnalyticTeacher"]
