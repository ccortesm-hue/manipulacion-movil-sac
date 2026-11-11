#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "glfw" if "--viewer" in sys.argv else "egl")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vx_compact_matplotlib")
RESOURCE_ROOT = Path(os.environ.get("VX_RESOURCE_ROOT", str(Path(__file__).resolve().parent.parent))).resolve()

import MM_DATA_VXR1 as shared
from MM_DATA_VXR1 import (np, spaces, Tuple, ACTION_DIM, VX_OBS_DIM, OBS_DIM,
                        VX_EXTRA_NAMES, wrap_angle, _VXReference)

class _PurePursuitReference(_VXReference):
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        public_action = np.clip(
            np.asarray(action, dtype=np.float32), -1.0, 1.0
        )
        if public_action.shape != (ACTION_DIM,):
            raise ValueError(
                f"ZennoVXR2 espera accion ({ACTION_DIM},), recibio {public_action.shape}."
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
            observation, reward, terminated, truncated, info = super(_VXReference, self).step(v7_action)
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

class ZennoVXR2(_PurePursuitReference):
    """VX-R2: observación 124D, residual sobre referencias Pure Pursuit antes del PID."""

    REVISION = 'VX_R2_P1_SAC_PP_REFERENCE_124D_7D_DELTA008_UNIFORM_COSTS'
    TIME_COST_PER_SECOND = 0.03
    ENERGY_COST_PER_JOULE = 0.0005
    # Incremento máximo de la referencia por decisión, igual en P0/P1/P2.
    ARM_DELTA_SCALE_BY_PHASE = np.full((3, 5), 0.008, dtype=np.float32)
    ACTION_SEMANTICS = 'incremental_arm_pure_pursuit_reference_residual'
    PP_LINEAR_FRACTION = 0.20
    PP_ANGULAR_FRACTION = 0.20
    RESIDUAL_FRACTION = 0.20
    RESIDUAL_TAU_S = 0.15
    RESIDUAL_SLEW_PER_S = 1.0
    W_BASE_ACTION_MAG = 0.0
    W_BASE_ACTION_SLEW = 0.0

    def __init__(self, *, residual_enabled=True, p1_budget_steps=4500, **kwargs):
        self.p1_budget_steps = self._validate_p1_budget(p1_budget_steps)
        self.residual_enabled = bool(residual_enabled)
        self._reset_vx()
        super().__init__(**kwargs)
        self.MAX_STEPS_0, self.MAX_STEPS_1, self.MAX_STEPS_2 = 1000, self.p1_budget_steps, 1000
        self._set_resource_cost_weights(self.dt)
        self.observation_space = spaces.Box(-np.inf, np.inf, (VX_OBS_DIM,), np.float32)

    @staticmethod
    def _validate_p1_budget(value):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
            raise ValueError('p1_budget_steps debe ser un entero positivo.')
        return int(value)

    @property
    def phase_budgets(self):
        return (int(self.MAX_STEPS_0), int(self.MAX_STEPS_1), int(self.MAX_STEPS_2))

    @property
    def max_episode_steps(self):
        """Cota total coherente para evaluadores; P2 ya incluye home."""
        return sum(self.phase_budgets)

    def set_p1_budget_steps(self, value):
        """Usar entre episodios al cambiar el horizonte del curriculum."""
        self.p1_budget_steps = self._validate_p1_budget(value)
        self.MAX_STEPS_1 = self.p1_budget_steps

    def _set_resource_cost_weights(self, dt):
        # Keep inherited reward branches and telemetry on the same physical units.
        self.W_TIME = self.W_TIME_P2 = self.TIME_COST_PER_SECOND * float(dt)
        self.W_EN_ARM = self.W_EN_BASE = self.ENERGY_COST_PER_JOULE
        self.W_EN_ARM_P2 = self.W_EN_BASE_P2 = self.ENERGY_COST_PER_JOULE

    def _compute_phase_reward(self, phase_eval, Ts, r_release_ready):
        self._set_resource_cost_weights(Ts)
        result = super()._compute_phase_reward(phase_eval, Ts, r_release_ready)
        if int(phase_eval) == 1:
            # Capture after stop_count is updated but BEFORE stage_setup_phase2
            # clears navigation state. This telemetry never advances physics.
            self._r2_last_p1_diagnostics = self.phase1_diagnostics()
        return result

    def _reset_vx(self):
        self._r2_requested_residual = np.zeros(2)
        self._r2_pp_nominal = np.zeros(2)
        self._r2_pp_corrected = np.zeros(2)
        self._r2_raw_wheel_commands = {}
        self._r2_nominal_wheel_commands = {}
        self._r2_requested_wheel_delta = {}
        self._r2_pp_speed_scale = 1.0
        self._r2_last_p1_diagnostics = None
        self._vx_filter = np.zeros(2)
        self._vx_nominal = np.zeros(2)
        self._vx_delta = np.zeros(2)
        self._vx_saturation = 0.0
        self._vx_active = False
        self._vx_prev_sp = None
        self._vx_relative_velocity = np.zeros(3)
        self._last_track_err = self._last_cte_signed = self._last_yaw_track_err = 0.0
        self._last_v_des_cmd = self._last_omega_des_cmd = 0.0

    def phase1_diagnostics(self):
        """Read-only P1 handoff conditions and command-level torque telemetry.

        Torque arrays follow left-wheel IDs, then right-wheel IDs. They are
        actuator commands (Nm, unit wheel gears), not measured joint forces.
        At reward time the fields describe the step before phase transitions.
        """
        geom, progress, length, endpoint, preview, valid = self._route_state()
        goal = self._goal_xy() - self._base_xy()
        heading = float(wrap_angle(np.arctan2(goal[1], goal[0]) - self._base_yaw()))
        route_heading = None
        route_goal_gap = None
        if valid:
            points = np.asarray(self._waypoints)[:, :2]
            tangent = points[-1] - points[-2]
            terminal_yaw = float(np.arctan2(tangent[1], tangent[0]))
            route_heading = float(wrap_angle(terminal_yaw - self._base_yaw()))
            route_goal_gap = float(wrap_angle(heading - route_heading))
        checks = {
            'route_valid': bool(valid),
            'nav_done': bool(self._nav_done),
            'endpoint_reached': bool(valid and endpoint <= self.FINAL_WAYPOINT_TOL),
            'heading_aligned': bool(abs(heading) <= self.FINAL_ROUTE_HEADING_TOL),
            'stop_count_satisfied': bool(self._stop_count >= self.STOP_COUNT_MIN),
            'gripping': bool(self.gripping),
            'cube_not_dropped': bool(not self.cube_dropped),
        }
        ids, nominal, requested, applied = [], [], [], []
        for side, wheel_ids in enumerate((self.wheel_left_ids, self.wheel_right_ids)):
            for aid in wheel_ids:
                limit = self._actuator_tau_limit(aid)
                ids.append(int(aid))
                nominal.append(float(self._vx_nominal[side] * limit))
                requested.append(float(self._r2_requested_wheel_delta.get(int(aid), 0.)) if self._vx_active else 0.0)
                applied.append(float(self._vx_delta[side] * limit))
        return {
            'phase': int(self.phase_state), 'phase_steps': int(self._phase_steps),
            'budget_steps': int(self.MAX_STEPS_1),
            'remaining_steps': max(0, int(self.MAX_STEPS_1) - int(self._phase_steps)),
            'checks': checks, 'ready': bool(all(checks.values())),
            'blocking_conditions': [name for name, passed in checks.items() if not passed],
            'endpoint_distance_m': float(endpoint) if valid else None,
            'heading_error_rad': heading, 'route_progress_m': float(progress),
            'route_terminal_heading_error_rad': route_heading,
            'goal_minus_route_terminal_heading_rad': route_goal_gap,
            'route_terminal_heading_aligned': bool(valid and abs(route_heading) <= self.FINAL_ROUTE_HEADING_TOL),
            'route_length_m': float(length), 'cte_m': float(geom[0]),
            'endpoint_tolerance_m': float(self.FINAL_WAYPOINT_TOL),
            'heading_tolerance_rad': float(self.FINAL_ROUTE_HEADING_TOL),
            'initial_aligning': bool(self._initial_aligning),
            'reverse_active': bool(getattr(self, '_reverse_active', False)),
            'terminal_region': bool(valid and not self._initial_aligning and endpoint <= self.PURE_PURSUIT_TERMINAL_POSE_DIST_M),
            'base_mode': str(getattr(self, '_last_base_mode', '')),
            'base_stopped': bool(self._base_is_stopped()),
            'stop_count': int(self._stop_count), 'stop_count_required': int(self.STOP_COUNT_MIN),
            'terminal_route_stable_count': int(getattr(self, '_terminal_route_stable_count', 0)),
            'terminal_route_stable_steps_required': int(self.FINAL_ROUTE_STABLE_STEPS),
            'carry_error_m': float(np.linalg.norm(self._cube_pos() - self._site_xpos('setpoint'))),
            'carry_error_is_handoff_gate': False,
            'residual_active': bool(self._vx_active),
            'requested_action': self._r2_requested_residual.tolist(),
            'filtered_action': self._vx_filter.tolist(),
            'wheel_actuator_ids': ids,
            'wheel_nominal_command_nm': nominal,
            'wheel_requested_residual_nm': requested,
            'wheel_applied_residual_nm': applied,
            'residual_level': 'pure_pursuit_reference',
            'pp_nominal_reference': self._r2_pp_nominal.tolist(),
            'pp_corrected_reference': self._r2_pp_corrected.tolist(),
            'pp_reference_delta': (self._r2_pp_corrected-self._r2_pp_nominal).tolist(),
            'sample': 'current_state_before_transition_when_called_from_reward',
        }

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
        yaw_abs, cte_abs = abs(float(kwargs.get('yaw_abs',0.))), abs(float(kwargs.get('cte_abs',0.)))
        scale = 1.
        if yaw_abs > self.YAW_STOP:
            scale = 0.
        elif yaw_abs > self.YAW_SLOW_2:
            scale *= self.YAW_SLOW_GAIN_2
        elif yaw_abs > self.YAW_SLOW_1:
            scale *= self.YAW_SLOW_GAIN_1
        if cte_abs > self.CTE_SLOW_2:
            scale *= self.CTE_SLOW_GAIN_2
        elif cte_abs > self.CTE_SLOW_1:
            scale *= self.CTE_SLOW_GAIN_1
        self._r2_pp_speed_scale = float(scale)
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
        # The inherited name is an API hook, NOT a torque residual in R2.
        if abs(u_lin)>1e-12 or abs(u_ang)>1e-12:
            raise RuntimeError('R2 prohíbe residuales de torque: corregir referencias antes del PID')
        self._r2_raw_wheel_commands = {}
        for side, ids in enumerate((self.wheel_left_ids, self.wheel_right_ids)):
            for aid in ids:
                limit = self._actuator_tau_limit(aid)
                raw = float((tau_each_L,tau_each_R)[side])
                self._r2_raw_wheel_commands[aid] = raw
                self.data.ctrl[aid] = np.clip(raw,-limit,limit) if limit>0 else 0.

    _R2_PID_STATE = ('_Accel_lin_des','_Accel_ang_des','_e_vel1','_e_vel2',
        '_e_omega1','_e_omega2','_v_des_f','_omega_des_f','_omega_des_prev',
        '_last_base_mode','_last_v_des_cmd','_last_omega_des_cmd')

    def _control_base_vel(self, Ts, v_des, omega_des, u_lin=0., u_ang=0., **kwargs):
        """Two algebraic PID queries from SAME state, ONE committed controller step.

        The nominal query measures a local counterfactual, not a second agent or
        a second physical simulation. All its PID state and ctrl writes are undone.
        Observation torque channels keep the same units/meaning as R1.
        """
        nominal_refs=np.array([v_des,omega_des],dtype=float)
        corrected=nominal_refs.copy()
        active=bool(kwargs.get('mode','tracking')=='tracking' and self._vx_active)
        if active and (abs(u_lin)>0. or abs(u_ang)>0.):
            corrected[1]=np.clip(omega_des+self.PP_ANGULAR_FRACTION*self.OMEGA_MAX*u_ang,
                                 -self.OMEGA_MAX,self.OMEGA_MAX)
            # A positive residual cannot restart a yaw-stopped nominal command
            # or erase its geometric slowdown. Uplift is <=20% of nominal v,
            # also bounded by the calibrated curvature/yaw/CTE speed envelope.
            curvature_cap = (float(self.V_CURV) /
                (1. + float(self.K_CURV)*abs(corrected[1]))) * self._r2_pp_speed_scale
            ceiling = min(float(self.V_MAX),
                          max(0., float(v_des))*(1.+self.PP_LINEAR_FRACTION),
                          max(float(v_des), curvature_cap))
            corrected[0]=np.clip(v_des+self.PP_LINEAR_FRACTION*self.V_MAX*u_lin,
                                  0., max(0., ceiling))
        self._r2_pp_nominal=nominal_refs.copy()
        self._r2_pp_corrected=corrected.copy()
        saved={name:getattr(self,name) for name in self._R2_PID_STATE}
        saved_ctrl=self.data.ctrl.copy()
        try:
            super()._control_base_vel(Ts,float(v_des),float(omega_des),u_lin=0.,u_ang=0.,**kwargs)
            nominal_ctrl=self.data.ctrl.copy()
            nominal_raw=self._r2_raw_wheel_commands.copy()
        finally:
            for name,value in saved.items():setattr(self,name,value)
            self.data.ctrl[:]=saved_ctrl
        # This is the only committed update of PID memory.
        super()._control_base_vel(Ts,float(corrected[0]),float(corrected[1]),u_lin=0.,u_ang=0.,**kwargs)
        actual_raw=self._r2_raw_wheel_commands.copy()
        self._r2_nominal_wheel_commands = {int(aid):float(nominal_ctrl[aid])
            for aid in self.wheel_left_ids+self.wheel_right_ids}
        self._r2_requested_wheel_delta = {}
        nominal,delta,saturation=[],[],[]
        for ids in (self.wheel_left_ids,self.wheel_right_ids):
            ns,ds=[],[]
            for aid in ids:
                limit=self._actuator_tau_limit(aid)
                if limit<=0:continue
                applied=float(self.data.ctrl[aid]-nominal_ctrl[aid])
                requested=float(actual_raw[aid]-nominal_raw[aid])
                self._r2_requested_wheel_delta[int(aid)] = requested
                ns.append(nominal_ctrl[aid]/limit);ds.append(applied/limit)
                # Same R1 notion: requested equivalent correction not realized
                # at the actuator due to clipping; zero for zero intervention.
                saturation.append(((requested-applied)/limit)**2)
            nominal.append(float(np.mean(ns)) if ns else 0.)
            delta.append(float(np.mean(ds)) if ds else 0.)
        self._vx_nominal=np.asarray(nominal)
        self._vx_delta=np.asarray(delta)
        self._vx_saturation=float(np.mean(saturation)) if saturation else 0.

    def _get_obs(self):
        original = super()._get_obs()
        geom, s, length, endpoint, preview, valid = self._route_state()
        theta = self._base_yaw()
        velocity = float(np.cos(theta)*self.data.qvel[0]+np.sin(theta)*self.data.qvel[1])
        goal = self._goal_xy()-self._base_xy()
        terminal_yaw = wrap_angle(np.arctan2(goal[1], goal[0])-theta)
        carry = self._cube_pos()-self._site_xpos('setpoint')
        budget = self.phase_budgets[int(self.phase_state)]
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
        self._r2_requested_residual = action[5:7].copy()
        self._r2_last_p1_diagnostics = None
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
                    vx_budgets=list(self.phase_budgets), p2_home_budget_steps=0,
                    p2_total_budget_steps=1000, p2_home_included_in_total=True)
        info.update(experiment_revision=self.REVISION,
                    max_episode_steps=self.max_episode_steps,
                    p1_diagnostics=self._r2_last_p1_diagnostics,
                    residual_level='pure_pursuit_reference', torque_residual_enabled=False,
                    pp_nominal_reference=self._r2_pp_nominal.tolist(),
                    pp_corrected_reference=self._r2_pp_corrected.tolist(),
                    pp_reference_delta=(self._r2_pp_corrected-self._r2_pp_nominal).tolist())
        return self._get_obs(), float(reward), terminated, truncated, info

def main(argv=None):
    shared.ACTIVE_ENVIRONMENT = ZennoVXR2
    shared.ACTIVE_EXPERIMENT = 'r2'
    return shared.main(argv)

if __name__ == '__main__':
    main()
