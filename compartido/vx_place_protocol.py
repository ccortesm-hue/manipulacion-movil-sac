"""Common, read-only stable-placement measurement for VX and the baseline.

The driver owns the controller and its native time budgets. This module never
changes either. Call substep() immediately after EACH mj_step and end_interval()
after exactly five such calls. Geometry/contact samples are recomputed at the
POST-integration state on a separate MjData; the live simulation is not forwarded.
Power is the absolute actuator force-times-transmission-velocity sampled after
each live step, not an electrical/battery estimate. The gripper is a separate
energy group and IS included in total_energy_J.

Placement is physical, not a native environment success flag: the cube must be
released, nearly stationary, over the goal pedestal and supported by that exact
pedestal for 20 consecutive 10-ms intervals. No return-home criterion is used.
The common 200-s horizon and 0.20-m fall threshold are evaluation criteria; they
do not rewrite the policy's observation, phase budget or trained controller.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import mujoco
import numpy as np


PROTOCOL_VERSION = "vx_stable_place_common_v1"
DT_PHYSICS_S = 0.002
FRAMESKIP = 5
DT_CONTROL_S = DT_PHYSICS_S * FRAMESKIP
GLOBAL_HORIZON_S = 200.0
GLOBAL_HORIZON_STEPS = round(GLOBAL_HORIZON_S / DT_CONTROL_S)
STABLE_PLACE_STEPS = 20
PLACE_XY_TOL_M = 0.05
PLACE_SPEED_TOL_M_S = 0.03
PLACE_HEIGHT_TOL_M = 0.010
FINGER_STRONG_FORCE_N = 0.8
SUPPORT_FORCE_EPS_N = 1e-6
CUBE_FALL_CENTER_Z_M = 0.20
TANGENT_SPAN_M = 0.32
GRIPPER_CONTROL_TOL = 1e-6

ARM_ACTUATORS = tuple(f"motor_j{i}" for i in range(1, 6))
BASE_ACTUATORS = ("tau_fl", "tau_cl", "tau_rl", "tau_fr", "tau_cr", "tau_rr")
GRIPPER_ACTUATORS = ("motor_j6",)
FINGER_GEOMS = ("left_pad1", "left_pad2", "right_pad1", "right_pad2")
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def protocol_metadata() -> dict[str, Any]:
    """JSON-compatible, explicit definition of this measurement contract."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dt_physics_s": DT_PHYSICS_S,
        "frameskip": FRAMESKIP,
        "dt_control_s": DT_CONTROL_S,
        "global_horizon_s": GLOBAL_HORIZON_S,
        "global_horizon_steps": GLOBAL_HORIZON_STEPS,
        "stable_place_steps": STABLE_PLACE_STEPS,
        "place_xy_tolerance_m": PLACE_XY_TOL_M,
        "cube_speed_tolerance_m_s": PLACE_SPEED_TOL_M_S,
        "cube_bottom_to_goal_top_tolerance_m": PLACE_HEIGHT_TOL_M,
        "goal_geometry": "pedestal_geom_2 (upright cylinder only)",
        "height_definition": "cube box center_z - sum(abs(R_zj)*box_half_size_j) - cylinder_top_z",
        "support_definition": "sum world-Fz ON cube from active cube-goal contacts > epsilon",
        "support_force_epsilon_N": SUPPORT_FORCE_EPS_N,
        "release_definition": "gripper ctrl at lower limit AND each pad summed normal contact force < threshold",
        "finger_strong_force_N": FINGER_STRONG_FORCE_N,
        "gripper_control_tolerance": GRIPPER_CONTROL_TOL,
        "cube_fall_center_z_m": CUBE_FALL_CENTER_Z_M,
        "fall_comparison": "<=",
        "return_home_required": False,
        "success_requires_native_phase_or_flag": False,
        "native_deadline_terminates": False,
        "positions_sample": "post-integration t_end on auxiliary MjData",
        "energy_sample": "sum of abs(actuator_force * actuator_velocity) * 0.002 after each live mj_step",
        "energy_total_includes_gripper": True,
        "energy_groups": {"arm": list(ARM_ACTUATORS), "base": list(BASE_ACTUATORS),
                          "gripper": list(GRIPPER_ACTUATORS)},
        "cte_definition": "nearest segment left normal dot (base_xy - clipped projection)",
        "distance_to_route_definition": "Euclidean distance to nearest clipped segment projection",
        "yaw_reference": "chord between interpolated arc points s-0.16 and s+0.16 m, clipped at endpoints",
        "tangent_span_m": TANGENT_SPAN_M,
        "route_progress_units": "fraction of route arc length [0,1] (not monotonically forced)",
        "tracking_summary_cohort": "phase == 1 AND supplied tracking flag",
    }


def principal_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class RouteGeometry:
    """Independent polyline projection: lateral and Euclidean errors differ.

    A point beyond a straight endpoint can have zero lateral CTE and positive
    Euclidean route distance. A left-side point has positive signed CTE. Yaw
    uses a spatially smoothed tangent; lateral CTE uses the actual closest
    segment normal. Ties at corners follow the first segment deterministically.
    """

    def __init__(self, path_xy: np.ndarray):
        points = np.asarray(path_xy, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("path_xy must be an Nx2 array with at least two points")
        if not np.isfinite(points).all():
            raise ValueError("Route points must be finite")
        keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-10]
        self.points = points[keep].copy()
        if len(self.points) < 2:
            raise ValueError("Route must contain at least two distinct points")
        self.segments = np.diff(self.points, axis=0)
        self.lengths = np.linalg.norm(self.segments, axis=1)
        self.arc = np.r_[0.0, np.cumsum(self.lengths)]
        self.total_length_m = float(self.arc[-1])
        self.tangents = self.segments / self.lengths[:, None]
        self.normals = np.c_[-self.tangents[:, 1], self.tangents[:, 0]]

    def point_at_arc(self, arc_m: float) -> np.ndarray:
        s = float(np.clip(arc_m, 0.0, self.total_length_m))
        i = min(int(np.searchsorted(self.arc, s, side="right")) - 1,
                len(self.segments) - 1)
        i = max(i, 0)
        return self.points[i] + ((s - self.arc[i]) / self.lengths[i]) * self.segments[i]

    def sample(self, xy: np.ndarray, base_yaw_rad: float) -> dict[str, float]:
        p = np.asarray(xy, dtype=np.float64)
        if p.shape != (2,) or not np.isfinite(p).all() or not np.isfinite(base_yaw_rad):
            raise ValueError("Pose must contain finite XY and yaw")
        offsets = p - self.points[:-1]
        ratios = np.clip(np.sum(offsets * self.segments, axis=1) / self.lengths**2, 0.0, 1.0)
        projections = self.points[:-1] + ratios[:, None] * self.segments
        differences = p - projections
        squared = np.sum(differences**2, axis=1)
        i = int(np.argmin(squared))
        s = float(self.arc[i] + ratios[i] * self.lengths[i])
        chord = (self.point_at_arc(s + TANGENT_SPAN_M / 2.0)
                 - self.point_at_arc(s - TANGENT_SPAN_M / 2.0))
        if np.linalg.norm(chord) <= 1e-10:
            chord = self.segments[i]
        path_yaw = float(np.arctan2(chord[1], chord[0]))
        signed = float(np.dot(self.normals[i], differences[i]))
        return {
            "cte_signed_m": signed,
            "cte_abs_m": abs(signed),
            "distance_to_route_m": float(np.sqrt(squared[i])),
            "path_yaw_rad": path_yaw,
            "heading_error_rad": principal_angle(base_yaw_rad - path_yaw),
            "route_progress": s / self.total_length_m,
        }


def _named_id(model, kind, name: str) -> int:
    index = int(mujoco.mj_name2id(model, kind, name))
    if index < 0:
        raise ValueError(f"Required model object is missing: {name}")
    return index


def _actuator_ids(model, names: tuple[str, ...]) -> np.ndarray:
    return np.array([_named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                     for name in names], dtype=np.int32)


def contact_force_on_geom(model, data, contact_index: int, geom_id: int) -> np.ndarray:
    """World force on the requested geom; inactive constraints contribute zero.

    MuJoCo contact-frame rows are world axes. Contact force acts from geom1
    toward geom2; consequently its sign reverses for geom1. See MuJoCo's
    Computation/Contact documentation (normal is contact-frame X, not Z).
    """
    contact = data.contact[contact_index]
    if geom_id not in (int(contact.geom1), int(contact.geom2)):
        raise ValueError("Requested geom does not participate in the contact")
    if int(contact.efc_address) < 0:
        return np.zeros(3, dtype=np.float64)
    force = np.zeros(6, dtype=np.float64)
    mujoco.mj_contactForce(model, data, contact_index, force)
    result = np.asarray(contact.frame).reshape(3, 3).T @ force[:3]
    return result if int(contact.geom2) == geom_id else -result


@dataclass(frozen=True)
class PlacementSample:
    cube_x: float
    cube_y: float
    cube_z: float
    cube_goal_dist_xy_m: float
    cube_speed_m_s: float
    cube_bottom_z_m: float
    goal_top_z_m: float
    cube_goal_height_error_m: float
    goal_support_fz_N: float
    finger_contact_max_N: float
    gripper_open: bool
    place_candidate: bool


class Recorder:
    """In-memory recorder with identical measurement/termination for both arms.

    Public state is read through summary()/traces(). These methods return copies;
    no file is written and no model/data attribute is assigned. A native deadline
    is recorded, NEVER used to stop. Terminal reasons are stable_place, cube_drop
    and global_deadline, with success at exactly 200 s counted before deadline.
    """

    def __init__(self, model, data, path: np.ndarray):
        if not np.isclose(float(model.opt.timestep), DT_PHYSICS_S, atol=1e-12, rtol=0):
            raise ValueError("The common protocol requires a 0.002-s physical timestep")
        if mujoco.get_mjcb_control() is not None:
            raise ValueError("Auxiliary mj_forward requires no process-global control callback")
        self.model = model
        self.data = data
        self.route = RouteGeometry(path)
        self.arm_ids = _actuator_ids(model, ARM_ACTUATORS)
        self.base_ids = _actuator_ids(model, BASE_ACTUATORS)
        self.gripper_ids = _actuator_ids(model, GRIPPER_ACTUATORS)
        all_ids = np.r_[self.arm_ids, self.base_ids, self.gripper_ids]
        if len(set(all_ids.tolist())) != 12 or model.nu != 12:
            raise ValueError("Protocol requires exactly five arm, six wheel and one gripper actuators")
        self.cube_body_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        cube_geoms = np.flatnonzero(model.geom_bodyid == self.cube_body_id)
        if len(cube_geoms) != 1 or int(model.geom_type[cube_geoms[0]]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            raise ValueError("Cube body must contain exactly one box geom")
        self.cube_geom_id = int(cube_geoms[0])
        if not np.allclose(model.geom_pos[self.cube_geom_id], 0.0, atol=1e-12, rtol=0):
            raise ValueError("Cube geom and body centers must coincide for this protocol")
        self.goal_geom_id = _named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "pedestal_geom_2")
        if int(model.geom_type[self.goal_geom_id]) != int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            raise ValueError("Goal pedestal must be a cylinder")
        self.base_site_id = _named_id(model, mujoco.mjtObj.mjOBJ_SITE, "chassis_frame")
        self.finger_ids = tuple(_named_id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in FINGER_GEOMS)
        grip_id = int(self.gripper_ids[0])
        if not bool(model.actuator_ctrllimited[grip_id]):
            raise ValueError("The gripper must have a finite, limited control range")
        self.gripper_open_ctrl = float(model.actuator_ctrlrange[grip_id, 0])
        self._aux = mujoco.MjData(model)
        self._state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC), dtype=np.float64)
        self._time_origin = float(data.time)
        self._last_physics_time = float(data.time)
        self._interval_substeps = 0
        self._interval_energy = np.zeros(3, dtype=np.float64)
        self._total_energy = np.zeros(3, dtype=np.float64)
        self._phase_energy = np.zeros((3, 3), dtype=np.float64)
        self._phase_steps = np.zeros(3, dtype=np.int64)
        self._rows: list[dict[str, Any]] = []
        self._stable_count = 0
        self._stable_place_step: int | None = None
        self._native_deadline_first_step: int | None = None
        self._termination_reason: str | None = None
        self._measurement_wall_time_s = 0.0
        self._last_placement: PlacementSample | None = None

    def substep(self) -> None:
        """Integrate one physical interval, immediately after the live mj_step."""
        if self._termination_reason is not None:
            raise RuntimeError("Recorder already reached a terminal event")
        if self._interval_substeps >= FRAMESKIP:
            raise RuntimeError("Call end_interval after exactly five physical steps")
        now = float(self.data.time)
        if not np.isclose(now - self._last_physics_time, DT_PHYSICS_S, atol=2e-9, rtol=0):
            raise RuntimeError("substep must follow one and only one 0.002-s mj_step")
        force = np.asarray(self.data.actuator_force)
        velocity = np.asarray(self.data.actuator_velocity)
        power = np.abs(force * velocity)
        if power.shape != (12,) or not np.isfinite(power).all():
            raise ValueError("Non-finite or incompatible actuator power")
        groups = (self.arm_ids, self.base_ids, self.gripper_ids)
        self._interval_energy += np.array([np.sum(power[ids]) for ids in groups]) * DT_PHYSICS_S
        self._interval_substeps += 1
        self._last_physics_time = now

    def _fresh_auxiliary_state(self):
        # INTEGRATION includes time, qpos/qvel/act, warmstart, ctrl, applied forces,
        # mocap, equality activity, userdata and plugin state in this MuJoCo API.
        mujoco.mj_getState(self.model, self.data, self._state, STATE_SPEC)
        if not np.isfinite(self._state).all():
            raise ValueError("Non-finite live integration state")
        mujoco.mj_setState(self.model, self._aux, self._state, STATE_SPEC)
        mujoco.mj_forward(self.model, self._aux)
        return self._aux

    def _placement_sample(self, fresh) -> PlacementSample:
        model = self.model
        cube = np.asarray(fresh.geom_xpos[self.cube_geom_id])
        cube_rotation = np.asarray(fresh.geom_xmat[self.cube_geom_id]).reshape(3, 3)
        goal = np.asarray(fresh.geom_xpos[self.goal_geom_id])
        goal_rotation = np.asarray(fresh.geom_xmat[self.goal_geom_id]).reshape(3, 3)
        if not np.isclose(abs(goal_rotation[2, 2]), 1.0, atol=1e-8, rtol=0):
            raise ValueError("Goal cylinder must remain upright for the planar support-height criterion")
        half_extent_z = float(np.dot(np.abs(cube_rotation[2]), model.geom_size[self.cube_geom_id]))
        cube_bottom = float(cube[2] - half_extent_z)
        goal_top = float(goal[2] + model.geom_size[self.goal_geom_id, 1])
        height_error = cube_bottom - goal_top
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(model, fresh, mujoco.mjtObj.mjOBJ_BODY,
                                self.cube_body_id, velocity, 0)
        speed = float(np.linalg.norm(velocity[3:]))
        xy_distance = float(np.linalg.norm(cube[:2] - goal[:2]))
        support_fz = 0.0
        pad_normal = {index: 0.0 for index in self.finger_ids}
        local_force = np.zeros(6, dtype=np.float64)
        for ci in range(fresh.ncon):
            contact = fresh.contact[ci]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if self.cube_geom_id not in (g1, g2) or int(contact.efc_address) < 0:
                continue
            other = g2 if g1 == self.cube_geom_id else g1
            if other == self.goal_geom_id:
                support_fz += float(contact_force_on_geom(model, fresh, ci, self.cube_geom_id)[2])
            elif other in pad_normal:
                mujoco.mj_contactForce(model, fresh, ci, local_force)
                # Sum all contact points on each pad: splitting a load over a
                # contact manifold must not turn a strong grasp into release.
                pad_normal[other] += max(0.0, float(local_force[0]))
        finger_max = max(pad_normal.values())
        gripper_open = bool(abs(float(fresh.ctrl[self.gripper_ids[0]]) - self.gripper_open_ctrl)
                            <= GRIPPER_CONTROL_TOL)
        candidate = bool(
            xy_distance <= PLACE_XY_TOL_M
            and abs(height_error) <= PLACE_HEIGHT_TOL_M
            and speed <= PLACE_SPEED_TOL_M_S
            and support_fz > SUPPORT_FORCE_EPS_N
            and gripper_open
            and finger_max < FINGER_STRONG_FORCE_N
        )
        return PlacementSample(float(cube[0]), float(cube[1]), float(cube[2]),
                               xy_distance, speed, cube_bottom, goal_top,
                               height_error, support_fz, finger_max, gripper_open, candidate)

    def end_interval(self, phase: int, tracking: bool,
                     native_deadline: bool = False) -> str | None:
        """Append one POST-step row, then return a common terminal reason or None.

        phase describes the phase that produced these five physical steps, not
        a phase possibly entered by the reward after the interval. tracking is
        likewise the mode active during this interval. Success uses no phase gate.
        """
        start = perf_counter()
        if self._termination_reason is not None:
            raise RuntimeError("Recorder already reached a terminal event")
        if phase not in (0, 1, 2):
            raise ValueError("phase must be 0, 1 or 2")
        if self._interval_substeps != FRAMESKIP:
            raise RuntimeError("end_interval requires exactly five recorded physical steps")
        step = len(self._rows) + 1
        elapsed = float(self.data.time) - self._time_origin
        if not np.isclose(elapsed, step * DT_CONTROL_S, atol=2e-8, rtol=0):
            raise RuntimeError("Physical time and recorded control intervals disagree")
        fresh = self._fresh_auxiliary_state()
        placement = self._placement_sample(fresh)
        base_xy = np.asarray(fresh.site_xpos[self.base_site_id, :2])
        base_rotation = np.asarray(fresh.site_xmat[self.base_site_id]).reshape(3, 3)
        base_yaw = float(np.arctan2(base_rotation[1, 0], base_rotation[0, 0]))
        geometry = self.route.sample(base_xy, base_yaw)
        self._stable_count = self._stable_count + 1 if placement.place_candidate else 0
        if native_deadline and self._native_deadline_first_step is None:
            self._native_deadline_first_step = step
        if self._stable_count >= STABLE_PLACE_STEPS:
            self._termination_reason = "stable_place"
            self._stable_place_step = step
        elif placement.cube_z <= CUBE_FALL_CENTER_Z_M:
            self._termination_reason = "cube_drop"
        elif step >= GLOBAL_HORIZON_STEPS:
            self._termination_reason = "global_deadline"
        self._total_energy += self._interval_energy
        self._phase_energy[int(phase)] += self._interval_energy
        self._phase_steps[int(phase)] += 1
        row = {
            "t_s": step * DT_CONTROL_S,
            "phase": int(phase),
            "x": float(base_xy[0]),
            "y": float(base_xy[1]),
            "base_yaw_rad": base_yaw,
            **geometry,
            "tracking_valid": bool(tracking and int(phase) == 1),
            **asdict(placement),
            "energy_arm_J": float(self._interval_energy[0]),
            "energy_base_J": float(self._interval_energy[1]),
            "energy_gripper_J": float(self._interval_energy[2]),
            "stable_place_count": self._stable_count,
            "native_deadline": bool(native_deadline),
        }
        self._rows.append(row)
        self._last_placement = placement
        self._interval_energy.fill(0.0)
        self._interval_substeps = 0
        self._measurement_wall_time_s += perf_counter() - start
        return self._termination_reason

    def traces(self) -> dict[str, np.ndarray]:
        """Independent arrays; interval energy fields are increments, not sums."""
        if not self._rows:
            names = ("t_s", "phase", "x", "y", "base_yaw_rad", "path_yaw_rad",
                     "cte_signed_m", "cte_abs_m", "distance_to_route_m", "route_progress",
                     "tracking_valid", "cube_x", "cube_y", "cube_z", "cube_goal_dist_xy_m",
                     "energy_arm_J", "energy_base_J", "energy_gripper_J", "heading_error_rad",
                     "cube_speed_m_s", "cube_bottom_z_m", "goal_top_z_m", "cube_goal_height_error_m",
                     "goal_support_fz_N", "finger_contact_max_N", "gripper_open", "place_candidate",
                     "stable_place_count", "native_deadline")
        else:
            names = tuple(self._rows[0])
        boolean = {"tracking_valid", "gripper_open", "place_candidate", "native_deadline"}
        integer = {"phase", "stable_place_count"}
        return {name: np.array([row[name] for row in self._rows],
                               dtype=bool if name in boolean else np.int64 if name in integer else np.float64)
                for name in names}

    def summary(self) -> dict[str, Any]:
        traces = self.traces()
        valid = traces["tracking_valid"]

        def rms(name: str):
            values = traces[name][valid]
            return float(np.sqrt(np.mean(values**2))) if values.size else None

        signed = traces["cte_signed_m"][valid]
        return {
            "protocol_version": PROTOCOL_VERSION,
            "total_steps": len(self._rows),
            "total_time_s": len(self._rows) * DT_CONTROL_S,
            "total_energy_arm_J": float(self._total_energy[0]),
            "total_energy_base_J": float(self._total_energy[1]),
            "total_energy_gripper_J": float(self._total_energy[2]),
            "total_energy_J": float(np.sum(self._total_energy)),
            "phase_time_s": (self._phase_steps * DT_CONTROL_S).tolist(),
            "phase_energy_arm_J": self._phase_energy[:, 0].tolist(),
            "phase_energy_base_J": self._phase_energy[:, 1].tolist(),
            "phase_energy_gripper_J": self._phase_energy[:, 2].tolist(),
            "success_place": self._termination_reason == "stable_place",
            "termination_reason": self._termination_reason,
            "stable_place_step": self._stable_place_step,
            "stable_place_count": self._stable_count,
            "cube_goal_dist_xy_m": (self._last_placement.cube_goal_dist_xy_m
                                    if self._last_placement is not None else None),
            "cte_rms_m": rms("cte_signed_m"),
            "cte_signed_mean_m": float(np.mean(signed)) if signed.size else None,
            "distance_to_route_rms_m": rms("distance_to_route_m"),
            "yaw_rms_rad": rms("heading_error_rad"),
            "tracking_samples": int(np.count_nonzero(valid)),
            "native_deadline_first_step": self._native_deadline_first_step,
            "placement_after_native_deadline": bool(
                self._stable_place_step is not None
                and self._native_deadline_first_step is not None
                and self._stable_place_step >= self._native_deadline_first_step),
            "measurement_wall_time_s": self._measurement_wall_time_s,
        }


PHYSICAL_MODEL_FIELDS = (
    "body_mass", "body_inertia", "body_pos", "body_quat", "body_ipos", "body_iquat",
    "geom_pos", "geom_quat", "geom_size", "geom_friction", "geom_solref", "geom_solimp",
    "dof_damping", "dof_armature", "dof_frictionloss", "jnt_range", "jnt_stiffness",
    "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange", "actuator_forcerange",
    "actuator_gear", "eq_data",
)


def physical_snapshot(model, data) -> dict[str, Any]:
    """Read-only JSON snapshot INCLUDING control and solver integration memory.

    This complements, but does not replace, XML/source hashes: model topology,
    assets and arbitrary user/plugin callbacks are not fully encoded here.
    """
    state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC), dtype=np.float64)
    mujoco.mj_getState(model, data, state, STATE_SPEC)
    if not np.isfinite(state).all():
        raise ValueError("Non-finite physical integration state")
    data_fields = ("qpos", "qvel", "act", "ctrl", "qacc_warmstart", "qfrc_applied",
                   "xfrc_applied", "mocap_pos", "mocap_quat", "eq_active", "userdata")
    return {
        "mujoco_version": mujoco.__version__,
        "integration_state_spec": int(STATE_SPEC),
        "integration_state": state.tolist(),
        "time_s": float(data.time),
        "data": {name: np.asarray(getattr(data, name)).tolist() for name in data_fields},
        "model": {name: np.asarray(getattr(model, name)).tolist() for name in PHYSICAL_MODEL_FIELDS},
        "options": {name: (np.asarray(getattr(model.opt, name)).tolist()) for name in
                    ("timestep", "gravity", "wind", "density", "viscosity", "integrator",
                     "solver", "cone", "iterations", "tolerance", "disableflags", "enableflags")},
    }


def compare_physical_snapshots(first: dict, second: dict, *, atol: float = 1e-8) -> dict[str, Any]:
    """Report all numerical differences; mismatched fields/shapes fail closed."""
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative")
    differences: dict[str, float | None] = {}

    def compare(a, b, prefix):
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                differences[prefix + ".keys"] = None
            for key in sorted(set(a) & set(b)):
                compare(a[key], b[key], f"{prefix}.{key}" if prefix else key)
        elif isinstance(a, str) or isinstance(b, str):
            differences[prefix] = 0.0 if a == b else None
        else:
            aa, bb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
            if aa.shape != bb.shape or not np.isfinite(aa).all() or not np.isfinite(bb).all():
                differences[prefix] = None
            else:
                differences[prefix] = float(np.max(np.abs(aa - bb))) if aa.size else 0.0

    compare(first, second, "")
    return {"matched": all(value is not None and value <= atol for value in differences.values()),
            "absolute_tolerance": float(atol), "max_abs_difference": differences}
