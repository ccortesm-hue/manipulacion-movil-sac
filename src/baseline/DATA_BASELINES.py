from __future__ import annotations
import os
import sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "glfw" if "--viewer" in sys.argv else "egl")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vx_compact_matplotlib")
RESOURCE_ROOT = Path(os.environ.get("VX_RESOURCE_ROOT", str(Path(__file__).resolve().parent.parent))).resolve()
import hashlib
def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
import csv
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
LATERAL_ACCEPTANCE_LIMIT_M = 0.020
ORIENTATION_ACCEPTANCE_LIMIT_RAD = float(np.deg2rad(3.0))
SCHEMA_VERSION = 7
TRACKING_COLUMNS = (
    "tracking_time_s",
    "tracking_error_m",
    "signed_tracking_error_m",
    "orientation_error_rad",
    "path_heading_error_rad",
    "path_heading_rad",
    "guidance_orientation_error_rad",
    "guidance_heading_rad",
    "geometric_orientation_error_rad",
    "smooth_route_orientation_error_rad",
    "base_yaw_rad",
    "smooth_path_yaw_rad",
    "omega_requested_rad_s",
    "omega_pid_reference_rad_s",
    "omega_measured_rad_s",
    "wheel_saturation_fraction",
    "waypoint_orientation_error_rad",
    "chassis_x_m",
    "chassis_y_m",
    "waypoint_reference_x_m",
    "waypoint_reference_y_m",
    "waypoint_heading_rad",
)
def finite(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)
def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value
def scalar_items(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        str(key): value
        for key, value in row.items()
        if not isinstance(value, (dict, list, tuple, np.ndarray))
    }
def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in scalar_items(row):
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            flat = scalar_items(row)
            writer.writerow({key: flat.get(key) for key in keys})
def zero_phase_fir(
    time_s: Sequence[float], values: Sequence[float], window_s: float = 0.25
) -> np.ndarray:
    t = np.asarray(time_s, dtype=np.float64)
    x = np.asarray(values, dtype=np.float64)
    if len(x) < 5 or len(t) != len(x):
        return x.copy()
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0.0:
        return x.copy()
    samples = max(5, int(round(window_s / dt)))
    samples += int(samples % 2 == 0)
    samples = min(samples, len(x) if len(x) % 2 else len(x) - 1)
    if samples < 5:
        return x.copy()
    kernel = np.hanning(samples)
    kernel /= float(np.sum(kernel))
    half = samples // 2
    return np.convolve(np.pad(x, (half, half), mode="reflect"), kernel, mode="valid")
def signal_stats(values: Sequence[float], *, absolute: bool = False) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if absolute:
        array = np.abs(array)
    if not len(array):
        return {"rms": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "rms": float(np.sqrt(np.mean(np.square(array)))),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }
def enrich_episode_from_trace(episode: Dict[str, Any], trace: Mapping[str, Any]) -> None:
    cte_signed = np.asarray(
        trace.get("signed_tracking_error_m", []), dtype=np.float64
    )
    cte_abs = np.abs(cte_signed)
    yaw_route = np.asarray(
        trace.get(
            "path_heading_error_rad",
            trace.get("smooth_route_orientation_error_rad", trace.get("orientation_error_rad", [])),
        ),
        dtype=np.float64,
    )
    yaw_controller = np.asarray(
        trace.get("guidance_orientation_error_rad", trace.get("orientation_error_rad", [])),
        dtype=np.float64,
    )
    cte = signal_stats(cte_abs, absolute=True)
    yaw_ctrl = signal_stats(yaw_controller, absolute=True)
    yaw_smooth = signal_stats(yaw_route, absolute=True)
    episode.update(
        {
            "tracking_samples": int(len(cte_abs)),
            "cte_rms_m": cte["rms"],
            "cte_mean_abs_m": cte["mean"],
            "cte_p95_m": cte["p95"],
            "cte_max_m": cte["max"],
            "cte_signed_mean_m": float(np.mean(cte_signed)) if len(cte_signed) else 0.0,
            "orientation_controller_rms_rad": yaw_ctrl["rms"],
            "orientation_controller_mean_abs_rad": yaw_ctrl["mean"],
            "orientation_controller_p95_rad": yaw_ctrl["p95"],
            "orientation_controller_max_rad": yaw_ctrl["max"],
            "orientation_route_rms_rad": yaw_smooth["rms"],
            "orientation_route_mean_abs_rad": yaw_smooth["mean"],
            "orientation_route_p95_rad": yaw_smooth["p95"],
            "orientation_route_max_rad": yaw_smooth["max"],
        }
    )
    path = np.asarray(trace.get("planned_path_xy", []), dtype=np.float64)
    chassis_x = np.asarray(trace.get("chassis_x_m", []), dtype=np.float64)
    chassis_y = np.asarray(trace.get("chassis_y_m", []), dtype=np.float64)
    chassis_yaw = np.asarray(trace.get("base_yaw_rad", []), dtype=np.float64)
    if (
        path.ndim == 2 and path.shape[1:] == (2,) and len(path) >= 2
        and len(chassis_x) and len(chassis_y) and len(chassis_yaw)
    ):
        point = np.array([chassis_x[-1], chassis_y[-1]], dtype=np.float64)
        segments = np.diff(path, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        best_distance = math.inf
        best_arc = 0.0
        for index, segment in enumerate(segments):
            length2 = float(np.dot(segment, segment))
            ratio = 0.0 if length2 <= 1.0e-12 else float(
                np.clip(np.dot(point - path[index], segment) / length2, 0.0, 1.0)
            )
            projection = path[index] + ratio * segment
            distance = float(np.linalg.norm(point - projection))
            if distance < best_distance:
                best_distance = distance
                best_arc = float(cumulative[index] + ratio * lengths[index])
        total_length = float(cumulative[-1])
        terminal_heading = float(np.arctan2(segments[-1, 1], segments[-1, 0]))
        terminal_error = float(
            (terminal_heading - float(chassis_yaw[-1]) + np.pi) % (2.0 * np.pi) - np.pi
        )
        episode["route_completion_fraction"] = float(
            np.clip(best_arc / max(total_length, 1.0e-12), 0.0, 1.0)
        )
        episode["endpoint_position_error_m"] = float(np.linalg.norm(point - path[-1]))
        episode["terminal_heading_error_rad"] = terminal_error
    else:
        episode["route_completion_fraction"] = math.nan
        episode["endpoint_position_error_m"] = math.nan
        episode["terminal_heading_error_rad"] = math.nan
    time_s = np.asarray(trace.get("tracking_time_s", []), dtype=np.float64)
    if len(time_s) == len(cte_signed) and len(time_s) >= 5:
        ripple = cte_signed - zero_phase_fir(time_s, cte_signed)
        episode["cte_ripple_rms_m"] = float(np.sqrt(np.mean(np.square(ripple))))
    else:
        episode["cte_ripple_rms_m"] = 0.0
    if len(time_s) == len(yaw_route) and len(time_s) >= 5:
        ripple = yaw_route - zero_phase_fir(time_s, yaw_route)
        episode["orientation_route_ripple_rms_rad"] = float(
            np.sqrt(np.mean(np.square(ripple)))
        )
    else:
        episode["orientation_route_ripple_rms_rad"] = 0.0
    saturation = np.asarray(trace.get("wheel_saturation_fraction", []), dtype=np.float64)
    episode["wheel_saturation_mean"] = (
        float(np.mean(saturation)) if len(saturation) else 0.0
    )
    episode["wheel_any_saturation_fraction"] = (
        float(np.mean(saturation > 0.0)) if len(saturation) else 0.0
    )
def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA foreign_keys=ON;

        CREATE TABLE run_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE episodes (
            episode_id INTEGER PRIMARY KEY,
            method TEXT NOT NULL,
            episode INTEGER NOT NULL,
            map_id INTEGER NOT NULL,
            mass_kg REAL NOT NULL,
            repetition INTEGER NOT NULL,
            seed INTEGER,
            phase0_success INTEGER NOT NULL,
            phase1_success INTEGER NOT NULL,
            phase2_success INTEGER NOT NULL,
            return_home_success INTEGER NOT NULL,
            success_overall INTEGER NOT NULL,
            timeout INTEGER NOT NULL,
            cube_dropped INTEGER NOT NULL,
            reward REAL,
            total_steps INTEGER,
            duration_s REAL NOT NULL,
            energy_total_j REAL NOT NULL,
            energy_arm_j REAL NOT NULL,
            energy_base_j REAL NOT NULL,
            cube_goal_xy_error_m REAL,
            base_goal_distance_m REAL,
            stop_distance_error_m REAL,
            base_goal_heading_error_rad REAL,
            cube_setpoint_rms_m REAL,
            cube_setpoint_max_m REAL,
            route_completion_fraction REAL,
            endpoint_position_error_m REAL,
            terminal_heading_error_rad REAL,
            tracking_samples INTEGER NOT NULL,
            cte_rms_m REAL,
            cte_mean_abs_m REAL,
            cte_p95_m REAL,
            cte_max_m REAL,
            cte_signed_mean_m REAL,
            orientation_controller_rms_rad REAL,
            orientation_controller_mean_abs_rad REAL,
            orientation_controller_p95_rad REAL,
            orientation_controller_max_rad REAL,
            orientation_route_rms_rad REAL,
            orientation_route_mean_abs_rad REAL,
            orientation_route_p95_rad REAL,
            orientation_route_max_rad REAL,
            cte_ripple_rms_m REAL,
            orientation_route_ripple_rms_rad REAL,
            wheel_saturation_mean REAL,
            wheel_any_saturation_fraction REAL,
            action_saturation_global REAL,
            metrics_json TEXT NOT NULL
        );

        CREATE TABLE tracking_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id INTEGER NOT NULL,
            sample_idx INTEGER NOT NULL,
            t_s REAL NOT NULL,
            cte_abs_m REAL,
            cte_signed_m REAL,
            orientation_error_rad REAL,
            path_heading_error_rad REAL,
            path_heading_rad REAL,
            guidance_orientation_error_rad REAL,
            guidance_heading_rad REAL,
            geometric_orientation_error_rad REAL,
            smooth_orientation_error_rad REAL,
            base_yaw_rad REAL,
            smooth_path_yaw_rad REAL,
            omega_requested_rad_s REAL,
            omega_pid_reference_rad_s REAL,
            omega_measured_rad_s REAL,
            wheel_saturation_fraction REAL,
            waypoint_orientation_error_rad REAL,
            chassis_x_m REAL,
            chassis_y_m REAL,
            waypoint_reference_x_m REAL,
            waypoint_reference_y_m REAL,
            waypoint_heading_rad REAL,
            FOREIGN KEY(episode_id) REFERENCES episodes(episode_id)
        );

        CREATE TABLE energy_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id INTEGER NOT NULL,
            sample_idx INTEGER NOT NULL,
            t_s REAL NOT NULL,
            phase INTEGER,
            energy_total_step_j REAL,
            energy_arm_step_j REAL,
            energy_base_step_j REAL,
            power_total_w REAL,
            FOREIGN KEY(episode_id) REFERENCES episodes(episode_id)
        );

        CREATE TABLE cube_setpoint_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id INTEGER NOT NULL,
            sample_idx INTEGER NOT NULL,
            t_s REAL NOT NULL,
            error_m REAL NOT NULL,
            FOREIGN KEY(episode_id) REFERENCES episodes(episode_id)
        );

        CREATE TABLE trajectory_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id INTEGER NOT NULL,
            sample_idx INTEGER NOT NULL,
            t_s REAL NOT NULL,
            base_x_m REAL,
            base_y_m REAL,
            cube_x_m REAL,
            cube_y_m REAL,
            cube_z_m REAL,
            FOREIGN KEY(episode_id) REFERENCES episodes(episode_id)
        );

        CREATE TABLE action_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id INTEGER NOT NULL,
            sample_idx INTEGER NOT NULL,
            t_s REAL NOT NULL,
            phase INTEGER,
            joint_1 REAL, joint_2 REAL, joint_3 REAL, joint_4 REAL, joint_5 REAL,
            base_linear REAL, base_angular REAL,
            saturation_global REAL,
            FOREIGN KEY(episode_id) REFERENCES episodes(episode_id)
        );

        CREATE TABLE planned_paths (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id INTEGER NOT NULL,
            waypoint_idx INTEGER NOT NULL,
            x_m REAL NOT NULL,
            y_m REAL NOT NULL,
            FOREIGN KEY(episode_id) REFERENCES episodes(episode_id)
        );

        CREATE INDEX idx_episode_map_mass ON episodes(map_id, mass_kg);
        CREATE INDEX idx_tracking_episode ON tracking_samples(episode_id);
        CREATE INDEX idx_energy_episode ON energy_samples(episode_id);
        CREATE INDEX idx_cube_setpoint_episode ON cube_setpoint_samples(episode_id);
        CREATE INDEX idx_trajectory_episode ON trajectory_samples(episode_id);
        CREATE INDEX idx_action_episode ON action_samples(episode_id);
        """
    )
def _trace_value(trace: Mapping[str, Any], key: str, index: int) -> Any:
    values = trace.get(key, [])
    return values[index] if index < len(values) else None
def _insert_episode(
    connection: sqlite3.Connection,
    method: str,
    episode_id: int,
    episode: Mapping[str, Any],
) -> None:
    columns = (
        "method", "episode", "map_id", "mass_kg", "repetition", "seed",
        "phase0_success", "phase1_success", "phase2_success",
        "return_home_success", "success_overall", "timeout", "cube_dropped",
        "reward", "total_steps", "duration_s", "energy_total_j", "energy_arm_j",
        "energy_base_j", "cube_goal_xy_error_m", "base_goal_distance_m",
        "stop_distance_error_m", "base_goal_heading_error_rad", "cube_setpoint_rms_m",
        "cube_setpoint_max_m", "route_completion_fraction",
        "endpoint_position_error_m", "terminal_heading_error_rad", "tracking_samples",
        "cte_rms_m", "cte_mean_abs_m", "cte_p95_m", "cte_max_m",
        "cte_signed_mean_m", "orientation_controller_rms_rad",
        "orientation_controller_mean_abs_rad", "orientation_controller_p95_rad",
        "orientation_controller_max_rad", "orientation_route_rms_rad",
        "orientation_route_mean_abs_rad", "orientation_route_p95_rad",
        "orientation_route_max_rad", "cte_ripple_rms_m",
        "orientation_route_ripple_rms_rad", "wheel_saturation_mean",
        "wheel_any_saturation_fraction", "action_saturation_global",
    )
    boolean_columns = {
        "phase0_success", "phase1_success", "phase2_success", "return_home_success",
        "success_overall", "timeout", "cube_dropped",
    }
    defaults = {
        "episode": episode_id,
        "repetition": 1,
        "seed": None,
        "reward": None,
        "total_steps": None,
        "duration_s": 0.0,
        "energy_total_j": 0.0,
        "energy_arm_j": 0.0,
        "energy_base_j": 0.0,
        "tracking_samples": 0,
        "action_saturation_global": 0.0,
    }
    values = []
    for column in columns:
        if column == "method":
            value = method
        else:
            value = episode.get(column, defaults.get(column))
        if column in boolean_columns:
            value = int(bool(value))
        values.append(value)
    placeholders = ",".join("?" for _ in columns)
    connection.execute(
        f"INSERT INTO episodes (episode_id,{','.join(columns)},metrics_json) "
        f"VALUES (?,{placeholders},?)",
        [episode_id, *values, json.dumps(json_safe(episode), ensure_ascii=False)],
    )
def _insert_trace(
    connection: sqlite3.Connection,
    episode_id: int,
    trace: Mapping[str, Any],
) -> None:
    n_tracking = len(trace.get("tracking_time_s", []))
    tracking_rows = []
    keys = TRACKING_COLUMNS[1:]
    for index in range(n_tracking):
        tracking_rows.append(
            (
                episode_id,
                index,
                finite(_trace_value(trace, "tracking_time_s", index), 0.0),
                *[finite(_trace_value(trace, key, index)) for key in keys],
            )
        )
    if tracking_rows:
        connection.executemany(
            """INSERT INTO tracking_samples (
                episode_id,sample_idx,t_s,cte_abs_m,cte_signed_m,
                orientation_error_rad,path_heading_error_rad,
                path_heading_rad,guidance_orientation_error_rad,guidance_heading_rad,
                geometric_orientation_error_rad,
                smooth_orientation_error_rad,base_yaw_rad,smooth_path_yaw_rad,
                omega_requested_rad_s,omega_pid_reference_rad_s,
                omega_measured_rad_s,wheel_saturation_fraction,
                waypoint_orientation_error_rad,chassis_x_m,chassis_y_m,
                waypoint_reference_x_m,waypoint_reference_y_m,waypoint_heading_rad
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            tracking_rows,
        )
    energy_time = trace.get("energy_time_s", [])
    if energy_time:
        rows = []
        for index in range(len(energy_time)):
            rows.append(
                (
                    episode_id,
                    index,
                    finite(energy_time[index], 0.0),
                    _trace_value(trace, "energy_phase", index),
                    finite(_trace_value(trace, "energy_total_step_j", index)),
                    finite(_trace_value(trace, "energy_arm_step_j", index)),
                    finite(_trace_value(trace, "energy_base_step_j", index)),
                    finite(_trace_value(trace, "power_total_w", index)),
                )
            )
        connection.executemany(
            """INSERT INTO energy_samples (
                episode_id,sample_idx,t_s,phase,energy_total_step_j,
                energy_arm_step_j,energy_base_step_j,power_total_w
            ) VALUES (?,?,?,?,?,?,?,?)""",
            rows,
        )
    cube_setpoint_time = trace.get("cube_setpoint_time_s", [])
    cube_setpoint_error = trace.get("cube_setpoint_error_m", [])
    if cube_setpoint_time and cube_setpoint_error:
        count = min(len(cube_setpoint_time), len(cube_setpoint_error))
        connection.executemany(
            """INSERT INTO cube_setpoint_samples
               (episode_id,sample_idx,t_s,error_m) VALUES (?,?,?,?)""",
            [
                (
                    episode_id,
                    index,
                    finite(cube_setpoint_time[index], 0.0),
                    finite(cube_setpoint_error[index], 0.0),
                )
                for index in range(count)
            ],
        )
    trajectory_time = trace.get("trajectory_time_s", [])
    if trajectory_time:
        rows = []
        for index in range(len(trajectory_time)):
            rows.append(
                (
                    episode_id,
                    index,
                    finite(trajectory_time[index], 0.0),
                    finite(_trace_value(trace, "base_x_m", index)),
                    finite(_trace_value(trace, "base_y_m", index)),
                    finite(_trace_value(trace, "cube_x_m", index)),
                    finite(_trace_value(trace, "cube_y_m", index)),
                    finite(_trace_value(trace, "cube_z_m", index)),
                )
            )
        connection.executemany(
            """INSERT INTO trajectory_samples (
                episode_id,sample_idx,t_s,base_x_m,base_y_m,cube_x_m,cube_y_m,cube_z_m
            ) VALUES (?,?,?,?,?,?,?,?)""",
            rows,
        )
    action_time = trace.get("action_time_s", [])
    actions = trace.get("actions", [])
    if action_time and actions:
        rows = []
        for index, action in enumerate(actions):
            vector = np.asarray(action, dtype=np.float64).reshape(-1)
            padded = np.full(7, np.nan, dtype=np.float64)
            padded[: min(7, len(vector))] = vector[:7]
            rows.append(
                (
                    episode_id,
                    index,
                    finite(_trace_value(trace, "action_time_s", index), 0.0),
                    _trace_value(trace, "action_phase", index),
                    *padded.tolist(),
                    finite(_trace_value(trace, "action_saturation", index)),
                )
            )
        connection.executemany(
            """INSERT INTO action_samples (
                episode_id,sample_idx,t_s,phase,joint_1,joint_2,joint_3,joint_4,
                joint_5,base_linear,base_angular,saturation_global
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
    path = trace.get("planned_path_xy", [])
    if path:
        connection.executemany(
            """INSERT INTO planned_paths (episode_id,waypoint_idx,x_m,y_m)
               VALUES (?,?,?,?)""",
            [
                (episode_id, index, float(point[0]), float(point[1]))
                for index, point in enumerate(path)
            ],
        )
def save_database(
    path: Path,
    method: str,
    metadata: Mapping[str, Any],
    episodes: Sequence[Dict[str, Any]],
    traces: Sequence[Mapping[str, Any]],
) -> None:
    if path.exists():
        raise FileExistsError(f"La base de datos ya existe: {path}")
    connection = sqlite3.connect(path)
    try:
        _create_schema(connection)
        run_metadata = {
            "schema_version": SCHEMA_VERSION,
            "method": method,
            **dict(metadata),
        }
        connection.executemany(
            "INSERT INTO run_metadata (key,value) VALUES (?,?)",
            [
                (str(key), json.dumps(json_safe(value), ensure_ascii=False))
                for key, value in run_metadata.items()
            ],
        )
        for index, (episode, trace) in enumerate(zip(episodes, traces), start=1):
            _insert_episode(connection, method, index, episode)
            _insert_trace(connection, index, trace)
        connection.commit()
    finally:
        connection.close()
def aggregate(episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    count = len(episodes)
    rates = {
        key: float(np.mean([bool(row.get(key, False)) for row in episodes]))
        if count
        else 0.0
        for key in (
            "phase0_success",
            "phase1_success",
            "phase2_success",
            "return_home_success",
            "success_overall",
        )
    }
    numeric = {}
    for key in (
        "reward",
        "duration_s",
        "energy_total_j",
        "energy_arm_j",
        "energy_base_j",
        "cube_setpoint_rms_m",
        "cube_setpoint_max_m",
        "route_completion_fraction",
        "endpoint_position_error_m",
        "terminal_heading_error_rad",
        "cte_rms_m",
        "cte_p95_m",
        "orientation_route_rms_rad",
        "orientation_route_p95_rad",
        "action_saturation_global",
    ):
        values = np.asarray([finite(row.get(key)) for row in episodes], dtype=np.float64)
        values = values[np.isfinite(values)]
        numeric[key] = {
            "mean": float(np.mean(values)) if len(values) else None,
            "std": float(np.std(values)) if len(values) else None,
        }
    return {"episodes": count, "rates": rates, "metrics": numeric}
def save_episode_plots(
    output_dir: Path,
    episodes: Sequence[Mapping[str, Any]],
    traces: Sequence[Mapping[str, Any]],
) -> None:
    plot_dir = output_dir / "tracking_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    histogram_dir = output_dir / "tracking_histograms"
    histogram_dir.mkdir(parents=True, exist_ok=True)
    for episode, trace in zip(episodes, traces):
        time_s = np.asarray(trace.get("tracking_time_s", []), dtype=np.float64)
        cte = np.asarray(trace.get("signed_tracking_error_m", []), dtype=np.float64)
        position_error = np.abs(cte)
        chassis_x = np.asarray(trace.get("chassis_x_m", []), dtype=np.float64)
        chassis_y = np.asarray(trace.get("chassis_y_m", []), dtype=np.float64)
        ref_x = np.asarray(trace.get("waypoint_reference_x_m", []), dtype=np.float64)
        ref_y = np.asarray(trace.get("waypoint_reference_y_m", []), dtype=np.float64)
        chassis_yaw = np.asarray(trace.get("base_yaw_rad", []), dtype=np.float64)
        path_yaw = np.asarray(
            trace.get("path_heading_rad", trace.get("smooth_path_yaw_rad", [])),
            dtype=np.float64,
        )
        guidance_yaw = np.asarray(trace.get("guidance_heading_rad", []), dtype=np.float64)
        waypoint_yaw = np.asarray(trace.get("waypoint_heading_rad", []), dtype=np.float64)
        yaw_error = np.asarray(
            trace.get(
                "path_heading_error_rad",
                trace.get("smooth_route_orientation_error_rad", trace.get("orientation_error_rad", [])),
            ),
            dtype=np.float64,
        )
        guidance_error = np.asarray(
            trace.get("guidance_orientation_error_rad", trace.get("orientation_error_rad", [])),
            dtype=np.float64,
        )
        arrays = (
            position_error, cte, chassis_x, chassis_y, ref_x, ref_y,
            chassis_yaw, path_yaw, guidance_yaw, waypoint_yaw, yaw_error,
            guidance_error,
        )
        if not len(time_s) or any(len(values) != len(time_s) for values in arrays):
            continue
        path = np.asarray(trace.get("planned_path_xy", []), dtype=np.float64)
        yaw_chassis_unwrapped = np.unwrap(chassis_yaw)
        yaw_path_unwrapped = np.unwrap(path_yaw)
        yaw_guidance_unwrapped = np.unwrap(guidance_yaw)
        yaw_waypoint_unwrapped = np.unwrap(waypoint_yaw)
        if len(yaw_path_unwrapped):
            turns = np.round(
                (yaw_chassis_unwrapped[0] - yaw_path_unwrapped[0]) / (2.0 * np.pi)
            )
            yaw_path_unwrapped += float(turns) * 2.0 * np.pi
        if len(yaw_guidance_unwrapped):
            turns = np.round(
                (yaw_chassis_unwrapped[0] - yaw_guidance_unwrapped[0]) / (2.0 * np.pi)
            )
            yaw_guidance_unwrapped += float(turns) * 2.0 * np.pi
        if len(yaw_waypoint_unwrapped):
            turns = np.round(
                (yaw_chassis_unwrapped[0] - yaw_waypoint_unwrapped[0]) / (2.0 * np.pi)
            )
            yaw_waypoint_unwrapped += float(turns) * 2.0 * np.pi
        fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), constrained_layout=True)
        ax_xy, ax_cte, ax_yaw, ax_yaw_error = axes.ravel()
        if path.ndim == 2 and path.shape[1:] == (2,) and len(path) >= 2:
            ax_xy.plot(
                path[:, 0], path[:, 1], color="black", linewidth=1.2,
                marker=".", markersize=2.0, label="Waypoints / ruta",
            )
        ax_xy.plot(ref_x, ref_y, color="tab:orange", linewidth=1.0,
                   label="Proyeccion sobre waypoints")
        ax_xy.plot(chassis_x, chassis_y, color="tab:blue", linewidth=1.5,
                   label="chassis_frame medido")
        ax_xy.scatter(chassis_x[0], chassis_y[0], color="tab:green", s=28, zorder=4,
                      label="Inicio tracking")
        ax_xy.scatter(chassis_x[-1], chassis_y[-1], color="tab:red", s=28, zorder=4,
                      label="Fin tracking")
        ax_xy.set_xlabel("X mundo [m]")
        ax_xy.set_ylabel("Y mundo [m]")
        ax_xy.set_aspect("equal", adjustable="datalim")
        ax_xy.set_title("Posicion real de chassis_frame respecto a waypoints")
        ax_xy.legend(fontsize=7)
        ax_cte.plot(time_s, 100.0 * cte, color="tab:blue", linewidth=1.0,
                    label="CTE perpendicular firmado")
        ax_cte.plot(time_s, 100.0 * position_error, color="tab:orange", linewidth=0.8,
                    alpha=0.75, label="Magnitud |CTE|")
        ax_cte.axhline(0.0, color="black", linewidth=0.7)
        lateral_limit_cm = 100.0 * LATERAL_ACCEPTANCE_LIMIT_M
        ax_cte.axhspan(
            -lateral_limit_cm, lateral_limit_cm,
            color="tab:green", alpha=0.08, label="Banda aceptable +/-2 cm",
        )
        ax_cte.axhline(lateral_limit_cm, color="tab:red", linestyle="--", linewidth=0.9)
        ax_cte.axhline(-lateral_limit_cm, color="tab:red", linestyle="--", linewidth=0.9)
        ax_cte.set_xlabel("Tiempo de seguimiento [s]")
        ax_cte.set_ylabel("Error de posicion [cm]")
        ax_cte.set_title("Error lateral chassis_frame-ruta sin filtrado")
        ax_cte.legend(fontsize=7)
        ax_yaw.plot(time_s, np.rad2deg(yaw_chassis_unwrapped), color="tab:blue",
                    linewidth=1.1, label="Yaw real chassis_frame")
        ax_yaw.plot(time_s, np.rad2deg(yaw_path_unwrapped), color="tab:orange",
                    linewidth=1.0, label="Tangente continua de la ruta")
        ax_yaw.set_xlabel("Tiempo de seguimiento [s]")
        ax_yaw.set_ylabel("Orientacion desenvolvida [deg]")
        ax_yaw.set_title("Orientacion real respecto a la tangente de la ruta")
        ax_yaw.legend(fontsize=7)
        ax_yaw_error.plot(time_s, np.rad2deg(yaw_error), color="tab:purple",
                          linewidth=1.0,
                          label="tangente de ruta - yaw chassis_frame")
        ax_yaw_error.axhline(0.0, color="black", linewidth=0.7)
        orientation_limit_deg = float(np.rad2deg(ORIENTATION_ACCEPTANCE_LIMIT_RAD))
        ax_yaw_error.axhspan(
            -orientation_limit_deg, orientation_limit_deg,
            color="tab:green", alpha=0.08, label="Banda aceptable +/-3 deg",
        )
        ax_yaw_error.axhline(
            orientation_limit_deg, color="tab:red", linestyle="--", linewidth=0.9
        )
        ax_yaw_error.axhline(
            -orientation_limit_deg, color="tab:red", linestyle="--", linewidth=0.9
        )
        ax_yaw_error.set_xlabel("Tiempo de seguimiento [s]")
        ax_yaw_error.set_ylabel("Error de orientacion real [deg]")
        ax_yaw_error.set_title("Error de orientacion de trayectoria sin filtrado")
        ax_yaw_error.legend(fontsize=7)
        for axis in axes.ravel():
            axis.grid(alpha=0.25)
        fig.suptitle(
            f"Mapa {episode['map_id']} | masa {episode['mass_kg']:g} kg | "
            f"episodio {episode['episode']} | medicion: site chassis_frame"
        )
        fig.savefig(
            plot_dir
            / f"ep{int(episode['episode']):03d}_map{int(episode['map_id'])}_"
              f"mass{float(episode['mass_kg']):.2f}.png",
            dpi=170,
        )
        plt.close(fig)
        cte_cm = 100.0 * cte[np.isfinite(cte)]
        yaw_deg = np.rad2deg(yaw_error[np.isfinite(yaw_error)])
        hist_fig, hist_axes = plt.subplots(
            1, 2, figsize=(11.5, 4.5), constrained_layout=True
        )
        for axis, values, color, xlabel, title in (
            (
                hist_axes[0], cte_cm, "tab:blue", "CTE perpendicular [cm]",
                "Distribucion del error lateral",
            ),
            (
                hist_axes[1], yaw_deg, "tab:purple", "Error tangencial [deg]",
                "Distribucion del error de orientacion",
            ),
        ):
            axis.hist(values, bins=40, density=True, color=color, alpha=0.78)
            axis.axvline(0.0, color="black", linewidth=0.8)
            limit = lateral_limit_cm if axis is hist_axes[0] else orientation_limit_deg
            axis.axvspan(-limit, limit, color="tab:green", alpha=0.08)
            axis.axvline(limit, color="tab:red", linestyle="--", linewidth=0.9)
            axis.axvline(-limit, color="tab:red", linestyle="--", linewidth=0.9)
            if len(values):
                rms = float(np.sqrt(np.mean(np.square(values))))
                p95 = float(np.percentile(np.abs(values), 95.0))
                mean = float(np.mean(values))
                axis.text(
                    0.98, 0.96,
                    f"media={mean:.2f}\nRMS={rms:.2f}\nP95 abs={p95:.2f}",
                    transform=axis.transAxes, ha="right", va="top", fontsize=8,
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
                )
            axis.set_xlabel(xlabel)
            axis.set_ylabel("Densidad")
            axis.set_title(title)
            axis.grid(axis="y", alpha=0.25)
        hist_fig.suptitle(
            f"Mapa {episode['map_id']} | masa {episode['mass_kg']:g} kg | "
            f"episodio {episode['episode']} | muestras crudas"
        )
        hist_fig.savefig(
            histogram_dir
            / f"ep{int(episode['episode']):03d}_map{int(episode['map_id'])}_"
              f"mass{float(episode['mass_kg']):.2f}_hist.png",
            dpi=170,
        )
        plt.close(hist_fig)
def save_overview_plot(output_dir: Path, episodes: Sequence[Mapping[str, Any]]) -> None:
    maps = sorted({int(row["map_id"]) for row in episodes})
    if not maps:
        return
    cte_rms = []
    cte_p95 = []
    cte_max = []
    yaw_rms = []
    yaw_p95 = []
    yaw_max = []
    success = []
    completion = []
    duration = []
    energy = []
    for map_id in maps:
        rows = [row for row in episodes if int(row["map_id"]) == map_id]
        cte_rms.append(np.mean([finite(row.get("cte_rms_m"), 0.0) for row in rows]))
        cte_p95.append(np.mean([finite(row.get("cte_p95_m"), 0.0) for row in rows]))
        cte_max.append(np.max([finite(row.get("cte_max_m"), 0.0) for row in rows]))
        yaw_rms.append(
            np.mean([finite(row.get("orientation_route_rms_rad"), 0.0) for row in rows])
        )
        yaw_p95.append(
            np.mean([finite(row.get("orientation_route_p95_rad"), 0.0) for row in rows])
        )
        yaw_max.append(
            np.max([finite(row.get("orientation_route_max_rad"), 0.0) for row in rows])
        )
        success.append(100.0 * np.mean([bool(row.get("phase2_success")) for row in rows]))
        completion.append(
            100.0 * np.mean([
                finite(row.get("route_completion_fraction"), 0.0) for row in rows
            ])
        )
        duration.append(np.mean([finite(row.get("duration_s"), 0.0) for row in rows]))
        energy.append(np.mean([finite(row.get("energy_total_j"), 0.0) for row in rows]))
    x = np.arange(len(maps), dtype=np.float64)
    width = 0.25
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    axes[0, 0].bar(x - width, 100.0 * np.asarray(cte_rms), width, label="RMS")
    axes[0, 0].bar(x, 100.0 * np.asarray(cte_p95), width, label="P95")
    axes[0, 0].bar(x + width, 100.0 * np.asarray(cte_max), width, label="Maximo")
    axes[0, 0].set_ylabel("Error lateral [cm]")
    axes[0, 0].set_title("Seguimiento de trayectoria")
    axes[0, 0].axhline(
        100.0 * LATERAL_ACCEPTANCE_LIMIT_M,
        color="tab:red", linestyle="--", linewidth=1.0, label="Limite 2 cm",
    )
    axes[0, 1].bar(x - width, np.rad2deg(yaw_rms), width, label="RMS")
    axes[0, 1].bar(x, np.rad2deg(yaw_p95), width, label="P95")
    axes[0, 1].bar(x + width, np.rad2deg(yaw_max), width, label="Maximo")
    axes[0, 1].set_ylabel("Error de orientacion [deg]")
    axes[0, 1].set_title("Orientacion respecto a la tangente de ruta")
    axes[0, 1].axhline(
        np.rad2deg(ORIENTATION_ACCEPTANCE_LIMIT_RAD),
        color="tab:red", linestyle="--", linewidth=1.0, label="Limite 3 deg",
    )
    axes[1, 0].bar(
        x - width / 2, success, width, color="#59a14f", label="Exito P2"
    )
    axes[1, 0].bar(
        x + width / 2, completion, width, color="#4c78a8", label="Ruta recorrida"
    )
    axes[1, 0].set_ylim(0.0, 105.0)
    axes[1, 0].set_ylabel("Exito P2 [%]")
    axes[1, 0].set_title("Exito y completitud de ruta")
    axes[1, 0].legend()
    energy_axis = axes[1, 1].twinx()
    axes[1, 1].bar(x - width / 2, duration, width, label="Tiempo", color="#4c78a8")
    energy_axis.bar(x + width / 2, energy, width, label="Energia", color="#f28e2b")
    axes[1, 1].set_ylabel("Tiempo [s]")
    energy_axis.set_ylabel("Energia [J]")
    axes[1, 1].set_title("Tiempo y energia")
    for axis in axes.flat:
        axis.set_xticks(x, [str(value) for value in maps])
        axis.set_xlabel("Mapa")
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend()
    axes[0, 1].legend()
    fig.savefig(output_dir / "metrics_overview.png", dpi=180)
    plt.close(fig)
def save_dataset(
    *,
    output_dir: Path,
    method: str,
    metadata: Mapping[str, Any],
    episodes: List[Dict[str, Any]],
    traces: List[Dict[str, Any]],
    plots: bool = True,
) -> Dict[str, Any]:
    if len(episodes) != len(traces):
        raise ValueError(
            f"Episodios y trazas no coinciden: {len(episodes)} != {len(traces)}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    for episode, trace in zip(episodes, traces):
        enrich_episode_from_trace(episode, trace)
    summary = aggregate(episodes)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "metadata": dict(metadata),
        "summary": summary,
        "episodes": episodes,
    }
    write_csv(output_dir / "episodes.csv", episodes)
    (output_dir / "summary.json").write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "traces.json").write_text(
        json.dumps(json_safe(traces), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    save_database(output_dir / "dataset.sqlite", method, metadata, episodes, traces)
    if plots:
        save_episode_plots(output_dir, episodes, traces)
        save_overview_plot(output_dir, episodes)
    return summary
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/zenno_data_baselines_matplotlib")
import numpy as np
HERE = RESOURCE_ROOT
ROUTES_DIR = HERE / "RUTAS_SAC"
OUTPUT_ROOT = HERE / "DATA_BASELINES_RESULTS"
VALID_MAPS = (1, 2, 3, 4)
DEFAULT_MASSES = (0.50, 0.75, 1.00, 1.25, 1.50)
PROGRAM_NAME = "DATA_BASELINES.py"
DEFAULT_SEED = 20_260_711
OUTPUT_GUARD_ROOT: Optional[Path] = None
def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("el valor debe ser >= 1")
    return number
def nonnegative_float(value: str) -> float:
    number = float(value)
    if number < 0.0:
        raise argparse.ArgumentTypeError("el valor debe ser >= 0")
    return number
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera datos del baseline clasico con los controladores actuales.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--maps", nargs="+", type=int, default=list(VALID_MAPS))
    parser.add_argument("--masses", nargs="+", type=float, default=list(DEFAULT_MASSES))
    parser.add_argument("--repetitions", "--reps", type=positive_int, default=None)
    parser.add_argument(
        "--profile",
        choices=("nominal", "robust"),
        default="nominal",
        help="nominal: estados fijos; robust: perturbaciones del cubo.",
    )
    parser.add_argument("--cube-xy-noise", type=nonnegative_float, default=None)
    parser.add_argument("--cube-yaw-noise-deg", type=nonnegative_float, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--routes-dir", type=Path, default=ROUTES_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-steps", type=positive_int, default=20_000)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser
def resolve_profile(args: argparse.Namespace) -> Tuple[int, float, float]:
    robust = args.profile == "robust"
    repetitions = args.repetitions if args.repetitions is not None else (5 if robust else 1)
    xy_noise = args.cube_xy_noise
    if xy_noise is None:
        xy_noise = 0.010 if robust else 0.0
    yaw_noise_deg = args.cube_yaw_noise_deg
    if yaw_noise_deg is None:
        yaw_noise_deg = 5.0 if robust else 0.0
    return int(repetitions), float(xy_noise), float(yaw_noise_deg)
def validate_cases(args: argparse.Namespace) -> Tuple[List[int], List[float]]:
    maps = sorted({int(value) for value in args.maps})
    masses = sorted({float(value) for value in args.masses})
    invalid = sorted(set(maps) - set(VALID_MAPS))
    if not maps or invalid:
        raise ValueError(f"Mapas invalidos: {invalid}; validos={VALID_MAPS}")
    if not masses or any(value <= 0.0 for value in masses):
        raise ValueError("Todas las masas deben ser positivas.")
    return maps, masses
def default_output(profile: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_ROOT / f"{stamp}_{profile}"
def route_for_map(routes_dir: Path, map_id: int) -> List[List[float]]:
    payload = json.loads((routes_dir / f"map_{map_id}.json").read_text(encoding="utf-8"))
    points = np.asarray(payload.get("path", []), dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 2:
        raise ValueError(f"Ruta invalida para mapa {map_id}")
    return points.tolist()
def canonicalize(
    raw_results: Sequence[Dict[str, Any]],
    raw_traces: Sequence[Dict[str, Any]],
    routes_dir: Path,
    repetitions: int,
    seed: int,
    maps: Sequence[int],
    masses: Sequence[float],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    episodes: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    counters: Dict[Tuple[int, float], int] = {}
    for index, (result, trace) in enumerate(zip(raw_results, raw_traces), start=1):
        map_id = int(result["map_id"])
        mass = float(result["cube_mass_kg"])
        key = (map_id, mass)
        counters[key] = counters.get(key, 0) + 1
        repetition = counters[key]
        map_index = list(maps).index(map_id)
        mass_index = list(masses).index(mass)
        episode_seed = int(
            seed + map_id * 100_000 + mass_index * 1_000 + repetition - 1
        )
        phase_steps = dict(result.get("phase_steps", {}))
        total_steps = int(round(float(result["total_time_s"]) / 0.010))
        episode = {
            "episode": index,
            "map_id": map_id,
            "mass_kg": mass,
            "repetition": repetition,
            "seed": episode_seed,
            "phase0_success": bool(result.get("success_pick", False)),
            "phase1_success": bool(result.get("success_transport", False)),
            "phase2_success": bool(result.get("success_place", False)),
            "return_home_success": bool(result.get("success_return_home", False)),
            "success_overall": bool(result.get("success_overall", False)),
            "timeout": bool(result.get("timeout", False)),
            "cube_dropped": False,
            "reward": None,
            "total_steps": total_steps,
            "duration_s": float(result["total_time_s"]),
            "energy_total_j": float(result.get("energy_total_j", 0.0)),
            "energy_arm_j": float(result.get("energy_arm_j", 0.0)),
            "energy_base_j": float(result.get("energy_base_j", 0.0)),
            "cube_goal_xy_error_m": float(result.get("cube_goal_distance_xy_m", np.nan)),
            "base_goal_distance_m": float(result.get("base_goal_distance_m", np.nan)),
            "stop_distance_error_m": float(result.get("stop_distance_error_m", np.nan)),
            "base_goal_heading_error_rad": float(
                result.get("base_goal_heading_error_rad", np.nan)
            ),
            "cube_setpoint_rms_m": float(result.get("cube_setpoint_rms_m", 0.0)),
            "cube_setpoint_max_m": float(result.get("cube_setpoint_max_m", 0.0)),
            "action_saturation_global": 0.0,
            "controller": (
                "PID_PSO_6D_MODEL_DYNAMICS_STRICT_20MM_3DEG+pure_pursuit"
            ),
            "arm_controller": "IK_DLS_PSO_10MS",
            "phase_steps": phase_steps,
            "nav_mode_steps": dict(result.get("nav_mode_steps", {})),
            "controller_compute_p99_us": float(
                result.get("controller_compute_p99_us", 0.0)
            ),
        }
        canonical_trace = dict(trace)
        canonical_trace["episode"] = index
        canonical_trace["map_id"] = map_id
        canonical_trace["mass_kg"] = mass
        canonical_trace["planned_path_xy"] = route_for_map(routes_dir, map_id)
        episodes.append(episode)
        traces.append(canonical_trace)
    expected = len(counters) * repetitions
    if len(episodes) != expected:
        raise RuntimeError(
            f"Cobertura incompleta: episodios={len(episodes)}, esperados={expected}."
        )
    return episodes, traces
def _extract_baseline_main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    maps, masses = validate_cases(args)
    repetitions, xy_noise, yaw_noise_deg = resolve_profile(args)
    routes_dir = args.routes_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else default_output(args.profile)
    )
    if OUTPUT_GUARD_ROOT is not None:
        guard_root = OUTPUT_GUARD_ROOT.expanduser().resolve()
        try:
            output_dir.relative_to(guard_root)
        except ValueError as error:
            raise ValueError(
                f"La salida debe permanecer dentro de {guard_root}: {output_dir}"
            ) from error
    if output_dir.exists():
        raise FileExistsError(f"La carpeta de salida ya existe: {output_dir}")
    os.environ["MUJOCO_GL"] = "glfw" if args.viewer else "egl"
    import baselines as tuned
    mm = tuned.mm
    tuned.configure_tuned_controllers()
    tuned.validate_routes(routes_dir, maps)
    mm.MAPS_DIR = str(routes_dir)
    mm.LOAD_ROUTES_ON_START = True
    mm.SAVE_ROUTES_PER_MAP = False
    mm.SAVE_GENERATED_ROUTE_PLOTS = False
    mm.GENERATE_NEW_RRT_ROUTE_EACH_EPISODE = False
    mm.MAP_CACHE.clear()
    mm.MAPS_LIST = list(maps)
    mm.FIXED_MAP_ID = None
    mm.MAP_SELECT_MODE = "cycle"
    mm.EPISODES_PER_MAP = len(masses) * repetitions
    mm.BLOCK_EPISODES = repetitions
    mm.CUBE_MASS_BY_BLOCK_KG = list(masses)
    mm.AUTO_STOP_AFTER_EVAL = True
    mm.MAX_DECISION_STEPS = int(args.max_steps)
    mm.HEADLESS = not bool(args.viewer)
    mm.REAL_TIME = bool(args.viewer)
    mm.TRACKING_CONTROLLER = "pure_pursuit"
    mm.SAVE_TRACKING_ERROR_PLOTS = False
    mm.PLOT_ONLY_PURE_TRACKING = True
    mm.GOAL_SUCCESS_DIST_TOL = 0.050
    mm.CUBE_XY_NOISE_RANGE = float(xy_noise)
    mm.CUBE_YAW_NOISE_RANGE_RAD = float(np.deg2rad(yaw_noise_deg))
    np.random.seed(int(args.seed))
    mm._cube_noise_rng = np.random.default_rng(int(args.seed))
    original_reset = mm.reset_world_like_rl_for_map
    reset_counts: Dict[Tuple[int, float], int] = {}
    def paired_seed_reset(map_id: int, cube_mass_kg=None):
        mass_value = float(cube_mass_kg)
        key = (int(map_id), mass_value)
        repetition_index = reset_counts.get(key, 0)
        reset_counts[key] = repetition_index + 1
        map_index = maps.index(int(map_id))
        mass_index = masses.index(mass_value)
        episode_seed = int(
            args.seed + int(map_id) * 100_000 + mass_index * 1_000
            + repetition_index
        )
        mm._cube_noise_rng = np.random.default_rng(episode_seed)
        return original_reset(map_id, cube_mass_kg=cube_mass_kg)
    mm.reset_world_like_rl_for_map = paired_seed_reset
    total = len(maps) * len(masses) * repetitions
    print("=" * 84)
    print("DATA BASELINES | PID + PURE PURSUIT + IK ACTUALES")
    print(f"Mapas={maps} | masas={masses} | repeticiones={repetitions} | total={total}")
    print(f"PID={tuned.PID_PROFILE} | IK={tuned.IK_PROFILE} | Ts={mm.DT_CONTROL:.3f}s")
    print(f"Rutas={routes_dir} | stop={tuned.STOP_DISTANCE_M:.2f}m")
    print(f"Ruido cubo=+/-{xy_noise:.3f}m, +/-{yaw_noise_deg:.1f}deg")
    print("=" * 84)
    started = datetime.now()
    try:
        mm.main()
    finally:
        mm.reset_world_like_rl_for_map = original_reset
    raw_results = list(mm.EPISODE_RESULTS)
    raw_traces = list(mm.EPISODE_TRACES)
    if len(raw_results) != total or len(raw_traces) != total:
        raise RuntimeError(
            f"Ejecucion incompleta: resultados={len(raw_results)}, "
            f"trazas={len(raw_traces)}, esperados={total}."
        )
    episodes, traces = canonicalize(
        raw_results,
        raw_traces,
        routes_dir,
        repetitions,
        int(args.seed),
        maps,
        masses,
    )
    metadata = {
        "program": PROGRAM_NAME,
        "schema_version": SCHEMA_VERSION,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "profile": args.profile,
        "maps": maps,
        "masses_kg": masses,
        "repetitions": repetitions,
        "seed": int(args.seed),
        "episode_seed_policy": "base + map_id*100000 + mass_index*1000 + repetition",
        "paired_with_data_sac": True,
        "cube_xy_noise_m": xy_noise,
        "cube_yaw_noise_deg": yaw_noise_deg,
        "model_xml": str(Path(mm.XML).resolve()),
        "routes_dir": str(routes_dir),
        "physical_timestep_s": float(mm.Ts),
        "frameskip": int(mm.FRAMESKIP_CONTROL),
        "control_timestep_s": float(mm.DT_CONTROL),
        "pid_profile": tuned.PID_PROFILE,
        "pid_gains": dict(tuned.PID_GAINS_TUNED),
        "lookahead_m": float(tuned.LOOKAHEAD_M),
        "pure_pursuit_law": "omega=v*2*sin(alpha)/Ld + Kpsi*e_path - Ky*cte",
        "pure_pursuit_path_heading_gain": float(
            mm.PURE_PURSUIT_PATH_HEADING_GAIN
        ),
        "pure_pursuit_cte_omega_gain": float(mm.PURE_PURSUIT_CTE_OMEGA_GAIN),
        "terminal_pose_distance_m": float(mm.PURE_PURSUIT_TERMINAL_POSE_DIST_M),
        "terminal_endpoint_tolerance_m": float(mm.FINAL_WAYPOINT_TOL),
        "terminal_heading_tolerance_deg": float(
            np.rad2deg(mm.FINAL_ROUTE_HEADING_TOL)
        ),
        "reverse_position_tolerance_m": float(mm.REV_STOP_TOL),
        "identified_mobile_mass_kg": float(mm.m),
        "identified_mobile_yaw_inertia_kg_m2": float(mm.J),
        "stop_distance_m": float(tuned.STOP_DISTANCE_M),
        "phase2_xy_success_tolerance_m": 0.050,
        "ik_profile": tuned.IK_PROFILE,
        "ik_params": dict(tuned.IK_PARAMS_TUNED),
        "tracking_metric": (
            "signed perpendicular CTE from chassis_frame and continuous path "
            "tangent error at the projected arc length; lookahead heading is "
            "stored only as a controller diagnostic; raw signals"
        ),
        "pose_source": "MuJoCo ground-truth site chassis_frame",
        "plot_filter": "none",
    }
    summary = save_dataset(
        output_dir=output_dir,
        method="baseline_pid_ik_pure_pursuit",
        metadata=metadata,
        episodes=episodes,
        traces=traces,
        plots=not bool(args.no_plots),
    )
    rates = summary["rates"]
    print(
        f"[OK] P0={100 * rates['phase0_success']:.1f}% "
        f"P1={100 * rates['phase1_success']:.1f}% "
        f"P2={100 * rates['phase2_success']:.1f}%"
    )
    print(f"[OK] Datos guardados en: {output_dir}")
import json
import os
from pathlib import Path
import sqlite3
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/vx_baseline_paired')
import numpy as np
ROOT = RESOURCE_ROOT
MASSES = (.5, .75, 1., 1.25, 1.5)
PHYSICAL_FIELDS = ('body_mass', 'body_inertia', 'geom_friction', 'dof_damping',
                   'dof_armature', 'actuator_gainprm', 'actuator_biasprm',
                   'actuator_ctrlrange', 'actuator_forcerange')
def initial_state(model, data, map_id, mass):
    return dict(qpos=data.qpos.tolist(), qvel=data.qvel.tolist(),
                map_id=int(map_id), cube_mass_kg=float(mass),
                physical_parameters={name: getattr(model, name).tolist() for name in PHYSICAL_FIELDS})
def disable_cube_xy_clamp(mm):
    mm.clamp_cube_xy_to_pedestal = lambda xy: np.asarray(xy, dtype=np.float64).copy()
def main(argv=None):
    legacy = sys.modules[__name__]
    import baselines as tuned
    mm = tuned.mm
    legacy.PROGRAM_NAME = 'DATA_BASELINES.py'
    legacy.OUTPUT_GUARD_ROOT = ROOT / 'data_vx'
    legacy.OUTPUT_ROOT = ROOT / 'data_vx' / 'paired_comparison' / 'baseline'
    original_parser = legacy.build_parser
    def build_parser():
        parser = original_parser()
        parser.set_defaults(profile='robust', repetitions=10,
                            cube_xy_noise=.01, cube_yaw_noise_deg=float(np.rad2deg(.1)))
        return parser
    legacy.build_parser = build_parser
    args = build_parser().parse_args(argv)
    if any(mass not in MASSES for mass in args.masses):
        raise ValueError('Only the canonical five masses are allowed')
    disable_cube_xy_clamp(mm)
    original_reset = mm.reset_world_like_rl_for_map
    original_home = mm._set_home_pose
    counters, states, current = {}, [], {}
    def paired_reset(map_id, cube_mass_kg=None):
        mass = float(cube_mass_kg)
        key = (int(map_id), mass)
        rep = counters.get(key, 0)
        counters[key] = rep + 1
        seed = int(args.seed + int(map_id) * 100000 + MASSES.index(mass) * 1000 + rep)
        mm._cube_noise_rng = np.random.default_rng(seed)
        current.update(map_id=map_id, mass=mass, seed=seed)
        return original_reset(map_id, cube_mass_kg=cube_mass_kg)
    def set_home_and_record(model, data, ik):
        result = original_home(model, data, ik)
        states.append(dict(seed=current['seed'], state=initial_state(
            model, data, current['map_id'], current['mass'])))
        return result
    mm.reset_world_like_rl_for_map = paired_reset
    mm._set_home_pose = set_home_and_record
    original_save = legacy.save_dataset
    def save_with_protocol(**kwargs):
        episodes = kwargs['episodes']
        if len(states) != len(episodes):
            raise ValueError('Missing baseline initial-state records')
        for episode, record in zip(episodes, states):
            if (episode['map_id'], episode['mass_kg']) != (record['state']['map_id'], record['state']['cube_mass_kg']):
                raise ValueError('Initial-state/episode ordering mismatch')
            episode['seed'] = record['seed']
        kwargs['metadata'].update({
            'initial_state_protocol': 'unclamped_xy_common_noise_v1',
            'cube_xy_clamp': 'none', 'initial_state_table': 'initial_states',
            'phase_budgets': None, 'global_deadline_steps': int(args.max_steps),
            'budgets_note': 'Original classical state machine; not student phase deadlines',
            'episode_seed_policy': 'base + map*100000 + canonical_mass_idx*1000 + rep_idx',
            'route_hashes': {p.name: sha256(p) for p in sorted(ROUTES_DIR.glob('*.json'))},
            'extractor_sha256': sha256(__file__),
            'baseline_sources_sha256': {p.name: sha256(p) for p in (
                Path(__file__), Path(tuned.__file__))},
        })
        result = original_save(**kwargs)
        database = Path(kwargs['output_dir']) / 'dataset.sqlite'
        with sqlite3.connect(database) as con:
            con.execute('CREATE TABLE initial_states (episode_id INTEGER PRIMARY KEY, episode_seed INTEGER NOT NULL, state_json TEXT NOT NULL)')
            ids = con.execute('SELECT episode_id FROM episodes ORDER BY episode_id').fetchall()
            if len(ids) != len(states):
                raise ValueError('Database episode count differs from initial states')
            con.executemany('INSERT INTO initial_states VALUES (?,?,?)',
                            [(eid[0], record['seed'], json.dumps(record['state'])) for eid, record in zip(ids, states)])
        return result
    legacy.save_dataset = save_with_protocol
    _extract_baseline_main(argv)
if __name__ == '__main__':
    main()
