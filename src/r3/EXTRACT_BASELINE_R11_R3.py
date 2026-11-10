from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import concurrent.futures
import contextlib
import csv
import hashlib
import json
import multiprocessing
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "VX_R3")]

import mujoco
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from EVALUATE_R3 import CurriculumSAC, ZennoVXR3, baseline_module, physical_hash, snapshot
from evaluate_zenno_vx_r1_1 import _SpaceOnlyEnv
from extract_vx_place_robust import BaselineDriver, dump, evaluation_environment, make_cases
from vx_final_student_contract import load_environment_class, validate_final_student
from vx_place_initials import canonicalize_and_audit
from vx_place_protocol import Recorder, protocol_metadata

METHODS = ("baseline", "r1_1", "r3")
MASSES = (.5, .75, 1., 1.25, 1.5)
R1_MANIFEST = ROOT / "models/vx_r1_r2_sequence_20260909/r1_1_student/final/final_student.json"
R3_SELECTION = ROOT / "VX_R3/comparisons/baseline_r3_300k_robust_20260924_205029/config.json"
WORKER = {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def models_and_cases(args):
    if not 1 <= args.reps <= 1000 or not 0 <= args.seed < 2**32 - 500000:
        raise ValueError("Repeticiones o semilla inválidas")
    if not 1 <= args.workers <= 4:
        raise ValueError("Use 1 a 4 trabajadores")
    if not args.maps or len(set(args.maps)) != len(args.maps) or not set(args.maps) <= {1, 2, 3, 4}:
        raise ValueError("Mapas inválidos o duplicados")
    if not args.masses or len(set(args.masses)) != len(args.masses) or not set(args.masses) <= set(MASSES):
        raise ValueError("Masas inválidas o duplicadas")
    r1 = validate_final_student(args.r1_manifest, expected_experiment="r1_1")
    selected = json.loads(R3_SELECTION.read_text())
    r3_path = args.r3_model.resolve() if args.r3_model else Path(selected["checkpoint"]).resolve()
    if not r3_path.is_file():
        raise FileNotFoundError(r3_path)
    if args.r3_model is None and digest(r3_path) != selected["model_sha256"]:
        raise ValueError("El R3 seleccionado ya no coincide con la evaluación publicada")
    panel = [dict(c, profile="robust") for c in make_cases(args.maps, args.masses, args.reps, args.seed)]
    if args.limit_cases is not None:
        if args.limit_cases < 1:
            raise ValueError("limit-cases debe ser positivo")
        panel = panel[:args.limit_cases]
    if len({c["case_seed"] for c in panel}) != len(panel):
        raise ValueError("Se repiten semillas")
    return r1, r3_path, panel


def init_worker(r1_manifest, r3_path):
    torch.set_num_threads(1)
    bundle = validate_final_student(r1_manifest, expected_experiment="r1_1")
    norm = VecNormalize.load(bundle["vecnormalize_path"], DummyVecEnv([_SpaceOnlyEnv]))
    norm.training = norm.norm_reward = False
    r1 = SAC.load(bundle["model_path"], device="cpu", custom_objects={"buffer_size": 1})
    r1.policy.set_training_mode(False)
    r3 = CurriculumSAC.load(r3_path, device="cpu", custom_objects={"buffer_size": 1})
    if (r1.observation_space.shape, r1.action_space.shape) != ((124,), (7,)):
        raise ValueError("R1.1 no es el actor final 124D/7D")
    if (r3.observation_space.shape, r3.action_space.shape) != ((130,), (6,)):
        raise ValueError("R3 no es el actor final 130D/6D")
    if np.shape(norm.obs_rms.mean) != (124,):
        raise ValueError("Normalizador R1.1 incompatible")
    mm, ik = baseline_module()
    WORKER.update(r1=r1, norm=norm, r3=snapshot(r3), mm=mm, ik=ik,
                  r1_class=evaluation_environment(load_environment_class(bundle)),
                  r1_kwargs=bundle["manifest"]["environment"]["kwargs"])


def episode(case, method, route, length):
    mm, ik = WORKER["mm"], WORKER["ik"]
    cls = WORKER["r1_class"] if method == "r1_1" else ZennoVXR3
    kwargs = WORKER["r1_kwargs"] if method == "r1_1" else {"p1_budget_steps": 4500}
    env = cls(render_mode=None, debug_rewards=False, **kwargs)
    try:
        env.set_eval_mode(True)
        env.FIXED_MAP_ID = case["map_id"]
        env.set_eval_mass(case["mass_kg"])
        env.CUBE_XY_NOISE_RANGE, env.CUBE_YAW_NOISE_RANGE_RAD = .01, .1
        env.reset(seed=case["case_seed"])
        mm.CUBE_XY_NOISE_RANGE, mm.CUBE_YAW_NOISE_RANGE_RAD = .01, .1
        mm._current_map_id = case["map_id"]
        mm._cube_noise_rng = np.random.default_rng(case["case_seed"])
        mm.reset_world_like_rl_for_map(case["map_id"], cube_mass_kg=case["mass_kg"])
        mm._set_home_pose(mm.model, mm.data, ik)
        audit = canonicalize_and_audit(env, mm)
        if not audit["passed"]:
            raise AssertionError("La auditoría física inicial falló")
        signature = physical_hash(env)
        obs = env._get_obs()
        driver = BaselineDriver(mm, ik) if method == "baseline" else None
        recorder = Recorder(mm.model, mm.data, route) if driver else (
            env.place_recorder if method == "r3" else Recorder(env.model, env.data, route))
        if method == "r1_1":
            env.place_recorder = recorder
        counts, carry, delta, base_error, stop = [0, 0, 0], [], [], None, None
        for _ in range(20000):
            phase = driver.phase if driver else int(env.phase_state)
            counts[phase] += 1
            if driver:
                driver.control()
                for _ in range(5):
                    mujoco.mj_step(mm.model, mm.data)
                    recorder.substep()
                stop = recorder.end_interval(phase, phase == 1)
            elif method == "r1_1":
                normalized = WORKER["norm"].normalize_obs(np.asarray(obs, dtype=np.float32).copy())
                action = WORKER["r1"].predict(normalized, deterministic=True)[0]
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    raise RuntimeError("La terminación nativa escapó del observador común")
                stop = recorder.end_interval(phase, phase == 1,
                    native_deadline=counts[phase] >= env.phase_budgets[phase])
                failures = [d["reason"] for d in env.place_native_diagnostics
                            if d["reason"] not in ("place_success_arm_home", "phase2_home_budget")]
                if stop is None and failures:
                    stop = failures[-1]
                if stop is None and counts[phase] >= env.phase_budgets[phase] and int(env.phase_state) == phase:
                    stop = f"phase{phase}_budget"
            else:
                action = WORKER["r3"].predict(obs, deterministic=True)[0]
                obs, _, terminated, truncated, info = env.step(action)
                stop = info["task_stop"]
                if bool(terminated or truncated) != bool(stop):
                    raise AssertionError("R3: parada incoherente")
                if info.get("lookahead_active"):
                    delta.append(float(info["lookahead_applied_m"] - info["lookahead_nominal_m"]))
            fresh = recorder._fresh_auxiliary_state()
            next_phase = driver.phase if driver else int(env.phase_state)
            if phase == 1:
                carry.append(float(np.linalg.norm(
                    fresh.body("cube").xpos - fresh.site("setpoint").xpos)))
            if phase == 1 and next_phase == 2:
                base_error = float(np.linalg.norm(
                    fresh.site_xpos[recorder.base_site_id, :2] - route[-1, :2]))
            if stop is not None:
                break
        if stop is None:
            raise RuntimeError("20000 intervalos sin criterio terminal")
        trace = recorder.traces()
        arc = trace["route_progress"] * length
        gate = (trace["phase"] == 1) & (arc >= .30) & (arc <= length - .55)
        trace["tracking_valid"] = gate
        yaw = (trace["base_yaw_rad"] - trace["path_yaw_rad"] + np.pi) % (2*np.pi) - np.pi
        row = recorder.summary()
        for key, series, rms in (
            ("cte_rms_m", trace["cte_signed_m"], True),
            ("cte_signed_mean_m", trace["cte_signed_m"], False),
            ("distance_to_route_rms_m", trace["distance_to_route_m"], True),
            ("yaw_rms_rad", yaw, True)):
            values = series[gate]
            row[key] = float(np.sqrt(np.mean(values**2)) if rms else np.mean(values)) if len(values) else None
        work11 = trace["energy_arm_J"] + trace["energy_base_J"]
        work12 = work11 + trace["energy_gripper_J"]
        row.update(case, method=method, success_place=stop == "stable_place",
            termination_reason=stop, initial_hash=signature, phase_steps=counts,
            tracking_samples=int(gate.sum()), base_terminal_error_m=base_error,
            carry_rms_m=float(np.sqrt(np.mean(np.square(carry)))) if carry else None,
            lookahead_delta_rms_m=float(np.sqrt(np.mean(np.square(delta)))) if delta else 0.,
            work_11_J=float(work11.sum()), work_12_J=float(work12.sum()),
            work_rms_11_J=float(np.sqrt(np.mean(work11**2))),
            work_rms_12_J=float(np.sqrt(np.mean(work12**2))))
        checks = {"passed": True,
                  "max_abs_differences_before": audit["max_abs_differences_before"],
                  "max_abs_differences_after": audit["max_abs_differences_after"]}
        return row, trace, signature, checks
    finally:
        env.close()


def run_case(case, output):
    output = Path(output)
    route = np.asarray(json.loads(
        (ROOT / "RUTAS_SAC" / f'map_{case["map_id"]}.json').read_text())["path"])
    length = float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum())
    rows, signatures, checks = [], {}, {}
    for method in METHODS:
        row, trace, signatures[method], checks[method] = episode(case, method, route, length)
        folder = output / "traces" / method
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / f'{case["case_id"]}.npz', **trace)
        dump(output / "episodes" / method / f'{case["case_id"]}.json', row)
        rows.append(row)
    if len(set(signatures.values())) != 1:
        raise AssertionError(f'Estados iniciales distintos: {case["case_id"]}')
    dump(output / "initial_states" / f'{case["case_id"]}.json',
         dict(case=case, signatures=signatures, audits=checks, all_methods_equal=True))
    return rows


def worker(case, output):
    with (Path(output) / "logs" / f'{case["case_id"]}.log').open("w") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            return run_case(case, output)


def aggregate(rows):
    grouped = {}
    for row in rows:
        group = grouped.setdefault(row["case_id"], {})
        if row["method"] in group:
            raise ValueError("Caso/método duplicado")
        group[row["method"]] = row
    if any(set(g) != set(METHODS) for g in grouped.values()):
        raise ValueError("Faltan métodos")
    metrics = ("total_time_s", "work_11_J", "work_12_J", "work_rms_11_J",
               "work_rms_12_J", "cte_rms_m", "yaw_rms_rad", "cube_goal_dist_xy_m",
               "carry_rms_m", "base_terminal_error_m")
    result = {}
    for scope, map_id in [("global", None)] + [(f"map_{m}", m) for m in (1, 2, 3, 4)]:
        cases = [g for g in grouped.values() if map_id is None or g["baseline"]["map_id"] == map_id]
        if not cases:
            continue
        common = [g for g in cases if all(g[m]["success_place"] for m in METHODS)]
        values = {}
        for key in metrics:
            valid = [g for g in common if all(g[m].get(key) is not None and
                     np.isfinite(g[m][key]) for m in METHODS)]
            values[key] = dict(n=len(valid), **{
                m: float(np.mean([g[m][key] for g in valid])) if valid else None for m in METHODS})
            if key == "cube_goal_dist_xy_m":
                values["cube_goal_dist_xy_rms_m"] = dict(n=len(valid), **{
                    m: float(np.sqrt(np.mean([g[m][key]**2 for g in valid])))
                    if valid else None for m in METHODS})
        result[scope] = dict(n_cases=len(cases), common_successes=len(common),
            success_counts={m: sum(g[m]["success_place"] for g in cases) for m in METHODS},
            failure_reasons={m: {reason: sum(g[m]["termination_reason"] == reason
                for g in cases if not g[m]["success_place"]) for reason in sorted({
                g[m]["termination_reason"] for g in cases if not g[m]["success_place"]})}
                for m in METHODS}, metrics_common_successes=values)
    return result


def report(output, rows):
    groups = aggregate(rows)
    dump(output / "summary.json", dict(complete=True, methods=METHODS,
         comparison_cohort="success in all three methods on exactly the same case",
         groups=groups, episodes=rows))
    units = {"total_time_s": "s", "work_11_J": "J", "work_12_J": "J",
             "work_rms_11_J": "J/10ms", "work_rms_12_J": "J/10ms",
             "cte_rms_m": "m", "yaw_rms_rad": "rad",
             "cube_goal_dist_xy_m": "m", "cube_goal_dist_xy_rms_m": "m",
             "carry_rms_m": "m", "base_terminal_error_m": "m"}
    with (output / "comparison.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("scope", "metric", "unit", "n_common", *METHODS))
        for scope, g in groups.items():
            writer.writerow((scope, "success_count", "episodes", g["n_cases"],
                             *(g["success_counts"][m] for m in METHODS)))
            for key, data in g["metrics_common_successes"].items():
                writer.writerow((scope, key, units[key], data["n"],
                                 *(data[m] for m in METHODS)))
    with (output / "failures.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=(
            "case_id", "method", "map_id", "mass_kg", "rep", "case_seed",
            "termination_reason", "total_time_s"), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(row for row in rows if not row["success_place"])
    return groups


def completed_rows(output, case):
    cid = case["case_id"]
    audit_path = output / "initial_states" / f"{cid}.json"
    if not audit_path.is_file():
        return None
    audit = json.loads(audit_path.read_text())
    if not audit.get("all_methods_equal") or audit.get("case") != case:
        raise ValueError(f"Invalid resumed audit: {cid}")
    rows = []
    for method in METHODS:
        ep = output / "episodes" / method / f"{cid}.json"
        trace = output / "traces" / method / f"{cid}.npz"
        if not ep.is_file() or not trace.is_file():
            return None
        row = json.loads(ep.read_text())
        if row["method"] != method or row["initial_hash"] != audit["signatures"][method]:
            raise ValueError(f"Incompatible resumed episode: {cid}/{method}")
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--r1-manifest", type=Path, default=R1_MANIFEST)
    parser.add_argument("--r3-model", type=Path, help="Por defecto: R3 de 300k pasos")
    parser.add_argument("--seed", type=int, default=510_000_000)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--maps", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--masses", type=float, nargs="+", default=list(MASSES))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit-cases", type=int, help="Prueba corta, no informe final")
    parser.add_argument("--resume", action="store_true", help="Continúa casos pendientes de la misma salida")
    args = parser.parse_args()
    bundle, r3, panel = models_and_cases(args)
    output = args.output.expanduser().resolve()
    if not output.is_relative_to(ROOT / "data_vx"):
        parser.error("Salida obligatoria dentro de INCREMENTAL_SAC/data_vx")
    if args.resume:
        if not output.is_dir():
            parser.error("--resume requiere una salida existente")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "logs").mkdir()
    sources = [Path(__file__), ROOT / "VX_R3/VX_R3.py", ROOT / "VX_R3/EVALUATE_R3.py",
               ROOT / "zenno_vx_r1_1.py", ROOT / "extract_vx_place_robust.py",
               ROOT / "vx_place_protocol.py", ROOT / "vx_place_initials.py",
               ROOT / "VX/baselines.py"] + sorted((ROOT / "RUTAS_SAC").glob("*.json"))
    config = dict(methods=list(METHODS), cases=panel, seed=args.seed,
         repetitions=args.reps, r1_manifest=str(bundle["manifest_path"]),
         r1_model_sha256=digest(bundle["model_path"]),
         r1_normalizer_sha256=digest(bundle["vecnormalize_path"]),
         r3_model=str(r3), r3_model_sha256=digest(r3),
         source_sha256={str(p): digest(p) for p in sources},
         placement=protocol_metadata(), deterministic=True,
         baseline_runs_per_case=1, no_training=True, inference="CPU")
    if args.resume:
        if json.loads((output / "config.json").read_text()) != config:
            raise ValueError("Fuentes, casos o modelos cambiaron; reanudación rechazada")
        reused = {case["case_id"]: completed_rows(output, case) for case in panel}
        rows = [row for case in panel for row in (reused[case["case_id"]] or [])]
        pending = [case for case in panel if reused[case["case_id"]] is None]
    else:
        dump(output / "config.json", config)
        rows, pending = [], panel
    dump(output / "status.json", dict(state="RUNNING", completed_cases=len(rows)//3,
                                      total_cases=len(panel)))
    print("START", json.dumps(dict(output=str(output), cases=len(panel),
                                   pending=len(pending), episodes=3*len(panel),
                                   seed=args.seed)), flush=True)
    started = time.monotonic()
    try:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=init_worker,
                initargs=(str(bundle["manifest_path"]), str(r3))) as pool:
            futures = [pool.submit(worker, case, str(output)) for case in pending]
            for future in concurrent.futures.as_completed(futures):
                rows.extend(future.result())
                done = len(rows)//3
                dump(output / "status.json", dict(state="RUNNING", completed_cases=done,
                     total_cases=len(panel), elapsed_s=time.monotonic()-started))
                if done in (1, max(1, len(panel)//2), len(panel)):
                    print("PROGRESS", json.dumps(dict(cases=done, total=len(panel),
                          elapsed_s=round(time.monotonic()-started, 1))), flush=True)
        if len(rows) != 3*len(panel):
            raise RuntimeError("Extracción incompleta")
        rows.sort(key=lambda r: (r["map_id"], r["mass_kg"], r["rep"], r["method"]))
        groups = report(output, rows)
        dump(output / "status.json", dict(state="COMPLETE", completed_cases=len(panel),
             total_cases=len(panel), elapsed_s=time.monotonic()-started))
        print("COMPLETE", json.dumps(dict(success=groups["global"]["success_counts"],
              common=groups["global"]["common_successes"], output=str(output))), flush=True)
    except BaseException:
        dump(output / "error.json", dict(traceback=traceback.format_exc()))
        dump(output / "status.json", dict(state="FAILED", completed_cases=len(rows)//3,
             total_cases=len(panel), elapsed_s=time.monotonic()-started))
        raise


if __name__ == "__main__":
    main()
