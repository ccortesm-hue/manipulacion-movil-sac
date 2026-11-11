#!/usr/bin/env python3
"""Ablación OFF de R1.1/R3; reutiliza el panel ON de la comparación de tesis."""
from __future__ import annotations

import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import concurrent.futures
import contextlib
import csv
from datetime import datetime
import json
import multiprocessing
import signal
import sys
import time
import traceback
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
import EXTRACT_BASELINE_R11_R3 as common

np = common.np
ROOT = common.ROOT
DEFAULT_REFERENCE = ROOT / "data_vx/baseline_r11_r3_common_20260929"
AGENTS = ("r1_1", "r3")
CASE_FIELDS = ("case_id", "map_id", "mass_kg", "rep", "case_seed", "profile")
ZERO_TOLERANCE = 1e-12
METRICS = {
    "total_time_s": "s", "work_11_J": "J", "work_12_J": "J",
    "work_rms_11_J": "J/intervalo de 10 ms", "cte_rms_m": "m",
    "cte_signed_mean_m": "m", "yaw_rms_rad": "rad",
    "cube_goal_dist_xy_m": "m", "carry_rms_m": "m",
    "base_terminal_error_m": "m",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_reference(folder):
    """Rechaza paneles parciales, fuentes distintas y checkpoints sustituidos."""
    folder = Path(folder).expanduser().resolve()
    config, summary = read_json(folder / "config.json"), read_json(folder / "summary.json")
    if not summary.get("complete") or set(config.get("methods", [])) != {"baseline", *AGENTS}:
        raise ValueError("La referencia debe ser la comparación conjunta completa de tres métodos")
    if not config.get("deterministic") or config.get("inference") != "CPU":
        raise ValueError("La ablación requiere la inferencia determinista CPU original")
    if config.get("placement") != common.protocol_metadata():
        raise ValueError("El observador de colocación cambió respecto de la referencia")
    cases = config["cases"]
    if not cases or len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Casos de referencia vacíos o duplicados")
    if len({c["case_seed"] for c in cases}) != len(cases):
        raise ValueError("Semillas duplicadas en la referencia")
    by_case = {c["case_id"]: {} for c in cases}
    case_lookup = {c["case_id"]: c for c in cases}
    for row in summary["episodes"]:
        cid, method = row["case_id"], row["method"]
        if cid not in by_case or method not in config["methods"] or method in by_case[cid]:
            raise ValueError("Episodios de referencia duplicados o ajenos al manifiesto")
        if any(row.get(k) != case_lookup[cid].get(k) for k in CASE_FIELDS):
            raise ValueError(f"Caso no pareado en referencia: {cid}/{method}")
        by_case[cid][method] = row
    if any(set(group) != set(config["methods"]) for group in by_case.values()):
        raise ValueError("Faltan episodios activos o baseline en la referencia")
    sources = config.get("source_sha256", {})
    if not sources or any(common.digest(p) != sha for p, sha in sources.items()):
        raise ValueError("Alguna fuente o ruta difiere de la extracción activa; no mezclar protocolos")
    bundle = common.validate_final_student(config["r1_manifest"], expected_experiment="r1_1")
    r3 = Path(config["r3_model"]).resolve()
    for path, expected in (
        (bundle["model_path"], config["r1_model_sha256"]),
        (bundle["vecnormalize_path"], config["r1_normalizer_sha256"]),
        (r3, config["r3_model_sha256"]),
    ):
        if common.digest(path) != expected:
            raise ValueError(f"Checkpoint o normalizador distinto: {path}")
    return dict(folder=folder, config=config, cases=cases, by_case=by_case,
                bundle=bundle, r3=r3)


def select_cases(reference, args):
    cases = reference["cases"]
    if args.maps:
        cases = [c for c in cases if c["map_id"] in args.maps]
    if args.case_ids:
        unknown = set(args.case_ids) - {c["case_id"] for c in reference["cases"]}
        if unknown:
            raise ValueError(f"Casos no presentes en la referencia: {sorted(unknown)}")
        cases = [c for c in cases if c["case_id"] in args.case_ids]
    if args.limit_cases is not None:
        if args.limit_cases < 1:
            raise ValueError("limit-cases debe ser positivo")
        cases = cases[:args.limit_cases]
    if not cases:
        raise ValueError("Panel vacío")
    return cases


def _register_env(env):
    env.off_samples = []
    env.off_p1_start_hash = None
    common.WORKER["off_env"] = env


def residual_off_r11_class(base_class):
    class R11ResidualOff(base_class):
        def __init__(self, **kwargs):
            # Conservar las acciones solicitadas y su historia; abrir la rama física.
            # Forzar los últimos canales a cero también en P0 alteraría esa historia.
            kwargs["residual_enabled"] = False
            self.off_torque_calls = 0
            self.off_torque_sample = None
            _register_env(self)
            super().__init__(**kwargs)

        def _apply_wheel_torque_with_residual(self, left, right, u_lin, u_ang):
            if max(abs(u_lin), abs(u_ang)) > ZERO_TOLERANCE:
                raise AssertionError("R1.1 OFF recibió una corrección en el sumador de torque")
            super()._apply_wheel_torque_with_residual(left, right, u_lin, u_ang)
            ids = list(self.wheel_left_ids) + list(self.wheel_right_ids)
            requested = [left] * len(self.wheel_left_ids) + [right] * len(self.wheel_right_ids)
            nominal = np.array([np.clip(tau, -self._actuator_tau_limit(aid),
                               self._actuator_tau_limit(aid)) for tau, aid in zip(requested, ids)])
            applied = self.data.ctrl[ids].copy()
            if np.max(np.abs(applied - nominal)) > ZERO_TOLERANCE:
                raise AssertionError("R1.1 OFF alteró físicamente el torque nominal del PID")
            self.off_torque_sample = (nominal, applied)
            self.off_torque_calls += 1

        def step(self, action):
            action = np.asarray(action, dtype=np.float32).copy()
            phase = int(self.phase_state)
            if phase == 1 and self.off_p1_start_hash is None:
                self.off_p1_start_hash = common.physical_hash(self)
            self.off_torque_sample = None
            result = super().step(action)
            if self._vx_active or np.any(np.abs(self._vx_delta) > ZERO_TOLERANCE):
                raise AssertionError("R1.1 OFF dejó activa la rama residual de torque")
            ids = list(self.wheel_left_ids) + list(self.wheel_right_ids)
            nominal, applied = self.off_torque_sample or (self.data.ctrl[ids].copy(),
                                                         self.data.ctrl[ids].copy())
            self.off_samples.append(dict(
                phase=phase, actor_arm_action=action[:5].copy(),
                actor_residual_request=action[5:7].copy(),
                wheel_nominal_command_nm=nominal, wheel_applied_command_nm=applied,
                wheel_applied_residual_nm=applied - nominal))
            return result

    return R11ResidualOff


class R3ResidualOff(common.ZennoVXR3):
    def __init__(self, **kwargs):
        _register_env(self)
        super().__init__(**kwargs)

    def step(self, action):
        requested = np.asarray(action, dtype=np.float32).copy()
        if requested.shape != (6,) or not np.isfinite(requested).all():
            raise ValueError("R3 OFF requiere seis acciones finitas")
        executed = requested.copy()
        executed[5] = 0.0
        phase = int(self.phase_state)
        if phase == 1 and self.off_p1_start_hash is None:
            self.off_p1_start_hash = common.physical_hash(self)
        result = super().step(executed)
        info = result[4]
        nominal = float(info["lookahead_nominal_m"])
        applied = float(info["lookahead_applied_m"])
        if max(abs(applied - nominal), abs(info["lookahead_filtered_m"] - nominal)) > ZERO_TOLERANCE:
            raise AssertionError("R3 OFF modificó el lookahead nominal")
        if np.max(np.abs(self._r2_pp_corrected - self._r2_pp_nominal)) > ZERO_TOLERANCE:
            raise AssertionError("R3 OFF activó un residual aditivo de velocidad")
        self.off_samples.append(dict(
            phase=phase, actor_arm_action=requested[:5].copy(),
            actor_residual_request=requested[5:6].copy(),
            executed_lookahead_request=0.0, lookahead_nominal_m=nominal,
            lookahead_applied_m=applied, lookahead_delta_m=applied - nominal))
        return result


def init_worker(r1_manifest, r3_path):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    common.init_worker(r1_manifest, r3_path)
    common.WORKER["r1_class"] = residual_off_r11_class(common.WORKER["r1_class"])
    common.ZennoVXR3 = R3ResidualOff


def verify_initial_and_pick(reference, case, agent, row, trace):
    folder = Path(reference)
    audit = read_json(folder / "initial_states" / f'{case["case_id"]}.json')
    active = read_json(folder / "episodes" / agent / f'{case["case_id"]}.json')
    if (audit.get("case") != case or not audit.get("all_methods_equal")
            or row["initial_hash"] != audit["signatures"][agent]
            or row["initial_hash"] != active["initial_hash"]):
        raise AssertionError(f'Estado inicial OFF distinto del ON: {case["case_id"]}/{agent}')
    path = folder / "traces" / agent / f'{case["case_id"]}.npz'
    with np.load(path, allow_pickle=False) as on:
        on_mask, off_mask = on["phase"] == 0, trace["phase"] == 0
        if int(on_mask.sum()) != int(off_mask.sum()):
            raise AssertionError("La ablación modificó la duración de P0 antes del residual")
        for key in on.files:
            if not np.array_equal(on[key][on_mask], trace[key][off_mask]):
                raise AssertionError(f"P0 difiere del ON histórico: {case['case_id']}/{agent}/{key}")
        checked = list(on.files)
    return dict(initial_matches_active=True, p0_matches_active=True,
                p0_samples=int(off_mask.sum()), p0_fields_checked=checked,
                reference_trace_sha256=common.digest(path),
                reference_audit_sha256=common.digest(folder / "initial_states" / f'{case["case_id"]}.json'))


def run_case(case, output, reference, agents):
    output = Path(output)
    route = np.asarray(read_json(ROOT / "RUTAS_SAC" / f'map_{case["map_id"]}.json')["path"])
    length = float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum())
    rows, audits = [], {}
    for agent in agents:
        row, trace, signature, checks = common.episode(case, agent, route, length)
        env = common.WORKER.pop("off_env")
        samples = env.off_samples
        if len(samples) != len(trace["phase"]):
            raise AssertionError("La telemetría OFF no cubre todas las decisiones")
        if not np.array_equal([s["phase"] for s in samples], trace["phase"]):
            raise AssertionError("Las fases de la telemetría OFF no están alineadas")
        proof = verify_initial_and_pick(reference, case, agent, row, trace)
        for key in samples[0]:
            trace["ablation_" + key] = np.asarray([s[key] for s in samples])
        physical_key = "wheel_applied_residual_nm" if agent == "r1_1" else "lookahead_delta_m"
        maximum = float(np.max(np.abs(trace["ablation_" + physical_key])))
        if maximum > ZERO_TOLERANCE:
            raise AssertionError("Residual aplicado distinto de cero")
        proof.update(residual_applied_max_abs=maximum,
                     residual_unit="N m" if agent == "r1_1" else "m",
                     p1_reached=bool(np.any(trace["phase"] == 1)),
                     p1_start_hash=env.off_p1_start_hash,
                     wheel_torque_calls=getattr(env, "off_torque_calls", None))
        method = agent + "_off"
        trace_path = output / "traces" / method / f'{case["case_id"]}.npz'
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(trace_path, **trace)
        row.update(agent=agent, method=method, residual_enabled=False,
                   ablation="same_actor_residual_off", ablation_checks=proof,
                   trace_sha256=common.digest(trace_path))
        common.dump(output / "episodes" / method / f'{case["case_id"]}.json', row)
        rows.append(row)
        audits[agent] = dict(initial_hash=signature, initial_audit=checks, **proof)
    common.dump(output / "initial_states" / f'{case["case_id"]}.json',
                dict(case=case, all_match_active_reference=True, audits=audits))
    return rows


def worker(case, output, reference, agents):
    with (Path(output) / "logs" / f'{case["case_id"]}.log').open("w") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            return run_case(case, output, reference, agents)


def completed_rows(output, case, agents, reference):
    audit_path = output / "initial_states" / f'{case["case_id"]}.json'
    if not audit_path.exists():
        return None
    audit = read_json(audit_path)
    if audit.get("case") != case or not audit.get("all_match_active_reference"):
        raise ValueError("Auditoría de reanudación incompatible")
    rows = []
    for agent in agents:
        method, cid = agent + "_off", case["case_id"]
        path = output / "episodes" / method / f"{cid}.json"
        trace = output / "traces" / method / f"{cid}.npz"
        if not path.is_file() or not trace.is_file():
            return None
        row = read_json(path)
        if (row.get("method") != method or any(row.get(k) != case.get(k) for k in CASE_FIELDS)
                or row.get("trace_sha256") != common.digest(trace)
                or row["ablation_checks"]["residual_applied_max_abs"] > ZERO_TOLERANCE):
            raise ValueError(f"Episodio OFF modificado o incompatible: {cid}/{method}")
        if row["ablation_checks"]["reference_trace_sha256"] != common.digest(
                reference / "traces" / agent / f"{cid}.npz"):
            raise ValueError("La traza ON cambió tras iniciar la ablación")
        rows.append(row)
    return rows


def aggregate(rows, reference, agents):
    result = {}
    for agent in agents:
        off_rows = [r for r in rows if r["agent"] == agent]
        if len({r["case_id"] for r in off_rows}) != len(off_rows):
            raise ValueError("Episodios OFF duplicados")
        scopes = {}
        for scope, mid in [("global", None)] + [(f"map_{m}", m) for m in range(1, 5)]:
            pairs = [(reference["by_case"][r["case_id"]][agent], r) for r in off_rows
                     if mid is None or r["map_id"] == mid]
            if not pairs:
                continue
            common_success = [(on, off) for on, off in pairs if on["success_place"] and off["success_place"]]
            metrics = {}
            for key, unit in METRICS.items():
                valid = [(on, off) for on, off in common_success if all(
                    r.get(key) is not None and np.isfinite(r[key]) for r in (on, off))]
                for metric, rms in [(key, False)] + ([("cube_goal_dist_xy_rms_m", True)]
                                                    if key == "cube_goal_dist_xy_m" else []):
                    def value(index):
                        values = [p[index][key] for p in valid]
                        return float(np.sqrt(np.mean(np.square(values))) if rms else np.mean(values)) if valid else None
                    on_value, off_value = value(0), value(1)
                    change = None if (off_value is None or abs(off_value) < 1e-12
                                      or key == "cte_signed_mean_m") else 100 * (on_value / off_value - 1)
                    metrics[metric] = dict(unit=unit, n_pairs=len(valid), active=on_value,
                                          off=off_value, active_vs_off_percent=change)
            scopes[scope] = dict(
                n_cases=len(pairs), success_active=sum(on["success_place"] for on, _ in pairs),
                success_off=sum(off["success_place"] for _, off in pairs), both_success=len(common_success),
                active_only=sum(on["success_place"] and not off["success_place"] for on, off in pairs),
                off_only=sum(off["success_place"] and not on["success_place"] for on, off in pairs),
                both_failed=sum(not on["success_place"] and not off["success_place"] for on, off in pairs),
                metrics_paired_success=metrics)
        result[agent] = scopes
    return result


def report(output, rows, reference, agents):
    groups = aggregate(rows, reference, agents)
    common.dump(output / "summary.json", dict(
        complete=True, methods=[a + "_off" for a in agents], episodes=rows,
        active_reference=str(reference["folder"]), groups=groups,
        continuous_metrics_cohort="same case successful in ON and OFF, separately for each actor",
        active_vs_off_percent_definition="100*(active/off-1); negative means lower, not overall superiority"))
    with (output / "comparison_on_off.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("agent", "scope", "metric", "unit", "n_pairs", "active", "off", "active_vs_off_percent"))
        for agent, scopes in groups.items():
            for scope, group in scopes.items():
                writer.writerow((agent, scope, "success_count", "episodes", group["n_cases"],
                                 group["success_active"], group["success_off"], ""))
                for metric, data in group["metrics_paired_success"].items():
                    writer.writerow((agent, scope, metric, data["unit"], data["n_pairs"],
                                     data["active"], data["off"], data["active_vs_off_percent"]))
    with (output / "case_comparison.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow((*CASE_FIELDS, "agent", "success_active", "success_off", "stop_active", "stop_off"))
        for row in rows:
            on = reference["by_case"][row["case_id"]][row["agent"]]
            writer.writerow((*(row[k] for k in CASE_FIELDS), row["agent"], on["success_place"],
                             row["success_place"], on["termination_reason"], row["termination_reason"]))
    with (output / "failures_off.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=(*CASE_FIELDS, "agent", "termination_reason", "total_time_s"),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(row for row in rows if not row["success_place"])
    return groups


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    p.add_argument("--output", type=Path, help="Carpeta nueva dentro de INCREMENTAL_SAC/data_vx")
    p.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    p.add_argument("--methods", nargs="+", choices=AGENTS, default=list(AGENTS))
    p.add_argument("--maps", type=int, nargs="+", choices=range(1, 5))
    p.add_argument("--case-ids", nargs="+")
    p.add_argument("--limit-cases", type=int, help="Prueba corta; no constituye el panel final")
    p.add_argument("--verify-only", action="store_true", help="Valida referencia y modelos sin simular")
    p.add_argument("--resume", action="store_true", help="Continúa casos pendientes de --output")
    return p


def main():
    args = parser().parse_args()
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("Métodos duplicados")
    reference = load_reference(args.reference)
    panel = select_cases(reference, args)
    if args.verify_only:
        print("VERIFIED", json.dumps(dict(reference=str(reference["folder"]), cases=len(panel),
              methods=args.methods, new_episodes=len(panel) * len(args.methods),
              seed=reference["config"]["seed"], baseline_new_episodes=0)), flush=True)
        return
    if args.resume and args.output is None:
        raise ValueError("--resume requiere --output con la carpeta de la corrida anterior")
    stamp = datetime.now(ZoneInfo("America/Santiago")).strftime("%Y%m%d_%H%M%S")
    output = (args.output or ROOT / "data_vx" / f"ablation_off_r11_r3_{stamp}_{os.getpid()}").expanduser().resolve()
    if not output.is_relative_to(ROOT / "data_vx") or output.is_relative_to(reference["folder"]):
        raise ValueError("Salida nueva dentro de data_vx y fuera de la referencia activa")
    config = dict(
        experiment="r11_r3_residual_off", cases=panel, methods=args.methods,
        reference=str(reference["folder"]), seed=reference["config"]["seed"],
        reference_config_sha256=common.digest(reference["folder"] / "config.json"),
        reference_summary_sha256=common.digest(reference["folder"] / "summary.json"),
        reference_sources=reference["config"]["source_sha256"],
        extractor_sha256=common.digest(__file__),
        r1_manifest_sha256=common.digest(reference["bundle"]["manifest_path"]),
        r1_model_sha256=reference["config"]["r1_model_sha256"],
        r1_normalizer_sha256=reference["config"]["r1_normalizer_sha256"],
        r3_model_sha256=reference["config"]["r3_model_sha256"],
        placement=reference["config"]["placement"], phase_budgets=[1000, 4500, 1000],
        xy_noise_m=.01, yaw_noise_rad=.1, inference="CPU deterministic; one Torch thread per worker",
        no_training=True, baseline_new_episodes=0, active_new_episodes=0,
        off_r11="same actor actions and history; physical residual_enabled=False; delta torque zero",
        off_r3="same actor arm actions; executed lookahead channel zero; L=0.30 m",
        versions=dict(numpy=np.__version__, torch=common.torch.__version__, mujoco=common.mujoco.__version__))
    if args.resume:
        if read_json(output / "config.json") != config:
            raise ValueError("La referencia, el extractor, los modelos o los casos cambiaron; no reanudar")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "logs").mkdir()
        common.dump(output / "config.json", config)
    reused = {c["case_id"]: completed_rows(output, c, args.methods, reference["folder"]) for c in panel}
    rows = [r for c in panel for r in (reused[c["case_id"]] or [])]
    pending = [c for c in panel if reused[c["case_id"]] is None]
    start = time.monotonic()
    def status(state):
        common.dump(output / "status.json", dict(state=state, completed_cases=len(rows) // len(args.methods),
                    total_cases=len(panel), elapsed_s=time.monotonic() - start))
    status("RUNNING")
    print("START", json.dumps(dict(output=str(output), cases=len(panel), pending_cases=len(pending),
          new_episodes=len(pending) * len(args.methods), baseline_new_episodes=0)), flush=True)
    pool = None
    futures = []
    try:
        if pending:
            pool = concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
                mp_context=multiprocessing.get_context("spawn"), initializer=init_worker,
                initargs=(str(reference["bundle"]["manifest_path"]), str(reference["r3"])))
            futures = [pool.submit(worker, c, str(output), str(reference["folder"]), args.methods) for c in pending]
            for future in concurrent.futures.as_completed(futures):
                rows.extend(future.result())
                done = len(rows) // len(args.methods)
                status("RUNNING")
                if done in (1, max(1, len(panel) // 2), len(panel)):
                    print("PROGRESS", json.dumps(dict(cases=done, total=len(panel),
                          elapsed_s=round(time.monotonic() - start, 1))), flush=True)
        if len(rows) != len(panel) * len(args.methods):
            raise RuntimeError("Panel OFF incompleto")
        rows.sort(key=lambda r: (r["map_id"], r["mass_kg"], r["rep"], r["agent"]))
        groups = report(output, rows, reference, args.methods)
        status("COMPLETE")
        print("COMPLETE", json.dumps(dict(output=str(output), success={a: {
              "active": groups[a]["global"]["success_active"], "off": groups[a]["global"]["success_off"]}
              for a in args.methods})), flush=True)
    except BaseException as error:
        for future in futures:
            future.cancel()
        if not isinstance(error, KeyboardInterrupt):
            common.dump(output / "error.json", dict(traceback=traceback.format_exc()))
        status("PAUSED" if isinstance(error, KeyboardInterrupt) else "FAILED")
        raise
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    main()
