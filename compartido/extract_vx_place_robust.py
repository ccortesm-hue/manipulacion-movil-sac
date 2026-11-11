#!/usr/bin/env python3
"""Paired baseline / final R1.1 extraction ending at physical stable placement.

New evaluation protocol only: never trains or rewrites historical modules.
Both methods have 200 simulated seconds, identical physical initial conditions,
and hold the current arm reference after release (no return-home command).
The native R1.1 phase budgets remain in the actor's observation unchanged.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import csv
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / 'models/vx_r1_r2_sequence_20260909/r1_1_student/final/final_student.json'
MASSES = (.5, .75, 1., 1.25, 1.5)
METHODS = ('baseline', 'r1_1')
WORKER = {}
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/vx_place_mpl')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def make_cases(maps, masses, reps, seed):
    return [dict(case_id=f'm{m}_mass{k}_rep{r:02d}', map_id=m,
                 mass_kg=float(mass), rep=r, case_seed=seed+m*100000+k*1000+r)
            for m in maps for k, mass in enumerate(MASSES) if mass in masses
            for r in range(reps)]


def evaluation_environment(base_class):
    """Keep all trained controller/reward logic until release, no native stop."""
    import mujoco
    import numpy as np

    class PlaceOnly(base_class):
        P2_RETURN_HOME_AFTER_RELEASE = False

        def reset(self, **kwargs):
            self.place_recorder = None
            self.place_hold_reference = None
            self.place_release_step = None
            self.place_hold_intervals = 0
            self.place_native_diagnostics = []
            return super().reset(**kwargs)

        def _apply_arm_action(self, arm_act):
            if int(self.phase_state) == 2 and self._released:
                if self.place_hold_reference is None:
                    self.place_hold_reference = self.data.ctrl[self.arm_act_ids[:5]].copy()
                self.arm_target = self.place_hold_reference.astype(np.float32).copy()
                self.data.ctrl[self.arm_act_ids[:5]] = self.place_hold_reference
                self._p2_returning_home = False
                self.place_hold_intervals += 1
            else:
                super()._apply_arm_action(arm_act)

        def _apply_gripper_logic(self, release_ready):
            super()._apply_gripper_logic(release_ready)
            if int(self.phase_state) == 2 and self._released and self.place_release_step is None:
                self.place_release_step = int(self._steps)
                self.place_hold_reference = self.data.ctrl[self.arm_act_ids[:5]].copy()

        def _check_termination(self, phase_eval):
            native = super()._check_termination(phase_eval)
            if native[0] or native[1]:
                entry = dict(step=int(self._steps), reason=native[2] or native[3])
                if not self.place_native_diagnostics or entry['reason'] != self.place_native_diagnostics[-1]['reason']:
                    self.place_native_diagnostics.append(entry)
            # New common physical observer is the ONLY episode stopping rule.
            return False, False, '', ''

        def do_simulation(self, ctrl, n_frames):
            if int(n_frames) != 5:
                raise ValueError('Common protocol requires five physics substeps')
            self.data.ctrl[:] = ctrl
            for _ in range(5):
                mujoco.mj_step(self.model, self.data)
                if self.place_recorder is not None:
                    self.place_recorder.substep()

    return PlaceOnly


class BaselineDriver:
    """Native MM state-machine prefix, delegating PID, PP and IK unchanged.

    The only changed tail is HOLD_RELEASE instead of hover/carry/return-home.
    Phase labels describe the command-producing state, not the next state.
    """
    def __init__(self, mm, ik):
        self.mm, self.ik = mm, ik
        self.state = 'ARM_PICK'
        self.hold = None
        self.release_step = None
        self.hold_intervals = 0
        self.steps = 0

    @property
    def phase(self):
        return 0 if self.state == 'ARM_PICK' else 2 if self.state in ('ARM_PLACE', 'HOLD_RELEASE') else 1

    def control(self):
        mm, state = self.mm, self.state
        phase = self.phase
        self.steps += 1
        if state in ('REVERSE', 'NAV_TO_GOAL'):
            mm.control_base_step_nav_to_goal()
        else:
            mm.stop_wheels()
        if state in ('ARM_PICK', 'REVERSE', 'PLAN_GOAL', 'NAV_TO_GOAL') and (mm._grasp_successful or mm._carry_done):
            if mm._carry_done:
                mm._hold_carry_posture(mm.model, mm.data)
            else:
                mm._hold_grip_closed(mm.model, mm.data)
        if state == 'ARM_PICK':
            mm.run_arm_pick_and_carry_to_setpoint(mm.model, mm.data, self.ik)
            if mm._carry_done:
                self.state = 'REVERSE'
                mm.start_reverse(distance_m=.35)
        elif state == 'REVERSE':
            if mm._reverse_done:
                self.state = 'PLAN_GOAL'
        elif state == 'PLAN_GOAL':
            mm._reset_place_task()
            if not mm.replan_path_to_goal_for_current_map():
                raise RuntimeError('Frozen RRT route unavailable; extraction invalid, not a controller failure')
            mm.stop_wheels()
            mm.reset_pid_accums()
            mm.zero_base_qvel()
            self.state = 'NAV_TO_GOAL'
        elif state == 'NAV_TO_GOAL':
            if mm._reached_final:
                self.state = 'ARM_PLACE'
        elif state == 'ARM_PLACE':
            mm.run_arm_place_on_goal(mm.model, mm.data, self.ik)
            if mm._drop_latched:
                self.hold = mm.data.ctrl[mm.arm_act_ids].copy()
                self.release_step = self.steps
                self.state = 'HOLD_RELEASE'
        elif state == 'HOLD_RELEASE':
            mm.data.ctrl[mm.arm_act_ids] = self.hold
            mm._open_grip(mm.model, mm.data)
            self.hold_intervals += 1
        else:
            raise RuntimeError(f'Unexpected baseline state: {state}')
        return phase


def init_worker(manifest, output):
    import numpy as np
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from vx_final_student_contract import validate_final_student, load_environment_class
    from evaluate_zenno_vx_r1_1 import _SpaceOnlyEnv
    import DATA_BASELINES_V9 as adapter

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    out = Path(output)
    log = (out / 'logs' / f'worker_{os.getpid()}.log').open('a', buffering=1)
    # All native diagnostic printing is kept, but outside the concise main log.
    sys.stdout = log
    sys.stderr = log
    bundle = validate_final_student(manifest, expected_experiment='r1_1')
    env = evaluation_environment(load_environment_class(bundle))(
        render_mode=None, debug_rewards=False, **bundle['manifest']['environment']['kwargs'])
    env.CUBE_XY_NOISE_RANGE, env.CUBE_YAW_NOISE_RANGE_RAD = .01, .1
    env.set_eval_mode(True)
    if list(env.phase_budgets) != bundle['manifest']['phase_budgets']:
        raise ValueError('Native phase budgets changed')
    norm = VecNormalize.load(bundle['vecnormalize_path'], DummyVecEnv([_SpaceOnlyEnv]))
    norm.training, norm.norm_reward = False, False
    if np.shape(norm.obs_rms.mean) != (124,):
        raise ValueError('Invalid normalizer observation shape')
    model = SAC.load(bundle['model_path'], device='cpu', custom_objects={'buffer_size': 1})
    if model.observation_space.shape != (124,) or model.action_space.shape != (7,):
        raise ValueError('Only the assessed final 124D/7D single actor is allowed')
    model.policy.set_training_mode(False)
    for p in model.policy.parameters():
        p.requires_grad_(False)
    adapter.install_v9_protocol()
    import MM_TEST as tuned
    mm = tuned.mm
    tuned.configure_tuned_controllers()
    tuned.validate_routes(adapter.RUTAS_SAC, [1, 2, 3, 4])
    mm.MAPS_DIR, mm.MAPS_LIST = str(adapter.RUTAS_SAC), [1, 2, 3, 4]
    mm.HEADLESS, mm.REAL_TIME, mm.TRACKING_CONTROLLER = True, False, 'pure_pursuit'
    mm.LOAD_ROUTES_ON_START = True
    mm.SAVE_ROUTES_PER_MAP = False
    mm.SAVE_GENERATED_ROUTE_PLOTS = False
    mm.GENERATE_NEW_RRT_ROUTE_EACH_EPISODE = False
    mm.SAVE_TRACKING_ERROR_PLOTS = False
    mm.CUBE_XY_NOISE_RANGE, mm.CUBE_YAW_NOISE_RANGE_RAD = .01, .1
    mm.clamp_cube_xy_to_pedestal = lambda xy: np.asarray(xy, dtype=float).copy()
    mm.MAP_CACHE.clear()
    mm.load_map_routes_from_folder(mm.MAPS_DIR, mm.MAPS_LIST)
    ik = mm.ArmIK(mm.model, mm.data, mm.JOINTS, mm.EEF_SITE_NAME, mm.ACTS_ARM,
                  params=mm.IK_PARAMS_OVERRIDE)
    WORKER.update(env=env, model=model, norm=norm, mm=mm, ik=ik, output=out, log=log)


def one_pair(case):
    import mujoco
    import numpy as np
    from vx_place_protocol import Recorder
    from vx_place_initials import canonicalize_and_audit
    env, mm, ik = (WORKER[k] for k in ('env', 'mm', 'ik'))
    out = WORKER['output']
    start = time.monotonic()
    env.FIXED_MAP_ID = case['map_id']
    env.set_eval_mass(case['mass_kg'])
    env.reset(seed=case['case_seed'])
    mm._current_map_id = case['map_id']
    mm._cube_noise_rng = np.random.default_rng(case['case_seed'])
    mm.reset_world_like_rl_for_map(case['map_id'], cube_mass_kg=case['mass_kg'])
    mm._set_home_pose(mm.model, mm.data, ik)
    audit = canonicalize_and_audit(env, mm)
    if not audit['passed']:
        raise ValueError('Initial physical state audit failed')
    dump(out / 'initial_states' / (case['case_id'] + '.json'), audit)
    route = np.asarray(json.loads((ROOT / 'RUTAS_SAC' / f'map_{case["map_id"]}.json').read_text())['path'])
    route_length = float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum())
    rows = []
    for method in METHODS:
        model, data = (mm.model, mm.data) if method == 'baseline' else (env.model, env.data)
        recorder = Recorder(model, data, route)
        driver = BaselineDriver(mm, ik) if method == 'baseline' else None
        env.place_recorder = recorder if method == 'r1_1' else None
        obs = env._get_obs() if method == 'r1_1' else None
        # Native phase counters are diagnostic only. The baseline has no such
        # training budgets; compare against the same 1000/4500/1000 yardstick.
        phase_counts = [0, 0, 0]
        release_step = None
        for step in range(20000):
            phase = driver.phase if driver else int(env.phase_state)
            phase_counts[phase] += 1
            native_exceeded = phase_counts[phase] >= env.phase_budgets[phase]
            if driver:
                driver.control()
                for _ in range(5):
                    mujoco.mj_step(model, data)
                    recorder.substep()
                release_step = driver.release_step
            else:
                action, _ = WORKER['model'].predict(
                    WORKER['norm'].normalize_obs(np.asarray(obs, dtype=np.float32).copy()), deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    raise RuntimeError('Native stopping escaped the evaluation-only wrapper')
                release_step = env.place_release_step
            # The common observer additionally restricts tracking geometrically
            # to the interior of the route (see its shared gate below).
            reason = recorder.end_interval(phase, phase == 1, native_deadline=native_exceeded)
            if reason is not None:
                break
        else:
            raise RuntimeError('Common observer failed to enforce its deadline')
        traces = recorder.traces()
        # Same geometric tracking window, independent of each controller's mode
        # names. Keep stopped/intervening samples inside that window.
        arc = traces['route_progress'] * route_length
        tracking = ((traces['phase'] == 1) & (arc >= .30) & (arc <= route_length-.55))
        traces['tracking_valid'] = tracking
        summary = recorder.summary()
        for dest, values, rms in (
            ('cte_rms_m', traces['cte_signed_m'], True),
            ('cte_signed_mean_m', traces['cte_signed_m'], False),
            ('distance_to_route_rms_m', traces['distance_to_route_m'], True),
            ('yaw_rms_rad', (traces['base_yaw_rad']-traces['path_yaw_rad']+np.pi)%(2*np.pi)-np.pi, True)):
            selected = values[tracking]
            summary[dest] = float(np.sqrt(np.mean(selected**2)) if rms else np.mean(selected)) if len(selected) else None
        summary['tracking_samples'] = int(tracking.sum())
        relative = Path('traces') / method / (case['case_id'] + '.npz')
        np.savez_compressed(out / relative, **traces)
        summary.update(case, method=method, trace_path=str(relative),
                       initial_state_path=str(Path('initial_states') / (case['case_id'] + '.json')),
                       release_step=release_step,
                       hold_after_release_intervals=int(driver.hold_intervals if driver else env.place_hold_intervals),
                       native_phase_counts=phase_counts,
                       native_diagnostics=[] if driver else env.place_native_diagnostics,
                       baseline_final_state=driver.state if driver else None)
        dump(out / 'case_results' / method / (case['case_id'] + '.json'), summary)
        rows.append(summary)
        env.place_recorder = None
    return dict(case_id=case['case_id'], rows=rows, wall_time_s=time.monotonic()-start)


def source_hashes():
    from vx_final_student_contract import sha256
    names = ['extract_vx_place_robust.py', 'vx_place_protocol.py', 'vx_place_initials.py',
             'PDF_VX_PLACE.py', 'zenno_vx_r1_1.py', 'DATA_BASELINES_V9.py',
             'vx_final_student_contract.py', 'evaluate_zenno_vx_r1_1.py']
    paths = [ROOT / name for name in names]
    paths += [ROOT.parent / name for name in ('DATA_BASELINES.py', 'MM_TEST.py', 'NEW_MM_VISUAL.py')]
    for model_dir in (ROOT / 'Zenno_Model', ROOT.parent / 'Zenno_Model'):
        paths += sorted(model_dir.rglob('*.xml'))
        paths += sorted(model_dir.rglob('*.stl'))
        paths += sorted(model_dir.rglob('*.obj'))
    return {str(path): sha256(path) for path in paths}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--maps', type=int, nargs='+', default=[1, 2, 3, 4])
    p.add_argument('--masses', type=float, nargs='+', default=list(MASSES))
    p.add_argument('--reps', type=int, default=10)
    p.add_argument('--seed', type=int, default=20260909)
    p.add_argument('--workers', type=int, choices=(1, 2), default=2)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--skip-report', action='store_true')
    p.add_argument('--detach', action='store_true', help='Launch independently, keeping log/status and automatic report')
    return p


def run(args):
    from vx_final_student_contract import validate_final_student, sha256
    from vx_place_protocol import protocol_metadata
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT / 'data_vx'):
        raise ValueError('Outputs must be inside INCREMENTAL_SAC/data_vx')
    if len(set(args.maps)) != len(args.maps) or not set(args.maps) <= {1, 2, 3, 4}:
        raise ValueError('Invalid maps')
    if len(set(args.masses)) != len(args.masses) or not set(args.masses) <= set(MASSES):
        raise ValueError('Invalid masses')
    if args.reps < 1 or not 0 <= args.seed < 2**32-500000:
        raise ValueError('Invalid cases or seed')
    bundle = validate_final_student(args.manifest, expected_experiment='r1_1')
    cases = make_cases(args.maps, args.masses, args.reps, args.seed)
    hashes = source_hashes()
    protocol = dict(schema_version=1, protocol='vx_paired_robust_stable_place_v1',
        methods=list(METHODS), routes_dir=str(ROOT / 'RUTAS_SAC'),
        route_hashes=bundle['manifest']['route_hashes'],
        final_student_manifest=str(bundle['manifest_path']),
        final_student_manifest_sha256=sha256(bundle['manifest_path']),
        model_sha256=bundle['manifest']['artifacts']['model']['sha256'],
        vecnormalize_sha256=bundle['manifest']['artifacts']['vecnormalize']['sha256'],
        single_actor=True, observation_dim=124, action_dim=7, deterministic=True,
        device='cpu', training_performed=False, expected_cases=len(cases),
        cube_xy_noise_m=.01, cube_yaw_noise_rad=.1, cube_xy_clamp='none',
        noise_distribution='independent uniform x,y,yaw, paired physical reset',
        seed=args.seed, repetitions=args.reps, maps=args.maps, masses=args.masses,
        episode_seed_policy='base + map*100000 + canonical_mass_index*1000 + zero_based_rep',
        dt_control_s=.01, physical_timestep_s=.002, frameskip=5, global_deadline_s=200.,
        phase_budgets_in_observation=bundle['manifest']['phase_budgets'],
        native_phase_deadlines_enforced=False,
        native_deadline_note='Diagnostic same phase yardstick for baseline; baseline has no native phase truncations',
        success_definition='Physical cube support on goal pedestal, released and stable for 0.2 s; home not required',
        cost_cutoff='stable_place_no_home', energy_includes_gripper=True,
        energy_definition='Sum abs(actuator_force*actuator_velocity)*0.002 over all twelve actuators; absolute mechanical work, not electrical energy',
        failed_episode_cost_cutoff='common physical fall or 200 s global deadline; not included in paired-success efficiency comparisons',
        post_release_control='Both methods hold the last arm command, open gripper, no home or hover withdrawal until physical stable placement',
        tracking_mask='P1 and .30 <= closest arc length <= route length-.55 m; identical geometric window, including stops',
        cte_definition='signed = left normal dot position-minus-projection; abs = absolute signed lateral deviation; Euclidean distance stored separately',
        source_hashes=hashes, placement=protocol_metadata())
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'protocol.json').exists():
        if not args.resume or json.loads((out / 'protocol.json').read_text()) != protocol:
            raise ValueError('Output already contains a different run or --resume was not requested')
    elif args.resume:
        raise ValueError('Cannot resume a run without its protocol manifest')
    dump(out / 'protocol.json', protocol)
    dump(out / 'case_manifest.json', cases)
    for folder in ['logs', 'initial_states'] + [f'{kind}/{method}' for kind in ('case_results', 'traces') for method in METHODS]:
        (out / folder).mkdir(parents=True, exist_ok=True)
    done, pending = [], []
    for case in cases:
        files = [out / 'case_results' / method / (case['case_id'] + '.json') for method in METHODS]
        if args.resume and all(path.exists() for path in files):
            rows = [json.loads(path.read_text()) for path in files]
            for row in rows:
                if any(row[k] != v for k, v in case.items()) or not (out / row['trace_path']).exists():
                    raise ValueError('Cannot resume inconsistent completed case')
            done.extend(rows)
        else:
            pending.append(case)
    started = time.time()
    def status(stage, **extra):
        state = dict(stage=stage, pid=os.getpid(), output=str(out), expected_pairs=len(cases),
                     completed_pairs=len(done)//2, elapsed_wall_s=time.time()-started, **extra)
        dump(out / 'status.json', state)
        print(json.dumps(state, ensure_ascii=False), flush=True)
    status('EXTRACTING', workers=args.workers, pending_pairs=len(pending))
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
                mp_context=multiprocessing.get_context('spawn'), initializer=init_worker,
                initargs=(str(bundle['manifest_path']), str(out))) as pool:
            futures = {pool.submit(one_pair, case): case for case in pending}
            try:
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    done.extend(result['rows'])
                    status('EXTRACTING', last_case=result['case_id'], last_pair_wall_s=result['wall_time_s'],
                           successes={method:sum(bool(row['success_place']) for row in done if row['method']==method) for method in METHODS})
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        if source_hashes() != hashes:
            raise ValueError('A source or asset changed during extraction; results not released')
        validate_final_student(args.manifest, expected_experiment='r1_1')
        audits = [json.loads((out / 'initial_states' / (case['case_id'] + '.json')).read_text()) for case in cases]
        if len(done) != 2*len(cases) or not all(a['passed'] for a in audits):
            raise ValueError('Incomplete or physically mismatched paired dataset')
        done.sort(key=lambda row:(row['map_id'], row['mass_kg'], row['rep'], row['method']))
        summary = dict(complete=True, expected_cases=len(cases), methods=list(METHODS), episodes=done,
                       initial_state_audit=dict(passed=True, pairs=len(audits), snapshots='initial_states/*.json'),
                       source_hashes_unchanged=True, wall_time_s=time.time()-started)
        dump(out / 'summary.json', summary)
        with (out / 'episodes.csv').open('w', newline='') as file:
            fields = [k for k,v in done[0].items() if not isinstance(v, (list,dict))]
            writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(done)
        status('REPORTING')
        if not args.skip_report:
            subprocess.run([sys.executable, str(ROOT / 'PDF_VX_PLACE.py'), '--input-dir', str(out)], check=True)
        status('COMPLETE', successes={method:sum(bool(row['success_place']) for row in done if row['method']==method) for method in METHODS})
    except BaseException:
        status('FAILED', error=traceback.format_exc())
        raise
    finally:
        lock.close()


def detached_command(args):
    """Resolve user paths BEFORE the detached child changes working directory."""
    command = [sys.executable, str(Path(__file__).resolve()),
               '--manifest', str(args.manifest.resolve()),
               '--output-dir', str(args.output_dir.resolve()),
               '--maps', *map(str, args.maps), '--masses', *map(str, args.masses),
               '--reps', str(args.reps), '--seed', str(args.seed), '--workers', str(args.workers)]
    if args.resume:
        command.append('--resume')
    if args.skip_report:
        command.append('--skip-report')
    return command


def main():
    args = parser().parse_args()
    if args.detach:
        out = args.output_dir.resolve()
        if not out.is_relative_to(ROOT / 'data_vx'):
            raise ValueError('Invalid output root')
        if out.exists() and not args.resume:
            raise FileExistsError('Choose a fresh output directory')
        out.mkdir(parents=True, exist_ok=True)
        command = detached_command(args)
        with (out / 'pipeline.log').open('a') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        print(json.dumps(dict(pid=process.pid, output=str(out), log=str(out/'pipeline.log'),
                              status=str(out/'status.json'), automatically_generates_report=not args.skip_report)))
    else:
        # Defer to the already-running training when CPUs are contended.
        with contextlib.suppress(OSError):
            os.nice(5)
        run(args)


if __name__ == '__main__':
    main()
