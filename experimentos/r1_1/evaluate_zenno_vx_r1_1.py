#!/usr/bin/env python3
"""Diagnóstico pareado del residual de torque: episodios completos, sin entrenar.

API para el especialista: action_fn(observación_cruda, ambiente) -> acción 7D.
evaluate_cases mantiene el reset físico y la política determinista de P0/P2;
el callback puede sustituir sólo las acciones autorizadas durante P1.
"""
import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/vx_r1_1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import gymnasium as gym
import mujoco
import numpy as np
from zenno_vx_r1_1 import ZennoVXR11
from vx_pose_metrics import RoutePoseMetric

ROOT = Path(__file__).resolve().parent
MASSES = (.5, .75, 1., 1.25, 1.5)
DEFAULT_MODEL = ROOT/'models/zenno_vx_r1_run2/best/model.zip'


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def experiment_hashes():
    return dict(sources={name:sha256(ROOT/name) for name in
                        ('zenno_vx_r1_1.py', 'evaluate_zenno_vx_r1_1.py', 'vx_pose_metrics.py')},
                routes={p.name:sha256(p) for p in sorted((ROOT/'RUTAS_SAC').glob('*.json'))})


def canonical_cases(maps=(1, 2, 3, 4), masses=MASSES, reps=2, seed=20260711,
                    profile='robust'):
    """Índice de masa canónico: filtrar masas no cambia las semillas."""
    if reps < 1:
        raise ValueError('reps debe ser positivo')
    result = []
    for map_id in maps:
        if map_id not in (1, 2, 3, 4):
            raise ValueError('Mapa fuera de RUTAS_SAC')
        for mass in masses:
            matches = [i for i, value in enumerate(MASSES) if np.isclose(mass, value, atol=1e-12)]
            if len(matches) != 1:
                raise ValueError(f'Masa no canónica: {mass}')
            for rep in range(reps):
                result.append(dict(map=int(map_id), mass=float(mass), repetition=rep,
                                   seed=int(seed+map_id*100000+matches[0]*1000+rep),
                                   profile=profile))
    return result


class _SpaceOnlyEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (124,), dtype=np.float32)
    action_space = gym.spaces.Box(-1., 1., (7,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(124, dtype=np.float32), {}

    def step(self, action):
        raise RuntimeError('Ambiente exclusivo para cargar las estadísticas; no simula.')


def load_policy(model_path=DEFAULT_MODEL, vecnormalize_path=None, device='cpu'):
    """SAC de inferencia: replay de una fila; normalizador externo congelado.

    Acepta checkpoints de torque R1/R1.1, nunca R2 aunque compartan 124D/7D.
    Devuelve (model, norm); cerrar norm al terminar.
    """
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    model_path = Path(model_path).resolve()
    if model_path.is_dir():
        model_path = model_path/'model.zip'
    norm_path = Path(vecnormalize_path).resolve() if vecnormalize_path else model_path.with_name('vecnormalize.pkl')
    for path in (model_path, norm_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    config_path = next((p for p in (model_path.parent/'config.json', model_path.parent.parent/'config.json')
                        if p.is_file()), None)
    if config_path is None:
        raise ValueError('El checkpoint debe conservar config.json de su experimento')
    config = json.loads(config_path.read_text())
    revision = str(config.get('revision', ''))
    if not (revision.startswith('VX_R1_') or revision.startswith('VX_R11_')
            or revision.startswith('VX_R1_1_')):
        raise ValueError(f'No es un checkpoint documentado de torque R1/R1.1: {revision!r}')
    for name, expected in config.get('route_hashes', {}).items():
        route = ROOT/'RUTAS_SAC'/name
        if not route.is_file() or sha256(route) != expected:
            raise ValueError(f'Cambió la ruta empleada por el checkpoint: {name}')
    norm = VecNormalize.load(norm_path, DummyVecEnv([_SpaceOnlyEnv]))
    norm.training = False
    norm.norm_reward = False
    try:
        if np.shape(norm.obs_rms.mean) != (124,):
            raise ValueError('El normalizador debe corresponder a las 124 observaciones')
        model = SAC.load(model_path, env=norm, device=device,
                         custom_objects={'buffer_size':1, 'learning_starts':0})
        if model.observation_space.shape != (124,) or model.action_space.shape != (7,):
            raise ValueError('El actor debe tener 124 observaciones y 7 acciones')
    except BaseException:
        norm.close()
        raise
    return model, norm


def policy_action(model, norm):
    return lambda obs, env: model.predict(norm.normalize_obs(np.asarray(obs).copy()),
                                          deterministic=True)[0]


class DiagnosticEnv(ZennoVXR11):
    """Sólo instrumentación: integra |fuerza*velocidad| en cada paso físico."""

    def __init__(self, **kwargs):
        self._diag_energy = np.zeros(2)
        self._diag_pose_metric = self._diag_pose_path = None
        super().__init__(**kwargs)

    def reset(self, **kwargs):
        self._diag_pose_metric = self._diag_pose_path = None
        self._diag_energy = np.zeros(2)
        return super().reset(**kwargs)

    def step(self, action):
        self._diag_nominal = {}
        return super().step(action)

    def _apply_wheel_torque_with_residual(self, left, right, u_lin, u_ang):
        super()._apply_wheel_torque_with_residual(left, right, u_lin, u_ang)
        self._diag_nominal = {
            int(aid):float(np.clip(raw, -self._actuator_tau_limit(aid), self._actuator_tau_limit(aid)))
            for raw, ids in ((left, self.wheel_left_ids), (right, self.wheel_right_ids)) for aid in ids}

    def _stop_base(self):
        super()._stop_base()
        self._diag_nominal = {int(aid):0. for aid in self.wheel_left_ids+self.wheel_right_ids}

    def do_simulation(self, ctrl, n_frames):
        if int(n_frames) != int(self.frame_skip) or int(n_frames) != 5:
            raise ValueError('La comparación exige cinco pasos físicos por decisión')
        ctrl = np.asarray(ctrl, dtype=np.float64)
        if ctrl.shape != (self.model.nu,):
            raise ValueError('Forma incorrecta del control')
        self.data.ctrl[:] = ctrl
        self._diag_tracking = bool(
            getattr(self, '_last_base_mode', '') == 'tracking'
            and getattr(self, '_plan_ready', False)
            and not any(getattr(self, name, False) for name in
                        ('_reverse_active', '_initial_aligning', '_final_aligning',
                         '_final_radial_approach', '_nav_done')))
        self._diag_pose = None
        if self._diag_tracking:
            path = np.asarray(self._waypoints, dtype=float)
            if self._diag_pose_path is None or not np.array_equal(path, self._diag_pose_path):
                self._diag_pose_metric = RoutePoseMetric(path)
                self._diag_pose_path = path.copy()
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'chassis_frame')
            if sid < 0:
                raise RuntimeError('Falta chassis_frame para medir yaw geométrico')
            rotation = self.data.site_xmat[sid].reshape(3,3)
            yaw = float(np.arctan2(rotation[1,0], rotation[0,0]))
            self._diag_pose = self._diag_pose_metric.sample(self.data.site_xpos[sid,:2], yaw)
        wheel_ids = self.wheel_left_ids+self.wheel_right_ids
        self._diag_delta_nm = np.asarray([ctrl[aid]-self._diag_nominal.get(aid, ctrl[aid])
                                          for aid in wheel_ids])
        self._diag_wheel_commands = ctrl[wheel_ids].copy()
        self._diag_wheel_limits = np.asarray([self._actuator_tau_limit(aid) for aid in wheel_ids])
        self._diag_energy = np.zeros(2)
        for _ in range(int(n_frames)):
            mujoco.mj_step(self.model, self.data)
            for group, ids in enumerate((self.arm_act_ids[:5], wheel_ids)):
                force = self.data.actuator_force[ids]
                velocity = self.data.actuator_velocity[ids]
                self._diag_energy[group] += float(np.sum(np.abs(force*velocity)))*self.model.opt.timestep


def initial_state(env, obs):
    """Estado físico y memorias iniciales, excluyendo el interruptor ON/OFF."""
    data_fields = ('qpos', 'qvel', 'ctrl', 'act', 'qacc_warmstart', 'mocap_pos', 'mocap_quat',
                   'qfrc_applied', 'xfrc_applied', 'eq_active')
    model_fields = ('body_mass', 'body_pos', 'body_inertia', 'geom_friction')
    result = {name:np.asarray(getattr(env.data, name)).tolist()
              for name in data_fields if hasattr(env.data, name)}
    result.update({'model_'+name:np.asarray(getattr(env.model, name)).tolist() for name in model_fields})
    result.update(observation=np.asarray(obs).tolist(), arm_target=np.asarray(env.arm_target).tolist(),
                  map_id=int(env.map_id), cube_mass_kg=float(env._cube_mass),
                  phase=int(env.phase_state), phase_steps=int(env._phase_steps), time_s=float(env.data.time))
    for name in ('_Accel_lin_des', '_Accel_ang_des', '_e_vel1', '_e_vel2', '_e_omega1', '_e_omega2',
                 '_v_des_f', '_omega_des_f', '_omega_des_prev', '_vx_filter', '_vx_delta'):
        value = getattr(env, name, None)
        result[name] = None if value is None else np.asarray(value).tolist()
    return result


def state_hash(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def _rms(values):
    return float(np.sqrt(np.mean(values))) if values else None


def evaluate_episode(action_fn, env, case, variant='on'):
    if variant not in ('on', 'off'):
        raise ValueError('Variante inválida')
    env.residual_enabled = variant == 'on'
    env.set_eval_mode(True)
    env.FIXED_MAP_ID = int(case['map'])
    env.set_eval_mass(float(case['mass']))
    robust = case.get('profile', 'robust') == 'robust'
    env.CUBE_XY_NOISE_RANGE = .01 if robust else 0.
    env.CUBE_YAW_NOISE_RANGE_RAD = .10 if robust else 0.
    obs, _ = env.reset(seed=int(case['seed']))
    initial = initial_state(env, obs)
    energy = np.zeros((3, 2))
    seconds = np.zeros(3)
    cte, yaw, carry, speed, residual = [], [], [], [], []
    near_saturation, exact_saturation, active_seconds = [], [], 0.
    snapshots, handoffs = [], []
    tracking_seconds = terminal_seconds = 0.
    first_terminal = first_nav_done = None
    p1_seen_steps = 0
    previous_phase = 0
    info = {}
    terminated = truncated = False
    pending = {3000, 4500, int(env.MAX_STEPS_1)}
    for step in range(int(env.max_episode_steps)):
        previous_phase = int(env.phase_state)
        action = np.asarray(action_fn(obs, env), dtype=np.float32).reshape(-1)
        if action.shape != (7,) or not np.all(np.isfinite(action)) or np.any(np.abs(action) > 1.00001):
            raise ValueError('El callback debe devolver una acción finita 7D en [-1,1]')
        obs, _, terminated, truncated, info = env.step(action)
        seconds[previous_phase] += float(env.dt)
        energy[previous_phase] += env._diag_energy
        if variant == 'off' and np.max(np.abs(env._diag_delta_nm), initial=0.) > 1e-10:
            raise AssertionError('OFF produjo una corrección de torque: instrumentación incoherente')
        if previous_phase == 1:
            p1_seen_steps += 1
            diag = info.get('p1_diagnostics') or env.phase1_diagnostics()
            carry.append(float(info.get('vx_carry_error_m', 0.))**2)
            speed.append(float(info.get('vx_relative_speed_m_s', 0.))**2)
            if env._diag_tracking:
                tracking_seconds += env.dt
                if env._diag_pose is not None:
                    cte.append(float(env._diag_pose['cte_signed_m'])**2)
                    yaw.append(float(env._diag_pose['heading_error_rad'])**2)
                residual.append(float(np.mean(env._diag_delta_nm**2)))
                ratio = np.abs(env._diag_wheel_commands)/env._diag_wheel_limits
                near_saturation.append(float(np.mean(ratio >= .996)))
                exact_saturation.append(float(np.mean(ratio >= 1.-1e-9)))
                active_seconds += env.dt*bool(getattr(env, '_vx_active', False))
            endpoint = diag.get('endpoint_m', diag.get('endpoint_distance_m'))
            terminal = bool(diag.get('terminal_region', False))
            if endpoint is not None and np.isfinite(endpoint):
                terminal |= endpoint <= env.PURE_PURSUIT_TERMINAL_POSE_DIST_M
            if terminal:
                terminal_seconds += env.dt
                if first_terminal is None:
                    first_terminal = p1_seen_steps*env.dt
            if diag.get('checks', {}).get('nav_done', False) and first_nav_done is None:
                first_nav_done = p1_seen_steps*env.dt
            if p1_seen_steps in pending or terminated or truncated or int(env.phase_state) != 1:
                snapshots.append(dict(p1_step=p1_seen_steps, p1_time_s=p1_seen_steps*env.dt,
                                      **diag))
        if int(env.phase_state) != previous_phase:
            handoffs.append(dict(source=previous_phase, target=int(env.phase_state),
                                 episode_time_s=float(np.sum(seconds)),
                                 carry_error_m=float(np.linalg.norm(env._site_xpos('setpoint')-env._cube_pos())),
                                 relative_speed_m_s=float(np.linalg.norm(env._vx_relative_velocity)),
                                 gripping=bool(env.gripping), arm_target=np.asarray(env.arm_target).tolist()))
        if terminated or truncated:
            break
    reason = info.get('truncation_reason') or info.get('termination_reason')
    if not (terminated or truncated):
        reason = 'evaluator_total_budget_exhausted'
    phases = [bool(info.get(f'phase{i}_success', False)) for i in range(3)]
    row = dict(**case, variant=variant, p1_budget_steps=int(env.MAX_STEPS_1),
               success=bool(info.get('strict_success', False)), phases=phases,
               ended=bool(terminated or truncated), terminal_phase=previous_phase, reason=reason,
               time_s=seconds.tolist(), energy_arm_j=energy[:,0].tolist(), energy_base_j=energy[:,1].tolist(),
               full_time_s=float(seconds.sum()), full_mechanical_work_j=float(energy.sum()),
               cte_rms_m=_rms(cte), yaw_rms_rad=_rms(yaw), carry_rms_m=_rms(carry),
               relative_speed_rms_m_s=_rms(speed), residual_rms_nm=_rms(residual),
               exact_saturation_fraction=float(np.mean(exact_saturation)) if exact_saturation else None,
               near_saturation_fraction=float(np.mean(near_saturation)) if near_saturation else None,
               p1_tracking_time_s=float(tracking_seconds), p1_terminal_time_s=float(terminal_seconds),
               residual_active_time_s=float(active_seconds), first_terminal_time_s=first_terminal,
               first_nav_done_time_s=first_nav_done, p1_snapshots=snapshots, handoffs=handoffs,
               initial_state_sha256=state_hash(initial), p1_reached=bool(phases[0]),
               p1_success_within_30s=bool(phases[1] and seconds[1] <= 30.+1e-8),
               p1_success_within_45s=bool(phases[1] and seconds[1] <= 45.+1e-8))
    return row, initial


def summarize(rows):
    result = []
    for budget, variant in sorted({(r['p1_budget_steps'], r['variant']) for r in rows}):
        group = [r for r in rows if r['p1_budget_steps'] == budget and r['variant'] == variant]
        reached = [r for r in group if r['p1_reached']]
        complete = [r for r in group if r['success']]
        summary = dict(p1_budget_steps=budget, variant=variant, episodes=len(group),
                       strict_success_rate=float(np.mean([r['success'] for r in group])),
                       phase_success_rates=np.mean([r['phases'] for r in group], axis=0).tolist(),
                       reached_p1=len(reached),
                       conditional_p1_success_rate=float(np.mean([r['phases'][1] for r in reached])) if reached else None,
                       conditional_p1_success_within_30s=float(np.mean([r['p1_success_within_30s'] for r in reached])) if reached else None,
                       conditional_p1_success_within_45s=float(np.mean([r['p1_success_within_45s'] for r in reached])) if reached else None,
                       failure_reasons=dict(Counter(str(r['reason']) for r in group if not r['success'])),
                       complete_success_cohort_n=len(complete))
        for metric in ('full_time_s', 'full_mechanical_work_j', 'cte_rms_m', 'yaw_rms_rad',
                       'carry_rms_m', 'relative_speed_rms_m_s', 'residual_rms_nm'):
            values = [r[metric] for r in complete if r[metric] is not None]
            summary['success_only_mean_'+metric] = float(np.mean(values)) if values else None
        result.append(summary)
    paired = []
    index = {(r['p1_budget_steps'], r.get('profile', 'robust'), r['map'], r['mass'], r['seed'], r['variant']):r for r in rows}
    for budget in sorted({r['p1_budget_steps'] for r in rows}):
        pairs = []
        for r in rows:
            if r['p1_budget_steps'] != budget or r['variant'] != 'on':
                continue
            off = index.get((budget, r.get('profile', 'robust'), r['map'], r['mass'], r['seed'], 'off'))
            if off is not None:
                pairs.append((r, off))
        both = [(on, off) for on, off in pairs if on['success'] and off['success']]
        entry = dict(p1_budget_steps=budget, pairs=len(pairs), both_success=len(both),
                     only_on_success=sum(on['success'] and not off['success'] for on, off in pairs),
                     only_off_success=sum(off['success'] and not on['success'] for on, off in pairs))
        for metric in ('full_time_s', 'full_mechanical_work_j', 'cte_rms_m', 'yaw_rms_rad',
                       'carry_rms_m', 'relative_speed_rms_m_s'):
            values = [on[metric]-off[metric] for on, off in both if on[metric] is not None and off[metric] is not None]
            entry['paired_success_delta_on_minus_off_'+metric] = float(np.mean(values)) if values else None
        paired.append(entry)
    return dict(groups=result, paired=paired,
                interpretation='No se calcula un ganador escalar. Costes incluyen home; trabajo mecánico absoluto, no consumo eléctrico. '
                               'Las medias success_only no son comparables causalmente sin usar paired. '
                               'Aumentar presupuesto también cambia la observación de tiempo restante durante P1.')


def evaluate_cases(action_fn, cases, output_dir, budgets=(3000, 4500, 6000),
                   variants=('on', 'off'), progress=None, metadata=None):
    """Retorna {groups, paired, rows}; persiste cada episodio terminado.

    El directorio debe ser nuevo. Perfiles nominal/robust son válidos; no hay
    autoreset vectorizado ni resets sintéticos P1/P2. El emparejamiento incluye
    todas las matrices físicas, controles, memoria PID y observación inicial.
    """
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f'No sobrescribir una evaluación: {output}')
    cases = [dict(case) for case in cases]
    if not cases or not budgets or not variants:
        raise ValueError('Casos, presupuestos y variantes no pueden estar vacíos')
    if len(set(budgets)) != len(budgets) or len(set(variants)) != len(variants):
        raise ValueError('Presupuestos o variantes duplicados')
    if any(v not in ('on', 'off') for v in variants):
        raise ValueError('Variantes permitidas: on/off')
    keys = [(c.get('profile', 'robust'), c['map'], c['mass'], c['seed']) for c in cases]
    if len(set(keys)) != len(keys):
        raise ValueError('Casos duplicados')
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='RUNNING', revision=ZennoVXR11.REVISION, budgets=list(budgets),
                    variants=list(variants), cases=cases, expected_episodes=len(cases)*len(budgets)*len(variants),
                    noise=dict(robust_xy_m=.01, robust_yaw_rad=.10), hashes=experiment_hashes(),
                    scope='diagnostic/development, no holdout claim', metadata=metadata or {})
    dump(output/'manifest.json', manifest)
    rows = []
    try:
        for budget in budgets:
            env = DiagnosticEnv(render_mode=None, debug_rewards=False, p1_budget_steps=int(budget))
            try:
                for index, case in enumerate(cases):
                    expected = None
                    for variant in (variants if index % 2 == 0 else tuple(reversed(variants))):
                        row, initial = evaluate_episode(action_fn, env, case, variant)
                        if expected is not None and row['initial_state_sha256'] != expected:
                            raise AssertionError(f'Reset ON/OFF no idéntico: {case}, presupuesto={budget}')
                        row['initial_pair_verified'] = expected is not None
                        expected = row['initial_state_sha256']
                        rows.append(row)
                        with (output/'episodes.jsonl').open('a') as file:
                            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                        state_path = output/'initial_states'/f'{budget}_{index:04d}.json'
                        if not state_path.exists():
                            dump(state_path, initial)
                        print('[R11-EVAL] '+json.dumps(dict(completed=len(rows), total=manifest['expected_episodes'],
                              budget=budget, variant=variant, map=case['map'], mass=case['mass'], seed=case['seed'],
                              success=row['success'], phases=row['phases'], reason=row['reason'])), flush=True)
                        if progress is not None:
                            progress(len(rows), row)
                    if len(variants) == 2:
                        rows[-2]['initial_pair_verified'] = rows[-1]['initial_pair_verified'] = True
                        with (output/'verified_pairs.jsonl').open('a') as file:
                            file.write(json.dumps(dict(p1_budget_steps=budget, case=case,
                                                       initial_state_sha256=expected))+'\n')
            finally:
                env.close()
        result = summarize(rows)
        result['rows'] = rows
        dump(output/'summary.json', result)
        columns = [key for key, value in rows[0].items() if not isinstance(value, (list, dict))]
        with (output/'episodes.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        manifest['status'] = 'COMPLETE'
        manifest['completed_episodes'] = len(rows)
        dump(output/'manifest.json', manifest)
        return result
    except BaseException as error:
        manifest.update(status='INTERRUPTED', completed_episodes=len(rows), error=repr(error))
        dump(output/'manifest.json', manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--vecnormalize', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--reps', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20260711)
    parser.add_argument('--budgets', type=int, nargs='+', default=[3000, 4500, 6000])
    parser.add_argument('--maps', type=int, nargs='+', default=[1, 2, 3, 4])
    parser.add_argument('--masses', type=float, nargs='+', default=list(MASSES))
    parser.add_argument('--profile', choices=('nominal', 'robust'), default='robust')
    parser.add_argument('--variants', choices=('on', 'off'), nargs='+', default=['on', 'off'])
    parser.add_argument('--cases-json', type=Path, help='Lista explícita de casos map/mass/seed/profile; sustituye la grilla')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('El directorio de salida ya existe')
    cases = json.loads(args.cases_json.read_text()) if args.cases_json else canonical_cases(
        args.maps, args.masses, args.reps, args.seed, args.profile)
    import torch
    torch.set_num_threads(2)
    model, norm = load_policy(args.model, args.vecnormalize, args.device)
    try:
        path = args.model/'model.zip' if args.model.is_dir() else args.model
        norm_path = args.vecnormalize or path.with_name('vecnormalize.pkl')
        result = evaluate_cases(policy_action(model, norm), cases, args.output, args.budgets, args.variants,
                                metadata=dict(model=str(path.resolve()), model_sha256=sha256(path),
                                              normalizer_sha256=sha256(norm_path), device=args.device,
                                              policy='single R1/R1.1 actor, deterministic, full physical P0->P1->P2'))
        print('COMPLETE '+json.dumps({k:result[k] for k in ('groups', 'paired')}, ensure_ascii=False), flush=True)
    finally:
        norm.close()


if __name__ == '__main__':
    main()
