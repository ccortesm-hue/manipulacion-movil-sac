#!/usr/bin/env python3
"""Curriculum P0 -> P2 con BC/DAgger sobre un solo actor SAC VX existente.

No llama learn(): recupera competencia antes de permitir otra etapa SAC online.
El profesor solo interviene en recolección; todas las evaluaciones son autónomas.
"""
import argparse
import contextlib
import copy
import hashlib
import json
import os
import pickle
from pathlib import Path
import sys
import traceback

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/vx_recovery')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from analytic_teacher_vx import VXAnalyticTeacher
from sac_vx import VXSAC
from train_vx import CASES, make_env, configure, predict, evaluate, dump, save, gate
from vx_p2_diagnostics import p2_state
from zenno_vx import ZennoVX

ROOT = Path(__file__).resolve().parent
DATA_KEYS = ('observations', 'actions', 'phase_ids', 'teacher_stages',
             'dagger_round', 'teacher_mask')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows(path, round_id, teacher_phases=(0, 1, 2)):
    with np.load(path, allow_pickle=False) as z:
        phases = z['phase_ids'].astype(np.int8)
        return dict(observations=z['observations'].astype(np.float32),
                    actions=z['actions'].astype(np.float32), phase_ids=phases,
                    teacher_stages=z['teacher_stages'].astype(str),
                    dagger_round=np.full(len(phases), round_id, np.int16),
                    teacher_mask=np.isin(phases, teacher_phases))


def join(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in DATA_KEYS}


def compact(metrics):
    return {k: metrics[k] for k in ('rates', 'phases', 'strict', 'conditional_p2')}


def acceptable(candidate, best, phase):
    """No empeora tasas nominal/robusta ni éxitos acumulados en ninguno de los paneles.

    P0 puede aceptarse aunque aún no aumente el éxito completo: recuperar acceso
    a P1 es un avance curricular. P2 requiere mejorar el éxito completo.
    """
    gains = []
    for name in best:
        new, old = candidate[name], best[name]
        for profile in ('nominal', 'robust'):
            if new['rates'][profile] + 1e-12 < old['rates'][profile]:
                return False
        if any(a + 1e-12 < b for a, b in zip(new['phases'], old['phases'])):
            return False
        gains.append(new['strict'] > old['strict'] + 1e-12)
        if phase == 0:
            gains.extend(a > b + 1e-12 for a, b in
                         zip(new['phases'][:2], old['phases'][:2]))
    return any(gains)


def make_specs(metrics, phase, seed, map1_reps):
    """Cobertura completa + ruido extra mapa 1 + reproducción de fallos de desarrollo."""
    specs = [(m, w, seed + i, True) for i, (m, w) in enumerate(CASES)]
    if phase == 0:
        specs += [(1, w, seed + 100 + i * 100 + r, True)
                  for i, w in enumerate((.5, .75, 1., 1.25, 1.5))
                  for r in range(map1_reps)]
    for panel in metrics.values():
        for row in panel['rows']:
            failed = not row['phases'][0] if phase == 0 else (
                row['phases'][1] and not row['success'])
            if failed:
                specs.append((row['map'], row['mass'], row['seed'], row['profile'] == 'robust'))
    return list(dict.fromkeys(specs))


def collect(model, norm, path, specs, phase_to_teach, beta, round_id):
    """P0: prefijos físicos hasta P1. P2: episodios reales P0 -> P1 -> P2.

    Fuera de la fase docente ejecuta exclusivamente el alumno. Las filas fuera
    de ella solo sirven para destilación; nunca se confunden con etiquetas IK.
    No se carga un estado P2 ni se rescata P0 durante la recolección P2.
    """
    env = make_env()
    arrays = {k: [] for k in DATA_KEYS}
    executed_rows, episode_ids, episodes = [], [], []
    try:
        for ep, (map_id, mass, seed, robust) in enumerate(specs):
            configure(env, (map_id, mass), robust)
            obs, _ = env.reset(seed=seed)
            teacher = VXAnalyticTeacher(env)
            counts = [0, 0, 0]
            trace = []
            for step in range(5000):
                phase = int(env.phase_state)
                student = predict(model, norm, obs)
                taught = phase == phase_to_teach
                # Track teacher FSM in P0/P2 only; P1 is entirely student-controlled.
                if taught:
                    label = teacher.action()
                    stage = teacher.stage
                    executed = (beta * label + (1. - beta) * student).astype(np.float32)
                else:
                    label, executed, stage = student, student, f'p{phase}_retention'
                # Retain sparse off-phase states; keep every supervised transition.
                if taught or step % 5 == 0:
                    values = (obs.copy(), label.copy(), phase, stage, round_id, taught)
                    for key, value in zip(DATA_KEYS, values):
                        arrays[key].append(value)
                    executed_rows.append(executed.copy()); episode_ids.append(ep)
                counts[phase] += 1
                obs, _, term, trunc, info = env.step(executed)
                if phase == 2 and (counts[2] == 1 or counts[2] % 25 == 0 or term or trunc):
                    trace.append(dict(step=counts[2], **p2_state(env)))
                if term or trunc or (phase_to_teach == 0 and int(env.phase_state) != 0):
                    break
            episodes.append(dict(map=map_id, mass=mass, seed=seed, robust=robust,
                beta=beta, counts=counts, phases=[bool(info.get(f'phase{i}_success', False))
                                                for i in range(3)],
                assisted_success=bool(info.get('strict_success', False)),
                reason=info.get('truncation_reason') or info.get('termination_reason'),
                p2_trace=trace))
            print('[COLLECT]', phase_to_teach, ep + 1, len(specs), episodes[-1]['phases'], flush=True)
    finally:
        env.close()
    data = dict(observations=np.asarray(arrays['observations'], np.float32),
                actions=np.asarray(arrays['actions'], np.float32),
                phase_ids=np.asarray(arrays['phase_ids'], np.int8),
                teacher_stages=np.asarray(arrays['teacher_stages'], str),
                dagger_round=np.asarray(arrays['dagger_round'], np.int16),
                teacher_mask=np.asarray(arrays['teacher_mask'], bool))
    np.savez_compressed(path, **data, executed_actions=np.asarray(executed_rows, np.float32),
                        episode_ids=np.asarray(episode_ids, np.int32),
                        replay_valid=np.zeros(len(episode_ids), bool))
    dump(path.with_suffix('.json'), dict(phase_to_teach=phase_to_teach, beta=beta,
         bc_only=True, episodes=episodes,
         assisted_phase_success=float(np.mean([r['phases'][phase_to_teach] for r in episodes]))))
    if not np.any(data['teacher_mask']):
        raise RuntimeError(f'No se alcanzó P{phase_to_teach}; revisar colección antes de continuar.')
    return data


class BalancedRows:
    """Balance por subetapa y prioridad explícita a etiquetas nuevas de la fase activa."""
    def __init__(self, data, phase, seed):
        self.rng = np.random.default_rng(seed)
        selected = (data['phase_ids'] == phase) & data['teacher_mask']
        indices = np.flatnonzero(selected)
        if not len(indices):
            raise ValueError(f'Sin etiquetas docentes de P{phase}')
        latest = np.max(data['dagger_round'][indices])
        recent = indices[data['dagger_round'][indices] == latest]
        self.groups = [[rows[data['teacher_stages'][rows] == stage]
                        for stage in np.unique(data['teacher_stages'][rows])]
                       for rows in (indices, recent)]
        self.retention = [np.flatnonzero(data['phase_ids'] == ph)
                          for ph in (0, 1, 2) if ph != phase]
        if any(not len(rows) for rows in self.retention):
            raise ValueError('Faltan estados de retención.')

    def fit(self, n):
        result = []
        for _ in range(n):
            groups = self.groups[int(self.rng.random() < .65)]
            result.append(self.rng.choice(groups[self.rng.integers(len(groups))]))
        return np.asarray(result)

    def retain(self, n):
        return np.concatenate([self.rng.choice(rows, n, replace=True) for rows in self.retention])


def update_block(model, anchor, norm, data, phase, steps, lr, seed):
    # CPU storage + small CUDA minibatches: do not fill 4 GB GPU with entire dataset.
    sampler = BalancedRows(data, phase, seed)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=lr)
    losses = []
    def inputs(idx):
        obs = norm.normalize_obs(data['observations'][idx].copy())
        return torch.as_tensor(obs, dtype=torch.float32, device=model.device)
    model.actor.set_training_mode(True)
    for _ in range(steps):
        fit_idx, retain_idx = sampler.fit(512), sampler.retain(256)
        x, kept = inputs(fit_idx), inputs(retain_idx)
        target = torch.as_tensor(data['actions'][fit_idx], device=model.device)
        # No override of P0 IK targets: source P0 failures must be allowed to change.
        with torch.no_grad():
            retained_target = anchor(kept, deterministic=True)
            base_target = anchor(x, deterministic=True)[:, 5:]
        pred = model.actor(x, deterministic=True)
        fit = (pred[:, :5] - target[:, :5]).square().mean()
        retention = (model.actor(kept, deterministic=True) - retained_target).square().mean()
        base = (pred[:, 5:] - base_target).square().mean()
        loss = fit + 10. * retention + .1 * base
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.)
        optimizer.step()
        losses.append([float(fit.detach()), float(retention.detach()), float(base.detach())])
    model.actor.optimizer = optimizer
    return dict(first=losses[0], last=losses[-1], mean=np.mean(losses, axis=0).tolist())


def validation(model, norm, out, tag, old_seed, dev_seed):
    return dict(legacy=evaluate(model, norm, out/f'{tag}_legacy40.json', seed=old_seed),
                development=evaluate(model, norm, out/f'{tag}_dev60.json', reps=2, seed=dev_seed))


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    d = 1 + z*z/n
    center = (p + z*z/(2*n)) / d
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return [float(center-half), float(center+half)]


def holdout_comparison(before, after):
    keys = lambda rows: [(r['profile'], r['map'], r['mass'], r['seed']) for r in rows]
    if keys(before['rows']) != keys(after['rows']):
        raise ValueError('Comparación final no pareada.')
    result = {}
    for profile in ('nominal', 'robust'):
        pairs = [(a['success'], b['success']) for a, b in zip(before['rows'], after['rows'])
                 if a['profile'] == profile]
        wins = sum(b for a, b in pairs)
        result[profile] = dict(n=len(pairs), source_successes=sum(a for a, b in pairs),
             best_successes=wins, recovered=sum(not a and b for a, b in pairs),
             regressed=sum(a and not b for a, b in pairs), rate=wins/len(pairs),
             # Nominal is a deterministic scenario census, not independent noise trials.
             wilson95_descriptive=wilson(wins, len(pairs)) if profile == 'robust' else None)
    return result


def preflight(args):
    source = json.loads((args.data_source/'config.json').read_text())
    initial = json.loads((args.init_run/'config.json').read_text())
    if source['revision'] != ZennoVX.REVISION or initial['revision'] != ZennoVX.REVISION:
        raise ValueError('Revisión de ambiente incompatible.')
    for name in ('zenno_vx.py', 'analytic_teacher_vx.py', 'sac_vx.py', 'demo_regularized_sac_v9.py'):
        if sha(ROOT/name) != source['hashes'][name]:
            raise ValueError(f'Contrato fuente modificado: {name}')
    for name, expected in source['route_hashes'].items():
        if sha(ROOT/'RUTAS_SAC'/name) != expected:
            raise ValueError(f'Ruta modificada: {name}')
    # Pickle bytes can change on re-save without changing the observation transform.
    # These are trusted project checkpoints, also deserialized by SB3 below.
    with (args.init_run/'best/vecnormalize.pkl').open('rb') as f:
        initial_norm = pickle.load(f)
    with (args.data_source/'best/vecnormalize.pkl').open('rb') as f:
        source_norm = pickle.load(f)
    for attr in ('clip_obs', 'epsilon', 'norm_obs', 'norm_obs_keys'):
        if getattr(initial_norm, attr, None) != getattr(source_norm, attr, None):
            raise ValueError(f'Normalizador incompatible: {attr}')
    for attr in ('mean', 'var', 'count'):
        if not np.array_equal(getattr(initial_norm.obs_rms, attr), getattr(source_norm.obs_rms, attr)):
            raise ValueError(f'Estadísticas de observación diferentes: {attr}')
    for path in [args.init_run/'best/model.zip', args.data_source/'teacher.npz',
                 args.data_source/'dagger_1.npz', args.data_source/'dagger_2.npz']:
        if not path.is_file():
            raise FileNotFoundError(path)
    old_seed = int(source['seed']) + 1000000
    # Reproduce the already inspected holdout; it is now DEVELOPMENT, not final test.
    dev_seed = int(initial['seed']) + 5000000
    final_seed = args.seed + 20000000
    ranges = [(old_seed, old_seed+12000), (dev_seed, dev_seed+12000),
              (args.seed+100000, args.seed+args.rounds*100000+12000),
              (final_seed, final_seed+12000)]
    for i, (lo, hi) in enumerate(ranges):
        if any(max(lo, a) <= min(hi, b) for a, b in ranges[i+1:]):
            raise ValueError('Semillas de colección/desarrollo/test se solapan.')
    return source, old_seed, dev_seed, final_seed


def run(args, announce):
    source_config, old_seed, dev_seed, final_seed = preflight(args)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA no disponible. Esta corrida no cambia silenciosamente a CPU.')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = args.output
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(revision=ZennoVX.REVISION, device=torch.cuda.get_device_name(0),
        single_actor=True, sac_online_started=False, legacy_seed=old_seed,
        development_seed=dev_seed, final_seed=final_seed,
        hashes={n: sha(ROOT/n) for n in ('zenno_vx.py', 'finetune_vx_recovery.py',
                'analytic_teacher_vx.py', 'train_vx.py', 'sac_vx.py', 'vx_p2_diagnostics.py')},
        route_hashes=source_config['route_hashes'],
        initial_model_sha256=sha(args.init_run/'best/model.zip'),
        normalizer_sha256=sha(args.init_run/'best/vecnormalize.pkl'))
    dump(out/'config.json', config)
    norm = VecNormalize.load(args.init_run/'best/vecnormalize.pkl', DummyVecEnv([make_env]))
    norm.training = False
    try:
        model = VXSAC.load(args.init_run/'best/model.zip', env=norm, device='cuda')
        save(model, norm, out/'initial')
        save(model, norm, out/'best')
        best = validation(model, norm, out, 'initial', old_seed, dev_seed)
        dump(out/'best_metrics.json', best)
        announce('BASELINE', {k: compact(v) for k, v in best.items()})
        parts = [load_rows(args.data_source/'teacher.npz', 0)]
        parts += [load_rows(args.data_source/f'dagger_{i}.npz', i) for i in (1, 2)]
        for i in (1, 2):
            path = args.init_run/f'p2_dagger_{i}.npz'
            if path.is_file():
                parts.append(load_rows(path, i+2, teacher_phases=(2,)))
        history = []
        for rnd in range(1, args.rounds+1):
            for phase in (0, 2):
                # High assistance is not reduced just because another round elapsed.
                competent = min(v['rates']['robust'] for v in best.values()) >= .9
                beta = 1. if phase == 0 and rnd == 1 else (.5 if competent else .7)
                specs = make_specs(best, phase, args.seed+rnd*100000+phase*1000, args.map1_reps)
                extra = collect(model, norm, out/f'collect_r{rnd}_p{phase}.npz',
                                specs, phase, beta, 5+rnd*2+phase)
                parts.append(extra)
                data = join(parts)
                for block in range(1, args.blocks+1):
                    tag = f'r{rnd}_p{phase}_b{block}'
                    # Training-only frozen copy of CURRENT best, never a second deployed agent.
                    anchor = copy.deepcopy(model.actor).eval()
                    for p in anchor.parameters():
                        p.requires_grad_(False)
                    losses = update_block(model, anchor, norm, data, phase, args.steps,
                                          args.lr, args.seed+rnd*100+phase*10+block)
                    del anchor
                    metrics = validation(model, norm, out, tag, old_seed, dev_seed)
                    accepted = acceptable(metrics, best, phase)
                    save(model, norm, out/f'candidate_{tag}')
                    history.append(dict(tag=tag, phase=phase, accepted=accepted, beta=beta,
                        losses=losses, metrics={k: compact(v) for k, v in metrics.items()}))
                    if accepted:
                        best = metrics
                        save(model, norm, out/'best'); dump(out/'best_metrics.json', best)
                    else:
                        model = VXSAC.load(out/'best/model.zip', env=norm, device='cuda')
                    dump(out/'history.json', history)
                    announce('BLOCK', dict(tag=tag, accepted=accepted,
                        best={k: compact(v) for k, v in best.items()}))
                del data
            announce('ROUND', dict(round=rnd, total=args.rounds))
        # Untouched seeds, same device/cases for initial and best. Never select using this test.
        model = VXSAC.load(out/'best/model.zip', env=norm, device='cuda')
        final = evaluate(model, norm, out/'holdout_best.json', reps=args.final_reps, seed=final_seed)
        initial_model = VXSAC.load(out/'initial/model.zip', env=norm, device='cuda')
        before = evaluate(initial_model, norm, out/'holdout_initial.json', reps=args.final_reps, seed=final_seed)
        result = dict(validation={k: compact(v) for k, v in best.items()},
            holdout=holdout_comparison(before, final),
            attempted_bc_updates=len(history)*args.steps,
            accepted_bc_updates=sum(r['accepted'] for r in history)*args.steps,
            sac_online_started=False, single_actor=True,
            empirical_90_gate=all(gate(v, .9) for v in [*best.values(), final]),
            note='Gate empírico, no garantía de 90% poblacional. No arranca SAC automáticamente.')
        dump(out/'result.json', result)
        announce('FINAL', result)
    finally:
        norm.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-source', type=Path, default=ROOT/'models/vx_delta008_uniform_costs_run1')
    p.add_argument('--init-run', type=Path, default=ROOT/'models/vx_p2_retention_run1')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=970000)
    p.add_argument('--rounds', type=int, default=2)
    p.add_argument('--blocks', type=int, default=2)
    p.add_argument('--steps', type=int, default=250, help='Updates BC por bloque, no steps MuJoCo.')
    p.add_argument('--lr', type=float, default=5e-6)
    p.add_argument('--map1-reps', type=int, default=8, help='Prefijos P0 extra por masa en mapa 1.')
    p.add_argument('--final-reps', type=int, default=5, help='Repeticiones robustas por mapa/masa.')
    p.add_argument('--check-only', action='store_true', help='Valida contratos sin CUDA, rollouts ni entrenamiento.')
    args = p.parse_args()
    if min(args.rounds, args.blocks, args.steps, args.map1_reps, args.final_reps) <= 0:
        p.error('Presupuestos enteros positivos.')
    if not np.isfinite(args.lr) or args.lr <= 0 or args.final_reps > 100 or args.map1_reps > 100:
        p.error('LR finito positivo y repeticiones <= 100 (semillas sin solapamiento).')
    if args.seed < 0 or args.seed+20012000 >= 2**32:
        p.error('Seed fuera de rango.')
    for key in ('data_source', 'init_run', 'output'):
        setattr(args, key, getattr(args, key).resolve())
    if args.check_only:
        _, old, dev, final = preflight(args)
        print('CHECK_OK', dict(legacy_seed=old, development_seed=dev, final_seed=final,
                              model=str(args.init_run/'best/model.zip')))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    console = sys.stdout
    def announce(stage, data):
        print(stage, json.dumps(data, ensure_ascii=False), file=console, flush=True)
    announce('START', dict(output=str(args.output), attempted_updates=args.rounds*2*args.blocks*args.steps))
    with (args.output/'pipeline.log').open('w', buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                run(args, announce)
            except BaseException as exc:
                traceback.print_exc(file=log)
                dump(args.output/'error.json', dict(type=type(exc).__name__, message=str(exc)))
                announce('ERROR', dict(message=str(exc), log=str(args.output/'pipeline.log')))
                raise


if __name__ == '__main__':
    main()
