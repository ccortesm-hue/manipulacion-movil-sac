#!/usr/bin/env python3
"""Destila un profesor P1 SAC verificado en UN estudiante VX 124D/7D.

El profesor y el actor auxiliar sólo participan al generar etiquetas. La
inferencia del artefacto final ejecuta exclusivamente el actor del estudiante.
No lanzar antes de finalizar y verificar la evaluación del profesor P1.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import importlib
import json
import os
from pathlib import Path
import signal

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/vx_single_student')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np

from vx_single_student_data import (balanced_indices, bounded_merge, collect, dump,
                                    load_old_bc, replay_rows, save_npz, sha256)

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'models/zenno_vx_r1_run2/best/model.zip'
FAMILIES = {'r1_1': ('zenno_vx_r1_1', 'ZennoVXR11'), 'r2': ('zenno_vx_r2', 'ZennoVXR2')}


def teacher_gate(path):
    """Only a checkpoint with SAC updates and its COMPLETED hashed evaluation.

    This establishes provenance, not teacher superiority; measured development
    success is recorded separately. Initial/random and interrupted checkpoints
    cannot be distilled simply because their weights exist.
    """
    path = Path(path).resolve()
    if path.is_dir():
        path = path / 'model.zip'
    contract = json.loads((path.parent / 'contract.json').read_text())
    if int(contract.get('updates', 0)) <= 0 or int(contract.get('steps', 0)) <= 0:
        raise ValueError('El profesor no tiene actualizaciones SAC completadas.')
    development = path.parent.parent / (path.parent.name + '_development')
    manifest = json.loads((development / 'manifest.json').read_text())
    if manifest.get('status') != 'COMPLETE':
        raise ValueError('La evaluación del profesor no terminó.')
    if manifest.get('metadata', {}).get('teacher_sha256') != sha256(path):
        raise ValueError('La evaluación no corresponde a este checkpoint del profesor.')
    summary = json.loads((development / 'summary.json').read_text())
    rows = summary.get('rows', [])
    if not rows or len(rows) != manifest.get('expected_episodes'):
        raise ValueError('La evaluación del profesor está incompleta.')
    if {row['variant'] for row in rows} != {'on', 'off'}:
        raise ValueError('El profesor necesita una ablación ON/OFF completa.')
    return dict(model=str(path), model_sha256=sha256(path),
                updates=int(contract['updates']), steps=int(contract['steps']),
                evaluation=str(development / 'summary.json'),
                evaluation_sha256=sha256(development / 'summary.json'),
                evaluation_groups=summary.get('groups', []))


def rates(rows):
    result = {}
    for profile in ('nominal', 'robust'):
        selected = [row for row in rows if row.get('profile') == profile]
        if not selected:
            raise ValueError('La selección necesita perfiles nominal y robusto completos.')
        result[profile] = dict(success=float(np.mean([r['success'] for r in selected])),
                               phases=np.mean([r['phases'] for r in selected], axis=0).tolist())
    return result


def promotion(candidate, incumbent):
    """No phase/profile success regression; tie break is a DEVELOPMENT heuristic.

    Costs use only common successful cases. This balanced scalar selects a
    checkpoint, never replaces the Pareto-style experimental comparison.
    """
    c, b = rates(candidate), rates(incumbent)
    for profile in c:
        if c[profile]['success'] + 1e-12 < b[profile]['success']:
            return False, {'reason': 'profile_success_regression', 'rates': c}
        if any(x + 1e-12 < y for x, y in zip(c[profile]['phases'], b[profile]['phases'])):
            return False, {'reason': 'phase_success_regression', 'rates': c}
    improvement = any(c[p]['success'] > b[p]['success'] + 1e-12 or
                      any(x > y + 1e-12 for x, y in zip(c[p]['phases'], b[p]['phases'])) for p in c)
    if improvement:
        return True, {'reason': 'success_improvement_without_regression', 'rates': c}
    key = lambda row: (row.get('profile'), row['map'], row['mass'], row['seed'])
    old = {key(row): row for row in incumbent}
    metrics = ('full_time_s', 'full_mechanical_work_j', 'cte_rms_m', 'yaw_rms_rad', 'carry_rms_m')
    pairs = [(row, old[key(row)]) for row in candidate if row['success'] and
             key(row) in old and old[key(row)]['success']]
    means = {}
    for metric in metrics:
        values = [np.log(row[metric] / previous[metric]) for row, previous in pairs
                  if row.get(metric) is not None and previous.get(metric) is not None
                  and row[metric] > 0 and previous[metric] > 0]
        if values:
            means[metric] = float(np.mean(values))
    score = float(np.mean(list(means.values()))) if len(means) == len(metrics) else None
    return bool(score is not None and score < -.001), dict(
        reason='development_balanced_cost_tiebreak', common_success_pairs=len(pairs),
        mean_log_ratios=means, score=score, rates=c)


def build_manifest(family, environment_class, output, budget, validation, provenance,
                   training_counts, *, root=ROOT):
    output, root = Path(output), Path(root)
    module, class_name = FAMILIES[family]
    report = output / 'holdout_report.json'
    return dict(schema_version=1, role='final_single_actor', experiment=family,
        actor_count=1, observation_dim=124, action_dim=7, inference_composition='single_actor',
        environment={'module': module, 'class': class_name, 'revision': environment_class.REVISION,
                     'sha256': sha256(root / (module + '.py')),
                     'action_semantics': environment_class.ACTION_SEMANTICS,
                     'kwargs': {'p1_budget_steps': int(budget)}},
        phase_budgets=[1000, int(budget), 1000],
        artifacts={'model': {'path': 'model.zip', 'sha256': sha256(output / 'model.zip')},
                   'vecnormalize': {'path': 'vecnormalize.pkl', 'sha256': sha256(output / 'vecnormalize.pkl')}},
        route_hashes={p.name: sha256(p) for p in sorted((root / 'RUTAS_SAC').glob('*.json'))},
        validation={'status': 'evaluated', 'target_met': bool(validation['target_met']),
                    'report': {'path': report.name, 'sha256': sha256(report)}},
        training=training_counts, teacher_provenance=provenance,
        deployment_note='Only student.actor predicts actions; source/teacher are training-only, '
                        'not serialized as an inference composition.')


def fit_bc(model, norm, data, updates, seed, announce):
    import torch
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=1e-5)
    model.policy.set_training_mode(True)
    losses = []
    for update in range(1, updates + 1):
        indices = balanced_indices(data, 512, rng)
        observations = torch.as_tensor(norm.normalize_obs(data['observations'][indices].copy()),
                                       dtype=torch.float32, device=model.device)
        labels = torch.as_tensor(data['actions'][indices], dtype=torch.float32, device=model.device)
        weights = np.ones((len(indices), 7), np.float32)
        weights[data['origin'][indices] == 1, 5:] = 0.
        weights = torch.as_tensor(weights, device=model.device)
        prediction = model.actor(observations, deterministic=True)
        loss = ((prediction - labels).square() * weights).sum() / weights.sum().clamp_min(1.)
        if not torch.isfinite(loss):
            raise FloatingPointError('BC produjo una pérdida no finita.')
        optimizer.zero_grad()
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.)
        if not torch.isfinite(gradient):
            raise FloatingPointError('BC produjo gradientes no finitos.')
        optimizer.step()
        losses.append(float(loss.detach()))
        if update in {max(1, updates // 2), updates}:
            announce('BC_PROGRESS', completed=update, total=updates,
                     mean_loss=float(np.mean(losses[-100:])))
    model.policy.set_training_mode(False)


def fresh_student(source, norm, family, device, seed):
    """Copy the ACTOR only into fresh SAC critics/optimizers/replay."""
    import torch
    from sac_vx import VXSAC
    model = VXSAC('MlpPolicy', norm, device=device, seed=seed, learning_rate=3e-6,
                  buffer_size=60000, learning_starts=0, batch_size=256, gamma=.9995,
                  tau=.005, train_freq=1, gradient_steps=1, ent_coef='auto_0.001',
                  policy_kwargs=copy.deepcopy(source.policy_kwargs),
                  replay_buffer_kwargs={'handle_timeout_termination': False}, verbose=0)
    model.actor.load_state_dict(source.actor.state_dict())
    for parameter in model.actor.parameters():
        parameter.requires_grad_(True)
    if family == 'r2':
        with torch.no_grad():
            model.actor.mu.weight[5:].zero_()
            model.actor.mu.bias[5:].zero_()
            model.actor.log_std.weight[5:].zero_()
            model.actor.log_std.bias[5:].fill_(-2.)
    return model


def run(args):
    import gymnasium as gym
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from sac_vx import VXSAC
    from train_zenno_vx_r1_1 import load_source

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requerida; no se inicia silenciosamente en CPU.')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    module, class_name = FAMILIES[args.family]
    environment_class = getattr(importlib.import_module(module), class_name)
    evaluator = importlib.import_module('evaluate_zenno_vx_' + args.family)
    teacher_api = importlib.import_module('vx_' + args.family + '_teacher')
    provenance = teacher_gate(args.teacher)
    source, source_norm = load_source(args.source, args.device)
    teacher = teacher_api.load_teacher(args.teacher, args.device, source_path=args.source,
        norm_path=args.source.parent / 'vecnormalize.pkl', expected_budget=args.p1_budget)
    if int(getattr(teacher, '_n_updates', 0)) <= 0:
        raise ValueError('El archivo real del profesor no contiene actualizaciones SAC.')
    label_fn = teacher_api.composed_action(source, source_norm, teacher)
    factory = lambda: environment_class(render_mode=None, debug_rewards=False,
                                        p1_budget_steps=args.p1_budget)

    class TrainingCasesEnv(gym.Wrapper):
        def __init__(self):
            super().__init__(factory())
            self.case_rng = np.random.default_rng(args.seed)
            self.episode = 0

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                self.case_rng = np.random.default_rng(seed)
            index = self.episode % 20
            self.episode += 1
            self.env.set_eval_mode(True)
            self.env.FIXED_MAP_ID = 1 + index // 5
            self.env.set_eval_mass((.5, .75, 1., 1.25, 1.5)[index % 5])
            self.env.CUBE_XY_NOISE_RANGE = .01
            self.env.CUBE_YAW_NOISE_RANGE_RAD = .1
            return self.env.reset(seed=int(self.case_rng.integers(100_000_000, 200_000_000)), options=options)

        def step(self, action):
            obs, reward, term, trunc, info = self.env.step(action)
            # These are finite TASK deadlines, not artificial rollout cuts.
            info['TimeLimit.truncated'] = False
            return obs, reward, bool(term or trunc), False, info

    norm = VecNormalize.load(args.source.parent / 'vecnormalize.pkl', DummyVecEnv([TrainingCasesEnv]))
    norm.training = False
    norm.norm_reward = False
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output
    counts = dict(bc_updates_attempted=0, sac_updates_attempted=0, sac_steps_attempted=0,
                  critic_warmup_attempted=0, selected_sac_updates=0, selected_sac_steps=0,
                  actor_initialized_from=str(args.source), critics='fresh', replay='new family only')

    def announce(stage, **details):
        record = dict(stage=stage, **details)
        dump(output / 'status.json', record)
        with (output / 'milestones.jsonl').open('a') as file:
            file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        print(stage + ' ' + json.dumps(details, ensure_ascii=False), flush=True)

    config = dict(revision=environment_class.REVISION, artifact_role='single_student_training',
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  source_sha256=sha256(args.source), teacher=provenance,
                  route_hashes={p.name: sha256(p) for p in sorted((ROOT / 'RUTAS_SAC').glob('*.json'))},
                  source_files={p.name: sha256(p) for p in
                     (ROOT / (module + '.py'), Path(__file__).resolve(), ROOT / 'vx_single_student_data.py')},
                  observations='raw replay, frozen VecNormalize exactly once',
                  actor_count_at_deployment=1, warm_transfer='actor only; base mean reset for R2',
                  no_old_physical_replay=True, training_counts=counts)
    dump(output / 'config.json', config)
    model = fresh_student(source, norm, args.family, args.device, args.seed)
    model.set_logger(configure(str(output / 'sac_logs'), ['csv']))

    def action_fn(observation, env):
        return np.asarray(model.predict(norm.normalize_obs(np.asarray(observation).copy()),
                                        deterministic=True)[0], np.float32)

    def save(directory):
        directory.mkdir(parents=True, exist_ok=True)
        model.save(directory / 'model.zip')
        norm.save(directory / 'vecnormalize.pkl')
        dump(directory / 'checkpoint.json', dict(sac_updates=int(model._n_updates),
             sac_steps=int(model.num_timesteps), bc_updates_attempted=counts['bc_updates_attempted']))

    development_cases = (evaluator.canonical_cases(reps=1, seed=61_000_000, profile='nominal') +
                         evaluator.canonical_cases(reps=args.validation_reps, seed=61_010_000, profile='robust'))

    def evaluate(tag):
        announce('EVALUATE', tag=tag, cases=len(development_cases))
        result = evaluator.evaluate_cases(action_fn, development_cases, output / ('eval_' + tag),
                                         budgets=(args.p1_budget,), variants=('on',),
                                         metadata={'role': 'single_actor_student', 'stage': tag})
        return result['rows']

    try:
        announce('START', output=str(output), family=args.family, single_actor=True)
        save(output / 'initial')
        best_rows = evaluate('initial')
        save(output / 'best')
        best_tag = 'initial'

        def select(tag):
            nonlocal best_rows, best_tag
            save(output / tag)
            candidate_rows = evaluate(tag)
            accepted, decision = promotion(candidate_rows, best_rows)
            dump(output / tag / 'selection.json', dict(accepted=accepted, **decision))
            if accepted:
                best_rows, best_tag = candidate_rows, tag
                save(output / 'best')
            announce('SELECTION', candidate=tag, accepted=accepted, best=best_tag, **decision)

        pool = None
        if args.old_data:
            pool = load_old_bc(args.old_data, source_env_hash=sha256(ROOT / 'zenno_vx_r1.py'),
                               source_norm_hash=sha256(args.source.parent / 'vecnormalize.pkl'))
            announce('OLD_BC_ONLY', rows=len(pool['phase_ids']), phase1_excluded=True, replay_rows=0)
        training_cases = evaluator.canonical_cases(reps=1, seed=141_000_000, profile='robust')
        announce('TEACHER_COLLECTION', episodes=len(training_cases), real_handoffs=True)
        pool = collect(factory, label_fn, None, training_cases, output / 'teacher_episodes',
                       pool=pool, seed=args.seed, announce=announce)
        if not all(np.any(pool['phase_ids'] == phase) for phase in (0, 1, 2)):
            raise RuntimeError('El profesor no alcanzó todas las fases; no fabricar ejemplos P2.')
        fit_bc(model, norm, pool, args.bc_updates, args.seed, announce)
        counts['bc_updates_attempted'] += args.bc_updates
        select('bc')
        for round_index in range(1, args.dagger_rounds + 1):
            cases = evaluator.canonical_cases(reps=1, seed=141_000_000 + round_index * 1_000_000,
                                              profile='robust')
            announce('DAGGER_COLLECTION', round=round_index, rescue=False, episodes=len(cases))
            pool = collect(factory, label_fn, action_fn, cases, output / f'dagger_{round_index}_episodes',
                           pool=pool, seed=args.seed + round_index, announce=announce)
            fit_bc(model, norm, pool, args.dagger_updates, args.seed + round_index, announce)
            counts['bc_updates_attempted'] += args.dagger_updates
            select(f'dagger_{round_index}')
        save_npz(output / 'bounded_dataset.npz', pool)

        # Online RL starts from the validated incumbent, not the last rejected BC block.
        model = VXSAC.load(output / 'best/model.zip', env=norm, device=args.device,
                           custom_objects={'buffer_size': 60000, 'learning_starts': 0,
                                           'learning_rate': 3e-6})
        model.set_logger(configure(str(output / 'sac_logs'), ['csv']))
        indices = replay_rows(pool)
        if not len(indices):
            raise ValueError('No existen transiciones físicas nuevas para entrenar SAC.')
        if not all(np.any(pool['phase_ids'][indices] == phase) for phase in (0, 1, 2)):
            raise ValueError('El replay físico nuevo debe contener P0, P1 y P2; BC antiguo no las sustituye.')
        model.replay_buffer.reset()
        for index in indices:
            model.replay_buffer.add(pool['observations'][index:index + 1],
                pool['next_observations'][index:index + 1], pool['executed_actions'][index:index + 1],
                pool['rewards'][index:index + 1], pool['dones'][index:index + 1],
                [{'TimeLimit.truncated': False}])
        action_weights = np.ones((3, 7), np.float32)
        action_weights[:, 5:] = 0.  # Never BC-anchor base channels during SAC exploration.
        demo = dict(pool, teacher_stages=np.where(pool['window01'] | pool['window12'],
                                                  'real_handoff', 'phase_interior'))
        model.set_demo_regularization(demo, loss_weight=5., batch_size=512, seed=args.seed,
            normalize_q_actor_loss=True, q_objective_scale=.15,
            actor_warmup_updates=int(model._n_updates) + args.critic_warmup,
        action_weights_by_phase=action_weights)

        if args.sac_steps:
            announce('CRITIC_WARMUP', updates=args.critic_warmup, replay_rows=len(indices), actor_frozen=True)
            model._current_progress_remaining = 1.
            # train(0) computes means of empty loss lists in DemoRegularizedSAC
            # and records NaNs. A requested zero warmup means no train call.
            if args.critic_warmup:
                model.train(args.critic_warmup, batch_size=256)
            counts['critic_warmup_attempted'] += args.critic_warmup
            counts['sac_updates_attempted'] += args.critic_warmup
            announce('CRITIC_WARMUP_COMPLETE', updates=int(model._n_updates))

            class Milestone(BaseCallback):
                def __init__(self):
                    super().__init__()
                    self.next = model.num_timesteps + max(1, args.sac_steps // 2)

                def _on_step(self):
                    if self.num_timesteps >= self.next:
                        save(output / 'latest')
                        announce('SAC_PROGRESS', steps=self.num_timesteps, updates=int(model._n_updates))
                        self.next += max(1, args.sac_steps // 2)
                    return True

            before = int(model._n_updates)
            model.learn(args.sac_steps, reset_num_timesteps=False, callback=Milestone(), log_interval=1)
            counts['sac_updates_attempted'] += int(model._n_updates) - before
            counts['sac_steps_attempted'] += args.sac_steps
            select('sac')
        model = VXSAC.load(output / 'best/model.zip', env=norm, device=args.device,
                           custom_objects={'buffer_size': 1, 'learning_starts': 0})
        counts['selected_sac_updates'] = int(model._n_updates)
        counts['selected_sac_steps'] = int(model.num_timesteps)
        counts['selected_stage'] = best_tag
        counts['selected_has_online_sac'] = bool(model.num_timesteps > 0)
        final = output / 'final'
        save(final)
        holdout_cases = (evaluator.canonical_cases(reps=1, seed=71_000_000, profile='nominal') +
                         evaluator.canonical_cases(reps=args.holdout_reps, seed=71_010_000, profile='robust'))
        announce('HOLDOUT', cases=len(holdout_cases), selection_frozen=True,
                 model_sha256=sha256(final / 'model.zip'))
        evaluated = evaluator.evaluate_cases(action_fn, holdout_cases, output / 'holdout',
            budgets=(args.p1_budget,), variants=('on',),
            metadata={'role': 'single_actor_student', 'scope': 'independent_holdout',
                      'model_sha256': sha256(final / 'model.zip'), 'selection_frozen': True})
        holdout_rates = rates(evaluated['rows'])
        target_met = bool(holdout_rates['nominal']['success'] >= 1. and
                          holdout_rates['robust']['success'] >= args.target_success)
        report = dict(complete=True, scope='full_task', deterministic=True, holdout_independent=True,
            model_sha256=sha256(final / 'model.zip'), cases=len(evaluated['rows']),
            successes=sum(bool(row['success']) for row in evaluated['rows']), target_met=target_met,
            target_robust=args.target_success, rates=holdout_rates, rows=evaluated['rows'],
            selected_using='development only; no holdout model reselection', training_counts=counts)
        dump(final / 'holdout_report.json', report)
        dump(final / 'final_student.json', build_manifest(args.family, environment_class, final,
             args.p1_budget, report, provenance, counts))
        from vx_final_student_contract import validate_final_student
        validate_final_student(final / 'final_student.json', expected_experiment=args.family)
        config['training_counts'] = counts
        dump(output / 'config.json', config)
        dump(output / 'result.json', dict(status='COMPLETE', final_manifest=str(final / 'final_student.json'),
                                         target_met=target_met, holdout_rates=holdout_rates,
                                         training_counts=counts))
        announce('COMPLETE', final_manifest=str(final / 'final_student.json'), target_met=target_met,
                 rates=holdout_rates, selected_stage=best_tag, selected_sac_updates=int(model._n_updates))
    except BaseException as error:
        try:
            save(output / 'interrupted')
        except BaseException as save_error:
            dump(output / 'save_error.json', dict(error=repr(save_error)))
        announce('INTERRUPTED', error=repr(error), training_counts=counts)
        raise
    finally:
        norm.close()
        source_norm.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=tuple(FAMILIES), required=True)
    parser.add_argument('--teacher', type=Path, required=True, help='blockN/model.zip with completed ON/OFF evaluation')
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--p1-budget', type=int, choices=(3000, 4500, 6000), default=4500)
    parser.add_argument('--old-data', type=Path, help='Optional P0/P2 labels; requires matching .contract.json')
    parser.add_argument('--bc-updates', type=int, default=3000)
    parser.add_argument('--dagger-rounds', type=int, default=2)
    parser.add_argument('--dagger-updates', type=int, default=1000)
    parser.add_argument('--sac-steps', type=int, default=20000)
    parser.add_argument('--critic-warmup', type=int, default=2000)
    parser.add_argument('--validation-reps', type=int, default=1)
    parser.add_argument('--holdout-reps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=15110000)
    parser.add_argument('--target-success', type=float, default=.98)
    args = parser.parse_args()
    if min(args.bc_updates, args.dagger_updates, args.validation_reps, args.holdout_reps) < 1:
        parser.error('BC, DAgger y repeticiones deben ser positivos.')
    if min(args.dagger_rounds, args.sac_steps, args.critic_warmup) < 0 or not 0 < args.target_success <= 1:
        parser.error('Presupuestos no negativos y target-success en (0,1].')
    for key in ('output', 'source', 'teacher', 'old_data'):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    if args.source.is_dir():
        args.source = args.source / 'model.zip'
    if args.teacher.is_dir():
        args.teacher = args.teacher / 'model.zip'
    if args.output.exists():
        parser.error('output debe ser nuevo; no se sobrescriben corridas existentes.')

    def interrupt(signum, frame):
        raise InterruptedError(f'Señal {signum}')

    signal.signal(signal.SIGTERM, interrupt)
    lock_path = ROOT / 'models' / ('.single_student_' + args.family + '.lock')
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Ya hay un estudiante de esta familia entrenando.')
        run(args)


if __name__ == '__main__':
    main()
