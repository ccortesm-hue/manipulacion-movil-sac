#!/usr/bin/env python3
"""Profesor P1 SAC 2D: exploración residual real, brazo de R1 en lazo cerrado.

Este artefacto es un PROFESOR experimental, no el estudiante final de 7 acciones.
No usa BC, críticos heredados ni replay de acciones etiquetadas por un profesor.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/vx_r11')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from zenno_vx_r1_1 import ZennoVXR11
from vx_r1_1_teacher import ARM_MASK_REVISION, arm_observation, composed_action

ROOT = Path(__file__).resolve().parent
MASSES = (.5, .75, 1., 1.25, 1.5)
SOURCE = ROOT/'models/zenno_vx_r1_run2/best/model.zip'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def load_source(path, device):
    """Trusted local SB3 bundle; never allocate its historical replay buffer."""
    path = Path(path).resolve()
    config = json.loads((path.parent.parent/'config.json').read_text())
    if config.get('revision') != 'VX_R1_OVERLAP_CURRICULUM_124D_7D_DELTA008_UNIFORM_COSTS':
        raise ValueError('Se requiere el checkpoint R1 documentado, no R2 ni otra semántica.')
    if config['hashes']['zenno_vx_r1.py'] != sha(ROOT/'zenno_vx_r1.py'):
        raise ValueError('La fuente R1 fue modificada respecto de su checkpoint.')
    for name, digest in config['route_hashes'].items():
        if sha(ROOT/'RUTAS_SAC'/name) != digest:
            raise ValueError(f'Ruta modificada: {name}')
    with (path.parent/'vecnormalize.pkl').open('rb') as file:
        norm = pickle.load(file)
    # SB3 requires set_venv after unpickling, including for saving again.
    # This light environment never advances physics or updates statistics.
    from stable_baselines3.common.vec_env import DummyVecEnv
    from evaluate_zenno_vx_r1_1 import _SpaceOnlyEnv
    norm.set_venv(DummyVecEnv([_SpaceOnlyEnv]))
    norm.training = False
    norm.norm_reward = False
    model = SAC.load(path, device=device, custom_objects={'buffer_size': 1})
    if model.observation_space.shape != (124,) or model.action_space.shape != (7,):
        raise ValueError('El actor fuente debe tener contrato 124D/7D.')
    model.policy.set_training_mode(False)
    for parameter in model.policy.parameters():
        parameter.requires_grad_(False)
    return model, norm


class P1ResidualEnv(gym.Env):
    """Inicios reales de P0; termina al entregar a P2 o agotar la tarea P1.

    Observaciones externas normalizadas UNA vez con las estadísticas congeladas
    de R1. Replay almacena estas observaciones y las acciones residuales ejecutadas
    a través del filtro/limitador del ambiente, nunca etiquetas contrafactuales.
    """
    metadata = {'render_modes': []}

    def __init__(self, source, norm, *, budget=4500, seed=13110000, log=None):
        super().__init__()
        self.source, self.norm = source, norm
        self.env = ZennoVXR11(render_mode=None, debug_rewards=False, p1_budget_steps=budget)
        self.env.set_eval_mode(True)
        self.env.CUBE_XY_NOISE_RANGE = .01
        self.env.CUBE_YAW_NOISE_RANGE_RAD = .1
        self.action_space = gym.spaces.Box(-1., 1., (2,), dtype=np.float32)
        clip = float(norm.clip_obs)
        self.observation_space = gym.spaces.Box(-clip, clip, (124,), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.counter = 0
        self.log = Path(log) if log else None
        self._obs = None

    def normalize(self, obs):
        return self.norm.normalize_obs(np.asarray(obs, dtype=np.float32).copy()).astype(np.float32)

    def arm_action(self, obs):
        protected = arm_observation(obs, int(self.env.phase_state))
        return np.asarray(self.source.predict(self.normalize(protected), deterministic=True)[0], dtype=np.float32).copy()

    def record(self, value):
        if self.log:
            with self.log.open('a') as file:
                file.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        for _ in range(40):
            index = self.counter % 20
            self.counter += 1
            mid, mass = 1+index//5, MASSES[index % 5]
            # Disjoint from diagnostic (~20M), development (61M) and test (71M).
            episode_seed = int(self.rng.integers(100_000_000, 200_000_000))
            self.env.FIXED_MAP_ID = mid
            self.env.set_eval_mass(mass)
            obs, _ = self.env.reset(seed=episode_seed)
            steps = 0
            while int(self.env.phase_state) == 0:
                obs, _, term, trunc, info = self.env.step(self.arm_action(obs))
                steps += 1
                if term or trunc:
                    break
            self.case = dict(map=mid, mass=mass, seed=episode_seed, p0_steps=steps)
            if int(self.env.phase_state) == 1 and not (term or trunc):
                self._obs = obs
                self.reward_sum = 0.
                self.steps = 0
                self.squares = np.zeros(3)
                self.reward_terms = {}
                self.record(dict(event='P0_TO_P1', **self.case))
                return self.normalize(obs), dict(real_p0_handoff=True, **self.case)
            self.record(dict(event='P0_FAILED_NOT_HIDDEN', **self.case,
                             reason=info.get('termination_reason') or info.get('truncation_reason')))
        raise RuntimeError('40 reinicios P0 sin alcanzar P1; no se fabrican estados iniciales.')

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError('El profesor residual requiere dos acciones finitas.')
        action = np.clip(action, -1., 1.)
        full = self.arm_action(self._obs)
        full[5:] = action
        self._obs, reward, term, trunc, info = self.env.step(full)
        self.steps += 1
        self.reward_sum += reward
        self.squares += [float(np.mean(action**2)), float(np.mean(self.env._vx_filter**2)),
                         float(np.mean((25*self.env._vx_delta)**2))]
        for key, value in info.items():
            if key.startswith('vx_') and isinstance(value, (float, np.floating)) and ('cost' in key or key in ('vx_progress', 'vx_terminal_progress', 'vx_drop', 'vx_arrival_bonus')):
                self.reward_terms[key] = self.reward_terms.get(key, 0.)+float(value)
        handoff = bool(info.get('phase1_success')) and int(self.env.phase_state) == 2
        done = bool(term or trunc or handoff)
        if done:
            self.record(dict(event='P1_EPISODE', **self.case, success=handoff,
                steps=self.steps, return_p1=self.reward_sum,
                request_filter_torque_rms=np.sqrt(self.squares/max(1, self.steps)).tolist(),
                reward_components=self.reward_terms, diagnostic=info.get('p1_diagnostics'),
                reason=info.get('termination_reason') or info.get('truncation_reason')))
        # Finite P1 task termination, NOT a rollout cut with bootstrap from P2.
        info['TimeLimit.truncated'] = False
        info['p1_specialist_success'] = handoff
        return self.normalize(self._obs), float(reward), done, False, info

    def close(self):
        self.env.close()


def evaluation_cases(seed=61_000_000, reps=1):
    return [dict(map=mid, mass=mass, seed=seed+mid*100000+i*1000+r, profile='robust')
            for mid in (1, 2, 3, 4) for i, mass in enumerate(MASSES) for r in range(reps)]


def run(args):
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requerida: no fallback CPU silencioso.')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    source, norm = load_source(args.source, args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(revision=ZennoVXR11.REVISION, artifact_role='P1_2D_teacher_NOT_final_student',
        source_sha256=sha(args.source), normalizer_sha256=sha(args.source.parent/'vecnormalize.pkl'),
        source_arm='frozen closed-loop actor; no open-loop trajectory',
        arm_mask_revision=ARM_MASK_REVISION,
        arm_request_channel_mask='P1 only: raw[30,31,97,98,99,100,101]=0, public observation unchanged',
        residual_bc_weight=0., normalized_observations='external_frozen_once; replay normalized',
        budgets=[1000, args.p1_budget, 1000], training_starts='real source P0 rollouts',
        source_files={p.name: sha(p) for p in (ROOT/'zenno_vx_r1_1.py', Path(__file__).resolve(),
                      ROOT/'vx_r1_1_teacher.py', ROOT/'evaluate_zenno_vx_r1_1.py')},
        route_hashes={p.name: sha(p) for p in (ROOT/'RUTAS_SAC').glob('*.json')},
        gpu=torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
        initial_log_std=-2., entropy_coef=args.ent_coef, critic_initialization='fresh',
        bootstrap_at_p1_deadline=False, final_student_created=False)
    dump(out/'config.json', config)

    def announce(stage, **details):
        record = dict(stage=stage, pid=os.getpid(), **details)
        dump(out/'status.json', record)
        with (out/'milestones.jsonl').open('a') as file:
            file.write(json.dumps(record, ensure_ascii=False)+'\n')
        print(stage+' '+json.dumps(details, ensure_ascii=False), flush=True)

    env = P1ResidualEnv(source, norm, budget=args.p1_budget, seed=args.seed,
                        log=out/'training_episodes.jsonl')
    model = SAC('MlpPolicy', env, device=args.device, seed=args.seed,
        learning_rate=args.lr, buffer_size=args.buffer_size, learning_starts=args.warmup,
        batch_size=256, tau=.005, gamma=.9995, train_freq=1, gradient_steps=1,
        ent_coef=args.ent_coef, target_entropy=-2.,
        policy_kwargs=dict(net_arch=[128,128]),
        replay_buffer_kwargs=dict(handle_timeout_termination=False), verbose=0)
    # SB3's non-gSDE actor does not initialize this Linear bias from log_std_init.
    with torch.no_grad():
        model.actor.mu.weight.zero_(); model.actor.mu.bias.zero_()
        model.actor.log_std.weight.zero_(); model.actor.log_std.bias.fill_(-2.)
    model.set_logger(configure(str(out/'sac_logs'), ['csv']))

    def save(directory):
        directory.mkdir(parents=True, exist_ok=True)
        model.save(directory/'model.zip')
        norm.save(directory/'vecnormalize.pkl')
        dump(directory/'contract.json', dict(revision=ZennoVXR11.REVISION,
             role='P1_2D_teacher_NOT_final_student', observation_dim=124, action_dim=2,
             p1_budget=args.p1_budget, source_actor_sha256=sha(args.source),
             normalizer_sha256=sha(directory/'vecnormalize.pkl'),
             updates=int(model._n_updates), steps=int(model.num_timesteps)))

    class Milestone(BaseCallback):
        def __init__(self, block):
            super().__init__(); self.block = block; self.next = model.num_timesteps+args.steps//2
        def _on_step(self):
            if self.num_timesteps >= self.next:
                save(out/'latest')
                with torch.no_grad():
                    observation = torch.as_tensor(env.normalize(env._obs)[None,:], device=self.model.device)
                    mean, log_std, _ = self.model.actor.get_action_dist_params(observation)
                    action_std = float(log_std.exp().mean())
                    action_mean = float(torch.tanh(mean).abs().mean())
                    alpha = float(self.model.log_ent_coef.exp()) if self.model.log_ent_coef is not None else float(self.model.ent_coef_tensor)
                announce('TRAIN_PROGRESS', block=self.block, steps=self.num_timesteps,
                         sac_updates=int(self.model._n_updates), gaussian_std=action_std,
                         deterministic_action_abs=action_mean, entropy_coefficient=alpha)
                self.next += max(1, args.steps//2)
            return True

    try:
        save(out/'initial')
        from evaluate_zenno_vx_r1_1 import evaluate_cases
        cases = evaluation_cases(reps=args.eval_reps)
        source_action = composed_action(source, norm)
        evaluation_metadata = dict(source_sha256=sha(args.source),
            normalizer_sha256=sha(args.source.parent/'vecnormalize.pkl'),
            auxiliary_arm_mask=config['arm_request_channel_mask'], role='P1_teacher_not_final_student')
        announce('BASELINE_EVAL', cases=len(cases), variants=['on', 'off'])
        evaluate_cases(source_action, cases, out/'source_development', budgets=(args.p1_budget,),
                       metadata=evaluation_metadata)
        for block in range(1, args.blocks+1):
            announce('TRAIN_START', block=block, additional_p1_steps=args.steps,
                     pure_sac=True, residual_bc=False)
            model.learn(args.steps, reset_num_timesteps=False, callback=Milestone(block), log_interval=1)
            save(out/f'block{block}')
            if not model._n_updates:
                raise RuntimeError('No hubo actualizaciones SAC; no se considera entrenamiento terminado.')
            announce('BLOCK_EVAL', block=block, sac_updates=int(model._n_updates))
            specialist_action = composed_action(source, norm, model)
            evaluate_cases(specialist_action, cases, out/f'block{block}_development',
                           budgets=(args.p1_budget,), metadata={**evaluation_metadata,
                            'teacher_sha256':sha(out/f'block{block}/model.zip'),
                            'updates':int(model._n_updates)})
        save(out/'last')
        # Persist replay separately, never embedded in inference checkpoints.
        model.save_replay_buffer(out/'replay_buffer.pkl')
        dump(out/'result.json', dict(status='pilot_complete', sac_updates=int(model._n_updates),
            p1_training_steps=int(model.num_timesteps), final_student_created=False,
            note='Comparar development ON/OFF y source antes de promover/distilar. No es holdout.'))
        announce('COMPLETE', sac_updates=int(model._n_updates), p1_training_steps=int(model.num_timesteps),
                 final_student_created=False)
    except BaseException as error:
        try:
            save(out/'interrupted')
        except BaseException as save_error:
            dump(out/'save_error.json', dict(error=repr(save_error)))
        announce('INTERRUPTED', error=repr(error), sac_updates=int(model._n_updates))
        raise
    finally:
        env.close()
        norm.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--device', choices=('cuda','cpu'), default='cuda')
    parser.add_argument('--p1-budget', type=int, choices=(3000,4500,6000), default=4500)
    parser.add_argument('--steps', type=int, default=30000, help='Decisiones P1 por bloque; P0 no cuenta.')
    parser.add_argument('--blocks', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=2048)
    parser.add_argument('--buffer-size', type=int, default=60000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ent-coef', default='auto_0.01')
    parser.add_argument('--seed', type=int, default=13110000)
    parser.add_argument('--eval-reps', type=int, default=1)
    args = parser.parse_args()
    args.output = args.output.resolve(); args.source = args.source.resolve()
    if args.steps <= args.warmup or args.blocks < 1 or args.buffer_size < 2048 or args.eval_reps < 1:
        parser.error('steps debe superar warmup y presupuestos deben ser positivos.')
    def interrupt(signum, frame):
        raise InterruptedError(f'Señal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    lock_path = ROOT/'models/.zenno_vx_r1_1.lock'
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Ya hay una corrida R1.1 activa.')
        run(args)


if __name__ == '__main__':
    main()
