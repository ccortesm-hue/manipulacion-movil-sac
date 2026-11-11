#!/usr/bin/env python3
"""VX: un SAC, BC curricular, DAgger físico end-to-end y evaluación separada."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/zenno_vx_mpl')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

from analytic_teacher_vx import VXAnalyticTeacher
from sac_vx import VXSAC as DemoRegularizedSAC
from train_sac_v9 import StageBalancedSampler, initialize_actor_exploration
from zenno_vx import ZennoVX, VX_OBS_DIM, VX_EXTRA_NAMES
from vx_p2_diagnostics import p2_state

ROOT = Path(__file__).resolve().parent
CASES = [(m, w) for m in (1, 2, 3, 4) for w in (.5, .75, 1., 1.25, 1.5)]


def dump(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def make_env():
    env = ZennoVX(render_mode=None, debug_rewards=False)
    env.CUBE_XY_NOISE_RANGE = .01
    env.CUBE_YAW_NOISE_RANGE_RAD = .10
    return env


def configure(env, case, robust):
    env.FIXED_MAP_ID, mass = case
    env.set_eval_mode(True)
    env.set_eval_mass(mass)
    env.CUBE_XY_NOISE_RANGE = .01 if robust else 0.0
    env.CUBE_YAW_NOISE_RANGE_RAD = .10 if robust else 0.0


def predict(model, norm, obs):
    return model.predict(norm.normalize_obs(obs.copy()), deterministic=True)[0]


def collect(output, reps, seed, model=None, norm=None, beta=1.0, round_id=0):
    """Etiquetas BC y acciones físicas separadas; jamás reinicia directamente P2."""
    env = make_env()
    teacher = VXAnalyticTeacher(env)
    chunks, episodes = [], []
    keys = ('observations', 'next_observations', 'actions', 'executed_actions',
            'rewards', 'dones', 'timeouts', 'phase_ids', 'teacher_stages',
            'episode_ids', 'dagger_round', 'replay_valid')
    try:
        for case_index, case in enumerate(CASES):
            for rep in range(reps):
                ep = case_index*reps+rep
                configure(env, case, rep > 0 or model is not None)
                obs, _ = env.reset(seed=seed+ep)
                teacher.reset()
                rows = {k: [] for k in keys}
                for step in range(5000):
                    phase = int(env.phase_state)
                    label = teacher.action()
                    student = label if model is None else predict(model, norm, obs)
                    action = (beta*label+(1-beta)*student).astype(np.float32)
                    # During DAgger let the student's residual act in P1;
                    # its target is ignored in BC to avoid undoing RL improvements.
                    if phase == 1 and model is not None:
                        action[5:] = student[5:]
                    next_obs, reward, term, trunc, info = env.step(action)
                    values = (obs, next_obs, label, action, reward, term or trunc,
                              trunc and not term, phase, teacher.stage, ep,
                              round_id, model is None)
                    for key, value in zip(keys, values):
                        rows[key].append(value)
                    obs = next_obs
                    if term or trunc:
                        break
                success = bool(info.get('strict_success', False))
                episodes.append(dict(map=case[0], mass=case[1], seed=seed+ep,
                                     success=success, steps=step+1,
                                     reason=info.get('truncation_reason',''),
                                     teacher_stage=teacher.stage))
                print(f'[VX-COLLECT r{round_id}] {ep+1}/{len(CASES)*reps} map={case[0]} mass={case[1]} success={int(success)} steps={step+1}', flush=True)
                # Initial BC contains complete successful demonstrations only.
                # DAgger includes failures: recovery labels are the point.
                if success or model is not None:
                    chunks.append({k: np.asarray(v) for k,v in rows.items()})
    finally:
        env.close()
    if not chunks:
        raise RuntimeError('No hay demostraciones aceptadas. Revisar profesor/logs.')
    data = {k: np.concatenate([c[k] for c in chunks]) for k in keys}
    np.savez_compressed(output, **data)
    dump(Path(output).with_suffix('.json'), dict(episodes=episodes,
        success_rate=sum(e['success'] for e in episodes)/len(episodes),
        revision=ZennoVX.REVISION, observation_dim=VX_OBS_DIM,
        beta=beta, assisted=model is not None))
    return data


def clone(model, norm, data, steps, seed, initial=False):
    sampler = StageBalancedSampler(data['phase_ids'], data['teacher_stages'], seed,
                                   dagger_round=data['dagger_round'])
    model.actor.set_training_mode(True)
    for group in model.actor.optimizer.param_groups:
        group['lr'] = 1e-4 if initial else 3e-5
    for update in range(steps):
        frac = update/max(steps,1)
        phases = ((0,) if frac < .2 else (0,1) if frac < .5 else (0,1,2)) if initial else (0,1,2)
        # First balance phases, then sub-stages; P2 has more stages than P1.
        phase_choices = sampler.rng.choice(phases, size=512)
        idx = np.concatenate([sampler.sample(int(np.sum(phase_choices==ph)), (ph,),
                              dagger_fraction=0 if initial else .5)
                              for ph in phases if np.any(phase_choices==ph)])
        obs = torch.as_tensor(norm.normalize_obs(data['observations'][idx].copy()), device=model.device)
        target = torch.as_tensor(data['actions'][idx], dtype=torch.float32, device=model.device)
        weights = torch.ones_like(target)
        if not initial:
            p1 = torch.as_tensor(data['phase_ids'][idx] == 1, device=model.device)
            weights[p1,5:] = 0.0
        pred = model.actor(obs.float(), deterministic=True)
        loss = (weights*(pred-target).square()).sum()/weights.sum()
        model.actor.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 5.0)
        model.actor.optimizer.step()
        if (update+1) % max(steps//10,1) == 0:
            print(f'[VX-BC] {update+1}/{steps} loss={loss.item():.6f}', flush=True)


def evaluate(model, norm, output, reps=1, seed=8_100_000, residual=True):
    env = ZennoVX(render_mode=None, debug_rewards=False, residual_enabled=residual)
    rows = []
    try:
        for profile in ('nominal','robust'):
            for case_idx, case in enumerate(CASES):
                for rep in range(1 if profile == 'nominal' else reps):
                    episode_seed = seed + (10000 if profile == 'robust' else 0) + case_idx*100 + rep
                    configure(env, case, profile == 'robust')
                    obs, _ = env.reset(seed=episode_seed)
                    cte, yaw, carry, relative, residual_sq = [], [], [], [], []
                    energy = np.zeros(3); seconds = np.zeros(3)
                    p2_entry = None
                    p2_trace = []
                    simultaneous_steps = 0
                    last_p2 = None
                    for step in range(5000):
                        phase = int(env.phase_state)
                        obs, reward, term, trunc, info = env.step(predict(model, norm, obs))
                        seconds[phase] += env.dt
                        energy[phase] += float(info.get('energy_step_total_j', 0))
                        if phase == 2:
                            last_p2 = p2_state(env)
                            simultaneous_steps += int(last_p2['simultaneous'])
                            if len(p2_trace) == 0 or env._phase_steps % 25 == 0 or term or trunc:
                                p2_trace.append(dict(step=int(env._phase_steps), **last_p2))
                        if phase == 1:
                            carry.append(info.get('vx_carry_error_m',0)**2)
                            relative.append(info.get('vx_relative_speed_m_s',0)**2)
                            residual_sq.append(np.mean(np.asarray(info['vx_delta_torque_fraction'])**2))
                            if info.get('vx_tracking_active'):
                                cte.append(info['vx_cte_m']**2); yaw.append(info['vx_yaw_rad']**2)
                            if int(env.phase_state) == 2:
                                p2_entry = dict(carry_m=info.get('vx_carry_error_m'),
                                                relative_speed_m_s=info.get('vx_relative_speed_m_s'))
                        if term or trunc:
                            break
                    rms = lambda x: float(np.sqrt(np.mean(x))) if x else None
                    row = dict(profile=profile, map=case[0], mass=case[1], seed=episode_seed,
                        success=bool(info.get('strict_success',False)),
                        phases=[bool(info.get(f'phase{i}_success',False)) for i in range(3)],
                        energy_j=energy.tolist(), time_s=seconds.tolist(),
                        cte_rms_m=rms(cte), yaw_rms_rad=rms(yaw), carry_rms_m=rms(carry),
                        relative_speed_rms_m_s=rms(relative), residual_rms_fraction=rms(residual_sq),
                        p2_entry=p2_entry, p2_stage=last_p2['stage'] if last_p2 else None,
                        p2_trace=p2_trace, p2_simultaneous_steps=simultaneous_steps,
                        reason=info.get('truncation_reason') or info.get('termination_reason'))
                    rows.append(row)
                    print(f"[VX-EVAL] {profile} map={case[0]} mass={case[1]} success={int(row['success'])} phases={row['phases']}", flush=True)
    finally:
        env.close()
    rates = {p: float(np.mean([r['success'] for r in rows if r['profile']==p])) for p in ('nominal','robust')}
    phases = np.mean([r['phases'] for r in rows], axis=0).tolist()
    reached = sum(r['phases'][1] for r in rows)
    conditional = sum(r['phases'][2] for r in rows)/reached if reached else 0.0
    metrics = dict(rates=rates, phases=phases, conditional_p2=conditional,
                   strict=float(np.mean([r['success'] for r in rows])), rows=rows,
                   residual_enabled=residual, revision=ZennoVX.REVISION)
    dump(output, metrics)
    return metrics


def rank(metrics):
    return (min(metrics['rates'].values()), metrics['strict'], *metrics['phases'])


def gate(metrics, threshold):
    return (min(metrics['rates'].values()) >= threshold and
            metrics['phases'][0] >= .95 and metrics['phases'][1] >= .95 and
            metrics['conditional_p2'] >= threshold)


def save(model, norm, directory):
    directory.mkdir(parents=True, exist_ok=True)
    model.save(directory/'model.zip')
    norm.save(directory/'vecnormalize.pkl')


class ValidationCallback(BaseCallback):
    def __init__(self, norm, out, initial, freq, seed, timesteps):
        super().__init__()
        self.norm, self.out, self.best = norm, out, initial
        self.freq, self.seed, self.total = freq, seed, timesteps
        self.last = 0

    def _on_step(self):
        self.model._demo_loss_weight = 5.0-3.0*min(1.,self.num_timesteps/self.total)
        if self.num_timesteps-self.last < self.freq:
            return True
        self.last = self.num_timesteps
        metrics = evaluate(self.model, self.norm, self.out/f'eval_sac_{self.last}.json', seed=self.seed)
        save(self.model, self.norm, self.out/f'checkpoint_{self.last}')
        if rank(metrics) > rank(self.best):
            self.best = metrics
            save(self.model, self.norm, self.out/'best')
        # Evaluation is deterministic; a substantial competence drop ends this run.
        if metrics['strict'] < self.best['strict']-.15:
            dump(self.out/'early_stop.json', dict(reason='validation_regression', metrics=metrics))
            return False
        return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=910000)
    p.add_argument('--teacher-reps', type=int, default=3)
    p.add_argument('--bc-steps', type=int, default=20000)
    p.add_argument('--dagger-rounds', type=int, default=5)
    p.add_argument('--dagger-steps', type=int, default=8000)
    p.add_argument('--timesteps', type=int, default=300000)
    p.add_argument('--n-envs', type=int, default=4)
    p.add_argument('--eval-freq', type=int, default=20000)
    p.add_argument('--gate', type=float, default=.90)
    p.add_argument('--collect-only', action='store_true')
    args = p.parse_args()
    if any(getattr(args,k) <= 0 for k in ('teacher_reps','bc_steps','dagger_steps','timesteps','n_envs','eval_freq')) or args.dagger_rounds < 0 or not .9 <= args.gate <= 1:
        p.error('Presupuestos positivos, rondas >=0 y gate entre .90 y 1.')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA no disponible; activar el entorno Python con GPU.')
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f'Utilice un directorio nuevo: {out}')
    out.mkdir(parents=True)
    set_random_seed(args.seed, using_cuda=args.device.startswith('cuda'))
    dump(out/'config.json', dict(**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        revision=ZennoVX.REVISION, observation_dim=VX_OBS_DIM,
        extra_observations=list(VX_EXTRA_NAMES), budgets=[1000,3000,1000],
        resource_costs=dict(time_per_second=ZennoVX.TIME_COST_PER_SECOND,
                            energy_per_joule=ZennoVX.ENERGY_COST_PER_JOULE,
                            uniform_across_phases=True),
        hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                ('zenno_vx.py','analytic_teacher_vx.py','train_vx.py','sac_vx.py','demo_regularized_sac_v9.py')},
        route_hashes={path.name:hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in sorted((ROOT/'RUTAS_SAC').glob('*.json'))},
        single_sac_agent=True, single_actor=True, ent_coef=.001, initial_log_std=-4.5))
    data = collect(out/'teacher.npz', args.teacher_reps, args.seed)
    if args.collect_only:
        return
    teacher_metrics = json.loads((out/'teacher.json').read_text())
    covered = {(e['map'],e['mass']) for e in teacher_metrics['episodes'] if e['success']}
    if covered != set(CASES) or teacher_metrics['success_rate'] < .95:
        dump(out/'teacher_gate.json', dict(passed=False, reason='teacher_coverage_or_success'))
        print('[VX] Profesor insuficiente: revisar teacher.json antes de BC.',flush=True)
        return
    norm = VecNormalize(DummyVecEnv([make_env for _ in range(args.n_envs)]), norm_reward=False)
    norm.obs_rms.update(data['observations'].astype(np.float64))
    # Constant teacher residuals must not turn new nonzero residual states into
    # clipped 10-sigma observations during SAC. Freeze physically sensible floors.
    floor = np.full(VX_OBS_DIM, .01)
    floor[78:] = 1.0
    for name in ('cte_m','carry_x','carry_y','carry_z'):
        floor[78+VX_EXTRA_NAMES.index(name)] = .05
    for name in ('carry_vx','carry_vy','carry_vz'):
        floor[78+VX_EXTRA_NAMES.index(name)] = .10
    for name in ('delta_left','delta_right'):
        floor[78+VX_EXTRA_NAMES.index(name)] = .20
    norm.obs_rms.var = np.maximum(norm.obs_rms.var, floor**2)
    norm.training = False
    norm.seed(args.seed)
    try:
        model = DemoRegularizedSAC('MlpPolicy', norm, device=args.device,
            policy_kwargs=dict(net_arch=[256,256]), learning_rate=3e-6, gamma=.999,
            ent_coef=.001, batch_size=512, buffer_size=300000, learning_starts=0,
            train_freq=1, gradient_steps=1, seed=args.seed, verbose=1,
            tensorboard_log=str(out/'tensorboard'))
        initialize_actor_exploration(model,-4.5)
        clone(model,norm,data,args.bc_steps,args.seed,initial=True)
        validation_seed = args.seed+1000000
        best = evaluate(model,norm,out/'eval_bc.json',seed=validation_seed)
        save(model,norm,out/'best')
        for round_id in range(1,args.dagger_rounds+1):
            if gate(best,args.gate):
                break
            beta = max(.1,.9-.2*(round_id-1))
            extra = collect(out/f'dagger_{round_id}.npz',1,args.seed+2000000+1000*round_id,
                            model,norm,beta,round_id)
            data = {k:np.concatenate([data[k],extra[k]]) for k in data}
            clone(model,norm,data,args.dagger_steps,args.seed+round_id)
            metrics = evaluate(model,norm,out/f'eval_dagger_{round_id}.json',seed=validation_seed)
            save(model,norm,out/f'dagger_model_{round_id}')
            if rank(metrics)>rank(best):
                best=metrics
                save(model,norm,out/'best')
            else:
                model=DemoRegularizedSAC.load(out/'best/model.zip',env=norm,device=args.device)
        dump(out/'pre_sac_gate.json',dict(passed=gate(best,args.gate),metrics=best))
        if not gate(best,args.gate):
            print('[VX] Gate no superado. BC/DAgger guardados; SAC no iniciado.',flush=True)
            return
        model=DemoRegularizedSAC.load(out/'best/model.zip',env=norm,device=args.device)
        # Seed Q only with actions actually executed in successful teacher runs.
        valid=np.flatnonzero(data['replay_valid'])
        for start in range(0,len(valid)-args.n_envs+1,args.n_envs):
            idx=valid[start:start+args.n_envs]
            model.replay_buffer.add(data['observations'][idx],data['next_observations'][idx],
                data['executed_actions'][idx],data['rewards'][idx],data['dones'][idx],
                [{'TimeLimit.truncated':bool(x)} for x in data['timeouts'][idx]])
        weights=np.ones((3,7),dtype=np.float32); weights[1,5:]=0
        model.set_demo_regularization(data,5.0,512,args.seed,
            normalize_q_actor_loss=True,q_objective_scale=.15,actor_warmup_updates=1000,
            action_weights_by_phase=weights)
        callback=ValidationCallback(norm,out,best,args.eval_freq,validation_seed,args.timesteps)
        model.learn(args.timesteps,callback=callback,reset_num_timesteps=True,tb_log_name='vx')
        save(model,norm,out/'last')
        last=evaluate(model,norm,out/'eval_sac_last.json',seed=validation_seed)
        if rank(last)>rank(callback.best):
            save(model,norm,out/'best')
        model=DemoRegularizedSAC.load(out/'best/model.zip',env=norm,device=args.device)
        # Holdout seeds never participate in checkpoint selection.
        final=evaluate(model,norm,out/'holdout_220.json',reps=10,seed=args.seed+5000000)
        evaluate(model,norm,out/'holdout_220_residual_off.json',reps=10,seed=args.seed+5000000,residual=False)
        dump(out/'result.json',dict(target_met=min(final['rates'].values())>=.95,
                                   nominal=final['rates']['nominal'],robust=final['rates']['robust']))
    finally:
        norm.close()


if __name__=='__main__':
    main()
