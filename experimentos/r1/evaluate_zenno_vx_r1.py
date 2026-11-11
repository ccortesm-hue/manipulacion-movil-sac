#!/usr/bin/env python3
"""Evaluación autónoma R1: física completa, éxito estricto y transiciones."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/vx_r1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np
from zenno_vx_r1 import ZennoVXR1
from vx_p2_diagnostics import p2_state

ROOT = Path(__file__).resolve().parent
CASES = [(m, w) for m in (1,2,3,4) for w in (.5,.75,1.,1.25,1.5)]

def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)

def make_env():
    env = ZennoVXR1(render_mode=None, debug_rewards=False)
    env.CUBE_XY_NOISE_RANGE = .01
    env.CUBE_YAW_NOISE_RANGE_RAD = .10
    return env

def configure(env, case, robust):
    env.FIXED_MAP_ID, mass = case
    env.set_eval_mode(True)
    env.set_eval_mass(mass)
    env.CUBE_XY_NOISE_RANGE = .01 if robust else 0.
    env.CUBE_YAW_NOISE_RANGE_RAD = .10 if robust else 0.

def predict(model, norm, obs):
    return model.predict(norm.normalize_obs(obs.copy()), deterministic=True)[0]

def save(model, norm, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save(directory/'model.zip')
    norm.save(directory/'vecnormalize.pkl')

def handoff_state(env, source, target):
    return dict(source=source, target=target,
        arm_qpos=np.asarray(env._get_arm_qpos()).tolist(),
        arm_target=np.asarray(env.arm_target).tolist(),
        cube_xyz=np.asarray(env._cube_pos()).tolist(),
        carry_error_m=float(np.linalg.norm(env._site_xpos('setpoint')-env._cube_pos())),
        relative_speed_m_s=float(np.linalg.norm(env._vx_relative_velocity)),
        cube_speed_m_s=float(np.linalg.norm(env._cube_vel_vec())),
        gripping=bool(env.gripping))

def evaluate(model, norm, output, reps=1, seed=8_100_000, residual=True, cases=None, progress=None):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    episode_log = output.with_suffix('.episodes.jsonl')
    if output.exists() or episode_log.exists():
        raise FileExistsError(f'La evaluación ya existe o quedó parcial: {output}')
    env = ZennoVXR1(render_mode=None, debug_rewards=False, residual_enabled=residual)
    rows = []
    try:
        for profile in ('nominal','robust'):
            for case_idx, case in enumerate(CASES if cases is None else cases):
                for rep in range(1 if profile == 'nominal' else reps):
                    episode_seed = seed + (10000 if profile == 'robust' else 0) + case_idx*100 + rep
                    configure(env, case, profile == 'robust')
                    obs, _ = env.reset(seed=episode_seed)
                    cte, yaw, carry, relative, residual_sq = [], [], [], [], []
                    energy = np.zeros(3); seconds = np.zeros(3)
                    p2_entry = None
                    handoffs = []
                    p2_trace = []
                    simultaneous_steps = 0
                    last_p2 = None
                    for step in range(5000):
                        phase = int(env.phase_state)
                        obs, reward, term, trunc, info = env.step(predict(model, norm, obs))
                        seconds[phase] += env.dt
                        if int(env.phase_state) != phase:
                            handoffs.append(handoff_state(env, phase, int(env.phase_state)))
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
                        handoffs=handoffs, p2_entry=p2_entry, p2_stage=last_p2['stage'] if last_p2 else None,
                        p2_trace=p2_trace, p2_simultaneous_steps=simultaneous_steps,
                        reason=info.get('truncation_reason') or info.get('termination_reason'))
                    rows.append(row)
                    # Every finished episode survives an interrupted evaluation.
                    with episode_log.open('a') as file:
                        file.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                    if progress is not None:
                        progress(len(rows), row)
                    print(f"[VX-EVAL] {profile} map={case[0]} mass={case[1]} success={int(row['success'])} phases={row['phases']}", flush=True)
    finally:
        env.close()
    rates = {p: float(np.mean([r['success'] for r in rows if r['profile']==p])) for p in ('nominal','robust')}
    phases = np.mean([r['phases'] for r in rows], axis=0).tolist()
    reached = sum(r['phases'][1] for r in rows)
    conditional = sum(r['phases'][2] for r in rows)/reached if reached else 0.0
    metrics = dict(rates=rates, phases=phases, conditional_p2=conditional,
                   strict=float(np.mean([r['success'] for r in rows])), rows=rows,
                   residual_enabled=residual, revision=ZennoVXR1.REVISION)
    dump(output, metrics)
    return metrics

def main():
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from sac_vx import VXSAC
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reps', type=int, default=10)
    p.add_argument('--seed', type=int, default=41000000)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if not 1 <= args.reps <= 100: p.error('reps entre 1 y 100')
    if args.output.exists(): p.error('El reporte ya existe; usa otra ruta.')
    norm = VecNormalize.load(args.bundle/'vecnormalize.pkl', DummyVecEnv([make_env]))
    norm.training = False
    try:
        model = VXSAC.load(args.bundle/'model.zip', env=norm, device=args.device)
        result = evaluate(model, norm, args.output, reps=args.reps, seed=args.seed)
        print(json.dumps({k:result[k] for k in ('rates','phases','strict')}))
    finally:
        norm.close()

if __name__ == '__main__': main()
