"""Evaluación CPU pareada: baseline, inicial R3, residual OFF y ON."""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from VX_R3 import ROOT, ZennoVXR3, CurriculumSAC, np, torch
from extract_vx_place_robust import BaselineDriver, dump, make_cases
from vx_place_initials import canonicalize_and_audit
from vx_place_protocol import Recorder
import mujoco

METRICS = ('total_time_s','total_energy_J','cte_rms_m','yaw_rms_rad',
           'cube_goal_dist_xy_m','carry_rms_m','base_terminal_error_m')
MASSES = (.5,.75,1.,1.25,1.5)

def cases(profile='robust', reps=1, seed=310_000_000):
    masses = (1.,) if profile == 'nominal' else MASSES
    return [dict(c, profile=profile) for c in make_cases([1,2,3,4],masses,reps,seed)]

def baseline_module():
    sys.path.insert(0,str(ROOT/'VX'))
    import baselines as mm
    if Path(mm.__file__).resolve() != ROOT/'VX/baselines.py':
        raise RuntimeError('Módulo baseline incorrecto')
    mm.configure_tuned_controllers()
    mm.validate_routes(ROOT/'RUTAS_SAC',[1,2,3,4])
    mm.MAPS_DIR, mm.MAPS_LIST = str(ROOT/'RUTAS_SAC'), [1,2,3,4]
    mm.HEADLESS, mm.REAL_TIME, mm.TRACKING_CONTROLLER = True, False, 'pure_pursuit'
    mm.LOAD_ROUTES_ON_START = True
    mm.SAVE_ROUTES_PER_MAP = mm.SAVE_GENERATED_ROUTE_PLOTS = False
    mm.GENERATE_NEW_RRT_ROUTE_EACH_EPISODE = mm.SAVE_TRACKING_ERROR_PLOTS = False
    mm.clamp_cube_xy_to_pedestal = lambda xy: np.asarray(xy,dtype=float).copy()
    mm.MAP_CACHE.clear()
    mm.load_map_routes_from_folder(mm.MAPS_DIR,mm.MAPS_LIST)
    return mm, mm.ArmIK(mm.model,mm.data,mm.JOINTS,mm.EEF_SITE_NAME,
                       mm.ACTS_ARM,params=mm.IK_PARAMS_OVERRIDE)

def snapshot(model):
    actor = copy.deepcopy(model.actor,{id(model.actor.optimizer):None}).to('cpu')
    actor.set_training_mode(False)
    actor.requires_grad_(False)
    return actor

def physical_hash(env):
    return hashlib.sha256(b''.join(np.asarray(a).tobytes() for a in
        (env.data.qpos,env.data.qvel,env.data.ctrl,env.data.qacc_warmstart))).hexdigest()

def evaluate(model, initial, panel, output, baseline=False):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    actors={'current':snapshot(model),'initial':snapshot(initial)}
    mm,ik=baseline_module()
    rows=[]
    methods=('baseline','initial','off','on') if baseline else ('initial','off','on')
    for index,case in enumerate(panel):
        signatures=[]; handoffs={}
        route=np.asarray(json.loads((ROOT/'RUTAS_SAC'/f'map_{case["map_id"]}.json').read_text())['path'])
        length=float(np.linalg.norm(np.diff(route,axis=0),axis=1).sum())
        for method in methods:
            env=ZennoVXR3(render_mode=None,debug_rewards=False,p1_budget_steps=4500)
            try:
                xy,yaw=(.01,.1) if case['profile']=='robust' else (0.,0.)
                env.set_eval_mode(True); env.FIXED_MAP_ID=case['map_id']
                env.set_eval_mass(case['mass_kg'])
                env.CUBE_XY_NOISE_RANGE,env.CUBE_YAW_NOISE_RANGE_RAD=xy,yaw
                env.reset(seed=case['case_seed'])
                mm.CUBE_XY_NOISE_RANGE,mm.CUBE_YAW_NOISE_RANGE_RAD=xy,yaw
                mm._current_map_id=case['map_id']
                mm._cube_noise_rng=np.random.default_rng(case['case_seed'])
                mm.reset_world_like_rl_for_map(case['map_id'],cube_mass_kg=case['mass_kg'])
                mm._set_home_pose(mm.model,mm.data,ik)
                audit=canonicalize_and_audit(env,mm)
                signatures.append(physical_hash(env)); obs=env._get_obs()
                if method==methods[0]: dump(output/'initial_states'/f'{case["case_id"]}.json',audit)
                driver=BaselineDriver(mm,ik) if method=='baseline' else None
                recorder=Recorder(mm.model,mm.data,route) if driver else env.place_recorder
                counts=[0,0,0]; carry=[]; delta=[]; base_error=None; p1_hash=None
                for _ in range(20000):
                    phase=driver.phase if driver else int(env.phase_state)
                    counts[phase]+=1
                    if driver:
                        driver.control()
                        for _ in range(5): mujoco.mj_step(mm.model,mm.data); recorder.substep()
                        stop=recorder.end_interval(phase,phase==1)
                    else:
                        actor=actors['initial' if method=='initial' else 'current']
                        action=actor.predict(obs,deterministic=True)[0]
                        if method=='off': action[5]=0.
                        obs,_,done,_,info=env.step(action)
                        stop=info['task_stop']
                        if phase==0 and env.phase_state==1: p1_hash=physical_hash(env)
                        if info['lookahead_active']: delta.append(env._lh_applied-env.L_NOMINAL)
                        if method=='off' and abs(env._lh_applied-.3)>1e-10:
                            raise AssertionError('OFF alteró el lookahead nominal')
                        if np.any(np.abs(env._r2_pp_corrected-env._r2_pp_nominal)>1e-10):
                            raise AssertionError('Se activó un residual de velocidad prohibido')
                    fresh=recorder._fresh_auxiliary_state()
                    next_phase=driver.phase if driver else int(env.phase_state)
                    if phase==1:
                        carry.append(float(np.linalg.norm(fresh.body('cube').xpos-fresh.site('setpoint').xpos)))
                    if phase==1 and next_phase==2:
                        base_error=float(np.linalg.norm(fresh.site_xpos[recorder.base_site_id,:2]-route[-1,:2]))
                    if stop is not None: break
                traces=recorder.traces(); row=recorder.summary()
                tracking=(traces['phase']==1)&(traces['route_progress']*length>=.30)&(traces['route_progress']*length<=length-.55)
                yaw_error=(traces['base_yaw_rad']-traces['path_yaw_rad']+np.pi)%(2*np.pi)-np.pi
                for name,array in (('cte_rms_m',traces['cte_signed_m']),('yaw_rms_rad',yaw_error)):
                    row[name]=float(np.sqrt(np.mean(array[tracking]**2))) if tracking.any() else None
                row.update(case,method=method,success_place=stop=='stable_place',termination_reason=stop,
                    phase_steps=counts,base_terminal_error_m=base_error,
                    carry_rms_m=float(np.sqrt(np.mean(np.square(carry)))) if carry else None,
                    lookahead_delta_rms_m=float(np.sqrt(np.mean(np.square(delta)))) if delta else 0.,
                    initial_hash=signatures[-1],p1_hash=p1_hash)
                rows.append(row); handoffs[method]=p1_hash
                dump(output/'episodes'/method/f'{case["case_id"]}.json',row)
            finally: env.close()
        if len(set(signatures))!=1 or handoffs['on']!=handoffs['off']:
            raise AssertionError('Estados iniciales o handoff ON/OFF distintos')
        if index+1 in (max(1,len(panel)//2),len(panel)):
            print('EVAL_PROGRESS',json.dumps({'cases':index+1,'total':len(panel)}),flush=True)
    report=assess(rows)
    dump(output/'summary.json',dict(rows=rows,assessment=report,complete=True,
        protocol=dict(agent_budgets=[1000,4500,1000],baseline_global_cap_s=200,
        placement='support + release + height + speed stable 0.2s; no home',
        inference='CPU for all snapshots',energy='absolute work, all 12 actuators',
        off='same current arm, fixed L=0.30m; diagnostic, not classical baseline')))
    return report

def assess(rows):
    groups={}
    for r in rows:
        group=groups.setdefault(r['case_id'],{})
        if r['method'] in group: raise ValueError('Caso/método duplicado')
        group[r['method']]=r
    if not groups: raise ValueError('Panel vacío')
    methods=set(next(iter(groups.values())))
    if any(set(g)!=methods for g in groups.values()): raise ValueError('Panel incompleto')
    rates={m:float(np.mean([g[m]['success_place'] for g in groups.values()])) for m in methods}
    comparisons={}; safe=True
    for ref in ('initial','off'):
        for mid in (1,2,3,4):
            subset=[g for g in groups.values() if g['on']['map_id']==mid]
            safe &= sum(g['on']['success_place'] for g in subset)>=sum(g[ref]['success_place'] for g in subset)
        paired=[g for g in groups.values() if g['on']['success_place'] and g[ref]['success_place']]
        ratios={}
        for metric in METRICS:
            valid=[g for g in paired if all(g[m].get(metric) is not None for m in ('on',ref))]
            if valid:
                a,b=[np.mean([g[m][metric] for g in valid]) for m in ('on',ref)]
                ratios[metric]=float(max(a,1e-6)/max(b,1e-6))
        comparisons[ref]=dict(pairs=len(paired),ratios=ratios)
    active=any(g['on']['lookahead_delta_rms_m']>1e-4 for g in groups.values())
    guards=all(len(v['ratios'])==len(METRICS) and all(
        x<=(1.20 if k=='total_energy_J' else 1.10) for k,x in v['ratios'].items()) for v in comparisons.values())
    eligible=bool(safe and active and guards and rates['on']>=.90)
    ratios=comparisons['off']['ratios']
    balance=float(np.exp(np.mean(np.log(list(ratios.values()))))) if ratios else None
    improved=rates['on']>rates['off'] or (balance is not None and balance<=.99)
    return dict(n_cases=len(groups),rates=rates,no_map_success_regression=bool(safe),active=active,
                metric_guard=guards,comparisons=comparisons,balance_vs_off=balance,
                curriculum_eligible=eligible,recommended=bool(eligible and improved and rates['on']>=.95))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True); p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--reps',type=int,default=10)
    p.add_argument('--seed',type=int,default=410_000_000); p.add_argument('--baseline',action='store_true')
    a=p.parse_args(); torch.set_num_threads(2)
    m=CurriculumSAC.load(a.model,device='cpu'); initial=CurriculumSAC.load(a.initial,device='cpu')
    print(json.dumps(evaluate(m,initial,cases(reps=a.reps,seed=a.seed),a.output,a.baseline),indent=2))
