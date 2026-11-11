"""Paired baseline/initial/current OFF/current ON; stable placement, no home."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from VX_R2_1_COUPLED import ROOT, ResidualSAC, ZennoVXR21, load_norm, np, torch
from extract_vx_place_robust import BaselineDriver, evaluation_environment, dump, make_cases
from vx_place_protocol import Recorder
from vx_place_initials import canonicalize_and_audit
from evaluate_zenno_vx_r1_1 import initial_state, state_hash
import mujoco

METHODS = ('baseline','reference','off','on')
METRICS = ('total_time_s','total_energy_J','cte_rms_m','yaw_rms_rad',
           'cube_goal_dist_xy_m','carry_rms_m','base_terminal_error_m')


def evaluation_actors(model, reference_model):
    """Snapshot actors ONLY on CPU; never move live SAC optimizers/replay.

    CPU/GPU float32 inference is not bitwise identical. Mixing backends before
    the P1 handoff can change MuJoCo trajectories even with identical weights.
    All paired rollouts use the same backend, including during CUDA training.
    """
    actors={name:copy.deepcopy(source.actor,{id(source.actor.optimizer):None}).to('cpu')
            for name,source in (('current',model),('reference',reference_model))}
    for actor in actors.values():
        actor.set_training_mode(False)
        actor.requires_grad_(False)
    return actors


def baseline_module():
    sys.path.insert(0,str(ROOT/'VX'))
    import baselines as mm
    if Path(mm.__file__).resolve() != ROOT/'VX/baselines.py':
        raise RuntimeError('Unexpected baseline module')
    mm.configure_tuned_controllers()
    mm.validate_routes(ROOT/'RUTAS_SAC',[1,2,3,4])
    mm.MAPS_DIR, mm.MAPS_LIST = str(ROOT/'RUTAS_SAC'), [1,2,3,4]
    mm.HEADLESS, mm.REAL_TIME, mm.TRACKING_CONTROLLER = True, False, 'pure_pursuit'
    mm.LOAD_ROUTES_ON_START = True
    mm.SAVE_ROUTES_PER_MAP = mm.SAVE_GENERATED_ROUTE_PLOTS = False
    mm.GENERATE_NEW_RRT_ROUTE_EACH_EPISODE = mm.SAVE_TRACKING_ERROR_PLOTS = False
    mm.CUBE_XY_NOISE_RANGE, mm.CUBE_YAW_NOISE_RANGE_RAD = .01,.1
    mm.clamp_cube_xy_to_pedestal = lambda xy: np.asarray(xy,dtype=float).copy()
    mm.MAP_CACHE.clear()
    mm.load_map_routes_from_folder(mm.MAPS_DIR,mm.MAPS_LIST)
    return mm, mm.ArmIK(mm.model,mm.data,mm.JOINTS,mm.EEF_SITE_NAME,mm.ACTS_ARM,params=mm.IK_PARAMS_OVERRIDE)


def evaluate(model, norm, cases, output, announce=lambda *a,**k:None, reference_model=None):
    if reference_model is None:
        raise ValueError('Frozen initial checkpoint required for anti-regression evaluation')
    output = Path(output); output.mkdir(parents=True,exist_ok=False)
    actors=evaluation_actors(model,reference_model)
    mm, ik = baseline_module()
    rows = []
    cls = evaluation_environment(ZennoVXR21)
    for index,case in enumerate(cases):
        signatures = []
        handoffs = []
        route = np.asarray(json.loads((ROOT/'RUTAS_SAC'/f'map_{case["map_id"]}.json').read_text())['path'])
        length = float(np.linalg.norm(np.diff(route,axis=0),axis=1).sum())
        for method in METHODS:
            env = cls(render_mode=None,debug_rewards=False,p1_budget_steps=4500)
            try:
                env.set_eval_mode(True); env.FIXED_MAP_ID=case['map_id']; env.set_eval_mass(case['mass_kg'])
                env.CUBE_XY_NOISE_RANGE,env.CUBE_YAW_NOISE_RANGE_RAD=.01,.1
                env.reset(seed=case['case_seed'])
                mm._current_map_id=case['map_id']; mm._cube_noise_rng=np.random.default_rng(case['case_seed'])
                mm.reset_world_like_rl_for_map(case['map_id'],cube_mass_kg=case['mass_kg'])
                mm._set_home_pose(mm.model,mm.data,ik)
                audit=canonicalize_and_audit(env,mm)
                obs=env._get_obs()
                signatures.append(state_hash(initial_state(env,obs)))
                if method=='baseline':
                    dump(output/'initial_states'/f'{case["case_id"]}.json',audit)
                physical,data=(mm.model,mm.data) if method=='baseline' else (env.model,env.data)
                recorder=Recorder(physical,data,route)
                driver=BaselineDriver(mm,ik) if method=='baseline' else None
                env.place_recorder=recorder if driver is None else None
                counts=[0,0,0]; requests=[]; deltas=[]; carry=[]; p1_hash=None; base_error=None
                arm_deviation={1:[],2:[]}; history_clipped=0; history_samples=0
                stop=None
                for step in range(20000):
                    phase=driver.phase if driver else int(env.phase_state)
                    counts[phase]+=1
                    if driver:
                        driver.control()
                        for _ in range(5):
                            mujoco.mj_step(physical,data); recorder.substep()
                    else:
                        policy=actors['reference' if method=='reference' else 'current']
                        normalized=norm.normalize_obs(obs.copy())
                        action=np.asarray(policy.predict(normalized,deterministic=True)[0],np.float32)
                        if phase in (1,2):
                            with torch.no_grad():
                                prior=policy.prior_logits(torch.as_tensor(normalized[None],device=policy.device)).tanh().cpu().numpy()[0]
                            arm_deviation[phase].append(float(np.mean((action[:5]-prior)**2)))
                        if phase==1:
                            history_clipped+=int(np.sum(np.abs(normalized[30:32])>=norm.clip_obs-1e-5))
                            history_samples+=2
                        if method=='off': action[5:]=0
                        obs,_,_,_,_=env.step(action)
                        if phase==0 and env.phase_state==1:
                            p1_hash=state_hash(initial_state(env,obs))
                        if phase==1:
                            requests.append(action[5:].copy())
                            delta=env._r2_pp_corrected-env._r2_pp_nominal
                            deltas.append(delta.copy())
                            if method=='off' and np.any(np.abs(delta)>1e-10):
                                raise AssertionError('OFF produced a physical reference correction')
                    next_phase=driver.phase if driver else int(env.phase_state)
                    fresh=recorder._fresh_auxiliary_state()
                    if phase==1:
                        carry.append(float(np.linalg.norm(fresh.body('cube').xpos-fresh.site('setpoint').xpos)))
                    if phase==1 and next_phase==2:
                        base_error=float(np.linalg.norm(fresh.site_xpos[recorder.base_site_id,:2]-route[-1,:2]))
                    stop=recorder.end_interval(phase,phase==1,
                        native_deadline=not driver and counts[phase]>=env.phase_budgets[phase])
                    if stop is not None: break
                    if not driver:
                        # Keep R2 native failure/deadline semantics, but home is not task success.
                        failures=[d for d in env.place_native_diagnostics
                                  if d['reason'] not in ('place_success_arm_home','phase2_home_budget')]
                        if failures:
                            stop=failures[-1]['reason']; break
                        if counts[phase]>=env.phase_budgets[phase]:
                            stop=f'phase{phase}_budget'; break
                traces=recorder.traces(); summary=recorder.summary()
                tracking=(traces['phase']==1)&(traces['route_progress']*length>=.30)&(traces['route_progress']*length<=length-.55)
                for key,values in [('cte_rms_m',traces['cte_signed_m']),
                                   ('yaw_rms_rad',(traces['base_yaw_rad']-traces['path_yaw_rad']+np.pi)%(2*np.pi)-np.pi)]:
                    summary[key]=float(np.sqrt(np.mean(values[tracking]**2))) if tracking.any() else None
                summary['cte_signed_mean_m']=float(np.mean(traces['cte_signed_m'][tracking])) if tracking.any() else None
                summary['tracking_samples']=int(tracking.sum())
                delta_rms=np.sqrt(np.mean(np.square(deltas),axis=0)).tolist() if deltas else [0.,0.]
                summary.update(case,method=method,termination_reason=stop,
                    success_place=stop=='stable_place',native_phase_counts=counts,
                    carry_rms_m=float(np.sqrt(np.mean(np.square(carry)))) if carry else None,
                    base_terminal_error_m=base_error,
                    arm_correction_rms_by_phase={str(k):float(np.sqrt(np.mean(v))) if v else 0. for k,v in arm_deviation.items()},
                    history_clip_fraction=history_clipped/max(1,history_samples),
                    requested_action_rms=float(np.sqrt(np.mean(np.square(requests)))) if requests else 0.,
                    reference_delta_rms_v_omega=delta_rms,initial_hash=signatures[-1],p1_hash=p1_hash)
                rows.append(summary)
                dump(output/'episodes'/method/f'{case["case_id"]}.json',summary)
                if method!='baseline': handoffs.append(p1_hash)
            finally:
                env.close()
        if len(set(signatures))!=1 or len(set(handoffs))!=1:
            dump(output/'pairing_error.json',dict(case=case,
                initial_hashes=dict(zip(METHODS,signatures)),
                handoff_hashes=dict(zip(METHODS[1:],handoffs)),
                inference_device='cpu',reason='Exact paired-state check failed'))
            raise AssertionError('Unpaired initial states or pre-residual P1 handoffs')
        if index+1 in (max(1,len(cases)//2),len(cases)):
            announce('EVAL_PROGRESS',cases=index+1,total=len(cases))
    report=assess(rows)
    dump(output/'summary.json',dict(complete=True,rows=rows,assessment=report,
        protocol=dict(placement='physical support, release, speed and height stable for 0.2 s; no home',
            inference_device='cpu',actor_snapshots=True,
            agent_phase_steps=[1000,4500,1000],baseline_global_steps=20000,
            energy='absolute mechanical work of all 12 actuators including gripper',
            perturbations=dict(xy_m=.01,yaw_rad=.1),off='zero base action only; current learned arm remains active',
            reference='initial checkpoint of this coupled run; same cases and normalizer',
            tracking='CTE RMS, signed mean and yaw share route-progress interior window')))
    return report


def assess(rows):
    groups={}
    for row in rows:
        group=groups.setdefault(row['case_id'],{})
        if row['method'] in group: raise ValueError('Duplicate case/method')
        group[row['method']]=row
    if not groups or any(set(g)!=set(METHODS) for g in groups.values()):
        raise ValueError('Incomplete paired evaluation')
    rates={m:sum(g[m]['success_place'] for g in groups.values())/len(groups) for m in METHODS}
    no_regression=all(sum(g['on']['success_place'] for g in groups.values() if g['on']['map_id']==mid)
                      >=sum(g['off']['success_place'] for g in groups.values() if g['off']['map_id']==mid)
                      for mid in {g['on']['map_id'] for g in groups.values()})
    paired=[g for g in groups.values() if g['on']['success_place'] and g['off']['success_place']]
    common=[g for g in groups.values() if all(g[m]['success_place'] for m in METHODS)]
    ratios={}; table={}
    for metric in METRICS:
        valid=[g for g in paired if all(g[m][metric] is not None and np.isfinite(g[m][metric]) for m in ('on','off'))]
        if valid:
            a,b=[np.mean([g[m][metric] for g in valid]) for m in ('on','off')]
            ratios[metric]=float(max(a,1e-6)/max(b,1e-6))
        valid3=[g for g in common if all(g[m][metric] is not None and np.isfinite(g[m][metric]) for m in METHODS)]
        table[metric]={m:float(np.mean([g[m][metric] for g in valid3])) if valid3 else None for m in METHODS}
    complete=len(ratios)==len(METRICS) and bool(paired)
    balance=float(np.exp(np.mean(np.log(list(ratios.values()))))) if complete else None
    activity=any(g['on']['reference_delta_rms_v_omega'][0]>1e-4 or
                 g['on']['reference_delta_rms_v_omega'][1]>1e-3 for g in groups.values())
    bounded=complete and all(v<=1.05 for v in ratios.values())
    improved=complete and (rates['on']>rates['off'] or balance<=.99)
    reference_pairs=[g for g in groups.values() if g['on']['success_place'] and g['reference']['success_place']]
    reference_ratios={}
    for metric in METRICS:
        valid=[g for g in reference_pairs if all(g[m][metric] is not None and np.isfinite(g[m][metric]) for m in ('on','reference'))]
        if valid:
            a,b=[np.mean([g[m][metric] for g in valid]) for m in ('on','reference')]
            reference_ratios[metric]=float(max(a,1e-6)/max(b,1e-6))
    reference_no_regression=all(sum(g['on']['success_place'] for g in groups.values() if g['on']['map_id']==mid)
        >=sum(g['reference']['success_place'] for g in groups.values() if g['reference']['map_id']==mid)
        for mid in {g['on']['map_id'] for g in groups.values()})
    reference_guard=len(reference_ratios)==len(METRICS) and all(v<=1.05 for v in reference_ratios.values())
    reference_balance=float(np.exp(np.mean(np.log(list(reference_ratios.values()))))) if len(reference_ratios)==len(METRICS) else None
    reference_improved=reference_balance is not None and (rates['on']>rates['reference'] or reference_balance<=.99)
    qualified=bool(no_regression and rates['on']>=.90 and activity and bounded and improved
                   and reference_no_regression and reference_guard and reference_improved)
    return dict(qualified=qualified,success_rates=rates,common_successes=len(common),
                common_successes_on_off=len(paired),raw_common_metrics=table,ratios_on_off=ratios,
                balance_ratio=balance,ratios_on_reference=reference_ratios,
                reference_balance_ratio=reference_balance,reference_no_success_regression=reference_no_regression,
                reference_metric_guard=reference_guard,reference_improvement=reference_improved,
                residual_physically_active=activity,no_success_regression=no_regression,
                metric_guard=bounded,improvement=improved,
                note='Predeclared descriptive development screen, not a significance test or guarantee')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reps',type=int,default=10)
    p.add_argument('--seed',type=int,default=341000000)
    p.add_argument('--maps',type=int,nargs='+',default=[1,2,3,4])
    p.add_argument('--masses',type=float,nargs='+',default=[.5,.75,1.,1.25,1.5])
    args=p.parse_args()
    if not 1<=args.reps<1000: p.error('reps must be in [1,999]')
    if not set(args.maps)<={1,2,3,4} or not set(args.masses)<={.5,.75,1.,1.25,1.5}:
        p.error('Unknown map or mass')
    if len(set(args.maps))!=len(args.maps) or len(set(args.masses))!=len(args.masses):
        p.error('Duplicated maps or masses')
    if not 0<=args.seed<2**32-500000: p.error('Seed out of range')
    digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    checkpoint=json.loads(args.model.with_name('checkpoint.json').read_text())
    if digest(args.model)!=checkpoint['model_sha256'] or digest(args.model.with_name('vecnormalize.pkl'))!=checkpoint['normalizer_sha256']:
        raise ValueError('Checkpoint/normalizer mismatch')
    config=json.loads((args.model.parent.parent/'config.json').read_text())
    for path,expected in config['source_files'].items():
        if digest(path)!=expected: raise ValueError(f'Changed source: {path}')
    for name,expected in config['source']['route_hashes'].items():
        if digest(ROOT/'RUTAS_SAC'/name)!=expected: raise ValueError(f'Changed route: {name}')
    torch.set_num_threads(1)
    model=ResidualSAC.load(args.model,device='cpu',custom_objects={'buffer_size':1})
    initial_path=args.model.parent.parent/'initial/model.zip'
    initial_meta=json.loads(initial_path.with_name('checkpoint.json').read_text())
    if digest(initial_path)!=initial_meta['model_sha256']:
        raise ValueError('Initial reference checksum mismatch')
    reference=ResidualSAC.load(initial_path,device='cpu',custom_objects={'buffer_size':1})
    norm=load_norm(args.model.with_name('vecnormalize.pkl'))
    try:
        report=evaluate(model,norm,make_cases(args.maps,args.masses,args.reps,args.seed),args.output,
                        lambda stage,**kw:print(stage,json.dumps(kw),flush=True),reference_model=reference)
        print(json.dumps(report,indent=2))
    finally: norm.close()


if __name__=='__main__': main()
