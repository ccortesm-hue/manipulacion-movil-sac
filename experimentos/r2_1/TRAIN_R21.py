"""Restricted SAC fine-tune of final R2 actor; no BC, DAgger or teacher at inference."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time
import gymnasium as gym
from VX_R2_1 import (ROOT,HERE,SOURCE,ZennoVXR21,np,torch,build,load_norm,
                     protected_state,assert_protected)
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from EVALUATE_R21 import evaluate, make_cases, dump


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TrainingEnv(gym.Env):
    """Physically roll P0, then retain every P1/P2 transition in replay.

    Only P1 residual parameters have gradients. P2 still supplies Bellman targets
    across the real handoff. P0 failures are logged and NEVER hidden in evaluation.
    """
    def __init__(self,norm,seed,log=None):
        self.env=ZennoVXR21(render_mode=None,debug_rewards=False,p1_budget_steps=4500)
        self.norm,self.model,self.log=norm,None,log
        self.rng=np.random.default_rng(seed); self.order=[]
        self.action_space=self.env.action_space
        self.observation_space=gym.spaces.Box(-norm.clip_obs,norm.clip_obs,(124,),dtype=np.float32)

    def normalize(self,obs):
        return self.norm.normalize_obs(np.asarray(obs,np.float32).copy()).astype(np.float32)

    def record(self,event,**data):
        if self.log:
            with self.log.open('a') as f: f.write(json.dumps(dict(event=event,**data),allow_nan=False)+'\n')

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        if seed is not None: self.rng=np.random.default_rng(seed)
        if self.model is None: raise RuntimeError('Bind the single actor before reset')
        for _ in range(40):
            if not self.order: self.order=self.rng.permutation(20).tolist()
            i=self.order.pop(); mid,mass=1+i//5,(.5,.75,1.,1.25,1.5)[i%5]
            self.case=dict(map=mid,mass=mass,seed=int(self.rng.integers(100000000,200000000)))
            e=self.env; e.set_eval_mode(True); e.FIXED_MAP_ID=mid; e.set_eval_mass(mass)
            e.CUBE_XY_NOISE_RANGE,e.CUBE_YAW_NOISE_RANGE_RAD=.01,.1
            obs,_=e.reset(seed=self.case['seed']); term=trunc=False; steps=0; info={}
            while e.phase_state==0 and not(term or trunc):
                action=self.model.predict(self.normalize(obs),deterministic=True)[0]
                obs,_,term,trunc,info=e.step(action); steps+=1
            if e.phase_state==1 and not(term or trunc):
                self.steps=0; self.phases=[0,0,0]; self.reward_sum=0.
                self.record('P0_TO_P1',**self.case,p0_steps=steps)
                return self.normalize(obs),dict(real_handoff=True,**self.case)
            self.record('P0_FAILURE',**self.case,p0_steps=steps,
                        reason=info.get('termination_reason') or info.get('truncation_reason'))
        raise RuntimeError('40 real P0 failures; no synthetic P1 starts permitted')

    def step(self,action):
        phase=int(self.env.phase_state)
        obs,r,term,trunc,info=self.env.step(action)
        self.steps+=1; self.phases[phase]+=1; self.reward_sum+=r
        done=bool(term or trunc)
        if done:
            self.record('EPISODE_END',**self.case,steps=self.steps,phase_steps=self.phases,
                        reward=self.reward_sum,placed=bool(self.env._placed_success),
                        reason=info.get('termination_reason') or info.get('truncation_reason'))
        info['TimeLimit.truncated']=False
        return self.normalize(obs),float(r),done,False,info

    def close(self): self.env.close()


def validate_source(path):
    manifest=json.loads(path.with_name('final_student.json').read_text())
    if manifest['experiment']!='r2': raise ValueError('Historical final R2 required')
    for name,file in [('model',path),('vecnormalize',path.with_name('vecnormalize.pkl'))]:
        if sha(file)!=manifest['artifacts'][name]['sha256']: raise ValueError(f'Changed source {name}')
    if sha(ROOT/'zenno_vx_r2.py')!=manifest['environment']['sha256']:
        raise ValueError('R2 environment no longer matches source manifest')
    for name,value in manifest['route_hashes'].items():
        if sha(ROOT/'RUTAS_SAC'/name)!=value: raise ValueError(f'Changed route {name}')
    return manifest


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,default=SOURCE)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--steps',type=int,default=100000)
    p.add_argument('--block-steps',type=int,default=25000)
    p.add_argument('--prefill',type=int,default=5000)
    p.add_argument('--critic-warmup',type=int,default=2000)
    p.add_argument('--buffer-size',type=int,default=200000)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--authority',type=float,default=.5)
    p.add_argument('--actor-lr',type=float,default=1e-5)
    p.add_argument('--critic-lr',type=float,default=3e-4)
    p.add_argument('--seed',type=int,default=21092026)
    p.add_argument('--dev-reps',type=int,default=2)
    p.add_argument('--holdout-reps',type=int,default=10)
    p.add_argument('--check',action='store_true',help='Load, verify frozen arm and real P0→P1 reset; no learning')
    return p


def run(args):
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing silent CPU training')
    if min(args.steps,args.block_steps,args.prefill,args.batch_size,args.buffer_size,args.critic_warmup)<1:
        raise ValueError('Positive training budgets required')
    if args.prefill<args.batch_size or args.buffer_size<args.prefill:
        raise ValueError('buffer_size >= prefill >= batch_size required')
    if not args.check and args.steps<=args.prefill+args.critic_warmup:
        raise ValueError('No actor updates would be performed')
    if not 1<=args.dev_reps<1000 or not 1<=args.holdout_reps<1000:
        raise ValueError('Evaluation repetitions in [1,999]')
    if not np.isfinite([args.authority,args.actor_lr,args.critic_lr]).all() or not 0<args.authority<=1 or min(args.actor_lr,args.critic_lr)<=0:
        raise ValueError('Invalid authority or learning rate')
    out=args.output.resolve()
    if not out.is_relative_to(HERE/'runs'): raise ValueError('Output must be inside VX_R2_1/runs')
    if out.exists(): raise FileExistsError(out)
    print('INITIALIZING',json.dumps(dict(device=args.device,output=str(out),source=str(args.source))),flush=True)
    source=validate_source(args.source)
    torch.set_num_threads(1)
    norm=load_norm(args.source.with_name('vecnormalize.pkl'))
    env=TrainingEnv(norm,args.seed)
    try:
        model,donor=build(env,norm,args); env.model=model
        frozen=protected_state(model.actor)
        # Verify identical deterministic arm outputs on source-distributed real observations.
        initial_obs=env.reset()[0]
        a=model.predict(initial_obs,deterministic=True)[0]
        b=donor.predict(initial_obs,deterministic=True)[0]
        np.testing.assert_allclose(a[:5],b[:5],atol=1e-6,rtol=0)
        assert_protected(model.actor,frozen)
        del donor
        if args.check:
            print('READY',json.dumps(dict(actor_count=1,obs_dim=124,action_dim=7,
                  real_p0_to_p1=True,arm_unchanged=True,device=args.device)),flush=True)
            return
        out.mkdir(parents=True,exist_ok=False); env.log=out/'training_episodes.jsonl'
        files=list(HERE.glob('*.py'))+[ROOT/'zenno_vx_r2.py',ROOT/'evaluate_zenno_vx_r2.py',
            ROOT/'evaluate_zenno_vx_r1_1.py',ROOT/'vx_pose_metrics.py',ROOT/'vx_place_protocol.py',
            ROOT/'vx_place_initials.py',ROOT/'extract_vx_place_robust.py',ROOT/'VX/baselines.py']
        hashes={str(p):sha(p) for p in files}
        dump(out/'config.json',dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            revision=ZennoVXR21.REVISION,source=source,source_files=hashes,
            authority_fixed_per_run=args.authority,phase_steps=[1000,4500,1000],
            fresh_critics_replay_optimizers=True,normalization='external frozen once',
            training='real P0 prefix, replay P1 and P2, gradients only P1 base outputs',
            reward='unchanged R2 including home in training; evaluation stops at stable placement',
            dev_seed=241000000,holdout_seed=341000000,training_seed_range=[100000000,200000000],
            gpu=torch.cuda.get_device_name(0) if args.device=='cuda' else None))
        model.set_logger(configure(str(out/'sac_logs'),['csv']))
        start=time.monotonic()
        def announce(stage,**data):
            record=dict(stage=stage,elapsed_s=time.monotonic()-start,**data)
            dump(out/'status.json',record)
            with (out/'milestones.jsonl').open('a') as f: f.write(json.dumps(record,allow_nan=False)+'\n')
            print(stage,json.dumps(data),flush=True)
        def save(tag):
            assert_protected(model.actor,frozen)
            if any(not torch.isfinite(p).all() for p in model.policy.parameters()):
                raise FloatingPointError('Nonfinite model')
            d=out/tag; d.mkdir(exist_ok=False)
            model.save(d/'model.zip'); norm.save(d/'vecnormalize.pkl')
            dump(d/'checkpoint.json',dict(steps=model.num_timesteps,critic_updates=model._n_updates,
                actor_updates=max(0,model._n_updates-args.critic_warmup),actor_count=1,
                arm_unchanged=True,authority=args.authority,model_sha256=sha(d/'model.zip'),
                normalizer_sha256=sha(d/'vecnormalize.pkl')))
            return d/'model.zip'
        class Milestones(BaseCallback):
            def __init__(self,steps):
                super().__init__(); self.at=model.num_timesteps+max(1,steps//2)
            def _on_step(self):
                if self.num_timesteps>=self.at:
                    assert_protected(model.actor,frozen)
                    announce('TRAIN_MIDPOINT',steps=self.num_timesteps,updates=model._n_updates)
                    self.at=float('inf')
                return True
        try:
            cases=make_cases([1,2,3,4],[.5,.75,1.,1.25,1.5],args.dev_reps,241000000)
            save('initial'); announce('INITIAL_EVALUATION')
            evaluate(model,norm,cases,out/'eval_initial',announce)
            selected=None; best_score=(float('inf'),float('inf'))
            while model.num_timesteps<args.steps:
                remaining=min(args.block_steps,args.steps-model.num_timesteps)
                announce('TRAIN_BLOCK_START',steps=model.num_timesteps,target=model.num_timesteps+remaining)
                model.learn(remaining,reset_num_timesteps=False,callback=Milestones(remaining),log_interval=1)
                checkpoint=save(f'block_{model.num_timesteps:07d}')
                announce('TRAIN_BLOCK_END',steps=model.num_timesteps,updates=model._n_updates)
                report=evaluate(model,norm,cases,checkpoint.parent/'development',announce)
                score=(-report['success_rates']['on'],report['balance_ratio'] if report['balance_ratio'] is not None else float('inf'))
                if report['qualified'] and score<best_score:
                    selected=checkpoint; best_score=score
                dump(out/'selection.json',dict(selected=str(selected) if selected else None,
                     reason='qualified active candidate' if selected else 'NO_QUALIFIED_ACTIVE_RESIDUAL',last=report))
                announce('DEVELOPMENT_RESULT',**report)
            holdout=None
            if selected:
                from VX_R2_1 import ResidualSAC
                candidate=ResidualSAC.load(selected,device='cpu',custom_objects={'buffer_size':1})
                announce('HOLDOUT_START',model=str(selected))
                holdout=evaluate(candidate,norm,make_cases([1,2,3,4],[.5,.75,1.,1.25,1.5],args.holdout_reps,341000000),out/'holdout',announce)
            if any(sha(p)!=digest for p,digest in hashes.items()): raise RuntimeError('Sources changed during run')
            dump(out/'result.json',dict(complete=True,selected=str(selected) if selected else None,
                holdout=holdout,validated_active_residual=bool(holdout and holdout['qualified']),
                note='No historical checkpoint overwritten; no automatic thesis promotion'))
            announce('FINISHED',selected=str(selected) if selected else None,
                     validated_active_residual=bool(holdout and holdout['qualified']))
        except BaseException as exc:
            announce('INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'ERROR',error=repr(exc))
            raise
    finally:
        env.close(); norm.close()


if __name__=='__main__':
    args=parser().parse_args()
    with (HERE/'.training.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args)
