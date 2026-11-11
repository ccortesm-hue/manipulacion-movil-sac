"""SAC puro con currículo de dificultad y autoridad, sin BC/DAgger nuevos."""
import argparse
from datetime import datetime
import hashlib
import json
import os
import pickle
import random
import signal
import time
import fcntl
import traceback
from pathlib import Path
import gymnasium as gym
from VX_R3 import (ROOT, ZennoVXR3, CurriculumActor, CurriculumPolicy,
                   CurriculumSAC, PhysicalFeatures, SAC, np, torch)
from EVALUATE_R3 import cases, evaluate, MASSES
from extract_vx_place_robust import dump
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'models/zenno_vx_r1_run2/best/model.zip'
NORM=SOURCE.with_name('vecnormalize.pkl')
STAGES=('lookahead_nominal','lookahead_robust','coupled_P1_P2','full_P0_P1_P2')
REWARD_SCALE=.01


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def provenance():
    paths=list(HERE.glob('*.py'))+[ROOT/'VX/MM_DATA_VXR1.py',ROOT/'VX/MM_DATA_VXR2.py',
        ROOT/'VX/baselines.py',ROOT/'extract_vx_place_robust.py',ROOT/'vx_place_protocol.py',
        ROOT/'vx_place_initials.py',SOURCE,NORM]+list((ROOT/'RUTAS_SAC').glob('*.json'))
    return {str(p):sha(p) for p in paths}


class CurriculumEnv(gym.Wrapper):
    def __init__(self, seed, log=None):
        super().__init__(ZennoVXR3(render_mode=None,debug_rewards=False,p1_budget_steps=4500))
        self.rng=np.random.default_rng(seed); self.stage=0; self.queue=[]
        self.episodes=0; self.log=log

    def set_stage(self, stage):
        self.stage=int(stage); self.queue=[]

    def reset(self, *, seed=None, options=None):
        if seed is not None: self.rng=np.random.default_rng(seed)
        masses=(1.,) if self.stage==0 else MASSES
        if not self.queue: self.queue=self.rng.permutation(4*len(masses)).tolist()
        index=self.queue.pop(); mass=masses[index%len(masses)]; mid=1+index//len(masses)
        self.env.set_eval_mode(True); self.env.FIXED_MAP_ID=mid; self.env.set_eval_mass(mass)
        self.env.CUBE_XY_NOISE_RANGE=0. if self.stage==0 else .01
        self.env.CUBE_YAW_NOISE_RANGE_RAD=0. if self.stage==0 else .1
        episode_seed=int(self.rng.integers(100_000_000,200_000_000))
        self.case=dict(map_id=mid,mass_kg=mass,seed=episode_seed)
        self.returns=[0.,0.,0.]; self.lookahead=[]
        return self.env.reset(seed=episode_seed,options=options)

    def step(self, action):
        phase=int(self.env.phase_state)
        obs,reward,term,trunc,info=self.env.step(action)
        self.returns[phase]+=reward
        if info['lookahead_active']: self.lookahead.append(info['lookahead_applied_m']-.3)
        if term or trunc:
            self.episodes+=1
            row=dict(self.case,episode=self.episodes,stage=STAGES[self.stage],
                phase_returns_raw=self.returns,phase_steps=self.env.phase_counts,
                success_place=info['success_place'],reason=info['task_stop'],
                lookahead_delta_rms_m=float(np.sqrt(np.mean(np.square(self.lookahead)))) if self.lookahead else 0.)
            if self.log:
                with self.log.open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
        return obs,REWARD_SCALE*reward,term,trunc,info


def build(vec, device, seed, buffer_size):
    with NORM.open('rb') as f: norm=pickle.load(f)
    source=SAC.load(SOURCE,device=device,custom_objects={'buffer_size':1})
    model=CurriculumSAC(CurriculumPolicy,vec,device=device,seed=seed,
        learning_rate=1e-4,buffer_size=buffer_size,learning_starts=5000,
        batch_size=256,train_freq=4,gradient_steps=1,gamma=.999,tau=.005,
        ent_coef=.0001,replay_buffer_kwargs={'handle_timeout_termination':False},
        policy_kwargs=dict(net_arch=[128,128],features_extractor_class=PhysicalFeatures),verbose=0)
    model.actor.initialize_arm(source,norm)
    for module in (model.actor,model.critic,model.critic_target):
        module.features_extractor.configure(norm)
    return model


def save_resume(model, wrapper, output, state):
    # Nueva generación primero; latest.json se sustituye SOLO tras escritura completa.
    target=output/'checkpoints'/f'{model.num_timesteps}_{time.time_ns()}'
    target.mkdir(parents=True,exist_ok=False)
    model.save(target/'model.zip'); model.save_replay_buffer(target/'replay.pkl')
    runtime=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        sampler=wrapper.rng.bit_generator.state,queue=wrapper.queue,episodes=wrapper.episodes)
    with (target/'runtime.pkl').open('wb') as f: pickle.dump(runtime,f)
    dump(target/'state.json',dict(state,steps=int(model.num_timesteps),
        sac_gradient_updates=int(model._n_updates),
        lookahead_mean_weight_norm=float(model.actor.mu.weight.detach().norm().cpu()),
        lookahead_mean_bias=float(model.actor.mu.bias.detach().cpu()[0]),
        model_sha256=sha(target/'model.zip'),replay_sha256=sha(target/'replay.pkl')))
    dump(output/'latest.json',dict(checkpoint=str(target)))
    return target


class Progress(BaseCallback):
    def __init__(self, stop, announce, size):
        super().__init__(); self.stop=stop; self.announce=announce; self.size=size
    def _on_training_start(self):
        self.middle=self.model.num_timesteps+self.size//2; self.reported=False
    def _on_step(self):
        if not self.reported and self.num_timesteps>=self.middle:
            self.announce('TRAIN_MIDPOINT',steps=self.num_timesteps,
                          updates=self.model._n_updates); self.reported=True
        return not self.stop['requested']


def run(args):
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA no disponible; no se cambia silenciosamente a CPU')
    if args.block_steps<1 or any(s<1 for s in args.stage_steps) or args.buffer_size<256:
        raise ValueError('Presupuestos y buffer inválidos')
    torch.set_num_threads(2)
    if args.resume and args.output: raise ValueError('Usa --resume o --output, no ambos')
    if args.check and args.resume: raise ValueError('--check sólo valida una transferencia nueva')
    if args.resume:
        output=args.resume.resolve()
        config=json.loads((output/'config.json').read_text())
        if config['hashes']!=provenance(): raise ValueError('Código/fuentes/rutas cambiaron; no reanudar otra semántica')
        args.stage_steps=config['stage_steps']; args.block_steps=config['block_steps']
    else:
        output=args.output.resolve() if args.output else HERE/'runs'/f'r3_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}'
        if output.exists(): raise FileExistsError('No se sobreescriben carpetas: '+str(output))
        if not args.check:
            output.mkdir(parents=True,exist_ok=False)
            dump(output/'config.json',dict(source=str(SOURCE),normalizer=str(NORM),hashes=provenance(),
                seed=args.seed,stage_steps=args.stage_steps,block_steps=args.block_steps,
                algorithm='SAC',BC_updates=0,DAgger_updates=0,reward_scale=REWARD_SCALE,
                phase_budgets=[1000,4500,1000],success='physical stable placement; no home',
                source_history='R1 transferred arm may include prior imitation and RL',
                actor_count=1,action_dim=6,observation_dim=130))
    lock=None
    if not args.check:
        lock=(output/'.run.lock').open('a')
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise RuntimeError('Ya hay una corrida usando esta carpeta')
    wrapper=CurriculumEnv(args.seed,None if args.check else output/'episodes.jsonl')
    vec=DummyVecEnv([lambda:wrapper]); stop={'requested':False}; old_handlers={}
    state=dict(stage=0,stage_steps=0,completed=False)
    def announce(event, **details):
        row=dict(event=event,stage=STAGES[state['stage']],**details)
        print(event+' '+json.dumps(details,allow_nan=False),flush=True)
        if not args.check:
            dump(output/'status.json',row)
            with (output/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    try:
        if args.resume:
            target=Path(json.loads((output/'latest.json').read_text())['checkpoint'])
            state=json.loads((target/'state.json').read_text())
            if state['completed']: raise ValueError('Esta corrida ya terminó')
            if sha(target/'model.zip')!=state['model_sha256'] or sha(target/'replay.pkl')!=state['replay_sha256']:
                raise ValueError('Checkpoint incompleto o modificado')
            model=CurriculumSAC.load(target/'model.zip',env=vec,device=args.device,force_reset=True)
            model.load_replay_buffer(target/'replay.pkl')
            with (target/'runtime.pkl').open('rb') as f: rt=pickle.load(f)
            random.setstate(rt['python']); np.random.set_state(rt['numpy']); torch.set_rng_state(rt['torch'])
            if args.device=='cuda' and rt['cuda']: torch.cuda.set_rng_state_all(rt['cuda'])
            wrapper.set_stage(state['stage']); wrapper.rng.bit_generator.state=rt['sampler']
            wrapper.queue=rt['queue']; wrapper.episodes=rt['episodes']
            vec._seeds=[None] # Preserve restored case sampler; next rollout starts physical P0.
        else:
            model=build(vec,args.device,args.seed,512 if args.check else args.buffer_size)
        model.set_logger(configure(None if args.check else str(output/'sac_logs'),['csv'] if not args.check else []))
        if args.check:
            obs=vec.reset(); action=model.predict(obs,deterministic=False)[0]
            obs,reward,done,info=vec.step(action)
            if action.shape!=(1,6) or not np.isfinite(obs).all(): raise AssertionError('Contrato inválido')
            announce('READY_NOT_TRAINED',device=str(model.device),source=str(SOURCE),
                     action_dim=6,observation_dim=130,actor_count=1)
            return
        if not args.resume:
            model.save(output/'initial.zip'); save_resume(model,wrapper,output,state)
        initial=CurriculumSAC.load(output/'initial.zip',device='cpu')
        for signum in (signal.SIGINT,signal.SIGTERM):
            old_handlers[signum]=signal.getsignal(signum)
            signal.signal(signum,lambda *_:stop.update(requested=True))
        announce('START',output=str(output),steps=model.num_timesteps,device=str(model.device))
        while state['stage']<4 and not stop['requested']:
            stage=state['stage']; model.actor.curriculum_stage.fill_(stage)
            if wrapper.stage!=stage:
                wrapper.set_stage(stage); model._last_obs=None # New stage starts at P0, never a synthetic handoff.
            if not state.get('pending_evaluation'):
                remaining=args.stage_steps[stage]-state['stage_steps']
                if remaining<=0:
                    announce('CURRICULUM_BLOCKED',reason='No superó la puerta de validación; no se promueve automáticamente')
                    break
                steps=min(args.block_steps,remaining); before=model.num_timesteps
                model.learn(steps,callback=Progress(stop,announce,steps),reset_num_timesteps=False)
                state['stage_steps']+=model.num_timesteps-before
                if not stop['requested']:
                    state['pending_evaluation']=f's{stage}_{model.num_timesteps}'
                save_resume(model,wrapper,output,state)
                if stop['requested']: break
            tag=state['pending_evaluation']+f'_{time.time_ns()}'
            report=evaluate(model,initial,cases('nominal' if stage==0 else 'robust'),output/'evaluations'/tag)
            announce('BLOCK_EVALUATED',steps=model.num_timesteps,assessment=report)
            if stop['requested']: break
            if report['curriculum_eligible']:
                model.save(output/f'passed_stage_{stage}.zip')
                if stage==3:
                    model.save(output/'candidate.zip')
                    holdout_dir=output/f'holdout_{time.time_ns()}'
                    holdout=evaluate(model,initial,cases(reps=10,seed=410_000_000),
                                     holdout_dir,baseline=True)
                    state['completed']=True
                    dump(output/'result.json',dict(candidate=str(output/'candidate.zip'),
                        holdout=str(holdout_dir),
                        assessment=holdout,recommended=holdout['recommended'],
                        note='Candidato evaluado; no reemplaza automáticamente ningún modelo histórico'))
                    announce('FINISHED',assessment=holdout)
                    save_resume(model,wrapper,output,state)
                    break
                state.update(stage=stage+1,stage_steps=0)
                state.pop('pending_evaluation',None)
                model.actor.curriculum_stage.fill_(stage+1)
                wrapper.set_stage(stage+1); model._last_obs=None
                save_resume(model,wrapper,output,state)
                announce('STAGE_PROMOTED')
            else:
                state.pop('pending_evaluation',None)
                save_resume(model,wrapper,output,state)
        if stop['requested'] and not state['completed']:
            save_resume(model,wrapper,output,state)
            announce('PAUSED_SAVED',resume=f'bash {HERE}/run_curriculum.sh --resume {output}')
    except Exception:
        if not args.check:
            dump(output/'error.json',dict(traceback=traceback.format_exc(),state=state))
        raise
    finally:
        vec.close()
        for signum,handler in old_handlers.items(): signal.signal(signum,handler)
        if lock is not None: lock.close()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path); p.add_argument('--resume',type=Path)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--seed',type=int,default=23092026)
    p.add_argument('--stage-steps',type=int,nargs=4,default=[150000,200000,150000,200000])
    p.add_argument('--block-steps',type=int,default=50000)
    p.add_argument('--buffer-size',type=int,default=200000)
    p.add_argument('--check',action='store_true')
    return p


if __name__=='__main__':
    run(parser().parse_args())
