#!/usr/bin/env python3
"""Experimento zenno_vx_r1: curriculum solapado A -> B -> C, un actor SAC."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import sys
import traceback

os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('MPLCONFIGDIR','/tmp/vx_r1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure as configure_logger
from stable_baselines3.common.utils import get_schedule_fn
from sac_vx import VXSAC
from zenno_vx_r1 import ZennoVXR1
from evaluate_zenno_vx_r1 import CASES,make_env,evaluate,dump,save
from vx_r1_curriculum import (collect,merge,load_physical_teacher,update,freeze_actor,
                             promote,competent)
from finetune_vx_recovery import compact,holdout_comparison

ROOT=Path(__file__).resolve().parent
SOURCE_REVISION='VX_R1_ROUTE_REWARD_124D_7D_DELTA008_UNIFORM_COSTS'
PROBE_CASES=[(m,.5) for m in (1,2,3,4)]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight(args):
    source=json.loads((args.data_source/'config.json').read_text())
    initial=json.loads((args.init_run/'config.json').read_text())
    if source['revision']!=SOURCE_REVISION or initial['revision']!=SOURCE_REVISION:
        raise ValueError('Checkpoint de partida incompatible con VX original.')
    for name in ('zenno_vx.py','analytic_teacher_vx.py','sac_vx.py','demo_regularized_sac_v9.py'):
        if sha(ROOT/name)!=source['hashes'][name]:raise ValueError(f'Fuente cambió: {name}')
    # Check the independent clone is still exactly the same physical contract.
    expected=(ROOT/'zenno_vx.py').read_text().replace('ZennoVX','ZennoVXR1').replace(
        SOURCE_REVISION,ZennoVXR1.REVISION).splitlines()[2:]
    if expected!=(ROOT/'zenno_vx_r1.py').read_text().splitlines()[2:]:
        raise ValueError('R1 ya no es un clon físico exacto. Revisar protocolo antes de entrenar.')
    for name,digest in source['route_hashes'].items():
        if sha(ROOT/'RUTAS_SAC'/name)!=digest:raise ValueError(f'Ruta modificada: {name}')
    norms=[]
    for run in (args.data_source,args.init_run):
        with (run/'best/vecnormalize.pkl').open('rb') as f:norms.append(pickle.load(f))
    for attr in ('mean','var','count'):
        if not np.array_equal(getattr(norms[0].obs_rms,attr),getattr(norms[1].obs_rms,attr)):
            raise ValueError(f'Normalizadores incompatibles: {attr}')
    for attr in ('norm_obs','clip_obs','epsilon','norm_obs_keys'):
        if getattr(norms[0],attr,None)!=getattr(norms[1],attr,None):
            raise ValueError(f'Transformación de observación diferente: {attr}')
    for path in (args.data_source/'teacher.npz',args.init_run/'best/model.zip'):
        if not path.is_file():raise FileNotFoundError(path)
    return source


def specs(seed,noise=1.,reps=1):
    return [(m,w,seed+i*100+r,noise) for i,(m,w) in enumerate(CASES) for r in range(reps)]


def training_specs(seed,panels,noise):
    result=specs(seed,noise)
    # Five known P0 seed failures plus varied map1 examples remain in scope.
    result.extend((1,w,seed+5000+i,1.) for i,w in enumerate((.5,.75,1.,1.25,1.5)))
    # Bound targeted overhead while retaining all map/mass coverage above.
    failed=[]
    for panel in panels.values():
        for r in panel['rows']:
            if not r['success']:
                failed.append((r['map'],r['mass'],r['seed'],float(r['profile']=='robust')))
    unique=list(dict.fromkeys(failed))
    rng=np.random.default_rng(seed)
    if len(unique)>15:unique=[unique[i] for i in rng.choice(len(unique),15,replace=False)]
    return list(dict.fromkeys(result+unique))


def validate(model,norm,out,tag,announce):
    def progress(done,row):
        if done%20==0:announce('EVAL',dict(tag=tag,completed_in_panel=done))
    return dict(legacy=evaluate(model,norm,out/f'{tag}_legacy40.json',seed=1910000,progress=progress),
        development=evaluate(model,norm,out/f'{tag}_dev60.json',seed=5930000,reps=2,progress=progress))


def sac_pilot(model,norm,data,args,best,announce):
    """SAC sólo tras gate; recalibra Q, replay físico y BC brazo sin forzar residual cero."""
    out=args.output
    model.learning_rate=3e-6
    model.lr_schedule=get_schedule_fn(3e-6)
    model.gradient_steps=1
    model.learning_starts=0
    model.set_logger(configure_logger(str(out/'sac_logs'),['csv']))
    model.replay_buffer.reset()
    valid=np.flatnonzero(data['replay_valid'])
    rng=np.random.default_rng(args.seed+444)
    order=[]
    for phase in (0,1,2):
        rows=valid[data['phase_ids'][valid]==phase]
        if not len(rows):raise ValueError('Replay físico sin cobertura de todas las fases.')
        order.extend(rng.choice(rows,min(len(rows),50000),replace=False))
    for i in rng.permutation(order):
        # Teacher LABELS are never passed as actions to Q.
        model.replay_buffer.add(data['observations'][i:i+1],data['next_observations'][i:i+1],
            data['executed_actions'][i:i+1],data['rewards'][i:i+1],data['dones'][i:i+1],
            [{'TimeLimit.truncated':bool(data['timeouts'][i])}])
    weights=np.ones((3,7),np.float32);weights[1,5:]=0.
    model.set_demo_regularization(data,5.,512,args.seed,normalize_q_actor_loss=True,
        q_objective_scale=.15,actor_warmup_updates=model._n_updates+5000,
        action_weights_by_phase=weights)
    announce('CRITIC_WARMUP',dict(updates=5000,actor_frozen=True))
    model.train(gradient_steps=5000,batch_size=512)
    class Checkpoint(BaseCallback):
        def __init__(self):
            super().__init__();self.best=best;self.last=0
        def _on_step(self):
            if self.num_timesteps-self.last<10000:return True
            self.last=self.num_timesteps
            metrics=validate(self.model,norm,out,f'sac_{self.last}',announce)
            accepted=promote(metrics,self.best)
            save(self.model,norm,out/f'sac_candidate_{self.last}')
            if accepted:
                self.best=metrics;save(self.model,norm,out/'best');dump(out/'best_metrics.json',metrics)
            announce('SAC_EVAL',dict(steps=self.last,accepted=accepted,
                                   candidate={k:compact(v) for k,v in metrics.items()}))
            return all(metrics[k]['strict']>=self.best[k]['strict']-.1 for k in self.best)
    callback=Checkpoint()
    model.learn(args.sac_steps,callback=callback,reset_num_timesteps=True)
    final=validate(model,norm,out,'sac_last',announce)
    save(model,norm,out/'sac_last')
    if promote(final,callback.best):
        callback.best=final;save(model,norm,out/'best');dump(out/'best_metrics.json',final)
    return callback.best


def run(args,announce):
    source=preflight(args)
    if not torch.cuda.is_available():raise RuntimeError('CUDA requerida; no fallback CPU silencioso.')
    torch.set_num_threads(2);torch.manual_seed(args.seed);np.random.seed(args.seed)
    out=args.output
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(experiment='zenno_vx_r1',revision=ZennoVXR1.REVISION,single_actor=True,
        device=torch.cuda.get_device_name(0),source_revision=SOURCE_REVISION,
        source_model_sha256=sha(args.init_run/'best/model.zip'),
        normalizer_sha256=sha(args.init_run/'best/vecnormalize.pkl'),
        hashes={n:sha(ROOT/n) for n in ('zenno_vx_r1.py','analytic_teacher_vx_r1.py',
             'train_zenno_vx_r1.py','evaluate_zenno_vx_r1.py','vx_r1_curriculum.py','sac_vx.py')},
        route_hashes=source['route_hashes'],final_seed=args.seed+30000000,
        stage_steps=dict(A=args.a_steps,B=args.b_steps,C=args.c_steps))
    dump(out/'config.json',config)
    norm=VecNormalize.load(args.init_run/'best/vecnormalize.pkl',DummyVecEnv([make_env]))
    norm.training=False;norm.norm_reward=False
    try:
        model=VXSAC.load(args.init_run/'best/model.zip',env=norm,device='cuda')
        save(model,norm,out/'initial');save(model,norm,out/'best')
        parts=[load_physical_teacher(args.data_source/'teacher.npz')]
        announce('TEACHER_AUDIT',dict(episodes=40,noise='20 nominal + 20 robust'))
        audit=collect(None,norm,out/'teacher_audit.npz',
            specs(args.seed,0.)+specs(args.seed+10000,1.),'teacher',1.,1,announce)
        report=json.loads((out/'teacher_audit.json').read_text())
        if report['success_rate']<.95:
            dump(out/'result.json',dict(status='teacher_gate_failed',success=report['success_rate'],
                training_started=False,target_met=False))
            announce('STOP',dict(reason='teacher_gate_failed',success=report['success_rate']))
            return
        parts.append(audit)
        best=validate(model,norm,out,'initial',announce)
        dump(out/'best_metrics.json',best)
        announce('BASELINE',{k:compact(v) for k,v in best.items()})
        # Paired autonomous states for conservative retention, not another teacher actor.
        parts.append(collect(model,norm,out/'initial_student.npz',specs(args.seed+20000),
                             'C',0.,2,announce))
        history=[]
        attempted=accepted_updates=0
        for cycle in range(1,args.cycles+1):
            model=VXSAC.load(out/'best/model.zip',env=norm,device='cuda')
            anchor=freeze_actor(model)
            model.actor.optimizer=torch.optim.Adam(model.actor.parameters(),lr=args.lr)
            robust=min(v['rates']['robust'] for v in best.values())
            beta=.7 if robust<.8 else .35 if robust<.95 else 0.
            noise=.5 if robust<.8 else 1.
            cycle_start=attempted
            for stage,steps in (('A',args.a_steps),('B',args.b_steps),('C',args.c_steps)):
                tag=f'cycle{cycle}_{stage}'
                announce('STAGE',dict(cycle=cycle,stage=stage,beta=0. if stage=='C' else beta,
                    provisional=True,steps=steps))
                extra=collect(model,norm,out/f'{tag}.npz',training_specs(
                    args.seed+cycle*100000+ord(stage)*1000,best,noise),stage,beta,
                    10+cycle*3+'ABC'.index(stage),announce)
                parts.append(extra)
                data=merge(parts)
                losses=update(model,norm,data,stage,steps,args.lr,args.seed+cycle*10+ord(stage),
                              anchor,announce)
                attempted+=steps
                save(model,norm,out/f'candidate_{tag}')
                # Eight-case diagnostic probe; never rejects A or B before integration.
                probe=evaluate(model,norm,out/f'probe_{tag}.json',seed=7930000,cases=PROBE_CASES)
                history.append(dict(cycle=cycle,stage=stage,updates=steps,losses=losses,
                                    probe=compact(probe),provisional=True))
                dump(out/'history.json',history)
                announce('STAGE_DONE',dict(tag=tag,probe=compact(probe),provisional=True))
                del data
            del anchor
            candidate=validate(model,norm,out,f'cycle{cycle}_integrated',announce)
            accepted=promote(candidate,best)
            if accepted:
                best=candidate;save(model,norm,out/'best');dump(out/'best_metrics.json',best)
                accepted_updates+=attempted-cycle_start
            history.append(dict(cycle=cycle,stage='INTEGRATED',accepted=accepted,
                candidate={k:compact(v) for k,v in candidate.items()},best={k:compact(v) for k,v in best.items()}))
            dump(out/'history.json',history)
            announce('CYCLE',history[-1])
            # A perfect nominal panel and near-100 robust panel may finish BC early.
            if competent(best,args.target) and all(v['rates']['nominal']==1. for v in best.values()):break
        data=merge(parts)
        model=VXSAC.load(out/'best/model.zip',env=norm,device='cuda')
        online=bool(args.sac_steps and competent(best,.95))
        dump(out/'pre_sac_gate.json',dict(passed=competent(best,.95),started=online,
             metrics={k:compact(v) for k,v in best.items()}))
        if online:best=sac_pilot(model,norm,data,args,best,announce)
        del data,parts
        model=VXSAC.load(out/'best/model.zip',env=norm,device='cuda')
        announce('FINAL_TEST',dict(episodes_per_model=20+20*args.final_reps,untouched_seeds=True))
        final=evaluate(model,norm,out/'holdout_best.json',reps=args.final_reps,seed=args.seed+30000000)
        initial=VXSAC.load(out/'initial/model.zip',env=norm,device='cuda')
        before=evaluate(initial,norm,out/'holdout_initial.json',reps=args.final_reps,seed=args.seed+30000000)
        result=dict(status='complete',experiment='zenno_vx_r1',single_actor=True,
            attempted_bc_updates=attempted,accepted_bc_updates=accepted_updates,sac_online_started=online,
            validation={k:compact(v) for k,v in best.items()},holdout=holdout_comparison(before,final),
            target_met=final['rates']['nominal']==1. and final['rates']['robust']>=args.target,
            note='Objetivo empírico finito; no garantiza éxito universal.')
        dump(out/'result.json',result);announce('FINAL',result)
    finally:norm.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-source',type=Path,default=ROOT/'models/vx_delta008_uniform_costs_run1')
    p.add_argument('--init-run',type=Path,default=ROOT/'models/vx_p2_retention_run1')
    p.add_argument('--seed',type=int,default=1100000)
    p.add_argument('--cycles',type=int,default=3)
    p.add_argument('--a-steps',type=int,default=500)
    p.add_argument('--b-steps',type=int,default=1000)
    p.add_argument('--c-steps',type=int,default=1000)
    p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--sac-steps',type=int,default=50000)
    p.add_argument('--final-reps',type=int,default=10)
    p.add_argument('--target',type=float,default=.98)
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    if min(args.cycles,args.a_steps,args.b_steps,args.c_steps,args.final_reps)<=0:p.error('Presupuestos positivos.')
    if not (0<args.lr<.001 and .9<=args.target<=1 and 0<=args.sac_steps and args.final_reps<=100):
        p.error('LR/target/presupuestos fuera del rango del experimento.')
    # Keep current protocol seeds disjoint from both old development panels and test.
    if args.seed!=1100000 or args.cycles>10:p.error('Esta revisión fija seed=1100000 y máximo 10 ciclos.')
    for key in ('output','data_source','init_run'):setattr(args,key,getattr(args,key).resolve())
    if args.check_only:
        preflight(args);print('CHECK_OK zenno_vx_r1: clon, checkpoints, normalizador y rutas compatibles.');return
    # One R1 launch at a time; never kill or resume any historical experiment.
    with (ROOT/'models/.zenno_vx_r1.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Ya hay una corrida zenno_vx_r1 activa.')
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'pid.txt').write_text(str(os.getpid())+'\n')
        console=sys.stdout
        def announce(stage,data):
            message=stage+' '+json.dumps(data,ensure_ascii=False)
            print(message,file=console,flush=True)
            with (args.output/'milestones.log').open('a') as f:f.write(message+'\n')
            dump(args.output/'status.json',dict(stage=stage,details=data,pid=os.getpid()))
        announce('START',dict(experiment='zenno_vx_r1',output=str(args.output),
            max_bc_updates=args.cycles*(args.a_steps+args.b_steps+args.c_steps)))
        with (args.output/'pipeline.log').open('w',buffering=1) as log:
            with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                try:run(args,announce)
                except (KeyboardInterrupt,InterruptedError) as exc:
                    dump(args.output/'interrupted.json',dict(type=type(exc).__name__,message=str(exc)))
                    announce('INTERRUPTED',dict(message=str(exc),completed_episode_shards_preserved=True))
                    raise SystemExit(130)
                except BaseException as exc:
                    traceback.print_exc(file=log)
                    dump(args.output/'error.json',dict(type=type(exc).__name__,message=str(exc)))
                    announce('ERROR',dict(message=str(exc)));raise


if __name__=='__main__':
    def stop_requested(signum,frame):
        raise InterruptedError(f'Interrupción solicitada: {signal.Signals(signum).name}')
    signal.signal(signal.SIGTERM,stop_requested)
    main()
