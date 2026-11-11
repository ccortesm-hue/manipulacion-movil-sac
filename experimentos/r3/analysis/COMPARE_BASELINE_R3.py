"""Baseline clásico vs VX_R3 activo: 200 casos físicos pareados, sin entrenamiento."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
os.environ.setdefault('MUJOCO_GL','egl')
import argparse
import concurrent.futures
import contextlib
import csv
import hashlib
import json
import math
import multiprocessing
import time
import traceback
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from EVALUATE_R3 import (ROOT, CurriculumSAC, ZennoVXR3, np, torch, mujoco,
    cases, snapshot, baseline_module, canonicalize_and_audit, physical_hash,
    BaselineDriver, Recorder, dump, METRICS)

def evaluate_pair(model, panel, output):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    actors={'current':snapshot(model)}
    mm,ik=baseline_module()
    rows=[]
    methods=('baseline','on')
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
                lookahead_trace=[]; handoff_diagnostics=None
                for _ in range(20000):
                    phase=driver.phase if driver else int(env.phase_state)
                    counts[phase]+=1
                    if driver:
                        driver.control()
                        for _ in range(5): mujoco.mj_step(mm.model,mm.data); recorder.substep()
                        stop=recorder.end_interval(phase,phase==1)
                    else:
                        actor=actors['current']
                        action=actor.predict(obs,deterministic=True)[0]
                        # Residual activo: no se sobrescribe ninguna acción.
                        obs,_,done,_,info=env.step(action)
                        stop=info['task_stop']
                        if phase==0 and env.phase_state==1: p1_hash=physical_hash(env)
                        if phase==1: handoff_diagnostics=info.get('p1_diagnostics')
                        lookahead_trace.append((float(action[5]),float(env._lh_applied),bool(info['lookahead_active'])))
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
                row['cte_signed_mean_m']=float(np.mean(traces['cte_signed_m'][tracking])) if tracking.any() else None
                row['tracking_samples']=int(tracking.sum())
                row['p1_handoff_diagnostics']=handoff_diagnostics
                trace_dir=output/'traces'/method; trace_dir.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(trace_dir/f'{case["case_id"]}.npz',**traces,lookahead=np.asarray(lookahead_trace))
                row.update(case,method=method,success_place=stop=='stable_place',termination_reason=stop,
                    phase_steps=counts,base_terminal_error_m=base_error,
                    carry_rms_m=float(np.sqrt(np.mean(np.square(carry)))) if carry else None,
                    lookahead_delta_rms_m=float(np.sqrt(np.mean(np.square(delta)))) if delta else 0.,
                    initial_hash=signatures[-1],p1_hash=p1_hash)
                rows.append(row); handoffs[method]=p1_hash
                dump(output/'episodes'/method/f'{case["case_id"]}.json',row)
            finally: env.close()
        if len(set(signatures))!=1:
            raise AssertionError('Estados físicos iniciales baseline/R3 distintos')
        if index+1 in (max(1,len(panel)//2),len(panel)):
            print('EVAL_PROGRESS',json.dumps({'cases':index+1,'total':len(panel)}),flush=True)
    report={'n_cases':len(panel)}
    dump(output/'summary.json',dict(rows=rows,assessment=report,complete=True,
        protocol=dict(agent_budgets=[1000,4500,1000],baseline_global_cap_s=200,
        placement='support + release + height + speed stable 0.2s; no home',
        inference='CPU for all snapshots',energy='absolute work, all 12 actuators',
        comparison='classical baseline versus R3 with learned lookahead active')))
    return rows


WORKER_MODEL=None


def init_worker(model_path):
    global WORKER_MODEL
    torch.set_num_threads(1)
    WORKER_MODEL=CurriculumSAC.load(model_path,device='cpu')


def worker(case, root):
    root=Path(root)
    with (root/'logs'/f'{case["case_id"]}.log').open('w') as log:
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            return evaluate_pair(WORKER_MODEL,[case],root/'cases'/case['case_id'])


def aggregate(rows):
    groups={}
    for row in rows:
        pair=groups.setdefault(row['case_id'],{})
        if row['method'] in pair: raise ValueError('Duplicado')
        pair[row['method']]=row
    if any(set(g)!=set(('baseline','on')) for g in groups.values()):
        raise ValueError('Comparación incompleta')
    output={}
    for label,mid in [('global',None)]+[(f'map_{m}',m) for m in (1,2,3,4)]:
        sub=[g for g in groups.values() if mid is None or g['on']['map_id']==mid]
        if not sub: continue
        both=[g for g in sub if all(g[m]['success_place'] for m in ('baseline','on'))]
        counts={m:sum(g[m]['success_place'] for g in sub) for m in ('baseline','on')}
        b_only=sum(g['baseline']['success_place'] and not g['on']['success_place'] for g in sub)
        r_only=sum(g['on']['success_place'] and not g['baseline']['success_place'] for g in sub)
        metrics={}
        for key in METRICS:
            valid=[g for g in both if all(g[m].get(key) is not None and np.isfinite(g[m][key]) for m in ('baseline','on'))]
            if not valid: continue
            b,r=[float(np.mean([g[m][key] for g in valid])) for m in ('baseline','on')]
            metrics[key]=dict(baseline=b,r3_active=r,n_pairs=len(valid),
                change_percent=None if abs(b)<1e-12 else 100*(r/b-1))
        output[label]=dict(n_cases=len(sub),success_counts=counts,
            success_rates={m:counts[m]/len(sub) for m in counts},
            both_success=len(both),baseline_only=b_only,r3_only=r_only,
            both_failed=len(sub)-len(both)-b_only-r_only,metrics_paired_success=metrics,
            failure_reasons={m:{reason:sum(g[m]['termination_reason']==reason for g in sub if not g[m]['success_place'])
                for reason in sorted({g[m]['termination_reason'] for g in sub if not g[m]['success_place']})}
                for m in ('baseline','on')})
    return output


def write_reports(root, rows):
    stats=aggregate(rows)
    dump(root/'summary.json',dict(complete=True,n_cases=len(rows)//2,groups=stats,rows=rows))
    fields=['scope','metric','baseline','r3_active','change_percent','n_pairs']
    with (root/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for scope,g in stats.items():
            writer.writerow(dict(scope=scope,metric='success_percent',
                baseline=100*g['success_rates']['baseline'],r3_active=100*g['success_rates']['on'],
                change_percent='',n_pairs=g['n_cases']))
            for metric,value in g['metrics_paired_success'].items():
                writer.writerow(dict(scope=scope,metric=metric,**value))
    with (root/'failures.csv').open('w',newline='') as f:
        fields=['method','map_id','mass_kg','case_seed','termination_reason','total_time_s']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        writer.writerows(r for r in rows if not r['success_place'])
    text=['# Baseline clásico frente a VX_R3 activo','',
        'Checkpoint de 300 000 pasos; no se modificó ni reentrenó el controlador.',
        'Semillas reservadas, masas 0.5–1.5 kg, cuatro mapas, ruido inicial XY ±0.01 m y yaw ±0.1 rad.',
        'Mismos estados físicos iniciales auditados; colocación estable durante 0.2 s, sin home.',
        'R3: límites por fase 1000/4500/1000 pasos. Baseline: máximo global de 200 s.',
        'Energía: trabajo mecánico absoluto de los 12 actuadores, no consumo eléctrico.',
        'CTE RMS, CTE firmado y yaw usan la misma ventana interior de ruta (0.30 m a longitud−0.55 m).',
        'Tiempo y energía se cortan al éxito físico o al fallo; las tablas de métricas usan SOLO éxitos comunes.',
        'Los porcentajes de cambio negativos indican valores menores, no necesariamente superioridad global.','']
    labels={'total_time_s':('Tiempo [s]',1),'total_energy_J':('Trabajo [J]',1),
        'cte_rms_m':('CTE RMS [mm]',1000),'yaw_rms_rad':('Yaw RMS [°]',180/math.pi),
        'cube_goal_dist_xy_m':('Error colocación [mm]',1000),'carry_rms_m':('Error transporte [mm]',1000),
        'base_terminal_error_m':('Error posición final base [mm]',1000)}
    for scope,g in stats.items():
        text += [f'## {scope}','',f'Éxitos: baseline {g["success_counts"]["baseline"]}/{g["n_cases"]}; R3 {g["success_counts"]["on"]}/{g["n_cases"]}.',
            f'Éxitos comunes: {g["both_success"]}; solo baseline: {g["baseline_only"]}; solo R3: {g["r3_only"]}; ambos fallan: {g["both_failed"]}.','',
            '| Métrica | Baseline | R3 activo | Cambio |','|---|---:|---:|---:|']
        for k,v in g['metrics_paired_success'].items():
            label,scale=labels[k]
            change='—' if v['change_percent'] is None else f'{v["change_percent"]:+.2f}%'
            text.append(f'| {label} | {v["baseline"]*scale:.3f} | {v["r3_active"]*scale:.3f} | {change} |')
        text += ['']
    text += ['## Lectura del resultado','',
        'Priorizar primero éxito y fallos pareados, después contrastar precisión, tiempo y energía de los mismos episodios.',
        'No interpretar una mejora entre éxitos como compensación automática de episodios perdidos.',
        'El fallo de handoff identificado en R3 NO fue corregido para esta comparación.',
        'Un agotamiento de tiempo describe la terminación, no demuestra por sí solo que ampliar el horizonte resuelva la causa.','']
    (root/'comparison.md').write_text('\n'.join(text)+'\n')
    return stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--reps',type=int,default=10)
    p.add_argument('--seed',type=int,default=410_000_000)
    p.add_argument('--limit-cases',type=int)
    a=p.parse_args()
    if a.workers<1 or a.reps<1: raise ValueError('Workers/reps deben ser positivos')
    panel=cases(reps=a.reps,seed=a.seed)
    if a.limit_cases is not None: panel=panel[:a.limit_cases]
    if not panel: raise ValueError('Panel vacío')
    root=a.output.resolve(); root.mkdir(parents=True,exist_ok=False); (root/'logs').mkdir()
    def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    dependencies=[Path(__file__),Path(__file__).resolve().parents[1]/'EVALUATE_R3.py',
        Path(__file__).resolve().parents[1]/'VX_R3.py',ROOT/'VX/MM_DATA_VXR1.py',ROOT/'VX/MM_DATA_VXR2.py',
        ROOT/'VX/baselines.py',ROOT/'vx_place_protocol.py',ROOT/'vx_place_initials.py',
        ROOT/'extract_vx_place_robust.py']+list((ROOT/'RUTAS_SAC').glob('*.json'))
    dump(root/'config.json',dict(checkpoint=str(a.model.resolve()),model_sha256=sha(a.model),
        sources={str(path):sha(path) for path in dependencies},cases=panel,workers=a.workers,
        inference='CPU, one torch thread per worker',methods=['baseline','on'],
        no_training=True,no_controller_changes=True))
    started=time.time(); all_rows=[]
    dump(root/'status.json',dict(state='RUNNING',completed_cases=0,total=len(panel)))
    print('START',json.dumps(dict(output=str(root),cases=len(panel),episodes=2*len(panel))),flush=True)
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,
                mp_context=multiprocessing.get_context('spawn'),
                initializer=init_worker,initargs=(str(a.model.resolve()),)) as pool:
            pending=[pool.submit(worker,case,str(root)) for case in panel]
            for future in concurrent.futures.as_completed(pending):
                all_rows.extend(future.result())
                n=len(all_rows)//2
                dump(root/'status.json',dict(state='RUNNING',completed_cases=n,total=len(panel),
                    elapsed_s=time.time()-started))
                if n in (1,max(1,len(panel)//2),len(panel)):
                    print('PROGRESS',json.dumps(dict(cases=n,total=len(panel),elapsed_s=time.time()-started)),flush=True)
        all_rows.sort(key=lambda r:(r['map_id'],r['mass_kg'],r['rep'],r['method']))
        if len(all_rows)!=2*len(panel): raise ValueError('Faltan episodios')
        stats=write_reports(root,all_rows)
        dump(root/'status.json',dict(state='COMPLETE',completed_cases=len(panel),total=len(panel),
            elapsed_s=time.time()-started))
        print('COMPLETE',json.dumps(stats['global']),flush=True)
    except BaseException:
        dump(root/'error.json',dict(traceback=traceback.format_exc()))
        dump(root/'status.json',dict(state='FAILED',completed_cases=len(all_rows)//2,total=len(panel)))
        raise


if __name__=='__main__':
    main()
