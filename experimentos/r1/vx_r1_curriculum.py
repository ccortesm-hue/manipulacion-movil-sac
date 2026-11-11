"""Datos físicos y BC acumulativo con ventanas de transición para VX-R1."""
import copy
from pathlib import Path
import numpy as np
import torch
from analytic_teacher_vx_r1 import VXR1AnalyticTeacher
from evaluate_zenno_vx_r1 import make_env, configure, predict, dump, handoff_state
from vx_p2_diagnostics import p2_state

FIELDS = ('observations', 'next_observations', 'actions', 'executed_actions',
          'rewards', 'dones', 'timeouts', 'phase_ids', 'teacher_stages', 'episode_ids',
          'dagger_round', 'window01', 'window12', 'segment_a', 'segment_b',
          'replay_valid', 'retain_mask')


def windows(phases, before=100, after=200):
    """Post-hoc, dentro de UN episodio. Nunca inventa transiciones que no ocurrieron."""
    phases = np.asarray(phases)
    n = len(phases)
    w01, w12 = np.zeros(n, bool), np.zeros(n, bool)
    a, b = phases == 0, phases == 2
    for i in range(1, n):
        if phases[i-1] == 0 and phases[i] == 1:
            w01[max(0,i-before):min(n,i+after)] = True
            a[i:min(n,i+after)] = True
        if phases[i-1] == 1 and phases[i] == 2:
            w12[max(0,i-before):min(n,i+after)] = True
            b[max(0,i-after):i] = True
    return w01, w12, a, b


def cast(rows):
    floats = {'observations','next_observations','actions','executed_actions','rewards'}
    ints = {'phase_ids','episode_ids','dagger_round'}
    return {k: np.asarray(rows[k], dtype=np.float32 if k in floats else
                         np.int32 if k in ints else str if k == 'teacher_stages' else bool)
            for k in FIELDS}


def merge(parts):
    return {k:np.concatenate([p[k] for p in parts]) for k in FIELDS}


def load_physical_teacher(path):
    """Solo el profesor original verificado; no convierte etiquetas DAgger en replay."""
    with np.load(path, allow_pickle=False) as z:
        data = {k:z[k].copy() for k in z.files}
    if not np.all(data['replay_valid']):
        raise ValueError('La semilla inicial debe contener demostraciones físicas docentes.')
    if not np.array_equal(data['actions'], data['executed_actions']):
        raise ValueError('El dataset fuente no es profesor puro.')
    n = len(data['phase_ids'])
    for name in ('window01','window12','segment_a','segment_b','retain_mask'):
        data[name] = np.zeros(n, bool)
    for ep in np.unique(data['episode_ids']):
        idx = np.flatnonzero(data['episode_ids'] == ep)
        for name, value in zip(('window01','window12','segment_a','segment_b'), windows(data['phase_ids'][idx])):
            data[name][idx] = value
    # Source teacher states are not automatically declared successes of the student.
    data['dagger_round'] = np.zeros(n, np.int32)
    return cast(data)


def assistance(mode, phase, env, beta):
    if mode == 'teacher': return 1.
    if mode == 'C': return 0.  # DAgger puro: toda acción ejecutada pertenece al alumno.
    if mode == 'A': return beta
    if mode != 'B': raise ValueError(mode)
    if phase == 2: return beta
    if phase == 1:
        _, s, length, distance, _, valid = env._route_state()
        if valid and (length-s <= .4 or distance <= .4): return beta
    return 0.


def collect(model, norm, path, specs, mode, beta, round_id, announce=None):
    """Episodios completos; profesor continuo etiqueta cada estado, incluso en beta=0.

    A: mezcla docente en todo el recorrido para enseñar pick + arranque estable.
    B: prefijo alumno; asistencia solo cerca del final P1 y en P2.
    C: alumno autónomo. No rescata fallos ni carga estados intermedios.
    Fuera de teacher, la base P1 siempre ejecuta los residuales del alumno.
    """
    if mode == 'C': beta = 0.
    if mode == 'teacher': beta = 1.
    path = Path(path)
    shard_dir = path.with_suffix('.episodes')
    if path.exists() or shard_dir.exists():
        raise FileExistsError(f'Colección existente o parcial: {path}')
    shard_dir.mkdir(parents=True)
    env = make_env()
    episodes, parts = [], []
    try:
        for ep, (map_id, mass, seed, noise) in enumerate(specs):
            configure(env, (map_id, mass), noise > 0)
            env.CUBE_XY_NOISE_RANGE *= noise
            env.CUBE_YAW_NOISE_RANGE_RAD *= noise
            obs, _ = env.reset(seed=seed)
            teacher = VXR1AnalyticTeacher(env)
            rows = {k:[] for k in FIELDS}
            handoffs, trace, agreements = [], [], []
            counts = [0,0,0]
            for step in range(5000):
                phase = int(env.phase_state)
                label = teacher.action()  # One uninterrupted oracle FSM for the whole episode.
                student = label if model is None else predict(model, norm, obs)
                mix = assistance(mode, phase, env, beta)
                action = (mix*label+(1.-mix)*student).astype(np.float32)
                if mode != 'teacher' and phase == 1:
                    action[5:] = student[5:]
                nxt, reward, term, trunc, info = env.step(action)
                counts[phase] += 1
                values = dict(observations=obs.copy(), next_observations=nxt.copy(),
                    actions=label.copy(), executed_actions=action.copy(), rewards=reward,
                    dones=term or trunc,
                    # The 1000/3000/1000 limits are task deadlines, not artificial
                    # rollout cuts: timeout is a failure and must not bootstrap Q.
                    timeouts=False, phase_ids=phase, teacher_stages=teacher.stage,
                    episode_ids=ep, dagger_round=round_id, replay_valid=True)
                for k, value in values.items(): rows[k].append(value)
                agreements.append(float(np.sqrt(np.mean((label[:5]-student[:5])**2))))
                if int(env.phase_state) != phase:
                    handoffs.append(handoff_state(env,phase,int(env.phase_state)))
                if phase == 2 and (counts[2] == 1 or counts[2] % 25 == 0 or term or trunc):
                    trace.append(dict(step=counts[2], **p2_state(env)))
                obs = nxt
                if term or trunc: break
            for k, values in zip(('window01','window12','segment_a','segment_b'), windows(rows['phase_ids'])):
                rows[k] = values
            success = bool(info.get('strict_success',False))
            # Distillation only on successful AUTONOMOUS student episodes with
            # compatible teacher actions; no blanket anchor on source failures.
            rows['retain_mask'] = ((np.asarray(agreements) <= .05) & success & (mode == 'C'))
            data = cast(rows)
            if len(data['dones']) == 5000 and not data['dones'][-1]:
                raise RuntimeError('Episodio excede contrato físico; no guardar replay incompleto.')
            parts.append(data)
            episodes.append(dict(map=map_id,mass=mass,seed=seed,noise_fraction=noise,
                mode=mode,beta=beta,success=success,counts=counts,
                phases=[bool(info.get(f'phase{i}_success',False)) for i in range(3)],
                handoffs=handoffs,p2_trace=trace,
                reason=info.get('truncation_reason') or info.get('termination_reason')))
            # Commit one COMPLETE physical episode at a time, before announcing
            # progress. A signal can lose the current episode, not prior ones.
            shard = shard_dir/f'episode_{ep:04d}.npz'
            temporary = shard.with_suffix('.npz.tmp')
            with temporary.open('wb') as file:
                np.savez_compressed(file,**data)
            temporary.replace(shard)
            dump(shard.with_suffix('.json'),episodes[-1])
            dump(shard_dir/'progress.json',dict(completed=ep+1,total=len(specs),
                 complete=False,mode=mode,beta=beta))
            print('[R1-COLLECT]',mode,ep+1,len(specs),success,counts,flush=True)
            if announce is not None and (ep+1 == len(specs) or (ep+1) % 10 == 0):
                announce('COLLECT',dict(mode=mode,completed=ep+1,total=len(specs),
                    assisted=mode!='C',successes=sum(r['success'] for r in episodes)))
    finally:
        env.close()
    result = merge(parts)
    # Unique within this file. Across files sampling does not use episode ids.
    offset = 0
    for ep, part in enumerate(parts):
        result['episode_ids'][offset:offset+len(part['phase_ids'])] = ep
        offset += len(part['phase_ids'])
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as file:
        np.savez_compressed(file,**result)
    temporary.replace(path)
    dump(path.with_suffix('.json'),dict(episodes=episodes,mode=mode,beta=beta,
        success_rate=float(np.mean([r['success'] for r in episodes])),
        physical_replay_action_key='executed_actions',
        labels_are_not_replay_actions=True,
        transition01_rows=int(result['window01'].sum()),transition12_rows=int(result['window12'].sum())))
    dump(shard_dir/'progress.json',dict(completed=len(episodes),total=len(specs),
         complete=True,mode=mode,beta=beta))
    return result


class OverlapSampler:
    """50% cobertura de fases/subetapas; 50% segmento activo, con ventanas explícitas."""
    def __init__(self,data,stage,seed):
        self.rng = np.random.default_rng(seed)
        self.data,self.stage = data,stage
        self.groups = {}
        phase_ids = data['phase_ids']
        for phase in (0,1,2):
            self.groups[phase] = [np.flatnonzero((phase_ids==phase)&(data['teacher_stages']==name))
                for name in np.unique(data['teacher_stages'][phase_ids==phase])]
            if not self.groups[phase]: raise ValueError(f'Falta P{phase} en datos.')
        focus = data['segment_a'] if stage=='A' else data['segment_b'] if stage=='B' else (
            data['window01'] | data['window12'])
        self.focus = np.flatnonzero(focus)
        if not len(self.focus): raise ValueError(f'Sin transiciones para etapa {stage}.')
        self.recent = self.focus[data['dagger_round'][self.focus] == max(data['dagger_round'][self.focus])]
        self.retained = np.flatnonzero(data['retain_mask'])
        self.transition = np.flatnonzero(data['window01'] if stage=='A' else
                                        data['window12'] if stage=='B' else focus)
        if not len(self.transition): raise ValueError(f'Sin ventana explícita {stage}.')

    def sample(self,n=512):
        # One quarter active segment; one quarter actual transition window;
        # half all phases uniformly (then substage), preventing P1-length bias.
        idx=[]
        for _ in range(n//2):
            groups=self.groups[int(self.rng.integers(3))]
            idx.append(self.rng.choice(groups[self.rng.integers(len(groups))]))
        idx.extend(self.rng.choice(self.transition,n//4,replace=True))
        for _ in range(n-len(idx)):
            idx.append(self.rng.choice(self.recent if self.rng.random()<.65 else self.focus))
        return np.asarray(idx,np.int64)


def update(model,norm,data,stage,steps,lr,seed,anchor,announce=None):
    sampler = OverlapSampler(data,stage,seed)
    # Preserve Adam state through A/B/C. Only set a new optimizer at cycle start.
    for group in model.actor.optimizer.param_groups: group['lr']=lr
    model.actor.set_training_mode(True)
    losses=[]
    def inputs(idx):
        return torch.as_tensor(norm.normalize_obs(data['observations'][idx].copy()),
                               dtype=torch.float32,device=model.device)
    for step in range(steps):
        idx=sampler.sample()
        x=inputs(idx)
        targets=torch.as_tensor(data['actions'][idx],device=model.device)
        pred=model.actor(x,deterministic=True)
        fit=(pred[:,:5]-targets[:,:5]).square().mean()
        with torch.no_grad(): ref=anchor(x,deterministic=True)
        base=(pred[:,5:]-ref[:,5:]).square().mean()
        retention=torch.zeros((),device=model.device)
        if len(sampler.retained):
            kept=sampler.rng.choice(sampler.retained,256,replace=True)
            rx=inputs(kept)
            # Retain actions that actually produced a successful autonomous
            # episode, not an old actor's prediction on newly shifted states.
            rt=torch.as_tensor(data['executed_actions'][kept],device=model.device)
            retention=(model.actor(rx,deterministic=True)[:,:5]-rt[:,:5]).square().mean()
        loss=fit+retention+.5*base
        if not torch.isfinite(loss): raise FloatingPointError('BC loss no finita.')
        model.actor.optimizer.zero_grad();loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.actor.parameters(),1.)
        if not torch.isfinite(grad): raise FloatingPointError('Gradiente no finito.')
        model.actor.optimizer.step()
        losses.append([float(fit.detach()),float(retention.detach()),float(base.detach())])
        if announce and (step+1 in {steps//2,steps}):
            announce('UPDATE',dict(stage=stage,step=step+1,total=steps,losses=losses[-1]))
    return dict(first=losses[0],last=losses[-1],mean=np.mean(losses,axis=0).tolist())


def freeze_actor(model):
    anchor=copy.deepcopy(model.actor).eval()
    for p in anchor.parameters(): p.requires_grad_(False)
    return anchor


def promote(candidate,best):
    """Se aplica SOLO al ciclo integrado, nunca entre A y B/C."""
    improved=False
    for name in best:
        c,b=candidate[name],best[name]
        for profile in ('nominal','robust'):
            if c['rates'][profile]+1e-12 < b['rates'][profile]: return False
            improved |= c['rates'][profile]>b['rates'][profile]+1e-12
        if any(x+1e-12<y for x,y in zip(c['phases'][:2],b['phases'][:2])): return False
    return bool(improved)


def competent(panels,threshold=.95):
    return all(min(m['rates'].values())>=threshold and min(m['phases'][:2])>=.95
               for m in panels.values())
