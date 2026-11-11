"""Datos acotados para destilar un profesor P1 en un único actor VX de 7 acciones.

Las etiquetas DAgger nunca sustituyen a las acciones físicas en el replay.
Los datos antiguos P0/P2 son exclusivamente etiquetas BC, aun si contienen
columnas de recompensa o transiciones. Sólo se aceptan con contrato verificable.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

FLOATS = ('observations', 'next_observations', 'actions', 'executed_actions', 'rewards')
BOOLS = ('dones', 'replay_valid', 'window01', 'window12')
INTS = ('phase_ids', 'episode_ids', 'origin')
FIELDS = FLOATS + BOOLS + INTS


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_npz(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as destination:
        np.savez_compressed(destination, **data)
    temporary.replace(path)


def windows(phases, before=100, after=200):
    phases = np.asarray(phases)
    result = [np.zeros(len(phases), bool), np.zeros(len(phases), bool)]
    for target, mask in zip((1, 2), result):
        for index in np.flatnonzero((phases[1:] == target) & (phases[:-1] == target - 1)) + 1:
            mask[max(0, index - before):min(len(phases), index + after)] = True
    return tuple(result)


def cast(data):
    return {key: np.asarray(data[key], dtype=np.float32 if key in FLOATS else
                           bool if key in BOOLS else np.int32) for key in FIELDS}


def validate_data(data):
    n = len(data['phase_ids'])
    if n == 0 or any(len(data[key]) != n for key in FIELDS):
        raise ValueError('Dataset vacío o columnas con longitudes diferentes.')
    for key, width in (('observations', 124), ('next_observations', 124),
                       ('actions', 7), ('executed_actions', 7)):
        if data[key].shape != (n, width) or not np.isfinite(data[key]).all():
            raise ValueError(f'{key} no cumple el contrato finito {width}D.')
    if not np.isfinite(data['rewards']).all() or not np.isin(data['phase_ids'], (0, 1, 2)).all():
        raise ValueError('Fases o recompensas inválidas.')
    if not np.isin(data['origin'], (0, 1)).all():
        raise ValueError('Origen desconocido: 0=física nueva, 1=etiquetas antiguas.')
    for name in ('actions', 'executed_actions'):
        if np.any(np.abs(data[name]) > 1.00001):
            raise ValueError('Acciones fuera de [-1,1].')
    legacy = data['origin'] == 1
    if np.any(legacy & (data['replay_valid'] | (data['phase_ids'] == 1))):
        raise ValueError('Datos antiguos no pueden aportar P1 ni transiciones al replay.')
    return data


def balanced_indices(data, count, rng, replace=True):
    """75% por fase uniforme; 25% ventanas reales P0→P1/P1→P2."""
    groups = [np.flatnonzero(data['phase_ids'] == phase) for phase in (0, 1, 2)]
    groups = [group for group in groups if len(group)]
    if not groups:
        raise ValueError('No hay fases para muestrear.')
    if not replace:
        # Capacity allocation is phase-balanced without assuming equal lengths.
        chosen = []
        remaining = min(count, len(data['phase_ids']))
        order = sorted(groups, key=len)
        for index, group in enumerate(order):
            take = min(len(group), remaining // (len(order) - index))
            chosen.append(rng.choice(group, take, replace=False))
            remaining -= take
        result = np.concatenate(chosen)
        rng.shuffle(result)
        return result
    focus = np.flatnonzero(data['window01'] | data['window12'])
    focused = count // 4 if len(focus) else 0
    choice = rng.integers(len(groups), size=count - focused)
    result = np.empty(count, np.int64)
    for index, group in enumerate(groups):
        positions = np.flatnonzero(choice == index)
        result[positions] = rng.choice(group, len(positions), replace=True)
    if focused:
        result[-focused:] = rng.choice(focus, focused, replace=True)
    rng.shuffle(result)
    return result


def bounded_merge(parts, max_rows=60000, seed=0):
    """Cap the in-memory pool without concatenating all observation matrices.

    At most the current bounded pool plus ONE complete episode is held by the
    collector. Persistent episode shards retain the full original trajectories.
    """
    if max_rows < 3:
        raise ValueError('Capacidad insuficiente para las tres fases.')
    parts = [validate_data(part) for part in parts if part is not None]
    phases = np.concatenate([part['phase_ids'] for part in parts])
    indices = balanced_indices({'phase_ids': phases}, min(max_rows, len(phases)),
                               np.random.default_rng(seed), replace=False)
    lengths = np.asarray([len(part['phase_ids']) for part in parts])
    boundaries = np.r_[0, lengths.cumsum()]
    result = {}
    for key in FIELDS:
        shape = (len(indices),) + parts[0][key].shape[1:]
        result[key] = np.empty(shape, dtype=parts[0][key].dtype)
        for index, part in enumerate(parts):
            mask = (indices >= boundaries[index]) & (indices < boundaries[index + 1])
            result[key][mask] = part[key][indices[mask] - boundaries[index]]
    return validate_data(result)


def load_old_bc(path, *, source_env_hash, source_norm_hash, max_rows=60000):
    """Require path.contract.json; retain ONLY raw 124D P0/P2 labels.

    Contract: schema_version=1, role='bc_labels', observation_dim=124,
    action_dim=7, observations='raw', source_revision=R1 exact,
    source_environment_sha256, normalizer_sha256, dataset_sha256.
    A matching architecture alone is deliberately insufficient.
    """
    path = Path(path).resolve()
    contract_path = path.with_suffix('.contract.json')
    contract = json.loads(contract_path.read_text())
    if contract.get('role') not in ('bc_labels', 'BC_ONLY_P0_P2_NOT_REPLAY'):
        raise ValueError('El artefacto no se declara exclusivamente como etiquetas BC.')
    if contract.get('role') == 'BC_ONLY_P0_P2_NOT_REPLAY' and (
            contract.get('bc_only') is not True or contract.get('replay_compatible') is not False):
        raise ValueError('El contrato debe excluir expresamente el replay antiguo.')
    expected = dict(schema_version=1, observation_dim=124,
                    action_dim=7, observations='raw',
                    source_revision='VX_R1_OVERLAP_CURRICULUM_124D_7D_DELTA008_UNIFORM_COSTS',
                    source_environment_sha256=source_env_hash,
                    normalizer_sha256=source_norm_hash, dataset_sha256=sha256(path))
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError('El contrato BC antiguo no demuestra compatibilidad con la fuente R1.')
    with np.load(path, allow_pickle=False) as loaded:
        phases = np.asarray(loaded['phase_ids'])
        selected = np.flatnonzero(np.isin(phases, (0, 2)))
        if not len(selected):
            raise ValueError('No hay etiquetas P0/P2 reutilizables.')
        if len(selected) > max_rows:
            local = balanced_indices({'phase_ids': phases[selected]}, max_rows,
                                     np.random.default_rng(0), replace=False)
            selected = selected[local]
        obs = np.asarray(loaded['observations'][selected], np.float32)
        labels = np.asarray(loaded['actions'][selected], np.float32)
        selected_phases = phases[selected]
    n = len(selected)
    data = dict(observations=obs, next_observations=np.zeros((n, 124)), actions=labels,
                executed_actions=np.zeros((n, 7)), rewards=np.zeros(n), dones=np.ones(n),
                phase_ids=selected_phases, episode_ids=np.full(n, -1), origin=np.ones(n),
                replay_valid=np.zeros(n), window01=np.zeros(n), window12=np.zeros(n))
    return validate_data(cast(data))


def replay_rows(data):
    validate_data(data)
    return np.flatnonzero(data['replay_valid'] & (data['origin'] == 0))


def collect(env_factory, label_fn, student_fn, cases, output, *, pool=None,
            max_rows=60000, seed=0, announce=None):
    """Full physical teacher/student episodes, no resets at phase boundaries.

    student_fn=None means pure teacher; otherwise all seven executed actions
    belong to the student, without rescue or phase-specific mixing.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    env = env_factory()
    summaries = []
    dump(output / 'manifest.json', dict(status='RUNNING', expected_episodes=len(cases),
         collection='teacher' if student_fn is None else 'student_dagger',
         observation_dim=124, action_dim=7, observations='raw',
         physical_replay_action_key='executed_actions', labels_are_not_replay_actions=True))
    try:
        for episode, case in enumerate(cases):
            env.set_eval_mode(True)
            env.FIXED_MAP_ID = int(case['map'])
            env.set_eval_mass(float(case['mass']))
            robust = case.get('profile', 'robust') == 'robust'
            env.CUBE_XY_NOISE_RANGE = .01 if robust else 0.
            env.CUBE_YAW_NOISE_RANGE_RAD = .10 if robust else 0.
            obs, _ = env.reset(seed=int(case['seed']))
            rows = {key: [] for key in FIELDS if key not in ('window01', 'window12')}
            counts = [0, 0, 0]
            ended = False
            for _ in range(sum((env.MAX_STEPS_0, env.MAX_STEPS_1, env.MAX_STEPS_2)) + 1):
                phase = int(env.phase_state)
                label = np.asarray(label_fn(obs, env), np.float32)
                executed = label.copy() if student_fn is None else np.asarray(student_fn(obs, env), np.float32)
                if label.shape != (7,) or executed.shape != (7,):
                    raise ValueError('Profesor y estudiante deben devolver acción completa 7D.')
                if not np.isfinite(label).all() or np.any(np.abs(label) > 1.00001):
                    raise ValueError('El profesor generó una etiqueta inválida.')
                if not np.isfinite(executed).all() or np.any(np.abs(executed) > 1.00001):
                    raise ValueError('No ejecutar acciones inválidas ni recortarlas silenciosamente.')
                nxt, reward, terminated, truncated, info = env.step(executed)
                ended = bool(terminated or truncated)
                values = dict(observations=obs.copy(), next_observations=nxt.copy(),
                              actions=label.copy(), executed_actions=executed.copy(),
                              rewards=reward, dones=ended, phase_ids=phase,
                              episode_ids=episode, origin=0, replay_valid=True)
                for key, value in values.items():
                    rows[key].append(value)
                counts[phase] += 1
                obs = nxt
                if ended:
                    break
            if not ended:
                raise RuntimeError('No convertir un corte del evaluador en transición terminal física.')
            rows['window01'], rows['window12'] = windows(rows['phase_ids'])
            data = validate_data(cast(rows))
            name = f'episode_{episode:04d}'
            save_npz(output / (name + '.npz'), data)
            summary = dict(**case, counts=counts, success=bool(info.get('strict_success', False)),
                           phases=[bool(info.get(f'phase{i}_success')) for i in (0, 1, 2)],
                           reason=info.get('truncation_reason') or info.get('termination_reason'),
                           complete=True, rows=len(data['phase_ids']),
                           window01=int(data['window01'].sum()), window12=int(data['window12'].sum()))
            dump(output / (name + '.json'), summary)
            summaries.append(summary)
            pool = bounded_merge([pool, data], max_rows, seed + episode)
            dump(output / 'progress.json', dict(completed=episode + 1, total=len(cases),
                 successes=sum(r['success'] for r in summaries), pool_rows=len(pool['phase_ids'])))
            if announce:
                announce('COLLECT_EPISODE', **summary, completed=episode + 1, total=len(cases))
        dump(output / 'summary.json', dict(complete=True, episodes=summaries,
             successes=sum(r['success'] for r in summaries), pool_rows=len(pool['phase_ids'])))
        manifest = json.loads((output / 'manifest.json').read_text())
        manifest['status'] = 'COMPLETE'
        dump(output / 'manifest.json', manifest)
    except BaseException as error:
        dump(output / 'interrupted.json', dict(error=repr(error), completed=len(summaries)))
        raise
    finally:
        env.close()
    return pool
