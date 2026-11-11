"""Canonical initial dynamic state for NEW placement comparisons only.

Call AFTER both independent resets and the baseline's _set_home_pose. Physics
must already match exactly and poses/velocities within the documented reset
tolerances. This module never imports experiment/controller sources, changes
model parameters, advances physics or loads a policy. Refresh the environment
observation after this function, before applying its frozen normalizer.
"""
from __future__ import annotations

import hashlib
import numpy as np
import mujoco

PROTOCOL = 'vx_place_shared_initial_dynamic_state_v1'
QPOS_ATOL = 1e-8
QVEL_ATOL = 1e-10
MODEL_PREFIXES = ('body_', 'geom_', 'dof_', 'jnt_', 'actuator_', 'eq_',
                  'pair_', 'tendon_', 'site_', 'sensor_', 'wrap_')
MODEL_FIELDS = ('qpos0', 'qpos_spring', 'exclude_signature')
DIMENSIONS = ('nq', 'nv', 'nu', 'na', 'nbody', 'njnt', 'ngeom', 'neq',
              'nmocap', 'ntendon', 'npair', 'nexclude', 'nsensor', 'nplugin')
ASSET_FIELDS = ('mesh_vert', 'mesh_face', 'mesh_pos', 'mesh_quat', 'mesh_scale',
                'hfield_data', 'hfield_size', 'hfield_nrow', 'hfield_ncol')
DYNAMIC_FIELDS = ('qpos', 'qvel', 'ctrl', 'act', 'time', 'qfrc_applied',
                  'xfrc_applied', 'eq_active', 'mocap_pos', 'mocap_quat',
                  'userdata', 'plugin_state')
SNAPSHOT_FIELDS = DYNAMIC_FIELDS + ('qacc_warmstart',)


class InitialStateMismatch(ValueError):
    """Failed, JSON-serializable evidence is available as exception.audit."""
    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = audit


def _numeric(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in 'biuf' or not np.isfinite(array).all():
        raise ValueError(f'{name}: expected finite numeric data')
    return array.copy()


def _maximum_difference(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f'Array shape differs: {a.shape} versus {b.shape}')
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.


def _physical(model):
    arrays = {}
    for name in sorted(set(MODEL_FIELDS) | {name for name in dir(model)
                                          if name.startswith(MODEL_PREFIXES)}):
        if hasattr(model, name):
            arrays[name] = _numeric(getattr(model, name), name)
    dimensions = {name: int(getattr(model, name)) for name in DIMENSIONS if hasattr(model, name)}
    options = {name: _numeric(getattr(model.opt, name), 'opt.' + name)
               for name in dir(model.opt) if not name.startswith('_')}
    # Collision geometry may live in mesh/heightfield assets, not geom_size.
    assets = {}
    for name in ASSET_FIELDS:
        if hasattr(model, name):
            value = _numeric(getattr(model, name), name)
            assets[name] = dict(shape=list(value.shape), dtype=str(value.dtype),
                               sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())
    return dict(arrays=arrays, dimensions=dimensions, options=options,
                asset_digests=assets, names_sha256=hashlib.sha256(bytes(model.names)).hexdigest())


def _physical_json(snapshot):
    # Flat arrays remain easy to compare with the older initial_states table.
    return {**{name: value.tolist() for name, value in snapshot['arrays'].items()},
            'dimensions': snapshot['dimensions'],
            'options': {name: value.tolist() for name, value in snapshot['options'].items()},
            'asset_digests': snapshot['asset_digests'], 'names_sha256': snapshot['names_sha256']}


def _dynamic(data):
    return {name: _numeric(getattr(data, name), name) for name in SNAPSHOT_FIELDS}


def _state_json(dynamic, physical):
    return dict(data={name: value.tolist() for name, value in dynamic.items()},
                physical_parameters=_physical_json(physical))


def _validate_reference_home(env):
    """Do not bless a progressed episode or an arbitrary initial command."""
    if int(env.phase_state) != 0 or float(env.data.time) != 0.:
        raise ValueError('Reference must be a fresh P0 reset at simulation time zero')
    if np.max(np.abs(env.data.qvel), initial=0.) > QVEL_ATOL:
        raise ValueError('Reference reset must start at rest')
    arms = list(env.arm_act_ids[:5])
    wheels = list(env.wheel_left_ids) + list(env.wheel_right_ids)
    grip = int(env.grip_aid)
    if len(arms) != 5 or len(wheels) != 6 or len(set(arms + wheels + [grip])) != 12:
        raise ValueError('Expected five arm, six wheel and one gripper actuator')
    if len(env.data.ctrl) != 12:
        raise ValueError('Unaccounted actuators in reference reset')
    home = _numeric(env.q_arm_home, 'q_arm_home').reshape(5)
    for index, aid in enumerate(arms):
        jid = int(env.model.actuator_trnid[aid, 0])
        q = float(env.data.qpos[int(env.model.jnt_qposadr[jid])])
        if abs(q - home[index]) > QPOS_ATOL:
            raise ValueError('Reference arm is not at its declared home pose')
        target = np.clip(home[index], *env.model.actuator_ctrlrange[aid])
        # VX stores arm_target in float32 before writing MuJoCo float64 ctrl.
        # Accept only those two exact representations, not an arbitrary wider
        # reset-pose tolerance (which remains 1e-8 for physical qpos).
        valid_targets = (float(target), float(np.float32(target)))
        if float(env.data.ctrl[aid]) not in valid_targets:
            raise ValueError('Reference arm control does not hold its home target')
    if np.any(env.data.ctrl[wheels] != 0.):
        raise ValueError('Reference wheel torques must start at zero')
    if float(env.data.ctrl[grip]) != float(env.grip_open):
        raise ValueError('Reference gripper must start open')


def canonicalize_and_audit(env, mm) -> dict:
    """Audit BEFORE writing, copy reference dynamic state, forward, audit EXACT.

    ``env`` may be a Gym wrapper; ``mm`` supplies model/data as NEW_MM_VISUAL
    does. Returns JSON-safe ``passed``, ``before``/``after`` snapshots and
    ``max_abs_differences_before``/``max_abs_differences_after`` dictionaries.
    Physical mismatches (including solver options) or nontrivial qpos/qvel
    differences raise InitialStateMismatch BEFORE any state is changed.
    A postcondition failure restores both input dynamic states and raises.
    """
    env = getattr(env, 'unwrapped', env)
    report = dict(protocol=PROTOCOL, passed=False, physics_modified=False,
                  physics_advanced=False, reference='env',
                  tolerances=dict(qpos=QPOS_ATOL, qvel=QVEL_ATOL, physical_parameters=0.),
                  observation_must_be_refreshed=True)
    originals = None
    copied = False
    try:
        reference_physics, baseline_physics = _physical(env.model), _physical(mm.model)
        reference, baseline = _dynamic(env.data), _dynamic(mm.data)
        originals = (reference, baseline)
        report['before'] = dict(env=_state_json(reference, reference_physics),
                                baseline=_state_json(baseline, baseline_physics))
        maxima = {name: _maximum_difference(reference[name], baseline[name])
                  for name in SNAPSHOT_FIELDS}
        report['max_abs_differences_before'] = dict(data=maxima, physical_parameters={})
        errors = []
        for group in ('dimensions', 'asset_digests', 'names_sha256'):
            if reference_physics[group] != baseline_physics[group]:
                errors.append('physical.' + group)
        for group in ('arrays', 'options'):
            a, b = reference_physics[group], baseline_physics[group]
            if set(a) != set(b):
                errors.append('physical.' + group + '.fields')
                continue
            for name in a:
                key = ('opt.' if group == 'options' else '') + name
                delta = _maximum_difference(a[name], b[name])
                report['max_abs_differences_before']['physical_parameters'][key] = delta
                if not np.array_equal(a[name], b[name]):
                    errors.append('physical.' + key)
        for name, tolerance in (('qpos', QPOS_ATOL), ('qvel', QVEL_ATOL)):
            if maxima[name] > tolerance:
                errors.append(name + f' difference {maxima[name]} > {tolerance}')
        if errors:
            raise ValueError('Initial conditions differ before canonicalization: ' + ', '.join(errors))
        _validate_reference_home(env)
        if hasattr(env, 'frame_skip') and hasattr(mm, 'FRAMESKIP_CONTROL'):
            if int(env.frame_skip) != int(mm.FRAMESKIP_CONTROL):
                raise ValueError('Control frameskip differs')
        # This is the first mutation. Model parameters and controller memory
        # remain untouched; only reset-time MuJoCo state is standardized.
        copied = True
        for target, model in ((env.data, env.model), (mm.data, mm.model)):
            for name in DYNAMIC_FIELDS:
                if name == 'time':
                    target.time = float(reference[name])
                else:
                    getattr(target, name)[:] = reference[name]
            target.qacc_warmstart[:] = 0.
            mujoco.mj_forward(model, target)
            target.qacc_warmstart[:] = 0.
        reference_after, baseline_after = _dynamic(env.data), _dynamic(mm.data)
        reference_physics_after, baseline_physics_after = _physical(env.model), _physical(mm.model)
        report['after'] = dict(env=_state_json(reference_after, reference_physics_after),
                               baseline=_state_json(baseline_after, baseline_physics_after))
        for label in ('env', 'baseline'):
            if report['before'][label]['physical_parameters'] != report['after'][label]['physical_parameters']:
                raise ValueError('Physical parameters changed during forward: ' + label)
        differences = {name: _maximum_difference(reference_after[name], baseline_after[name])
                       for name in SNAPSHOT_FIELDS}
        report['max_abs_differences_after'] = dict(data=differences,
            physical_parameters=report['max_abs_differences_before']['physical_parameters'].copy())
        if any(not np.array_equal(reference_after[name], baseline_after[name]) for name in SNAPSHOT_FIELDS):
            raise ValueError('Dynamic state is not exactly equal after canonicalization')
        report.update(passed=True, status='CANONICALIZED_AND_AUDITED',
                      canonicalized_fields=list(DYNAMIC_FIELDS) + ['qacc_warmstart'],
                      warmstart_policy='zero_before_forward_and_after_forward')
        return report
    except Exception as error:
        if copied and originals is not None:
            for target, model, saved in ((env.data, env.model, originals[0]),
                                          (mm.data, mm.model, originals[1])):
                for name in DYNAMIC_FIELDS:
                    if name == 'time':
                        target.time = float(saved[name])
                    else:
                        getattr(target, name)[:] = saved[name]
                mujoco.mj_forward(model, target)
                target.qacc_warmstart[:] = saved['qacc_warmstart']
            report['input_dynamic_state_restored'] = True
        report.update(status='REJECTED', error=str(error))
        raise InitialStateMismatch(str(error), report) from error
