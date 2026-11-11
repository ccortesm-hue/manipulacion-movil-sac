"""Read-only provenance gate for FINAL single-actor VX students, not teachers.

This module deliberately does not import torch, MuJoCo or pickle. A valid
manifest is necessary, not sufficient: extraction also loads the checkpoint
and checks its actual observation/action spaces and prediction shape.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENVIRONMENTS = {
    'r1_1': ('zenno_vx_r1_1', 'ZennoVXR11'),
    'r2': ('zenno_vx_r2', 'ZennoVXR2'),
}
ROUTE_NAMES = {f'map_{i}.json' for i in range(1, 5)}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def artifact(entry, manifest_dir):
    """Resolve one named local artifact and require an exact content hash."""
    if not isinstance(entry, dict) or not isinstance(entry.get('path'), str):
        raise ValueError('An artifact must declare path and sha256')
    path = Path(entry['path']).expanduser()
    path = (path if path.is_absolute() else manifest_dir / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = entry.get('sha256')
    if not isinstance(expected, str) or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(f'Artifact hash mismatch: {path}')
    return path


def validate_final_student(manifest_path, *, expected_experiment=None, root=ROOT):
    """Reject 2D P1 teachers, composed inference and stale/unassessed exports.

    Below-target success is NOT rejected: failures are valid experimental
    evidence. `target_met` is kept separate from architectural validity.
    """
    path = Path(manifest_path).expanduser().resolve()
    root = Path(root).resolve()
    manifest = json.loads(path.read_text(encoding='utf-8'))
    required = dict(schema_version=1, role='final_single_actor', actor_count=1,
                    observation_dim=124, action_dim=7, inference_composition='single_actor')
    for key, value in required.items():
        if manifest.get(key) != value:
            raise ValueError(f'Not a final 124D/7D single actor: {key}={manifest.get(key)!r}')
    experiment = manifest.get('experiment')
    if experiment not in ENVIRONMENTS or (expected_experiment and experiment != expected_experiment):
        raise ValueError(f'Wrong experiment: {experiment!r}')
    environment = manifest.get('environment', {})
    module, class_name = ENVIRONMENTS[experiment]
    if (environment.get('module'), environment.get('class')) != (module, class_name):
        raise ValueError('Environment family/class differs from declared experiment')
    env_path = root / (module + '.py')
    if environment.get('sha256') != sha256(env_path):
        raise ValueError(f'Environment source changed after export: {env_path}')
    if not environment.get('revision') or not environment.get('action_semantics'):
        raise ValueError('Environment revision/action semantics must be explicit')
    kwargs = environment.get('kwargs', {})
    if not isinstance(kwargs, dict) or kwargs.get('residual_enabled', True) is not True:
        raise ValueError('Final comparison requires the declared residual ON')
    if any(k in kwargs for k in ('render_mode', 'debug_rewards')):
        raise ValueError('Rendering/logging is configured by extractor, not model contract')
    budgets = manifest.get('phase_budgets')
    if not isinstance(budgets, list) or len(budgets) != 3 or any(type(v) is not int or v < 1 for v in budgets):
        raise ValueError('Explicit positive integer P0/P1/P2 budgets required')
    if budgets[0] != 1000 or budgets[2] != 1000 or kwargs.get('p1_budget_steps', 3000) != budgets[1]:
        raise ValueError('Phase budgets and environment kwargs disagree')
    routes = manifest.get('route_hashes', {})
    if not ROUTE_NAMES <= set(routes):
        raise ValueError('All four RRT route hashes are required')
    for name, digest in routes.items():
        if Path(name).name != name or not name.endswith('.json'):
            raise ValueError('Invalid route filename')
        if sha256(root / 'RUTAS_SAC' / name) != digest:
            raise ValueError(f'Route changed: {name}')
    artifacts = manifest.get('artifacts', {})
    model = artifact(artifacts.get('model'), path.parent)
    normalizer = artifact(artifacts.get('vecnormalize'), path.parent)
    if model.suffix != '.zip' or normalizer.suffix != '.pkl':
        raise ValueError('Expected SAC ZIP and VecNormalize pickle')
    validation = manifest.get('validation', {})
    if validation.get('status') not in ('evaluated', 'passed') or type(validation.get('target_met')) is not bool:
        raise ValueError('A finished evaluation, separately from target_met, is required')
    report_path = artifact(validation.get('report'), path.parent)
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report.get('complete') is not True or report.get('scope') != 'full_task' or report.get('deterministic') is not True:
        raise ValueError('Validation must be complete deterministic full-task evaluation')
    if report.get('model_sha256') != artifacts['model']['sha256']:
        raise ValueError('Validation belongs to a different checkpoint')
    n, successes = report.get('cases'), report.get('successes')
    if type(n) is not int or type(successes) is not int or n < 1 or not 0 <= successes <= n:
        raise ValueError('Invalid validation counts')
    return dict(manifest=manifest, manifest_path=path, model_path=model,
                vecnormalize_path=normalizer, validation_report=report,
                validation_path=report_path, environment_path=env_path)


def load_environment_class(bundle):
    """Runtime revision check; call only after the read-only provenance gate."""
    import importlib
    declared = bundle['manifest']['environment']
    cls = getattr(importlib.import_module(declared['module']), declared['class'])
    if cls.REVISION != declared['revision'] or cls.ACTION_SEMANTICS != declared['action_semantics']:
        raise ValueError('Runtime environment revision/action semantics differ from export')
    return cls
