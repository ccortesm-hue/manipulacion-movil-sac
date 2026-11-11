"""Composición del profesor 2D con brazo congelado, sin canal artificial al brazo.

No es la política final desplegable. El SAC temporal sigue observando los 124
canales reales; únicamente el controlador auxiliar del brazo recibe una copia
sin las solicitudes/memorias sintéticas del residual. Así un residual inactivo
no puede cambiar el brazo simplemente escribiendo otra acción en la observación.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from zenno_vx_r1_1 import VX_EXTRA_NAMES, ZennoVXR11


ARM_MASK_REVISION = 'R11_FROZEN_ARM_SYNTHETIC_RESIDUAL_CHANNELS_ZERO_V1'
ARM_MASK_EXTRA_NAMES = ('delta_left', 'delta_right', 'residual_filter_lin',
                        'residual_filter_ang', 'residual_active')
ARM_MASK_INDICES = (30, 31) + tuple(78 + VX_EXTRA_NAMES.index(name)
                                   for name in ARM_MASK_EXTRA_NAMES)


def arm_observation(raw, phase):
    """Copia 124D para el brazo; P0/P2 idénticas, P1 con siete canales a cero.

    Se enmascara ANTES de normalizar. Pose, velocidad, fuerzas nominales PID,
    estado del cubo, ruta y tiempo restante conservan sus valores físicos.
    """
    observation = np.asarray(raw, dtype=np.float32).copy()
    if observation.shape != (124,) or not np.all(np.isfinite(observation)):
        raise ValueError('El brazo auxiliar requiere una observación finita 124D.')
    if int(phase) not in (0, 1, 2):
        raise ValueError('Fase fuera de P0/P1/P2.')
    if int(phase) == 1:
        observation[list(ARM_MASK_INDICES)] = 0.
    return observation


def composed_action(source, norm, teacher=None):
    """Devuelve callback (obs_cruda, env) -> acción7D para evaluaciones parejas.

    source: actor histórico 7D congelado. teacher: profesor 2D opcional.
    Tanto source-only como teacher usan la MISMA copia enmascarada para el brazo.
    OFF envía solicitudes cero además de desactivar físicamente el residual.
    El análisis histórico de R1 sin máscara es un experimento separado.
    """
    def action_fn(raw, env):
        phase = int(env.phase_state)
        observation = np.asarray(raw, dtype=np.float32).copy()
        if observation.shape != (124,) or not np.all(np.isfinite(observation)):
            raise ValueError('El profesor requiere una observación finita 124D.')
        normalized = norm.normalize_obs(observation.copy())
        action = np.asarray(source.predict(normalized, deterministic=True)[0],
                            dtype=np.float32).copy()
        if action.shape != (7,):
            raise ValueError('El controlador fuente debe devolver siete acciones.')
        if phase == 1:
            masked = norm.normalize_obs(arm_observation(observation, phase))
            arm = np.asarray(source.predict(masked, deterministic=True)[0],
                             dtype=np.float32)
            if arm.shape != (7,):
                raise ValueError('El controlador del brazo debe devolver siete acciones.')
            action[:5] = arm[:5]
            if not bool(env.residual_enabled):
                action[5:] = 0.
            elif teacher is not None:
                residual = np.asarray(teacher.predict(normalized, deterministic=True)[0],
                                      dtype=np.float32)
                if residual.shape != (2,):
                    raise ValueError('El profesor residual debe devolver dos acciones.')
                action[5:] = residual
        if not np.all(np.isfinite(action)) or np.any(np.abs(action) > 1.00001):
            raise ValueError('Política fuera del contrato finito [-1,1].')
        return np.clip(action, -1., 1.).astype(np.float32)
    return action_fn


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_teacher(path, device='cpu', *, source_path=None, norm_path=None,
                 expected_budget=None):
    """Carga SOLO un profesor R1.1 2D confiable y local; replay de una fila.

    norm_path identifica el normalizador FUENTE original. El profesor se evalúa
    normalizando una única vez con éste, como durante su entrenamiento.
    """
    from stable_baselines3 import SAC

    path = Path(path).resolve()
    if path.is_dir():
        path = path/'model.zip'
    contract_path = path.parent/'contract.json'
    config_path = path.parent.parent/'config.json'
    for required in (path, contract_path, config_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    contract = json.loads(contract_path.read_text())
    config = json.loads(config_path.read_text())
    role = 'P1_2D_teacher_NOT_final_student'
    if contract.get('revision') != ZennoVXR11.REVISION or config.get('revision') != ZennoVXR11.REVISION:
        raise ValueError('El checkpoint no corresponde a la semántica R1.1 de torque.')
    if contract.get('role') != role or config.get('artifact_role') != role:
        raise ValueError('Se requiere el profesor temporal P1 de dos acciones.')
    if (contract.get('observation_dim'), contract.get('action_dim')) != (124, 2):
        raise ValueError('El profesor debe declarar contrato 124D/2D.')
    budget = contract.get('p1_budget')
    if budget != config.get('p1_budget') or (expected_budget is not None and budget != expected_budget):
        raise ValueError('El presupuesto P1 no coincide con el entrenamiento del profesor.')
    mask_revision = config.get('arm_mask_revision')
    if mask_revision != ARM_MASK_REVISION:
        raise ValueError('Falta el protocolo común de aislamiento del brazo frente al residual.')
    if source_path is not None:
        digest = _sha(source_path)
        if digest != contract.get('source_actor_sha256') or digest != config.get('source_sha256'):
            raise ValueError('El actor auxiliar no es el utilizado al entrenar el profesor.')
    if norm_path is not None and _sha(norm_path) != config.get('normalizer_sha256'):
        raise ValueError('El normalizador fuente no corresponde al entrenamiento.')
    saved_norm_hash = contract.get('normalizer_sha256')
    if saved_norm_hash is not None and _sha(path.parent/'vecnormalize.pkl') != saved_norm_hash:
        raise ValueError('El normalizador guardado del profesor fue modificado.')
    for name, expected in config.get('route_hashes', {}).items():
        if _sha(Path(__file__).resolve().parent/'RUTAS_SAC'/name) != expected:
            raise ValueError(f'Ruta diferente respecto del profesor: {name}')
    model = SAC.load(path, device=device, custom_objects={'buffer_size': 1})
    if model.observation_space.shape != (124,) or model.action_space.shape != (2,):
        raise ValueError('El archivo SAC no cumple el contrato 124D/2D.')
    model.policy.set_training_mode(False)
    for parameter in model.policy.parameters():
        parameter.requires_grad_(False)
    return model
