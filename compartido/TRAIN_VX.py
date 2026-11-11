#!/usr/bin/env python3
"""Entrenamiento consolidado VX: profesor SAC P1, estudiante BC/DAgger/SAC.

Reutiliza los ambientes consolidados; no importa fuentes Python históricas.
Los checkpoints fuente son recursos necesarios para esta transferencia, no
un entrenamiento desde cero. El profesor 2D solo participa en entrenamiento;
el estudiante final ejecuta un único actor 7D.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('MPLCONFIGDIR','/tmp/vx_train_compact')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
from MM_DATA_VXR1 import (ZennoVXR11, VX_EXTRA_NAMES, RoutePoseMetric,
                        RESOURCE_ROOT, validate_final_student, verify_consolidated_sources)
from MM_DATA_VXR2 import ZennoVXR2
CODE_ROOT = Path(__file__).resolve().parent
ROOT = RESOURCE_ROOT

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



from typing import Dict, List

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update


class DemoRegularizedSAC(SAC):
    """Añade una pérdida BC balanceada por fase al actor SAC estándar."""

    def set_demo_regularization(
        self,
        dataset: Dict[str, np.ndarray],
        loss_weight: float,
        batch_size: int,
        seed: int,
        *,
        normalize_q_actor_loss: bool = False,
        q_objective_scale: float = 1.0,
        actor_warmup_updates: int = 0,
        action_weights_by_phase: np.ndarray | None = None,
    ) -> None:
        self._demo_observations = np.asarray(
            dataset["observations"], dtype=np.float32
        ).copy()
        self._demo_actions = np.asarray(
            dataset["actions"], dtype=np.float32
        ).copy()
        self._demo_loss_weight = float(max(loss_weight, 0.0))
        self._demo_batch_size = int(max(batch_size, 1))
        self._demo_rng = np.random.default_rng(int(seed))
        self._normalize_q_actor_loss = bool(normalize_q_actor_loss)
        self._q_objective_scale = float(max(q_objective_scale, 0.0))
        self._actor_warmup_updates = int(max(actor_warmup_updates, 0))
        self._demo_phase_ids = (
            np.asarray(dataset["phase_ids"], dtype=np.int64).copy()
            if "phase_ids" in dataset else None
        )
        if action_weights_by_phase is None:
            self._demo_action_weights_by_phase = None
        else:
            weights = np.asarray(action_weights_by_phase, dtype=np.float32)
            expected = (3, int(self.action_space.shape[0]))
            if weights.shape != expected:
                raise ValueError(
                    f"action_weights_by_phase debe tener forma {expected}, "
                    f"no {weights.shape}."
                )
            if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
                raise ValueError(
                    "action_weights_by_phase debe ser finito y no negativo."
                )
            self._demo_action_weights_by_phase = weights.copy()
        self._demo_group_indices = None
        if "phase_ids" in dataset:
            phase_ids = np.asarray(dataset["phase_ids"], dtype=np.int64)
            if "teacher_stages" in dataset:
                stages = np.asarray(dataset["teacher_stages"]).astype(str)
                groups = {
                    f"p{phase}:{stage}": np.flatnonzero(
                        (phase_ids == phase) & (stages == stage)
                    )
                    for phase in (0, 1, 2)
                    for stage in np.unique(stages[phase_ids == phase])
                }
            else:
                # Compatibilidad con datasets antiguos: en P2 se separan las
                # órdenes activas del tramo release/home (acción docente cero).
                active = np.max(np.abs(self._demo_actions), axis=1) > 1.0e-6
                groups = {
                    "p0": np.flatnonzero(phase_ids == 0),
                    "p1": np.flatnonzero(phase_ids == 1),
                    "p2:active": np.flatnonzero((phase_ids == 2) & active),
                    "p2:passive": np.flatnonzero((phase_ids == 2) & ~active),
                }
            groups = {name: rows for name, rows in groups.items() if rows.size > 0}
            if groups and all(
                any(name.startswith(f"p{phase}") for name in groups)
                for phase in (0, 1, 2)
            ):
                self._demo_group_indices = groups

    def _sample_demo_indices(self) -> np.ndarray:
        groups = getattr(self, "_demo_group_indices", None)
        if not groups:
            return self._demo_rng.integers(
                0,
                self._demo_observations.shape[0],
                size=self._demo_batch_size,
            )
        names = tuple(groups)
        selected_groups = self._demo_rng.integers(
            0, len(names), size=self._demo_batch_size
        )
        indices = np.empty(self._demo_batch_size, dtype=np.int64)
        for group_id, name in enumerate(names):
            positions = np.flatnonzero(selected_groups == group_id)
            if positions.size:
                indices[positions] = self._demo_rng.choice(
                    groups[name], size=positions.size, replace=True
                )
        return indices

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses: List[float] = []
        ent_coefs: List[float] = []
        actor_losses: List[float] = []
        actor_sac_losses: List[float] = []
        critic_losses: List[float] = []
        demo_losses: List[float] = []
        q_abs_means: List[float] = []
        q_actor_scales: List[float] = []
        skipped_critic_updates = 0
        skipped_actor_updates = 0
        frozen_actor_updates = 0

        demo_enabled = bool(
            getattr(self, "_demo_loss_weight", 0.0) > 0.0
            and getattr(self, "_demo_observations", None) is not None
        )

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(  # type: ignore[union-attr]
                batch_size, env=self._vec_normalize_env
            )
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else self.gamma
            )
            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(
                replay_data.observations
            )
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(
                    self.log_ent_coef
                    * (log_prob + self.target_entropy).detach()
                ).mean()
                ent_coef_losses.append(float(ent_coef_loss.item()))
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(float(ent_coef.item()))

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                next_q_values = torch.cat(
                    self.critic_target(
                        replay_data.next_observations, next_actions
                    ),
                    dim=1,
                )
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (
                    (1 - replay_data.dones) * discounts * next_q_values
                )

            current_q_values = self.critic(
                replay_data.observations, replay_data.actions
            )
            critic_loss = 0.5 * sum(
                torch.nn.functional.mse_loss(current_q, target_q_values)
                for current_q in current_q_values
            )
            critic_losses.append(float(critic_loss.item()))
            self.critic.optimizer.zero_grad()
            if torch.isfinite(critic_loss):
                critic_loss.backward()
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), max_norm=10.0
                )
                critic_grads_finite = bool(torch.isfinite(critic_grad_norm)) and all(
                    parameter.grad is None
                    or bool(torch.isfinite(parameter.grad).all())
                    for parameter in self.critic.parameters()
                )
                if critic_grads_finite:
                    self.critic.optimizer.step()
                else:
                    skipped_critic_updates += 1
                    self.critic.optimizer.zero_grad(set_to_none=True)
            else:
                skipped_critic_updates += 1

            q_values_pi = torch.cat(
                self.critic(replay_data.observations, actions_pi), dim=1
            )
            min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
            mean_abs_q = min_qf_pi.detach().abs().mean()
            if bool(getattr(self, "_normalize_q_actor_loss", False)):
                # TD3+BC: la escala absoluta del crítico no puede borrar la
                # imitación. El gradiente conserva la dirección de Q, pero su
                # contribución media queda acotada y comparable con BC.
                q_actor_scale = float(
                    getattr(self, "_q_objective_scale", 1.0)
                ) / torch.clamp(mean_abs_q, min=1.0)
            else:
                q_actor_scale = torch.ones_like(mean_abs_q)
            actor_sac_loss = (
                ent_coef * log_prob - q_actor_scale * min_qf_pi
            ).mean()
            actor_loss = actor_sac_loss
            q_abs_means.append(float(mean_abs_q.item()))
            q_actor_scales.append(float(q_actor_scale.item()))

            if demo_enabled:
                indices = self._sample_demo_indices()
                demo_observations = self._demo_observations[indices].copy()
                if self._vec_normalize_env is not None:
                    demo_observations = self._vec_normalize_env.normalize_obs(
                        demo_observations
                    )
                demo_obs_tensor = torch.as_tensor(
                    demo_observations, dtype=torch.float32, device=self.device
                )
                demo_action_tensor = torch.as_tensor(
                    self._demo_actions[indices],
                    dtype=torch.float32,
                    device=self.device,
                )
                demo_prediction = self.actor(
                    demo_obs_tensor, deterministic=True
                )
                phase_ids = getattr(self, "_demo_phase_ids", None)
                action_weights = getattr(
                    self, "_demo_action_weights_by_phase", None
                )
                if phase_ids is not None and action_weights is not None:
                    sampled_phases = np.asarray(
                        phase_ids[indices], dtype=np.int64
                    )
                    demo_weight_tensor = torch.as_tensor(
                        action_weights[sampled_phases],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    squared_error = torch.square(
                        demo_prediction - demo_action_tensor
                    )
                    demo_loss = torch.sum(
                        demo_weight_tensor * squared_error
                    ) / torch.clamp(torch.sum(demo_weight_tensor), min=1.0)
                else:
                    demo_loss = torch.nn.functional.mse_loss(
                        demo_prediction, demo_action_tensor
                    )
                actor_loss = actor_loss + self._demo_loss_weight * demo_loss
                demo_losses.append(float(demo_loss.item()))

            actor_sac_losses.append(float(actor_sac_loss.item()))
            actor_losses.append(float(actor_loss.item()))
            self.actor.optimizer.zero_grad()
            actor_update_number = int(self._n_updates + gradient_step)
            actor_is_warm = actor_update_number >= int(
                getattr(self, "_actor_warmup_updates", 0)
            )
            if not actor_is_warm:
                frozen_actor_updates += 1
            elif torch.isfinite(actor_loss):
                actor_loss.backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), max_norm=5.0
                )
                actor_grads_finite = bool(torch.isfinite(actor_grad_norm)) and all(
                    parameter.grad is None
                    or bool(torch.isfinite(parameter.grad).all())
                    for parameter in self.actor.parameters()
                )
                if actor_grads_finite:
                    self.actor.optimizer.step()
                else:
                    skipped_actor_updates += 1
                    self.actor.optimizer.zero_grad(set_to_none=True)
            else:
                skipped_actor_updates += 1

            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )
                polyak_update(
                    self.batch_norm_stats, self.batch_norm_stats_target, 1.0
                )

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", float(np.mean(ent_coefs)))
        self.logger.record("train/actor_loss", float(np.mean(actor_losses)))
        self.logger.record(
            "train/actor_sac_loss", float(np.mean(actor_sac_losses))
        )
        self.logger.record("train/critic_loss", float(np.mean(critic_losses)))
        self.logger.record("train/q_abs_mean", float(np.mean(q_abs_means)))
        self.logger.record(
            "train/q_actor_scale", float(np.mean(q_actor_scales))
        )
        if demo_losses:
            self.logger.record("train/demo_bc_loss", float(np.mean(demo_losses)))
        self.logger.record(
            "train/skipped_critic_updates", int(skipped_critic_updates)
        )
        self.logger.record(
            "train/skipped_actor_updates", int(skipped_actor_updates)
        )
        self.logger.record(
            "train/frozen_actor_updates", int(frozen_actor_updates)
        )
        if ent_coef_losses:
            self.logger.record(
                "train/ent_coef_loss", float(np.mean(ent_coef_losses))
            )


__all__ = ["DemoRegularizedSAC"]

import numpy as np



class VXSAC(DemoRegularizedSAC):
    def _excluded_save_params(self):
        # The dataset lives in NPZ, not inside the deployed actor/critics file.
        return super()._excluded_save_params() + [k for k in self.__dict__ if k.startswith('_demo_')]

    def _sample_demo_indices(self):
        groups = getattr(self, '_demo_group_indices', None)
        if not groups:
            return super()._sample_demo_indices()
        phases = self._demo_rng.integers(0, 3, self._demo_batch_size)
        result = np.empty(self._demo_batch_size, dtype=np.int64)
        for phase in (0, 1, 2):
            names = [n for n in groups if n.startswith(f'p{phase}:')]
            if not names:
                return super()._sample_demo_indices()
            positions = np.flatnonzero(phases == phase)
            choices = self._demo_rng.integers(len(names), size=positions.size)
            for group_id, name in enumerate(names):
                selected = positions[choices == group_id]
                result[selected] = self._demo_rng.choice(groups[name], size=selected.size)
        return result

#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/vx_r2')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import gymnasium as gym
import mujoco
import numpy as np



ROOT = RESOURCE_ROOT
MASSES = (.5, .75, 1., 1.25, 1.5)
DEFAULT_MODEL = ROOT/'models/zenno_vx_r1_run2/best/model.zip'








def canonical_cases(maps=(1, 2, 3, 4), masses=MASSES, reps=2, seed=20260711,
                    profile='robust'):
    """Índice de masa canónico: filtrar masas no cambia las semillas."""
    if reps < 1:
        raise ValueError('reps debe ser positivo')
    result = []
    for map_id in maps:
        if map_id not in (1, 2, 3, 4):
            raise ValueError('Mapa fuera de RUTAS_SAC')
        for mass in masses:
            matches = [i for i, value in enumerate(MASSES) if np.isclose(mass, value, atol=1e-12)]
            if len(matches) != 1:
                raise ValueError(f'Masa no canónica: {mass}')
            for rep in range(reps):
                result.append(dict(map=int(map_id), mass=float(mass), repetition=rep,
                                   seed=int(seed+map_id*100000+matches[0]*1000+rep),
                                   profile=profile))
    return result


class _SpaceOnlyEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (124,), dtype=np.float32)
    action_space = gym.spaces.Box(-1., 1., (7,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(124, dtype=np.float32), {}

    def step(self, action):
        raise RuntimeError('Ambiente exclusivo para cargar las estadísticas; no simula.')




def policy_action(model, norm):
    return lambda obs, env: model.predict(norm.normalize_obs(np.asarray(obs).copy()),
                                          deterministic=True)[0]


def make_env():
    """Compatibility for archived physical-collection utilities, not SAC reset."""
    env = Environment(render_mode=None, debug_rewards=False, p1_budget_steps=4500)
    env.CUBE_XY_NOISE_RANGE = .01
    env.CUBE_YAW_NOISE_RANGE_RAD = .10
    return env


def configure_case(env, case, robust):
    env.FIXED_MAP_ID, mass = case
    env.set_eval_mode(True)
    env.set_eval_mass(mass)
    env.CUBE_XY_NOISE_RANGE = .01 if robust else 0.
    env.CUBE_YAW_NOISE_RANGE_RAD = .10 if robust else 0.


def predict(model, norm, obs):
    return model.predict(norm.normalize_obs(np.asarray(obs).copy()), deterministic=True)[0]


def handoff_state(env, source, target):
    return dict(source=source, target=target,
        arm_qpos=np.asarray(env._get_arm_qpos()).tolist(),
        arm_target=np.asarray(env.arm_target).tolist(),
        cube_xyz=np.asarray(env._cube_pos()).tolist(),
        carry_error_m=float(np.linalg.norm(env._site_xpos('setpoint')-env._cube_pos())),
        relative_speed_m_s=float(np.linalg.norm(env._vx_relative_velocity)),
        cube_speed_m_s=float(np.linalg.norm(env._cube_vel_vec())), gripping=bool(env.gripping))




def initial_state(env, obs):
    """Estado físico y memorias iniciales, excluyendo el interruptor ON/OFF."""
    data_fields = ('qpos', 'qvel', 'ctrl', 'act', 'qacc_warmstart', 'mocap_pos', 'mocap_quat',
                   'qfrc_applied', 'xfrc_applied', 'eq_active')
    model_fields = ('body_mass', 'body_pos', 'body_inertia', 'geom_friction')
    result = {name:np.asarray(getattr(env.data, name)).tolist()
              for name in data_fields if hasattr(env.data, name)}
    result.update({'model_'+name:np.asarray(getattr(env.model, name)).tolist() for name in model_fields})
    result.update(observation=np.asarray(obs).tolist(), arm_target=np.asarray(env.arm_target).tolist(),
                  map_id=int(env.map_id), cube_mass_kg=float(env._cube_mass),
                  phase=int(env.phase_state), phase_steps=int(env._phase_steps), time_s=float(env.data.time))
    for name in ('_Accel_lin_des', '_Accel_ang_des', '_e_vel1', '_e_vel2', '_e_omega1', '_e_omega2',
                 '_v_des_f', '_omega_des_f', '_omega_des_prev', '_vx_filter', '_vx_delta'):
        value = getattr(env, name, None)
        result[name] = None if value is None else np.asarray(value).tolist()
    return result


def state_hash(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def _rms(values):
    return float(np.sqrt(np.mean(values))) if values else None


def evaluate_episode(action_fn, env, case, variant='on'):
    if variant not in ('on', 'off'):
        raise ValueError('Variante inválida')
    env.residual_enabled = variant == 'on'
    env.set_eval_mode(True)
    env.FIXED_MAP_ID = int(case['map'])
    env.set_eval_mass(float(case['mass']))
    robust = case.get('profile', 'robust') == 'robust'
    env.CUBE_XY_NOISE_RANGE = .01 if robust else 0.
    env.CUBE_YAW_NOISE_RANGE_RAD = .10 if robust else 0.
    obs, _ = env.reset(seed=int(case['seed']))
    initial = initial_state(env, obs)
    energy = np.zeros((3, 2))
    seconds = np.zeros(3)
    cte, yaw, carry, speed, residual = [], [], [], [], []
    near_saturation, exact_saturation, active_seconds = [], [], 0.
    snapshots, handoffs = [], []
    tracking_seconds = terminal_seconds = 0.
    first_terminal = first_nav_done = None
    p1_seen_steps = 0
    previous_phase = 0
    info = {}
    terminated = truncated = False
    pending = {3000, 4500, int(env.MAX_STEPS_1)}
    for step in range(int(env.max_episode_steps)):
        previous_phase = int(env.phase_state)
        action = np.asarray(action_fn(obs, env), dtype=np.float32).reshape(-1)
        if action.shape != (7,) or not np.all(np.isfinite(action)) or np.any(np.abs(action) > 1.00001):
            raise ValueError('El callback debe devolver una acción finita 7D en [-1,1]')
        obs, _, terminated, truncated, info = env.step(action)
        seconds[previous_phase] += float(env.dt)
        energy[previous_phase] += env._diag_energy
        if variant == 'off' and np.max(np.abs(env._diag_delta_nm), initial=0.) > 1e-10:
            raise AssertionError('OFF produjo una corrección de torque: instrumentación incoherente')
        if previous_phase == 1:
            p1_seen_steps += 1
            diag = info.get('p1_diagnostics') or env.phase1_diagnostics()
            carry.append(float(info.get('vx_carry_error_m', 0.))**2)
            speed.append(float(info.get('vx_relative_speed_m_s', 0.))**2)
            if env._diag_tracking:
                tracking_seconds += env.dt
                if env._diag_pose is not None:
                    cte.append(float(env._diag_pose['cte_signed_m'])**2)
                    yaw.append(float(env._diag_pose['heading_error_rad'])**2)
                residual.append(float(np.mean(env._diag_delta_nm**2)))
                ratio = np.abs(env._diag_wheel_commands)/env._diag_wheel_limits
                near_saturation.append(float(np.mean(ratio >= .996)))
                exact_saturation.append(float(np.mean(ratio >= 1.-1e-9)))
                active_seconds += env.dt*bool(getattr(env, '_vx_active', False))
            endpoint = diag.get('endpoint_m', diag.get('endpoint_distance_m'))
            terminal = bool(diag.get('terminal_region', False))
            if endpoint is not None and np.isfinite(endpoint):
                terminal |= endpoint <= env.PURE_PURSUIT_TERMINAL_POSE_DIST_M
            if terminal:
                terminal_seconds += env.dt
                if first_terminal is None:
                    first_terminal = p1_seen_steps*env.dt
            if diag.get('checks', {}).get('nav_done', False) and first_nav_done is None:
                first_nav_done = p1_seen_steps*env.dt
            if p1_seen_steps in pending or terminated or truncated or int(env.phase_state) != 1:
                snapshots.append(dict(p1_step=p1_seen_steps, p1_time_s=p1_seen_steps*env.dt,
                                      **diag))
        if int(env.phase_state) != previous_phase:
            handoffs.append(dict(source=previous_phase, target=int(env.phase_state),
                                 episode_time_s=float(np.sum(seconds)),
                                 carry_error_m=float(np.linalg.norm(env._site_xpos('setpoint')-env._cube_pos())),
                                 relative_speed_m_s=float(np.linalg.norm(env._vx_relative_velocity)),
                                 gripping=bool(env.gripping), arm_target=np.asarray(env.arm_target).tolist()))
        if terminated or truncated:
            break
    reason = info.get('truncation_reason') or info.get('termination_reason')
    if not (terminated or truncated):
        reason = 'evaluator_total_budget_exhausted'
    phases = [bool(info.get(f'phase{i}_success', False)) for i in range(3)]
    row = dict(**case, variant=variant, p1_budget_steps=int(env.MAX_STEPS_1),
               success=bool(info.get('strict_success', False)), phases=phases,
               ended=bool(terminated or truncated), terminal_phase=previous_phase, reason=reason,
               time_s=seconds.tolist(), energy_arm_j=energy[:,0].tolist(), energy_base_j=energy[:,1].tolist(),
               full_time_s=float(seconds.sum()), full_mechanical_work_j=float(energy.sum()),
               cte_rms_m=_rms(cte), yaw_rms_rad=_rms(yaw), carry_rms_m=_rms(carry),
               relative_speed_rms_m_s=_rms(speed), residual_rms_nm=_rms(residual),
               exact_saturation_fraction=float(np.mean(exact_saturation)) if exact_saturation else None,
               near_saturation_fraction=float(np.mean(near_saturation)) if near_saturation else None,
               p1_tracking_time_s=float(tracking_seconds), p1_terminal_time_s=float(terminal_seconds),
               residual_active_time_s=float(active_seconds), first_terminal_time_s=first_terminal,
               first_nav_done_time_s=first_nav_done, p1_snapshots=snapshots, handoffs=handoffs,
               initial_state_sha256=state_hash(initial), p1_reached=bool(phases[0]),
               p1_success_within_30s=bool(phases[1] and seconds[1] <= 30.+1e-8),
               p1_success_within_45s=bool(phases[1] and seconds[1] <= 45.+1e-8))
    return row, initial


def summarize(rows):
    result = []
    for budget, variant in sorted({(r['p1_budget_steps'], r['variant']) for r in rows}):
        group = [r for r in rows if r['p1_budget_steps'] == budget and r['variant'] == variant]
        reached = [r for r in group if r['p1_reached']]
        complete = [r for r in group if r['success']]
        summary = dict(p1_budget_steps=budget, variant=variant, episodes=len(group),
                       strict_success_rate=float(np.mean([r['success'] for r in group])),
                       phase_success_rates=np.mean([r['phases'] for r in group], axis=0).tolist(),
                       reached_p1=len(reached),
                       conditional_p1_success_rate=float(np.mean([r['phases'][1] for r in reached])) if reached else None,
                       conditional_p1_success_within_30s=float(np.mean([r['p1_success_within_30s'] for r in reached])) if reached else None,
                       conditional_p1_success_within_45s=float(np.mean([r['p1_success_within_45s'] for r in reached])) if reached else None,
                       failure_reasons=dict(Counter(str(r['reason']) for r in group if not r['success'])),
                       complete_success_cohort_n=len(complete))
        for metric in ('full_time_s', 'full_mechanical_work_j', 'cte_rms_m', 'yaw_rms_rad',
                       'carry_rms_m', 'relative_speed_rms_m_s', 'residual_rms_nm'):
            values = [r[metric] for r in complete if r[metric] is not None]
            summary['success_only_mean_'+metric] = float(np.mean(values)) if values else None
        result.append(summary)
    paired = []
    index = {(r['p1_budget_steps'], r.get('profile', 'robust'), r['map'], r['mass'], r['seed'], r['variant']):r for r in rows}
    for budget in sorted({r['p1_budget_steps'] for r in rows}):
        pairs = []
        for r in rows:
            if r['p1_budget_steps'] != budget or r['variant'] != 'on':
                continue
            off = index.get((budget, r.get('profile', 'robust'), r['map'], r['mass'], r['seed'], 'off'))
            if off is not None:
                pairs.append((r, off))
        both = [(on, off) for on, off in pairs if on['success'] and off['success']]
        entry = dict(p1_budget_steps=budget, pairs=len(pairs), both_success=len(both),
                     only_on_success=sum(on['success'] and not off['success'] for on, off in pairs),
                     only_off_success=sum(off['success'] and not on['success'] for on, off in pairs))
        for metric in ('full_time_s', 'full_mechanical_work_j', 'cte_rms_m', 'yaw_rms_rad',
                       'carry_rms_m', 'relative_speed_rms_m_s'):
            values = [on[metric]-off[metric] for on, off in both if on[metric] is not None and off[metric] is not None]
            entry['paired_success_delta_on_minus_off_'+metric] = float(np.mean(values)) if values else None
        paired.append(entry)
    return dict(groups=result, paired=paired,
                interpretation='No se calcula un ganador escalar. Costes incluyen home; trabajo mecánico absoluto, no consumo eléctrico. '
                               'Las medias success_only no son comparables causalmente sin usar paired. '
                               'Aumentar presupuesto también cambia la observación de tiempo restante durante P1.')


def evaluate_cases(action_fn, cases, output_dir, budgets=(3000, 4500, 6000),
                   variants=('on', 'off'), progress=None, metadata=None):
    """Retorna {groups, paired, rows}; persiste cada episodio terminado.

    El directorio debe ser nuevo. Perfiles nominal/robust son válidos; no hay
    autoreset vectorizado ni resets sintéticos P1/P2. El emparejamiento incluye
    todas las matrices físicas, controles, memoria PID y observación inicial.
    """
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f'No sobrescribir una evaluación: {output}')
    cases = [dict(case) for case in cases]
    if not cases or not budgets or not variants:
        raise ValueError('Casos, presupuestos y variantes no pueden estar vacíos')
    if len(set(budgets)) != len(budgets) or len(set(variants)) != len(variants):
        raise ValueError('Presupuestos o variantes duplicados')
    if any(v not in ('on', 'off') for v in variants):
        raise ValueError('Variantes permitidas: on/off')
    keys = [(c.get('profile', 'robust'), c['map'], c['mass'], c['seed']) for c in cases]
    if len(set(keys)) != len(keys):
        raise ValueError('Casos duplicados')
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='RUNNING', revision=Environment.REVISION, budgets=list(budgets),
                    variants=list(variants), cases=cases, expected_episodes=len(cases)*len(budgets)*len(variants),
                    noise=dict(robust_xy_m=.01, robust_yaw_rad=.10), hashes=experiment_hashes(),
                    scope='diagnostic/development, no holdout claim', metadata=metadata or {})
    dump(output/'manifest.json', manifest)
    rows = []
    try:
        for budget in budgets:
            env = DiagnosticEnv(render_mode=None, debug_rewards=False, p1_budget_steps=int(budget))
            try:
                for index, case in enumerate(cases):
                    expected = None
                    for variant in (variants if index % 2 == 0 else tuple(reversed(variants))):
                        row, initial = evaluate_episode(action_fn, env, case, variant)
                        if expected is not None and row['initial_state_sha256'] != expected:
                            raise AssertionError(f'Reset ON/OFF no idéntico: {case}, presupuesto={budget}')
                        row['initial_pair_verified'] = expected is not None
                        expected = row['initial_state_sha256']
                        rows.append(row)
                        with (output/'episodes.jsonl').open('a') as file:
                            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                        state_path = output/'initial_states'/f'{budget}_{index:04d}.json'
                        if not state_path.exists():
                            dump(state_path, initial)
                        print('[R2-EVAL] '+json.dumps(dict(completed=len(rows), total=manifest['expected_episodes'],
                              budget=budget, variant=variant, map=case['map'], mass=case['mass'], seed=case['seed'],
                              success=row['success'], phases=row['phases'], reason=row['reason'])), flush=True)
                        if progress is not None:
                            progress(len(rows), row)
                    if len(variants) == 2:
                        rows[-2]['initial_pair_verified'] = rows[-1]['initial_pair_verified'] = True
                        with (output/'verified_pairs.jsonl').open('a') as file:
                            file.write(json.dumps(dict(p1_budget_steps=budget, case=case,
                                                       initial_state_sha256=expected))+'\n')
            finally:
                env.close()
        result = summarize(rows)
        result['rows'] = rows
        dump(output/'summary.json', result)
        columns = [key for key, value in rows[0].items() if not isinstance(value, (list, dict))]
        with (output/'episodes.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        manifest['status'] = 'COMPLETE'
        manifest['completed_episodes'] = len(rows)
        dump(output/'manifest.json', manifest)
        return result
    except BaseException as error:
        manifest.update(status='INTERRUPTED', completed_episodes=len(rows), error=repr(error))
        dump(output/'manifest.json', manifest)
        raise





class DiagnosticTorque(ZennoVXR11):
    """Sólo instrumentación: integra |fuerza*velocidad| en cada paso físico."""

    def __init__(self, **kwargs):
        self._diag_energy = np.zeros(2)
        self._diag_pose_metric = self._diag_pose_path = None
        super().__init__(**kwargs)

    def reset(self, **kwargs):
        self._diag_pose_metric = self._diag_pose_path = None
        self._diag_energy = np.zeros(2)
        return super().reset(**kwargs)

    def step(self, action):
        self._diag_nominal = {}
        return super().step(action)

    def _apply_wheel_torque_with_residual(self, left, right, u_lin, u_ang):
        super()._apply_wheel_torque_with_residual(left, right, u_lin, u_ang)
        self._diag_nominal = {
            int(aid):float(np.clip(raw, -self._actuator_tau_limit(aid), self._actuator_tau_limit(aid)))
            for raw, ids in ((left, self.wheel_left_ids), (right, self.wheel_right_ids)) for aid in ids}

    def _stop_base(self):
        super()._stop_base()
        self._diag_nominal = {int(aid):0. for aid in self.wheel_left_ids+self.wheel_right_ids}

    def do_simulation(self, ctrl, n_frames):
        if int(n_frames) != int(self.frame_skip) or int(n_frames) != 5:
            raise ValueError('La comparación exige cinco pasos físicos por decisión')
        ctrl = np.asarray(ctrl, dtype=np.float64)
        if ctrl.shape != (self.model.nu,):
            raise ValueError('Forma incorrecta del control')
        self.data.ctrl[:] = ctrl
        self._diag_tracking = bool(
            getattr(self, '_last_base_mode', '') == 'tracking'
            and getattr(self, '_plan_ready', False)
            and not any(getattr(self, name, False) for name in
                        ('_reverse_active', '_initial_aligning', '_final_aligning',
                         '_final_radial_approach', '_nav_done')))
        self._diag_pose = None
        if self._diag_tracking:
            path = np.asarray(self._waypoints, dtype=float)
            if self._diag_pose_path is None or not np.array_equal(path, self._diag_pose_path):
                self._diag_pose_metric = RoutePoseMetric(path)
                self._diag_pose_path = path.copy()
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'chassis_frame')
            if sid < 0:
                raise RuntimeError('Falta chassis_frame para medir yaw geométrico')
            rotation = self.data.site_xmat[sid].reshape(3,3)
            yaw = float(np.arctan2(rotation[1,0], rotation[0,0]))
            self._diag_pose = self._diag_pose_metric.sample(self.data.site_xpos[sid,:2], yaw)
        wheel_ids = self.wheel_left_ids+self.wheel_right_ids
        self._diag_delta_nm = np.asarray([ctrl[aid]-self._diag_nominal.get(aid, ctrl[aid])
                                          for aid in wheel_ids])
        self._diag_wheel_commands = ctrl[wheel_ids].copy()
        self._diag_wheel_limits = np.asarray([self._actuator_tau_limit(aid) for aid in wheel_ids])
        self._diag_energy = np.zeros(2)
        for _ in range(int(n_frames)):
            mujoco.mj_step(self.model, self.data)
            for group, ids in enumerate((self.arm_act_ids[:5], wheel_ids)):
                force = self.data.actuator_force[ids]
                velocity = self.data.actuator_velocity[ids]
                self._diag_energy[group] += float(np.sum(np.abs(force*velocity)))*self.model.opt.timestep
class DiagnosticPP(ZennoVXR2):
    """Sólo instrumentación: integra |fuerza*velocidad| en cada paso físico."""

    def __init__(self, **kwargs):
        self._diag_energy = np.zeros(2)
        self._diag_pose_metric = self._diag_pose_path = None
        super().__init__(**kwargs)

    def reset(self, **kwargs):
        self._diag_pose_metric = self._diag_pose_path = None
        self._diag_energy = np.zeros(2)
        return super().reset(**kwargs)

    def step(self, action):
        self._diag_nominal = {}
        return super().step(action)

    def _control_base_vel(self, *args, **kwargs):
        super()._control_base_vel(*args, **kwargs)
        # Baseline of the local PID counterfactual, NOT the actual corrected
        # PID input torque passed to the low-level wheel writer.
        self._diag_nominal = self._r2_nominal_wheel_commands.copy()

    def _stop_base(self):
        super()._stop_base()
        self._diag_nominal = {int(aid):0. for aid in self.wheel_left_ids+self.wheel_right_ids}

    def do_simulation(self, ctrl, n_frames):
        if int(n_frames) != int(self.frame_skip) or int(n_frames) != 5:
            raise ValueError('La comparación exige cinco pasos físicos por decisión')
        ctrl = np.asarray(ctrl, dtype=np.float64)
        if ctrl.shape != (self.model.nu,):
            raise ValueError('Forma incorrecta del control')
        self.data.ctrl[:] = ctrl
        self._diag_tracking = bool(
            getattr(self, '_last_base_mode', '') == 'tracking'
            and getattr(self, '_plan_ready', False)
            and not any(getattr(self, name, False) for name in
                        ('_reverse_active', '_initial_aligning', '_final_aligning',
                         '_final_radial_approach', '_nav_done')))
        self._diag_pose = None
        if self._diag_tracking:
            path = np.asarray(self._waypoints, dtype=float)
            if self._diag_pose_path is None or not np.array_equal(path, self._diag_pose_path):
                self._diag_pose_metric = RoutePoseMetric(path)
                self._diag_pose_path = path.copy()
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'chassis_frame')
            if sid < 0:
                raise RuntimeError('Falta chassis_frame para medir yaw geométrico')
            rotation = self.data.site_xmat[sid].reshape(3,3)
            yaw = float(np.arctan2(rotation[1,0], rotation[0,0]))
            self._diag_pose = self._diag_pose_metric.sample(self.data.site_xpos[sid,:2], yaw)
        wheel_ids = self.wheel_left_ids+self.wheel_right_ids
        self._diag_delta_nm = np.asarray([ctrl[aid]-self._diag_nominal.get(aid, ctrl[aid])
                                          for aid in wheel_ids])
        self._diag_wheel_commands = ctrl[wheel_ids].copy()
        self._diag_wheel_limits = np.asarray([self._actuator_tau_limit(aid) for aid in wheel_ids])
        self._diag_energy = np.zeros(2)
        for _ in range(int(n_frames)):
            mujoco.mj_step(self.model, self.data)
            for group, ids in enumerate((self.arm_act_ids[:5], wheel_ids)):
                force = self.data.actuator_force[ids]
                velocity = self.data.actuator_velocity[ids]
                self._diag_energy[group] += float(np.sum(np.abs(force*velocity)))*self.model.opt.timestep

def experiment_hashes():
    return dict(sources={p.name:sha256(p) for p in (CODE_ROOT/'TRAIN_VX.py',CODE_ROOT/'MM_DATA_VXR1.py',CODE_ROOT/'MM_DATA_VXR2.py')},
                routes={p.name:sha256(p) for p in (ROOT/'RUTAS_SAC').glob('*.json')})

import hashlib
import json
from pathlib import Path

import numpy as np




ARM_MASK_REVISION = 'R2_FROZEN_ARM_SYNTHETIC_RESIDUAL_CHANNELS_ZERO_V1'
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
            elif FAMILY == 'r2':
                # R1 base actions are TORQUE labels, not PP reference labels.
                action[5:] = 0.
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
    """Carga SOLO un profesor R2 2D confiable y local; replay de una fila.

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
    if contract.get('revision') != Environment.REVISION or config.get('revision') != Environment.REVISION:
        raise ValueError('El checkpoint no corresponde a la semántica R2 de referencias Pure Pursuit.')
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
        if _sha(ROOT/'RUTAS_SAC'/name) != expected:
            raise ValueError(f'Ruta diferente respecto del profesor: {name}')
    for name, expected in config.get('source_files', {}).items():
        if source_digest(name) != expected:
            raise ValueError(f'Código modificado respecto del profesor R2: {name}')
    model = SAC.load(path, device=device, custom_objects={'buffer_size': 1})
    if model.observation_space.shape != (124,) or model.action_space.shape != (2,):
        raise ValueError('El archivo SAC no cumple el contrato 124D/2D.')
    model.policy.set_training_mode(False)
    for parameter in model.policy.parameters():
        parameter.requires_grad_(False)
    return model

#!/usr/bin/env python3
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/vx_r2')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure as configure_logger



ROOT = RESOURCE_ROOT
MASSES = (.5, .75, 1., 1.25, 1.5)
SOURCE = ROOT/'models/zenno_vx_r1_run2/best/model.zip'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def load_source(path, device):
    """Trusted local SB3 bundle; never allocate its historical replay buffer."""
    path = Path(path).resolve()
    config = json.loads((path.parent.parent/'config.json').read_text())
    if config.get('revision') != 'VX_R1_OVERLAP_CURRICULUM_124D_7D_DELTA008_UNIFORM_COSTS':
        raise ValueError('Se requiere el checkpoint R1 documentado, no R2 ni otra semántica.')
    if config['hashes']['zenno_vx_r1.py'] != SOURCE_HASHES['zenno_vx_r1.py']:
        raise ValueError('La fuente R1 fue modificada respecto de su checkpoint.')
    for name, digest in config['route_hashes'].items():
        if sha(ROOT/'RUTAS_SAC'/name) != digest:
            raise ValueError(f'Ruta modificada: {name}')
    with (path.parent/'vecnormalize.pkl').open('rb') as file:
        norm = pickle.load(file)
    # SB3 requires set_venv after unpickling, including for saving again.
    # This light environment never advances physics or updates statistics.
    from stable_baselines3.common.vec_env import DummyVecEnv

    norm.set_venv(DummyVecEnv([_SpaceOnlyEnv]))
    norm.training = False
    norm.norm_reward = False
    model = SAC.load(path, device=device, custom_objects={'buffer_size': 1})
    if model.observation_space.shape != (124,) or model.action_space.shape != (7,):
        raise ValueError('El actor fuente debe tener contrato 124D/7D.')
    model.policy.set_training_mode(False)
    for parameter in model.policy.parameters():
        parameter.requires_grad_(False)
    return model, norm


class P1ResidualEnv(gym.Env):
    """Inicios reales de P0; termina al entregar a P2 o agotar la tarea P1.

    Observaciones externas normalizadas UNA vez con las estadísticas congeladas
    de R1. Replay almacena estas observaciones y las acciones residuales ejecutadas
    a través del filtro/limitador del ambiente, nunca etiquetas contrafactuales.
    """
    metadata = {'render_modes': []}

    def __init__(self, source, norm, *, budget=4500, seed=13110000, log=None):
        super().__init__()
        self.source, self.norm = source, norm
        self.env = Environment(render_mode=None, debug_rewards=False, p1_budget_steps=budget)
        self.env.set_eval_mode(True)
        self.env.CUBE_XY_NOISE_RANGE = .01
        self.env.CUBE_YAW_NOISE_RANGE_RAD = .1
        self.action_space = gym.spaces.Box(-1., 1., (2,), dtype=np.float32)
        clip = float(norm.clip_obs)
        self.observation_space = gym.spaces.Box(-clip, clip, (124,), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.counter = 0
        self.log = Path(log) if log else None
        self._obs = None

    def normalize(self, obs):
        return self.norm.normalize_obs(np.asarray(obs, dtype=np.float32).copy()).astype(np.float32)

    def arm_action(self, obs):
        protected = arm_observation(obs, int(self.env.phase_state))
        return np.asarray(self.source.predict(self.normalize(protected), deterministic=True)[0], dtype=np.float32).copy()

    def record(self, value):
        if self.log:
            with self.log.open('a') as file:
                file.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        for _ in range(40):
            index = self.counter % 20
            self.counter += 1
            mid, mass = 1+index//5, MASSES[index % 5]
            # Disjoint from diagnostic (~20M), development (61M) and test (71M).
            episode_seed = int(self.rng.integers(100_000_000, 200_000_000))
            self.env.FIXED_MAP_ID = mid
            self.env.set_eval_mass(mass)
            obs, _ = self.env.reset(seed=episode_seed)
            steps = 0
            while int(self.env.phase_state) == 0:
                obs, _, term, trunc, info = self.env.step(self.arm_action(obs))
                steps += 1
                if term or trunc:
                    break
            self.case = dict(map=mid, mass=mass, seed=episode_seed, p0_steps=steps)
            if int(self.env.phase_state) == 1 and not (term or trunc):
                self._obs = obs
                self.reward_sum = 0.
                self.steps = 0
                self.squares = np.zeros(3)
                self.reward_terms = {}
                self.record(dict(event='P0_TO_P1', **self.case))
                return self.normalize(obs), dict(real_p0_handoff=True, **self.case)
            self.record(dict(event='P0_FAILED_NOT_HIDDEN', **self.case,
                             reason=info.get('termination_reason') or info.get('truncation_reason')))
        raise RuntimeError('40 reinicios P0 sin alcanzar P1; no se fabrican estados iniciales.')

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError('El profesor residual requiere dos acciones finitas.')
        action = np.clip(action, -1., 1.)
        full = self.arm_action(self._obs)
        full[5:] = action
        self._obs, reward, term, trunc, info = self.env.step(full)
        self.steps += 1
        self.reward_sum += reward
        self.squares += [float(np.mean(action**2)), float(np.mean(self.env._vx_filter**2)),
                         float(np.mean((25*self.env._vx_delta)**2))]
        for key, value in info.items():
            if key.startswith('vx_') and isinstance(value, (float, np.floating)) and ('cost' in key or key in ('vx_progress', 'vx_terminal_progress', 'vx_drop', 'vx_arrival_bonus')):
                self.reward_terms[key] = self.reward_terms.get(key, 0.)+float(value)
        handoff = bool(info.get('phase1_success')) and int(self.env.phase_state) == 2
        done = bool(term or trunc or handoff)
        if done:
            self.record(dict(event='P1_EPISODE', **self.case, success=handoff,
                steps=self.steps, return_p1=self.reward_sum,
                request_filter_torque_rms=np.sqrt(self.squares/max(1, self.steps)).tolist(),
                reward_components=self.reward_terms, diagnostic=info.get('p1_diagnostics'),
                reason=info.get('termination_reason') or info.get('truncation_reason')))
        # Finite P1 task termination, NOT a rollout cut with bootstrap from P2.
        info['TimeLimit.truncated'] = False
        info['p1_specialist_success'] = handoff
        return self.normalize(self._obs), float(reward), done, False, info

    def close(self):
        self.env.close()


def evaluation_cases(seed=61_000_000, reps=1):
    return [dict(map=mid, mass=mass, seed=seed+mid*100000+i*1000+r, profile='robust')
            for mid in (1, 2, 3, 4) for i, mass in enumerate(MASSES) for r in range(reps)]


def run_teacher(args):
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requerida: no fallback CPU silencioso.')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    source, norm = load_source(args.source, args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(revision=Environment.REVISION, artifact_role='P1_2D_teacher_NOT_final_student',
        source_sha256=sha(args.source), normalizer_sha256=sha(args.source.parent/'vecnormalize.pkl'),
        source_arm='frozen closed-loop actor; no open-loop trajectory',
        arm_mask_revision=ARM_MASK_REVISION,
        arm_request_channel_mask='P1 only: raw[30,31,97,98,99,100,101]=0, public observation unchanged',
        residual_bc_weight=0., normalized_observations='external_frozen_once; replay normalized',
        budgets=[1000, args.p1_budget, 1000], training_starts='real source P0 rollouts',
        source_files={p.name: sha(p) for p in (CODE_ROOT/'MM_DATA_VXR1.py', CODE_ROOT/'MM_DATA_VXR2.py', Path(__file__).resolve())},
        route_hashes={p.name: sha(p) for p in (ROOT/'RUTAS_SAC').glob('*.json')},
        gpu=torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
        initial_log_std=-2., entropy_coef=args.ent_coef, critic_initialization='fresh',
        bootstrap_at_p1_deadline=False, final_student_created=False,
        residual_level='pure_pursuit_reference' if FAMILY == 'r2' else 'pid_torque', torque_residual_enabled=FAMILY == 'r1_1',
        reference_limits=dict(linear_fraction=.20, angular_fraction=.20,
            positive_speed_uplift_fraction_of_nominal=.20, lookahead_adaptive=False) if FAMILY == 'r2' else None,
        transferred_phases=['P0_actor', 'P2_actor', 'P1_auxiliary_arm_only'],
        imported_replay_transitions=0, source_base_actions_discarded=FAMILY == 'r2')
    dump(out/'config.json', config)

    def announce(stage, **details):
        record = dict(stage=stage, pid=os.getpid(), **details)
        dump(out/'status.json', record)
        with (out/'milestones.jsonl').open('a') as file:
            file.write(json.dumps(record, ensure_ascii=False)+'\n')
        print(stage+' '+json.dumps(details, ensure_ascii=False), flush=True)

    env = P1ResidualEnv(source, norm, budget=args.p1_budget, seed=args.seed,
                        log=out/'training_episodes.jsonl')
    model = SAC('MlpPolicy', env, device=args.device, seed=args.seed,
        learning_rate=args.lr, buffer_size=args.buffer_size, learning_starts=args.warmup,
        batch_size=256, tau=.005, gamma=.9995, train_freq=1, gradient_steps=1,
        ent_coef=args.ent_coef, target_entropy=-2.,
        policy_kwargs=dict(net_arch=[128,128]),
        replay_buffer_kwargs=dict(handle_timeout_termination=False), verbose=0)
    # SB3's non-gSDE actor does not initialize this Linear bias from log_std_init.
    with torch.no_grad():
        model.actor.mu.weight.zero_(); model.actor.mu.bias.zero_()
        model.actor.log_std.weight.zero_(); model.actor.log_std.bias.fill_(-2.)
    model.set_logger(configure_logger(str(out/'sac_logs'), ['csv']))

    def save(directory):
        directory.mkdir(parents=True, exist_ok=True)
        model.save(directory/'model.zip')
        norm.save(directory/'vecnormalize.pkl')
        dump(directory/'contract.json', dict(revision=Environment.REVISION,
             role='P1_2D_teacher_NOT_final_student', observation_dim=124, action_dim=2,
             p1_budget=args.p1_budget, source_actor_sha256=sha(args.source),
             normalizer_sha256=sha(directory/'vecnormalize.pkl'),
             updates=int(model._n_updates), steps=int(model.num_timesteps)))

    class Milestone(BaseCallback):
        def __init__(self, block):
            super().__init__(); self.block = block; self.next = model.num_timesteps+args.steps//2
        def _on_step(self):
            if self.num_timesteps >= self.next:
                save(out/'latest')
                with torch.no_grad():
                    observation = torch.as_tensor(env.normalize(env._obs)[None,:], device=self.model.device)
                    mean, log_std, _ = self.model.actor.get_action_dist_params(observation)
                    action_std = float(log_std.exp().mean())
                    action_mean = float(torch.tanh(mean).abs().mean())
                    alpha = float(self.model.log_ent_coef.exp()) if self.model.log_ent_coef is not None else float(self.model.ent_coef_tensor)
                announce('TRAIN_PROGRESS', block=self.block, steps=self.num_timesteps,
                         sac_updates=int(self.model._n_updates), gaussian_std=action_std,
                         deterministic_action_abs=action_mean, entropy_coefficient=alpha)
                self.next += max(1, args.steps//2)
            return True

    try:
        save(out/'initial')

        cases = evaluation_cases(reps=args.eval_reps)
        source_action = composed_action(source, norm)
        evaluation_metadata = dict(source_sha256=sha(args.source),
            normalizer_sha256=sha(args.source.parent/'vecnormalize.pkl'),
            auxiliary_arm_mask=config['arm_request_channel_mask'], role='P1_teacher_not_final_student')
        announce('BASELINE_EVAL', cases=len(cases), variants=['on', 'off'])
        evaluate_cases(source_action, cases, out/'source_development', budgets=(args.p1_budget,),
                       metadata=evaluation_metadata)
        for block in range(1, args.blocks+1):
            announce('TRAIN_START', block=block, additional_p1_steps=args.steps,
                     pure_sac=True, residual_bc=False)
            model.learn(args.steps, reset_num_timesteps=False, callback=Milestone(block), log_interval=1)
            save(out/f'block{block}')
            if not model._n_updates:
                raise RuntimeError('No hubo actualizaciones SAC; no se considera entrenamiento terminado.')
            announce('BLOCK_EVAL', block=block, sac_updates=int(model._n_updates))
            specialist_action = composed_action(source, norm, model)
            evaluate_cases(specialist_action, cases, out/f'block{block}_development',
                           budgets=(args.p1_budget,), metadata={**evaluation_metadata,
                            'teacher_sha256':sha(out/f'block{block}/model.zip'),
                            'updates':int(model._n_updates)})
        save(out/'last')
        # Persist replay separately, never embedded in inference checkpoints.
        model.save_replay_buffer(out/'replay_buffer.pkl')
        dump(out/'result.json', dict(status='pilot_complete', sac_updates=int(model._n_updates),
            p1_training_steps=int(model.num_timesteps), final_student_created=False,
            note='Comparar development ON/OFF y source antes de promover/distilar. No es holdout.'))
        announce('COMPLETE', sac_updates=int(model._n_updates), p1_training_steps=int(model.num_timesteps),
                 final_student_created=False)
    except BaseException as error:
        try:
            save(out/'interrupted')
        except BaseException as save_error:
            dump(out/'save_error.json', dict(error=repr(save_error)))
        announce('INTERRUPTED', error=repr(error), sac_updates=int(model._n_updates))
        raise
    finally:
        env.close()
        norm.close()


def teacher_cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--device', choices=('cuda','cpu'), default='cuda')
    parser.add_argument('--p1-budget', type=int, choices=(3000,4500,6000), default=4500)
    parser.add_argument('--steps', type=int, default=30000, help='Decisiones P1 por bloque; P0 no cuenta.')
    parser.add_argument('--blocks', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=2048)
    parser.add_argument('--buffer-size', type=int, default=60000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ent-coef', default='auto_0.01')
    parser.add_argument('--seed', type=int, default=13110000)
    parser.add_argument('--eval-reps', type=int, default=1)
    args = parser.parse_args(argv)
    args.output = args.output.resolve(); args.source = args.source.resolve()
    if args.steps <= args.warmup or args.blocks < 1 or args.buffer_size < 2048 or args.eval_reps < 1:
        parser.error('steps debe superar warmup y presupuestos deben ser positivos.')
    def interrupt(signum, frame):
        raise InterruptedError(f'Señal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    lock_path = ROOT/'models'/('.teacher_'+FAMILY+'.lock')
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Ya hay una corrida R2 activa.')
        run_teacher(args)



#!/usr/bin/env python3

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import signal

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/vx_single_student')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np


ROOT = RESOURCE_ROOT
SOURCE = ROOT / 'models/zenno_vx_r1_run2/best/model.zip'
FAMILIES = {'r1_1': ('MM_DATA_VXR1', 'ZennoVXR11'), 'r2': ('MM_DATA_VXR2', 'ZennoVXR2')}


def teacher_gate(path):
    """Only a checkpoint with SAC updates and its COMPLETED hashed evaluation.

    This establishes provenance, not teacher superiority; measured development
    success is recorded separately. Initial/random and interrupted checkpoints
    cannot be distilled simply because their weights exist.
    """
    path = Path(path).resolve()
    if path.is_dir():
        path = path / 'model.zip'
    contract = json.loads((path.parent / 'contract.json').read_text())
    if int(contract.get('updates', 0)) <= 0 or int(contract.get('steps', 0)) <= 0:
        raise ValueError('El profesor no tiene actualizaciones SAC completadas.')
    development = path.parent.parent / (path.parent.name + '_development')
    manifest = json.loads((development / 'manifest.json').read_text())
    if manifest.get('status') != 'COMPLETE':
        raise ValueError('La evaluación del profesor no terminó.')
    if manifest.get('metadata', {}).get('teacher_sha256') != sha256(path):
        raise ValueError('La evaluación no corresponde a este checkpoint del profesor.')
    summary = json.loads((development / 'summary.json').read_text())
    rows = summary.get('rows', [])
    if not rows or len(rows) != manifest.get('expected_episodes'):
        raise ValueError('La evaluación del profesor está incompleta.')
    if {row['variant'] for row in rows} != {'on', 'off'}:
        raise ValueError('El profesor necesita una ablación ON/OFF completa.')
    return dict(model=str(path), model_sha256=sha256(path),
                updates=int(contract['updates']), steps=int(contract['steps']),
                evaluation=str(development / 'summary.json'),
                evaluation_sha256=sha256(development / 'summary.json'),
                evaluation_groups=summary.get('groups', []))


def rates(rows):
    result = {}
    for profile in ('nominal', 'robust'):
        selected = [row for row in rows if row.get('profile') == profile]
        if not selected:
            raise ValueError('La selección necesita perfiles nominal y robusto completos.')
        result[profile] = dict(success=float(np.mean([r['success'] for r in selected])),
                               phases=np.mean([r['phases'] for r in selected], axis=0).tolist())
    return result


def promotion(candidate, incumbent):
    """No phase/profile success regression; tie break is a DEVELOPMENT heuristic.

    Costs use only common successful cases. This balanced scalar selects a
    checkpoint, never replaces the Pareto-style experimental comparison.
    """
    c, b = rates(candidate), rates(incumbent)
    for profile in c:
        if c[profile]['success'] + 1e-12 < b[profile]['success']:
            return False, {'reason': 'profile_success_regression', 'rates': c}
        if any(x + 1e-12 < y for x, y in zip(c[profile]['phases'], b[profile]['phases'])):
            return False, {'reason': 'phase_success_regression', 'rates': c}
    improvement = any(c[p]['success'] > b[p]['success'] + 1e-12 or
                      any(x > y + 1e-12 for x, y in zip(c[p]['phases'], b[p]['phases'])) for p in c)
    if improvement:
        return True, {'reason': 'success_improvement_without_regression', 'rates': c}
    key = lambda row: (row.get('profile'), row['map'], row['mass'], row['seed'])
    old = {key(row): row for row in incumbent}
    metrics = ('full_time_s', 'full_mechanical_work_j', 'cte_rms_m', 'yaw_rms_rad', 'carry_rms_m')
    pairs = [(row, old[key(row)]) for row in candidate if row['success'] and
             key(row) in old and old[key(row)]['success']]
    means = {}
    for metric in metrics:
        values = [np.log(row[metric] / previous[metric]) for row, previous in pairs
                  if row.get(metric) is not None and previous.get(metric) is not None
                  and row[metric] > 0 and previous[metric] > 0]
        if values:
            means[metric] = float(np.mean(values))
    score = float(np.mean(list(means.values()))) if len(means) == len(metrics) else None
    return bool(score is not None and score < -.001), dict(
        reason='development_balanced_cost_tiebreak', common_success_pairs=len(pairs),
        mean_log_ratios=means, score=score, rates=c)


def build_manifest(family, environment_class, output, budget, validation, provenance,
                   training_counts, *, root=ROOT):
    output, root = Path(output), Path(root)
    module, class_name = FAMILIES[family]
    report = output / 'holdout_report.json'
    return dict(schema_version=1, role='final_single_actor', experiment=family,
        actor_count=1, observation_dim=124, action_dim=7, inference_composition='single_actor',
        environment={'module': module, 'class': class_name, 'revision': environment_class.REVISION,
                     'sha256': sha256(CODE_ROOT / (module + '.py')),
                     'common_sha256': sha256(CODE_ROOT / 'MM_DATA_VXR1.py'),
                     'action_semantics': environment_class.ACTION_SEMANTICS,
                     'kwargs': {'p1_budget_steps': int(budget)}},
        phase_budgets=[1000, int(budget), 1000],
        artifacts={'model': {'path': 'model.zip', 'sha256': sha256(output / 'model.zip')},
                   'vecnormalize': {'path': 'vecnormalize.pkl', 'sha256': sha256(output / 'vecnormalize.pkl')}},
        route_hashes={p.name: sha256(p) for p in sorted((root / 'RUTAS_SAC').glob('*.json'))},
        validation={'status': 'evaluated', 'target_met': bool(validation['target_met']),
                    'report': {'path': report.name, 'sha256': sha256(report)}},
        training=training_counts, teacher_provenance=provenance,
        deployment_note='Only student.actor predicts actions; source/teacher are training-only, '
                        'not serialized as an inference composition.')


def fit_bc(model, norm, data, updates, seed, announce):
    import torch
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=1e-5)
    model.policy.set_training_mode(True)
    losses = []
    for update in range(1, updates + 1):
        indices = balanced_indices(data, 512, rng)
        observations = torch.as_tensor(norm.normalize_obs(data['observations'][indices].copy()),
                                       dtype=torch.float32, device=model.device)
        labels = torch.as_tensor(data['actions'][indices], dtype=torch.float32, device=model.device)
        weights = np.ones((len(indices), 7), np.float32)
        weights[data['origin'][indices] == 1, 5:] = 0.
        weights = torch.as_tensor(weights, device=model.device)
        prediction = model.actor(observations, deterministic=True)
        loss = ((prediction - labels).square() * weights).sum() / weights.sum().clamp_min(1.)
        if not torch.isfinite(loss):
            raise FloatingPointError('BC produjo una pérdida no finita.')
        optimizer.zero_grad()
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.)
        if not torch.isfinite(gradient):
            raise FloatingPointError('BC produjo gradientes no finitos.')
        optimizer.step()
        losses.append(float(loss.detach()))
        if update in {max(1, updates // 2), updates}:
            announce('BC_PROGRESS', completed=update, total=updates,
                     mean_loss=float(np.mean(losses[-100:])))
    model.policy.set_training_mode(False)


def fresh_student(source, norm, family, device, seed):
    """Copy the ACTOR only into fresh SAC critics/optimizers/replay."""
    import torch
    model = VXSAC('MlpPolicy', norm, device=device, seed=seed, learning_rate=3e-6,
                  buffer_size=60000, learning_starts=0, batch_size=256, gamma=.9995,
                  tau=.005, train_freq=1, gradient_steps=1, ent_coef='auto_0.001',
                  policy_kwargs=copy.deepcopy(source.policy_kwargs),
                  replay_buffer_kwargs={'handle_timeout_termination': False}, verbose=0)
    model.actor.load_state_dict(source.actor.state_dict())
    for parameter in model.actor.parameters():
        parameter.requires_grad_(True)
    if family == 'r2':
        with torch.no_grad():
            model.actor.mu.weight[5:].zero_()
            model.actor.mu.bias[5:].zero_()
            model.actor.log_std.weight[5:].zero_()
            model.actor.log_std.bias[5:].fill_(-2.)
    return model


def run_student(args):
    import gymnasium as gym
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure as configure_logger
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requerida; no se inicia silenciosamente en CPU.')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    module, class_name = FAMILIES[args.family]
    environment_class = Environment

    provenance = teacher_gate(args.teacher)
    source, source_norm = load_source(args.source, args.device)
    teacher = load_teacher(args.teacher, args.device, source_path=args.source,
        norm_path=args.source.parent / 'vecnormalize.pkl', expected_budget=args.p1_budget)
    if int(getattr(teacher, '_n_updates', 0)) <= 0:
        raise ValueError('El archivo real del profesor no contiene actualizaciones SAC.')
    label_fn = composed_action(source, source_norm, teacher)
    factory = lambda: environment_class(render_mode=None, debug_rewards=False,
                                        p1_budget_steps=args.p1_budget)

    class TrainingCasesEnv(gym.Wrapper):
        def __init__(self):
            super().__init__(factory())
            self.case_rng = np.random.default_rng(args.seed)
            self.episode = 0

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                self.case_rng = np.random.default_rng(seed)
            index = self.episode % 20
            self.episode += 1
            self.env.set_eval_mode(True)
            self.env.FIXED_MAP_ID = 1 + index // 5
            self.env.set_eval_mass((.5, .75, 1., 1.25, 1.5)[index % 5])
            self.env.CUBE_XY_NOISE_RANGE = .01
            self.env.CUBE_YAW_NOISE_RANGE_RAD = .1
            return self.env.reset(seed=int(self.case_rng.integers(100_000_000, 200_000_000)), options=options)

        def step(self, action):
            obs, reward, term, trunc, info = self.env.step(action)
            # These are finite TASK deadlines, not artificial rollout cuts.
            info['TimeLimit.truncated'] = False
            return obs, reward, bool(term or trunc), False, info

    norm = VecNormalize.load(args.source.parent / 'vecnormalize.pkl', DummyVecEnv([TrainingCasesEnv]))
    norm.training = False
    norm.norm_reward = False
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output
    counts = dict(bc_updates_attempted=0, sac_updates_attempted=0, sac_steps_attempted=0,
                  critic_warmup_attempted=0, selected_sac_updates=0, selected_sac_steps=0,
                  actor_initialized_from=str(args.source), critics='fresh', replay='new family only')

    def announce(stage, **details):
        record = dict(stage=stage, **details)
        dump(output / 'status.json', record)
        with (output / 'milestones.jsonl').open('a') as file:
            file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        print(stage + ' ' + json.dumps(details, ensure_ascii=False), flush=True)

    config = dict(revision=environment_class.REVISION, artifact_role='single_student_training',
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  source_sha256=sha256(args.source), teacher=provenance,
                  route_hashes={p.name: sha256(p) for p in sorted((ROOT / 'RUTAS_SAC').glob('*.json'))},
                  source_files={p.name: sha256(p) for p in
                     (CODE_ROOT / (module + '.py'), CODE_ROOT / 'MM_DATA_VXR1.py', Path(__file__).resolve())},
                  observations='raw replay, frozen VecNormalize exactly once',
                  actor_count_at_deployment=1, warm_transfer='actor only; base mean reset for R2',
                  no_old_physical_replay=True, training_counts=counts)
    dump(output / 'config.json', config)
    model = fresh_student(source, norm, args.family, args.device, args.seed)
    model.set_logger(configure_logger(str(output / 'sac_logs'), ['csv']))

    def action_fn(observation, env):
        return np.asarray(model.predict(norm.normalize_obs(np.asarray(observation).copy()),
                                        deterministic=True)[0], np.float32)

    def save(directory):
        directory.mkdir(parents=True, exist_ok=True)
        model.save(directory / 'model.zip')
        norm.save(directory / 'vecnormalize.pkl')
        dump(directory / 'checkpoint.json', dict(sac_updates=int(model._n_updates),
             sac_steps=int(model.num_timesteps), bc_updates_attempted=counts['bc_updates_attempted']))

    development_cases = (canonical_cases(reps=1, seed=61_000_000, profile='nominal') +
                         canonical_cases(reps=args.validation_reps, seed=61_010_000, profile='robust'))

    def evaluate(tag):
        announce('EVALUATE', tag=tag, cases=len(development_cases))
        result = evaluate_cases(action_fn, development_cases, output / ('eval_' + tag),
                                         budgets=(args.p1_budget,), variants=('on',),
                                         metadata={'role': 'single_actor_student', 'stage': tag})
        return result['rows']

    try:
        announce('START', output=str(output), family=args.family, single_actor=True)
        save(output / 'initial')
        best_rows = evaluate('initial')
        save(output / 'best')
        best_tag = 'initial'

        def select(tag):
            nonlocal best_rows, best_tag
            save(output / tag)
            candidate_rows = evaluate(tag)
            accepted, decision = promotion(candidate_rows, best_rows)
            dump(output / tag / 'selection.json', dict(accepted=accepted, **decision))
            if accepted:
                best_rows, best_tag = candidate_rows, tag
                save(output / 'best')
            announce('SELECTION', candidate=tag, accepted=accepted, best=best_tag, **decision)

        pool = None
        if args.old_data:
            pool = load_old_bc(args.old_data, source_env_hash=SOURCE_HASHES['zenno_vx_r1.py'],
                               source_norm_hash=sha256(args.source.parent / 'vecnormalize.pkl'))
            announce('OLD_BC_ONLY', rows=len(pool['phase_ids']), phase1_excluded=True, replay_rows=0)
        training_cases = canonical_cases(reps=1, seed=141_000_000, profile='robust')
        announce('TEACHER_COLLECTION', episodes=len(training_cases), real_handoffs=True)
        pool = collect(factory, label_fn, None, training_cases, output / 'teacher_episodes',
                       pool=pool, seed=args.seed, announce=announce)
        if not all(np.any(pool['phase_ids'] == phase) for phase in (0, 1, 2)):
            raise RuntimeError('El profesor no alcanzó todas las fases; no fabricar ejemplos P2.')
        fit_bc(model, norm, pool, args.bc_updates, args.seed, announce)
        counts['bc_updates_attempted'] += args.bc_updates
        select('bc')
        for round_index in range(1, args.dagger_rounds + 1):
            cases = canonical_cases(reps=1, seed=141_000_000 + round_index * 1_000_000,
                                              profile='robust')
            announce('DAGGER_COLLECTION', round=round_index, rescue=False, episodes=len(cases))
            pool = collect(factory, label_fn, action_fn, cases, output / f'dagger_{round_index}_episodes',
                           pool=pool, seed=args.seed + round_index, announce=announce)
            fit_bc(model, norm, pool, args.dagger_updates, args.seed + round_index, announce)
            counts['bc_updates_attempted'] += args.dagger_updates
            select(f'dagger_{round_index}')
        save_npz(output / 'bounded_dataset.npz', pool)

        # Online RL starts from the validated incumbent, not the last rejected BC block.
        model = VXSAC.load(output / 'best/model.zip', env=norm, device=args.device,
                           custom_objects={'buffer_size': 60000, 'learning_starts': 0,
                                           'learning_rate': 3e-6})
        model.set_logger(configure_logger(str(output / 'sac_logs'), ['csv']))
        indices = replay_rows(pool)
        if not len(indices):
            raise ValueError('No existen transiciones físicas nuevas para entrenar SAC.')
        if not all(np.any(pool['phase_ids'][indices] == phase) for phase in (0, 1, 2)):
            raise ValueError('El replay físico nuevo debe contener P0, P1 y P2; BC antiguo no las sustituye.')
        model.replay_buffer.reset()
        for index in indices:
            model.replay_buffer.add(pool['observations'][index:index + 1],
                pool['next_observations'][index:index + 1], pool['executed_actions'][index:index + 1],
                pool['rewards'][index:index + 1], pool['dones'][index:index + 1],
                [{'TimeLimit.truncated': False}])
        action_weights = np.ones((3, 7), np.float32)
        action_weights[:, 5:] = 0.  # Never BC-anchor base channels during SAC exploration.
        demo = dict(pool, teacher_stages=np.where(pool['window01'] | pool['window12'],
                                                  'real_handoff', 'phase_interior'))
        model.set_demo_regularization(demo, loss_weight=5., batch_size=512, seed=args.seed,
            normalize_q_actor_loss=True, q_objective_scale=.15,
            actor_warmup_updates=int(model._n_updates) + args.critic_warmup,
        action_weights_by_phase=action_weights)

        if args.sac_steps:
            announce('CRITIC_WARMUP', updates=args.critic_warmup, replay_rows=len(indices), actor_frozen=True)
            model._current_progress_remaining = 1.
            # train(0) computes means of empty loss lists in DemoRegularizedSAC
            # and records NaNs. A requested zero warmup means no train call.
            if args.critic_warmup:
                model.train(args.critic_warmup, batch_size=256)
            counts['critic_warmup_attempted'] += args.critic_warmup
            counts['sac_updates_attempted'] += args.critic_warmup
            announce('CRITIC_WARMUP_COMPLETE', updates=int(model._n_updates))

            class Milestone(BaseCallback):
                def __init__(self):
                    super().__init__()
                    self.next = model.num_timesteps + max(1, args.sac_steps // 2)

                def _on_step(self):
                    if self.num_timesteps >= self.next:
                        save(output / 'latest')
                        announce('SAC_PROGRESS', steps=self.num_timesteps, updates=int(model._n_updates))
                        self.next += max(1, args.sac_steps // 2)
                    return True

            before = int(model._n_updates)
            model.learn(args.sac_steps, reset_num_timesteps=False, callback=Milestone(), log_interval=1)
            counts['sac_updates_attempted'] += int(model._n_updates) - before
            counts['sac_steps_attempted'] += args.sac_steps
            select('sac')
        model = VXSAC.load(output / 'best/model.zip', env=norm, device=args.device,
                           custom_objects={'buffer_size': 1, 'learning_starts': 0})
        counts['selected_sac_updates'] = int(model._n_updates)
        counts['selected_sac_steps'] = int(model.num_timesteps)
        counts['selected_stage'] = best_tag
        counts['selected_has_online_sac'] = bool(model.num_timesteps > 0)
        if not counts['selected_has_online_sac']:
            announce('WARNING_SELECTED_NO_ONLINE_SAC', selected_stage=best_tag,
                     note='El candidato SAC no fue seleccionado; no atribuirle SAC online al modelo final.')
        final = output / 'final'
        save(final)
        holdout_cases = (canonical_cases(reps=1, seed=71_000_000, profile='nominal') +
                         canonical_cases(reps=args.holdout_reps, seed=71_010_000, profile='robust'))
        announce('HOLDOUT', cases=len(holdout_cases), selection_frozen=True,
                 model_sha256=sha256(final / 'model.zip'))
        evaluated = evaluate_cases(action_fn, holdout_cases, output / 'holdout',
            budgets=(args.p1_budget,), variants=('on',),
            metadata={'role': 'single_actor_student', 'scope': 'independent_holdout',
                      'model_sha256': sha256(final / 'model.zip'), 'selection_frozen': True})
        holdout_rates = rates(evaluated['rows'])
        target_met = bool(holdout_rates['nominal']['success'] >= 1. and
                          holdout_rates['robust']['success'] >= args.target_success)
        report = dict(complete=True, scope='full_task', deterministic=True, holdout_independent=True,
            model_sha256=sha256(final / 'model.zip'), cases=len(evaluated['rows']),
            successes=sum(bool(row['success']) for row in evaluated['rows']), target_met=target_met,
            target_robust=args.target_success, rates=holdout_rates, rows=evaluated['rows'],
            selected_using='development only; no holdout model reselection', training_counts=counts)
        dump(final / 'holdout_report.json', report)
        dump(final / 'final_student.json', build_manifest(args.family, environment_class, final,
             args.p1_budget, report, provenance, counts))
        validate_final_student(final / 'final_student.json', expected_experiment=args.family)
        config['training_counts'] = counts
        dump(output / 'config.json', config)
        dump(output / 'result.json', dict(status='COMPLETE', final_manifest=str(final / 'final_student.json'),
                                         target_met=target_met, holdout_rates=holdout_rates,
                                         training_counts=counts))
        announce('COMPLETE', final_manifest=str(final / 'final_student.json'), target_met=target_met,
                 rates=holdout_rates, selected_stage=best_tag, selected_sac_updates=int(model._n_updates))
    except BaseException as error:
        try:
            save(output / 'interrupted')
        except BaseException as save_error:
            dump(output / 'save_error.json', dict(error=repr(save_error)))
        announce('INTERRUPTED', error=repr(error), training_counts=counts)
        raise
    finally:
        norm.close()
        source_norm.close()


def student_cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=tuple(FAMILIES), required=True)
    parser.add_argument('--teacher', type=Path, required=True, help='blockN/model.zip with completed ON/OFF evaluation')
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--p1-budget', type=int, choices=(3000, 4500, 6000), default=4500)
    parser.add_argument('--old-data', type=Path, help='Optional P0/P2 labels; requires matching .contract.json')
    parser.add_argument('--bc-updates', type=int, default=3000)
    parser.add_argument('--dagger-rounds', type=int, default=2)
    parser.add_argument('--dagger-updates', type=int, default=1000)
    parser.add_argument('--sac-steps', type=int, default=20000)
    parser.add_argument('--critic-warmup', type=int, default=2000)
    parser.add_argument('--validation-reps', type=int, default=1)
    parser.add_argument('--holdout-reps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=15110000)
    parser.add_argument('--target-success', type=float, default=.98)
    args = parser.parse_args(argv)
    if min(args.bc_updates, args.dagger_updates, args.validation_reps, args.holdout_reps) < 1:
        parser.error('BC, DAgger y repeticiones deben ser positivos.')
    if min(args.dagger_rounds, args.sac_steps, args.critic_warmup) < 0 or not 0 < args.target_success <= 1:
        parser.error('Presupuestos no negativos y target-success en (0,1].')
    for key in ('output', 'source', 'teacher', 'old_data'):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    if args.source.is_dir():
        args.source = args.source / 'model.zip'
    if args.teacher.is_dir():
        args.teacher = args.teacher / 'model.zip'
    if args.output.exists():
        parser.error('output debe ser nuevo; no se sobrescriben corridas existentes.')

    def interrupt(signum, frame):
        raise InterruptedError(f'Señal {signum}')

    signal.signal(signal.SIGTERM, interrupt)
    lock_path = ROOT / 'models' / ('.single_student_' + args.family + '.lock')
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Ya hay un estudiante de esta familia entrenando.')
        run_student(args)




SOURCE_HASHES = {'zenno_vx_r1.py': '9de9e722480de8166b32e36dd2076b06baee04b455d58e1993ab14c59f065267', 'zenno_vx_r1_1.py': '126a3ae62fa333ef84434c7e9a47b854637ace6e3ccc46e41258ec87135b0846', 'zenno_vx_r2.py': 'ac7042132bf142990cf798826480a7bdc257e2f45f4b6339d353872931009bc2', 'train_zenno_vx_r2.py': '24943d7eceb6e50769a23acb8305082eac809d26b830d5c48fc959d68afbf098', 'train_zenno_vx_r1_1.py': '114e82c4676721d6bb672edce8a22011d4b8d24112ac4a2a524a3bb746b14244', 'vx_r2_teacher.py': 'c11104ea83b8d64d0fe2a5a09883471bf47595a1252173ae5ca5b06d2336c7b7', 'vx_r1_1_teacher.py': '3ce064db0dabac5b22be81b411a474ca6159de0840edfc7a9c22db6e353968a3', 'evaluate_zenno_vx_r1_1.py': '55f4f22f005ea83b165d0d76be0886a715900f7cf695b3a047cd95d0523b15e4', 'evaluate_zenno_vx_r2.py': '02aae17351776a115cda78d50cdf3f3c2ea64ec5a5186304baa78bfebc962b7e', 'vx_pose_metrics.py': 'f432ba199c6a571c60a986fd26ec6fe7580935ba459a6722d375fbfdaf967529'}

def source_digest(name):
    if name in SOURCE_HASHES:
        return SOURCE_HASHES[name]
    if name not in ('TRAIN_VX.py','MM_DATA_VXR1.py','MM_DATA_VXR2.py'):
        raise ValueError('Unrecognized provenance source: '+name)
    return sha256(CODE_ROOT/name)

def select_family(family):
    global FAMILY, Environment, DiagnosticEnv, ARM_MASK_REVISION
    if family not in ('r1_1','r2'):
        raise ValueError('Familia debe ser r1_1 o r2')
    FAMILY = family
    Environment = ZennoVXR11 if family == 'r1_1' else ZennoVXR2
    DiagnosticEnv = DiagnosticTorque if family == 'r1_1' else DiagnosticPP
    ARM_MASK_REVISION = ('R11' if family == 'r1_1' else 'R2') + '_FROZEN_ARM_SYNTHETIC_RESIDUAL_CHANNELS_ZERO_V1'

select_family('r1_1')

def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('stage',choices=('teacher','student'))
    parser.add_argument('--family',choices=('r1_1','r2'),required=True)
    if argv is None:
        argv=sys.argv[1:]
    if not argv or argv in (['-h'],['--help']):
        parser.print_help()
        print('Ayuda detallada: TRAIN_VX.py teacher|student --family r1_1|r2 --help')
        return
    args, remaining=parser.parse_known_args(argv)
    select_family(args.family)
    verify_consolidated_sources()
    (ROOT/'models').mkdir(parents=True,exist_ok=True)
    if args.stage == 'teacher':
        teacher_cli(remaining)
    else:
        student_cli(['--family',args.family]+remaining)

if __name__ == '__main__':
    main()
