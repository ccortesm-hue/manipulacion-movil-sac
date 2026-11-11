"""SAC V9 con regularización por demostraciones, sin dependencias heredadas."""

from __future__ import annotations

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
