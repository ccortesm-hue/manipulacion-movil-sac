#!/home/claudio/Documentos/Zenno_Ambient_v1/Zenno_v1/bin/python
"""Entrena desde cero un único SAC V9 incremental y end-to-end.

El curriculum conserva las transiciones reales: BC avanza P0 -> P0/P1 ->
P0/P1/P2 con episodios docentes completos y el ajuste SAC siempre comienza
físicamente en P0. No se permite cargar pesos ni normalizadores anteriores.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "demonstrations" / "v9_teacher_demos.npz"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "sac_v9_incremental_run1"
os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/zenno_v9_matplotlib")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3.common.callbacks import BaseCallback, CallbackList  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization  # noqa: E402

from demo_regularized_sac_v9 import DemoRegularizedSAC  # noqa: E402
from zenno_v9 import ACTION_DIM, OBS_DIM, ZennoV9  # noqa: E402

INFO_KEYWORDS = (
    "phase_state", "map_id", "cube_mass", "phase0_success",
    "phase1_success", "phase2_success", "strict_success", "terminal_phase",
    "cube_goal_xy_error", "cube_z_abs_error", "release_ready_count",
    "environment_revision", "action_semantics", "smoothness_penalty",
    "residual_authority",
)


def make_env_factory(train_masses: Optional[Sequence[float]] = None):
    def factory() -> ZennoV9:
        env = ZennoV9(render_mode=None, debug_rewards=False)
        if train_masses is not None:
            env.set_train_masses(list(train_masses))
        return env
    return factory


def build_train_env(
    log_dir: Path, n_envs: int, seed: int,
    train_masses: Optional[Sequence[float]],
    init_vecnormalize: Optional[Path] = None,
) -> VecNormalize:
    raw = make_vec_env(
        make_env_factory(train_masses), n_envs=int(n_envs), seed=int(seed),
        monitor_dir=str(log_dir), monitor_kwargs={"info_keywords": INFO_KEYWORDS},
    )
    if init_vecnormalize is not None:
        env = VecNormalize.load(str(init_vecnormalize), raw)
        env.training = True
        env.norm_obs = True
        env.norm_reward = False
        env.clip_obs = 10.0
        env.gamma = 0.999
        env.epsilon = 1e-8
        return env
    return VecNormalize(
        raw, norm_obs=True, norm_reward=False, clip_obs=10.0,
        gamma=0.999, epsilon=1e-8,
    )


def build_eval_env(seed: int) -> VecNormalize:
    raw = make_vec_env(make_env_factory(), n_envs=1, seed=int(seed))
    env = VecNormalize(
        raw, norm_obs=True, norm_reward=False, clip_obs=10.0,
        gamma=0.999, epsilon=1e-8,
    )
    env.training = False
    return env


def base_env(env: VecNormalize) -> ZennoV9:
    base = env.venv.envs[0].unwrapped
    if not isinstance(base, ZennoV9):
        raise TypeError(type(base))
    return base


def csv_optional(text: str, cast):
    values = [cast(item.strip()) for item in text.split(",") if item.strip()]
    return values or None


def load_dataset(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = (
            "observations", "next_observations", "actions", "rewards", "dones",
            "phase_ids", "map_ids", "masses_kg", "teacher_is_analytic",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"Dataset V9 incompleto; faltan {missing}.")
        dataset = {key: np.asarray(payload[key]).copy() for key in required}
        for optional_key in (
            "teacher_stages", "episode_ids", "dagger_round", "executed_actions"
        ):
            if optional_key in payload:
                dataset[optional_key] = np.asarray(payload[optional_key]).copy()
        count = int(dataset["observations"].shape[0])
        dataset["replay_valid"] = (
            np.asarray(payload["replay_valid"], dtype=np.bool_).copy()
            if "replay_valid" in payload else np.ones(count, dtype=np.bool_)
        )
    count = int(dataset["observations"].shape[0])
    if not bool(np.all(np.asarray(dataset["teacher_is_analytic"], dtype=np.bool_))):
        raise ValueError(
            "El dataset contiene demostraciones no analíticas. "
            "Ejecute collect_demos_v9.py antes de entrenar."
        )
    expected = {
        "observations": (count, OBS_DIM), "next_observations": (count, OBS_DIM),
        "actions": (count, ACTION_DIM), "phase_ids": (count,),
        "replay_valid": (count,), "teacher_is_analytic": (count,),
    }
    for key, shape in expected.items():
        if dataset[key].shape != shape:
            raise ValueError(f"{key}: {dataset[key].shape} != {shape}")
    return dataset


def update_observation_normalizer(env: VecNormalize, data: Dict[str, np.ndarray]) -> None:
    env.obs_rms.update(np.asarray(data["observations"], dtype=np.float32))
    env.obs_rms.update(np.asarray(data["next_observations"], dtype=np.float32))


class StageBalancedSampler:
    """Balancea fases y subetapas para no aprender sólo acciones cero."""

    def __init__(
        self, phase_ids: np.ndarray, teacher_stages: Optional[np.ndarray], seed: int,
        dagger_round: Optional[np.ndarray] = None,
    ):
        self.rng = np.random.default_rng(int(seed))
        phases = np.asarray(phase_ids, dtype=np.int64)
        if teacher_stages is None:
            self.indices = {
                (phase, f"p{phase}"): np.flatnonzero(phases == phase)
                for phase in (0, 1, 2)
            }
        else:
            stages = np.asarray(teacher_stages).astype(str)
            self.indices = {
                (phase, stage): np.flatnonzero(
                    (phases == phase) & (stages == stage)
                )
                for phase in (0, 1, 2)
                for stage in np.unique(stages[phases == phase])
            }
        self.indices = {
            key: rows for key, rows in self.indices.items() if rows.size > 0
        }
        rounds = (
            np.zeros(phases.shape[0], dtype=np.int16)
            if dagger_round is None
            else np.asarray(dagger_round, dtype=np.int16)
        )
        self.dagger_indices = {
            key: rows[rounds[rows] > 0] for key, rows in self.indices.items()
        }
        self.latest_dagger_round = int(np.max(rounds))
        self.latest_dagger_indices = {
            key: rows[rounds[rows] == self.latest_dagger_round]
            for key, rows in self.indices.items()
        }
        missing = [
            phase
            for phase in (0, 1, 2)
            if not any(key[0] == phase for key in self.indices)
        ]
        if missing:
            raise ValueError(f"Dataset sin demostraciones para fases {missing}.")

    def sample(
        self, batch_size: int, allowed_phases: Sequence[int] = (0, 1, 2),
        dagger_fraction: float = 0.0,
        latest_dagger_fraction: float = 0.0,
    ) -> np.ndarray:
        allowed = np.asarray(tuple(allowed_phases), dtype=np.int64)
        if allowed.size == 0 or np.any((allowed < 0) | (allowed > 2)):
            raise ValueError(f"Fases BC inválidas: {allowed_phases}")
        groups = [key for key in self.indices if key[0] in set(allowed.tolist())]
        selected = self.rng.integers(0, len(groups), size=int(batch_size))
        result = np.empty(int(batch_size), dtype=np.int64)
        for group_id, key in enumerate(groups):
            positions = np.flatnonzero(selected == group_id)
            if positions.size:
                dagger_rows = self.dagger_indices[key]
                use_dagger = (
                    self.rng.random(positions.size)
                    < float(np.clip(dagger_fraction, 0.0, 1.0))
                ) & bool(dagger_rows.size)
                if np.any(use_dagger):
                    dagger_positions = positions[use_dagger]
                    latest_rows = self.latest_dagger_indices[key]
                    use_latest = (
                        self.rng.random(dagger_positions.size)
                        < float(np.clip(latest_dagger_fraction, 0.0, 1.0))
                    ) & bool(latest_rows.size)
                    if np.any(use_latest):
                        result[dagger_positions[use_latest]] = self.rng.choice(
                            latest_rows,
                            size=int(np.sum(use_latest)),
                            replace=True,
                        )
                    if np.any(~use_latest):
                        result[dagger_positions[~use_latest]] = self.rng.choice(
                            dagger_rows,
                            size=int(np.sum(~use_latest)),
                            replace=True,
                        )
                if np.any(~use_dagger):
                    result[positions[~use_dagger]] = self.rng.choice(
                        self.indices[key], size=int(np.sum(~use_dagger)), replace=True
                    )
        return result


def scaled_demo_dataset(
    model: DemoRegularizedSAC, data: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    result = {key: value for key, value in data.items()}
    result["actions"] = np.asarray(
        model.policy.scale_action(np.asarray(data["actions"], dtype=np.float32)),
        dtype=np.float32,
    )
    return result


def behavior_clone(
    model: DemoRegularizedSAC, env: VecNormalize, data: Dict[str, np.ndarray],
    gradient_steps: int, batch_size: int, seed: int,
    dagger_fraction: float = 0.0,
    latest_dagger_fraction: float = 0.0,
    curriculum_enabled: bool = True,
) -> List[float]:
    if int(gradient_steps) <= 0:
        return []
    sampler = StageBalancedSampler(
        data["phase_ids"], data.get("teacher_stages"), seed,
        dagger_round=data.get("dagger_round"),
    )
    observations = np.asarray(data["observations"], dtype=np.float32)
    targets = np.asarray(
        model.policy.scale_action(np.asarray(data["actions"], dtype=np.float32)),
        dtype=np.float32,
    )
    losses: List[float] = []
    model.actor.set_training_mode(True)
    for update in range(int(gradient_steps)):
        fraction = float(update) / max(float(gradient_steps), 1.0)
        if not bool(curriculum_enabled):
            curriculum_phases = (0, 1, 2)
            curriculum_stage = "ALL_DAGGER"
        elif fraction < 0.20:
            curriculum_phases = (0,)
            curriculum_stage = "P0"
        elif fraction < 0.50:
            curriculum_phases = (0, 1)
            curriculum_stage = "P0_P1"
        else:
            curriculum_phases = (0, 1, 2)
            curriculum_stage = "P0_P1_P2"
        indices = sampler.sample(
            int(batch_size), curriculum_phases,
            dagger_fraction=float(dagger_fraction),
            latest_dagger_fraction=float(latest_dagger_fraction),
        )
        obs_tensor = torch.as_tensor(
            env.normalize_obs(observations[indices].copy()),
            dtype=torch.float32, device=model.device,
        )
        target_tensor = torch.as_tensor(
            targets[indices], dtype=torch.float32, device=model.device,
        )
        loss = torch.nn.functional.mse_loss(
            model.actor(obs_tensor, deterministic=True), target_tensor
        )
        model.actor.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.actor.parameters(), max_norm=5.0)
        model.actor.optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        if (update + 1) % max(int(gradient_steps) // 10, 1) == 0:
            print(
                f"[V9-BC:{curriculum_stage}] {update + 1}/{gradient_steps} "
                f"loss={float(np.mean(losses[-100:])):.6f}"
            )
    return losses


def prefill_replay(
    model: DemoRegularizedSAC, data: Dict[str, np.ndarray], seed: int,
    transitions_per_phase: int = 0,
) -> int:
    replay = model.replay_buffer
    if replay is None:
        raise RuntimeError("Replay SAC no inicializado.")
    valid = np.flatnonzero(np.asarray(data["replay_valid"], dtype=np.bool_))
    if valid.size == 0:
        return 0
    rng = np.random.default_rng(int(seed))
    if int(transitions_per_phase) > 0:
        phase_ids = np.asarray(data["phase_ids"], dtype=np.int64)
        phase_orders = []
        for phase in (0, 1, 2):
            rows = valid[phase_ids[valid] == phase]
            if rows.size == 0:
                raise ValueError(f"Replay docente sin filas válidas para P{phase}.")
            phase_orders.append(rng.choice(
                rows,
                size=int(transitions_per_phase),
                replace=rows.size < int(transitions_per_phase),
            ))
        order = rng.permutation(np.concatenate(phase_orders))
    else:
        order = rng.permutation(valid)
    scaled_actions = np.asarray(
        model.policy.scale_action(np.asarray(data["actions"], dtype=np.float32)),
        dtype=np.float32,
    )
    n_envs = int(replay.n_envs)
    added = 0
    for start in range(0, int(order.size), n_envs):
        indices = order[start : start + n_envs]
        if indices.size < n_envs:
            indices = np.concatenate(
                [indices, rng.choice(order, size=n_envs - indices.size, replace=True)]
            )
        replay.add(
            np.asarray(data["observations"][indices], dtype=np.float32),
            np.asarray(data["next_observations"][indices], dtype=np.float32),
            scaled_actions[indices],
            np.asarray(data["rewards"][indices], dtype=np.float32),
            np.asarray(data["dones"][indices], dtype=np.float32),
            [{"TimeLimit.truncated": False, "expert_demo": True} for _ in range(n_envs)],
        )
        added += int(indices.size)
    print(f"[V9-DEMO] Replay precargado con {added} transiciones fisicas.")
    return added


def initialize_actor_exploration(
    model: DemoRegularizedSAC, initial_log_std: float
) -> None:
    """Evita que el primer rollout destruya una media ya clonada por BC."""
    log_std_layer = getattr(model.actor, "log_std", None)
    if not isinstance(log_std_layer, torch.nn.Linear):
        raise TypeError("La política SAC no expone una capa log_std lineal.")
    with torch.no_grad():
        log_std_layer.weight.zero_()
        log_std_layer.bias.fill_(float(initial_log_std))
    print(f"[V9-INIT] Exploración inicial log_std={float(initial_log_std):.3f}.")


def dataset_cases(data: Dict[str, np.ndarray]) -> List[Tuple[int, float]]:
    return sorted(
        {(int(m), float(kg)) for m, kg in zip(data["map_ids"], data["masses_kg"])}
    )


class V9EvaluationCallback(BaseCallback):
    def __init__(
        self, eval_env: VecNormalize, train_env: VecNormalize,
        cases: Sequence[Tuple[int, float]], save_dir: Path,
        eval_freq_steps: int, max_steps: int = 4_700,
    ):
        super().__init__(verbose=1)
        self.eval_env = eval_env
        self.train_env = train_env
        self.cases = list(cases)
        self.save_dir = Path(save_dir)
        self.eval_freq_steps = int(eval_freq_steps)
        self.max_steps = int(max_steps)
        self.last_eval_at = 0
        self.best_key = (-np.inf,) * 5

    def _init_callback(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.last_eval_at = int(self.num_timesteps)

    def evaluate(self) -> Dict[str, object]:
        sync_envs_normalization(self.train_env, self.eval_env)
        self.eval_env.training = False
        base = base_env(self.eval_env)
        results: List[Dict[str, object]] = []
        for index, (map_id, mass_kg) in enumerate(self.cases):
            base.FIXED_MAP_ID = int(map_id)
            base.set_eval_mode(True)
            base.set_eval_mass(float(mass_kg))
            self.eval_env.seed(700_000 + index)
            observation = self.eval_env.reset()
            reached = 0
            final_info: dict = {}
            for step in range(1, self.max_steps + 1):
                action, _ = self.model.predict(observation, deterministic=True)
                observation, _reward, dones, infos = self.eval_env.step(action)
                final_info = infos[0]
                reached = max(reached, int(final_info.get("phase_state", base.phase_state)))
                if bool(dones[0]):
                    break
            results.append({
                "map_id": int(map_id), "mass_kg": float(mass_kg), "steps": int(step),
                "strict_success": bool(final_info.get("strict_success", False)),
                "phase0_success": bool(final_info.get("phase0_success", False)),
                "phase1_success": bool(final_info.get("phase1_success", False)),
                "phase2_success": bool(final_info.get("phase2_success", False)),
                "phase_reached": int(reached),
                "terminal_phase": int(final_info.get("terminal_phase", -1)),
            })
        base.FIXED_MAP_ID = None
        count = max(len(results), 1)
        strict = float(sum(row["strict_success"] for row in results) / count)
        p2_reach = float(sum(int(row["phase_reached"]) >= 2 for row in results) / count)
        p1 = float(sum(row["phase1_success"] for row in results) / count)
        p0 = float(sum(row["phase0_success"] for row in results) / count)
        mean_phase = float(np.mean([row["phase_reached"] for row in results]))
        key = (strict, p2_reach, p1, p0, mean_phase)
        payload = {
            "timesteps": int(self.num_timesteps), "strict_success_rate": strict,
            "phase2_reach_rate": p2_reach, "phase1_success_rate": p1,
            "phase0_success_rate": p0, "mean_phase_reached": mean_phase,
            "candidate_key": list(key), "results": results,
        }
        with (self.save_dir / "evaluation_history.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        if key > self.best_key:
            self.best_key = key
            self.model.save(str(self.save_dir / "best_model"))
            self.train_env.save(str(self.save_dir / "env_zenno_v9.pkl"))
            with (self.save_dir / "best_metrics.json").open("w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, ensure_ascii=False)
                file.write("\n")
            print(
                f"[V9-BEST] strict={100*strict:.1f}% P2reach={100*p2_reach:.1f}% "
                f"P1={100*p1:.1f}% P0={100*p0:.1f}%"
            )
        return payload

    def _on_step(self) -> bool:
        if int(self.num_timesteps) - self.last_eval_at >= self.eval_freq_steps:
            self.last_eval_at = int(self.num_timesteps)
            self.evaluate()
        return True


class SaveV9Callback(BaseCallback):
    def __init__(self, env: VecNormalize, directory: Path, frequency: int):
        super().__init__(verbose=0)
        self.env, self.directory, self.frequency = env, Path(directory), int(frequency)
        self.last_save = 0

    def _init_callback(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.last_save = int(self.num_timesteps)

    def _on_step(self) -> bool:
        if int(self.num_timesteps) - self.last_save >= self.frequency:
            self.last_save = int(self.num_timesteps)
            prefix = self.directory / f"checkpoint_{self.num_timesteps}"
            self.model.save(str(prefix))
            self.env.save(str(prefix) + "_env.pkl")
        return True


class DemoWeightAnnealingCallback(BaseCallback):
    """Reduce gradualmente BC para evitar olvido sin congelar la mejora SAC."""

    def __init__(self, start_weight: float, end_weight: float, total_steps: int):
        super().__init__(verbose=0)
        self.start_weight = float(start_weight)
        self.end_weight = float(end_weight)
        self.total_steps = max(int(total_steps), 1)

    def _on_step(self) -> bool:
        progress = float(np.clip(self.num_timesteps / self.total_steps, 0.0, 1.0))
        weight = self.start_weight + progress * (
            self.end_weight - self.start_weight
        )
        self.model._demo_loss_weight = float(weight)
        self.logger.record("train/demo_bc_weight", float(weight))
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--offline-only", action="store_true")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=842)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--ent-coef", default="auto_0.03")
    parser.add_argument("--bc-steps", type=int, default=12_000)
    parser.add_argument("--bc-batch-size", type=int, default=512)
    parser.add_argument("--bc-loss-weight", type=float, default=20.0)
    parser.add_argument("--bc-loss-weight-final", type=float, default=2.0)
    parser.add_argument("--demo-replay-per-phase", type=int, default=0)
    parser.add_argument("--normalize-q-actor-loss", action="store_true")
    parser.add_argument("--q-objective-scale", type=float, default=1.0)
    parser.add_argument("--actor-warmup-updates", type=int, default=0)
    parser.add_argument("--initial-log-std", type=float, default=-3.0)
    parser.add_argument(
        "--require-bc-strict-rate",
        type=float,
        default=0.0,
        help=(
            "No inicia SAC online si la política BC no supera esta tasa "
            "nominal end-to-end (0 desactiva el gate)."
        ),
    )
    parser.add_argument("--train-masses", default="0.5,0.75,1.0,1.25,1.5")
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    args = parser.parse_args()

    dataset_path = args.dataset.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if (output_dir / "training_config.json").is_file():
        raise FileExistsError(
            f"El run ya existe: {output_dir}. Usa otro --output para conservar "
            "la trazabilidad y comenzar un SAC realmente nuevo."
        )
    data = load_dataset(dataset_path)
    log_dir, tensorboard_dir = output_dir / "logs", output_dir / "tensorboard"
    best_dir, checkpoint_dir = output_dir / "best_model", output_dir / "checkpoints"
    for directory in (log_dir, tensorboard_dir, best_dir, checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)

    train_env = build_train_env(
        log_dir, args.n_envs, args.seed, csv_optional(args.train_masses, float),
        init_vecnormalize=None,
    )
    eval_env = build_eval_env(args.seed + 10_000)
    update_observation_normalizer(train_env, data)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ent_coef = (
        str(args.ent_coef)
        if str(args.ent_coef).startswith("auto")
        else float(args.ent_coef)
    )
    model = DemoRegularizedSAC(
        "MlpPolicy", train_env, learning_rate=float(args.learning_rate),
        buffer_size=int(args.buffer_size), learning_starts=0,
        batch_size=int(args.batch_size), tau=0.005, gamma=float(args.gamma),
        # Con cuatro entornos, gradient_steps=-1 ejecuta cuatro updates por
        # paso vectorizado y mantiene una razón update/transición cercana a 1.
        train_freq=(1, "step"), gradient_steps=-1, ent_coef=ent_coef,
        policy_kwargs={"net_arch": [256, 256]}, tensorboard_log=str(tensorboard_dir),
        seed=int(args.seed), device=device, verbose=1,
    )
    print("[V9-INIT] Actor, críticos, entropía y normalizador creados desde cero.")
    bc_losses = behavior_clone(
        model, train_env, data, args.bc_steps, args.bc_batch_size, args.seed + 100_000
    )
    initialize_actor_exploration(model, args.initial_log_std)
    prefill_replay(
        model,
        data,
        args.seed + 200_000,
        transitions_per_phase=args.demo_replay_per_phase,
    )
    model.set_demo_regularization(
        scaled_demo_dataset(model, data), loss_weight=float(args.bc_loss_weight),
        batch_size=int(args.bc_batch_size), seed=int(args.seed + 300_000),
        normalize_q_actor_loss=bool(args.normalize_q_actor_loss),
        q_objective_scale=float(args.q_objective_scale),
        actor_warmup_updates=int(args.actor_warmup_updates),
    )
    cases = dataset_cases(data)
    config = {
        "revision": ZennoV9.REVISION, "dataset": str(dataset_path),
        "action_semantics": "incremental_normalized_all_phases",
        "initialization": "random_fresh_sac_no_checkpoint",
        "init_model": None,
        "init_vecnormalize": None,
        "dataset_transitions": int(data["observations"].shape[0]),
        "dataset_cases": [{"map_id": m, "mass_kg": kg} for m, kg in cases],
        "phase_counts": {
            str(phase): int(np.sum(data["phase_ids"] == phase)) for phase in (0, 1, 2)
        },
        "device": device, "offline_only": bool(args.offline_only),
        "ent_coef": ent_coef,
        "train_freq": [1, "step"],
        "gradient_steps": -1,
        "total_timesteps": int(args.total_timesteps), "bc_steps": int(args.bc_steps),
        "bc_initial_loss": None if not bc_losses else float(bc_losses[0]),
        "bc_final_loss": None if not bc_losses else float(bc_losses[-1]),
        "bc_loss_weight": float(args.bc_loss_weight),
        "bc_loss_weight_final": float(args.bc_loss_weight_final),
        "bc_sampler": "teacher_stage_balanced",
        "demo_replay_per_phase": int(args.demo_replay_per_phase),
        "normalize_q_actor_loss": bool(args.normalize_q_actor_loss),
        "q_objective_scale": float(args.q_objective_scale),
        "actor_warmup_updates": int(args.actor_warmup_updates),
        "initial_log_std": float(args.initial_log_std),
        "require_bc_strict_rate": float(args.require_bc_strict_rate),
        "bc_curriculum": ["P0", "P0_P1", "P0_P1_P2"],
        "online_curriculum": "end_to_end_from_physical_P0_with_real_handoffs",
        "reward_normalized": False,
        "actions_scaled_for_actor_and_replay": True,
    }
    with (output_dir / "training_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)
        file.write("\n")

    eval_callback = V9EvaluationCallback(
        eval_env, train_env, cases, best_dir, args.eval_freq
    )
    # Evaluar el actor clonado en lazo cerrado P0 -> P1 -> P2
    # antes de iniciar las actualizaciones SAC con interacción en línea.
    eval_callback.init_callback(model)
    bc_metrics = eval_callback.evaluate()
    bc_dir = output_dir / "bc_model"
    bc_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(bc_dir / "model_bc"))
    train_env.save(str(bc_dir / "env_zenno_v9_bc.pkl"))
    with (bc_dir / "bc_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(bc_metrics, file, indent=2, ensure_ascii=False)
        file.write("\n")

    bc_strict_rate = float(bc_metrics["strict_success_rate"])
    if bool(args.offline_only) or (
        float(args.require_bc_strict_rate) > 0.0
        and bc_strict_rate + 1.0e-12 < float(args.require_bc_strict_rate)
    ):
        reason = "offline_only" if bool(args.offline_only) else "bc_gate_failed"
        gate_payload = {
            "reason": reason,
            "online_sac_started": False,
            "bc_strict_success_rate": bc_strict_rate,
            "required_bc_strict_rate": float(args.require_bc_strict_rate),
        }
        with (output_dir / "bc_gate.json").open("w", encoding="utf-8") as file:
            json.dump(gate_payload, file, indent=2, ensure_ascii=False)
            file.write("\n")
        print(
            f"[V9-BC-GATE] online_sac=0 strict={100*bc_strict_rate:.1f}% "
            f"required={100*float(args.require_bc_strict_rate):.1f}% "
            f"reason={reason}"
        )
        train_env.close()
        eval_env.close()
        return

    callbacks = CallbackList([
        eval_callback,
        SaveV9Callback(train_env, checkpoint_dir, args.checkpoint_freq),
        DemoWeightAnnealingCallback(
            args.bc_loss_weight, args.bc_loss_weight_final, args.total_timesteps
        ),
    ])
    try:
        if args.offline_only:
            eval_callback.init_callback(model)
            eval_callback.evaluate()
        else:
            model.learn(
                total_timesteps=int(args.total_timesteps), callback=callbacks,
                reset_num_timesteps=True, tb_log_name="v9_incremental", progress_bar=False,
            )
    finally:
        model.save(str(output_dir / "model_final"))
        train_env.save(str(output_dir / "env_zenno_v9_final.pkl"))
        try:
            model.save_replay_buffer(str(output_dir / "replay_final.pkl"))
        except Exception as error:
            print(f"[WARN] No se pudo guardar replay V9: {error}")
        train_env.close()
        eval_env.close()
    print(f"[V9] Entrenamiento finalizado: {output_dir}")


if __name__ == "__main__":
    main()
