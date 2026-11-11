"""R2.1: one 124D/7D SAC actor, frozen encoder/arm, trainable P1 residual."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/vxr21_mpl')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import Actor, SACPolicy
from stable_baselines3.common.utils import update_learning_rate
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from evaluate_zenno_vx_r2 import _SpaceOnlyEnv
from zenno_vx_r2 import ZennoVXR2

SOURCE = ROOT/'models/vx_r1_r2_sequence_20260909/r2_student/final/model.zip'


class ZennoVXR21(ZennoVXR2):
    """Same R2 dynamics, reward, observation, PP lookahead and phase budgets."""
    REVISION = 'VX_R2_1_SINGLE_ACTOR_P1_SAC_FROZEN_ARM_V1'


class ResidualActor(Actor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.use_sde or self.action_space.shape != (7,):
            raise ValueError('Required: Gaussian seven-action actor without gSDE')
        self.register_buffer('p1_midpoint', torch.tensor(.5))
        self.register_buffer('authority', torch.tensor(.5))
        self.protect_arm()

    def protect_arm(self):
        for p in self.parameters():
            p.requires_grad_(False)
        for handle in getattr(self, '_masks', []):
            handle.remove()
        self._masks = []
        for layer in (self.mu, self.log_std):
            for p in layer.parameters():
                p.requires_grad_(True)
                def mask(gradient):
                    result = gradient.clone()
                    result[:5] = 0
                    return result
                self._masks.append(p.register_hook(mask))

    def configure(self, norm, authority):
        if not 0 < authority <= 1 or norm.training or not norm.norm_obs:
            raise ValueError('Authority in (0,1], frozen observation normalization required')
        low = np.zeros((1,124), np.float32)
        high = low.copy(); high[0,72] = 1
        a, b = norm.normalize_obs(low)[0,72], norm.normalize_obs(high)[0,72]
        if not np.isfinite([a,b]).all() or b-a < 1e-5:
            raise ValueError('P1 cannot be distinguished after normalization')
        self.p1_midpoint.fill_(float((a+b)/2))
        self.authority.fill_(authority)

    def _action(self, obs, deterministic):
        mean, log_std, _ = self.get_action_dist_params(obs)
        active = (obs[:,72:73] > self.p1_midpoint).to(mean.dtype)
        distribution = torch.distributions.Normal(mean[:,5:], log_std[:,5:].clamp(-4.6,-1.6).exp())
        latent = mean[:,5:] if deterministic else distribution.rsample()
        unit = latent.tanh()
        # Exact Jacobian for the scaled tanh distribution; entropy only in P1.
        log_prob = (distribution.log_prob(latent)-torch.log(1-unit.square()+1e-6)
                    -torch.log(self.authority)).sum(1)*active[:,0]
        return torch.cat((mean[:,:5].tanh().detach(), self.authority*unit*active), 1), log_prob

    def forward(self, obs, deterministic=False):
        return self._action(obs, deterministic)[0]

    def action_log_prob(self, obs):
        return self._action(obs, False)


class ResidualPolicy(SACPolicy):
    def make_actor(self, features_extractor=None):
        return ResidualActor(**self._update_features_extractor(self.actor_kwargs, features_extractor)).to(self.device)


class ResidualSAC(SAC):
    def __init__(self, *args, actor_lr=1e-5, critic_lr=3e-4, critic_warmup=2000, **kwargs):
        self.actor_lr, self.critic_lr, self.critic_warmup = actor_lr, critic_lr, critic_warmup
        super().__init__(*args, **kwargs)

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        # Never use SB3 uniform seven-dimensional exploration: arm is protected.
        return super()._sample_action(0, action_noise, n_envs)

    def _update_learning_rate(self, optimizers):
        update_learning_rate(self.actor.optimizer, self.actor_lr)
        update_learning_rate(self.critic.optimizer, self.critic_lr)

    def train(self, gradient_steps, batch_size=256):
        if gradient_steps != 1:
            raise ValueError('One update per call makes the critic-only boundary explicit')
        if self._n_updates >= self.critic_warmup:
            return super().train(gradient_steps, batch_size)
        step = self.actor.optimizer.step
        self.actor.optimizer.step = lambda *args, **kwargs: None
        try:
            return super().train(gradient_steps, batch_size)
        finally:
            self.actor.optimizer.step = step


def protected_state(actor):
    return {k:(v[:5] if k in ('mu.weight','mu.bias','log_std.weight','log_std.bias') else v).detach().cpu().clone()
            for k,v in actor.state_dict().items()}


def assert_protected(actor, expected):
    actual = protected_state(actor)
    if actual.keys() != expected.keys() or any(not torch.equal(actual[k],v) for k,v in expected.items()):
        raise RuntimeError('Frozen arm/encoder/authority was modified')


def load_norm(path):
    norm = VecNormalize.load(path, DummyVecEnv([_SpaceOnlyEnv]))
    norm.training, norm.norm_reward = False, False
    if norm.obs_rms.mean.shape != (124,):
        raise ValueError('Expected 124D normalizer')
    return norm


def build(env, norm, args):
    source = SAC.load(args.source, device='cpu', custom_objects={'buffer_size':1})
    if source.observation_space.shape != (124,) or source.action_space.shape != (7,):
        raise ValueError('Expected R2 124D/7D source')
    model = ResidualSAC(ResidualPolicy, env, device=args.device, seed=args.seed,
        actor_lr=args.actor_lr, critic_lr=args.critic_lr, critic_warmup=args.critic_warmup,
        learning_rate=args.critic_lr, buffer_size=args.buffer_size, learning_starts=args.prefill,
        batch_size=args.batch_size, gamma=.9995, tau=.005, train_freq=1, gradient_steps=1,
        ent_coef=.001, target_entropy=-2., policy_kwargs={'net_arch':[256,256]},
        replay_buffer_kwargs={'handle_timeout_termination':False}, verbose=0)
    result = model.actor.load_state_dict(source.actor.state_dict(), strict=False)
    if set(result.missing_keys) != {'authority','p1_midpoint'} or result.unexpected_keys:
        raise ValueError(f'Incompatible actor: {result}')
    with torch.no_grad():
        model.actor.log_std.weight[5:].zero_()
        model.actor.log_std.bias[5:].fill_(np.log(.15))
    model.actor.configure(norm,args.authority)
    model.actor.protect_arm()
    return model, source
