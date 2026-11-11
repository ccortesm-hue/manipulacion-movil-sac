"""Single coupled SAC policy; unique module name prevents historical-policy collisions."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/vxr21_coupled_mpl')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np
import torch
from torch.nn import functional as F
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import Actor, SACPolicy
from stable_baselines3.common.utils import update_learning_rate, polyak_update
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from evaluate_zenno_vx_r2 import _SpaceOnlyEnv
from zenno_vx_r2 import ZennoVXR2

SOURCE = ROOT/'models/vx_r1_r2_sequence_20260909/r2_student/final/model.zip'


class ZennoVXR21(ZennoVXR2):
    """Unchanged physics/reward, 124 observations and seven incremental actions."""
    REVISION = 'VX_R2_1_COUPLED_SAC_ARM_P1_P2_BASE_P1_V1'


class ResidualActor(Actor):
    """One actor with a frozen prior path and a trainable joint correction path.

    P0 is exactly the prior policy. In P1/P2 the arm action is
    tanh(prior_logit + arm_authority*tanh(z_arm)); base is authority*tanh(z_base)
    in P1 only. The final arm action remains in [-1,1], hence <=0.008 rad/step.
    The correction network receives all observations, not a separate teacher.
    """
    HISTORY = (30, 31, 97, 98, 99, 100)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.use_sde or self.action_space.shape != (7,):
            raise ValueError('Seven-action Gaussian actor without gSDE required')
        for p in self.parameters():
            p.requires_grad_(False)
        self.coupled_features = torch.nn.Sequential(
            torch.nn.Linear(124, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 128), torch.nn.ReLU())
        self.coupled_mu = torch.nn.Linear(128, 7)
        torch.nn.init.zeros_(self.coupled_mu.weight)
        torch.nn.init.zeros_(self.coupled_mu.bias)
        self.coupled_log_std = torch.nn.Parameter(
            torch.tensor([np.log(.08)]*5+[np.log(.15)]*2, dtype=torch.float32))
        self.register_buffer('p1_midpoint', torch.tensor(.5))
        self.register_buffer('p2_midpoint', torch.tensor(.5))
        self.register_buffer('released_midpoint', torch.tensor(.5))
        self.register_buffer('authority', torch.tensor(.5))
        self.register_buffer('arm_authority', torch.tensor(.5))
        self.register_buffer('history_zero', torch.zeros(len(self.HISTORY)))

    def configure(self, norm, authority, arm_authority):
        if norm.training or not norm.norm_obs or not (0 < authority <= 1 and 0 < arm_authority <= 1):
            raise ValueError('Frozen observation normalizer and authorities in (0,1] required')
        low = np.zeros((1,124), np.float32)
        for index, buffer in ((72,self.p1_midpoint),(73,self.p2_midpoint),(52,self.released_midpoint)):
            high = low.copy(); high[0,index] = 1
            a,b = norm.normalize_obs(low)[0,index], norm.normalize_obs(high)[0,index]
            if not np.isfinite([a,b]).all() or b-a < 1e-5:
                raise ValueError('Phase indistinguishable after normalization')
            buffer.fill_(float((a+b)/2))
        self.authority.fill_(authority)
        self.arm_authority.fill_(arm_authority)
        self.history_zero.copy_(torch.as_tensor(norm.normalize_obs(low)[0,list(self.HISTORY)],
                                               device=self.history_zero.device))

    def gates(self, obs):
        p1 = (obs[:,72:73] > self.p1_midpoint).to(obs.dtype)
        p2 = ((obs[:,73:74] > self.p2_midpoint) &
              (obs[:,52:53] <= self.released_midpoint)).to(obs.dtype)
        arm = torch.maximum(p1,p2).expand(-1,5).clone()
        # Inherited environment locks J1/J5 during transport. Do not explore
        # commands that physics would discard; these joints remain trainable in P2.
        arm[:,0:1]=p2
        arm[:,4:5]=p2
        return arm, p1

    def prior_logits(self, obs):
        # The historical arm saw these command-history channels at zero.
        # Only the prior path is masked; the trainable path sees the real history.
        reference = obs.clone()
        reference[:,list(self.HISTORY)] = self.history_zero
        with torch.no_grad():
            return self.get_action_dist_params(reference)[0][:,:5]

    def _action(self, obs, deterministic):
        arm_gate, base_gate = self.gates(obs)
        prior = self.prior_logits(obs)
        mean = self.coupled_mu(self.coupled_features(obs))
        std = self.coupled_log_std.clamp(-4.6, -1.6).exp()
        distribution = torch.distributions.Normal(mean, std)
        z = mean if deterministic else distribution.rsample()
        unit = z.tanh()
        arm = (prior + self.arm_authority*unit[:,:5]*arm_gate).tanh()
        base = self.authority*unit[:,5:]*base_gate
        # Exact change-of-variables density on active dimensions; no entropy in P0.
        log_tanh = 2*(np.log(2.)-z-F.softplus(-2*z))
        lp = distribution.log_prob(z)-log_tanh
        arm_logits = prior+self.arm_authority*unit[:,:5]
        log_arm_tanh = 2*(np.log(2.)-arm_logits-F.softplus(-2*arm_logits))
        arm_lp = (lp[:,:5]-torch.log(self.arm_authority)-log_arm_tanh)*arm_gate
        base_lp = (lp[:,5:]-torch.log(self.authority))*base_gate
        return torch.cat((arm,base),1), arm_lp.sum(1)+base_lp.sum(1)

    def forward(self, obs, deterministic=False):
        return self._action(obs,deterministic)[0]

    def action_log_prob(self, obs):
        return self._action(obs,False)

    def arm_anchor_loss(self, obs):
        gate,_ = self.gates(obs)
        actual = self.forward(obs,True)[:,:5]
        prior = self.prior_logits(obs).tanh()
        return ((actual-prior).square()*gate).sum()/gate.sum().clamp_min(1)


class ResidualPolicy(SACPolicy):
    def make_actor(self, features_extractor=None):
        return ResidualActor(**self._update_features_extractor(self.actor_kwargs,features_extractor)).to(self.device)


class ResidualSAC(SAC):
    """SAC with fixed entropy, critic warmup and a soft arm-prior regularizer."""
    def __init__(self,*args,actor_lr=1e-5,critic_lr=3e-4,critic_warmup=2000,
                 arm_anchor_coef=2.,**kwargs):
        self.actor_lr,self.critic_lr,self.critic_warmup=actor_lr,critic_lr,critic_warmup
        self.arm_anchor_coef=arm_anchor_coef
        super().__init__(*args,**kwargs)

    def _sample_action(self,learning_starts,action_noise=None,n_envs=1):
        return super()._sample_action(0,action_noise,n_envs)

    def _update_learning_rate(self,optimizers):
        update_learning_rate(self.actor.optimizer,self.actor_lr)
        update_learning_rate(self.critic.optimizer,self.critic_lr)

    def train(self,gradient_steps,batch_size=256):
        # Adapted from the installed SB3 2.7 SAC update, with an explicit anchor.
        if gradient_steps != 1 or self.ent_coef_optimizer is not None or self.use_sde:
            raise ValueError('One update, fixed entropy and no gSDE required')
        self.policy.set_training_mode(True)
        self._update_learning_rate([])
        replay=self.replay_buffer.sample(batch_size,env=self._vec_normalize_env)
        discounts=replay.discounts if replay.discounts is not None else self.gamma
        alpha=self.ent_coef_tensor
        with torch.no_grad():
            next_action,next_lp=self.actor.action_log_prob(replay.next_observations)
            next_q=torch.cat(self.critic_target(replay.next_observations,next_action),1).min(1,keepdim=True).values
            target=replay.rewards+(1-replay.dones)*discounts*(next_q-alpha*next_lp[:,None])
        q=self.critic(replay.observations,replay.actions)
        critic_loss=.5*sum(F.mse_loss(value,target) for value in q)
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
        actor_loss=anchor=torch.zeros((),device=self.device)
        if self._n_updates >= self.critic_warmup:
            for p in self.critic.parameters(): p.requires_grad_(False)
            try:
                actions,lp=self.actor.action_log_prob(replay.observations)
                q_pi=torch.cat(self.critic(replay.observations,actions),1).min(1).values
                anchor=self.actor.arm_anchor_loss(replay.observations)
                actor_loss=(alpha*lp-q_pi).mean()+self.arm_anchor_coef*anchor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
            finally:
                for p in self.critic.parameters(): p.requires_grad_(True)
        if self._n_updates % self.target_update_interval == 0:
            polyak_update(self.critic.parameters(),self.critic_target.parameters(),self.tau)
            polyak_update(self.batch_norm_stats,self.batch_norm_stats_target,1.)
        self._n_updates+=1
        for name,value in dict(n_updates=self._n_updates,critic_loss=critic_loss.item(),
                               actor_loss=actor_loss.item(),arm_anchor_mse=anchor.item(),
                               ent_coef=alpha.item()).items():
            self.logger.record('train/'+name,value)


def protected_state(actor):
    return {k:v.detach().cpu().clone() for k,v in actor.state_dict().items()
            if not k.startswith('coupled_')}


def assert_protected(actor,expected):
    actual=protected_state(actor)
    if actual.keys()!=expected.keys() or any(not torch.equal(actual[k],v) for k,v in expected.items()):
        raise RuntimeError('Protected prior/P0/authority was modified')


def load_norm(path):
    norm=VecNormalize.load(path,DummyVecEnv([_SpaceOnlyEnv]))
    norm.training,norm.norm_reward=False,False
    if norm.obs_rms.mean.shape!=(124,): raise ValueError('Expected 124D normalizer')
    return norm


def build(env,norm,args):
    source=SAC.load(args.source,device='cpu',custom_objects={'buffer_size':1})
    if source.observation_space.shape!=(124,) or source.action_space.shape!=(7,):
        raise ValueError('Expected historical R2 source')
    if torch.count_nonzero(source.actor.mu.weight[5:]) or torch.count_nonzero(source.actor.mu.bias[5:]):
        raise ValueError('This protocol requires the verified zero-base historical R2 source')
    # Source had std=.01 on prior base commands: values >=.1 were clipped at 10.
    # New path uses fixed physical scaling; the prior path receives normalized zero.
    norm.obs_rms.mean[30:32]=0.
    norm.obs_rms.var[30:32]=.5**2
    model=ResidualSAC(ResidualPolicy,env,device=args.device,seed=args.seed,
        actor_lr=args.actor_lr,critic_lr=args.critic_lr,critic_warmup=args.critic_warmup,
        arm_anchor_coef=args.arm_anchor_coef,learning_rate=args.critic_lr,
        buffer_size=args.buffer_size,learning_starts=args.prefill,batch_size=args.batch_size,
        gamma=.9995,tau=.005,train_freq=1,gradient_steps=1,ent_coef=.001,
        policy_kwargs={'net_arch':[256,256]},
        replay_buffer_kwargs={'handle_timeout_termination':False},verbose=0)
    result=model.actor.load_state_dict(source.actor.state_dict(),strict=False)
    allowed={'p1_midpoint','p2_midpoint','released_midpoint','authority','arm_authority','history_zero'}
    if result.unexpected_keys or any(not(k.startswith('coupled_') or k in allowed) for k in result.missing_keys):
        raise ValueError(f'Incompatible actor: {result}')
    model.actor.configure(norm,args.authority,args.arm_authority)
    return model,source
