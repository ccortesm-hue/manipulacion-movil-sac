"""VX R3: SAC curricular desde R1; residual geométrico y adaptación del brazo."""
import os
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl_vx_r25')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'VX'))

import numpy as np
import torch
from torch import nn
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import Actor, SACPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import update_learning_rate
from MM_DATA_VXR2 import ZennoVXR2
sys.path.insert(0, str(ROOT))
from extract_vx_place_robust import evaluation_environment
from vx_place_protocol import Recorder
import json

OBS_DIM, ACTION_DIM = 130, 6
ARM_HISTORY = (30, 31, 97, 98, 99, 100)


class LookaheadEnvironment(ZennoVXR2):
    """La acción pública es [dq0..dq4, a_L]; jamás inyecta delta-v/delta-omega."""
    REVISION = 'VX_R3_CURRICULUM_LOOKAHEAD_V1'
    ACTION_SEMANTICS = 'incremental_arm_pure_pursuit_lookahead_residual'
    L_NOMINAL, L_MIN, L_MAX = .30, .15, .60
    L_FILTER_TAU, L_RATE_MAX, L_TERMINAL_BLEND = .25, .30, .30
    L_MAG_COST, L_RATE_COST = .10, .02

    def __init__(self, **kwargs):
        self._reset_lookahead()
        super().__init__(**kwargs)
        if not np.isclose(self.LOOKAHEAD_DIST, self.L_NOMINAL):
            raise ValueError('La fuente debe tener lookahead nominal 0.30 m')
        self._legacy_public_space = self.action_space
        self.action_space = spaces.Box(-1., 1., (ACTION_DIM,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)

    def _reset_lookahead(self):
        self._lh_request = 0.
        self._lh_filtered = self.L_NOMINAL
        self._lh_applied = self.L_NOMINAL
        self._lh_previous = self.L_NOMINAL
        self._lh_rate = 0.
        self._lh_target = None
        self._lh_used = False
        self._lh_updated = False

    def reset_model(self):
        self._reset_lookahead()
        return super().reset_model()

    def lookahead_available(self):
        return bool(int(self.phase_state) == 1 and self._plan_ready
            and len(self._waypoints) >= 2 and self.base_phase >= 3
            and not self._reverse_active and not self._initial_aligning and not self._nav_done)

    def _choose_lookahead(self, x, y, start_idx, lookahead):
        # La misma función de proyección/interpolación original, sin regenerar rutas.
        if not self._lh_updated:
            enabled = self.lookahead_available()
            requested = (self.L_NOMINAL + self._lh_request *
                ((self.L_MAX-self.L_NOMINAL) if self._lh_request >= 0
                 else (self.L_NOMINAL-self.L_MIN))) if enabled else self.L_NOMINAL
            change = self.dt/(self.L_FILTER_TAU+self.dt)*(requested-self._lh_filtered)
            self._lh_filtered = float(np.clip(self._lh_filtered + np.clip(change,
                -self.L_RATE_MAX*self.dt, self.L_RATE_MAX*self.dt), self.L_MIN, self.L_MAX))
            endpoint = float(np.linalg.norm(np.asarray(self._waypoints[-1])-np.array([x,y]))) if enabled else 0.
            u = float(np.clip((endpoint-self.PURE_PURSUIT_TERMINAL_POSE_DIST_M)/self.L_TERMINAL_BLEND, 0, 1))
            authority = u*u*(3-2*u) if enabled else 0.
            desired = self.L_NOMINAL + authority*(self._lh_filtered-self.L_NOMINAL)
            self._lh_applied = float(self._lh_previous + np.clip(desired-self._lh_previous,
                -self.L_RATE_MAX*self.dt, self.L_RATE_MAX*self.dt))
            self._lh_rate = (self._lh_applied-self._lh_previous)/self.dt
            self._lh_used = bool(enabled and (authority > 0 or abs(self._lh_applied-self.L_NOMINAL)>1e-10))
            self._lh_updated = True
        result = super()._choose_lookahead(x, y, start_idx, self._lh_applied)
        self._lh_target = np.asarray(result[:2], dtype=float)
        return result

    def _control_base_vel(self, Ts, v_des, omega_des, u_lin=0., u_ang=0., **kwargs):
        if abs(u_lin)>1e-12 or abs(u_ang)>1e-12:
            raise RuntimeError('R3 no admite residuales aditivos de velocidad')
        return super()._control_base_vel(Ts, v_des, omega_des, u_lin=0., u_ang=0., **kwargs)

    def _get_obs(self):
        original = super()._get_obs()
        theta = self._base_yaw()
        delta = np.zeros(2) if self._lh_target is None else self._lh_target-self._base_xy()
        rotation = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
        goal_body = rotation@delta
        enabled = self.lookahead_available()
        if enabled:
            endpoint = np.linalg.norm(np.asarray(self._waypoints[-1])-self._base_xy())
            enabled = endpoint > self.PURE_PURSUIT_TERMINAL_POSE_DIST_M
        obs = np.r_[original, self._lh_filtered, self._lh_applied, goal_body,
                    self._lh_rate, float(enabled)].astype(np.float32)
        if obs.shape != (OBS_DIM,) or not np.isfinite(obs).all():
            raise RuntimeError('Observación R3 inválida')
        return obs

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise ValueError('R3 requiere seis acciones finitas')
        action = np.clip(action, -1, 1)
        phase = int(self.phase_state)
        self._lh_request = float(action[5]) if phase == 1 else 0.
        self._lh_previous = self._lh_applied
        self._lh_updated = self._lh_used = False
        self._lh_rate = 0.
        # Compatibilidad interna con el controlador heredado, NO dos acciones RL más.
        legacy_action = np.r_[action[:5], 0., 0.].astype(np.float32)
        public = self.action_space
        self.action_space = self._legacy_public_space
        try:
            obs, reward, term, trunc, info = super().step(legacy_action)
        finally:
            self.action_space = public
        if not self._lh_updated:
            self._lh_filtered = self._lh_applied = self.L_NOMINAL
            self._lh_target = None
        if abs(info.get('vx_residual_cost', 0.)) > 1e-9:
            raise RuntimeError('Se activó indebidamente el coste residual de torque/velocidad')
        # Penalización con unidades explícitas, sin derivar el torque PID.
        magnitude = -self.dt*self.L_MAG_COST*((self._lh_applied-self.L_NOMINAL)/.30)**2 if self._lh_used else 0.
        rate = -self.dt*self.L_RATE_COST*(self._lh_rate/self.L_RATE_MAX)**2 if self._lh_used else 0.
        reward = float(reward+magnitude+rate)
        self.reward = reward
        for name in ('pp_nominal_reference','pp_corrected_reference','pp_reference_delta'):
            info.pop(name, None)
        info.update(environment_revision=self.REVISION, experiment_revision=self.REVISION,
            action_dim=ACTION_DIM, action_semantics=self.ACTION_SEMANTICS,
            residual_level='pure_pursuit_lookahead', velocity_residual_enabled=False,
            torque_residual_enabled=False, lookahead_request=self._lh_request,
            lookahead_nominal_m=self.L_NOMINAL, lookahead_filtered_m=self._lh_filtered,
            lookahead_applied_m=self._lh_applied, lookahead_rate_m_s=self._lh_rate,
            lookahead_active=self._lh_used, lookahead_magnitude_cost=magnitude,
            lookahead_rate_cost=rate, pp_generated_reference=self._r2_pp_corrected.tolist(),
            lookahead_target_xy=None if self._lh_target is None else self._lh_target.tolist())
        return self._get_obs(), reward, term, trunc, info


class PhysicalFeatures(BaseFeaturesExtractor):
    """Escalas congeladas apropiadas al residual, embebidas en actor y críticos."""
    def __init__(self, observation_space):
        super().__init__(observation_space, OBS_DIM)
        self.register_buffer('center', torch.zeros(OBS_DIM))
        self.register_buffer('scale', torch.ones(OBS_DIM))

    def configure(self, norm):
        mean = np.r_[norm.obs_rms.mean, .3,.3,0.,0.,0.,0.]
        std = np.r_[np.sqrt(norm.obs_rms.var+norm.epsilon), .3,.3,.6,.6,.3,1.]
        std[[30,31,99,100,101]] = 1.
        std[[97,98]] = np.maximum(std[[97,98]], .2)
        self.center.copy_(torch.as_tensor(mean, device=self.center.device, dtype=torch.float32))
        self.scale.copy_(torch.as_tensor(std, device=self.scale.device, dtype=torch.float32))

    def forward(self, obs):
        return ((obs-self.center)/self.scale).clamp(-10.,10.)



class ZennoVXR3(evaluation_environment(LookaheadEnvironment)):
    """Misma colocación física estable en entrenamiento y evaluación, sin home."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.W_P2_HOME_PROGRESS = self.W_P2_HOME_EVENT = 0.

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        route = json.loads((ROOT/'RUTAS_SAC'/f'map_{self.FIXED_MAP_ID}.json').read_text())['path']
        self.place_recorder = Recorder(self.model, self.data, np.asarray(route))
        self.phase_counts = [0, 0, 0]
        return obs, info

    def step(self, action):
        phase = int(self.phase_state)
        obs, reward, _, _, info = super().step(action)
        self.phase_counts[phase] += 1
        stop = self.place_recorder.end_interval(phase, phase == 1,
            native_deadline=self.phase_counts[phase] >= self.phase_budgets[phase])
        failures = [d['reason'] for d in self.place_native_diagnostics
                    if d['reason'] not in ('place_success_arm_home', 'phase2_home_budget')]
        if stop is None and failures:
            stop = failures[-1]
        if stop is None and self.phase_counts[phase] >= self.phase_budgets[phase] and int(self.phase_state) == phase:
            stop = f'phase{phase}_budget'
        if stop is not None and stop != 'stable_place':
            penalty = float(self.PHASE_FAILURE_PENALTIES[phase])
            reward += penalty
            info['phase_failure_penalty'] = penalty
        info.update(success_place=stop == 'stable_place', task_stop=stop,
                    TimeLimit_truncated=False, home_required=False)
        return obs, float(reward), stop is not None, False, info


class CurriculumActor(Actor):
    """Un actor 6D: prior del brazo R1 y correcciones por fase; no profesores."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        width = self.mu.in_features
        self.mu = nn.Linear(width, 1)
        self.log_std = nn.Linear(width, 1)
        self.arm_encoder = nn.Sequential(nn.Linear(124,256),nn.ReLU(),
                                         nn.Linear(256,256),nn.ReLU())
        self.arm_mu = nn.Linear(256,5)
        self.corrections = nn.ModuleList([
            nn.Sequential(nn.Linear(130,128),nn.ReLU(),nn.Linear(128,128),
                          nn.ReLU(),nn.Linear(128,10)) for _ in range(3)])
        for net in self.corrections:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
            with torch.no_grad(): net[-1].bias[5:].fill_(np.log(.1))
        nn.init.zeros_(self.mu.weight); nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.log_std.weight)
        nn.init.constant_(self.log_std.bias, np.log(.1))
        self.register_buffer('arm_center', torch.zeros(124,dtype=torch.float64))
        self.register_buffer('arm_scale', torch.ones(124,dtype=torch.float64))
        self.register_buffer('arm_clip', torch.tensor(10.))
        self.register_buffer('curriculum_stage', torch.tensor(0,dtype=torch.long))
        self.register_buffer('arm_authority', torch.tensor(.15))
        for module in (self.arm_encoder,self.arm_mu):
            module.requires_grad_(False)

    def initialize_arm(self, source, norm):
        if source.observation_space.shape != (124,) or source.action_space.shape != (7,):
            raise ValueError('Se requiere el checkpoint original R1 124D/7D')
        self.arm_encoder.load_state_dict(source.actor.latent_pi.state_dict(), strict=True)
        with torch.no_grad():
            self.arm_mu.weight.copy_(source.actor.mu.weight[:5])
            self.arm_mu.bias.copy_(source.actor.mu.bias[:5])
            self.arm_center.copy_(torch.as_tensor(norm.obs_rms.mean,device=self.device))
            self.arm_scale.copy_(torch.as_tensor(np.sqrt(norm.obs_rms.var+norm.epsilon),device=self.device))
            self.arm_clip.fill_(norm.clip_obs)

    def prior_logits(self, obs):
        raw = obs[:,:124].clone()
        # Los canales de residual de torque NO representan el nuevo lookahead.
        raw[:,list(ARM_HISTORY)] = 0.
        z = ((raw-self.arm_center)/self.arm_scale).clamp(-self.arm_clip,self.arm_clip)
        return self.arm_mu(self.arm_encoder(z.to(obs.dtype))).detach()

    def _action(self, obs, deterministic):
        features = self.extract_features(obs, self.features_extractor)
        phase = obs[:,71:74].argmax(1)
        stage = int(self.curriculum_stage)
        outputs = torch.stack([net(features) for net in self.corrections],1)
        pars = outputs[torch.arange(len(obs),device=obs.device),phase]
        arm_mean, arm_std = pars[:,:5], pars[:,5:].clamp(-4.6,-1.2).exp()
        arm_dist = torch.distributions.Normal(arm_mean,arm_std)
        z = arm_mean if deterministic else arm_dist.rsample()
        correction = z.tanh()
        mask = torch.zeros_like(correction)
        if stage >= 2:
            mask[phase == 1,1:4] = 1. # J1/J5 están bloqueadas por el ambiente P1.
            mask[(phase == 2) & (obs[:,52] < .5)] = 1.
        if stage >= 3:
            mask[phase == 0] = 1.
        arm = (self.prior_logits(obs)+self.arm_authority*correction*mask).tanh()
        jac = (torch.log(self.arm_authority)+torch.log(1-correction.square()+1e-6)
               +torch.log(1-arm.square()+1e-6))
        arm_lp = ((arm_dist.log_prob(z)-jac)*mask).sum(1)
        latent = self.latent_pi(features)
        mean = self.mu(latent)
        dist = torch.distributions.Normal(mean,self.log_std(latent).clamp(-4.6,-1.2).exp())
        sample = mean if deterministic else dist.rsample()
        residual = sample.tanh()
        available = (obs[:,129:130] > .5).to(obs.dtype)
        lp = (dist.log_prob(sample)-torch.log(1-residual.square()+1e-6)).sum(1)
        return torch.cat((arm,residual*available),1), arm_lp+lp*available[:,0]

    def forward(self, obs, deterministic=False):
        return self._action(obs,deterministic)[0]

    def action_log_prob(self, obs):
        return self._action(obs,False)


class CurriculumPolicy(SACPolicy):
    def make_actor(self, features_extractor=None):
        return CurriculumActor(**self._update_features_extractor(
            self.actor_kwargs,features_extractor)).to(self.device)


class CurriculumSAC(SAC):
    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        # Incluso al llenar replay: prior + exploración acotada, jamás brazo aleatorio.
        return super()._sample_action(0,action_noise,n_envs)
