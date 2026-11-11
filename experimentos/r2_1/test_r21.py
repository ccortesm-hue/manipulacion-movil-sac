"""Contract tests: frozen arm, restricted exploration, critic warmup, reload, gates."""
from types import SimpleNamespace
import numpy as np
import pytest
from VX_R2_1 import (SOURCE,build,load_norm,protected_state,assert_protected,
                     ResidualSAC,torch)
from TRAIN_R21 import TrainingEnv,validate_source
from EVALUATE_R21 import assess,METRICS,METHODS
from stable_baselines3.common.logger import configure


@pytest.fixture
def setup():
    torch.set_num_threads(1)
    validate_source(SOURCE)
    norm=load_norm(SOURCE.with_name('vecnormalize.pkl'))
    env=TrainingEnv(norm,123)
    args=SimpleNamespace(source=SOURCE,device='cpu',seed=123,actor_lr=1e-4,
        critic_lr=3e-4,critic_warmup=2,buffer_size=64,prefill=16,batch_size=8,authority=.5)
    model,source=build(env,norm,args); env.model=model
    model.set_logger(configure(None,[]))
    try: yield model,source,norm,env
    finally: env.close(); norm.close()


def test_phase_exploration_and_source_arm(setup):
    model,source,norm,env=setup
    for phase in range(3):
        raw=np.zeros((8,124),np.float32); raw[:,71+phase]=1
        obs=norm.normalize_obs(raw)
        det=model.predict(obs,deterministic=True)[0]
        original=source.predict(obs,deterministic=True)[0]
        stochastic=model.predict(obs,deterministic=False)[0]
        np.testing.assert_allclose(det[:,:5],original[:,:5],atol=1e-6,rtol=0)
        np.testing.assert_array_equal(det[:,:5],stochastic[:,:5])
        if phase!=1: assert not np.any(stochastic[:,5:])
        else: assert np.any(stochastic[:,5:]) and np.max(np.abs(stochastic[:,5:]))<=.5


def test_updates_protect_arm_and_reload(setup,tmp_path):
    model,_,norm,_=setup
    raw=np.zeros((1,124),np.float32);raw[0,72]=1
    obs=norm.normalize_obs(raw)
    frozen=protected_state(model.actor)
    original={k:v.clone() for k,v in model.actor.state_dict().items()}
    for _ in range(32):
        a=model.predict(obs,deterministic=False)[0]
        model.replay_buffer.add(obs,obs,a,np.array([.3],np.float32),np.array([0.]),[{}])
    for _ in range(2): model.train(1,8)
    assert all(torch.equal(v,model.actor.state_dict()[k]) for k,v in original.items())
    for _ in range(5): model.train(1,8)
    assert_protected(model.actor,frozen)
    assert not torch.equal(original['mu.weight'][5:],model.actor.mu.weight[5:])
    model.save(tmp_path/'model.zip')
    loaded=ResidualSAC.load(tmp_path/'model.zip',device='cpu',custom_objects={'buffer_size':1})
    np.testing.assert_array_equal(model.predict(obs,deterministic=True)[0],loaded.predict(obs,deterministic=True)[0])


def rows():
    return [dict(case_id='a',map_id=1,method=m,success_place=True,
                 reference_delta_rms_v_omega=[.01,.01],**{k:1. for k in METRICS}) for m in METHODS]


def test_selection_requires_benefit_activity_and_complete_pairs():
    r=rows()
    assert not assess(r)['qualified']
    r[2].update({k:.98 for k in METRICS})
    assert assess(r)['qualified']
    r[2]['reference_delta_rms_v_omega']=[0.,0.]
    assert not assess(r)['qualified']
    r[2]['reference_delta_rms_v_omega']=[.01,.01]
    r[2]['cube_goal_dist_xy_m']=1.06
    assert not assess(r)['qualified']
    with pytest.raises(ValueError): assess(r[:2])


def test_real_p0_handoff(setup):
    model,source,norm,env=setup
    mean=norm.obs_rms.mean.copy(); var=norm.obs_rms.var.copy()
    obs,info=env.reset(seed=234)
    assert info['real_handoff'] and env.env.phase_state==1
    np.testing.assert_allclose(model.predict(obs,deterministic=True)[0][:5],
                               source.predict(obs,deterministic=True)[0][:5],atol=1e-6,rtol=0)
    np.testing.assert_array_equal(mean,norm.obs_rms.mean)
    np.testing.assert_array_equal(var,norm.obs_rms.var)


def test_short_physical_sac_rollout(setup):
    model,_,_,env=setup
    model.learning_starts=8
    frozen=protected_state(model.actor)
    model.learn(24,reset_num_timesteps=False,log_interval=1)
    assert model.num_timesteps==24 and model._n_updates==16
    assert env.env.phase_state==1
    assert_protected(model.actor,frozen)
