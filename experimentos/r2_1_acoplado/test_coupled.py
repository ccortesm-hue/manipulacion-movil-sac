"""Coupled policy contracts; tiny synthetic/physical tests, not performance claims."""
from types import SimpleNamespace
import numpy as np
import pytest
from VX_R2_1_COUPLED import SOURCE,build,load_norm,protected_state,assert_protected,ResidualSAC,torch
from TRAIN_R21 import TrainingEnv,validate_source
from EVALUATE_R21 import assess,METRICS,METHODS,evaluate,make_cases,evaluation_actors
from stable_baselines3.common.logger import configure


@pytest.fixture
def setup():
    torch.set_num_threads(1)
    validate_source(SOURCE)
    norm=load_norm(SOURCE.with_name('vecnormalize.pkl'))
    env=TrainingEnv(norm,123)
    args=SimpleNamespace(source=SOURCE,device='cpu',seed=123,actor_lr=1e-4,
        critic_lr=3e-4,critic_warmup=2,buffer_size=64,prefill=16,batch_size=8,
        authority=.5,arm_authority=.5,arm_anchor_coef=2.)
    model,source=build(env,norm,args);env.model=model
    model.set_logger(configure(None,[]))
    try: yield model,source,norm,env
    finally: env.close();norm.close()


def phase_obs(norm,phase,n=8):
    raw=np.zeros((n,124),np.float32);raw[:,71+phase]=1
    return norm.normalize_obs(raw)


def test_initial_policy_exploration_limits_and_history_scale(setup):
    model,source,norm,_=setup
    assert np.allclose(norm.obs_rms.var[30:32],.25)
    for phase in range(3):
        obs=phase_obs(norm,phase)
        det=model.predict(obs,deterministic=True)[0]
        original=source.predict(obs,deterministic=True)[0]
        sample=model.predict(obs,deterministic=False)[0]
        np.testing.assert_allclose(det[:,:5],original[:,:5],atol=1e-6,rtol=0)
        assert not np.any(det[:,5:])
        assert np.max(np.abs(sample[:,:5]))<=1 # <=.008 rad for every joint
        assert np.max(np.abs(sample[:,5:]))<=.5
        if phase==0: np.testing.assert_array_equal(det,sample)
        else: assert np.any(sample[:,:5]!=det[:,:5])
        if phase==1: np.testing.assert_array_equal(det[:,[0,4]],sample[:,[0,4]])
        if phase!=1: assert not np.any(sample[:,5:])
        tensor=torch.as_tensor(obs,device=model.device)
        _,lp=model.actor.action_log_prob(tensor)
        assert torch.isfinite(lp).all()
        if phase==0: assert torch.count_nonzero(lp)==0
    raw=np.zeros((1,124),np.float32);raw[:,30:32]=.2
    np.testing.assert_allclose(norm.normalize_obs(raw)[:,30:32],.4,atol=1e-6)
    # After release the environment returns home automatically: no phantom entropy.
    raw=np.zeros((1,124),np.float32);raw[:,73]=1;raw[:,52]=1
    obs=torch.as_tensor(norm.normalize_obs(raw),device=model.device)
    a,lp=model.actor.action_log_prob(obs)
    torch.testing.assert_close(a,model.actor(obs,deterministic=True),rtol=0,atol=0)
    assert torch.count_nonzero(lp)==0


def test_sac_updates_both_arm_and_base_and_reload(setup,tmp_path):
    model,_,norm,_=setup
    obs=phase_obs(norm,1,1)
    frozen=protected_state(model.actor)
    p0=phase_obs(norm,0)
    p0_before=model.predict(p0,deterministic=True)[0]
    before={k:v.clone() for k,v in model.actor.state_dict().items()}
    for _ in range(32):
        a=model.predict(obs,deterministic=False)[0]
        model.replay_buffer.add(obs,obs,a,np.array([.3],np.float32),np.array([0.]),[{}])
    for _ in range(2):model.train(1,8)
    assert all(torch.equal(v,model.actor.state_dict()[k]) for k,v in before.items())
    for _ in range(5):model.train(1,8)
    assert_protected(model.actor,frozen)
    assert not torch.equal(before['coupled_mu.weight'][:5],model.actor.coupled_mu.weight[:5])
    assert not torch.equal(before['coupled_mu.weight'][5:],model.actor.coupled_mu.weight[5:])
    np.testing.assert_array_equal(p0_before,model.predict(p0,deterministic=True)[0])
    model.save(tmp_path/'model.zip')
    loaded=ResidualSAC.load(tmp_path/'model.zip',device='cpu',custom_objects={'buffer_size':1})
    for phase in range(3):
        x=phase_obs(norm,phase)
        np.testing.assert_array_equal(model.predict(x,deterministic=True)[0],loaded.predict(x,deterministic=True)[0])
    assert loaded.arm_anchor_coef==2.


def test_p2_arm_gradients_without_base_action(setup):
    model,_,norm,_=setup
    obs=torch.as_tensor(phase_obs(norm,2),device=model.device)
    model.actor.zero_grad()
    a=model.actor(obs,deterministic=True)
    a[:,:5].sum().backward()
    assert torch.count_nonzero(model.actor.coupled_mu.weight.grad[:5])>0
    assert torch.count_nonzero(model.actor.coupled_mu.weight.grad[5:])==0
    assert torch.count_nonzero(a[:,5:])==0


def test_anchor_is_soft_and_prior_unchanged(setup):
    model,_,norm,_=setup
    obs=torch.as_tensor(phase_obs(norm,1),device=model.device)
    assert model.actor.arm_anchor_loss(obs).item()==0
    with torch.no_grad():model.actor.coupled_mu.bias[:5].add_(.3)
    loss=model.actor.arm_anchor_loss(obs)
    assert loss.item()>0
    model.actor.zero_grad();loss.backward()
    assert torch.count_nonzero(model.actor.coupled_mu.bias.grad[:5])>0
    assert all(p.grad is None for p in model.actor.latent_pi.parameters())


def test_selection_rejects_degraded_reference_even_if_on_beats_off():
    def rows():
        return [dict(case_id='a',map_id=1,method=m,success_place=True,
            reference_delta_rms_v_omega=[.01,.01],**{k:1. for k in METRICS}) for m in METHODS]
    r=rows();on=next(x for x in r if x['method']=='on')
    assert not assess(r)['qualified']
    on.update({k:.98 for k in METRICS});assert assess(r)['qualified']
    next(x for x in r if x['method']=='reference').update({k:.9 for k in METRICS})
    assert not assess(r)['qualified']
    with pytest.raises(ValueError):assess(r[:-1])
    r=rows();r[-1].update({k:.98 for k in METRICS});r[-1]['reference_delta_rms_v_omega']=[0.,0.]
    assert not assess(r)['qualified']


def test_real_handoff_and_short_physical_training(setup):
    model,_,norm,env=setup
    mean=norm.obs_rms.mean.copy();var=norm.obs_rms.var.copy()
    obs,info=env.reset(seed=234)
    assert info['real_handoff'] and env.env.phase_state==1
    frozen=protected_state(model.actor)
    model.learning_starts=8
    model.learn(24,reset_num_timesteps=False,log_interval=1)
    assert model.num_timesteps==24 and model._n_updates==16
    assert_protected(model.actor,frozen)
    np.testing.assert_array_equal(mean,norm.obs_rms.mean)
    np.testing.assert_array_equal(var,norm.obs_rms.var)


def test_complete_physical_comparison_initial(setup,tmp_path):
    model,_,norm,_=setup
    report=evaluate(model,norm,make_cases([1],[1.],1,401000000),tmp_path/'smoke',reference_model=model)
    assert set(report['success_rates'])==set(METHODS)
    assert all(report['success_rates'][m]==1 for m in METHODS)
    assert not report['residual_physically_active'] and not report['qualified']
    for values in report['raw_common_metrics'].values():
        assert values['reference']==values['off']==values['on']


def test_cpu_snapshots_preserve_live_actor_and_actions(setup):
    model,_,norm,_=setup
    before={k:v.clone() for k,v in model.actor.state_dict().items()}
    snapshots=evaluation_actors(model,model)
    for actor in snapshots.values():
        assert actor.device.type=='cpu'
        assert not actor.training
        assert actor.optimizer is None
        assert not any(p.requires_grad for p in actor.parameters())
        for phase in range(3):
            obs=phase_obs(norm,phase)
            np.testing.assert_array_equal(actor.predict(obs,deterministic=True)[0],
                                          model.predict(obs,deterministic=True)[0])
    assert any(p.requires_grad for p in model.actor.parameters())
    assert all(torch.equal(v,model.actor.state_dict()[k]) for k,v in before.items())
    with torch.no_grad(): snapshots['current'].coupled_mu.bias.add_(1.)
    assert all(torch.equal(v,model.actor.state_dict()[k]) for k,v in before.items())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA integration requires GPU access')
def test_cuda_training_actor_cpu_reference_pairing(setup,tmp_path):
    model,_,norm,_=setup
    model.save(tmp_path/'initial.zip')
    gpu=ResidualSAC.load(tmp_path/'initial.zip',device='cuda',custom_objects={'buffer_size':1})
    reference=ResidualSAC.load(tmp_path/'initial.zip',device='cpu',custom_objects={'buffer_size':1})
    optimizer=gpu.actor.optimizer
    before={k:v.clone() for k,v in gpu.actor.state_dict().items()}
    # Exact case that aborted the user's launch, not just the old CPU smoke case.
    report=evaluate(gpu,norm,make_cases([1],[.5],1,241000000),tmp_path/'gpu_pairing',reference_model=reference)
    assert all(report['success_rates'][m]==1 for m in METHODS)
    for values in report['raw_common_metrics'].values():
        assert values['reference']==values['off']==values['on']
    assert gpu.device.type=='cuda' and gpu.actor.device.type=='cuda'
    assert gpu.actor.optimizer is optimizer
    assert all(torch.equal(v,gpu.actor.state_dict()[k]) for k,v in before.items())
    assert any(p.requires_grad for p in gpu.actor.parameters())
