"""Contratos, gradientes SAC, persistencia y lookahead físico."""
import json
import pytest
from TRAIN_R3 import (build, CurriculumEnv, DummyVecEnv, torch, np,
                      CurriculumSAC, save_resume)
from EVALUATE_R3 import assess, METRICS, snapshot
from stable_baselines3.common.logger import configure


@pytest.fixture(scope='module')
def setup_model():
    torch.set_num_threads(2)
    env=CurriculumEnv(123)
    vec=DummyVecEnv([lambda:env])
    model=build(vec,'cpu',123,512)
    obs=vec.reset()
    yield model,env,vec,obs
    vec.close()


def phase_batch(obs):
    batch=torch.as_tensor(np.repeat(obs,3,axis=0))
    batch[:,71:74]=torch.eye(3); batch[:,52]=0; batch[:,129]=1
    batch[[0,2],129]=0
    return batch


def test_initial_transfer_and_neutral_lookahead(setup_model):
    model,env,vec,obs=setup_model
    actor=model.actor; actor.curriculum_stage.fill_(0)
    batch=phase_batch(obs)
    actions=actor(batch,deterministic=True)
    torch.testing.assert_close(actions[:,:5],actor.prior_logits(batch).tanh(),rtol=0,atol=0)
    assert torch.count_nonzero(actions[:,5])==0
    assert env.env.phase_budgets==(1000,4500,1000)
    assert all(not p.requires_grad for p in actor.arm_encoder.parameters())
    assert all(not p.requires_grad for p in actor.arm_mu.parameters())


def test_stage_gates_and_finite_gradients(setup_model):
    model,_,_,obs=setup_model
    actor=model.actor; batch=phase_batch(obs)
    prior=actor.prior_logits(batch).tanh()
    for stage in range(4):
        actor.curriculum_stage.fill_(stage)
        torch.manual_seed(15)
        actions,lp=actor.action_log_prob(batch)
        assert torch.isfinite(actions).all() and torch.isfinite(lp).all()
        assert torch.max(abs(actions))<=1
        if stage<2: torch.testing.assert_close(actions[:,:5],prior,rtol=0,atol=0)
        if stage<3: torch.testing.assert_close(actions[0,:5],prior[0],rtol=0,atol=0)
        if stage>=2:
            assert not torch.equal(actions[1,1:4],prior[1,1:4])
            torch.testing.assert_close(actions[1,[0,4]],prior[1,[0,4]],rtol=0,atol=0)
            assert not torch.equal(actions[2,:5],prior[2])
        actor.zero_grad(); (actions.sum()+.001*lp.sum()).backward()
        assert all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None)
    actor.curriculum_stage.fill_(0)


def test_snapshot_save_load_and_replay(setup_model,tmp_path):
    model,env,vec,obs=setup_model
    cpu=snapshot(model)
    np.testing.assert_array_equal(cpu.predict(obs,deterministic=True)[0],model.predict(obs,deterministic=True)[0])
    a=model.predict(obs,deterministic=False)[0]
    next_obs,reward,done,infos=vec.step(a)
    model.replay_buffer.add(obs,next_obs,a,reward,done,infos)
    target=save_resume(model,env,tmp_path,dict(stage=0,stage_steps=0,completed=False))
    restored=CurriculumSAC.load(target/'model.zip',device='cpu')
    restored.load_replay_buffer(target/'replay.pkl')
    np.testing.assert_array_equal(restored.predict(obs,deterministic=True)[0],model.predict(obs,deterministic=True)[0])
    assert restored.replay_buffer.size()==model.replay_buffer.size()
    assert json.loads((tmp_path/'latest.json').read_text())['checkpoint']==str(target)


def test_physical_lookahead_and_transition(setup_model):
    model,wrapper,vec,_=setup_model
    env=wrapper.env; env.set_eval_mode(True); env.FIXED_MAP_ID=1; env.set_eval_mass(1.)
    env.CUBE_XY_NOISE_RANGE=env.CUBE_YAW_NOISE_RANGE_RAD=0.
    obs,_=env.reset(seed=310102000)
    model.actor.curriculum_stage.fill_(0)
    reached=False; changes=[]
    for _ in range(5500):
        action=model.predict(obs,deterministic=True)[0]
        if env.lookahead_available(): reached=True; action[5]=.8
        previous=obs.copy()
        obs,_,done,_,info=env.step(action)
        model.replay_buffer.add(previous[None],obs[None],action[None],np.array([0.]),
                                np.array([done]),[info])
        if info['lookahead_active']:
            changes.append(info['lookahead_applied_m']-.3)
            assert abs(info['lookahead_rate_m_s'])<=env.L_RATE_MAX+1e-7
            assert env.L_MIN<=env._lh_applied<=env.L_MAX
            assert np.max(abs(env._r2_pp_corrected-env._r2_pp_nominal))<1e-12
            if len(changes)>=30: break
        if done: break
    assert reached and len(changes)==30
    assert max(changes)>.01
    assert np.isfinite(obs).all()
    x,y=env._base_xy()
    env._reset_lookahead()
    target0=np.asarray(env._choose_lookahead(x,y,0,.3)[:2])
    env._reset_lookahead(); env._lh_request=.8
    target1=np.asarray(env._choose_lookahead(x,y,0,.3)[:2])
    assert np.linalg.norm(target1-target0)>1e-6


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA no disponible')
def test_real_sac_update_cuda_and_protected_prior(setup_model,tmp_path):
    source,_,_,_=setup_model
    source.save(tmp_path/'source.zip')
    source.save_replay_buffer(tmp_path/'replay.pkl')
    model=CurriculumSAC.load(tmp_path/'source.zip',device='cuda')
    model.load_replay_buffer(tmp_path/'replay.pkl')
    model.set_logger(configure(None,[]))
    prior={k:v.clone() for k,v in model.actor.state_dict().items() if k.startswith('arm_')}
    before=model.actor.mu.weight.detach().clone()
    model.train(gradient_steps=4,batch_size=64)
    assert model._n_updates==4
    assert not torch.equal(before,model.actor.mu.weight)
    assert all(torch.equal(v,model.actor.state_dict()[k]) for k,v in prior.items())
    assert all(torch.isfinite(p).all() for p in model.policy.parameters())


def test_training_pause_resume(tmp_path,monkeypatch):
    import os
    import signal
    import TRAIN_R3 as trainer
    original=trainer.Progress._on_step
    threshold=[4]
    def stop_soon(self):
        if self.num_timesteps>=threshold[0]: os.kill(os.getpid(),signal.SIGINT)
        return original(self)
    monkeypatch.setattr(trainer.Progress,'_on_step',stop_soon)
    folder=tmp_path/'pause_resume'
    trainer.run(trainer.parser().parse_args(['--output',str(folder),'--device','cpu',
        '--buffer-size','512','--block-steps','20','--stage-steps','20','20','20','20']))
    assert json.loads((folder/'status.json').read_text())['event']=='PAUSED_SAVED'
    first=json.loads((folder/'latest.json').read_text())['checkpoint']
    threshold[0]=8
    trainer.run(trainer.parser().parse_args(['--resume',str(folder),'--device','cpu']))
    second=json.loads((folder/'latest.json').read_text())['checkpoint']
    from pathlib import Path
    assert first!=second
    assert json.loads((Path(second)/'state.json').read_text())['steps']==8
    assert json.loads((folder/'status.json').read_text())['event']=='PAUSED_SAVED'
    restored=CurriculumSAC.load(Path(second)/'model.zip',device='cpu')
    restored.load_replay_buffer(Path(second)/'replay.pkl')
    assert restored.replay_buffer.size()==6 # Two interrupted, unrecorded control steps.


def test_gate_rejects_zero_residual_and_failures():
    rows=[]
    for mid in range(1,5):
        for method in ('initial','off','on'):
            rows.append(dict(case_id=str(mid),map_id=mid,method=method,
                success_place=True,lookahead_delta_rms_m=0.,**{m:1. for m in METRICS}))
    assert not assess(rows)['curriculum_eligible']
    for row in rows:
        if row['method']=='on': row['lookahead_delta_rms_m']=.02
    assert assess(rows)['curriculum_eligible']
    assert not assess(rows)['recommended']
    rows[-1]['success_place']=False
    assert not assess(rows)['curriculum_eligible']
