"""Pruebas de aislamiento físico y agregación de la ablación, sin entrenar."""
from types import SimpleNamespace
import numpy as np
import pytest

import EXTRACT_RESIDUAL_OFF_R11_R3 as off


class FakeR11:
    def __init__(self, residual_enabled=True, **kwargs):
        self.residual_enabled = residual_enabled
        self.phase_state = 0
        self._vx_active = False
        self._vx_delta = np.zeros(2)
        self.data = SimpleNamespace(ctrl=np.zeros(6))
        self.wheel_left_ids = [0, 1, 2]
        self.wheel_right_ids = [3, 4, 5]

    def _actuator_tau_limit(self, aid):
        return 2.

    def _apply_wheel_torque_with_residual(self, left, right, u_lin, u_ang):
        self.data.ctrl[:] = np.clip([left] * 3 + [right] * 3, -2., 2.)

    def step(self, action):
        self.last_action = action.copy()
        self._apply_wheel_torque_with_residual(3., -1., 0., 0.)
        return None, 0., False, False, {}


def test_r11_opens_physical_branch_without_rewriting_action_history():
    env = off.residual_off_r11_class(FakeR11)(residual_enabled=True)
    action = np.array([.1, .2, -.3, .4, -.5, .7, -.8], dtype=np.float32)
    original = action.copy()
    env.step(action)
    assert not env.residual_enabled
    np.testing.assert_array_equal(env.last_action, original)
    np.testing.assert_array_equal(action, original)
    np.testing.assert_array_equal(env.off_samples[0]['actor_arm_action'], original[:5])
    np.testing.assert_array_equal(env.off_samples[0]['wheel_applied_residual_nm'], np.zeros(6))
    assert env.off_torque_calls == 1


def test_r11_rejects_nonzero_physical_residual():
    env = off.residual_off_r11_class(FakeR11)()
    with pytest.raises(AssertionError, match='sumador'):
        env._apply_wheel_torque_with_residual(1., 1., .01, 0.)


def test_r3_changes_only_lookahead_action(monkeypatch):
    def parent_step(env, action):
        env.last_action = action.copy()
        return None, 0., False, False, dict(lookahead_nominal_m=.3,
                    lookahead_applied_m=.3, lookahead_filtered_m=.3)
    monkeypatch.setattr(off.common.ZennoVXR3, 'step', parent_step)
    env = off.R3ResidualOff.__new__(off.R3ResidualOff)
    off._register_env(env)
    env.phase_state = 0
    env._r2_pp_corrected = np.array([.1, .2])
    env._r2_pp_nominal = np.array([.1, .2])
    action = np.array([.1, -.2, .3, -.4, .5, .8], dtype=np.float32)
    original = action.copy()
    env.step(action)
    np.testing.assert_array_equal(env.last_action[:5], original[:5])
    assert env.last_action[5] == 0.
    np.testing.assert_array_equal(action, original)
    assert env.off_samples[0]['lookahead_delta_m'] == 0.


def sample(cid, success, value, agent='r3'):
    return dict(case_id=cid, map_id=1, agent=agent, success_place=success,
                **{key: value for key in off.METRICS})


def test_aggregation_retains_all_failures_but_compares_common_successes():
    active = {'a': sample('a', True, 2.), 'b': sample('b', True, 90.),
              'c': sample('c', False, 80.), 'd': sample('d', False, 70.)}
    rows = [sample('a', True, 4.), sample('b', False, 100.),
            sample('c', True, 60.), sample('d', False, 50.)]
    reference = dict(by_case={k: {'r3': v} for k, v in active.items()})
    group = off.aggregate(rows, reference, ['r3'])['r3']['global']
    assert (group['n_cases'], group['success_active'], group['success_off']) == (4, 2, 2)
    assert (group['both_success'], group['active_only'], group['off_only'], group['both_failed']) == (1, 1, 1, 1)
    metric = group['metrics_paired_success']['total_time_s']
    assert metric['n_pairs'] == 1 and metric['active'] == 2. and metric['off'] == 4.
    assert metric['active_vs_off_percent'] == -50.
    assert group['metrics_paired_success']['cte_signed_mean_m']['active_vs_off_percent'] is None


def test_aggregation_handles_zero_denominator_and_duplicate_rows():
    reference = dict(by_case={'a': {'r3': sample('a', True, 0.)}})
    rows = [sample('a', True, 0.)]
    group = off.aggregate(rows, reference, ['r3'])['r3']['global']
    assert group['metrics_paired_success']['total_time_s']['active_vs_off_percent'] is None
    with pytest.raises(ValueError, match='duplicados'):
        off.aggregate(rows * 2, reference, ['r3'])


def test_case_selection_never_generates_new_seeds():
    reference = dict(cases=[dict(case_id='a', map_id=1, case_seed=510100000),
                            dict(case_id='b', map_id=2, case_seed=510200000)])
    args = SimpleNamespace(maps=[2], case_ids=['b'], limit_cases=None)
    assert off.select_cases(reference, args) == [reference['cases'][1]]
    args.case_ids = ['unknown']
    with pytest.raises(ValueError, match='no presentes'):
        off.select_cases(reference, args)
