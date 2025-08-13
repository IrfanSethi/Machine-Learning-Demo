import math

from ml_platformer import config as C
from ml_platformer.main import compute_reward
from ml_platformer.player import InputState


def test_time_penalty_scales_with_dt():
    # Setup: no progress, no events
    prev_dist = 100.0
    new_dist = 100.0
    prev_x = 0.0
    new_x = 0.0
    base = compute_reward(prev_dist, new_dist, prev_x, new_x,
                          reached_exit=False, fell=False, dt=1.0,
                          idle_weight=0.0, episode_time=1.0,
                          reached_timeout=False, furthest_x_reward=0.0,
                          cur_input=InputState())
    half = compute_reward(prev_dist, new_dist, prev_x, new_x,
                          reached_exit=False, fell=False, dt=0.5,
                          idle_weight=0.0, episode_time=1.0,
                          reached_timeout=False, furthest_x_reward=0.0,
                          cur_input=InputState())
    # Penalty is linear in dt, other terms zero
    assert math.isclose(base, 2 * half, rel_tol=1e-6)


def test_forward_vs_backward_dx():
    # When equidistant before/after, only dx terms differ
    prev_dist = 50.0
    new_dist = 50.0
    dt = 0.0

    # Move right by +10px
    r_forward = compute_reward(prev_dist, new_dist, 0.0, 10.0,
                               False, False, dt, 0.0, 1.0, False, 0.0,
                               InputState())
    # Move left by -10px
    r_backward = compute_reward(prev_dist, new_dist, 0.0, -10.0,
                                False, False, dt, 0.0, 1.0, False, 0.0,
                                InputState())
    assert r_forward > 0
    assert r_backward < 0
    assert C.REWARD_PROGRESS_X_SCALE > C.LEFT_MOVE_PENALTY_PER_PX
    assert r_forward > abs(r_backward)


def test_terminal_rewards():
    # Exit should grant base + time bonus
    t = 20.0
    r = compute_reward(100.0, 10.0, 0.0, 500.0, True, False, 0.0, 0.0, t, False, 0.0, InputState())
    assert r >= C.REWARD_REACH_EXIT
    # Timeout applies a negative penalty when reached_timeout=True
    r_to = compute_reward(0.0, 0.0, 0.0, 0.0, False, False, 0.0, 0.0, 1.0, True, 0.0, InputState())
    assert math.isclose(r_to, C.TIMEOUT_PENALTY, rel_tol=1e-6)
