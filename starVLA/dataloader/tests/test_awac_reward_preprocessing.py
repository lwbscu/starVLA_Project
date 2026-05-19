# Copyright 2025 starVLA community. All rights reserved.
"""Unit tests for AWAC reward preprocessing."""

import numpy as np
import pandas as pd
import pytest

from starVLA.dataloader.awac_reward_preprocessing import (
    AWACRewardConfig,
    compute_discounted_chunk_return,
    compute_done_flags,
    compute_step_rewards,
    process_episode,
)


def test_compute_step_rewards_success():
    step = compute_step_rewards(5, episode_success=True, step_penalty=-1.0, success_reward=0.0, failure_reward=-3000.0)
    assert np.allclose(step[:-1], -1.0)
    assert step[-1] == 0.0


def test_compute_step_rewards_failure():
    step = compute_step_rewards(3, episode_success=False)
    assert step[-1] == -3000.0


def test_discounted_chunk_return_horizon_3():
    step = np.array([-1.0, -1.0, 0.0])
    chunk = compute_discounted_chunk_return(step, action_horizon=3, gamma=0.5)
    assert chunk[0] == pytest.approx(-1.0 + 0.5 * (-1.0) + 0.25 * 0.0)
    assert chunk[2] == pytest.approx(0.0)


def test_done_last_h_frames():
    done = compute_done_flags(10, action_horizon=3)
    assert done.sum() == 3
    assert done[-1] and done[-3] and not done[-4]


def test_process_episode_writes_columns():
    df = pd.DataFrame({"success": [False, False, True]})
    cfg = AWACRewardConfig(action_horizon=2, gamma=0.9)
    out = process_episode(df, cfg)
    assert "reward" in out.columns
    assert "done" in out.columns
    assert "step_reward" in out.columns
    assert out["done"].iloc[-1]
    assert out["done"].iloc[-2]
