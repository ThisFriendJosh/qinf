from qinf.intrinsic import CuriosityReward, CompressionGainReward


def test_curiosity_reward_non_zero():
    r = CuriosityReward(beta=0.5)
    obs1 = {"flat": [0.0, 0.0, 0.0, 0.0]}
    obs2 = {"flat": [1.0, 1.0, 1.0, 1.0]}
    reward = r.compute(obs1, obs2, 0, {})
    assert reward > 0.0


def test_compression_gain_reward_non_zero_after_revisit():
    r = CompressionGainReward(beta=1.0)
    obs = {"flat": [1, 2, 3]}
    _ = r.compute(obs, obs, 0, {})  # first visit -> zero
    reward = r.compute(obs, obs, 0, {})  # revisit -> positive gain
    assert reward > 0.0


def test_config_toggles_disable_curiosity():
    cfg = {"intrinsic": {"curiosity": {"enabled": False, "beta": 0.5}}}
    cur_cfg = cfg["intrinsic"]["curiosity"]
    r = CuriosityReward(beta=cur_cfg["beta"] if cur_cfg["enabled"] else 0.0)
    obs1 = {"flat": [0.0, 0.0, 0.0]}
    obs2 = {"flat": [1.0, 1.0, 1.0]}
    assert r.compute(obs1, obs2, 0, {}) == 0.0
