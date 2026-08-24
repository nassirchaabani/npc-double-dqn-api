from src.environment import NPCGridWorld
from src.model import encode_observation


def test_environment_observation_and_step():
    env = NPCGridWorld(size=5)
    observation, info = env.reset(seed=1)
    assert observation.shape == (6,)
    assert info == {}
    next_observation, reward, terminated, truncated, info = env.step(0)
    assert next_observation.shape == (6,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "success" in info


def test_relative_observation_encoding():
    encoded = encode_observation([0, 0, 4, 4, 2, 1])
    assert encoded.shape == (6,)
    assert encoded.tolist() == [1.0, 1.0, 0.5, 0.25, 0.0, 0.0]
