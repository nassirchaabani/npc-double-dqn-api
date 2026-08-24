from src.compare_baselines import evaluate_policy, random_policy


def test_random_baseline_is_reproducible():
    first = evaluate_policy(random_policy(seed=7), episodes=10)
    second = evaluate_policy(random_policy(seed=7), episodes=10)
    assert first == second
    assert 0.0 <= first["success_rate"] <= 1.0