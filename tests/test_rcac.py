from reasonforge.curriculum import RCACConfig, RCACSampler, TaskEstimate


def _sampler(seed: int = 7) -> RCACSampler:
    return RCACSampler(
        [
            TaskEstimate("easy", "reason", p=0.95, q=0.99, u=0.2),
            TaskEstimate("middle", "reason", p=0.55, q=0.99, u=0.5),
            TaskEstimate("noisy", "tool", p=0.55, q=0.4, u=1.0),
            TaskEstimate("tool", "tool", p=0.50, q=0.95, u=0.8),
        ],
        RCACConfig(),
        seed=seed,
    )


def test_reliability_gate_is_hard() -> None:
    probabilities = _sampler().distribution()
    assert probabilities["noisy"] == 0.0
    assert probabilities["middle"] > probabilities["easy"]


def test_deterministic_selection_and_round_trip() -> None:
    sampler = _sampler()
    first = sampler.select(11)
    restored = RCACSampler.from_state_dict(sampler.state_dict())
    assert first == _sampler().select(11)
    assert restored.select(20) == sampler.select(20)


def test_balancing_reduces_repeated_family_weight() -> None:
    sampler = _sampler()
    before = sampler.score(sampler.tasks["middle"])
    sampler.family_counts["reason"] = 8
    assert sampler.score(sampler.tasks["middle"]) < before
