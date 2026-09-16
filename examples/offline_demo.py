"""Illustrate token attribution without a model, API request, or training run."""
from types import SimpleNamespace

from trata_slime_muse.turn_credit_synthesis_spans import (
    TurnSpan,
    anchored_objective_group_advantages,
    build_response_token_weights,
)


def example():
    descriptions = [
        "Read evidence later used in the answer",
        "Reasonable search with no match",
        "Repeat the same read without new information",
        "Commit an unsupported conclusion",
    ]
    turns = [TurnSpan(i, 4 * i, 4 * (i + 1), text, (), ())
             for i, text in enumerate(descriptions)]
    # Four illustrative positions per turn; the fourth is a masked wrapper token.
    mask = [1, 1, 1, 0] * len(turns)
    positive = [0.8, 0.0, 0.0, 0.0]
    negative = [0.0, 0.0, 0.6, 0.9]
    success, _ = build_response_token_weights(
        response_length=len(mask), loss_mask=mask, turns=turns,
        raw_vectors=[positive], channel="success_credit")
    failure, _ = build_response_token_weights(
        response_length=len(mask), loss_mask=mask, turns=turns,
        raw_vectors=[negative], channel="failure_blame")
    rewards = [0.2, 0.6, 0.7]
    samples = [SimpleNamespace(metadata={}) for _ in rewards]
    contrast = anchored_objective_group_advantages(samples, rewards)[0]
    advantages = [abs(contrast) * (s - f) for s, f in zip(success, failure)]
    return contrast, turns, success, failure, advantages


if __name__ == "__main__":
    contrast, turns, success, failure, advantages = example()
    print("Synthetic attribution example (no judge calls)")
    print(f"Episode score: 0.20; sibling mean: 0.50; centered score: {contrast:+.2f}")
    print("A low-scoring rollout can still receive positive credit for useful work.\n")
    for turn in turns:
        start, end = turn.response_start, turn.response_end
        values = ", ".join(f"{x:+.2f}" for x in advantages[start:end])
        print(f"Turn {turn.turn_index}: {turn.assistant_content}")
        print(f"  success={success[start]:.1f}, failure={failure[start]:.1f}")
        print(f"  token advantages: [{values}]")
