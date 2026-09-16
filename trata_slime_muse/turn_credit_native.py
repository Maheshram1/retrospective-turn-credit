"""Gemini per-turn objective credit without protocol-specific loss terms."""

from __future__ import annotations

from typing import Any

from trata_slime_muse.turn_credit_synthesis_spans import (
    TurnCreditError,
    _negative_advantage_scale,
)


def apply_native_turn_credit(args: Any, rollout_data: dict[str, Any]) -> None:
    """Apply only the success/failure maps scaled by sibling-group contrast."""
    del args
    import torch
    from megatron.core import mpu
    from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

    if not mpu.is_pipeline_last_stage():
        return

    rewards = rollout_data["rewards"]
    kl = rollout_data["kl"]
    loss_masks = rollout_data["loss_masks"]
    success_weights = rollout_data.get("turn_credit_weights")
    failure_weights = rollout_data.get("turn_failure_weights")
    response_lengths = rollout_data["response_lengths"]
    total_lengths = rollout_data["total_lengths"]
    if success_weights is None or failure_weights is None:
        raise TurnCreditError("actor batch is missing Gemini objective maps")
    if not len(success_weights) == len(failure_weights) == len(rewards):
        raise TurnCreditError("native turn-credit batch length mismatch")

    negative_scale = _negative_advantage_scale()
    cp_size = mpu.get_context_parallel_world_size()
    advantages = []
    selected_channel_empty: list[bool] = []

    for (
        reward,
        local_kl,
        raw_mask,
        raw_success,
        raw_failure,
        response_len,
        total_len,
    ) in zip(
        rewards,
        kl,
        loss_masks,
        success_weights,
        failure_weights,
        response_lengths,
        total_lengths,
        strict=True,
    ):
        success = raw_success[:response_len].to(
            device=local_kl.device, dtype=torch.float32
        )
        failure = raw_failure[:response_len].to(
            device=local_kl.device, dtype=torch.float32
        )
        mask = raw_mask[:response_len].to(
            device=local_kl.device, dtype=torch.bool
        )
        if success.numel() != response_len or failure.numel() != response_len:
            raise TurnCreditError("native turn-credit tensor length mismatch")
        for name, weights in (("success", success), ("failure", failure)):
            if (
                not torch.isfinite(weights).all()
                or (weights < 0).any()
                or (weights > 1).any()
            ):
                raise TurnCreditError(f"actor {name} weights contain invalid values")
            if not mask.any() and weights.abs().sum() != 0:
                raise TurnCreditError(
                    f"fully masked trajectory has nonzero {name} weights"
                )

        centered_reward = torch.as_tensor(
            reward, device=success.device, dtype=success.dtype
        )
        if centered_reward.numel() != 1 or not torch.isfinite(centered_reward).all():
            raise TurnCreditError("centered GRPO reward must be one finite scalar")
        selected_channel_empty.append(
            bool(
                centered_reward.item() != 0.0
                and mask.any()
                and success[mask].sum().item() == 0.0
                and failure[mask].sum().item() == 0.0
            )
        )
        full = centered_reward.abs() * (
            success - negative_scale * failure
        )
        full = torch.where(mask, full, torch.zeros_like(full))
        advantages.append(
            slice_log_prob_with_cp(full, total_len, response_len)
            if cp_size > 1
            else full
        )

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = [advantage.clone() for advantage in advantages]
    rollout_data["turn_credit_selected_channel_empty"] = selected_channel_empty
    rollout_data["protocol_error_additive_advantage_sums"] = [
        0.0
    ] * len(advantages)
    rollout_data["protocol_error_effective_advantage_contributions"] = [
        0.0
    ] * len(advantages)

