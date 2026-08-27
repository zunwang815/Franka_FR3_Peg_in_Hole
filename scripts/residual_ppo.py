"""PPO variant with an explicit action-residual regularizer.

The stock RSL-RL PPO objective only constrains the new policy relative to the
previous rollout policy.  For a frozen-teacher transfer this is not enough:
the residual can drift from the teacher over successive iterations.  This
small subclass adds ``coef * ||residual_action||^2`` to the minibatch loss.
"""

from __future__ import annotations

import torch
from torch import nn

from rsl_rl.algorithms import PPO


class ResidualPPO(PPO):
    """PPO update with a teacher-action residual penalty.

    The coefficient is set by the training entrypoint before the runner is
    constructed, avoiding an extra unknown field in the RSL-RL config schema.
    """

    default_residual_penalty_coef = 0.0

    def __init__(self, policy, *args, **kwargs):
        super().__init__(policy, *args, **kwargs)
        self.residual_penalty_coef = float(self.default_residual_penalty_coef)
        if self.residual_penalty_coef < 0.0:
            raise ValueError("residual penalty coefficient must be non-negative")
        if not hasattr(policy, "residual_penalty"):
            raise TypeError("ResidualPPO requires a policy with residual_penalty()")
        print(
            f"Residual PPO: teacher-action penalty coefficient={self.residual_penalty_coef:g}",
            flush=True,
        )

    def update(self):  # noqa: C901
        if self.rnd is not None or self.symmetry is not None:
            raise NotImplementedError("ResidualPPO currently supports plain PPO only")

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_residual_penalty = 0.0

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            _old_mu_batch,
            _old_sigma_batch,
            hid_states_batch,
            masks_batch,
            _rnd_state_batch,
        ) in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std() + 1.0e-8
                    )

            self.policy.act(
                obs_batch,
                masks=masks_batch,
                hidden_states=hid_states_batch[0],
            )
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(
                critic_obs_batch,
                masks=masks_batch,
                hidden_states=hid_states_batch[1],
            )
            entropy_batch = self.policy.entropy

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            residual_penalty = self.policy.residual_penalty(obs_batch)
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + self.residual_penalty_coef * residual_penalty
            )

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_residual_penalty += residual_penalty.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return {
            "value_function": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "residual_penalty": mean_residual_penalty / num_updates,
        }
