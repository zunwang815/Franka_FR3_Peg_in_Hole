"""Frozen-teacher actor with a small trainable residual action head."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules import ActorCritic

from scripts.geometric_teacher import action_from_policy_observation


class ResidualActorCritic(ActorCritic):
    """Keep the transferred actor fixed and learn only a bounded action residual.

    The old actor remains the teacher.  A zero-initialized residual head makes
    the initial policy exactly the transferred checkpoint, while PPO can learn
    a small correction for the IK-induced Jacobian mismatch.  The critic stays
    trainable so value estimates can adapt to the new reset distribution.
    """

    default_freeze_critic = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        residual_hidden_dims=(64, 64),
        residual_scale=0.15,
        **kwargs,
    ):
        super().__init__(num_actor_obs, num_critic_obs, num_actions, **kwargs)
        layers: list[nn.Module] = []
        in_features = num_actor_obs
        for hidden in residual_hidden_dims:
            layers.extend((nn.Linear(in_features, hidden), nn.Tanh()))
            in_features = hidden
        layers.append(nn.Linear(in_features, num_actions))
        self.residual = nn.Sequential(*layers)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.residual_scale = float(residual_scale)

        # Preserve the teacher actor and its exploration scale.  The critic is
        # intentionally left trainable for the changed IK reset distribution.
        for parameter in self.actor.parameters():
            parameter.requires_grad = False
        if self.default_freeze_critic:
            for parameter in self.critic.parameters():
                parameter.requires_grad = False
        if self.noise_std_type == "scalar":
            self.std.requires_grad = False
        else:
            self.log_std.requires_grad = False

        print(
            f"Residual policy: frozen actor + trainable residual {residual_hidden_dims}, "
            f"scale={self.residual_scale}, freeze_critic={self.default_freeze_critic}"
        )

    def update_distribution(self, observations):
        with torch.no_grad():
            teacher_mean = self.actor(observations)
        # Bound the complete residual, including the final linear layer.  The
        # hidden Tanh activations alone do not prevent an outlier observation
        # from producing an arbitrarily large action correction.
        mean = teacher_mean + self.residual_action(observations)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act_inference(self, observations):
        with torch.no_grad():
            teacher_mean = self.actor(observations)
        return teacher_mean + self.residual_action(observations)

    def residual_action(self, observations):
        """Return the bounded action correction for a batch of observations."""
        return self.residual_scale * torch.tanh(self.residual(observations))

    def residual_penalty(self, observations):
        """Mean squared bounded correction used by the constrained PPO update."""
        return self.residual_action(observations).pow(2).mean()

    def load_state_dict(self, state_dict, strict=True):
        # Transfer checkpoints contain teacher actor/critic/std but no
        # residual head; keep the zero initialization for missing residual
        # parameters.  Residual checkpoints load all keys normally.
        return super().load_state_dict(state_dict, strict=False)


class GeometricTeacherResidualActorCritic(ActorCritic):
    """Train only a bounded residual around the relative-observation teacher.

    Unlike :class:`ResidualActorCritic`, this variant has no transferred PPO
    actor. The frozen teacher action is computed directly from the policy's
    relative peg-to-hole and peg-axis observations, so the initial policy is
    useful even when no checkpoint exists.
    """

    default_residual_scale = 0.15
    default_teacher_position_scale = 0.005
    default_teacher_kp_position = 0.8
    default_teacher_kp_orientation = 0.8
    default_teacher_approach_depth_mm = -10.0
    default_teacher_insert_depth_mm = 30.0
    default_teacher_alignment_gate_mm = 1.0
    default_teacher_tilt_gate_deg = 2.0
    default_freeze_critic = False

    def __init__(self, num_actor_obs, num_critic_obs, num_actions, **kwargs):
        super().__init__(num_actor_obs, num_critic_obs, num_actions, **kwargs)
        layers: list[nn.Module] = []
        in_features = num_actor_obs
        for hidden in (64, 64):
            layers.extend((nn.Linear(in_features, hidden), nn.Tanh()))
            in_features = hidden
        layers.append(nn.Linear(in_features, num_actions))
        self.residual = nn.Sequential(*layers)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.residual_scale = float(self.default_residual_scale)

        for parameter in self.actor.parameters():
            parameter.requires_grad = False
        if self.default_freeze_critic:
            for parameter in self.critic.parameters():
                parameter.requires_grad = False
        if self.noise_std_type == "scalar":
            self.std.requires_grad = False
        else:
            self.log_std.requires_grad = False
        print(
            "Geometric teacher residual policy: frozen relative-observation "
            f"teacher + residual scale={self.residual_scale}, "
            f"freeze_critic={self.default_freeze_critic}",
            flush=True,
        )

    def teacher_action(self, observations):
        return action_from_policy_observation(
            observations,
            position_scale=self.default_teacher_position_scale,
            kp_position=self.default_teacher_kp_position,
            kp_orientation=self.default_teacher_kp_orientation,
            approach_depth_mm=self.default_teacher_approach_depth_mm,
            insert_depth_mm=self.default_teacher_insert_depth_mm,
            alignment_gate_mm=self.default_teacher_alignment_gate_mm,
            tilt_gate_deg=self.default_teacher_tilt_gate_deg,
        )

    def residual_action(self, observations):
        return self.residual_scale * torch.tanh(self.residual(observations))

    def update_distribution(self, observations):
        with torch.no_grad():
            teacher_mean, _ = self.teacher_action(observations)
        mean = teacher_mean + self.residual_action(observations)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act_inference(self, observations):
        with torch.no_grad():
            teacher_mean, _ = self.teacher_action(observations)
        return teacher_mean + self.residual_action(observations)

    def residual_penalty(self, observations):
        return self.residual_action(observations).pow(2).mean()
