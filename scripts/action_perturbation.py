"""Shared no-delay action perturbation for stress training and evaluation.

The perturbation is intentionally limited to zero-delay disturbances:
per-control-step normalized Gaussian action noise and a per-episode control
gain multiplier.  Keeping this wrapper shared makes teacher/PPO comparisons
use the same disturbance definition.
"""

from __future__ import annotations


def wrap_with_action_perturbation(
    env,
    *,
    gym,
    torch,
    action_noise_std: float = 0.0,
    action_gain_noise_std_pct: float = 0.0,
    reset_warmup_steps: int = 0,
):
    """Wrap a vector Isaac Lab environment with zero-delay action noise."""

    noise_std = float(action_noise_std)
    gain_std = float(action_gain_noise_std_pct) / 100.0
    if noise_std < 0.0 or gain_std < 0.0:
        raise ValueError("action perturbation scales must be non-negative")
    if reset_warmup_steps < 0:
        raise ValueError("reset_warmup_steps must be non-negative")
    if noise_std == 0.0 and gain_std == 0.0:
        return env

    class ActionPerturbationWrapper(gym.Wrapper):
        def __init__(self, wrapped_env):
            super().__init__(wrapped_env)
            self._episode_gain = None
            # Publicly inspectable provenance for visualization/evaluation
            # records. Values describe the action that was most recently sent
            # to the wrapped environment, before any completed environments
            # resample their next-episode gain.
            self._last_applied_gain = None
            self._last_action_noise = None

        def _ensure_gain(self, actions):
            if self._episode_gain is None:
                self._episode_gain = torch.ones(
                    actions.shape[0], device=actions.device, dtype=actions.dtype
                )
                self._resample_gain(torch.arange(
                    actions.shape[0], device=actions.device, dtype=torch.long
                ))

        def _resample_gain(self, env_ids):
            if self._episode_gain is None or gain_std == 0.0:
                return
            sampled = 1.0 + torch.randn(
                env_ids.numel(),
                device=self._episode_gain.device,
                dtype=self._episode_gain.dtype,
            ) * gain_std
            self._episode_gain[env_ids] = sampled.clamp_min(0.05)

        def reset(self, **kwargs):
            result = self.env.reset(**kwargs)
            self._episode_gain = None
            self._last_applied_gain = None
            self._last_action_noise = None
            if reset_warmup_steps > 0:
                obs, info = result
                zero_action = torch.zeros(
                    self.unwrapped.num_envs,
                    self.action_space.shape[-1],
                    device=self.unwrapped.device,
                    dtype=torch.float32,
                )
                for _ in range(reset_warmup_steps):
                    obs, _, _, _, _ = self.env.step(zero_action)
                return obs, info
            return result

        def step(self, actions):
            self._ensure_gain(actions)
            perturbed = actions.clone()
            if noise_std > 0.0:
                action_noise = torch.randn_like(perturbed) * noise_std
                perturbed = perturbed + action_noise
            else:
                action_noise = torch.zeros_like(perturbed)
            self._last_action_noise = action_noise.detach().clone()
            self._last_applied_gain = self._episode_gain.detach().clone()
            perturbed = (
                perturbed * self._episode_gain.unsqueeze(-1)
            ).clamp(-1.0, 1.0)
            result = self.env.step(perturbed)
            _, _, terminated, truncated, _ = result
            done = terminated.bool() | truncated.bool()
            if done.any():
                self._resample_gain(torch.nonzero(done).flatten())
            return result

    return ActionPerturbationWrapper(env)
