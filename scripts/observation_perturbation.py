"""Shared fixed-per-episode hole-position observation error.

The teacher and PPO use the same interpretation of a 0.5 mm hole-position
uncertainty: a calibration/estimation bias sampled at reset and held fixed
for that episode.  This models an imperfect pose estimate without introducing
controller delay or frame-to-frame sensor jitter.
"""

from __future__ import annotations


def wrap_with_hole_xy_bias(
    env,
    *,
    gym,
    torch,
    bias_std_mm: float = 0.0,
    vector_start: int = 18,
):
    """Add a fixed Gaussian XY bias to the flattened peg-to-hole observation."""

    bias_std_m = float(bias_std_mm) / 1000.0
    if bias_std_m < 0.0:
        raise ValueError("bias_std_mm must be non-negative")
    if bias_std_m == 0.0:
        return env

    class HoleXYBiasWrapper(gym.Wrapper):
        def __init__(self, wrapped_env):
            super().__init__(wrapped_env)
            self._bias = None
            # Bias used by the policy observation that produced the most
            # recently applied action. This remains valid even when done
            # environments immediately resample the next episode's bias.
            self._last_policy_bias = None

        def _sample_bias(self, env_ids=None, device=None, dtype=None):
            if env_ids is None:
                count = self.unwrapped.num_envs
                env_ids = None
            else:
                count = int(env_ids.numel())
            if device is None:
                device = self.unwrapped.device
            if dtype is None:
                dtype = torch.float32
            sampled = torch.randn((count, 2), device=device, dtype=dtype) * bias_std_m
            if env_ids is None:
                self._bias = sampled
            else:
                self._bias[env_ids] = sampled

        def _apply(self, obs):
            if isinstance(obs, dict) and "policy" in obs:
                policy = obs["policy"].clone()
                policy[:, vector_start : vector_start + 2] += self._bias.to(policy.dtype)
                updated = dict(obs)
                updated["policy"] = policy
                return updated
            policy = obs.clone()
            policy[:, vector_start : vector_start + 2] += self._bias.to(policy.dtype)
            return policy

        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            policy = obs["policy"] if isinstance(obs, dict) else obs
            self._sample_bias(device=policy.device, dtype=policy.dtype)
            self._last_policy_bias = self._bias.detach().clone()
            return self._apply(obs), info

        def step(self, action):
            if self._bias is not None:
                self._last_policy_bias = self._bias.detach().clone()
            obs, reward, terminated, truncated, info = self.env.step(action)
            if self._bias is None:
                policy = obs["policy"] if isinstance(obs, dict) else obs
                self._sample_bias(device=policy.device, dtype=policy.dtype)
            done = terminated.bool() | truncated.bool()
            if done.any():
                self._sample_bias(
                    torch.nonzero(done).flatten(),
                    device=self._bias.device,
                    dtype=self._bias.dtype,
                )
            return self._apply(obs), reward, terminated, truncated, info

    return HoleXYBiasWrapper(env)
