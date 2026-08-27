"""PPO agent configuration for Franka Peg-in-Hole task using RSL-RL."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class FrankaPegInHolePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner config for peg-in-hole task.

    Tuned for contact-rich fine manipulation with sparse success signal.
    """

    # Training loop
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 200               # Frequent save to capture peak performance
    experiment_name = "franka_peg_in_hole_phase1"
    run_name = ""
    resume = False
    empirical_normalization = False

    # Policy network: [256, 256, 128] for contact-rich tasks
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )

    # PPO algorithm — lower entropy for stable convergence
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,             # Low: environment is now correct, focus on convergence
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class FrankaPegInHoleRelPPORunnerCfg(FrankaPegInHolePPORunnerCfg):
    """PPO runner config for RELATIVE IK peg-in-hole task (6D action)."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "franka_peg_in_hole_ik_rel"
