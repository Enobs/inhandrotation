"""RSL-RL PPO configuration for Wuji In-Hand Rotation."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlMLPModelCfg,
)


@configclass
class WujiInHandRotationPPORunnerCfg(RslRlOnPolicyRunnerCfg):

    num_steps_per_env = 8  # Sharpa-style short horizon for fast feedback
    max_iterations = 100000
    save_interval = 1000
    experiment_name = "wuji_inhand_rotation"
    run_name = ""
    logger = "tensorboard"
    empirical_normalization = False

    # -- observation groups: asymmetric actor-critic
    # actor sees: policy obs (proprioception + object state) = 76
    # critic sees: critic state (policy obs + DR privileged info) = 83
    obs_groups = {
        "actor": ["policy"],
        "critic": ["critic"],
    }

    # -- actor network
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.3,
        ),
    )

    # -- critic network
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )

    # -- PPO algorithm
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=4.0,  # Sharpa: critic_coef=4
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0005,  # lower to prevent std growth
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,  # back to stable value; Sharpa's 5e-3 is for their custom PPO
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.016,
        max_grad_norm=1.0,
    )
