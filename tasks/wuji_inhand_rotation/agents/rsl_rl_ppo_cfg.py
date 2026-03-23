"""RSL-RL PPO configuration for Wuji In-Hand Rotation."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlMLPModelCfg,
)


@configclass
class WujiInHandRotationPPORunnerCfg(RslRlOnPolicyRunnerCfg):

    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 1000
    experiment_name = "wuji_inhand_rotation"
    run_name = ""
    logger = "tensorboard"
    empirical_normalization = False

    # -- observation groups: map env obs groups to algo obs sets
    # "policy" group goes to both actor and critic for now (symmetric)
    # later can add "privileged" group for asymmetric actor-critic
    obs_groups = {
        "actor": ["policy"],
        "critic": ["policy"],
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
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.016,
        max_grad_norm=1.0,
    )
