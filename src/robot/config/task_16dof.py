"""16DOF tasks share the versioned 45/90-degree startup reference."""

import robot.naming as naming
from dataclasses import replace
from robot import robot_16dof

# 覆盖模块级几何常量
naming.WHEEL_RADIUS = robot_16dof.WHEEL_RADIUS
naming.WHEEL_TRACK = robot_16dof.WHEEL_TRACK
naming.STAND_BASE_HEIGHT = robot_16dof.STAND_BASE_HEIGHT
naming.STAND_JOINT_POS = robot_16dof.STAND_JOINT_POS

from robot.config import env_cfgs as _env_cfgs  # noqa: E402
from robot.config.rl_cfg import (  # noqa: E402
    flat_ppo_runner_cfg, stand_ppo_runner_cfg,
    rough_ppo_runner_cfg, crawl_ppo_runner_cfg,
)
from mjlab.tasks.registry import register_mjlab_task  # noqa: E402

_env_cfgs.get_robot_cfg = robot_16dof.get_robot_cfg


def apply_training_zero(cfg):
    """Use this when adding future 16DOF gait/obstacle tasks as well."""
    cfg.scene.entities["robot"] = robot_16dof.get_robot_cfg()
    # Start leg angles at the reference. Terrain/root velocity randomization
    # remains task-specific; later pose-noise curricula must be explicit.
    reset_joints = cfg.events.get("reset_joints")
    if reset_joints is not None:
        reset_joints.params["position_range"] = (0.0, 0.0)
        reset_joints.params["velocity_range"] = (0.0, 0.0)
    return cfg


def register() -> None:
    stand_agent = stand_ppo_runner_cfg()
    stand_agent.experiment_name = "robot_stand_16dof_zero4590_v1"
    stand_agent.max_iterations = 600
    register_mjlab_task(
        task_id="Robot-16DOF-Stand-v0",
        env_cfg=apply_training_zero(_env_cfgs.stand_env_cfg(play=False)),
        play_env_cfg=apply_training_zero(_env_cfgs.stand_env_cfg(play=True)),
        rl_cfg=stand_agent,
    )
    flat_agent = flat_ppo_runner_cfg()
    flat_agent.experiment_name = "robot_flat_16dof_zero4590_v1"
    register_mjlab_task(
        task_id="Robot-16DOF-Flat-v0",
        env_cfg=apply_training_zero(_env_cfgs.flat_env_cfg(play=False)),
        play_env_cfg=apply_training_zero(_env_cfgs.flat_env_cfg(play=True)),
        rl_cfg=flat_agent,
    )
    for label, env_builder, agent_builder in (
        ("Rough", _env_cfgs.rough_env_cfg, rough_ppo_runner_cfg),
        ("Crawl", _env_cfgs.crawl_env_cfg, crawl_ppo_runner_cfg),
    ):
        agent = agent_builder()
        agent.experiment_name = f"robot_{label.lower()}_16dof_zero4590_v1"
        register_mjlab_task(
            task_id=f"Robot-16DOF-{label}-v0",
            env_cfg=apply_training_zero(env_builder(play=False)),
            play_env_cfg=apply_training_zero(env_builder(play=True)),
            rl_cfg=agent,
        )

    # Preserve the input/output reference of the already trained model_500.
    def legacy_cfg(play):
        cfg = _env_cfgs.stand_env_cfg(play=play)
        asset = cfg.scene.entities["robot"]
        pose = {}
        for leg in ("FR", "FL", "RL", "RR"):
            sign = 1 if leg in ("FR", "FL") else -1
            pose.update({f"{leg}_hip_joint": 0.0,
                         f"{leg}_thigh_joint": sign * 0.065,
                         f"{leg}_calf_joint": -sign * 1.005,
                         f"{leg}_wheel_joint": 0.0})
        cfg.scene.entities["robot"] = replace(
            asset, init_state=replace(asset.init_state, pos=(0, 0, 0.376), joint_pos=pose))
        cfg.rewards["base_height_l2"].params["target_height"] = 0.376
        return cfg

    legacy_agent = stand_ppo_runner_cfg()
    legacy_agent.experiment_name = "robot_stand_16dof_v2"
    register_mjlab_task(
        task_id="Robot-16DOF-Stand-LegacyV2-v0",
        env_cfg=legacy_cfg(False), play_env_cfg=legacy_cfg(True), rl_cfg=legacy_agent)


if __name__ == "__main__":
    register()
    print("registered Robot-16DOF-Stand-v0, Robot-16DOF-Flat-v0")
