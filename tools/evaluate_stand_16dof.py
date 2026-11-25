"""Headless, matched-conditions evaluation for 16DOF standing checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import hashlib
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
for name, path in {
    "WARP_CACHE_PATH": CACHE / "warp",
    "CUDA_CACHE_PATH": CACHE / "cuda",
    "MPLCONFIGDIR": CACHE / "mpl",
    "TMP": CACHE / "tmp",
    "TEMP": CACHE / "tmp",
    "TMPDIR": CACHE / "tmp",
    "TORCHINDUCTOR_CACHE_DIR": CACHE / "torchinductor",
    "TRITON_CACHE_DIR": CACHE / "triton",
}.items():
    os.environ[name] = str(path)
    Path(path).mkdir(parents=True, exist_ok=True)
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["PYTHONPATH"] = str(ROOT / "src")

import torch
import mujoco

import robot
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import (
    load_env_cfg,
    load_rl_cfg,
    load_runner_cls,
)


OLD_STAND_POSE = {
    f"{leg}_{part}_joint": value
    for leg, sign in (("FL", 1.0), ("FR", 1.0), ("RL", -1.0), ("RR", -1.0))
    for part, value in (("hip", 0.0), ("thigh", sign * 0.4), ("calf", -sign * 1.7), ("wheel", 0.0))
}


def configure_eval(task: str, num_envs: int, stance: str):
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = num_envs
    cfg.seed = 42
    cfg.observations["actor"].enable_corruption = False

    # Standing evaluation has zero velocity command and no interval push.
    command = cfg.commands["twist"]
    command.ranges.lin_vel_x = (0.0, 0.0)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.heading_command = False
    command.ranges.heading = None
    command.rel_standing_envs = 1.0
    command.rel_forward_envs = 0.0
    command.rel_heading_envs = 0.0

    cfg.events.pop("push_robot", None)
    for name in (
        "base_com",
        "body_friction",
        "actuator_stiffness",
        "actuator_damping",
        "body_mass_base",
    ):
        cfg.events.pop(name, None)

    # Apply the same mild pose and velocity perturbation to both candidates.
    reset = cfg.events["reset_base"]
    reset.params["pose_range"] = {
        "x": (-0.03, 0.03),
        "y": (-0.03, 0.03),
        "yaw": (-0.10, 0.10),
    }
    reset.params["velocity_range"] = {
        "x": (-0.10, 0.10),
        "y": (-0.10, 0.10),
        "z": (-0.05, 0.05),
        "roll": (-0.10, 0.10),
        "pitch": (-0.10, 0.10),
        "yaw": (-0.10, 0.10),
    }

    if stance in ("old", "legacy-v2"):
        if stance == "legacy-v2":
            pose = {
                f"{leg}_{part}_joint": value
                for leg, sign in (("FR", 1), ("FL", 1), ("RL", -1), ("RR", -1))
                for part, value in (("hip", 0.0), ("thigh", sign * 0.065),
                                    ("calf", -sign * 1.005), ("wheel", 0.0))
            }
            reference_height = 0.376
        else:
            pose, reference_height = OLD_STAND_POSE, 0.284
        robot_cfg = cfg.scene.entities["robot"]
        cfg.scene.entities["robot"] = replace(
            robot_cfg,
            init_state=replace(
                robot_cfg.init_state,
                pos=(0.0, 0.0, reference_height),
                joint_pos=pose,
            ),
        )
        cfg.rewards["base_height_l2"].params["target_height"] = reference_height
    elif stance != "current":
        raise ValueError(f"Unknown stance: {stance}")

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stance", choices=("old", "legacy-v2", "current"), required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", choices=("nominal", "push"), default="nominal")
    parser.add_argument("--burn-in", type=int, default=100,
                        help="Control steps excluded from steady-state metrics")
    parser.add_argument("--export-replay", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.burn_in < args.steps:
        parser.error("--burn-in must be nonnegative and smaller than --steps")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    task = "Robot-16DOF-Stand-v0"
    env_cfg = configure_eval(task, args.num_envs, args.stance)
    env_cfg.seed = args.seed
    agent_cfg = load_rl_cfg(task)
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode=None)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device="cuda:0")
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location="cuda:0",
    )
    policy = runner.get_inference_policy(device="cuda:0")
    env.reset(seed=args.seed)

    if args.export_replay:
        model_path = checkpoint.parent / "replay_model.mjb"
        mujoco.mj_saveModel(env.sim.mj_model, str(model_path), None)
        model = env.sim.mj_model
        cfg_joint_pos = env_cfg.scene.entities["robot"].init_state.joint_pos
        initial_qpos = env.sim.data.qpos[0].detach().cpu().numpy().copy()
        initial_qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for jid in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            short = name.split("/")[-1] if name else ""
            if short in cfg_joint_pos:
                initial_qpos[model.jnt_qposadr[jid]] = cfg_joint_pos[short]
        action_terms = []
        for name, term in env.action_manager._terms.items():
            scale = term.scale
            if isinstance(scale, torch.Tensor):
                scale = scale[0].detach().cpu().tolist()
            else:
                scale = [float(scale)] * term.action_dim
            action_terms.append({
                "name": name,
                "joints": [f"robot/{n}" for n in term.target_names],
                "scale": scale,
                "offset": term.offset[0].detach().cpu().tolist()
                    if isinstance(term.offset, torch.Tensor)
                    else [float(term.offset)] * term.action_dim,
                "weights": term._weights,
                "delay_min": term.min_delay,
                "delay_max": term.max_delay,
            })
        observation_terms = env.observation_manager._group_obs_term_cfgs["actor"]
        obs_pos_cfg = observation_terms[3].params["asset_cfg"]
        obs_wheel_cfg = observation_terms[5].params["asset_cfg"]
        asset = env.scene["robot"]
        contract = {
            "model": model_path.name,
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "source_checkpoint": checkpoint.name,
            "obs_dim": 53,
            "action_dim": 16,
            "physics_dt": env.physics_dt,
            "decimation": env.cfg.decimation,
            "control_dt": env.step_dt,
            "initial_qpos": initial_qpos.tolist(),
            "leg_obs_joints": [f"robot/{n}" for n in asset.find_joints(obs_pos_cfg.joint_names)[1]],
            "wheel_obs_joints": [f"robot/{n}" for n in asset.find_joints(obs_wheel_cfg.joint_names)[1]],
            "default_joint_pos": cfg_joint_pos,
            "action_terms": action_terms,
            "filter_kind": "two_tap_delayed_raw_action_mix",
            "zero_command": True,
        }
        contract_path = checkpoint.with_suffix(".replay.json")
        contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
        print(f"Saved matched replay contract: {contract_path}")

    upright = []
    height = []
    ang_speed = []
    leg_effort = []
    lin_speed = []
    joint_pos = []
    failures_by_term = {name: 0 for name in env.termination_manager.active_terms}
    failed_envs = torch.zeros(args.num_envs, dtype=torch.bool, device="cuda:0")
    falls = torch.zeros((), dtype=torch.long, device="cuda:0")
    timeouts = torch.zeros((), dtype=torch.long, device="cuda:0")
    obs = wrapped.get_observations()
    robot_asset = env.scene["robot"]
    robot_data = robot_asset.data
    leg_ids = robot_asset.find_joints(
        (".*_hip_joint", ".*_thigh_joint", ".*_calf_joint")
    )[0]

    with torch.inference_mode():
        for step in range(args.steps):
            if args.profile == "push" and step in (250, 500, 750):
                # Reproducible velocity kicks, independent of policy RNG use.
                # 0.3 m/s horizontal delta, alternating direction across envs.
                phase = torch.arange(args.num_envs, device="cuda:0") % 4
                velocity = robot_data.root_link_vel_w.clone()
                velocity[:, 0] += (phase == 0) * 0.30 - (phase == 1) * 0.30
                velocity[:, 1] += (phase == 2) * 0.30 - (phase == 3) * 0.30
                robot_asset.write_root_link_velocity_to_sim(velocity)
            actions = policy(obs)
            if not torch.isfinite(actions).all():
                raise RuntimeError("Nonfinite policy output during evaluation")
            obs, _, done, extras = wrapped.step(actions)
            timeout = extras.get("time_outs", torch.zeros_like(done, dtype=torch.bool))
            falls += torch.count_nonzero(done.bool() & ~timeout.bool())
            timeouts += torch.count_nonzero(timeout)
            failed_envs |= done.bool() & ~timeout.bool()
            for name in failures_by_term:
                failures_by_term[name] += int(torch.count_nonzero(
                    env.termination_manager.get_term(name)).cpu())
            if not torch.isfinite(env.sim.data.qpos).all():
                raise RuntimeError("Nonfinite simulation state during evaluation")
            upright.append(torch.clamp(-robot_data.projected_gravity_b[:, 2], 0.0, 1.0))
            height.append(robot_data.root_link_pos_w[:, 2].clone())
            ang_speed.append(torch.linalg.vector_norm(robot_data.root_link_ang_vel_b, dim=-1))
            leg_effort.append(
                torch.mean(torch.abs(robot_data.qfrc_actuator[:, leg_ids]), dim=-1)
            )
            lin_speed.append(torch.linalg.vector_norm(robot_data.root_link_lin_vel_b, dim=-1))
            joint_pos.append(robot_data.joint_pos[:, leg_ids].clone())

    def stats(values: list[torch.Tensor]) -> dict[str, float]:
        data = torch.cat([value.reshape(-1) for value in values])
        return {
            "mean": float(data.mean().cpu()),
            "std": float(data.std(unbiased=False).cpu()),
            "p95": float(torch.quantile(data, 0.95).cpu()),
            "max": float(data.max().cpu()),
        }

    result = {
        "task": task,
        "checkpoint": str(checkpoint),
        "stance": args.stance,
        "seed": args.seed,
        "profile": args.profile,
        "burn_in_steps": args.burn_in,
        "push_delta_m_s": 0.30 if args.profile == "push" else 0.0,
        "push_times_s": [5, 10, 15] if args.profile == "push" else [],
        "observation_noise": False,
        "domain_randomization": False,
        "delay_range_control_steps": [0, 2],
        "auto_reset_on_failure": True,
        "failed_envs": int(failed_envs.sum().cpu()),
        "termination_counts": failures_by_term,
        "num_envs": args.num_envs,
        "steps_per_env": args.steps,
        "evaluated_control_steps": args.num_envs * args.steps,
        "target_base_height_m": float(env_cfg.rewards["base_height_l2"].params["target_height"]),
        "initial_base_height_m": float(env_cfg.scene.entities["robot"].init_state.pos[2]),
        "falls": int(falls.cpu()),
        "timeouts": int(timeouts.cpu()),
        "upright": stats(upright),
        "base_height_m": stats(height),
        "base_ang_speed_rad_s": stats(ang_speed),
        "mean_abs_leg_torque_nm": stats(leg_effort),
        "base_lin_speed_m_s": stats(lin_speed),
        "steady_state": {
            "upright": stats(upright[args.burn_in:]),
            "base_height_m": stats(height[args.burn_in:]),
            "base_ang_speed_rad_s": stats(ang_speed[args.burn_in:]),
            "mean_abs_leg_torque_nm": stats(leg_effort[args.burn_in:]),
            "base_lin_speed_m_s": stats(lin_speed[args.burn_in:]),
            "leg_joint_names": robot_asset.find_joints(
                (".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"))[1],
            "mean_leg_joint_pos_rad": torch.stack(joint_pos[args.burn_in:]).mean(
                dim=(0, 1)).cpu().tolist(),
        },
    }
    output = checkpoint.with_name(
        f"{checkpoint.stem}_eval_{args.stance}_{args.profile}_seed{args.seed}.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved evaluation: {output}")
    wrapped.close()


if __name__ == "__main__":
    main()
