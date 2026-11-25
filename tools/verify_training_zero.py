"""Verify that the new reference gives zero observations and correct targets."""

import argparse
import json
import os
from pathlib import Path
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
args = parser.parse_args()
ROOT = args.root.resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "mjlab" / "src")]
for variable, subdir in {
    "WARP_CACHE_PATH": "warp", "CUDA_CACHE_PATH": "cuda", "MPLCONFIGDIR": "mpl",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor", "TRITON_CACHE_DIR": "triton",
    "TMP": "tmp", "TEMP": "tmp", "TMPDIR": "tmp",
}.items():
    path = ROOT / ".cache" / subdir
    path.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(path)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import mujoco
import numpy as np
import torch
import robot
from robot import robot_16dof
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg


def main():
    reference = robot_16dof.ZERO_REFERENCE
    pose = reference["joint_reference_rad"]
    tasks = [f"Robot-16DOF-{name}-v0" for name in ("Stand", "Flat", "Rough", "Crawl")]
    for task in tasks:
        for play in (False, True):
            cfg = load_env_cfg(task, play=play)
            initial = cfg.scene.entities["robot"].init_state
            for name, value in pose.items():
                assert np.isclose(initial.joint_pos[name], value), (task, name)
            assert np.isclose(initial.pos[2], reference["base_height_m"])
            reset_joints = cfg.events.get("reset_joints")
            if reset_joints is not None:
                assert reset_joints.params["position_range"] == (0.0, 0.0)
        assert "zero4590" in load_rl_cfg(task).experiment_name
    legacy = load_env_cfg("Robot-16DOF-Stand-LegacyV2-v0")
    assert legacy.scene.entities["robot"].init_state.joint_pos["FR_thigh_joint"] == 0.065
    assert legacy.rewards["base_height_l2"].params["target_height"] == 0.376

    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "16dof" / "16dof.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for name, value in pose.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert model.jnt_range[jid, 0] <= value <= model.jnt_range[jid, 1]
        assert np.isclose(data.qpos[model.jnt_qposadr[jid]], value)
        for actuator in np.flatnonzero(model.actuator_trnid[:, 0] == jid):
            assert np.isclose(data.ctrl[actuator], value)

    cfg = load_env_cfg("Robot-16DOF-Stand-v0", play=True)
    cfg.scene.num_envs = 8
    cfg.observations["actor"].enable_corruption = False
    cfg.events = {"reset_scene": cfg.events["reset_scene"]}
    cfg.curriculum = {}
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        env.reset(seed=42)
        obs = env.get_observations()["actor"]
        position_error = float(obs[:, 9:21].abs().max().item())
        assert position_error < 1e-6, position_error
        env.action_manager.process_action(torch.zeros((8, 16), device="cuda:0"))
        term = env.action_manager._terms["leg_joint_pos"]
        expected = torch.tensor([pose[name] for name in term.target_names], device="cuda:0")
        target_error = float((term._processed_actions - expected).abs().max().item())
        assert target_error < 1e-6, target_error
        result = {
            "pose_version": reference["version"],
            "base_height_m": reference["base_height_m"],
            "tasks_checked": tasks,
            "zero_observation_max_error_rad": position_error,
            "zero_action_target_max_error_rad": target_error,
            "xml_home_and_limits": "passed",
            "legacy_v2_reference": "preserved",
            "compiled_actuators": env.sim.mj_model.nu,
            "not_a_stability_evaluation": True,
        }
        output = ROOT / ".cache" / "zero4590_verification.json"
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
