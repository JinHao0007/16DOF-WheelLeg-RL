"""Open the 16DOF XML in MuJoCo's standard interactive Simulate GUI."""

from pathlib import Path
import argparse
import json
import math
import time

import mujoco
import numpy as np
from mujoco import viewer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-pose", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    model = mujoco.MjModel.from_xml_path(str(root / "assets" / "16dof" / "16dof.xml"))
    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home)
    if args.calibration_pose:
        angles = {}
        for leg in ("FR", "FL", "RL", "RR"):
            sign = 1 if leg in ("FR", "FL") else -1
            for part, angle in (("hip", 0.0), ("thigh", sign * math.pi / 4),
                                ("calf", -sign * math.pi / 2), ("wheel", 0.0)):
                name = f"{leg}_{part}_joint"
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if model.jnt_limited[jid] and not (
                    model.jnt_range[jid, 0] <= angle <= model.jnt_range[jid, 1]
                ):
                    raise ValueError(f"Joint limit exceeded: {name}")
                data.qpos[model.jnt_qposadr[jid]] = angle
                angles[name] = math.degrees(angle)
                for aid in np.flatnonzero(model.actuator_trnid[:, 0] == jid):
                    data.ctrl[aid] = angle if part != "wheel" else 0.0
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        wheel_bottoms = []
        for leg in ("FR", "FL", "RL", "RR"):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                  f"{leg}_wheel_link_collision")
            axis_z = abs(data.geom_xmat[gid].reshape(3, 3)[2, 2])
            radius, half_width = model.geom_size[gid, :2]
            extent_z = radius * math.sqrt(max(0, 1 - axis_z**2)) + half_width * axis_z
            wheel_bottoms.append(float(data.geom_xpos[gid, 2] - extent_z))
        data.qpos[2] += 0.0002 - min(wheel_bottoms)
        # Make the native keyframe reset return to this inspection pose.
        model.key_qpos[home] = data.qpos
        model.key_ctrl[home] = data.ctrl
        report = {"pose": "calibration_hip0_thigh45_calf90",
                  "base_height_m": float(data.qpos[2]),
                  "joint_degrees": angles, "policy_loaded": False}
        output = root / ".cache" / "calibration_pose_45_90.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    mujoco.mj_forward(model, data)
    if args.check:
        return
    model.stat.center[:] = data.xpos[1]
    model.stat.extent = 0.9
    print("Opening interactive MuJoCo Simulate: 16DOF XML, no policy.", flush=True)
    print("Space: pause/run. Right panel: Joint / Control sliders.", flush=True)
    if args.calibration_pose:
        running = False
        reset_requested = False

        def on_key(keycode):
            nonlocal running, reset_requested
            if keycode == 32:
                running = not running
            elif keycode in (ord("R"), ord("r")):
                reset_requested = True

        with viewer.launch_passive(model, data, key_callback=on_key,
                                   show_left_ui=True, show_right_ui=True) as handle:
            # The native GUI initializes some state on load; restore the exact
            # keyframe after it has opened, before any physics step is taken.
            with handle.lock():
                mujoco.mj_resetDataKeyframe(model, data, home)
                mujoco.mj_forward(model, data)
                handle.cam.lookat[:] = data.xpos[1]
                handle.cam.distance = 1.8
                handle.cam.azimuth = 125
                handle.cam.elevation = -20
            handle.sync()
            while handle.is_running():
                started = time.perf_counter()
                with handle.lock():
                    if reset_requested:
                        mujoco.mj_resetDataKeyframe(model, data, home)
                        running = False
                        reset_requested = False
                    if running:
                        mujoco.mj_step(model, data)
                    else:
                        mujoco.mj_forward(model, data)
                handle.set_texts((None, None, "45/90 degree pose",
                                 "SPACE: run/pause | R: reset\n" +
                                 ("PHYSICS RUNNING" if running else "POSE PAUSED")))
                handle.sync()
                time.sleep(max(0, model.opt.timestep - (time.perf_counter() - started)))
        return
    viewer.launch(model, data, show_left_ui=True, show_right_ui=True)


if __name__ == "__main__":
    main()
