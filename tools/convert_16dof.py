"""把 16DOF-URDF最终版/mujoco/16DOF-URDF-training.xml 转成 mjlab 可训练格式。

改造内容：
  1. 执行器顺序重排：8 组 hip/thigh -> 4 组 calf -> 4 组 wheel（mjlab 约定）
  2. 碰撞分组：只有 base 和 4 个轮子参与接触，腿的碰撞体关闭
  3. 关节阻尼补齐（armature / damping / frictionloss）
  4. 网格复制到 assets 目录，路径改为相对引用

输出：assets/16dof/16dof.xml + meshes/ + split/
"""

from __future__ import annotations

import pathlib
import shutil
import json

import mujoco

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / ".cache" / "src_16dof"
SRC_MESHES = SRC_DIR / "meshes"
SRC = SRC_DIR / "16DOF-URDF-training.xml"

DST_DIR = ROOT / "assets" / "16dof"
DST = DST_DIR / "16dof.xml"

LEGS = ("FL", "FR", "RL", "RR")
PARTS = ("hip", "thigh", "calf")

# PD：沿用 2027rc 已验证可用的一套（腿 kp=50 kd=1.5）
KP = {"hip": 50.0, "thigh": 50.0, "calf": 50.0}
KV = 1.5
WHEEL_KV = 1.0

# 力矩上限（取自原训练模型 16DOF-URDF-training.xml 的 forcerange）
EFFORT = {"hip": 30.0, "thigh": 30.0, "calf": 20.0}
EFFORT_WHEEL = 5.0

# 关节阻尼：沿用 2027rc 验证过的一套（抑制 PD 振荡）
ARMATURE = 0.01
DAMPING = 0.2
FRICTIONLOSS = 0.2

# Keep regenerated keyframes consistent with the versioned software zero.
REFERENCE = json.loads((DST_DIR / "training_zero.json").read_text(encoding="utf-8"))
STAND_THIGH = REFERENCE["joint_reference_rad"]["FR_thigh_joint"]
STAND_CALF = REFERENCE["joint_reference_rad"]["FR_calf_joint"]
STAND_HEIGHT = REFERENCE["base_height_m"]


def main() -> None:
    # 1) 复制网格
    dst_meshes = DST_DIR / "meshes"
    dst_split = DST_DIR / "split"
    dst_meshes.mkdir(parents=True, exist_ok=True)
    dst_split.mkdir(parents=True, exist_ok=True)
    for stl in SRC_MESHES.glob("*.STL"):
        shutil.copy2(stl, dst_meshes / stl.name)
    for stl in (SRC_DIR / "split").glob("*.STL"):
        shutil.copy2(stl, dst_split / stl.name)
    print(f"meshes: {len(list(dst_meshes.glob('*.STL')))} + split {len(list(dst_split.glob('*.STL')))}")

    spec = mujoco.MjSpec.from_file(str(SRC))
    # meshdir 保持源文件的 "."，网格引用分为 meshes/ 与 split/ 两个子目录

    # 2) 删除原执行器，按 mjlab 顺序重建
    for act in list(spec.actuators):
        spec.delete(act)

    def add_position(joint: str, kp: float, lo: float, hi: float, limit: float) -> None:
        a = spec.add_actuator()
        a.name = f"{joint}_servo"
        a.target = joint
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0] = kp
        a.biasprm[1] = -kp
        a.biasprm[2] = -KV
        a.ctrlrange = [lo, hi]
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        # 力矩上限必须显式设置，否则 forcerange=[0,0] 等于完全不能出力
        a.forcerange = [-limit, limit]
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE

    def add_velocity(joint: str, lim: float, limit: float) -> None:
        a = spec.add_actuator()
        a.name = f"{joint}_servo"
        a.target = joint
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0] = 1.0
        a.biasprm[2] = -WHEEL_KV
        a.ctrlrange = [-lim, lim]
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.forcerange = [-limit, limit]
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE

    # 关节限位查表（用于 ctrlrange）
    for name, bounds in REFERENCE["calf_limits_rad"].items():
        joint = spec.joint(name)
        joint.range = bounds
        joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE
    limits = {}
    for j in spec.joints:
        if j.name and j.type == mujoco.mjtJoint.mjJNT_HINGE:
            limits[j.name] = tuple(j.range) if hasattr(j, "range") else (0.0, 0.0)

    # 先 8 个 hip/thigh
    for leg in LEGS:
        for part in ("hip", "thigh"):
            lo, hi = limits.get(f"{leg}_{part}_joint", (-1.5, 1.5))
            add_position(f"{leg}_{part}_joint", KP[part], lo, hi, EFFORT[part])
    # 再 4 个 calf
    for leg in LEGS:
        lo, hi = limits[f"{leg}_calf_joint"]
        add_position(f"{leg}_calf_joint", KP["calf"], lo, hi, EFFORT["calf"])
    # 最后 4 个 wheel
    for leg in LEGS:
        add_velocity(f"{leg}_wheel_joint", 20.0, EFFORT_WHEEL)

    # 3) 碰撞分组：base + wheel 参与接触，腿关闭
    for g in spec.geoms:
        n = (g.name or "").lower()
        if "collision" not in n:
            continue
        if n.startswith("base") or "_wheel_link_collision" in n:
            g.contype = 1
            g.conaffinity = 1
            g.condim = 3
        else:
            g.contype = 0
            g.conaffinity = 0

    # 4) 关节阻尼
    for j in spec.joints:
        if j.name and j.type != mujoco.mjtJoint.mjJNT_FREE:
            j.armature = ARMATURE
            j.damping = [DAMPING] * 3
            j.frictionloss = FRICTIONLOSS

    # 5) 站姿：前后腿都向中间弯曲（方案 A），机身高度由正运动学反算得出
    for k in list(spec.keys):
        spec.delete(k)

    def leg_angles(leg: str) -> tuple[float, float]:
        if leg in ("RL", "RR"):
            return -STAND_THIGH, -STAND_CALF
        return STAND_THIGH, STAND_CALF

    # qpos 顺序 = 关节定义顺序（每腿 hip,thigh,calf,wheel 交错）
    qpos = [0.0, 0.0, STAND_HEIGHT, 1.0, 0.0, 0.0, 0.0]
    for leg in LEGS:
        thigh, calf = leg_angles(leg)
        qpos += [0.0, thigh, calf, 0.0]

    # ctrl 顺序必须跟执行器顺序一致：8 hip/thigh -> 4 calf -> 4 wheel
    ctrl = []
    for leg in LEGS:
        for part in ("hip", "thigh"):
            ctrl.append(0.0 if part == "hip" else leg_angles(leg)[0])
    for leg in LEGS:
        ctrl.append(leg_angles(leg)[1])
    for _ in LEGS:
        ctrl.append(0.0)

    key = spec.add_key()
    key.name = "home"
    key.qpos = qpos
    key.ctrl = ctrl
    print(f"qpos({len(qpos)}) = {qpos}")
    print(f"ctrl({len(ctrl)}) = {ctrl}")

    model = spec.compile()
    print(f"compiled: nq={model.nq} nv={model.nv} nu={model.nu} nkey={model.nkey}")
    print(f"mass = {model.body_mass.sum():.2f} kg")

    # 6) body 名统一成 go2w 约定（现有任务代码按此匹配）
    #    FL_wheel_Link -> FL_wheel_link ; FL_hip_Link -> FL_hip ; base -> base_link
    rename = {}
    for b in spec.bodies:
        n = b.name
        if not n:
            continue
        new = n
        if n == "base":
            new = "base_link"
        elif n.endswith("_wheel_Link"):
            new = n[:-len("_wheel_Link")] + "_wheel_link"   # FL_wheel_link
        elif n.endswith("_Link"):
            new = n[:-len("_Link")]                          # FL_hip
        if new != n:
            rename[n] = new

    xml = spec.to_xml()
    # 先替换长名（wheel），再替换短名；用正则带边界避免子串误伤
    import re

    for old, new in sorted(rename.items(), key=lambda kv: -len(kv[0])):
        if old == "base":
            xml = re.sub(r'(?<=name=)"base"', '"base_link"', xml)
            xml = re.sub(r'"base"(?=[ />])', '"base_link"', xml)
            xml = xml.replace('body1="base"', 'body1="base_link"')
            xml = xml.replace('body2="base"', 'body2="base_link"')
        else:
            xml = xml.replace(f'"{old}"', f'"{new}"')
            xml = xml.replace(f'name="{old}_', f'name="{new}_')
            xml = xml.replace(f'body="{old}"', f'body="{new}"')
    XML_HEADER = '<?xml version="1.0" encoding="utf-8"?>\n'
    DST.write_text(XML_HEADER + xml, encoding="utf-8")
    print("saved:", DST)
    print("renamed:", rename)

    m2 = mujoco.MjModel.from_xml_path(str(DST))
    print("reload ok: nu =", m2.nu, " ngeom =", m2.ngeom)
    for a in range(m2.nu):
        n = mujoco.mj_id2name(m2, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        print(f"  {a:2d} {n}")


if __name__ == "__main__":
    main()
