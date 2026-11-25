"""16DOF-URDF最终版 的模型配置（改造自 16DOF-URDF-training.xml）。"""

from pathlib import Path
import json
from dataclasses import replace
from types import MethodType

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg, BuiltinVelocityActuatorCfg
from mjlab.actuator.builtin_actuator import BuiltinPositionActuator
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

XML: Path = Path(__file__).resolve().parent.parent.parent / "assets" / "16dof" / "16dof.xml"
assert XML.exists(), f"missing model: {XML}"

LEG_ORDER = ("FL", "FR", "RL", "RR")
BACK_LEGS = ("RL", "RR")

# Software reference only: keep the original MuJoCo joint coordinates/limits.
ZERO_REFERENCE = json.loads((XML.parent / "training_zero.json").read_text(encoding="utf-8"))
POSE_VERSION = ZERO_REFERENCE["version"]
CALF_LIMITS = ZERO_REFERENCE["calf_limits_rad"]
STAND_THIGH = ZERO_REFERENCE["joint_reference_rad"]["FR_thigh_joint"]
STAND_CALF = ZERO_REFERENCE["joint_reference_rad"]["FR_calf_joint"]
STAND_BASE_HEIGHT = ZERO_REFERENCE["base_height_m"]

STAND_JOINT_POS = {"hip": 0.0, "thigh": STAND_THIGH, "calf": STAND_CALF}

#: 轮子几何（实测）
WHEEL_RADIUS = 0.0839
WHEEL_TRACK = 0.410

#: 力矩上限（取自原训练模型）
EFFORT_LEG = 30.0
EFFORT_CALF = 20.0
EFFORT_WHEEL = 5.0

TOTAL_MASS = 25.10


def _load_spec(path) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(path))
    # mjlab attaches its own terrain to every environment. The source XML also
    # contains a visual floor plane; leave it visible, but disable its collision
    # so each wheel does not receive a pair of coplanar floor/terrain contacts.
    xml_floor = next((geom for geom in spec.geoms if geom.name == "floor"), None)
    if xml_floor is not None:
        xml_floor.contype = 0
        xml_floor.conaffinity = 0
    return spec


def get_spec() -> mujoco.MjSpec:
    spec = _load_spec(XML)
    for name, limits in CALF_LIMITS.items():
        joint = spec.joint(name)
        joint.range = limits
        joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        for actuator in spec.actuators:
            if actuator.target == name:
                actuator.ctrlrange = limits
                actuator.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    return spec


def get_legacy_spec() -> mujoco.MjSpec:
    """Historical policy has calf references outside the new physical limits."""
    return _load_spec(XML.with_name("16dof_legacy_v2.xml"))


def _edit_calf_spec(self, spec, target_names):
    BuiltinPositionActuator.edit_spec(self, spec, target_names)
    for name in target_names:
        for actuator in spec.actuators:
            if actuator.target == name:
                actuator.inheritrange = 0.0
                actuator.ctrlrange = spec.joint(name).range
                actuator.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE


class CalfPositionActuatorCfg(BuiltinPositionActuatorCfg):
    def build(self, entity, target_ids, target_names):
        # Keep the exact builtin class: mjlab's fused control writer indexes by
        # type(instance). Customize only this instance's model construction.
        actuator = super().build(entity, target_ids, target_names)
        actuator.edit_spec = MethodType(_edit_calf_spec, actuator)
        return actuator


def stand_pose() -> dict[str, float]:
    """后腿 thigh / calf 与前腿反号（都向中间弯）。"""
    pose: dict[str, float] = dict(ZERO_REFERENCE["joint_reference_rad"])
    for leg in LEG_ORDER:
        pose[f"{leg}_wheel_joint"] = 0.0
    return pose


INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, STAND_BASE_HEIGHT),
    joint_pos=stand_pose(),
    joint_vel={".*": 0.0},
)

HIP_THIGH_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
    stiffness=50.0,
    damping=1.5,
    effort_limit=EFFORT_LEG,
)
CALF_ACTUATOR = CalfPositionActuatorCfg(
    target_names_expr=(".*_calf_joint",),
    stiffness=50.0,
    damping=1.5,
    effort_limit=EFFORT_CALF,
)
WHEEL_ACTUATOR = BuiltinVelocityActuatorCfg(
    target_names_expr=(".*_wheel_joint",),
    damping=1.0,
    effort_limit=EFFORT_WHEEL,
)

ARTICULATION = EntityArticulationInfoCfg(
    actuators=(HIP_THIGH_ACTUATOR, CALF_ACTUATOR, WHEEL_ACTUATOR),
    soft_joint_pos_limit_factor=0.95,
)


def get_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=INIT_STATE,
        spec_fn=get_spec,
        articulation=ARTICULATION,
    )


def get_legacy_robot_cfg() -> EntityCfg:
    calf = BuiltinPositionActuatorCfg(
        target_names_expr=(".*_calf_joint",), stiffness=50.0,
        damping=1.5, effort_limit=EFFORT_CALF,
    )
    return replace(
        get_robot_cfg(), spec_fn=get_legacy_spec,
        articulation=replace(ARTICULATION,
                             actuators=(HIP_THIGH_ACTUATOR, calf, WHEEL_ACTUATOR)),
    )
