"""16DOF-URDF最终版 的模型配置（改造自 16DOF-URDF-training.xml）。"""

from pathlib import Path
import json

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg, BuiltinVelocityActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

XML: Path = Path(__file__).resolve().parent.parent.parent / "assets" / "16dof" / "16dof.xml"
assert XML.exists(), f"missing model: {XML}"

LEG_ORDER = ("FL", "FR", "RL", "RR")
BACK_LEGS = ("RL", "RR")

# Software reference only: keep the original MuJoCo joint coordinates/limits.
ZERO_REFERENCE = json.loads((XML.parent / "training_zero.json").read_text(encoding="utf-8"))
POSE_VERSION = ZERO_REFERENCE["version"]
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


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(XML))
    # mjlab attaches its own terrain to every environment. The source XML also
    # contains a visual floor plane; leave it visible, but disable its collision
    # so each wheel does not receive a pair of coplanar floor/terrain contacts.
    xml_floor = next((geom for geom in spec.geoms if geom.name == "floor"), None)
    if xml_floor is not None:
        xml_floor.contype = 0
        xml_floor.conaffinity = 0
    return spec


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
CALF_ACTUATOR = BuiltinPositionActuatorCfg(
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
