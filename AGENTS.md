# 自研 16DOF 训练约定

- 所有生成物、虚拟环境、缓存、临时目录留在本项目所在的非 C 盘目录内。
- 后续自研 16DOF 行走、步态、爬行、越障任务统一从
  `assets/16dof/training_zero.json` 的 `zero-45-90-v1` 参考启动。
- 初始姿态：前腿 hip/thigh/calf = 0°/+45°/−90°；后腿为0°/−45°/+90°。
  软件腿角为模型角减参考角。不得通过修改关节轴、几何或原始限位来伪造清零。
- 新16DOF任务调用 `apply_training_zero(cfg)`；目标高度/步态可以另设，参考零点不变。
- wheel使用零速度目标，不将轮子连续转角作为腿部标定零点。
- 旧 `robot_stand_16dof_v2/model_500` 保留原参考和回放契约；mjlab回放使用
  `Robot-16DOF-Stand-LegacyV2-v0`。新参考任务不直接加载旧坐标约定的策略。
- go2w/2027RC 是其他模型，不把自研16DOF零点强加给它们。
- 本约定是训练软件坐标，不代表硬件编码器已经标定，也不代表新参考策略已经训练完成。
- 细节见 `docs/TRAINING_ZERO_REFERENCE.md`。
