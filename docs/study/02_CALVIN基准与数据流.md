# 02 CALVIN 基准与数据流

## 学习目标

理解 CALVIN ABC->D 的训练/评测目标，StarVLA 如何读取 LeRobot 格式训练数据，以及为什么训练数据和评测环境是两套东西。

## 算法直觉

CALVIN 长程操控不是单个动作预测问题，而是连续完成 5 个自然语言任务。ABC->D 的含义是：在 A/B/C 环境训练，在未见过的 D 环境评测。难点包括：

- 视觉背景变化。
- 相似场景中任务语义混淆。
- 多步任务之间状态传递。
- 抽屉、滑块、开关、旋转等交互动作细节。

## StarVLA 源码位置

- `examples/calvin/train_files/data_registry/data_config.py`
- `examples/calvin/train_files/modality.json`
- `starVLA/dataloader/lerobot_datasets.py`
- `starVLA/dataloader/gr00t_lerobot/datasets.py`
- `starVLA/dataloader/gr00t_lerobot/registry.py`
- `examples/calvin/eval_files/eval_calvin.py`

## 输入输出张量 / 数据结构

本项目训练样本经本地探索确认包含：

```text
sample_keys = ["action", "image", "lang", "robot_tag"]
action_shape = (8, 7)
image view count = 2
include_state=true 时 state_shape = (1, 8)
```

`action` 的 7 维通常对应：

```text
[x, y, z, roll, pitch, yaw, gripper]
```

`modality.json` 中把 action/state/video 字段映射成 StarVLA 的统一 schema。policy server eval 时还会根据 `dataset_statistics.json` 做反归一化。

## 训练时怎么走

训练读取的是 LeRobot 格式数据：

```text
playground/Datasets/calvin/calvin_abc_d_lerobot_v2.1/
```

数据注册在：

```python
DATASET_NAMED_MIXTURES = {
    "calvin_abc_d": [
        ("calvin_abc_d_lerobot_v2.1", 1.0, "libero_franka"),
    ],
}
```

训练配置中关键字段：

```yaml
datasets:
  vla_data:
    dataset_py: lerobot_datasets
    data_root_dir: playground/Datasets/calvin
    data_mix: calvin_abc_d
    action_type: delta_qpos
    require_existing_parquet: true
```

`require_existing_parquet` 是当前本地数据不完整时的重要保护，避免 dataloader 采到不存在的 parquet。

## 推理时怎么走

评测不是读取 LeRobot 数据，而是启动原始 CALVIN 环境：

```text
/home/lwb/Projects/SII/starVLA_Projects/calvin/dataset/calvin_debug_dataset/validation
```

`eval_calvin.py` 做的事情：

1. 创建 CALVIN env。
2. 加载 eval sequence。
3. 根据 initial condition reset 环境。
4. 每个 subtask 用自然语言指令调用 policy client。
5. policy client 通过 websocket 向 StarVLA server 请求动作。
6. 环境执行动作。
7. task oracle 判断 subtask 是否成功。
8. 输出平均任务链长度和 Task1~5 成功率。

## 与 CALVIN 任务的关系

训练数据提供行为克隆样本，eval 环境提供闭环控制和任务判定。两者必须对齐：

- 图像视角要一致或可映射。
- 语言指令格式要接近。
- action 归一化/反归一化必须一致。
- gripper 方向必须一致。
- action horizon 和 server replan 节奏必须一致。

## 当前项目中的风险点

- LeRobot 训练数据和 CALVIN eval 原始环境路径不同，不能混用。
- `data_mix` 必须注册，否则 policy normalization 无法反推 robot type。
- `action_shape=(8,7)`，直接改 `action_horizon=16` 会造成标签长度不匹配。
- CALVIN debug eval 成功跑通不代表 ABC->D 完整指标好，只说明链路通。

## 可以改进的位置

- 固定保存每次 eval 的 sequence id、subtask、success/fail、mp4。
- 给 dataloader 加一个样本可视化脚本，对齐 static/wrist image 和 language。
- 对 action 统计做 sanity check：均值、尺度、gripper 二值分布。
