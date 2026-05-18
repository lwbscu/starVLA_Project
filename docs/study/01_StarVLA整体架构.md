# 01 StarVLA 整体架构

## 学习目标

理解 StarVLA 如何从 YAML 配置构建模型、加载数据、冻结模块、训练 action head、保存 checkpoint，并最终被 policy server 恢复用于评测。

## 算法直觉

StarVLA 的核心设计是把不同 VLA/WM 路线统一到同一个 framework 接口。无论是 `QwenOFT`、`QwenPI`、`QwenGR00T` 还是 `QwenAdapter`，训练器都只关心：

- 这个 framework 是否能吃 `vla` batch。
- 调 `compute_loss("vla", batch)` 能否返回 loss。
- checkpoint 能否用 `baseframework.from_pretrained()` 重建。
- 推理时是否实现 `predict_action(examples)` 并返回 `normalized_actions`。

## StarVLA 源码位置

- `starVLA/model/framework/base_framework.py`
- `starVLA/model/tools.py`
- `starVLA/training/train_starvla.py`
- `starVLA/training/trainer_utils/trainer_tools.py`
- `starVLA/model/framework/share_tools.py`

## 输入输出张量 / 数据结构

训练 batch 对 framework 来说通常是 `List[dict]`，每个样本至少包含：

```text
{
  "image": [PIL.Image, PIL.Image, ...],
  "lang": str,
  "action": np.ndarray, shape = [action_horizon, action_dim],
  "robot_tag": str,
  "state": optional np.ndarray
}
```

framework 的训练输出：

```text
{"action_loss": torch.Tensor scalar}
```

policy server 期望 framework 推理输出：

```text
{"normalized_actions": np.ndarray[B, T, D]}
```

其中 `T` 必须和训练时 `framework.action_model.action_horizon` 对齐。

## 训练时怎么走

`train_starvla.py` 是主训练入口。

关键流程：

1. 解析 `--config_yaml` 和命令行 dotlist override。
2. 调 `apply_config_compat(cfg)` 规范旧字段，比如 `action_horizon` 和 `future_action_window_size`。
3. 调 `build_dataloader(cfg, dataset_py=cfg.datasets.vla_data.dataset_py)` 创建 VLA dataloader。
4. 调 `build_framework(cfg)` 构建具体模型。
5. `VLATrainer.prepare_training()` 中执行：
   - 保存 `config.full.yaml`
   - 初始化 checkpoint 目录
   - 根据 `trainer.freeze_modules` 冻结模块
   - 打印 trainable parameter
   - DeepSpeed/Accelerate prepare
6. 训练循环中对 batch 调 `self.model.compute_loss(tag, batch, loss_scale)`。
7. loss backward、optimizer step、scheduler step。
8. 到 `save_interval` 时保存：
   - `steps_xxx_pytorch_model.pt`
   - `config.yaml`
   - `dataset_statistics.json`

## 推理时怎么走

推理不走 trainer。policy server 调：

```text
baseframework.from_pretrained(ckpt_path)
```

这个函数会：

1. 从 checkpoint 路径向上找 run dir。
2. 读取 `config.yaml` 和 `dataset_statistics.json`。
3. 用 config 再次 `build_framework(cfg)`。
4. 加载 checkpoint state dict。
5. 把 normalization stats 挂到模型上。

随后 server 调 framework 的 `predict_action(examples)`，再做 unnormalize。

## 与 CALVIN 任务的关系

CALVIN 评测不是直接调用训练脚本，而是：

```text
checkpoint -> policy server -> websocket client -> CALVIN env -> task oracle
```

所以训练成功不代表评测成功。必须同时保证：

- checkpoint 能恢复。
- `action_horizon` 能被 server 读出。
- `dataset_statistics.json` 能被 normalization processor 使用。
- `predict_action` 输出 shape 正确。

## 当前项目中的风险点

- `freeze_modules` 是字符串路径，例如 `qwen_vl_interface`。路径写错只会警告，可能导致误训 backbone。
- `from_pretrained` 使用严格加载，新增模块后 checkpoint key 不匹配会直接失败。
- `action_horizon` 和旧字段 `future_action_window_size` 必须保持一致。
- 保存的 `config.yaml` 是 eval 恢复模型的重要依据，不能只保存权重。

## 可以改进的位置

- 训练前增加自动校验：确认 frozen module 真被冻结。
- checkpoint 保存时额外写 `metadata.json`，记录 action head、action_horizon、obs_image_size、data_mix。
- 对 `freeze_modules` 路径错误从 warning 升级为 failure，避免假训练。
