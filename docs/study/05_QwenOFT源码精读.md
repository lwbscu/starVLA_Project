# 05 QwenOFT 源码精读

## 学习目标

精读当前保底主线 `Qwen3.5-0.8B + QwenOFT`，理解它为什么快、稳、适合两天内出结果，以及哪些地方最值得改。

## 算法直觉

OFT 可以理解为：冻结 Qwen3.5，用它编码图像和语言；再训练一个较轻的连续动作头，把视觉语言特征直接回归成未来若干步动作。它不做复杂采样，不依赖 action token，因此工程闭环最短。

## StarVLA 源码位置

- `starVLA/model/framework/VLM4A/QwenOFT.py`
- `starVLA/model/modules/action_model/MLP_ActionHeader.py`
- `examples/calvin/train_files/starvla_train_calvin_qwen35_oft_smoke.yaml`

## 输入输出张量 / 数据结构

训练样本：

```text
image: List[PIL.Image]
lang: str
action: np.ndarray[8, 7]
state: optional np.ndarray[1, 8]
```

OFT 输出：

```text
pred_actions: Tensor[B, action_horizon, action_dim]
loss: action_loss
```

推理输出：

```text
{"normalized_actions": np.ndarray[B, action_horizon, action_dim]}
```

## 训练时怎么走

`QwenOFTDefaultConfig` 定义默认结构：

- `framework.name = "QwenOFT"`
- `framework.qwenvl.base_vlm`
- `framework.qwenvl.attn_implementation`
- `framework.action_model.action_model_type = "MLP"`
- `framework.action_model.action_dim`
- `framework.action_model.action_horizon`

`__init__` 中关键动作：

1. `merge_framework_config` 合并默认配置和 YAML。
2. `get_vlm_model(config)` 创建 Qwen3.5 wrapper。
3. 读取 `action_horizon`，设置 `chunk_len`。
4. 根据 Qwen hidden size 配置 MLP action head。

`forward` 中关键流程：

1. 从 examples 中取 `image/lang/action`。
2. 若配置了 `datasets.vla_data.obs_image_size`，对图像 resize。
3. 调 `qwen_vl_interface.build_qwenvl_inputs`。
4. 调 Qwen3.5 forward，要求 `output_hidden_states=True`。
5. 从 hidden state 中取动作条件特征。
6. `action_model.predict_action(...)` 输出动作。
7. 取 label 的最后 `action_horizon` 步：

```python
actions_target = actions[:, -self.action_horizon:, :]
```

8. 计算动作损失并返回。

## 推理时怎么走

`predict_action` 和训练基本对齐：

1. policy server 传入 `{"image": [...], "lang": "..."}`
2. 转 PIL、resize。
3. Qwen3.5 编码图像和语言。
4. MLP action head 预测 normalized actions。
5. policy server 用 `PolicyNormProcessor` 反归一化。
6. CALVIN client 每步从 action chunk 中取动作执行。

## 与 CALVIN 任务的关系

`action_horizon=8` 是当前安全默认，因为本地数据样本就是 `(8,7)`。对于 CALVIN 5-step chain，horizon 不是“5 个任务”的长度，而是每次 replan 输出的低层动作块长度。horizon 太短可能频繁重规划，太长可能动作漂移；当前先用 8 保稳。

## 当前项目中的风险点

- OFT 容易快速学到动作分布，但泛化可能弱于 Adapter/LoRA MoE。
- 训练 loss 下降不等于 CALVIN 成功率上升。
- 若 `obs_image_size` 训练和 eval 不一致，视觉特征分布会变。
- 若 `state` 训练开启但 eval example 不传 state，会造成 train/eval 不一致。

## 可以改进的位置

- `obs_image_size=160`：提升目标定位。
- `include_state=true`：补充 proprio。
- 数据增强：颜色、裁剪、轻微扰动，但不能破坏 CALVIN 物体可见性。
- Failure-aware 后训练：对第一失败任务采样更多训练片段。
