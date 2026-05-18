# 07 LoRA 与 MoE 改进路线

## 学习目标

理解为什么当前不做 Qwen3.5 全参数训练，LoRA/MoE 可能如何帮助 CALVIN 多任务泛化，以及当前仓库要实现 LoRA 还缺哪些工程件。

## 算法直觉

LoRA 的思想是在大模型线性层旁边加低秩可训练矩阵，只训练少量参数。这样可以让 backbone 轻微适配机器人任务，而不破坏原本视觉语言能力。

MoE adapter 的思想是让不同任务或语义模式走不同专家。CALVIN 场景相似、任务容易混淆，MoE 可能帮助模型把“open drawer”“move slider”“rotate block”等任务分开。

## StarVLA 源码位置

当前仓库中没有完整 LoRA/PEFT 训练链路。已确认：

- `requirements.txt` 和源码中没有 PEFT LoRA 主线实现。
- 当前可用轻量改进更接近 `QwenAdapter`。
- 训练冻结逻辑在 `starVLA/training/trainer_utils/trainer_tools.py`。
- 参数分组在 `build_param_lr_groups`。

## 输入输出张量 / 数据结构

LoRA 不改变外部输入输出。它改变的是模型内部可训练参数。

输入仍然是：

```text
image/lang/action/state(optional)
```

输出仍然是：

```text
normalized_actions: [B, T, D]
```

但 checkpoint 会多出 LoRA adapter 权重，保存和加载必须适配。

## 训练时怎么走

如果后续实现 LoRA，建议流程：

1. 在 `get_vlm_model` 创建 Qwen3.5 后注入 LoRA。
2. target modules 优先选 attention projection 和 MLP projection。
3. 冻结原始 Qwen3.5 参数。
4. 训练 LoRA + action head。
5. checkpoint 保存必须包含 LoRA 参数。
6. policy server `from_pretrained` 必须能重建同样的 LoRA 结构再加载。

不建议第一晚做 LoRA，因为这会同时影响训练、保存、加载、eval 四条链路。

## 推理时怎么走

LoRA 推理时必须在加载 checkpoint 前重建同样的 LoRA module。如果只保存了 LoRA 权重但 server 不知道如何注入 LoRA，会出现 state dict missing/unexpected keys 或实际没有加载 LoRA 的问题。

## 与 CALVIN 任务的关系

LoRA/MoE 可能帮助跨环境泛化和任务区分，但它不是最先要解决的问题。CALVIN 分数首先要求：

- 动作尺度正确。
- gripper 正确。
- policy server 正确。
- 训练足够步数。
- eval 有 mp4/results。

在这些没有稳定前，LoRA 会增加排查难度。

## 当前项目中的风险点

- 没有 PEFT 依赖和保存加载约定。
- 全参数训练 Qwen3.5 容易让已有视觉语言能力漂移。
- LoRA target modules 选错会没效果或显存过高。
- MoE adapter 还涉及路由策略，CALVIN 两天时间内难以稳定验证。

## 可以改进的位置

- 先把 LoRA 写成二阶段加分项，不作为 P0/P1。
- 如果 OFT/Adapter 已稳定，可加一个最小 LoRA smoke：
  - 只注入 text attention q/k/v/o projection。
  - rank 从 8 或 16 开始。
  - 先 100 step 检查保存加载。
- MoE 先写进技术文档作为未来工作，除非 H200 主线已提前完成。
