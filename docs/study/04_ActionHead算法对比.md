# 04 Action Head 算法对比

## 学习目标

理解 OFT、PI/Flow-Matching、GR00T、FAST 在 StarVLA 中分别如何把 Qwen hidden states 转换成机器人动作，并明确两天实训中各自的优先级。

## 算法直觉

Action head 是 VLA 中从“看懂任务”到“产生动作”的桥。Qwen3.5 负责视觉语言编码，action head 负责控制输出。不同 head 的差别在于动作建模方式：

- OFT：直接连续回归，最快最稳。
- PI/Flow-Matching：生成式建模动作轨迹，表达力强但重。
- GR00T：DiT 风格复杂动作专家，更强但更吃资源。
- FAST：把连续动作离散成 action token，自回归生成。

## StarVLA 源码位置

- `starVLA/model/framework/VLM4A/QwenOFT.py`
- `starVLA/model/framework/VLM4A/QwenPI.py`
- `starVLA/model/framework/VLM4A/QwenGR00T.py`
- `starVLA/model/framework/VLM4A/QwenFast.py`
- `starVLA/model/modules/action_model/MLP_ActionHeader.py`
- `starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py`
- `starVLA/model/modules/action_model/GR00T_ActionHeader.py`
- `starVLA/model/modules/action_model/fast_ActionHeader.py`

## 输入输出张量 / 数据结构

统一目标：

```text
输入: Qwen hidden states + optional state
输出: normalized_actions, shape = [B, action_horizon, action_dim]
```

CALVIN 当前安全默认：

```text
action_horizon = 8
action_dim = 7
```

训练标签：

```text
actions_target = actions[:, -action_horizon:, :]
```

## 训练时怎么走

OFT：

```text
Qwen hidden -> action query/last hidden -> MLP action head -> L1/MSE style loss
```

PI/Flow-Matching：

```text
Qwen hidden -> DiT/flow matching head -> sampled noise/time target -> flow loss
```

GR00T：

```text
Qwen hidden -> GR00T action expert -> diffusion/velocity prediction loss
```

FAST：

```text
continuous action -> FAST tokenizer -> action token string
image/lang + action token labels -> Qwen LM loss
```

## 推理时怎么走

OFT/GR00T/PI 输出连续 normalized actions，再由 policy server 反归一化。

FAST 推理时需要 Qwen 自回归生成 action special tokens，再 decode 回连续动作。当前 Qwen3.5 普通权重没有这些 tokens，所以不适合作为第一晚主线。

## 与 CALVIN 任务的关系

CALVIN 既需要语义理解，也需要稳定低层控制。两天内最重要的是先产生可执行动作和可评测 checkpoint。因此：

- OFT 适合保底。
- Adapter 适合尝试多任务泛化。
- GR00T/PI 可做 H200 探针。
- FAST 工程链路太长，暂缓。

## 当前项目中的风险点

- 本地实测 `QwenPI` 在 8GB 上 backward OOM，H200 也要先 smoke。
- `QwenGR00T` 本地 1 step 可跑，但模型时间明显更高。
- `QwenFast` 需要 action-token 版 VLM，当前 Qwen3.5 普通权重不能直接用。
- `action_horizon=16` 当前会和 `(8,7)` 标签冲突，除非同步改数据窗口。

## 可以改进的位置

- OFT：先做 obs size、state、训练步数消融。
- Adapter：验证 action query 是否改善任务混淆。
- PI/GR00T：只在 H200 先跑 100 step，确认显存和 eval 兼容。
- FAST：后续单独做 Qwen3.5 action token 扩展，不抢主线。
