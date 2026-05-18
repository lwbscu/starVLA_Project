# 06 QwenAdapter 与 AdapterVLA 改进

## 学习目标

理解导师建议的 AdapterVLA 思路如何对应到当前 StarVLA 的 `QwenAdapter`，以及它相比 OFT 可能带来的泛化收益和工程风险。

## 算法直觉

AdapterVLA 的核心想法是：不大规模训练 LLM/VLM 主体，而是在 VLM 上插入轻量可训练结构，让它专门学习机器人动作条件。对 CALVIN 这种多任务相似场景，adapter/action query 可能比简单 OFT 更能区分任务语义。

## StarVLA 源码位置

- `starVLA/model/framework/VLM4A/QwenAdapter.py`
- `starVLA/model/modules/action_model/VLA_AdapterHeader.py`
- `starVLA/model/modules/vlm/__init__.py`

## 输入输出张量 / 数据结构

`QwenAdapter` 输入和 OFT 类似：

```text
image/lang/action/state(optional)
```

它额外引入：

```text
action_query: nn.Parameter[action_query_num, hidden_size]
dummy_action_token: "🔍"
dummy_action_prompt: repeated dummy tokens
```

Adapter head 输出：

```text
predicted_actions: Tensor[B, num_actions_chunk, action_dim]
```

## 训练时怎么走

`QwenAdapter` 的训练流程：

1. 在 instruction 后追加 prompt：

```text
Please predict the next N robot actions: <action>🔍🔍...<action>.
```

2. 用 Qwen processor tokenize 图像和文本。
3. 找到 dummy action token 在 `input_ids` 中的位置。
4. 在 embedding 层注册 forward hook。
5. hook 把 dummy token embedding 替换成可训练 `action_query`。
6. Qwen forward 输出多层 hidden states。
7. 代码抽取：
   - image token 对应 hidden states
   - action query 对应 hidden states
8. 拼接成多层视觉/action query 特征。
9. `VLA_Adapter_L1RegressionActionHead` 回归动作。
10. 与 gt action 做 L1 loss。

## 推理时怎么走

`predict_action` 和训练类似，也会插入 dummy action prompt、注册 hook、抽取 action query hidden states，再由 adapter head 输出 normalized actions。

因此，`QwenAdapter` 能否进入正式路线，必须验证：

- checkpoint 能保存/加载。
- `predict_action` 在 policy server 中可运行。
- 输出 `normalized_actions` shape 正确。
- CALVIN debug eval 能生成 mp4。

## 与 CALVIN 任务的关系

CALVIN 的任务之间场景很像，例如不同颜色方块、抽屉、滑块、开关可能同时出现。Adapter/action query 让模型有专门的可训练 token 去承载“当前任务应该控制什么”的信息，理论上有助于减少任务混淆。

## 当前项目中的风险点

- `QwenAdapter.py` 当前从 `starVLA.model.modules.vlm.QWen3` 导入 `IMAGE_TOKEN_INDEX`，而 Qwen3.5 wrapper 中 image token id 不同。这是 Qwen3.5 兼容 smoke 必须优先检查的点。
- `QwenAdapter` 会遍历多层 hidden states，显存开销可能明显高于 OFT。
- dummy token `"🔍"` 是否在 Qwen3.5 tokenizer 中稳定分成单 token，需要实测。
- `action_query_num=64` 和 `num_actions_chunk=8` 的组合需要和 CALVIN action label 对齐。

## 可以改进的位置

- 将 image token id 改为从当前 `qwen_vl_interface` 读取，而不是硬编码 Qwen3。
- 给 action dummy token 做 tokenizer 检查：必须确认 token count 等于 `action_query_num`。
- 先做 `100 step smoke + policy server + 1 seq eval`，再长训。
- 若 Adapter 稳定，再尝试 task-aware adapter 或 LoRA MoE。
