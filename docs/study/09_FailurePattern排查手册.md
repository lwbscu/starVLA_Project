# 09 Failure Pattern 排查手册

## 学习目标

把算法理解转成可执行 debug 顺序。每类失败都要能对应到可能的源码位置、检查对象和两天内是否值得修。

## 算法直觉

CALVIN 失败不能只说“模型不行”。同样是失败，可能来自完全不同层级：

- 模型没学到。
- 动作归一化错。
- gripper 方向反。
- camera 对齐错。
- instruction 格式错。
- horizon/replan 错。
- eval server 没加载正确 checkpoint。

Failure Pattern 的价值是缩小排查范围。

## StarVLA 源码位置

- `deployment/model_server/policy_norm_processor.py`
- `deployment/model_server/policy_wrapper.py`
- `examples/calvin/eval_files/eval_calvin.py`
- `starVLA/model/framework/VLM4A/QwenOFT.py`
- `starVLA/model/framework/VLM4A/QwenAdapter.py`
- `starVLA/dataloader/gr00t_lerobot/transform/state_action.py`

## 输入输出张量 / 数据结构

优先记录：

```text
normalized_actions: [T, 7]
unnormalized_actions: [T, 7]
first env action: [7]
server metadata
eval subtask
mp4 path
results.json
```

如果只有 success rate，没有 mp4 和动作数值，很难定位根因。

## 训练时怎么走

训练中重点看：

- loss 是否 finite。
- loss 是否长期不变。
- checkpoint 是否保存完整。
- `dataset_statistics.json` 是否包含 `franka`。
- trainable params 是否符合预期，尤其是 backbone 是否冻结。

## 推理时怎么走

eval 中重点看：

- server 是否加载正确 checkpoint。
- `action_chunk_size` 是否等于训练 horizon。
- 第一次 infer 是否返回非空动作。
- 动作尺度是否合理。
- gripper channel 是否按预期开/合。
- mp4 中机械臂是否朝目标移动。

## 与 CALVIN 任务的关系

常见 failure 与优先级：

| Failure | 高概率原因 | 优先级 | 两天内处理 |
| --- | --- | --- | --- |
| 完全不动 | action head 未学到、server 未返回有效动作、unnorm 出错 | 最高 | 必须修 |
| 动作爆炸 | normalization、q01/q99、mask、action scale | 最高 | 必须修 |
| 夹爪方向反 | gripper channel 阈值或符号 | 最高 | 必须修 |
| 碰不到目标 | camera 顺序、resize、语言任务混淆、训练不足 | 高 | 试 obs160 / Adapter |
| 抓到但放不对 | horizon 不够、state 缺失、长程状态不足 | 中 | 试 state / 更长训练 |
| 抽屉滑块失败 | 接触动力学难、动作精度不足 | 中 | 视出现频率决定 |
| 旋转姿态失败 | 姿态控制细、数据覆盖不足 | 低到中 | 后处理分析 |

## 当前项目中的风险点

- 本地 100 step eval 全部第一子任务失败，这说明 smoke 模型未收敛，不代表链路错。
- 若长训 5k 后仍完全不动，就不是“训练不够”一个解释了，需要查 normalization/server/action 输出。
- `QwenAdapter` 需要额外确认 Qwen3.5 image token id，不然可能抽错 hidden states。
- FAST/PI 的失败不能和 OFT 混为一谈，它们的失败层级不同。

## 可以改进的位置

排查命令和文件：

```bash
grep -n "server running" logs/**/terminal/policy_server.log
grep -n "action_chunk_size\\|available_unnorm_keys" logs/**/terminal/policy_server.log
grep -n "Average successful sequence length\\|Subtask:" logs/**/terminal/eval.log
find logs -path "*/mp4/*.mp4" | sort
```

建议新增 debug 输出：

- 每次 eval 第一条 normalized action min/max。
- 每次 eval 第一条 unnormalized action min/max。
- gripper channel 前 20 step 序列。
- static/wrist 第一帧保存到 `mp4/frames/`。
