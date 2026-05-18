# 算法深入理解学习索引

## 学习目标

这组文档用于系统理解本次 CALVIN ABC->D 考核相关算法和源码链路。重点不是泛泛解释 VLA，而是把当前 StarVLA 仓库中的真实代码、训练入口、评测入口和可改进位置串起来。

本项目当前主线：

```text
CALVIN -> LeRobot dataset -> StarVLA dataloader -> Qwen3.5 backbone
       -> action head -> policy server -> CALVIN eval -> results/mp4
```

## 建议阅读顺序

1. [01_StarVLA整体架构.md](./01_StarVLA整体架构.md)
2. [02_CALVIN基准与数据流.md](./02_CALVIN基准与数据流.md)
3. [03_Qwen3_5模型接入机制.md](./03_Qwen3_5模型接入机制.md)
4. [04_ActionHead算法对比.md](./04_ActionHead算法对比.md)
5. [05_QwenOFT源码精读.md](./05_QwenOFT源码精读.md)
6. [06_QwenAdapter与AdapterVLA改进.md](./06_QwenAdapter与AdapterVLA改进.md)
7. [07_LoRA与MoE改进路线.md](./07_LoRA与MoE改进路线.md)
8. [08_训练评测闭环.md](./08_训练评测闭环.md)
9. [09_FailurePattern排查手册.md](./09_FailurePattern排查手册.md)
10. [10_源码阅读索引.md](./10_源码阅读索引.md)

## 当前结论

保底路线是 `Qwen3.5-0.8B + QwenOFT`。它已经在本地完成 smoke train、100 step train、policy server、CALVIN debug eval 和 mp4 输出。

泛化尝试路线是 `QwenAdapter`。它更接近导师提到的 AdapterVLA 思路，但当前还需要验证 Qwen3.5 兼容性，不能直接假设长训稳定。

暂缓路线是 `FAST / PI / 全参数训练`。FAST 依赖 action-token 版 VLM；PI 本地 OOM，H200 也只应先做资源 smoke；全参数训练 Qwen3.5 在两天实训和 CALVIN 数据规模下风险太高。

## 源码阅读方法

读每个算法时都按同一条线：

```text
config.yaml -> build_framework -> framework.__init__
            -> dataloader sample schema
            -> forward loss
            -> checkpoint
            -> from_pretrained
            -> predict_action
            -> policy server unnormalize
            -> CALVIN env step
```

不要只看模型结构。CALVIN 分数来自完整闭环：数据字段、动作归一化、action horizon、server metadata、gripper 映射和视频中的真实行为都会影响结果。
