# H200 CALVIN 数据集分析

## 1. 结论

当前 H200 共享目录：

```text
/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d
```

包含两类数据：

```text
1. LeRobot / HuggingFace 格式训练数据
2. 原始 CALVIN npz 格式数据
```

训练主线应该使用完整数据：

```text
calvin_task_ABC_D
```

D 环境快速评测应该使用：

```text
task_D_D
```

但要注意：`task_D_D` 目前只有 `training/`，没有 `validation/`。它可以用于 D 环境 smoke / 可视化评估，但不能说成官方 validation 指标。

## 2. 目录含义

### 2.1 `calvin_task_ABC_D`

路径：

```text
/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d/calvin_task_ABC_D
```

格式：

```text
LeRobot / HuggingFace v2.1
```

证据：

```text
HAS meta/info.json
HAS meta/modality.json
HAS data/
HAS videos/
parquet_count=17870
mp4_count=35740
```

元信息：

```text
total_episodes: 17870
total_frames: 1071743
total_tasks: 389
total_videos: 35740
splits: {"train": "0:17870"}
fps: 10
```

特征：

```text
image: 200x200x3 video
wrist_image: 84x84x3 video
state: 8 dims
actions: 7 dims
task_index: language task index
```

用途：

```text
StarVLA 训练数据。当前 H200 训练脚本应该读取这个目录。
```

当前代码注册：

```python
"calvin_abc_d_h200": [
    ("calvin_task_ABC_D", 1.0, "libero_franka"),
]
```

对应训练变量：

```bash
export H200_CALVIN_DATA_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d
export H200_CALVIN_DATA_NAME=calvin_task_ABC_D
export H200_CALVIN_DATA_MIX=calvin_abc_d_h200
```

### 2.2 `calvin_task_ABC_D_hf`

路径：

```text
/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d/calvin_task_ABC_D_hf
```

格式：

```text
LeRobot / HuggingFace v2.1
```

证据：

```text
HAS meta/info.json
HAS meta/modality.json
HAS data/
HAS videos/
parquet_count=7870
mp4_count=35740
```

元信息显示：

```text
total_episodes: 17870
total_frames: 1071743
total_tasks: 389
splits: {"train": "0:17870"}
```

风险点：

```text
meta/info.json 声称 total_episodes=17870，但实际 parquet_count=7870。
这说明该目录可能不完整、懒加载未完全拉取，或者部分 parquet 缺失。
```

用途：

```text
不作为主训练目录。除非确认 parquet 补齐，否则优先不用。
```

### 2.3 `task_ABC_D/training`

路径：

```text
/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d/task_ABC_D/training
```

格式：

```text
原始 CALVIN npz 格式
```

证据：

```text
training_npz_count=750536
HAS training/
没有 meta/info.json
没有 parquet
没有 validation/
没有 training/.hydra/merged_config.yaml
```

含义：

```text
ABC->D 任务设置下的原始训练轨迹。
```

用途：

```text
不能被当前 StarVLA LeRobot dataloader 直接读取。
不能被当前 CALVIN eval 脚本直接作为 eval 环境读取，因为没有 validation/.hydra，也没有 training/.hydra。
```

结论：

```text
当前不直接使用它。它是原始训练源数据，不是当前自动训练脚本和自动 eval 脚本的直接输入。
```

### 2.4 `task_D_D/training`

路径：

```text
/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d/task_D_D/training
```

格式：

```text
原始 CALVIN npz 格式
```

证据：

```text
training_npz_count=145881
HAS training/
HAS training/.hydra/merged_config.yaml
HAS language annotation npy files
没有 validation/
```

含义：

```text
D 环境数据。
```

用途：

```text
可以作为 D 环境 smoke / 可视化 eval 的环境配置来源。
因为当前没有 validation/，评测脚本需要支持从 training/.hydra/merged_config.yaml 初始化环境。
```

限制：

```text
这不是官方 validation split。
用它做每轮 checkpoint 后的快速评估是合理的工程观察手段，但报告中不能把它写成正式 CALVIN validation 指标。
```

## 3. 当前推荐数据流

### 3.1 训练

训练使用全部 `calvin_task_ABC_D`，不从训练集切 eval 子集：

```text
LeRobot: calvin_task_ABC_D
episodes: 17870
frames: 1071743
```

训练脚本变量：

```bash
export H200_CALVIN_DATA_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d
export H200_CALVIN_DATA_NAME=calvin_task_ABC_D
export H200_CALVIN_DATA_MIX=calvin_abc_d_h200
```

### 3.2 每轮 checkpoint 后快速测试

使用 D 环境做少量序列可视化：

```bash
export H200_CALVIN_EVAL_DATASET_PATH=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d/task_D_D
```

脚本会从：

```text
task_D_D/training/.hydra/merged_config.yaml
```

初始化 D 环境。

默认每个阶段 eval：

```text
1k: 1 条序列
10k: 3 条序列
30k: 5 条序列
```

输出：

```text
results.json
mp4/*.mp4
```

### 3.3 正式指标

当前共享目录没有发现：

```text
task_ABC_D/validation/
task_D_D/validation/
```

因此正式 CALVIN validation 指标还缺少标准 `validation/` 目录。后续如果拿到正式 validation，应设置：

```bash
export H200_CALVIN_EVAL_DATASET_PATH=/path/to/dataset_with_validation
```

并确认：

```bash
test -f "${H200_CALVIN_EVAL_DATASET_PATH}/validation/.hydra/merged_config.yaml"
```

## 4. 当前代码需要遵守的边界

1. 训练目录必须是 LeRobot/HF 格式，至少包含：

```text
meta/info.json
meta/modality.json
data/
videos/
```

2. Eval 目录必须至少包含以下三种之一：

```text
validation/.hydra/merged_config.yaml
training/.hydra/merged_config.yaml
.hydra/merged_config.yaml
```

3. 如果 eval 使用 `training/.hydra`，只能标记为：

```text
D environment smoke eval / visual eval
```

不能标记为：

```text
official validation score
```

## 5. 对五路线快速探索的影响

当前可行流程：

```text
完整 calvin_task_ABC_D 训练数据 -> 训练 checkpoint -> task_D_D 环境初始化 eval -> 输出 results.json 和 mp4
```

这能满足当前一小时路线比较目标：

```text
1. 看 loss 是否下降
2. 看 checkpoint 是否能加载
3. 看 D 环境中动作是否完全错误
4. 看 mp4 中运动方向、夹爪、目标接近程度
```

但它不能替代最终正式报告中的完整 CALVIN validation。

## 6. 推荐立即使用的变量

```bash
export H200_CALVIN_DATA_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d
export H200_CALVIN_DATA_NAME=calvin_task_ABC_D
export H200_CALVIN_DATA_MIX=calvin_abc_d_h200
export H200_CALVIN_EVAL_DATASET_PATH=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d/task_D_D
```

## 7. 验收命令

```bash
test -f "${H200_CALVIN_DATA_ROOT}/${H200_CALVIN_DATA_NAME}/meta/info.json"
test -f "${H200_CALVIN_DATA_ROOT}/${H200_CALVIN_DATA_NAME}/meta/modality.json"
test -d "${H200_CALVIN_DATA_ROOT}/${H200_CALVIN_DATA_NAME}/data"
test -d "${H200_CALVIN_DATA_ROOT}/${H200_CALVIN_DATA_NAME}/videos"
test -f "${H200_CALVIN_EVAL_DATASET_PATH}/training/.hydra/merged_config.yaml"
```
