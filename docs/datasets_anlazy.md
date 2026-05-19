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

D-D 环境评测优先使用当前已检索到的独立目录：

```text
/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_d_d
```

证据：该目录有 `validation/.hydra/merged_config.yaml`。如果只是做快速链路 smoke，也可以使用项目内或公共 `calvin_debug_dataset`，因为它同时有 `training/.hydra` 和 `validation/.hydra`。

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

### 2.4 `calvin_d_d/validation`

路径：

```text
/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_d_d/validation
```

格式：

```text
原始 CALVIN npz 格式
```

证据：

```text
HAS validation/.hydra/merged_config.yaml
```

含义：

```text
D 环境数据。
```

用途：

```text
D-D 环境 eval 的优先环境配置来源。
```

注意：

```text
该目录当前只确认到 validation/.hydra。若后续需要训练 split 或做 train-set rollout eval，应使用 calvin_debug_dataset/training 或重新检索是否存在 training/.hydra。
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
export H200_CALVIN_EVAL_DATASET_PATH=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_d_d
```

脚本会从：

```text
calvin_d_d/validation/.hydra/merged_config.yaml
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

此前 `calvin_abc_d` 目录下没有发现：

```text
task_ABC_D/validation/
task_D_D/validation/
```

但当前公共目录已经发现独立 D-D validation 环境：

```text
/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_d_d/validation/.hydra/merged_config.yaml
```

后续如果要报告正式指标，应继续确认该目录的初始状态、语言标注、任务 oracle 与使用的 eval_sequences 是否匹配，再设置：

```bash
export H200_CALVIN_EVAL_DATASET_PATH=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_d_d
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
完整 calvin_task_ABC_D 训练数据 -> 训练 checkpoint -> calvin_d_d validation 环境初始化 eval -> 输出 results.json 和 mp4
```

这能满足当前一小时路线比较目标：

```text
1. 看 loss 是否下降
2. 看 checkpoint 是否能加载
3. 看 D 环境中动作是否完全错误
4. 看 mp4 中运动方向、夹爪、目标接近程度
```

如果要写入正式报告，需要先确认 `calvin_d_d/validation` 的任务 oracle、语言标注和 eval_sequences 与报告口径一致；否则只能作为 D-D 环境工程评估结果。

## 6. 推荐立即使用的变量

```bash
export H200_CALVIN_DATA_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d
export H200_CALVIN_DATA_NAME=calvin_task_ABC_D
export H200_CALVIN_DATA_MIX=calvin_abc_d_h200
export H200_CALVIN_EVAL_DATASET_PATH=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_d_d
```

## 7. 验收命令

```bash
test -f "${H200_CALVIN_DATA_ROOT}/${H200_CALVIN_DATA_NAME}/meta/info.json"
test -f "${H200_CALVIN_DATA_ROOT}/${H200_CALVIN_DATA_NAME}/meta/modality.json"
test -d "${H200_CALVIN_DATA_ROOT}/${H200_CALVIN_DATA_NAME}/data"
test -d "${H200_CALVIN_DATA_ROOT}/${H200_CALVIN_DATA_NAME}/videos"
test -f "${H200_CALVIN_EVAL_DATASET_PATH}/validation/.hydra/merged_config.yaml"
```

## 8. `/public/three/dataset` 跨数据集预训练扫描

新增只读扫描脚本：

```text
examples/calvin/scripts/inspect_dataset_layout.py
```

在 H200 上执行：

```bash
cd /inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project
export PROJECT_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project
export CONDA_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/26220216/miniconda3
export STAR_VLA_PYTHON="${CONDA_ROOT}/envs/starVLA_qwen35/bin/python"

export DATASET_SCAN_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/public/three/dataset
export DATASET_SCAN_OUT="${PROJECT_ROOT}/logs/dataset_scan_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${DATASET_SCAN_OUT}"

"${STAR_VLA_PYTHON}" examples/calvin/scripts/inspect_dataset_layout.py \
  "${DATASET_SCAN_ROOT}" \
  /inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_d_d \
  /inspire/qb-ilm2/project/26summer-camp-10/public/four/calvin/dataset/calvin_debug_dataset \
  --sample-parquet \
  --json-out "${DATASET_SCAN_OUT}/dataset_layout.json" \
  > "${DATASET_SCAN_OUT}/dataset_layout.md"

sed -n '1,220p' "${DATASET_SCAN_OUT}/dataset_layout.md"
```

这个报告会列出：

```text
1. 直接子目录
2. 扩展名计数：parquet/mp4/npz/npy/json/jsonl/hdf5/zarr 等
3. LeRobot root：meta/info.json、meta/modality.json、data/、videos/
4. CALVIN eval root：validation/.hydra、training/.hydra、direct .hydra
5. 每个 LeRobot root 的 parquet_count、mp4_count、feature_keys、modality_keys
6. 可选 sample parquet schema，用于确认 image/state/action/language 列名和维度
```

跨数据集预训练前，需要从扫描结果里确认以下信息：

```text
dataset_root        数据集父目录，传给 datasets.vla_data.data_root_dir
dataset_subdir      mixture 里的 data_name，相对 dataset_root
format              LeRobot v2/v3、原始 npz、hdf5、zarr 或其他
modality            image/wrist_image/state/action/language 的实际 key
state_dim           本体状态维度
action_dim          动作维度
fps                 帧率
episode/frame 数量  用于估算采样权重
robot_type          能否映射到已有 ROBOT_TYPE_CONFIG_MAP
embodiment_tag      能否映射到 EmbodimentTag
language 字段       task_index、language_instruction 或其他列
视频可读性          mp4 路径和 parquet 引用是否一致
```

只有确认以上字段后，才应该新增 `examples/<dataset>/train_files/data_registry/data_config.py` 或扩展 `DATASET_NAMED_MIXTURES`。不要仅凭目录名把数据加入混合训练。
