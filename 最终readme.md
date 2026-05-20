# 一键运行脚本

本文档按当前 `HEAD` 代码对齐最终训练、AWAC 后训练和 CALVIN ABC->D 评测。

## 0. 固定路径

```bash
cd /inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project

export PROJECT_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project
export PUBLIC_QWEN35_4B=/inspire/qb-ilm2/project/26summer-camp-10/public/Qwen/Qwen3.5-4B
export PUBLIC_CALVIN_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d
export H200_CALVIN_EVAL_DATASET_PATH=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_d_d
```

`PUBLIC_QWEN35_4B` 是公共 Qwen3.5-4B base model；CALVIN 训练数据是 `${PUBLIC_CALVIN_ROOT}/calvin_task_ABC_D`。

## 1. BC 主训练

目标：公共 Qwen3.5-4B + 已训好的 PI-State action head + LLM 语义 LoRA，8 卡，每卡 batch size 16，训练 20000 step。

```bash
cd /inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project
bash 最终训练脚本.sh
```

默认输出：

```text
logs/final_runs/pi_state_lora_sem_qwen35_4b_8gpu_b16_20000step/
```

关键 checkpoint：

```text
logs/final_runs/pi_state_lora_sem_qwen35_4b_8gpu_b16_20000step/checkpoints/pi_state_lora_sem_qwen35_4b_8gpu_b16_20000step/checkpoints/steps_20000_pytorch_model.pt
```

## 2. 后训练 Critic

当前代码逻辑：`train_awac_critic.py` 使用 `QwenPI`，加载并冻结 BC actor，只训练 Q critic。

```bash
cd /inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project
bash 最终后训练critic脚本.sh
```

默认输出：

```text
logs/final_awac_pi_lora_sem_onfly/awac_critic_pi_lora_sem_qwen35_4b_bc20k_onfly/
```

关键 checkpoint：

```text
logs/final_awac_pi_lora_sem_onfly/awac_critic_pi_lora_sem_qwen35_4b_bc20k_onfly/checkpoints/steps_50000_critic.pt
```

## 3. 后训练 Actor

当前代码逻辑：`train_awac_actor.py` 使用 `QwenPI_AWAC`，加载 frozen critic，并构造 frozen `actor_pi` 快照；优势为 `A = Q(s,a_pi) - Q(s,a_data)`。

```bash
cd /inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project
bash 最终后训练actor脚本.sh
```

默认输出：

```text
logs/final_awac_pi_lora_sem_onfly/awac_actor_pi_lora_sem_qwen35_4b_bc20k_onfly/
```

关键 checkpoint：

```text
logs/final_awac_pi_lora_sem_onfly/awac_actor_pi_lora_sem_qwen35_4b_bc20k_onfly/checkpoints/steps_30000_pytorch_model.pt
```

## 4. 一键评测脚本

快速 smoke：

```bash
cd /inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project

CKPT_PATH="${PROJECT_ROOT}/logs/final_awac_pi_lora_sem_onfly/awac_actor_pi_lora_sem_qwen35_4b_bc20k_onfly/checkpoints/steps_30000_pytorch_model.pt" \
H200_CALVIN_EVAL_DATASET_PATH="/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_d_d" \
NUM_SEQUENCES=5 \
WRITE_MP4=0 \
EVAL_GPU=0 \
EVAL_PORT=6200 \
./最终测评脚本.sh
```

正式评测：

```bash
cd /inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project

CKPT_PATH="${PROJECT_ROOT}/logs/final_awac_pi_lora_sem_onfly/awac_actor_pi_lora_sem_qwen35_4b_bc20k_onfly/checkpoints/steps_30000_pytorch_model.pt" \
H200_CALVIN_EVAL_DATASET_PATH="/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_d_d" \
NUM_SEQUENCES=1000 \
WRITE_MP4=0 \
EVAL_GPU=0 \
EVAL_PORT=6200 \
./最终测评脚本.sh
```

评测结果在 `results/model_eval/...` 下，关注 `results.json` 中 Task 1~5 成功率和平均链长。

## 5. TensorBoard 日志

BC/LoRA：

```bash
tensorboard --logdir logs/final_runs/pi_state_lora_sem_qwen35_4b_8gpu_b16_20000step/tensorboard --port 6006 --bind_all
```

Critic：

```bash
tensorboard --logdir logs/final_awac_pi_lora_sem_onfly/awac_critic_pi_lora_sem_qwen35_4b_bc20k_onfly/tensorboard --port 6007 --bind_all
```

Actor：

```bash
tensorboard --logdir logs/final_awac_pi_lora_sem_onfly/awac_actor_pi_lora_sem_qwen35_4b_bc20k_onfly/tensorboard --port 6008 --bind_all
```

## 6. LoRA 与后训练一致性

最终 BC checkpoint 是 `PI + LLM LoRA` 架构，所以后训练 critic 和 actor 都必须传同一套 LoRA 配置：

```text
LORA_ENABLED=true
LORA_TARGET_INCLUDE_PREFIXES=model
LORA_TARGET_EXCLUDE_PREFIXES=model.visual
LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

`train_awac_critic.py` 已与 `train_starvla.py`、`train_awac_actor.py` 对齐：`build_framework(cfg)` 后先 `apply_lora_if_enabled(...)`，再加载 `bc_checkpoint`。否则 LoRA checkpoint 会被无 LoRA 架构加载，LoRA 权重可能被 `strict=False` 丢掉。

## 7. 禁止项

- 不使用 `assume_success_if_missing=true`。
- 不写 public parquet，不在公共数据集目录生成 reward/done。
- 不把 `steps_*_critic.pt` 当 policy 做 CALVIN eval。
- 不把 LoRA checkpoint 用无 LoRA 架构加载。
- 不复用已有 run 目录覆盖日志；确需继续旧目录时显式设置 `ALLOW_EXISTING_LOG_DIR=true` 或 `ALLOW_EXISTING_RUN_DIR=true`。

## 8. 验收文件

```text
BC:     logs/final_runs/pi_state_lora_sem_qwen35_4b_8gpu_b16_20000step/checkpoints/pi_state_lora_sem_qwen35_4b_8gpu_b16_20000step/checkpoints/steps_20000_pytorch_model.pt
Critic: logs/final_awac_pi_lora_sem_onfly/awac_critic_pi_lora_sem_qwen35_4b_bc20k_onfly/checkpoints/steps_50000_critic.pt
Actor:  logs/final_awac_pi_lora_sem_onfly/awac_actor_pi_lora_sem_qwen35_4b_bc20k_onfly/checkpoints/steps_30000_pytorch_model.pt
Eval:   results/model_eval/.../results.json
```
