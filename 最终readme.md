# 一键运行脚本


# 一键评测脚本
cd /inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project

CKPT_PATH="/inspire/qb-ilm2/project/26summer-camp-10/26220216/starVLA_Project/你的权重路径/steps_xxx_pytorch_model.pt" \
H200_CALVIN_EVAL_DATASET_PATH="/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_d_d" \
NUM_SEQUENCES=1000 \
WRITE_MP4=0 \
EVAL_GPU=0 \
EVAL_PORT=6200 \
./最终测评脚本.sh


# tensorboard日志