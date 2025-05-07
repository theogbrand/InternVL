#!/bin/bash
#SBATCH --job-name=prm_training
#SBATCH --nodes=1
#SBATCH --partition=a3mega
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48 
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/home/ncs/ob1/InternVL/internvl_chat/SLURM_scripts/output_logs/prm_training/slurm-dp-%j.out
#SBATCH --error=/home/ncs/ob1/InternVL/internvl_chat/SLURM_scripts/output_logs/prm_training/slurm-dp-%j.err # Redirect 

echo "Starting SLURM script setup..."

# Activate environment if needed
echo "Loading uv module..."
module load uv
echo "Changing directory..."
cd /home/ncs/ob1/InternVL/internvl_chat || { echo "cd failed"; exit 1; }
echo "Activating virtual environment..."
source .prm_train/bin/activate
export PATH
export PYTHONPATH
export VIRTUAL_ENV
echo "Setup complete. Virtual env activated."

set -x

GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-128}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

OUTPUT_DIR='/home/ncs/ob1/InternVL/internvl_chat/prm/internvl2_5_8b_prm_finetune'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 8
# batch size per gpu: 4
# gradient accumulation steps: 4
# total batch size: 128
# epoch: 1
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=${MASTER_ADDR} \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "OpenGVLab/InternVL2_5-8B" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "./shell/data/internvl_prm_meta.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 1 \
  --learning_rate 4e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 8192 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
