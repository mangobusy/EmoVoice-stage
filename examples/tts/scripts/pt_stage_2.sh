#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/EmoVoice/src
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

code_dir=/root/autodl-tmp/EmoVoice/examples/tts
# 假设你将第一阶段的python脚本命名为 train_stage1_emotion.py
# script_name=slam_model_tts_2.py 

num_gpus_per_node=$(( $(echo ${CUDA_VISIBLE_DEVICES} | tr -cd ',' | wc -c) + 1 ))
num_nodes=1
num_gpus=$(( num_gpus_per_node * num_nodes ))

llm_path=/root/autodl-tmp/EmoVoice/checkpoint/Qwen2.5-0.5B
llm_name=Qwen2.5-0.5b
llm_dim=896

# vocabulary settings
code_layer=3
total_audio_vocabsize=4160
llm_vocabsize=170000
total_vocabsize=$((total_audio_vocabsize + llm_vocabsize))

# dataset settings (Stage 1 只需要文本和情感标签)
train_stage=2

echo "Starting Stage 2 Training: audio token prediction"
train_data_path="/root/autodl-tmp/data/VoiceAssistant-400K-v2/train_0.jsonl"
val_data_path="/root/autodl-tmp/data/VoiceAssistant-400K-v2/val_0.jsonl"

# training settings

batch_size_training=6
use_fp16=true
use_peft=false 
num_epochs=10
lr=1e-5
warmup_steps=500
total_steps=50000
validation_interval=2

split_size=0.01
# model settings
group_decode=true
group_decode_adapter_type=linear

# log settings
exp_name="stage2_audio_token_prediction"
wandb_entity_name=u03zs21-sun-yat-sen-university
wandb_project_name=SLAM-Omni
home_dir=/root/autodl-tmp/EmoVoice
output_dir=$home_dir/$exp_name
ckpt_path=/root/autodl-tmp/EmoVoice/stage1_emotion_regression/tts_latest/model.pt

if [ "$exp_name" = "debug" ]; then
    use_wandb=false
else
    use_wandb=true
fi
wandb_exp_name=$exp_name

# Hydra Arguments specifically for Stage 1
hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=$llm_name \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=$llm_dim \
++model_config.vocab_config.code_layer=$code_layer \
++model_config.vocab_config.total_audio_vocabsize=$total_audio_vocabsize \
++model_config.vocab_config.total_vocabsize=$total_vocabsize \
++dataset_config.dataset=speech_dataset_tts \
++dataset_config.train_data_path=$train_data_path \
++dataset_config.val_data_path=$val_data_path \
++dataset_config.seed=42 \
++dataset_config.split_size=0.01 \
++dataset_config.vocab_config.code_layer=$code_layer \
++dataset_config.vocab_config.total_audio_vocabsize=$total_audio_vocabsize \
++dataset_config.vocab_config.total_vocabsize=$total_vocabsize \
++dataset_config.load_emotion_label=true \
++train_config.training_stage=$train_stage \
++train_config.use_lm_loss_stage1=false \
++train_config.model_name=tts \
++train_config.num_epochs=$num_epochs \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=false \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=$warmup_steps \
++train_config.total_steps=$total_steps \
++train_config.lr=$lr \
++train_config.validation_interval=$validation_interval \
++train_config.batch_size_training=$batch_size_training \
++train_config.val_batch_size=1 \
++train_config.num_workers_dataloader=4 \
++train_config.output_dir=$output_dir \
++train_config.use_fp16=$use_fp16 \
++train_config.use_peft=$use_peft \
++log_config.use_wandb=$use_wandb \
++log_config.wandb_entity_name=$wandb_entity_name \
++log_config.wandb_project_name=$wandb_project_name \
++log_config.wandb_exp_name=$wandb_exp_name \
++log_config.wandb_dir=$output_dir \
++log_config.log_file=$output_dir/exp.log \
++log_config.log_interval=50 \
++ckpt_path=$ckpt_path \
"

if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
    if [ "$exp_name" = "debug" ]; then
        python -m debugpy --listen 5678 --wait-for-client $code_dir/finetune_tts.py \
            $hydra_args
    else
        python $code_dir/finetune_tts.py \
            $hydra_args
    fi
else
    torchrun \
        --nnodes $num_nodes \
        --nproc_per_node $num_gpus_per_node \
        --master_port=29503 \
        $code_dir/finetune_tts.py \
        ++train_config.enable_ddp=true \
        ++train_config.enable_fsdp=false \
        $hydra_args
fi
