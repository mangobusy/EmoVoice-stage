#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/data/Shizihui/MyModel/src
export CUDA_VISIBLE_DEVICES=0,1,2
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

code_dir=/data/Shizihui/MyModel/examples/tts
num_gpus_per_node=$(( $(echo ${CUDA_VISIBLE_DEVICES} | tr -cd ',' | wc -c) + 1 ))
num_nodes=1
num_gpus=$(( num_gpus_per_node * num_nodes ))

llm_path=/data/Shizihui/MyModel/ckp/Qwen2.5-0.5B
llm_name=Qwen2.5-0.5b
llm_dim=896                         # 896 1536 3584 8192  -> 0.5B 1.5B 3B 7B

# vocabulary settings
code_layer=3                        # 1 single semantic code layer   2 3 4 5 6 7 8 group semantic code layers 
total_audio_vocabsize=4160          # the vocab size of the codec token
llm_vocabsize=152000                # the vocab size of the LLM model (Qwen2 here)
EMOTION_BINS=10
# total_vocabsize=$((total_audio_vocabsize + llm_vocabsize))
total_vocabsize=$((total_audio_vocabsize + llm_vocabsize + EMOTION_BINS * 2))

EMOTION_LOSS_WEIGHT=5.0
AUDIO_LOSS_WEIGHT=1.0
TEXT_LOSS_WEIGHT=1.0

# code settings
num_latency_tokens=0                # number of latency tokens (in front of the generated audio tokens)
do_layershift=false                 # if false, tokens in each layers use the same codebook, otherwise, use different codebooks

# dataset settings
train_data_path="/data/Shizihui/Data_preprocess/Total/EN/4_dataset/4_dataset-train.jsonl"
val_data_path="/data/Shizihui/Data_preprocess/Total/EN/4_dataset/4_dataset-val.jsonl"
# train_data_path="/root/autodl-tmp/data/story_audio_w_emotion_tra/MsceneSpeech/MsceneSpeech_train_emotion.jsonl"
# val_data_path="/root/autodl-tmp/data/VoiceAssistant-400K-v2/val_0.jsonl"


# training settings
batch_size_training=4
use_fp16=true
use_peft=false
num_epochs=10
lr=1e-4
warmup_steps=1000
total_steps=100000

# validation settings
# -----------------------------------------------------------------------------------
# validation_interval=10000
validation_interval=500
# -----------------------------------------------------------------------------------

split_size=0.01
# model settings
group_decode=true
group_decode_adapter_type=linear

# log settings
exp_name="UT-EN-12"

wandb_entity_name=u03zs21-sun-yat-sen-university
wandb_project_name=SLAM-Omni

home_dir=/data/Shizihui/MyModel
output_dir=$home_dir/$exp_name

if [ "$exp_name" = "debug" ]; then
    use_wandb=false
else
    use_wandb=true
fi
wandb_exp_name=$exp_name

hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=$llm_name \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=$llm_dim \
++model_config.vocab_config.code_layer=$code_layer \
++model_config.vocab_config.total_audio_vocabsize=$total_audio_vocabsize \
++model_config.vocab_config.total_vocabsize=$total_vocabsize \
++model_config.group_decode=$group_decode \
++model_config.vocab_config.emotion_bins=$EMOTION_BINS \
++model_config.group_decode_adapter_type=$group_decode_adapter_type \
++dataset_config.dataset=speech_dataset_tts \
++dataset_config.train_data_path=$train_data_path \
++dataset_config.val_data_path=$val_data_path \
++dataset_config.seed=42 \
++dataset_config.split_size=$split_size \
++dataset_config.vocab_config.code_layer=$code_layer \
++dataset_config.vocab_config.total_audio_vocabsize=$total_audio_vocabsize \
++dataset_config.vocab_config.total_vocabsize=$total_vocabsize \
++dataset_config.num_latency_tokens=$num_latency_tokens \
++dataset_config.do_layershift=$do_layershift \
++dataset_config.use_text_stream=true \
++dataset_config.use_emotion_tokens=true \
++train_config.use_emotion_token_loss=true \
++train_config.emotion_token_loss_weight=$EMOTION_LOSS_WEIGHT \
++train_config.audio_token_loss_weight=$AUDIO_LOSS_WEIGHT \
++train_config.text_token_loss_weight=$TEXT_LOSS_WEIGHT \
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
++train_config.num_workers_dataloader=0 \
++train_config.output_dir=$output_dir \
++train_config.use_fp16=$use_fp16 \
++train_config.use_peft=$use_peft \
++train_config.shuffle_train=true \
++metric=acc \
++log_config.use_wandb=$use_wandb \
++log_config.wandb_entity_name=$wandb_entity_name \
++log_config.wandb_project_name=$wandb_project_name \
++log_config.wandb_exp_name=$wandb_exp_name \
++log_config.wandb_dir=$output_dir \
++log_config.log_file=$output_dir/exp.log \
++log_config.log_interval=100 \
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
