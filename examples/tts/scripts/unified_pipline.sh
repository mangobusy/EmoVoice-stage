#!/usr/bin/env bash
set -euo pipefail

# Example runner for emotion-token training + inference.
# Adjust paths and hyperparameters as needed.

PROJECT_ROOT="/root/autodl-tmp/EmoVoice"
CONFIG_FILE="/root/autodl-tmp/EmoVoice/examples/tts/tts_config.py"

TRAIN_DATA_PATH="/root/autodl-tmp/data/Data_preprocess/StoryTTS/StoryTTS_data.jsonl"
VAL_DATA_PATH="/root/autodl-tmp/data/Data_preprocess/StoryTTS/StoryTTS_val.jsonl"

LLM_PATH="/root/autodl-tmp/EmoVoice/checkpoint/Qwen2.5-0.5B"

TOTAL_AUDIO_VOCABSIZE=4160
LLM_VOCABSIZE=152000
TOTAL_VOCABSIZE=$((TOTAL_AUDIO_VOCABSIZE + LLM_VOCABSIZE))
CODE_LAYER=3
NUM_LATENCY_TOKENS=0

EMOTION_BINS=200
EMOTION_LOSS_WEIGHT=5.0
AUDIO_LOSS_WEIGHT=1.0

HYDRA_FLAGS=(
  "++model_config.llm_path=${LLM_PATH}"
  "++model_config.vocab_config.code_layer=${CODE_LAYER}"
  "++model_config.vocab_config.total_audio_vocabsize=${TOTAL_AUDIO_VOCABSIZE}"
  "++model_config.vocab_config.total_vocabsize=${TOTAL_VOCABSIZE}"
  "++model_config.vocab_config.emotion_bins=${EMOTION_BINS}"
  "++dataset_config.train_data_path=${TRAIN_DATA_PATH}"
  "++dataset_config.val_data_path=${VAL_DATA_PATH}"
  "++dataset_config.num_latency_tokens=${NUM_LATENCY_TOKENS}"
  "++dataset_config.use_emotion_tokens=true"
  "++train_config.use_emotion_token_loss=true"
  "++train_config.emotion_token_loss_weight=${EMOTION_LOSS_WEIGHT}"
  "++train_config.audio_token_loss_weight=${AUDIO_LOSS_WEIGHT}"
)

python "${PROJECT_ROOT}/src/slam_llm/pipeline/finetune.py" \
  ++model_config.file="examples/tts/model/slam_model_tts.py:model_factory" \
  ++dataset_config.file="examples/tts/speech_dataset_tts.py:get_speech_dataset" \
  "${HYDRA_FLAGS[@]}"

echo "Training finished. Run batch inference:"

echo "python ${PROJECT_ROOT}/examples/tts/generate_tts_batch.py \\
  ++model_config.file=examples/tts/model/slam_model_tts.py:model_factory \\
  ++dataset_config.file=examples/tts/speech_dataset_tts.py:get_speech_dataset \\
  ${HYDRA_FLAGS[*]} \\
  ++decode_config.max_new_tokens=3000"