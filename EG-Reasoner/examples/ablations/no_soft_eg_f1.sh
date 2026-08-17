#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd -- "${TRAINING_ROOT}/.." && pwd)"
cd "${TRAINING_ROOT}"

# User settings: edit these values before running the script.
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"  # Replace it with local model path
TRAIN_FILES="${PROJECT_ROOT}/data/train_set_parquet"
SEMANTIC_MODEL_DIR="${TRAINING_ROOT}/Science_Bert"
export OPENAI_API_KEY=""                  # Fill in your answer-judge API key
export OPENAI_BASE_URL=""                  # Optional OpenAI-compatible API base URL
export WANDB_API_KEY=""                    # Optional W&B API key
NUM_GPUS=4
NUM_FRAMES=32

if [[ ! -e "${TRAIN_FILES}" ]]; then
    echo "Training data not found: ${TRAIN_FILES}" >&2
    exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is required by the answer-judge reward." >&2
    exit 1
fi

export VERL_REWARD_MAX_WORKERS="32"

python3 -m verl.trainer.main \
    config=examples/eg_vqa.yaml \
    data.train_files="${TRAIN_FILES}" \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.reward.score_function_kwargs.semantic_model_dir="${SEMANTIC_MODEL_DIR}" \
    worker.reward.score_function=./examples/score_function/no_soft_eg_f1_reward.py:compute_score \
    trainer.project_name=EG-VQA-Ablations \
    trainer.experiment_name=No-Soft-EG-F1 \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    worker.rollout.limit_images="${NUM_FRAMES}"
