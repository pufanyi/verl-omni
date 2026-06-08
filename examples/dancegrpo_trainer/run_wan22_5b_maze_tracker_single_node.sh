#!/usr/bin/env bash
set -euo pipefail
set -x

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found at $PYTHON_BIN. Create/install .venv first or set PYTHON_BIN." >&2
  exit 1
fi

WAN_TRAINER_ROOT=${WAN_TRAINER_ROOT:-$(realpath ../Wan-Trainer)}
MODEL_PATH=${MODEL_PATH:-$WAN_TRAINER_ROOT/storage/models/Wan2.2-TI2V-5B-Diffusers}
WDS_DIR=${WDS_DIR:-$WAN_TRAINER_ROOT/storage/latents/maze_5b_line_to_ball_v1/webdataset/rl}
DATA_DIR=${DATA_DIR:-$REPO_ROOT/storage/data/maze_5b_line_to_ball_rl_parquet}
DATA_MAX_SAMPLES=${DATA_MAX_SAMPLES:-2048}
VAL_SIZE=${VAL_SIZE:-32}
RECREATE_DATA=${RECREATE_DATA:-0}

mkdir -p "$DATA_DIR"
if [[ "$RECREATE_DATA" == "1" || ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/test.parquet" ]]; then
  "$PYTHON_BIN" examples/dancegrpo_trainer/data_process/wan22_maze_wds_to_parquet.py \
    --webdataset-dir "$WDS_DIR" \
    --output-dir "$DATA_DIR" \
    --max-samples "$DATA_MAX_SAMPLES" \
    --val-size "$VAL_SIZE"
fi

export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-$REPO_ROOT/storage/hf_cache}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
mkdir -p "$HF_DATASETS_CACHE"
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$REPO_ROOT/storage/torchinductor_cache}
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
PYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR:-$REPO_ROOT/storage/debs/python312-dev/extracted/usr/include/python3.12}
PYTHON_INCLUDE_ROOT=${PYTHON_INCLUDE_ROOT:-$REPO_ROOT/storage/debs/python312-dev/extracted/usr/include}
if [[ -f "$PYTHON_INCLUDE_DIR/Python.h" ]]; then
  export CPATH="$PYTHON_INCLUDE_DIR:$PYTHON_INCLUDE_ROOT${CPATH:+:$CPATH}"
fi

export MAZE_TRACKER_REWARD_NUM_FRAMES=${MAZE_TRACKER_REWARD_NUM_FRAMES:-21}
export MAZE_TRACKER_REWARD_SEARCH_RADIUS=${MAZE_TRACKER_REWARD_SEARCH_RADIUS:-96}
export MAZE_TRACKER_REWARD_COLOR_SLACK=${MAZE_TRACKER_REWARD_COLOR_SLACK:-28.0}
export MAZE_TRACKER_REWARD_GOAL_TOLERANCE_CELLS=${MAZE_TRACKER_REWARD_GOAL_TOLERANCE_CELLS:-1.0}
export MAZE_TRACKER_REWARD_MAX_MEAN_ERROR_CELLS=${MAZE_TRACKER_REWARD_MAX_MEAN_ERROR_CELLS:-4.0}
export MAZE_TRACKER_REWARD_W_TRAJ=${MAZE_TRACKER_REWARD_W_TRAJ:-0.35}
export MAZE_TRACKER_REWARD_W_ONPATH=${MAZE_TRACKER_REWARD_W_ONPATH:-0.25}
export MAZE_TRACKER_REWARD_W_GOAL=${MAZE_TRACKER_REWARD_W_GOAL:-0.25}
export MAZE_TRACKER_REWARD_W_PROGRESS=${MAZE_TRACKER_REWARD_W_PROGRESS:-0.15}

NUM_GPUS=${NUM_GPUS:-8}
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_N=${ROLLOUT_N:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-1}
MAX_STEPS=${MAX_STEPS:-1}
SAVE_FREQ=${SAVE_FREQ:-100}
TEST_FREQ=${TEST_FREQ:-100}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-4}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-10}
VAL_NUM_INFERENCE_STEPS=${VAL_NUM_INFERENCE_STEPS:-20}
NUM_FRAMES=${NUM_FRAMES:-41}
SDE_WINDOW_SIZE=${SDE_WINDOW_SIZE:-6}
SDE_WINDOW_RANGE=${SDE_WINDOW_RANGE:-"[0,9]"}
NOISE_LEVEL=${NOISE_LEVEL:-0.3}
LR=${LR:-1e-5}
LORA_RANK=${LORA_RANK:-0}
LORA_ALPHA=${LORA_ALPHA:-32}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-wan22_5b_maze_tracker_single_node_${MAX_STEPS}step}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/storage/checkpoints/verl_omni/$EXPERIMENT_NAME}
LOG_ROLLOUT_DATA=${LOG_ROLLOUT_DATA:-1}
if [[ "$LOG_ROLLOUT_DATA" == "0" ]]; then
  ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-}
else
  ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-$REPO_ROOT/storage/rollouts/$EXPERIMENT_NAME}
fi
VAL_DATA_DIR=${VAL_DATA_DIR:-$REPO_ROOT/storage/validation/$EXPERIMENT_NAME}

"$PYTHON_BIN" -m verl_omni.trainer.main_diffusion \
  algorithm.adv_estimator=dance_grpo \
  actor_rollout_ref.model.algorithm=dance_grpo \
  actor_rollout_ref.actor.diffusion_loss.loss_mode=dance_grpo \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/test.parquet" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.val_batch_size=4 \
  data.max_prompt_length=1024 \
  data.dataloader_num_workers=1 \
  data.truncation=right \
  data.seed=42 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.attn_backend=native \
  actor_rollout_ref.model.custom_chat_template='"{% if messages %}{% for message in messages %}{% if message[\"role\"] == \"user\" %}{{ message[\"content\"] }}{% endif %}{% endfor %}{% endif %}</s>"' \
  actor_rollout_ref.model.lora_rank="$LORA_RANK" \
  actor_rollout_ref.model.lora_alpha="$LORA_ALPHA" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="$LR" \
  actor_rollout_ref.actor.optim.weight_decay=0.01 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=100 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE" \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params=10000 \
  actor_rollout_ref.actor.diffusion_loss.clip_ratio=0.0001 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
  actor_rollout_ref.rollout.name=vllm_omni \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.agent.num_workers="$((NUM_GPUS / ROLLOUT_TP))" \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.pipeline.height=384 \
  actor_rollout_ref.rollout.pipeline.width=384 \
  actor_rollout_ref.rollout.pipeline.num_frames="$NUM_FRAMES" \
  actor_rollout_ref.rollout.pipeline.num_inference_steps="$NUM_INFERENCE_STEPS" \
  actor_rollout_ref.rollout.pipeline.guidance_scale=1.0 \
  actor_rollout_ref.rollout.pipeline.max_sequence_length=1024 \
  actor_rollout_ref.rollout.algo.noise_level="$NOISE_LEVEL" \
  actor_rollout_ref.rollout.algo.sde_type=dance_sde \
  actor_rollout_ref.rollout.algo.sde_window_size="$SDE_WINDOW_SIZE" \
  actor_rollout_ref.rollout.algo.sde_window_range="$SDE_WINDOW_RANGE" \
  actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps="$VAL_NUM_INFERENCE_STEPS" \
  actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  reward.num_workers=4 \
  reward.reward_model.enable=False \
  reward.custom_reward_function.path=verl_omni/utils/reward_score/maze_tracker.py \
  reward.custom_reward_function.name=compute_score_maze_tracker \
  trainer.logger='["console", "tensorboard"]' \
  trainer.project_name=wan-dancegrpo-maze-5b-tracker \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.rollout_data_dir="$ROLLOUT_DATA_DIR" \
  trainer.validation_data_dir="$VAL_DATA_DIR" \
  trainer.log_val_generations="$LOG_VAL_GENERATIONS" \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node="$NUM_GPUS" \
  trainer.nnodes=1 \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.total_epochs=1 \
  trainer.total_training_steps="$MAX_STEPS" \
  "$@"
