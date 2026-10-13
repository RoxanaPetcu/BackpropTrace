# #!/bin/bash

# # 1. Environment Setup (Matching your PPO logic)
# module purge
# module load 2025
# module load Anaconda3/2025.06-1

# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate subsearch

# # 2. Automatically get the Retriever IP
# IP_FILE="/<path>/current_retriever_ip_backproptrace.txt"

# if [ -f "$IP_FILE" ]; then
#     RETRIEVER_IP=$(cat "$IP_FILE")
#     echo "Connecting to Retriever at: $RETRIEVER_IP"
# else
#     echo "ERROR: Retriever IP file not found at $IP_FILE"
#     exit 1
# fi

# export HF_HOME="/<path>/hf_cache"
# export TRITON_CACHE_DIR="/<path>/triton_cache"
# export PYTHONUNBUFFERED=1

# # Ensure the directories exist on scratch
# mkdir -p $HF_HOME
# mkdir -p $TRITON_CACHE_DIR

# # 3. NCCL and GPU configuration
# export NCCL_SOCKET_IFNAME=ib-bond0
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_P2P_DISABLE=1
# export NCCL_TIMEOUT=1800
# export NCCL_DEBUG=WARN

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# WAND_PROJECT='BackpropTrace'

# # export DATA_DIR='/<path>/projects/Search-R1/data/nq_search'
# # export DATA_DIR='/<path>/projects/Search-R1/data/nq_search_decomposed'
# # export DATA_DIR='/<path>/projects/Search-R1/data/nq_search_decomposed_val'
# export DATA_DIR='/<path>/projects/BackpropTrace/data/nq_hotpotqa_search_decomposed'

# # export BASE_MODEL='meta-llama/Llama-3.2-3B'
# # export BASE_MODEL='Qwen/Qwen2.5-3B'
# export BASE_MODEL='Qwen/Qwen2.5-3B'


# ### MODIFY HERE
# # model="llama3.2-3b"
# model="Qwen-Qwen2.5-3B" 
# prompt_decomposition="true"
# r_answerability="true"
# alpha=0.8
# beta=0.2
# lr="1e6"
# ### MODIFY HERE

# EXPERIMENT_NAME="subsearch-nq+hotpotqa-grpo-format-${model}"

# # Standard string comparison
# if [ "$prompt_decomposition" == "true" ]; then
#   EXPERIMENT_NAME="${EXPERIMENT_NAME}-prompt-decomp"
# fi

# if [ "$r_answerability" == "true" ]; then
#   EXPERIMENT_NAME="${EXPERIMENT_NAME}-r_answerability-alpha${alpha}-beta${beta}"
# fi
# if [ "$prompt_decomposition" == "true" ]; then
#   max_obs_length=1200
# else
#   max_obs_length=500
# fi

# EXPERIMENT_NAME="${EXPERIMENT_NAME}-lr${lr}"

# export EXPERIMENT_NAME
# echo "Saving to WandB as: $EXPERIMENT_NAME"



# export VLLM_ATTENTION_BACKEND=XFORMERS

# export SAVE_DIR="/<path>/verl_checkpoints/$EXPERIMENT_NAME"

# # 4. GRPO Execution FOR BASE:
# # actor_rollout_ref.model.path=$BASE_MODEL
# # actor_rollout_ref.actor.optim.lr=1e-6 \    
# # actor_rollout_ref.actor.kl_loss_coef=0.001 \
# # actor_rollout_ref.rollout.temperature=1 \

# # FOR INSTRUCT:
# # EM only
# # lr=5e-7
# # kl=0.003
# # temperature=0.8
# PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
#     data.train_files=$DATA_DIR/train.parquet \
#     data.val_files=$DATA_DIR/val.parquet \
#     data.train_batch_size=256 \
#     data.val_batch_size=128 \
#     data.max_prompt_length=4096 \
#     data.max_response_length=500 \
#     data.max_start_length=2048 \
#     data.max_obs_length=$max_obs_length \
#     data.shuffle_train_dataloader=True \
#     algorithm.adv_estimator=grpo \
#     actor_rollout_ref.model.path=$BASE_MODEL \
#     actor_rollout_ref.model.enable_gradient_checkpointing=true \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
#     actor_rollout_ref.actor.use_kl_loss=true \
#     actor_rollout_ref.actor.ppo_mini_batch_size=128 \
#     actor_rollout_ref.actor.ppo_micro_batch_size=128 \
#     actor_rollout_ref.actor.fsdp_config.param_offload=true \
#     actor_rollout_ref.actor.fsdp_config.grad_offload=true \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
#     actor_rollout_ref.rollout.name=vllm \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
#     actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
#     actor_rollout_ref.ref.fsdp_config.param_offload=True \
#     actor_rollout_ref.actor.kl_loss_coef=0.001 \
#     actor_rollout_ref.actor.kl_loss_type=low_var_kl \
#     actor_rollout_ref.rollout.n_agent=5 \
#     actor_rollout_ref.rollout.temperature=1 \
#     actor_rollout_ref.actor.state_masking=true \
#     trainer.logger=['wandb'] \
#     +trainer.val_only=false \
#     +trainer.val_before_train=true \
#     trainer.n_gpus_per_node=4 \
#     trainer.nnodes=1 \
#     trainer.save_freq=50 \
#     trainer.test_freq=50 \
#     trainer.project_name=$WAND_PROJECT \
#     trainer.experiment_name=$EXPERIMENT_NAME \
#     trainer.total_epochs=15 \
#     trainer.total_training_steps=200 \
#     trainer.default_local_dir=$SAVE_DIR \
#     trainer.default_hdfs_dir=null \
#     +trainer.dirs_exist_ok=false \
#     max_turns=4 \
#     retriever.url="http://${RETRIEVER_IP}:8000/retrieve" \
#     retriever.topk=3 \
#     2>&1 | tee $EXPERIMENT_NAME.log



#!/bin/bash

# 1. Environment Setup (Matching your PPO logic)
module purge
module load 2025
module load Anaconda3/2025.06-1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate subsearch

# 2. Automatically get the Retriever IP
IP_FILE="/<path>/current_retriever_ip_backproptrace.txt"

if [ -f "$IP_FILE" ]; then
    RETRIEVER_IP=$(cat "$IP_FILE")
    echo "Connecting to Retriever at: $RETRIEVER_IP"
else
    echo "ERROR: Retriever IP file not found at $IP_FILE"
    exit 1
fi

export HF_HOME="/<path>/hf_cache"
export TRITON_CACHE_DIR="/<path>/triton_cache"
export PYTHONUNBUFFERED=1

# Ensure the directories exist on scratch
mkdir -p $HF_HOME
mkdir -p $TRITON_CACHE_DIR

# 3. NCCL and GPU configuration
export NCCL_SOCKET_IFNAME=ib-bond0
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_P2P_DISABLE=1
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=WARN

export CUDA_VISIBLE_DEVICES=0,1,2,3
WAND_PROJECT='BackpropTrace'

# export DATA_DIR='/<path>/projects/Search-R1/data/nq_search'
# export DATA_DIR='/<path>/projects/Search-R1/data/nq_search_decomposed'
# export DATA_DIR='/<path>/projects/Search-R1/data/nq_search_decomposed_val'
# export DATA_DIR='/<path>/projects/BackpropTrace/data/subset_nq_hotpotqa_search_decomposed'
export DATA_DIR='/<path>/projects/BackpropTrace/data/nq_hotpotqa_search_decomposed'

# export BASE_MODEL='meta-llama/Llama-3.2-3B'
# export BASE_MODEL='Qwen/Qwen2.5-3B'
export BASE_MODEL='Qwen/Qwen2.5-3B'


### MODIFY HERE
# model="llama3.2-3b"
model="Qwen-Qwen2.5-3B" 
training_phase="experiment1_spanwise" # options: pretrain_format_baseline, experiment1_spanwise

# Defaults for spanwise Experiment 1 (overridden by pretrain baseline block below)
adv_estimator="grpo_spanwise_lm"
format_warmup_steps="0"
pre_warmup_steps="50"
# phase_steps="0"
phase_steps="50"
phase_decline_patience="1"
phase_decline_min_delta="0.0"
structure_token_reward="0.2"
valid_format_bonus="0.2"

# Runtime knobs (overridden per phase)
val_batch_size=32
ppo_micro_batch_size=32
rollout_n_agent=5
save_freq=100
val_before_train=false
total_epochs=2000
total_training_steps=600

# EXPERIMENT_NAME="Backprop-SubSample-Experiment1-format-pretraining-EM-nli-deberta-entailment-minus-neutral-NOT-BY-STEPS-nq+hotpotqa-format-${model}"
EXPERIMENT_NAME="Backprop-ALL-Experiment2-format-pretraining-EM-nli-deberta-entailment-minus-neutral-NOT-BY-STEPS-nq+hotpotqa-format-${model}"
# EXPERIMENT_NAME="Backprop-SubSample-Experiment2-format-pretraining-EM-reasoning-Qwen-NOT-BY-STEPS-nq+hotpotqa-format-${model}"
prompt_decomposition="true"
r_answerability="true"
think_judge_mode="nli"  # options: nli | reasoning_prompt
reasoning_prompt_model="Qwen/Qwen2.5-0.5B-Instruct"
alpha=0.8
beta=0.2
lr="1e6"
### MODIFY HERE

if [ "$training_phase" == "pretrain_format_baseline" ]; then
  # Stable Search-R1 style baseline: EM reward with standard GRPO.
  adv_estimator="grpo"
  val_batch_size=128
  ppo_micro_batch_size=128
  rollout_n_agent=5
  save_freq=50
  val_before_train=true
  total_epochs=15
  total_training_steps=50
  EXPERIMENT_NAME="Backprop-SubSample-Pretrain50-BaselineEM-${model}"
fi

# Standard string comparison
if [ "$prompt_decomposition" == "true" ]; then
  EXPERIMENT_NAME="${EXPERIMENT_NAME}-prompt-decomp"
fi

if [ "$r_answerability" == "true" ]; then
  EXPERIMENT_NAME="${EXPERIMENT_NAME}-r_answerability-alpha${alpha}-beta${beta}"
fi
if [ "$prompt_decomposition" == "true" ]; then
  max_obs_length=1200
else
  max_obs_length=500
fi

EXPERIMENT_NAME="${EXPERIMENT_NAME}-phases${phase_steps}"

if [ "$training_phase" == "pretrain_format_baseline" ]; then
  EXPERIMENT_NAME="${EXPERIMENT_NAME}-phase-pretrain50"
fi


EXPERIMENT_NAME="${EXPERIMENT_NAME}-lr${lr}"

export EXPERIMENT_NAME
echo "Saving to WandB as: $EXPERIMENT_NAME"



export VLLM_ATTENTION_BACKEND=XFORMERS

export SAVE_DIR="/<path>/verl_checkpoints/$EXPERIMENT_NAME"

extra_algorithm_args=()
if [ "$training_phase" != "pretrain_format_baseline" ]; then
  extra_algorithm_args+=(
    algorithm.pre_warmup_steps=$pre_warmup_steps
    algorithm.format_warmup_steps=$format_warmup_steps
    algorithm.phase_steps=$phase_steps
    algorithm.phase_decline_patience=$phase_decline_patience
    algorithm.phase_decline_min_delta=$phase_decline_min_delta
    algorithm.structure_token_reward=$structure_token_reward
    algorithm.valid_format_bonus=$valid_format_bonus
    algorithm.think_judge_mode=$think_judge_mode
    algorithm.reasoning_prompt_model=$reasoning_prompt_model
  )
fi

# 4. GRPO Execution FOR BASE:
# actor_rollout_ref.model.path=$BASE_MODEL
# actor_rollout_ref.actor.optim.lr=1e-6 \    
# actor_rollout_ref.actor.kl_loss_coef=0.001 \
# actor_rollout_ref.rollout.temperature=1 \

# FOR INSTRUCT:
# EM only
# lr=5e-7
# kl=0.003
# temperature=0.8
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/val.parquet \
    data.train_batch_size=256 \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=$max_obs_length \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=$adv_estimator \
    ${extra_algorithm_args[@]} \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=$ppo_micro_batch_size \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.n_agent=$rollout_n_agent \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=$val_before_train \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=$save_freq \
    trainer.test_freq=50 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=$total_epochs \
    trainer.total_training_steps=$total_training_steps \
    trainer.default_local_dir=$SAVE_DIR \
    trainer.default_hdfs_dir=null \
    +trainer.dirs_exist_ok=false \
    max_turns=4 \
    retriever.url="http://${RETRIEVER_IP}:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee $EXPERIMENT_NAME.log