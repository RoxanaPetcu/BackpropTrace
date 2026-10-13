# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np
import os
import functools

def _select_rm_score_fn(data_source, use_answerability=True):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        # if use_answerability:
        #     return qa_em.compute_score_em # qa_em.compute_score_em_plus_answerability_decomposition # qa_em.compute_score_em_plus_decomposition # qa_em.compute_score_em_plus_answerability
        # else:
        #     return qa_em.compute_score_em
        if use_answerability:
            return functools.partial(
                qa_em.compute_score_em,
                add_format=True,
                structure_format_score=0.2,
                final_format_score=0.1,
                retrieval_score=0.1,
                decomposition_bonus=0.0
            )
        else:
            return functools.partial(
                qa_em.compute_score_em,
                add_format=False
            )
    else:
        raise NotImplementedError


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, format_score=0., use_answerability=True) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score
        self.use_answerability = use_answerability

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        global_steps = int(data.meta_info.get('global_steps', 0))

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source, use_answerability=self.use_answerability)

            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            sample_id = f"SAMPLE_R{rank}_I{i}"
            qa_em.set_sample_id(sample_id)

            if self.use_answerability:
                score = compute_score_fn(
                    solution_str=sequences_str,
                    ground_truth=ground_truth,
                    format_score=self.format_score,
                    training_step=global_steps,
                )
            else:
                score = compute_score_fn(
                    solution_str=sequences_str,
                    ground_truth=ground_truth,
                    format_score=self.format_score,
                )

            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"[{sample_id}] ========== REASONING TRACE ==========")
                print(f"[{sample_id}] {sequences_str}")
                print(f"[{sample_id}] ========== END TRACE ==========")

        return reward_tensor


class SpanwiseLMRewardManager():
    """Reward manager with per-span rewards for think/search/answer."""

    def __init__(self,
                 tokenizer,
                 step_judge,
                 num_examine=0,
                 structure_token_reward=0.2,
                 valid_format_bonus=0.2,
                 no_answer_penalty=-0.5,
                 no_think_penalty=-0.3,
                 pre_warmup_steps=0,
                 format_warmup_steps=0,
                 phase_steps=50,
                 phase_decline_patience=3,
                 phase_decline_min_delta=0.0) -> None:
        self.tokenizer = tokenizer
        self.step_judge = step_judge
        self.num_examine = num_examine
        self.structure_token_reward = structure_token_reward
        self.valid_format_bonus = valid_format_bonus
        self.no_answer_penalty = no_answer_penalty
        self.no_think_penalty = no_think_penalty
        self.pre_warmup_steps = int(pre_warmup_steps)
        self.format_warmup_steps = int(format_warmup_steps)
        self.phase_steps = int(phase_steps)
        self.phase_decline_patience = int(phase_decline_patience)
        self.phase_decline_min_delta = float(phase_decline_min_delta)
        # Dynamic phase state (post-prewarmup): 1=think, 2=search, 3=EM
        self.current_phase = 1
        self._prev_think_mean = None
        self._prev_search_mean = None
        self._think_decline_streak = 0
        self._search_decline_streak = 0
        self._dynamic_phase_enabled = self.phase_steps > 0
        # Baseline Search-R1 reward behavior used during pre-warmup.
        self._baseline_reward_manager = RewardManager(tokenizer=tokenizer, num_examine=0, use_answerability=True)

    @staticmethod
    def _char_span_to_token_indices(offsets, char_start, char_end, token_cap):
        token_indices = []
        n = min(len(offsets), token_cap)
        for tok_idx in range(n):
            tok_start, tok_end = offsets[tok_idx]
            if tok_end <= tok_start:
                continue
            if tok_end <= char_start:
                continue
            if tok_start >= char_end:
                continue
            token_indices.append(tok_idx)
        return token_indices

    @staticmethod
    def _collect_structural_tag_token_indices(response_str, offsets, token_cap):
        """Collect token indices for structural tags that should receive format reward."""
        tag2type = {
            "think": 1,
            "search": 2,
            "answer": 3,
        }
        tagged = []
        tag_pattern = r"</?(think|search|information|answer)>"
        for m in re.finditer(tag_pattern, response_str):
            tag_name = m.group(1)
            # Keep information fully masked.
            if tag_name == "information":
                continue
            type_id = tag2type.get(tag_name)
            if type_id is None:
                continue
            tok_indices = SpanwiseLMRewardManager._char_span_to_token_indices(
                offsets=offsets,
                char_start=m.start(),
                char_end=m.end(),
                token_cap=token_cap,
            )
            for tok_idx in tok_indices:
                tagged.append((tok_idx, type_id))
        return tagged

    def _compute_answer_metrics(self, data: DataProto, device: torch.device):
        """Compute per-sample EM and strict has-answer flags for logging."""
        answer_em = []
        has_answer_flags = []

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = int(data_item.batch['attention_mask'][:prompt_length].sum())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = int(data_item.batch['attention_mask'][prompt_length:].sum())
            valid_response_ids = response_ids[:valid_response_length]

            sequence_ids = torch.cat((valid_prompt_ids, valid_response_ids))
            sequence_str = self.tokenizer.decode(sequence_ids)
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            answer_reward = qa_em.compute_score_em(
                solution_str=sequence_str,
                ground_truth=ground_truth,
                add_format=False,
            )
            answer_em.append(float(answer_reward))

            spans_local = qa_em.parse_reasoning_trace_spans(response_str)
            if not spans_local:
                has_answer_flags.append(0.0)
                continue

            last_type, last_text, _last_start, last_end = spans_local[-1]
            trailing_text = response_str[last_end:].strip()
            is_nonempty_answer = (last_type == 'answer') and (len(last_text.strip()) > 0)
            has_answer_flags.append(1.0 if is_nonempty_answer and trailing_text == '' else 0.0)

        answer_em_avg = torch.tensor(answer_em, dtype=torch.float32, device=device)
        has_answer_per_sample = torch.tensor(has_answer_flags, dtype=torch.float32, device=device)
        return answer_em_avg, has_answer_per_sample

    def __call__(self, data: DataProto):
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        span_type_mask = torch.zeros_like(data.batch['responses'], dtype=torch.long)
        phase_loss_mask = torch.ones_like(data.batch['responses'], dtype=torch.long)
        judge_think_avg = torch.zeros((len(data),), dtype=torch.float32, device=reward_tensor.device)
        judge_search_avg = torch.zeros((len(data),), dtype=torch.float32, device=reward_tensor.device)
        global_steps = int(data.meta_info.get('global_steps', 0))
        in_pre_warmup = (self.pre_warmup_steps > 0) and (global_steps <= self.pre_warmup_steps)
        in_format_warmup = global_steps < self.format_warmup_steps
        active_phase = self.current_phase if self._dynamic_phase_enabled else 0
        # active_phase = self.current_phase if self._dynamic_phase_enabled else 3
        current_phase_batch = torch.full((len(data),), active_phase, dtype=torch.long, device=reward_tensor.device)

        if in_pre_warmup:
            # During pre-warmup, mimic the baseline GRPO reward (single EM-style
            # outcome reward on terminal token). Advantage routing is switched to
            # standard GRPO in ray_trainer using the same step gate.
            reward_tensor = self._baseline_reward_manager(data)
            data.batch['span_type_mask'] = span_type_mask
            data.batch['phase_loss_mask'] = phase_loss_mask
            data.batch['judge_think_avg'] = judge_think_avg
            data.batch['judge_search_avg'] = judge_search_avg
            answer_em_avg, has_answer_per_sample = self._compute_answer_metrics(data, reward_tensor.device)
            data.batch['answer_em_avg'] = answer_em_avg
            data.batch['has_answer_per_sample'] = has_answer_per_sample
            # Phase id -1 denotes pre-warmup baseline stage.
            data.batch['current_phase'] = torch.full((len(data),), -1, dtype=torch.long, device=reward_tensor.device)
            # Reset dynamic phase state so transitions start clean after pre-warmup.
            self.current_phase = 1
            self._prev_think_mean = None
            self._prev_search_mean = None
            self._think_decline_streak = 0
            self._search_decline_streak = 0
            return reward_tensor

        # Phase 3: identical to pre-warmup — single terminal-token EM reward,
        # loss propagated to all non-information tokens via state_masking.
        if active_phase == 3:
            reward_tensor = self._baseline_reward_manager(data)
            data.batch['span_type_mask'] = span_type_mask
            data.batch['phase_loss_mask'] = phase_loss_mask
            data.batch['judge_think_avg'] = judge_think_avg
            data.batch['judge_search_avg'] = judge_search_avg
            answer_em_avg, has_answer_per_sample = self._compute_answer_metrics(data, reward_tensor.device)
            data.batch['answer_em_avg'] = answer_em_avg
            data.batch['has_answer_per_sample'] = has_answer_per_sample
            data.batch['current_phase'] = torch.full((len(data),), 3, dtype=torch.long, device=reward_tensor.device)
            return reward_tensor

        phase_think_scores = []
        phase_search_scores = []
        phase_answer_scores = []

        already_printed = 0

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = int(data_item.batch['attention_mask'][:prompt_length].sum())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = int(data_item.batch['attention_mask'][prompt_length:].sum())
            valid_response_ids = response_ids[:valid_response_length]

            sequence_ids = torch.cat((valid_prompt_ids, valid_response_ids))
            sequence_str = self.tokenizer.decode(sequence_ids)
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            question = qa_em.extract_question_from_solution(sequence_str)
            spans = qa_em.parse_reasoning_trace_spans(response_str)

            answer_reward = qa_em.compute_score_em(
                solution_str=sequence_str,
                ground_truth=ground_truth,
                add_format=False,
            )
            em_success = float(answer_reward) >= 0.999
            phase_answer_scores.append(float(answer_reward))

            try:
                tokenized_resp = self.tokenizer(
                    response_str,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                offsets = tokenized_resp.get('offset_mapping', [])
            except Exception:
                offsets = []

            previous_thinks = []
            last_think = ""
            think_scores = []
            search_scores = []

            for span_type, span_text, char_start, char_end in spans:
                if span_type == 'think':
                    raw_reward = float(self.step_judge['think_judge'](question, previous_thinks, span_text))
                    think_scores.append(raw_reward)
                    phase_think_scores.append(raw_reward)
                    type_id = 1
                    previous_thinks.append(span_text)
                    last_think = span_text
                elif span_type == 'search':
                    raw_reward = float(self.step_judge['search_judge'](last_think, span_text))
                    search_scores.append(raw_reward)
                    phase_search_scores.append(raw_reward)
                    type_id = 2
                elif span_type == 'answer':
                    raw_reward = float(answer_reward)
                    type_id = 3
                else:
                    raw_reward = 0.0
                    type_id = 0

                # If EM is correct, propagate reward 1.0 to all learnable spans
                # (think/search/answer), while keeping information spans masked.
                if em_success and type_id in (1, 2, 3):
                    raw_reward = 1.0

                # Mandatory warm-up: optimize formatting first by zeroing span-wise
                # semantic rewards (think/search/answer). Structure/format bonuses
                # are still added below.
                if in_format_warmup and type_id in (1, 2, 3):
                    raw_reward = 0.0

                # Keep phase masking only for non-EM-success traces.
                # Phase 1: think judge active; mask search, keep answer (EM).
                # Phase 2: search judge active; mask think, keep answer (EM).
                if (not em_success) and self._dynamic_phase_enabled and type_id in (1, 2, 3):
                    if active_phase == 1 and type_id == 2:
                        raw_reward = 0.0
                    elif active_phase == 2 and type_id == 1:
                        raw_reward = 0.0

                tok_indices = self._char_span_to_token_indices(
                    offsets=offsets,
                    char_start=char_start,
                    char_end=char_end,
                    token_cap=valid_response_length,
                )

                for tok_idx in tok_indices:
                    reward_tensor[i, tok_idx] = raw_reward
                    span_type_mask[i, tok_idx] = type_id

            # Add a small token-local format reward to structural tags.
            tagged_tokens = self._collect_structural_tag_token_indices(
                response_str=response_str,
                offsets=offsets,
                token_cap=valid_response_length,
            )
            for tok_idx, type_id in tagged_tokens:
                reward_tensor[i, tok_idx] += self.structure_token_reward
                if span_type_mask[i, tok_idx] == 0:
                    span_type_mask[i, tok_idx] = type_id

            # Add global format bonus over learnable reasoning tokens.
            is_valid_format, _ = qa_em.is_valid_sequence(sequence_str)
            if is_valid_format:
                learnable_mask = (span_type_mask[i] == 1) | (span_type_mask[i] == 2) | (span_type_mask[i] == 3)
                reward_tensor[i, learnable_mask] += self.valid_format_bonus

            # Penalize truncated/looping traces that never produced </answer>.
            # Applies in all phases (warm_up or not): the model must learn to terminate.
            has_answer = any(span_type == 'answer' for span_type, _, _, _ in spans)
            if not has_answer:
                last_valid_tok = valid_response_length - 1
                reward_tensor[i, last_valid_tok] += self.no_answer_penalty

            # Penalize traces that skip <think> tags entirely. The model must not
            # bypass the think judge by putting reasoning as free text outside tags.
            has_think = any(span_type == 'think' for span_type, _, _, _ in spans)
            if not has_think and not in_format_warmup:
                last_valid_tok = valid_response_length - 1
                reward_tensor[i, last_valid_tok] += self.no_think_penalty

            if think_scores:
                judge_think_avg[i] = float(np.mean(think_scores))
            if search_scores:
                judge_search_avg[i] = float(np.mean(search_scores))

            # Apply phase-specific token update masks so only desired spans receive
            # policy/KL/entropy updates in warmup phases.
            if self._dynamic_phase_enabled and active_phase in (1, 2):
                if active_phase == 1:
                    allowed_mask = (span_type_mask[i] == 1) | (span_type_mask[i] == 3)
                else:
                    allowed_mask = (span_type_mask[i] == 2) | (span_type_mask[i] == 3)

                sample_phase_mask = torch.zeros_like(span_type_mask[i], dtype=torch.long)
                sample_phase_mask[:valid_response_length] = allowed_mask[:valid_response_length].long()
                if sample_phase_mask[:valid_response_length].sum() == 0 and valid_response_length > 0:
                    # Fallback: always keep at least one token trainable.
                    sample_phase_mask[valid_response_length - 1] = 1
                phase_loss_mask[i] = sample_phase_mask

            if already_printed < self.num_examine:
                already_printed += 1
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                sample_id = f"SPAN_R{rank}_I{i}"
                print(f"[{sample_id}] ========== REASONING TRACE ==========")
                print(f"[{sample_id}] {sequence_str}")
                print(f"[{sample_id}] ========== END TRACE ==========")

        n = len(data)
        answer_em_avg = torch.tensor(
            [phase_answer_scores[i] if i < len(phase_answer_scores) else 0.0 for i in range(n)],
            dtype=torch.float32, device=reward_tensor.device,
        )
        # Strict has_answer_ratio:
        # count as 1 only if the final meaningful span is a non-empty <answer>
        # and there is no non-whitespace trailing text after </answer>.
        has_answer_flags = []
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch['responses']
            valid_response_length = int(data_item.batch['attention_mask'][prompt_length:].sum())
            valid_response_ids = response_ids[:valid_response_length]
            response_str = self.tokenizer.decode(valid_response_ids)
            spans_local = qa_em.parse_reasoning_trace_spans(response_str)

            if not spans_local:
                has_answer_flags.append(0.0)
                continue

            last_type, last_text, _last_start, last_end = spans_local[-1]
            trailing_text = response_str[last_end:].strip()
            is_nonempty_answer = (last_type == 'answer') and (len(last_text.strip()) > 0)
            has_answer_flags.append(1.0 if is_nonempty_answer and trailing_text == '' else 0.0)

        has_answer_per_sample = torch.tensor(
            has_answer_flags,
            dtype=torch.float32,
            device=reward_tensor.device,
        )

        # Dynamic phase transitions based on consecutive declines.
        if self._dynamic_phase_enabled:
            if self.current_phase == 1 and phase_think_scores:
                think_mean = float(np.mean(phase_think_scores))
                if self._prev_think_mean is not None and think_mean < (self._prev_think_mean - self.phase_decline_min_delta):
                    self._think_decline_streak += 1
                else:
                    self._think_decline_streak = 0
                self._prev_think_mean = think_mean
                if self._think_decline_streak >= self.phase_decline_patience:
                    self.current_phase = 2
                    self._think_decline_streak = 0
                    self._prev_search_mean = None
                    self._search_decline_streak = 0

            elif self.current_phase == 2 and phase_search_scores:
                search_mean = float(np.mean(phase_search_scores))
                if self._prev_search_mean is not None and search_mean < (self._prev_search_mean - self.phase_decline_min_delta):
                    self._search_decline_streak += 1
                else:
                    self._search_decline_streak = 0
                self._prev_search_mean = search_mean
                if self._search_decline_streak >= self.phase_decline_patience:
                    self.current_phase = 3
                    self._search_decline_streak = 0

        data.batch['span_type_mask'] = span_type_mask
        data.batch['phase_loss_mask'] = phase_loss_mask
        data.batch['judge_think_avg'] = judge_think_avg
        data.batch['judge_search_avg'] = judge_search_avg
        data.batch['answer_em_avg'] = answer_em_avg
        data.batch['has_answer_per_sample'] = has_answer_per_sample
        data.batch['current_phase'] = current_phase_batch
        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        token = os.getenv("HF_TOKEN")
        token = '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'
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN', 'HF_TOKEN': token}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer
    from verl.utils.reward_score import qa_em

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf

    ## ADDED
    qa_em.set_judge_fn(
        qa_em.init_mpnet_judge(
            # model_name="sentence-transformers/all-mpnet-base-v2",
            model_name="sentence-transformers/all-MiniLM-L12-v2",
            device="cpu",
            threshold=0.35,
        )
    )



    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    if config.algorithm.adv_estimator == 'grpo_spanwise_lm':
        step_judge = qa_em.init_llm_step_judge(
            model_name="all-MiniLM-L6-v2",
            think_judge_mode=config.algorithm.get('think_judge_mode', 'nli'),
            reasoning_prompt_model=config.algorithm.get('reasoning_prompt_model', 'Qwen/Qwen2.5-0.5B-Instruct'),
        )
        reward_fn = SpanwiseLMRewardManager(
            tokenizer=tokenizer,
            step_judge=step_judge,
            num_examine=0,
            structure_token_reward=float(config.algorithm.get('structure_token_reward', 0.2)),
            valid_format_bonus=float(config.algorithm.get('valid_format_bonus', 0.2)),
            no_answer_penalty=float(config.algorithm.get('no_answer_penalty', -0.5)),
            no_think_penalty=float(config.algorithm.get('no_think_penalty', -0.3)),
            pre_warmup_steps=int(config.algorithm.get('pre_warmup_steps', 0)),
            format_warmup_steps=int(config.algorithm.get('format_warmup_steps', 0)),
            phase_steps=int(config.algorithm.get('phase_steps', 50)),
            phase_decline_patience=int(config.algorithm.get('phase_decline_patience', 3)),
            phase_decline_min_delta=float(config.algorithm.get('phase_decline_min_delta', 0.0)),
        )
    else:
        reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, use_answerability=True)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, use_answerability=False)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
