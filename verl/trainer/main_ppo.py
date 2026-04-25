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
                 structure_token_reward=0.02,
                 valid_format_bonus=0.05,
                 warm_up_phases=False,
                 phase_threshold=0.6,
                 phase_ema_decay=0.9) -> None:
        self.tokenizer = tokenizer
        self.step_judge = step_judge
        self.num_examine = num_examine
        self.structure_token_reward = structure_token_reward
        self.valid_format_bonus = valid_format_bonus
        self.warm_up_phases = warm_up_phases
        self.phase_threshold = phase_threshold
        self.phase_ema_decay = phase_ema_decay
        # Phase state: 1=think-only, 2=search-only, 3=all
        self._current_phase = 1
        self._phase_ema = None

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

    def __call__(self, data: DataProto):
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        span_type_mask = torch.zeros_like(data.batch['responses'], dtype=torch.long)
        judge_think_avg = torch.zeros((len(data),), dtype=torch.float32, device=reward_tensor.device)
        judge_search_avg = torch.zeros((len(data),), dtype=torch.float32, device=reward_tensor.device)

        already_printed = 0
        # Collect across all samples for phase EMA update.
        all_think_scores = []
        all_search_scores = []

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            sequence_ids = torch.cat((valid_prompt_ids, valid_response_ids))
            sequence_str = self.tokenizer.decode(sequence_ids)
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            question = qa_em.extract_question_from_solution(sequence_str)
            spans = qa_em.parse_reasoning_trace_spans(response_str)
            valid_response_length = int(valid_response_length)

            answer_reward = qa_em.compute_score_em(
                solution_str=sequence_str,
                ground_truth=ground_truth,
                add_format=False,
            )

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
                    judge_score = float(self.step_judge['think_judge'](question, previous_thinks, span_text))
                    think_scores.append(judge_score)
                    type_id = 1
                    previous_thinks.append(span_text)
                    last_think = span_text
                    # Gate: think content only active in phase 1 or 3
                    raw_reward = judge_score if (not self.warm_up_phases or self._current_phase != 2) else 0.0
                elif span_type == 'search':
                    judge_score = float(self.step_judge['search_judge'](last_think, span_text))
                    search_scores.append(judge_score)
                    type_id = 2
                    # Gate: search content only active in phase 2 or 3
                    raw_reward = judge_score if (not self.warm_up_phases or self._current_phase != 1) else 0.0
                elif span_type == 'answer':
                    type_id = 3
                    # Gate: answer content only active in phase 3
                    raw_reward = float(answer_reward) if (not self.warm_up_phases or self._current_phase == 3) else 0.0
                else:
                    raw_reward = 0.0
                    type_id = 0

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

            if think_scores:
                judge_think_avg[i] = float(np.mean(think_scores))
            if search_scores:
                judge_search_avg[i] = float(np.mean(search_scores))
            all_think_scores.extend(think_scores)
            all_search_scores.extend(search_scores)

            if already_printed < self.num_examine:
                already_printed += 1
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                sample_id = f"SPAN_R{rank}_I{i}"
                print(f"[{sample_id}] ========== REASONING TRACE ==========")
                print(f"[{sample_id}] {sequence_str}")
                print(f"[{sample_id}] ========== END TRACE ==========")

        data.batch['span_type_mask'] = span_type_mask
        data.batch['judge_think_avg'] = judge_think_avg
        data.batch['judge_search_avg'] = judge_search_avg

        # Phase warm-up: update EMA and check for phase transition.
        if self.warm_up_phases and self._current_phase < 3:
            active_scores = all_think_scores if self._current_phase == 1 else all_search_scores
            if active_scores:
                batch_score = float(np.mean(active_scores))
                if self._phase_ema is None:
                    self._phase_ema = batch_score
                else:
                    self._phase_ema = (self.phase_ema_decay * self._phase_ema
                                       + (1.0 - self.phase_ema_decay) * batch_score)
                if self._phase_ema >= self.phase_threshold:
                    old_phase = self._current_phase
                    self._current_phase += 1
                    self._phase_ema = None  # reset EMA for new phase
                    print(f"[WarmUpPhases] Phase transition {old_phase} -> {self._current_phase} "
                          f"(EMA crossed threshold {self.phase_threshold:.2f})")

        data.batch['current_phase'] = torch.full(
            (len(data),), self._current_phase, dtype=torch.long, device=reward_tensor.device
        )
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
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            device="cpu",
            max_new_tokens=4,
        )
        reward_fn = SpanwiseLMRewardManager(
            tokenizer=tokenizer,
            step_judge=step_judge,
            num_examine=0,
            warm_up_phases=bool(getattr(config.algorithm, 'warm_up_phases', False)),
            phase_threshold=float(getattr(config.algorithm, 'phase_threshold', 0.6)),
            phase_ema_decay=float(getattr(config.algorithm, 'phase_ema_decay', 0.9)),
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
