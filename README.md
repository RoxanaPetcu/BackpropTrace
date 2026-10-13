# Local Proxies, Global Failures: Does Better Local Reasoning Improve Generation, or Do Models Strategize Differently?

This repository contains the code for a multi-objective optimization study on multi-hop QA reasoning.

We decompose the reasoning trace of a search agent into three stages: step-wise entailment, search query formulation, and answer generation, and attach a targeted **proxy reward** to each one, forming a cascading dependency chain. Rewards are assigned at the token level, so credit is attributed to the specific trace segment that produced it, rather than aggregated into a single scalar.

We train under two regimes: **simultaneous backpropagation** of all objectives, and **phased warm-up**, in which each component is optimized in isolation before the next is introduced.

## Installation

### BackpropTrace environment
```bash
conda create -n BackpropTrace python=3.9
conda activate BackpropTrace

pip install torch==2.4.0 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install vllm==0.6.3 flash-attn --no-build-isolation

pip install verl outlines sentence-transformers wandb
pip install -e .
```

### Retriever environment (optional)
If you would like to call a local retriever as the search engine, you can install the environment as follows. (We recommend using a seperate environment.)
```bash
conda create -n retriever python=3.10
conda activate retriever

conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c 
pip install transformers datasets pyserini

conda install -c pytorch -c nvidia faiss-gpu=1.8.0
pip install sentence-transformers openai pydantic-settings tiktoken

pip install uvicorn fastapi
```


## Quick start

Train a reasoning + search LLM on NQ dataset with e5 as the retriever and wikipedia as the corpus.

(1) Download the indexing and corpus.
```bash
save_path=/path/to/save/data
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

(2) Process the NQ dataset. Change template_type for prompt decomposition. 
```bash
python scripts/data_process/nq_search.py
```

(3) Launch a local retrieval server.
```bash
export RETRIEVER_IP_PATH=/path/to/save/IP/file
export DATA_PATH=/path/to/data
conda activate retriever
bash run_retriever.sh
```

(4) Run RL training (PPO) with Llama-3.2-3b-base.
```bash
export RETRIEVER_IP_PATH=/path/to/save/IP/file
export DATA_PATH=/path/to/data
export SAVE_DIR_BASE=/path//to/checkpoints
conda activate BackpropTrace
bash train_grpo.sh
```


## Appendix

Material that did not fit the paper's page limit.

### A. Experimental setup

Training merges Natural Questions and HotpotQA; validation is on HotpotQA. The backbone is Qwen2.5-3B, trained with GRPO through the verl framework on 4 NVIDIA H100 GPUs.

| Setting | Value |
|---|---|
| Group size | 5 |
| Rollout temperature | 1.0 |
| Training batch size | 512 |
| Validation batch size | 256 |
| Max prompt length | 4096 tokens |
| Max response length | 500 tokens |
| Observation length | 1200 tokens |
| Learning rate | 1 x 10^-6 |
| KL divergence coefficient (`kl_loss_coef`) | 0.001 |
| Proxy reward judge | `cross-encoder/nli-deberta-v3-small` |

### B. Step-Wise Entailment Reward

To ensure the quality of the nli-deberta-v3-small model for generating the next reasoning step, we performed qualitative human evaluations on random samples. Example are shown in Tables 4 and 5.

### C. Summary model performance

Proxy reward and exact match at the end of each warm-up phase and at the final evaluation checkpoint.

| Regime | Phase 1 Proxy | Phase 1 EM | Phase 2 Proxy | Phase 2 EM | Phase 3 Proxy | Phase 3 EM | Final Proxy | Final EM |
|---|---|---|---|---|---|---|---|---|
| Exp 1 | .913 | .265 | .928 | .195 | .203 | .393 | .203 | .393 |
| Exp 2 | .588 | .285 | .512 | .218 | .210 | .241 | .210 | .241 |
| Exp 3 (`wp1`) | .263 | .188 | – | – | – | – | .263 | .188 |
| Exp 4 (`wp-1`) | – | – | – | – | – | – | .003 | .003 |

Exp 1 is the clearest statement of the result: the proxy climbs from .913 to .928 between phases 1 and 2 while EM *falls* from .265 to .195. Only when the proxy collapses to .203 in phase 3 does EM recover to .393.

### C. Prompts and Examples

Table 3: Prompt template. The question is appended during training and inference.

The question is appended during both training and inference.

```text
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search>, and it will return the top searched results between <information> and </information>.

If the original query is complex or involves multiple parts, you are encouraged to decompose it into at most 3 smaller sub-questions, separated by ##. For example: <search> sub-question 1 ## sub-question 2 </search> and it will return the top searched results between <information> documents sub-question 1 ## documents sub-question 2 </information>.

You can search as many times as you want. Only decompose when the question has multiple independent parts (e.g., different entities, aspects, or comparisons). Do not decompose questions that do not need it.

If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer> without detailed illustrations. For example, <answer> Beijing </answer>.

Question: {question}
```


Table 4: Case study 1 (successful generation).

**Question:** When does the slender man movie come out?

```text
<think> I need to find the title of the Slender Man movie. </think>
<search> Slender Man movie </search>
<information> Doc 1 (Title: "Slender Man (film)") Slender Man is a 2018 American supernatural horror film directed by Sylvain White and written by David Birke, based on the character of the same name.
Doc 2 (Title: "Slender Man (film)") "Slender Man" was released in the United States on August 10, 2018, by Screen Gems.
Doc 3 (Title: "Slender Man") In 2016, Sony Pictures subsidiary Screen Gems partnered with Mythology Entertainment to bring a Slender Man film into theatres. </information>
<think> I found out that "Slender Man" was released in August 2018. However, I want to double-check this information. </think>
<search> release date of slender man film </search>
<information> Doc 1 (Title: "Slender Man (film)") "Slender Man" was released in the United States on August 10, 2018, by Screen Gems, grossed over $51 million worldwide. Doc 2 (Title: "Slender Man (film)") It was previously slated for May 18, and then August 24, 2018. Doc 3 (Title: "Slender Man (film)") "Slender Man" was released by Screen Gems on August 10, 2018. </information>
<think> After reviewing both sources, I confirm that the Slender Man movie was released on August 10, 2018. </think>
Answer: The Slender Man movie came out on August 10, 2018. My previous action is invalid. If I want to search, I should put the query between <search> and </search>. Let me try again.
<answer> August 10, 2018 </answer>
```
The answer is correct, but note the stray self-correction before the `<answer>` tag: the model emits a bare `Answer:` line, recognises the format violation, and recovers. This is the kind of trace the formatting warm-up (`wp0`) is meant to eliminate.

Table 5. Case study 2 (unsuccessful generation).

**Question:** Who played professor higgins in my fair lady?

```text
<think> First, I need to recall what information I already have. Since I am trying to answer the question and do not have a list of actors yet, I will search for it. </think>
<search> My Fair Lady cast of actor who plays professor Higgins </search>
<information> Doc 1 (Title: "My Fair Lady (film)") In London, Professor Henry Higgins (Rex Harrison), a scholar of phonetics, believes that the accent determines a person's prospects. 
Doc 2 (Title: "Edward Mulhare") His best-known stage role was as Professor Higgins in the original Broadway production of "My Fair Lady," having taken over from Rex Harrison in 1957. 
Doc 3 (Title: "Harry Hadden-Paton") Hadden-Paton is currently playing the lead role of Henry Higgins in the Lincoln Center Theater revival of "My Fair Lady" on Broadway. </information>
<answer> I found that Stanley Holloway played the role of Professor Higgins in the Broadway production of "My Fair Lady" in 1950. That makes him the actor who played Professor Higgins in My Fair Lady. </answer>
```

Every retrieved document names Rex Harrison, Edward Mulhare or Harry Hadden-Paton. The model answers "Stanley Holloway" — a name that appears in none of them — and asserts it confidently with a fabricated date. This is spurious grounding: a well-formed trace with a correct-looking search step, which the local proxies reward, producing an answer unsupported by the evidence it just retrieved.
