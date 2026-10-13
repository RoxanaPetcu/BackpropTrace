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
