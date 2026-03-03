# LoRA Fine-Tuning and KV Cache

Fine-tuned `LiquidAI/LFM2.5-1.2B-Instruct` on ELI5-style Q/A pairs using LoRA, benchmarked KV cache impact, and deployed the adapted model as a streaming inference server.

---

## What's in here

**LoRA fine-tuning** — adapted the base model with rank-8 LoRA on attention projections (`q_proj`, `k_proj`, `v_proj`), trained for 1 epoch on 45k filtered ELI5 examples. The adapter is ~30MB vs the 2.4GB base model.

The fine-tuned model ends up shorter and faster but worse on ELI5 quality — more repetitive, higher readability grade, loses the child-friendly tone. Supervised fine-tuning on Q/A pairs without a reward signal nudges the model toward mimicking answer structure rather than answer quality. Expected, but good to see in practice.

**KV cache benchmark** — measured decode throughput with and without cache:

| Model | Cache | Avg latency | Tokens/sec |
|-------|-------|-------------|------------|
| Base  | on    | 12.63s      | 18.57      |
| Base  | off   | 59.86s      | 3.92       |
| LoRA  | on    | 10.40s      | 19.18      |
| LoRA  | off   | 56.30s      | 3.63       |

~4.7–5.4× slower without cache on ~200-token outputs. Gap widens significantly at longer sequences.

**Deployment** — merged the adapter into the base weights, served via FastAPI with SSE streaming, containerised with Docker.

---

## Structure

```
data/eli5/              dataset prep and frozen eval prompts
models/lfm2dot5/        model inspection script
training/               LoRA training (PEFT + TRL SFTTrainer)
eval/                   baseline + LoRA eval scripts and results
experiments/            KV cache benchmark
adapters/eli5-lora-v1/  LoRA adapter weights
scripts/                merge adapter into base model
server/                 FastAPI streaming server
docker/                 Dockerfile
```

---

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
# dataset
python data/eli5/prepare_dataset.py

# train
python training/train_lora.py

# eval
python eval/run_baseline_eval.py
python eval/run_lora_eval.py
python eval/compare_metrics.py

# kv cache benchmark
python experiments/kv_cache_benchmark.py

# merge and serve
python scripts/merge_adapter.py
uvicorn server.main:app --host 0.0.0.0 --port 8000

# or in Docker (runs on CPU)
docker build -f docker/Dockerfile -t eli5-server .
docker run -p 8000:8000 eli5-server
```

```bash
# streaming test
curl -N -X POST http://localhost:8000/generate/stream \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Why is the sky blue?"}'
```

---

## Config

- **Base model:** `LiquidAI/LFM2.5-1.2B-Instruct`
- **Dataset:** `sentence-transformers/eli5`, 80–300 token answers, 50k examples
- **LoRA:** r=8, alpha=16, dropout=0.05
- **Training:** 1 epoch, lr=2e-4, effective batch=8, max_seq_len=1024
- **Hardware:** Apple Silicon MPS
