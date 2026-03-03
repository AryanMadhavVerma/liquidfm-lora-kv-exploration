import json
import os
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "LiquidAI/LFM2.5-1.2B-Instruct"
ADAPTER_PATH = "adapters/eli5-lora-v1/checkpoint-5625"
PROMPTS_PATH = "data/eli5/eval_prompts.json"
OUTPUTS_PATH = "eval/kv_cache_outputs.json"
METRICS_PATH = "eval/kv_cache_metrics.csv"

SYSTEM_PROMPT = (
    "You are an expert at explaining complex topics in a simple, easy-to-understand way. "
    "You will explain as if I'm a 5-10 year old"
)

NUM_PROMPTS = 10
MAX_NEW_TOKENS = 350


def estimate_kv_cache_mb(model, seq_len, batch_size=1):
    cfg = model.config
    num_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    num_attention_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
    num_kv_heads = getattr(cfg, "num_key_value_heads", None) or num_attention_heads
    hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)

    if not (num_layers and num_attention_heads and num_kv_heads and hidden_size):
        return None

    head_dim = hidden_size // num_attention_heads

    dtype = model.dtype or torch.float16
    try:
        bytes_per_element = torch.finfo(dtype).bits // 8
    except (TypeError, ValueError):
        bytes_per_element = torch.iinfo(dtype).bits // 8

    total_bytes = (
        batch_size
        * num_layers
        * 2
        * num_kv_heads
        * seq_len
        * head_dim
        * bytes_per_element
    )
    return total_bytes / (1024**2)


def load_model(variant):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="mps" if torch.backends.mps.is_available() else "cpu",
        dtype=torch.float16,
    )

    if variant == "lora":
        if not os.path.isdir(ADAPTER_PATH):
            print(f"adapter not found at {ADAPTER_PATH}, skipping lora")
            return None, None
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    model.eval()
    return model, tokenizer


def run_variant(model, tokenizer, prompts, model_label, use_cache):
    outputs = []
    metrics = []
    total_latency = 0.0
    total_tokens = 0

    for i, prompt in enumerate(prompts, start=1):
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        chat_prompt = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

        start = time.perf_counter()
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=use_cache,
            )
        end = time.perf_counter()

        generated_text = tokenizer.decode(gen[0], skip_special_tokens=True)
        assistant = generated_text.split("assistant")[-1].strip()

        latency = end - start
        gen_tokens = gen.shape[1] - inputs.input_ids.shape[1]
        total_seq_len = gen.shape[1]

        total_latency += latency
        total_tokens += gen_tokens

        kv_cache_mb = (
            estimate_kv_cache_mb(model, total_seq_len, batch_size=1) if use_cache else 0.0
        )

        outputs.append(
            {
                "model_variant": model_label,
                "use_cache": use_cache,
                "prompt": prompt,
                "generated_answer": assistant,
                "latency_s": latency,
                "generated_tokens": gen_tokens,
                "kv_cache_mb": kv_cache_mb,
            }
        )

        metrics.append(
            {
                "model_variant": model_label,
                "use_cache": use_cache,
                "latency_s": latency,
                "generated_tokens": gen_tokens,
                "tokens_per_sec": (gen_tokens / latency) if latency > 0 else 0.0,
                "kv_cache_mb": kv_cache_mb,
            }
        )

        if i % 5 == 0:
            print(f"{model_label} | use_cache={use_cache} | processed {i}/{len(prompts)}")

    avg_latency = total_latency / len(prompts)
    tokens_per_sec = total_tokens / total_latency if total_latency > 0 else 0.0
    summary = {
        "model_variant": model_label,
        "use_cache": use_cache,
        "avg_latency_s": avg_latency,
        "tokens_per_sec": tokens_per_sec,
    }

    return outputs, metrics, summary


def main():
    print("loading prompts...")
    with open(PROMPTS_PATH, "r") as f:
        prompts = json.load(f)
    prompts = prompts[:NUM_PROMPTS]
    print(f"using {len(prompts)} prompts")

    all_outputs = []
    all_metrics = []

    for variant in ("base", "lora"):
        print(f"loading {variant} model...")
        model, tokenizer = load_model(variant)
        if model is None:
            continue

        for use_cache in (True, False):
            print(f"running {variant} | use_cache={use_cache}")
            outputs, metrics, summary = run_variant(
                model, tokenizer, prompts, variant, use_cache
            )
            all_outputs.extend(outputs)
            all_metrics.extend(metrics)
            print(
                f"{variant} | use_cache={use_cache} | "
                f"avg_latency={summary['avg_latency_s']:.2f}s | "
                f"tokens/sec={summary['tokens_per_sec']:.2f}"
            )

        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print("saving outputs and metrics...")
    with open(OUTPUTS_PATH, "w") as f:
        json.dump(all_outputs, f, indent=2)
    pd.DataFrame(all_metrics).to_csv(METRICS_PATH, index=False)
    print(f"Saved outputs to: {OUTPUTS_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
