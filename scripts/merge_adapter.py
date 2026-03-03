"""
Merge LoRA adapter weights into the base model and save a standalone model.

After this runs, models/eli5-merged/ contains a self-contained model that
requires no PEFT at inference time. Run once; takes ~1-2 min on MPS.

Usage:
    python scripts/merge_adapter.py
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "LiquidAI/LFM2.5-1.2B-Instruct"
ADAPTER_PATH = "adapters/eli5-lora-v1"
OUTPUT_PATH = "models/eli5-merged"


def main():
    output_dir = Path(OUTPUT_PATH)
    if output_dir.exists():
        print(f"{OUTPUT_PATH} already exists — delete it first to re-merge.")
        return

    print("loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",   # merge on CPU to avoid MPS memory pressure
    )

    print("loading adapter...")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)

    print("merging adapter weights into base (this is the permanent bake-in)...")
    model = model.merge_and_unload()

    print(f"saving merged model to {OUTPUT_PATH}...")
    output_dir.mkdir(parents=True)
    model.save_pretrained(OUTPUT_PATH)

    print("saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)

    print(f"done — merged model saved to {OUTPUT_PATH}/")
    print(f"size: {sum(p.numel() * 2 for p in model.parameters()) / 1e9:.2f} GB (float16)")


if __name__ == "__main__":
    main()
