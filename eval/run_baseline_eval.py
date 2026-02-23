import json
import time
import pandas as pd
import textstat
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "LiquidAI/LFM2.5-1.2B-Instruct"
PROMPTS_PATH = "data/eli5/eval_prompts.json"
OUTPUTS_PATH = "eval/baseline_outputs.json"
METRICS_PATH = "eval/baseline_metrics.csv"

SYSTEM_PROMPT = "You are an expert at explaining complex topics in a simple, easy-to-understand way. You will explain as if I'm a 5-10 year old"

def main():
    print("loading model and tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="mps" if torch.backends.mps.is_available() else "cpu",
        dtype=torch.float16
    )

    print("loading prompts...")
    with open(PROMPTS_PATH, "r") as f:
        prompts = json.load(f)

    print(f"running baseline eval on {len(prompts)} prompts..")

    outputs = []
    metrics = []
    total_latency = 0.0
    total_tokens = 0

    for i, prompt in enumerate(prompts, start=1):
        conversation = [
            {"role":"system", "content": SYSTEM_PROMPT},
            {"role":"user", "content": prompt}
        ]
        chat_prompt = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

        start = time.time()
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=350,
                do_sample=False #deterministic
            )

        end = time.time()

        # gen which is model.generate will give tensor of shape [batch,seq_len] we are running with batch size 1 so gen has shape [1,seq_len]
        #do_sample does greedy decoding, doesnt sample takes the highest probablity logit as next token thoguhout the generation process at each step
        generated_text = tokenizer.decode(gen[0], skip_special_tokens=True)

        assistant = generated_text.split("assistant")[-1].strip() #content field and then strip it 
        latency = end - start
        gen_tokens = gen.shape[1] - inputs.input_ids.shape[1]

        total_latency  += latency
        total_tokens +=gen_tokens


        outputs.append({
            "prompt": prompt,
            "generated_answer": assistant,
            "latency_s": latency,
            "generated_tokens": gen_tokens,
        })

        try:
            readability = textstat.flesch_kincaid_grade(assistant)
        except Exception:
            readability = -1

        metrics.append({
            "prompt": prompt,
            "output_token_length": gen_tokens,
            "readability_grade": readability,
            "latency_s": latency,
        })

        if i % 5 ==0:
            print(f"Processed {i/len(prompts)}")
    
    avg_latency = total_latency / len(prompts)
    tokens_per_sec = total_tokens / total_latency
    print(f"Avg latency: {avg_latency:.2f}s | Tokens/sec: {tokens_per_sec:.2f}")

    print("saving outputs and metrics...")
    with open(OUTPUTS_PATH, "w") as f:
        json.dump(outputs, f, indent=2)
    pd.DataFrame(metrics).to_csv(METRICS_PATH, index=False)
    print(f"Saved outputs to: {OUTPUTS_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")

if __name__ == "__main__":
    main()










