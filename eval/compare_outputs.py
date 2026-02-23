import json

BASELINE_OUTPUTS_PATH = "eval/baseline_outputs.json"
LORA_OUTPUTS_PATH = "eval/lora_outputs.json"
OUTPUT_MD = "eval/eli5_comparison.md"
MAX_EXAMPLES = 10


def load_outputs(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    baseline = load_outputs(BASELINE_OUTPUTS_PATH)
    lora = load_outputs(LORA_OUTPUTS_PATH)

    lora_by_prompt = {row["prompt"]: row for row in lora}

    examples = []
    for row in baseline:
        prompt = row["prompt"]
        if prompt in lora_by_prompt:
            examples.append((row, lora_by_prompt[prompt]))
        if len(examples) >= MAX_EXAMPLES:
            break

    lines = []
    lines.append("# ELI5 Baseline vs LoRA (Sampled)\n")

    for i, (base, lora_row) in enumerate(examples, start=1):
        lines.append(f"## Example {i}")
        lines.append("Prompt:")
        lines.append(f"```\n{base['prompt']}\n```")

        lines.append(
            f"Baseline (tokens={base['generated_tokens']}, latency_s={base['latency_s']:.3f}):"
        )
        lines.append(f"```\n{base['generated_answer']}\n```")

        lines.append(
            f"LoRA (tokens={lora_row['generated_tokens']}, latency_s={lora_row['latency_s']:.3f}):"
        )
        lines.append(f"```\n{lora_row['generated_answer']}\n```")

    with open(OUTPUT_MD, "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote {len(examples)} examples to {OUTPUT_MD}")


if __name__ == "__main__":
    main()
