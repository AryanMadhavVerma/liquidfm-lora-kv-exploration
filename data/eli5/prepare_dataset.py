import json
from datasets import load_dataset

from sklearn.model_selection import train_test_split

MODEL_NAME = "LiquidAI/LFM2.5-1.2B-Instruct"
DATASET_NAME = "sentence-transformers/eli5"

MIN_TOKENS = 80
MAX_TOKENS = 300
TARGET_SAMPLES = 50_000

SYSTEM_PROMPT = """You are an expert at explaining complex topics in a simple, easy-to-understand way. You will explain as if I'm a 5-10 year old"""


def main():
    print("loading dataset")
    dataset = load_dataset(
        DATASET_NAME, split="train"
    )

    processed = []
    skipped = 0
    
    for row in dataset:
        question = row.get("question")
        answer = row.get("answer")

        if not question or not answer:
            skipped +=1
            continue
        
        #we vaguely assume token and word length to be the same here. also we dont care about the context_length limit which is a combination of system prompt question and answer. This is because context_length of model is huge enough
        token_length = len(answer.split())
        #we dont want the answee tobe too short or long
        if token_length < MIN_TOKENS or token_length > MAX_TOKENS:
            skipped +=1
            continue

        entry = {
            "messages": [
                {
                    "role": "system", "content": SYSTEM_PROMPT
                },
                {
                    "role": "user", "content": question
                },
                {
                    "role": "assistant", "content": answer
                }
            ]
        }

        processed.append(entry)

        if len(processed) >= TARGET_SAMPLES:
            break

    print(f"processed {len(processed)} samples, skipped {skipped} samples")

    output_dir = "data/eli5"
    processed_path = f"{output_dir}/processed.jsonl"

    with open(processed_path, "w") as f:
        for entry in processed:
            f.write(json.dumps(entry) + "\n")

    train_data, eval_data = train_test_split(processed, test_size=0.1, random_state=42)
    train_path = f"{output_dir}/train.jsonl"
    eval_path = f"{output_dir}/eval.jsonl"

    with open(train_path, "w") as f:
        for entry in train_data:
            f.write(json.dumps(entry) + "\n")   

    with open(eval_path, "w") as f:
        for entry in eval_data:  
            f.write(json.dumps(entry) + "\n")

    frozen_prompts = [item["messages"][1]["content"] for item in eval_data[:50]]
    prompts_path = f"{output_dir}/eval_prompts.json"

    with open(prompts_path, "w") as f:
        json.dump(frozen_prompts,f, indent=2)

    print(f"Saved processed dataset to: {processed_path}")
    print(f"Saved train split to: {train_path}")
    print(f"Saved eval split to: {eval_path}")
    print(f"Saved frozen eval prompts to: {prompts_path}")



if __name__=="__main__":
    main()
