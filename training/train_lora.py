#for a small base model and a curated dataset, 1 epoch is better for lora instruction finetuning as model already knows the language, we're only nudging style, more epoch would risk overfitting 
#lora adapts quickly with moderate learning rate

# training batch size 

# gradient accumulation we cant fit large batch in memory we simulate  larger batch by accumulating gradients across multiple small batches 
# if we want effect of batch size 8, but can only fit 1 in mmeoy we will run 1 example compute gradients, run another exmpale, add gradients, repeat and then update weights at once 

# during trraining passes, we need more meomry than inference we store model weights, activatiosn needed for backprop, gradients, optimiszer states, lora adapter weights 

# during gradient accumulation we just keep rtdaitrents of all passes in memory but after each forward pass we clear the actiuvations of the previous batch

#how batch size would affect memory? more number of sequences processed at once, multiples activations, attention tensors, nad gradients, model pramters dont get ultiplied 
# sequence length higher = more activations and attention weights linearly grows

#peft is the library we will use to implement LORA where we only train a small number of parameters isntead of the whole model 
#trl is a library that includes SFTTrainer which wrap s a lot of training boilerplate for supervised finetuing, it handles packing tokenization and loss masking for caht style data

#ful finetnng requires smaller LR 
#lora rquires higher LR has fewer parameters and is less likely to destabilize the base model so can tolerate larger steps 



from email import message
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

MODEL_NAME = "LiquidAI/LFM2.5-1.2B-Instruct"
DATA_DIR = "data/eli5"
OUTPUT_DIR = "adapters/eli5-lora-v1"

MAX_SEQ_LEN = 1024
BATCH_SIZE = 1
GRAD_ACCUM = 8
EPOCHS = 1
LR = 2e-4

train_path = f"{DATA_DIR}/train.jsonl"
eval_path = f"{DATA_DIR}/eval.jsonl"
dataset = load_dataset("json", data_files={"train": train_path, "eval": eval_path})

train_ds = dataset["train"]
eval_ds = dataset["eval"]

print(f"train={len(train_ds)} eval={len(eval_ds)}")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True) 

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def to_prompt_completion(example):
    messages = example["messages"]
    #assuming last messag is assitant 
    prompt_messages = messages[:-1]
    assistant_message = messages[-1]

    prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    completion = assistant_message["content"]
    if tokenizer.eos_token:
        completion += tokenizer.eos_token

    return {"prompt": prompt, "completion": completion}

train_pc = train_ds.map(to_prompt_completion, remove_columns=train_ds.column_names)
eval_pc = eval_ds.map(to_prompt_completion, remove_columns=eval_ds.column_names)

print(train_pc[0].keys())
print(train_pc[0]["prompt"][:200])
print(train_pc[0]["completion"][:200])


print("loading base model and lora config")

#we loaad the model and then create a lora config targeting the attentino projections

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map=device,
    dtype=torch.float16
)

#w' = w + (alpha/r) * B@A
#a - (r, in_features) B - (out_features, r) 
#r is the rank so the update is low-rank
#A and B are trained
#its liek a nudge to the wieght of some layers 
# rank controls the cpacity of LORA update higher r means more trainable parameters more expreesive update lower r is chepaer buyt less felexible
#  alpha is a scaling factor for the LoRA update. 
# the effective update is scaled by alpha / r. keeping the magnitude stable. its basically how strongly the adapter is allowed to steer the base weights
# lora dropout is the dropout applied to the lora branch not the attention mask
# same behaviour but it randomly zeroes parts of the input to the lora update, regularises it so that it doesnt overfit
# target_modules are the linear layers our lora injects into,

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

from trl import SFTConfig, SFTTrainer

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=300,
    save_strategy="steps",
    save_steps=300,
    save_total_limit=2,
    optim="adamw_torch",
    max_length=MAX_SEQ_LEN,
    completion_only_loss=True,
    gradient_checkpointing=True,
    use_mps_device=torch.backends.mps.is_available(),
    report_to=[],
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_pc,
    eval_dataset=eval_pc,
    processing_class=tokenizer,
)

trainer.train(resume_from_checkpoint="adapters/eli5-lora-v1/checkpoint-3000")
trainer.save_model(OUTPUT_DIR)




