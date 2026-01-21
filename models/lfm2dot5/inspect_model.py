import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "LiquidAI/LFM2.5-1.2B-Instruct"

print(f"loading model and tokenizer for: {model_name}")

try :
    #lets change from default float 32 to float 16, a 1.2b parameter model would take around 2gb instead of 4 with float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="mps" if torch.backends.mps.is_available() else "cpu"
    )
    print("model and tokenizer loaded successfully")
except Exception as e:
    print(f"error loading model: {e}")

print("verifying chat template")

print("analysing the chat template")
if tokenizer.chat_template:
    conversation = [
        {
            "role":"system",
            "content": "you are a helpful assistant"
        },
        {"role": "user", "content": "Hello, how are you?"}
    ]

    try:
        prompt = tokenizer.apply_chat_template(conversation,tokenize=False,add_generation_prompt=True)
        #tokenizer true would just convert into token ids to be fed to the model
        #this is the <im> token of the model and the new line before <im>
        print("applied chat template")
        print(f"this is the prompt: \n{prompt}")
    except Exception as e:
        print(f"error applying chat template {e}")

else:
    print("no chat template found")

#basic text generation without any template 

prompt = "the best thing about learning to code is: "
print(f"input prompt {prompt}")

try:
    #return tensors prt basically helps us aovid the part where we get a list of tokens which we then unsqueeze to add a batch dimension etc
    inputs = tokenizer(prompt,return_tensors="pt").to(model.device)
    #inputs is a map / key value pairs of input ids, attention mask (eventually during atateion whichtokens to nullify) we pass this kwargs into the generate funciton 
    outputs = model.generate(**inputs, max_new_tokens=50)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"generated text is: \n{generated_text}")
except Exception as e:
    print(f"erro rcame during ngeration: {e}")


#freezing base model weights 
#when we finetune a model using a technique like LoRA, we dont wnt to change the origiunal pretrained weights ofd the model. We want to train the new adapterlayers we add, to prevent the original layers from being updated
#we freeze them 
#bascially requires_grade=False would disale grdient flow 

print("freezing base model weights")
trainable_params_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable parameters before frezing = {trainable_params_before / 1e6:.2f}M")

for param in model.parameters():
    param.requires_grad = False
    
trainable_params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\n--- All base model weights have been frozen. ---")
print(f"Trainable parameters after freezing: {trainable_params_after}")





