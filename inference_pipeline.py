# Fine-tuned model name
model_name = "./7b1"

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
    logging,
)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # use the gpu
    device_map= "auto"
)


# Load the tokenizer from the model (llama2)
tokenizer = AutoTokenizer.from_pretrained(model_name)#, trust_remote_code=True, use_fast=False)

model.use_cache = True

model.eval()
logging.set_verbosity(logging.CRITICAL)
print("hello")

def format_instruction(sample):
	return f"""### Instruction:
You are a coding assistant that will write a Solution to resolve the following Task:

### Task:
{sample}

### Solution:
""".strip()

while(True):
    prompt = input("Enter Your Prompt: ")
    
    if(prompt=="exit"):
        break

    final = format_instruction(prompt)
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, torch_dtype = torch.float16, repetition_penalty = 1.1)
    result = pipe(f"{final}", do_sample = True, top_p = 0.5, temperature = 0.5, top_k = 10 ,num_return_sequences = 1, eos_token_id = tokenizer.eos_token_id,max_length = 2048)
    for output in result:
        print(output['generated_text'])

