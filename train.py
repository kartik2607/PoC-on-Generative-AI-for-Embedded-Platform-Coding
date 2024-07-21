import torch 
from transformers import ( 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    pipeline, 
) 
from peft import LoraConfig, PeftModel 
from trl import SFTTrainer 
from datasets import load_dataset 
 
# The model that you want to train from the Hugging Face hub 
model_name = "meta-llama/Llama-2-7b-hf" 
 
# The instruction dataset to use 
dataset_name = "final2i.jsonl" 
 
# Fine-tuned model name 
new_model = "7b" 
 
# Output directory where the model predictions and checkpoints will be stored 
output_dir = "./results" 
 
# Number of training epochs 
num_train_epochs = 8 
 
# Load base model 
model = AutoModelForCausalLM.from_pretrained( 
    model_name, 
    # use the gpu 
    device_map= "auto" 
) 
 
# don't use the cache 
model.config.use_cache = False 
 
# Load the tokenizer from the model (llama2) 
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False) 
tokenizer.pad_token = tokenizer.eos_token 
tokenizer.padding_side = "right" 
 
# Load the dataset 
dataset = load_dataset("json", data_files="final2i.jsonl") 
 
# Load LoRA configuration 
peft_config = LoraConfig( 
    lora_alpha=64,
    lora_dropout=0.1, 
    r=16, 
    bias="none", 
    task_type="CAUSAL_LM", 
) 
 
# Set training parameters 
training_arguments = TrainingArguments( 
    output_dir=output_dir, 
    num_train_epochs=num_train_epochs,      # uses the number of epochs earlier 
    per_device_train_batch_size=4,          # 4 seems reasonable 
    gradient_accumulation_steps=2,          # 2 is fine, as we're a small batch 
    optim="paged_adamw_32bit",              # default optimizer 
    save_steps=0,                           # we're not gonna save 
    logging_steps=10,                       # same value as used by Meta 
    learning_rate=2e-4,                     # standard learning rate 
    weight_decay=0.001,                     # standard weight decay 0.001 
    fp16=False,                             # set to true for A100 
    bf16=False,                             # set to true for A100 
    max_grad_norm=0.3,                      # standard setting 
    max_steps=-1,                           # needs to be -1, otherwise overrides epochs 
    warmup_ratio=0.03,                      # standard warmup ratio 
    group_by_length=True,                   # speeds up the training 
    lr_scheduler_type="cosine",           # constant seems better than cosine 
    report_to="tensorboard" 
) 
 
# Set supervised fine-tuning parameters 
trainer = SFTTrainer( 
    model=model, 
    train_dataset=dataset, 
    peft_config=peft_config,                # use our lora peft config 
    dataset_text_field="text", 
    max_seq_length=None,                    # no max sequence length 
    tokenizer=tokenizer,                    # use the llama tokenizer 
    args=training_arguments,                # use the training arguments 
    packing=False,                          # don't need packing 
) 
 
# Train model 
trainer.train() 
 
# Save trained model 
trainer.model.save_pretrained(new_model) 
 
# Empty VRAM 
del model 
del trainer 
del tokenizer 
 
import gc 
 
gc.collect() 
 
# Reload model in Float16 and merge it with LoRA weights 
base_model = AutoModelForCausalLM.from_pretrained( 
    model_name, 
    low_cpu_mem_usage=True, 
    return_dict=True, 
    torch_dtype=torch.float16, 
    device_map="auto" 
) 
model = PeftModel.from_pretrained(base_model, new_model) 
model = model.merge_and_unload() 
 
# Reload tokenizer to save it 
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True) 
tokenizer.pad_token = tokenizer.eos_token 
tokenizer.padding_side = "right" 
 
model.save_pretrained("7b1") 
tokenizer.save_pretrained("7b1") 