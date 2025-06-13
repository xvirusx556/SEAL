import json
import ipdb
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import numpy as np
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model with LoRA fine-tuning')
    parser.add_argument('--configs_and_indices', type=str, required=True, help='Path to configs and indices JSON file')
    parser.add_argument('--results', type=str, required=True, help='Path to results JSON file')
    parser.add_argument('--model_name', type=str, required=True, help='Name or path of the base model')
    parser.add_argument('--lora_rank', type=int, required=True, help='Rank for LoRA fine-tuning')
    parser.add_argument('--lora_alpha', type=int, required=True, help='Alpha parameter for LoRA')
    parser.add_argument('--num_train_epochs', type=int, required=True, help='Number of training epochs')
    parser.add_argument('--per_device_train_batch_size', type=int, required=True, help='Batch size per device for training')
    parser.add_argument('--gradient_accumulation_steps', type=int, required=True, help='Number of steps to accumulate gradients')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate for training')
    return parser.parse_args()

# Parse arguments
args = parse_args()

# Load the JSON files
with open(args.configs_and_indices, 'r') as f:
    configs_and_indices = json.load(f)

with open(args.results, 'r') as f:
    results = json.load(f)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer.pad_token = tokenizer.eos_token

# Configure LoRA
lora_config = LoraConfig(
    r=args.lora_rank, 
    lora_alpha=args.lora_alpha,
    lora_dropout=0.00,
    bias="none",
    task_type="CAUSAL_LM", 
    target_modules=["q_proj", "v_proj", "gate_proj", "down_proj", "up_proj"]
)

# Prepare model for LoRA training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Extract prompts and responses for correct cases
correct_prompts_responses = []

for sample_idx in range(len(results)):
    for task_idx in range(len(results[str(sample_idx)]['task_id'])):

        if results[str(sample_idx)]['correct'][str(task_idx)]:
            task_id = results[str(sample_idx)]['task_id'][str(task_idx)]
            prompt = configs_and_indices[task_id][str(sample_idx)]['prompt']
            response = configs_and_indices[task_id][str(sample_idx)]['response']
            correct_prompts_responses.append({
                'prompt': prompt,
                'response': response
            })

print(f"Found {len(correct_prompts_responses)} correct cases") 

training_data = []
for i in range(len(correct_prompts_responses)):
    training_data.append(correct_prompts_responses[i]['prompt'] + correct_prompts_responses[i]['response'] + '<|eot_id|>')

def create_dataset(training_data):
    # Create dataset with text field
    dataset = Dataset.from_dict({"text": training_data})
    
    def tokenize_function(examples):
        # Tokenize the full text
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=100000,
            padding="longest",
            return_tensors="pt"
        )
        
        # Create labels (same as input_ids initially)
        labels = tokenized["input_ids"].clone()
        
        # For each example, find where the response starts
        for i, input_ids in enumerate(tokenized["input_ids"]):
            # Find the last position of [128007, 271] sequence
            last_pos = -1
            for j in range(len(input_ids) - 1):
                if input_ids[j] == 128007 and input_ids[j + 1] == 271:
                    last_pos = j
                    print(f"Found [128007, 271] at position {j}")
            
            if last_pos != -1:
                # Set labels to -100 for all tokens before and including the last [128007, 271]
                labels[i, :last_pos+2] = -100
            
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels
        }
    
    # Apply tokenization and create labels
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

# Create the dataset
train_dataset = create_dataset(training_data)

# Training arguments
training_args = TrainingArguments(
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    optim="adamw_torch",
    lr_scheduler_type="cosine",
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Start training
trainer.train()

# Merge LoRA weights into the base model
model = model.merge_and_unload()

# First save to a temporary directory
temp_dir = "./temp_model"
model.save_pretrained(
    temp_dir,
    safe_serialization=True,
    max_shard_size="2GB"
)
tokenizer.save_pretrained(temp_dir)

# Load back as a regular model
print("Loading merged model to verify structure...")
clean_model = AutoModelForCausalLM.from_pretrained(
    temp_dir,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Final save to output directory
output_dir = "./RL_trained_model"
clean_model.save_pretrained(
    output_dir,
    safe_serialization=True,
    max_shard_size="2GB"
)
tokenizer.save_pretrained(output_dir)

# Clean up temporary directory
shutil.rmtree(temp_dir)

print(f"Model and tokenizer saved to {output_dir}")

# Verify the final saved model
print("Verifying final saved model...")
test_model = AutoModelForCausalLM.from_pretrained(
    output_dir,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("Model verification successful!")


