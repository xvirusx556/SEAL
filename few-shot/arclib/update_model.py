import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from typing import Dict, Any, Callable, List, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from tqdm import tqdm
import numpy as np
import torch
import datetime
import uuid

from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from torch.utils.data import DataLoader

from .arc import Example, Grid, Task
from .representers import WordGridRepresenter

import itertools
from typing import List

import numpy as np

from .arc import Task
from .augmenters import (
    Augmenter,
    Chain,
    Concat,
    Flip,
    IdentityAugmenter,
    IncreaseHeight,
    IncreaseResolution,
    IncreaseWidth,
    PermuteColors,
    PermuteExamples,
    RandomTranslateXY,
    Reflect,
    Repeat,
    Rotate,
    Transpose,
)
from .messagers import MessageRepresenter


class TTT:
    def __init__(self, model_name: str, state_dict_path: Optional[str] = None, lora_config: Optional[LoraConfig] = None):
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Make sure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        if state_dict_path is not None:
            state_dict = torch.load(state_dict_path)
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded state dict from {state_dict_path}")
        
        self.model = get_peft_model(model, lora_config) 

        # Store initial LoRA A parameter values
        self.initial_lora_A = {}
        for name, param in self.model.named_parameters():
            if "lora_A" in name:
                # Store a deep copy of the initial parameters
                self.initial_lora_A[name] = param.data.clone().detach()
    
    def update_model(
        self, 
        task_text_list: List[str], 
        output_dir: str,
        batch_size: int,
        gradient_accumulation_steps: int,
        learning_rate: float,
        num_train_epochs: int,
        lr_scheduler_type: str,
        loss_on_all_tokens: bool,
    ) -> str:
        """
        Update the model using the provided task by:
        1. Reset LoRA weights
        2. Create training data from the task
        3. Train the model using SFT
        4. Save the LoRA weights
        
        Args:
            task: ARC task to use for training
            task_processor: Function to process tasks (like get_preprocessed_tasks_single)
            representer: Representer for formatting tasks (like GPTTextMessageRepresenterForBarc)
            output_dir: Directory to save model
            batch_size: Batch size for training
            gradient_accumulation_steps: Number of gradient accumulation steps
            learning_rate: Learning rate for training
            num_train_epochs: Number of epochs for training
            lr_scheduler_type: Learning rate scheduler type
            
        Returns:
            Path to the saved model
        """
        # Reset the LoRA weights
        self.reset_lora()
        
        # Create training data
        training_data = self._tokenize_and_process(task_text_list, loss_on_all_tokens)
        
        # clear cache
        torch.cuda.empty_cache()

        # Train the model
        self._train_model(
            training_data,
            output_dir=output_dir,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            lr_scheduler_type=lr_scheduler_type,
        )
        
        # Save model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")
        return output_dir
        
    def reset_lora(self):
        """Reset all LoRA B parameters to zero and LoRA A parameters to their initial values."""
        for name, param in self.model.named_parameters():
            if "lora_B" in name:
                param.data.fill_(0.0)
            elif "lora_A" in name and name in self.initial_lora_A:
                # Restore the original values of LoRA A parameters
                param.data.copy_(self.initial_lora_A[name])
        
    def _tokenize_and_process(self, text_list: List[str], loss_on_all_tokens: bool):
        """
        Tokenize a list of texts and set up labels for instruction fine-tuning.
        Specifically looks for the second-to-last occurrence of the assistant header token sequence.
        Processes all texts in parallel for efficiency.
        
        Args:
            text_list: List of text strings to process
            
        Returns:
            Dictionary with tokenized inputs and labels
        """
        # Tokenize all texts in a batch
        outputs = self.tokenizer(
            text_list,
            truncation=True,
            max_length=8192,
            padding="longest",
            return_tensors="pt"
        )
        input_ids = outputs["input_ids"]
        
        # Process all samples in parallel
        batch_size = input_ids.shape[0]
        labels = input_ids.clone()
        
        # Find special sequences and set labels for all samples in parallel
        for i in range(batch_size):
            if loss_on_all_tokens:
                continue

            sample_input_ids = input_ids[i].tolist()
            
            # Find all occurrences of the special sequence
            # This is "<|start_header_id|>assistant<|end_header_id|>"
            special_indices = []
            for j in range(len(sample_input_ids) - 1):
                if sample_input_ids[j] == 128007 and sample_input_ids[j + 1] == 271:
                    special_indices.append(j + 1)  # include the 271 token in the conditioning
            
            # If we found multiple matches, use the second-to-last one
            if len(special_indices) == 4:
                special_index = special_indices[-2]
            else:
                print(f"Warning: Special sequence not found in sample {i}, using fallback strategy")
                special_index = int(len(sample_input_ids) * 0.8)
                assert False
            
            # Create labels: we want the model to predict tokens after the special sequence
            for j in range(special_index + 1):
                labels[i, j] = -100
        
        outputs["labels"] = labels
        return outputs
    
    def _train_model(
        self,
        training_data: Dict[str, Any],
        output_dir: str,
        batch_size: int,
        gradient_accumulation_steps: int,
        learning_rate: float,
        num_train_epochs: int,
        lr_scheduler_type: str,
    ):
        """
        Train the model using the provided text examples.
        
        Args:
            text_list: List of formatted training examples
            output_dir: Directory to save checkpoints
            batch_size: Batch size for training
            gradient_accumulation_steps: Number of gradient accumulation steps
            learning_rate: Learning rate for training
            num_train_epochs: Number of epochs for training
        
        Returns:
            Trained model and tokenizer
        """
        print(f"Training on {training_data['input_ids'].shape[0]} examples for {num_train_epochs} epochs, lr: {learning_rate}")
        # Prepare dataset
        ds = Dataset.from_dict(training_data)
        

        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            lr_scheduler_type=lr_scheduler_type,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            bf16=True,  # Use bfloat16 precision
            remove_unused_columns=False,
            optim="adamw_torch",
            warmup_steps=11,
          )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=ds,
            
        )
        
        # Train the model
        print("Starting training...")
        trainer.train()
        print("Training complete.")
        
        return self.model, self.tokenizer
