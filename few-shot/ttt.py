import os
import re
import json
import glob
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Optional
from datetime import datetime
from collections import Counter
from transformers import AutoTokenizer

from peft import LoraConfig
import arclib
from arclib.arc import Example, Task
from arclib.representers import (
    CompositeRepresenter,
    ConnectedComponentRepresenter,
    DelimitedGridRepresenter,
    DiffExampleRepresenter,
    GridRepresenter,
    ImageTaskRepresenter,
    PythonListGridRepresenter,
    TaskRepresenter,
    TextTaskRepresenter,
    TextExampleRepresenter,
    WordGridRepresenter,
)
from arclib.messagers import GPTTextMessageRepresenterForBarc, GPTTextMessageRepresenterV2
from arclib.update_model import TTT
from inference.preprocess import get_preprocessed_tasks_single

import itertools
from typing import List

import numpy as np

from arclib.arc import Task
from arclib.augmenters import (
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
from arclib.messagers import MessageRepresenter


def mode_array(array_list):
    """Return the most common array from a list of arrays."""
    tuple_shape_list = [(tuple(arr.flatten()), arr.shape) for arr in array_list]
    most_common_tuple_shape, _ = Counter(tuple_shape_list).most_common(1)[0]
    most_common_tuple, original_shape = most_common_tuple_shape
    mode_arr = np.array(most_common_tuple).reshape(original_shape)
    return mode_arr


def read_tasks_from_folder(task_folder: str, test: bool = False) -> List[Task]:
    """Read tasks from a folder of JSON files."""
    all_tasks = []
    for file in glob.glob(f"{task_folder}/*.json"):
        basename = os.path.basename(file)
        idx = basename.replace(".json", "")
        tasks = read_tasks_from_file(file, test=test)
        for i, task in enumerate(tasks):
            task.name = idx + "-" + str(i)
        all_tasks += tasks
    return all_tasks


def read_tasks_from_single_file(
    challenge_file: str, test: bool = False, solution_file: Optional[str] = None
) -> List[Task]:
    """Read tasks from a single JSON file with optional solutions."""
    with open(challenge_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if solution_file is not None:
        test = False
        with open(solution_file, "r", encoding="utf-8") as handle:
            solutions = json.load(handle)
            for key, value in solutions.items():
                for idx, solution in enumerate(value):
                    data[key]["test"][idx]["output"] = solution

    all_tasks = []
    for task_name, subtasks in data.items():
        parsed_tasks = Task.read_tasks_from_dict(subtasks, test=test)
        for i, task in enumerate(parsed_tasks):
            task.name = task_name + "-" + str(i)
            all_tasks.append(task)

    return all_tasks


def read_tasks_from_file(task_file: str, test: bool = False) -> List[Task]:
    """Read tasks from a JSON file."""
    with open(task_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return Task.read_tasks_from_dict(data, test=test)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


def get_augmenters(
    include_basic: bool = True,
    include_size: bool = True,
    include_chain: bool = True,
    include_repeat: bool = True,
    include_concat: bool = False,
) -> List[Augmenter]:
    basic_augmenters_to_apply = (
        [
            Rotate(90),
            Rotate(270),
            Rotate(180),
            Flip(0),
            Flip(1),
            Reflect(0, reverse=True),
            Reflect(1, reverse=True),
            Reflect(0, reverse=False),
            Reflect(1, reverse=False),
            RandomTranslateXY(),
            Transpose(),
        ]
        if include_basic
        else []
    )

    size_augmenters_to_apply = (
        [
            IncreaseResolution(2),
            IncreaseHeight(2),
            IncreaseWidth(2),
        ]
        if include_size
        else []
    )

    concat_augmenters_to_apply = (
        [
            Concat((IdentityAugmenter(), Rotate(180)), axis=0),
            Concat((IdentityAugmenter(), Rotate(180)), axis=1),
        ]
        if include_concat
        else []
    )

    chain_augmenters_to_apply = (
        [
            Chain([Rotate(90), IncreaseResolution(2)]),
            Chain([Rotate(270), IncreaseResolution(2)]),
            Chain([Rotate(180), IncreaseResolution(2)]),
            Chain([Flip(0), IncreaseResolution(2)]),
            Chain([Flip(1), IncreaseResolution(2)]),
            Chain([Transpose(), IncreaseResolution(2)]),
        ]
        if include_chain
        else []
    )

    repeat_augmenters_to_apply = (
        [
            Repeat(0, 2),
            Repeat(1, 2),
            Repeat(2, 2),
        ]
        if include_repeat
        else []
    )

    augmenters_to_apply = (
        basic_augmenters_to_apply
        + size_augmenters_to_apply
        + concat_augmenters_to_apply
        + chain_augmenters_to_apply
        + repeat_augmenters_to_apply
    )

    print("Augmenters to apply: ", augmenters_to_apply, "len: ", len(augmenters_to_apply))
    return augmenters_to_apply

def _tokenize_and_process(text: str, tokenizer):
    """
    Tokenize text and set up labels for instruction fine-tuning.
    Specifically looks for assistant header token sequence to identify the response part.
    """
    # Tokenize the entire string
    outputs = tokenizer(
        text,
        truncation=True,
    )
    input_ids = outputs["input_ids"]
    
    # Find the special sequence: token 128007 followed immediately by 271
    # This is "<|start_header_id|>assistant<|end_header_id|>"
    special_indexes = []
    for i in range(len(input_ids) - 1):
        if input_ids[i] == 128007 and input_ids[i + 1] == 271:
            special_indexes.append(i + 1)  # include the 271 token in the conditioning
            
    
    special_index = special_indexes[2] if len(special_indexes) == 4 else None
    # If special sequence not found, handle the error
    if special_index is None:
        print("Warning: Special sequence not found, using fallback strategy")
        # Fallback: set labels for the last 20% of tokens
        assert False
    
    # Create labels: we want the model to predict tokens after the special sequence
    labels = input_ids.copy()
    for j in range(special_index + 1):
        labels[j] = -100
    
    outputs["labels"] = labels
    return outputs

def format_and_filter(formatter, tokenizer, task, train_on_input: False):
    task = formatter.encode(task)
    data = {"input": task[0], "output": task[1]}
    task_text = tokenizer.apply_chat_template(task[0] + [task[1]], tokenize=False, add_generation_prompt=True)
    #messages = arc_to_messages(data, train_on_input=False)
    outputs = _tokenize_and_process(task_text, tokenizer)
    data["total_tokens"] = len(outputs["input_ids"])
    data["full_text"] = task_text
    return data


def get_test_time_train_data(
    original_task: Task, augmenters: List[Augmenter], n: int = 1, permute_n: int = 1, seed: int = 0
) -> List[Task]:
    rng = np.random.RandomState(seed)
    train_examples = original_task.train_examples.copy()
    initial_tasks = []
    N = len(train_examples)
    for i in range(len(train_examples)):
        examples = train_examples.copy()
        indices = set(range(N)) - {i}
        # we already remove i, so we need to remove n-1 more
        combs = list(itertools.combinations(indices, n - 1))
        combs = [indices - set(comb) for comb in combs]
        for comb in combs:
            initial_tasks.append(
                Task(name="", train_examples=[examples[j] for j in comb], test_example=examples[i])
            )

    augmented_tasks = []
    for augmenter in augmenters:
        for task in initial_tasks:
            task = augmenter.apply_to_task(task, to_input=True, to_output=True, rng=rng)
            # some augmentations increase shapes
            if not (task.max_height() <= 30 and task.max_width() <= 30):
                continue
            augmented_tasks.append(task)

    augmented_tasks = list(set(augmented_tasks + initial_tasks))

    color_and_permute_augmented_tasks = []

    for _ in range(permute_n):
        for task in augmented_tasks:
            if len(augmenters) != 0:
                new_task = PermuteColors().apply_to_task(task, to_input=True, to_output=True, rng=rng)
            else:
                new_task = task
            new_task = PermuteExamples().apply_to_task(
                new_task, rng=rng, to_input=True, to_output=True
            )
            color_and_permute_augmented_tasks.append(new_task)

    augmented_tasks = color_and_permute_augmented_tasks + augmented_tasks

    augmented_tasks = list(set(augmented_tasks))

    return augmented_tasks


def get_formatted_data(
    task: Task,
    augmenters: List[Augmenter],
    formatter: MessageRepresenter,
    tokenizer,
    leave_n: int = 1,
    permute_n: int = 1,
    seed: int = 0,
    max_tokens: int = 8192,
):

    train_data = get_test_time_train_data(
        task, augmenters, n=leave_n, permute_n=permute_n, seed=seed
    )

    formatted_data = []
    for task in train_data:
        formatted = format_and_filter(formatter, tokenizer, task, train_on_input=False)
        if formatted["total_tokens"] < max_tokens:
            formatted_data.append(formatted)

    return formatted_data


def process_task(
    task: Task,
    augmenters: List[Augmenter],
    formatter: MessageRepresenter,
    tokenizer,
    leave_n: List[int],
    permute_n: int = 1,
    Nmax: int = 250,
    seed: int = 0,
):
    rng = np.random.RandomState(seed)
    
    train = []
    # Generate training data for each n in leave_n
    for n in leave_n:
        leave_n_train_data = get_formatted_data(
            task, augmenters, formatter, tokenizer, leave_n=n, permute_n=permute_n, seed=seed
        )
        train.extend(leave_n_train_data)

    # Shuffle and limit the total number of examples if needed
    if len(train) > Nmax:
        rng.shuffle(train)
        train = train[:Nmax]

    return train


def main():
    # lora config
    lora_config = LoraConfig(
        r=128, 
        lora_alpha=16,
        lora_dropout=0.00,
        bias="none",
        task_type="CAUSAL_LM", 
        target_modules=["q_proj", "v_proj", "gate_proj", "down_proj", "up_proj"]
    )

    # training config
    batch_size = 2
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    num_train_epochs = 2
    lr_scheduler_type = "cosine"

    standard_formatter = TextTaskRepresenter(
        example_representer=TextExampleRepresenter(
            io_sep=" -> ",
            input_header="",
            output_header="",
            output_footer="#",
            grid_representer=PythonListGridRepresenter(),
        )
    )

    representer = GPTTextMessageRepresenterV2(task_representer=standard_formatter)


    #grid_converter = WordGridRepresenter()

    # Load tasks
    tasks = read_tasks_from_single_file(
        challenge_file="data/arc-agi_training_challenges.json",
        solution_file="data/arc-agi_training_solutions.json"
    )

    # Setup tokenizer and model
    #model_name = "barc0/Llama-3.1-ARC-Potpourri-Transduction-8B"
    #model_name = "/data/pulkitag/models/jyop/code/arc/debug/1B-17k"
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # setup ttt 
    ttt = TTT(
        model_name=model_name,
        lora_config=lora_config
    )

    # Evaluation settings
    n_tasks = 350
    # assert len(tasks) >= n_tasks
    
    correct = 0
    task_results = []

    self_edits = [{"augmenters": [True, True, True, True], "leave_n": [1, 2], "loss_on_everything": False}]


    #Generate loras for each task
    for i in range(n_tasks):
        task = tasks[i]
        
        self_edit = self_edits[0] #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! TODO 
        augmenters_to_apply = get_augmenters(
            include_basic=self_edit["augmenters"][0], include_size=self_edit["augmenters"][1], include_chain=self_edit["augmenters"][2], include_repeat=self_edit["augmenters"][3]
        )

        train_data = process_task(
            task=task,
            augmenters=augmenters_to_apply,
            formatter=representer,
            tokenizer=tokenizer,
            leave_n=self_edit["leave_n"],
            permute_n=1,
            Nmax=250,
            seed=0
        )

        if len(train_data) == 0:
            continue

        task_text_list = [data["full_text"] for data in train_data]

        if task.name.endswith("-0"):
            task.name = task.name[:-2]
        
        if task.name.endswith("-1"):
            continue

        adapter_path = ttt.update_model(
            task_text_list=task_text_list,
            output_dir=f"loras/lora-1B-train-350/{task.name}",
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            lr_scheduler_type=lr_scheduler_type,
            loss_on_all_tokens=self_edit["loss_on_everything"]
        )
  

if __name__ == "__main__":
    main()
    
   
 
    
