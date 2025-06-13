import itertools
from typing import Any, Dict, List, Tuple

import numpy as np

from arclib.arc import Task
from arclib.augmenters import (
    Flip,
    PermuteExamples,
    Rotate,
    Transpose,
    inverse,
)


def get_leave_n_test_task(
    original_task: Task, n: int = 1, permute_n: int = 2, seed: int = 0
) -> Tuple[Task, List[Task]]:
    rng = np.random.RandomState(seed)
    indices = list(range(len(original_task.train_examples)))
    leave_n_indices = [set(indices) - set(comb) for comb in itertools.combinations(indices, n)]

    test_tasks = []

    train_examples = original_task.train_examples.copy()

    for comb in leave_n_indices:

        new_task = Task(
            name=original_task.name,
            train_examples=[train_examples[j] for j in comb],
            test_example=original_task.test_example,
        )

        test_tasks.append(new_task)

        for _ in range(permute_n):
            permuted_task = PermuteExamples().apply_to_task(
                new_task, to_input=True, to_output=True, rng=rng
            )
            test_tasks.append(permuted_task)

    test_tasks = list(set(test_tasks))

    augmented_tasks = []
    for augmenter in (Transpose(), Flip(0), Flip(1), Rotate(90), Rotate(180)):
        # pick a random permutation from the test tasks
        # apply the augmenter to the task
        # random leave_n task
        new_task = rng.choice(test_tasks)

        augmented_task = augmenter.apply_to_task(new_task, to_input=True, to_output=True)
        if augmented_task in test_tasks:
            continue
        inverter = str(inverse(augmenter))
        augmented_task.inverter = inverter
        augmented_task.augmenter = augmenter
        augmented_tasks.append(augmented_task)

        # permuted_task = PermuteExamples().apply_to_task(augmented_task, to_input=True, to_output=True, rng=rng)
        # permuted_task.inverter = inverter
        # test_tasks.append(permuted_task)

    # get unique
    test_tasks = augmented_tasks + test_tasks
    test_tasks = list(set(test_tasks))

    return test_tasks


def get_augmented_test_tasks(
    task: Task, include_n: List[int] = [0, 1], permute_n: int = 2
) -> List[Task]:
    augmented_tests = []
    for n in include_n:
        augmented_tests += get_leave_n_test_task(task, n=n, permute_n=permute_n)
    return augmented_tests


def format_and_filter(formatter: Any, tokenizer: Any, task: Task) -> Dict[str, Any]:
    encoded_task = formatter.encode(task)
    if encoded_task[0] is None:
        return None
    data = {"input": encoded_task[0], "output": encoded_task[1]}
    # just to get the total tokens
    # this is kind of using output
    # but normally in test we will have outputs filled by input
    messages_w_output = tokenizer.apply_chat_template(
        data["input"] + [data["output"]], tokenize=False, add_generation_prompt=True
    )
    total_tokens = len(tokenizer.encode(messages_w_output))
    del messages_w_output
    # this is the real query
    messages = tokenizer.apply_chat_template(
        data["input"], tokenize=False, add_generation_prompt=True
    )
    if hasattr(task, "inverter"):
        inverter = task.inverter
    else:
        inverter = None

    if hasattr(task, "augmenter"):
        augmenter = task.augmenter
    else:
        augmenter = None

    return {
        "text": messages,
        "inverter": inverter,
        "total_tokens": total_tokens,
        "formatter": repr(formatter),
        "input": data["input"][-1],
        "augmenter": augmenter,
        "task": task,
    }


def get_formatted_test_tasks(
    task: Task, formatters: Any, tokenizer: Any, include_n: List[int] = [0, 1], permute_n: int = 2
) -> List[Dict[str, Any]]:
    formatted_tasks = []
    augmented_tests = get_augmented_test_tasks(task, include_n=include_n, permute_n=permute_n)
    for augmented_test in augmented_tests:
        for formatter in formatters:
            formatted_task = format_and_filter(formatter, tokenizer, augmented_test)
            if formatted_task is not None:
                formatted_tasks.append(formatted_task)
    return formatted_tasks


def get_preprocessed_tasks_single(
    task: Task,
    tokenizer: Any,
    formatters: Any,
    max_tokens: int = 8192,
    include_n: List[int] = [0],
    permute_n: int = 2,
):
    queries = get_formatted_test_tasks(
        task, formatters, tokenizer, include_n=include_n, permute_n=permute_n
    )
    filtered_queries = [query for query in queries if query["total_tokens"] < max_tokens]
    if len(filtered_queries) == 0:
        queries = get_formatted_test_tasks(
            task,
            formatters,
            tokenizer,
            include_n=include_n + [include_n[-1] + 1],
            permute_n=permute_n,
        )
        filtered_queries = [query for query in queries if query["total_tokens"] < max_tokens]
    return {"valid": len(filtered_queries) > 0, "task": task, "queries": filtered_queries}


def get_preprocessed_tasks(
    tasks: List[Task],
    tokenizer: Any,
    formatters: List,
    max_tokens: int = 8192,
    include_n: List[int] = [0],
    id_to_lora_path: Dict[str, str] = {},
    permute_n: int = 2,
) -> Dict[str, Dict[str, Any]]:
    task_name_to_processed_data = {}
    print("len(id_to_lora_path)", len(id_to_lora_path))
    for task in tasks:
        task_name = task.name
        task_id = task_name.split("-")[0]
        if len(id_to_lora_path) > 0 and task_id not in id_to_lora_path:
            task_name_to_processed_data[task_name] = {"valid": False, "task": task, "queries": []}
        else:
            task_name_to_processed_data[task_name] = get_preprocessed_tasks_single(
                task,
                tokenizer,
                formatters,
                max_tokens=max_tokens,
                include_n=include_n,
                permute_n=permute_n,
            )

    return task_name_to_processed_data
