"""
Evaluation script for ARC AGI Challenge

evaluate: function to evaluate a submission file
compare: function to compare two submission files
"""
import argparse
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
from transformers.models.tapas.tokenization_tapas import pd

from .arc import make_submission, read_tasks_from_single_file, to_list, to_tuple
from .voting import vote

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize

def plot_the_mistake(task, preds, save_folder, postfix=""):
    cmap = ListedColormap([
        '#000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ])
    norm = Normalize(vmin=0, vmax=9)
    args = {'cmap': cmap, 'norm': norm}
    height = 2
    width = len(task.train_examples) + 3
    figure_size = (width * 3, height * 3)
    figure, axes = plt.subplots(height, width, figsize=figure_size)

    examples  = task.train_examples + [task.test_example]
    for column, example in enumerate(examples):
        axes[0, column].imshow(example.input, **args)
        axes[1, column].imshow(example.output, **args)
        axes[0, column].axis('off')
        axes[1, column].axis('off')

    for pred in preds.values():
        column += 1
        axes[1, column].imshow(pred, **args)
        axes[1, column].axis('off')

    figure.suptitle(task.name, fontsize=20)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # save fig
    plt.savefig(f"{save_folder}/{task.name}{postfix}.png")
    plt.close()


def accuracy(task, attempts):
    output = to_tuple(task.test_example.output)
    attempts = [to_tuple(attempt) for attempt in attempts.values()]
    return output in attempts


def hamming_distance(task, attempt):
    output = task.test_example.output
    attempt = np.array(attempt)
    output_size = output.shape[0] * output.shape[1]
    # if shapes are different find the minimum shape
    min_width = min(output.shape[1], attempt.shape[1])
    min_height = min(output.shape[0], attempt.shape[0])
    output = output[:min_height, :min_width]
    attempt = attempt[:min_height, :min_width]
    # remaining number of elementx
    additional_elements = output_size - (min_height * min_width)
    return (int(np.sum(output != attempt)) + additional_elements) / output_size


def evaluate(
    data_file: str, solution_file: str, submission_file: str, task_info_file: Optional[str] = None, mistakes: bool = False, subpoints=False,
):
    tasks = read_tasks_from_single_file(data_file, solution_file=solution_file, test=False)

    eval_folder = os.path.dirname(submission_file)

    if mistakes:
        save_folder = os.path.join(eval_folder, "mistakes")
        os.makedirs(save_folder, exist_ok=True)

    with open(submission_file, "r") as handle:
        submission = json.load(handle)

    if task_info_file is not None:
        task_info = pd.read_csv(task_info_file)
    else:
        task_info = pd.DataFrame(submission.keys(), columns=["task_id"])

    task_info["correct"] = False

    task_id_to_tasks = {task.name.split("-")[0]: [] for task in tasks}
    for task in tasks:
        task_id_to_tasks[task.name.split("-")[0]].append(task)

    print("Attempted tasks: ", len(submission))

    corrects = 0
    competition_corrects = 0
    total = 0
    hamming = 0

    for task_name in submission.keys():
        subtask_predictions = submission[task_name]
        all_submitted = len(subtask_predictions) == len(task_id_to_tasks[task_name])
        all_true = True
        subcorrect = 0
        subtotal = len(task_id_to_tasks[task_name])
        for i, subtask_prediction in enumerate(subtask_predictions):
            task = [task for task in tasks if task.name == f"{task_name}-{i}"][0]
            label = accuracy(task, subtask_prediction)
            if mistakes and not label:
                plot_the_mistake(task, subtask_prediction, save_folder)
            corrects += label
            all_true = all_true and label
            subcorrect += label
            # hamming += (
            #     hamming_distance(task, subtask_prediction["attempt_1"])
            #     + hamming_distance(task, subtask_prediction["attempt_2"])
            # ) / 2
            total += 1
        if subpoints:
            competition_corrects += (subcorrect / subtotal)
        else:
            competition_corrects += all_true and all_submitted
        #
        # edit task_info
        task_info.loc[task_info["task_id"] == task_name, "correct"] = all_true and all_submitted
        print(task_name, all_true and all_submitted)

    print(f"Per Prediction Accuracy: {corrects} / {total} = {corrects / total}")
    # print(f"Per Prediction Hamming Distance: {hamming} / {total} = {hamming / total}")
    print(
        f"Competition Accuracy: {competition_corrects} / {len(task_id_to_tasks)} = {competition_corrects / len(task_id_to_tasks)}"
    )

    if "level" in task_info.columns:
        print("Per Level Solved Tasks")
        print(task_info.groupby("level")["correct"].sum())
        # save task_info to submission folder
        task_info.to_csv(
            os.path.join(os.path.dirname(submission_file), "task_info.csv"), index=False
        )

    return task_info


def compare(data_file: str, solution_file: str, submission_1_file: str, submission_2_file: str, plot_differents: bool = False, diff_folder: str = "diffs"):
    tasks = read_tasks_from_single_file(data_file, solution_file=solution_file, test=False)

    with open(submission_1_file, "r") as handle:
        submission_1 = json.load(handle)

    with open(submission_2_file, "r") as handle:
        submission_2 = json.load(handle)

    if plot_differents:
        submission_1_correct_path = os.path.join(diff_folder, "submission_1_correct")
        submission_2_correct_path = os.path.join(diff_folder, "submission_2_correct")
        os.makedirs(submission_1_correct_path, exist_ok=True)
        os.makedirs(submission_2_correct_path, exist_ok=True)

    correct_in_1_wrong_in_2 = 0
    correct_in_2_wrong_in_1 = 0

    for task_name in submission_1.keys():
        subtask_predictions_1 = submission_1[task_name]
        subtask_predictions_2 = submission_2.get(task_name, [])

        for i, subtask_prediction_1 in enumerate(subtask_predictions_1):
            task = [task for task in tasks if task.name == f"{task_name}-{i}"][0]
            label_1 = accuracy(task, subtask_prediction_1)
            label_2 = accuracy(task, subtask_predictions_2[i]) if i < len(subtask_predictions_2) else False

            if label_1 and not label_2:
                correct_in_1_wrong_in_2 += 1
                if plot_differents:
                    plot_the_mistake(task, subtask_prediction_1, submission_1_correct_path, postfix="CS1")
                    if i < len(subtask_predictions_2):
                        plot_the_mistake(task, subtask_predictions_2[i], submission_1_correct_path, postfix="WS2")
            elif label_2 and not label_1:
                correct_in_2_wrong_in_1 += 1
                if plot_differents:
                    plot_the_mistake(task, subtask_predictions_2[i], submission_2_correct_path)
                    plot_the_mistake(task, subtask_prediction_1, submission_2_correct_path, postfix="WS1")


    print(f"Tasks correct in submission_1 and wrong in submission_2: {correct_in_1_wrong_in_2}")
    print(f"Tasks correct in submission_2 and wrong in submission_1: {correct_in_2_wrong_in_1}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--data_file",
        type=str,
        default="/kaggle/input/arc-prize-2024/arc-agi_evaluation_challenges.json",
        help="Data file path to evaluate",
    )
    parser.add_argument(
        "--solution_file",
        type=str,
        default="/kaggle/input/arc-prize-2024/arc-agi_evaluation_solutions.json",
        help="Solution file path to evaluate",
    )
    parser.add_argument(
        "--submission_file",
        type=str,
        default=None,
        help="Submission file path to evaluate",
    )
    parser.add_argument(
        "--all_predictions_file",
        type=str,
        default=None,
        help="All outputs before voting",
    )
    parser.add_argument(
        "--task_info_file", type=str, default="tasks_info.csv", help="Task info file"
    )
    parser.add_argument(
        "--plot_mistakes", action="store_true", help="Plot mistakes for each task"
    )

    parser.add_argument(
        "--compare_submission_file", type=str, default=None, help="Compare two submission files"
    )
    parser.add_argument(
        "--diff_folder", type=str, default=None, help="Folder to save differences"
    )

    args = parser.parse_args()

    submission_file = args.submission_file

    if args.all_predictions_file is not None:
        print("Using all outputs by key instead of the submission file")
        outputs_by_key = json.load(open(args.all_predictions_file, "r"))
        # outputs_by_key = {key: outs for key, outs in outputs_by_key.items() if outs[0]['inverter'] is None or outs[0]['inverter'] == 'identity'}
        tasks = read_tasks_from_single_file(
            args.data_file, solution_file=args.solution_file, test=False
        )
        outputs = {}
        for task in tasks:
            name = task.name

            to_vote = [out for key, out in outputs_by_key.items() if name in key]
            to_vote = [out for sublist in to_vote for out in sublist]

            if len(to_vote) == 0:
                outputs[name] = [[[0]], [[0]]]
                continue
            else:
                attempt_1, attempt_2 = vote(to_vote)
                outputs[name] = [to_list(attempt_1), to_list(attempt_2)]

        predictions = [outputs[task.name] for task in tasks]


        submission_file = os.path.join(os.path.dirname(args.all_predictions_file), "submission.json")

        make_submission(tasks, predictions, submission_file)

    evaluate(args.data_file, args.solution_file, submission_file, args.task_info_file, mistakes=args.plot_mistakes)

    if args.compare_submission_file is not None:
        compare(args.data_file, args.solution_file, submission_file, args.compare_submission_file, plot_differents=args.plot_mistakes, diff_folder=args.diff_folder)
