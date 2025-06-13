"""
This module contains the voting functions for the ARCLib.

row_base_majority_voting: a function to vote based on the majority of rows or columns
get_all_type_of_votingsv2: a function to get the top-3 most common outputs
vote: a function to vote based on the category outputs
"""
import numpy as np

from .arc import to_tuple


def row_base_majority_voting(outputs, transpose=False):
    if transpose:
        outputs = [to_tuple(np.array(output).T) for output in outputs]
    unique_shapes = [np.array(output).shape for output in outputs]
    most_common_shape = max(set(unique_shapes), key=unique_shapes.count)

    output_rows = []
    for row in range(most_common_shape[0]):
        unique_rows = [
            output[row]
            for output in outputs
            if len(output) > row and len(output[row]) == most_common_shape[1]
        ]
        most_common_row = max(set(unique_rows), key=unique_rows.count)
        output_rows.append(most_common_row)

    output = np.array(output_rows)

    if transpose:
        output = output.T

    return to_tuple(output)


def get_all_type_of_votingsv2(outputs, row_first=True):
    if len(outputs) == 0:
        return (None, None, None)

    unique_outputs = list(set(outputs))
    counts = [outputs.count(output) for output in unique_outputs]

    most_common = unique_outputs[np.argmax(counts)]
    second_most_common = most_common
    third_most_common = most_common

    row_based_majority = row_base_majority_voting(outputs, transpose=False)
    col_based_majority = row_base_majority_voting(outputs, transpose=True)

    if len(unique_outputs) > 2:
        second_most_common = unique_outputs[np.argsort(counts)[-2]]
        third_most_common = unique_outputs[np.argsort(counts)[-3]]
    elif len(unique_outputs) > 1:
        second_most_common = unique_outputs[np.argsort(counts)[-2]]

    if second_most_common == most_common:
        second_most_common = (
            row_based_majority if row_based_majority != most_common else col_based_majority
        )

    if third_most_common in (most_common, second_most_common):
        third_most_common = (
            row_based_majority
            if row_based_majority not in (most_common, second_most_common)
            else col_based_majority
        )

    return most_common, second_most_common, third_most_common


def vote(outputs):
    # TODO: simplify the categories based on iverters.
    # The categories should've been assinged outside of this function.
    # This function just needs to vote based on the categories.
    outputs_by_category = {}
    for d in outputs:
        inverter = d["inverter"]
        if inverter is None:
            inverter = "identity"
        if inverter not in outputs_by_category:
            outputs_by_category[inverter] = []

        outputs_by_category[inverter].append(to_tuple(d["output"]))

    outputs_all = []
    for key in outputs_by_category:
        outputs_all += outputs_by_category[key]

    outputs_by_category["all"] = outputs_all

    row_first = False

    candidates = []
    for key in [
        "all",
        "identity",
        "Transpose()",
        "Flip(0)",
        "Flip(1)",
        "Rotate(180)",
        "Rotate(270)",
    ]:
        if key in outputs_by_category:
            category_candidates = get_all_type_of_votingsv2(
                outputs_by_category[key], row_first=row_first
            )
            for candidate in category_candidates:
                if candidate is not None:
                    candidates.append(candidate)

    C1, C2, C3 = get_all_type_of_votingsv2(candidates)

    cC1, cC2, cC3 = candidates.count(C1), candidates.count(C2), candidates.count(C3)
    try:
        if cC2 == cC3:
            c2_identity_counts = outputs_by_category["identity"].count(C2)
            c3_identity_counts = outputs_by_category["identity"].count(C3)
            if c2_identity_counts > c3_identity_counts:
                C3 = C2
            else:
                C2 = C3
    except:
        print("Error in ", id)

    return (C1, C2)
