import json
import os

filtered_tasks = [ # for the eval set 
    "59341089",
    "68b67ca3",
    "903d1b4a",
    "af24b4cc",
    "b1fc8b8e",
    "bc4146bd",
    "c7d4e6ad",
    "ce8d95cc",
    "358ba94e",
    "66e6c45b"
]

# filtered_tasks = [ # for training set 
#     "44f52bb0",
#     "4be741c5",
#     "5582e5ca",
#     "5614dbcf",
#     "5bd6f4ac",
#     "6150a2bd",
#     "67e8384a",
#     "6d0aefbc",
#     "6e02f1e3",
#     "8be77c9e",
#     "8d5021e8",
#     "8e1813be"
# ]


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  TODO CHANGED
# Input and output file paths
challenges_input = 'data/arc-agi_evaluation_challenges.json'
solutions_input = 'data/arc-agi_evaluation_solutions.json'
challenges_output = 'data/arc-agi_evaluation_challenges_filtered_1B_eval_set.json'
solutions_output = 'data/arc-agi_evaluation_solutions_filtered_1B_eval_set.json'

# Process challenges
print("Processing challenges...")
with open(challenges_input, 'r') as f:
    challenges = json.load(f)
filtered_challenges = {task_id: data for task_id, data in challenges.items() 
                      if task_id in filtered_tasks}
print(f"Filtered from {len(challenges)} to {len(filtered_challenges)} challenges")

# Process solutions
print("Processing solutions...")
with open(solutions_input, 'r') as f:
    solutions = json.load(f)
filtered_solutions = {task_id: data for task_id, data in solutions.items() 
                     if task_id in filtered_tasks}
print(f"Filtered from {len(solutions)} to {len(filtered_solutions)} solutions")

# Save filtered data
print("Saving filtered data...")
with open(challenges_output, 'w') as f:
    json.dump(filtered_challenges, f, indent=2)
with open(solutions_output, 'w') as f:
    json.dump(filtered_solutions, f, indent=2)

print("Done! Filtered files have been created:")
print(f"- {challenges_output}")
print(f"- {solutions_output}") 