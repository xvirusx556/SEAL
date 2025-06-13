
# for self-edit.py
self_edit_prompt = """
You are configuring a model training pipeline by selecting from predefined tools.

You must make two decisions:

1. **Data Generation Tools** — For each of the following, choose true or false:
    - use_basic_augmentations
    - use_size_augmentations
    - use_chain_augmentations
    - use_repeat_augmentations

2. **Training Configuration** — Choose one of:
    - "train_using_all_tokens"
    - "train_using_output_tokens"

Also specify:
    - learning_rate (float)
    - num_train_epochs (integer)

### Output Format

Respond with a valid JSON object. Do not include any explanation, markdown, or extra text. Use lowercase `true`/`false` for booleans and ensure correct JSON syntax.

Example output:

{
  "data_generation": {
    "use_basic_augmentations": ...,
    "use_size_augmentations": ...,
    "use_chain_augmentations": ...,
    "use_repeat_augmentations": ...
  },
  "training": {
    "strategy": ...,
    "learning_rate": ...,
    "num_train_epochs": ...
  }
}
"""

system_message = "You are a helpful assistant that provide the correct output for the given task immediately."
