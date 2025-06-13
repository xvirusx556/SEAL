from typing import Dict, List, Optional, Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import PreTrainedTokenizer
from vllm import LLM, EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest
from peft import PeftModel

def get_sampling_params(
    tokenizer: PreTrainedTokenizer,
    num_tokens: int,
    max_tokens: int,
    temperature: float = 0.0,
    n: int = 1,
) -> SamplingParams:
    max_new_tokens = max_tokens - num_tokens
    return SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        n=n,
        stop=[tokenizer.eos_token, "<|eot_id|>"],
        # best_of=10,
        # use_beam_search=True,
    )


def initialize_engine(
    model: str,
    enforce_eager: bool = False,
    enable_lora: bool = True,
    max_lora_rank: int = 64,
    quantization: Optional[str] = None,
    lora_repo: Optional[str] = None,
    lora_target_modules: Optional[List[str]] = None,
) -> LLMEngine:
    """Initialize the LLMEngine."""

    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model).to("cuda")

    return (model, tokenizer)


def process_requests(engine, test_prompts) -> Dict[str, List[str]]:
    """Continuously process a list of prompts and handle the outputs."""
    all_outputs: Dict[str, List[str]] = {}
    model, tokenizer = engine
    while test_prompts:
        print("number of promtps left:",  len(test_prompts))
        prompt, sampling_param, lora_request, idx = test_prompts.pop(0)
        find_start = prompt.find("<|begin_of_text|>") + len("<|begin_of_text|>")
        prompt = prompt[find_start:]
        lora_path = lora_request.lora_path
        
        # # Load LoRA weights
        # if lora_path:
        #     model = PeftModel.from_pretrained(model, lora_path)
        #     model = model.merge_and_unload()
        
        tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        request_outputs = model.generate(tokenized_prompt, do_sample=False, max_new_tokens=8192)

        # Get only the generated part by slicing off the input prompt
        input_length = tokenized_prompt.shape[1]
        generated_tokens = request_outputs[0][input_length:]
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        all_outputs[idx] = [text]
    import ipdb; ipdb.set_trace()
    return all_outputs
