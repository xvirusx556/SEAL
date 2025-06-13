
import itertools
import os
import time
from collections.abc import Iterable, Iterator
from typing import Any, Callable, Dict, List, Union

from litellm import RetryPolicy, Router


MESSAGE = Dict[str, Union[str, Dict]]
MESSAGES = List[MESSAGE]

LM_LIST: List[Dict[str, Any]] = [
    {
        "model_name": "gpt-4-0613",
        "litellm_params": {
            "model": "gpt-4-0613",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
    },
    {
        "model_name": "gpt-4-32k-0314",
        "litellm_params": {
            "model": "gpt-4-32k-0314",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
    },
    {
        "model_name": "gpt-4o",
        "litellm_params": {
            "model": "gpt-4o",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
    },
    {
        "model_name": "o1-mini",
        "litellm_params": {
            "model": "o1-mini",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
    },
    {
        "model_name": "claude-3-opus-20240229",
        "litellm_params": {
            "model": "claude-3-opus-20240229",
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
        },
    },
    {
        "model_name": "claude-3-5-sonnet-20240620",
        "litellm_params": {
            "model": "claude-3-5-sonnet-20240620",
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
        },
    },
    {
        "model_name": "gpt-4o-2024-08-06",
        "litellm_params": {
            "model": "gpt-4o-2024-08-06",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
    },
]

def setup_lm_api(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_password: str = "",
    timeout: float = 180.0,
    retry_after: int = 1,
    allowed_fails: int = 4,
    cache_responses: bool = True,
    cache_kwargs: Dict[str, Any] = {"namespace": "representer_ekin", "redis_flush_size": 5}
) -> None:
    retry_policy = RetryPolicy(
        ContentPolicyViolationErrorRetries=0,
        AuthenticationErrorRetries=0,
        BadRequestErrorRetries=1,
        TimeoutErrorRetries=4,
        RateLimitErrorRetries=5,
    )

    time.sleep(1)  # wait for cache to be ready

    router = Router(
        model_list=LM_LIST,
        retry_after=retry_after,
        allowed_fails=allowed_fails,
        retry_policy=retry_policy,
        cache_responses=cache_responses,
        redis_host=redis_host,
        redis_port=redis_port,
        timeout=timeout,
        redis_password=redis_password,
        cache_kwargs=cache_kwargs
    )

    # Attempt to set a test key in Redis to check if the cache is working
    test_key = "test_redis_key"
    test_value = "test_value"
    router.cache.redis_cache.set_cache(test_key, test_value)
    # throw and error

    return router


def batch(inputs: Union[List, Iterable], n: int):
    "Batch data into iterators of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")

    if not isinstance(inputs, Iterator):
        inputs = iter(inputs)

    while True:
        chunk_it = itertools.islice(inputs, n)
        try:
            first_el = next(chunk_it)
        except StopIteration:
            return

async def run_agent(
    inputs: List[MESSAGES],
    async_llm: Callable[MESSAGES, Any],
    batch_size: int = 20,
) -> Any:
    """Runs the API function in parallel on the inputs."""
    for task_inputs in batch(inputs, batch_size):
        tasks = [async_llm(task_input) for task_input in task_inputs]

        results = await asyncio.gather(*tasks, return_exceptions=False)

        for result in results:
            yield result

