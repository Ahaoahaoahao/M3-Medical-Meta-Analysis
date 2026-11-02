import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional
from openai import OpenAI, APITimeoutError

logger = logging.getLogger(__name__)

LLM_CALL_TIMEOUT = 180
DEFAULT_TEMPERATURE = 0

class BaseAgent(ABC):
    def __init__(self, client: OpenAI, model_name: str, system_prompt: str = ""):
        self.client = client
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.metrics = {
            'total_calls': 0, 'total_prompt_tokens': 0,
            'total_completion_tokens': 0, 'total_time': 0.0, 'error_count': 0
        }

    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        pass

    def _make_llm_call(self, prompt: str, temperature: float = DEFAULT_TEMPERATURE, json_mode: bool = False) -> Tuple[
        str, int, int, float]:
        start_time = time.time()
        try:
            messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}]
            call_kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "timeout": LLM_CALL_TIMEOUT
            }
            if json_mode:
                call_kwargs["response_format"] = {"type": "json_object"}

            response = self.client.chat.completions.create(**call_kwargs)
            content = response.choices[0].message.content.strip()
            p_tokens = response.usage.prompt_tokens
            c_tokens = response.usage.completion_tokens
            duration = round(time.time() - start_time, 2)

            logger.info(
                f"LLM Call Success - Prompt Tokens: {p_tokens}, Completion Tokens: {c_tokens}, Duration: {duration}s")

            self.metrics['total_calls'] += 1
            self.metrics['total_prompt_tokens'] += p_tokens
            self.metrics['total_completion_tokens'] += c_tokens
            self.metrics['total_time'] += duration
            return content, p_tokens, c_tokens, duration

        except APITimeoutError:
            self.metrics['error_count'] += 1
            logger.error(f"API Call Timeout (>{LLM_CALL_TIMEOUT}s) in {self.__class__.__name__}")
            raise
        except Exception as e:
            self.metrics['error_count'] += 1
            logger.error(f"API Call Error in {self.__class__.__name__}: {str(e)}", exc_info=True)
            raise
