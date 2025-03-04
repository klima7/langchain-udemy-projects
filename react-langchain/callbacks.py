from typing import Any

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult


class AgentCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        print(f"***Prompt to LLM was:***\n{prompts[0]}")
        print("******")
        
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        print(f"***LLM response:***\n{response.generations[0][0].text}")
        print("******")
