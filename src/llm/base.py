from abc import ABC, abstractmethod

class LLMClient(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 200) -> str:
        ...
