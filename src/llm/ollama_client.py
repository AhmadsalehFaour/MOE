import requests
from .base import LLMClient

class OllamaClient(LLMClient):
    def __init__(self, base_url: str = "http://127.0.0.1:11434", model: str = "llama3", timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 200) -> str:
        prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}\n\n[ASSISTANT]"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": max_tokens},
        }
        try:
            r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            return data.get("response", "").strip()
        except Exception as e:
            return f"[Ollama error: {e}]"
