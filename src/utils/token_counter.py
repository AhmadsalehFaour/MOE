def estimate_tokens(text: str) -> int:
    return max(1, len(text.split()) // 0.75)
