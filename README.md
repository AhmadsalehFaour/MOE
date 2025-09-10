# Mixture-of-Experts (MoE) Vision + Ollama Project

This project implements a Mixture-of-Experts captioning pipeline:
- **Vision Expert** (ResNet-50) returns top-k labels, confidence, entropy, colors, and a complexity score.
- **Gating Network** routes to either a template caption or a **local Ollama** LLM (e.g., `llama3`) to produce a richer caption.
- **Prompt Engineering** utilities compose grounded prompts for the LLM.

## Project Structure
(Modeled after your reference)
```
config/
src/
  llm/
  vision/
  prompt_engineering/
  utils/
  handlers/
  moe/
data/
examples/
notebooks/
requirements.txt
setup.py
README.md
Dockerfile
```

## Quickstart
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Make sure **Ollama** is running and pull a model:
   ```bash
   ollama pull llama3
   ```
3. Try an example:
   ```bash
   python examples/basic_caption.py --image path/to/your.jpg --ollama-url http://127.0.0.1:11434 --ollama-model llama3
   ```
