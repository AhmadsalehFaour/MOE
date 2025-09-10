# 🖼️ Mixture-of-Experts Image Captioner  

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)
![License](https://img.shields.io/badge/license-MIT-green)

This repository provides a **Mixture-of-Experts (MoE)** system for **image captioning** that combines:  

🔹 **Vision Expert (ResNet-50)** for object and color recognition  
🔹 **Gating Network** to decide whether to use a lightweight template or a full language model  
🔹 **Ollama LLM (e.g., llama3)** for natural, detailed captions  

The goal is to generate **concise, vivid, and grounded captions** for uploaded images while balancing **speed, interpretability, and richness**.

---

## ✨ Features

✅ **Hybrid routing**: Gating selects between:
- **Template captions** (fast, grounded in top labels and colors)  
- **LLM-generated captions** (richer, more descriptive)  

✅ **Vision expert**: ResNet-50 trained on ImageNet with color + complexity analysis  
✅ **Configurable thresholds**: Confidence and entropy bounds for gating  
✅ **Local LLM support**: Integration with [Ollama](https://ollama.com) for running llama3 or other models  
✅ **Streamlit UI** for interactive testing  
✅ **CLI tool** for batch or scripted usage  
✅ **Docker support** for containerized deployment  

---

## 🛠 Project Structure

```
.
├─ app.py                 # Streamlit web app
├─ basic_caption.py        # CLI tool for caption generation
├─ moe/
│  ├─ describer.py         # MoEImageDescriber logic
│  ├─ gate.py              # Gating network (template vs LLM)
│  ├─ classifier.py        # Vision classifier (ResNet-50 + color analysis)
│  ├─ ollama_client.py     # Client for Ollama LLM
│  ├─ templates.py         # Prompt templates
│  ├─ few_shot.py          # (Optional) few-shot support
│  └─ ...
├─ Dockerfile              # Build + run container
├─ model_config.yaml       # Model + device configuration
├─ prompt_templates.yaml   # Prompt formats
├─ logging_config.yaml     # Logging setup
└─ notebooks/              # Experiments and analysis
```

---

## 🚀 Usage

### ▶️ Run locally (Streamlit UI)
```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501).

### 💻 Run CLI
```bash
python basic_caption.py --image sample.jpg --device cpu
```

### 🐳 Run with Docker
```bash
docker build -t moe-captioner .
docker run -p 8501:8501 moe-captioner
```

---

## ⚙️ Configuration

⚡ **`model_config.yaml`**: select device (cpu/cuda), model paths, etc.  
⚡ **`prompt_templates.yaml`**: define caption prompts for LLM.  
⚡ **`logging_config.yaml`**: adjust logging level/format.  
⚡ **Thresholds** (`confidence`, `entropy`) control when the gate chooses template vs LLM.  

---

## 📌 Example

Upload an image → system classifies objects/colors → gate decides:  
- If **simple + high confidence** → template caption (e.g., “A photo of a dog with brown tones.”)  
- If **complex/uncertain** → llama3 generates natural caption.  

---

## 📸 Screenshots

_Add screenshots of the Streamlit UI here._

---

## 📜 License
Licensed under the **MIT License**. See `LICENSE` for details.
