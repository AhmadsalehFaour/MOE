import argparse
from PIL import Image
from src.moe.describer import MoEImageDescriber

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-model", default="llama3")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--entropy-hi", type=float, default=3.5)
    args = parser.parse_args()

    img = Image.open(args.image).convert("RGB")
    moe = MoEImageDescriber(device=args.device, ollama_url=args.ollama_url, ollama_model=args.ollama_model,
                            p_threshold=args.threshold, entropy_hi=args.entropy_hi)
    result = moe.describe(img, topk=5, max_tokens=200)

    print("Routing:", result['route'])
    print("Rationale:", result['rationale'])
    print("Labels:", result['labels'])
    print("Colors:", result['colors'])
    print("\nCaption:\n", result['caption'])

if __name__ == "__main__":
    main()
