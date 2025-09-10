def make_caption_prompt(labels, colors, complexity: float) -> str:
    labels_str = ", ".join([f"{l} ({p:.0%})" for l, p in labels])
    color_str = ", ".join(colors) if colors else "neutral"
    return (
        "Cues from the image classification and heuristics:\n"
        f"• Top labels: {labels_str}\n"
        f"• Dominant colors: {color_str}\n"
        f"• Complexity: {complexity:.2f}\n\n"
        "Write a natural caption grounded in these cues."
    )
