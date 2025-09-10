from dataclasses import dataclass

@dataclass
class GateDecision:
    route: str  # 'template' or 'llama'
    score: float
    rationale: dict

class GatingNetwork:
    def __init__(self, p_threshold: float = 0.55, entropy_hi: float = 3.5):
        self.p_threshold = p_threshold
        self.entropy_hi = entropy_hi

    def decide(self, vis) -> GateDecision:
        simple_scene = vis.complexity < 0.25
        use_template = (vis.p_max >= self.p_threshold) and (vis.entropy <= self.entropy_hi) and simple_scene
        route = 'template' if use_template else 'llama'
        score = float(vis.p_max if use_template else (1.0 - vis.p_max))
        rationale = {
            'p_max': vis.p_max,
            'entropy': vis.entropy,
            'complexity': vis.complexity,
            'colors': vis.colors,
            'top1': vis.topk_labels[0] if vis.topk_labels else None,
        }
        return GateDecision(route=route, score=score, rationale=rationale)
