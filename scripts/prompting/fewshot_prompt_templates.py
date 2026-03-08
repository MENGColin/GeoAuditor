#!/usr/bin/env python3
"""
Build few-shot and zero-shot prompt templates used in prompting experiments.
"""

FEW_SHOT_EXAMPLES = {
    "is_water_poor": [
        {"mean_prob": 0.82, "std_prob": 0.05, "min_prob": 0.75, "max_prob": 0.89, "label": 1,
         "reasoning": "Very high mean probability (0.82) with low variance indicates consistent water poverty."},
        {"mean_prob": 0.12, "std_prob": 0.04, "min_prob": 0.07, "max_prob": 0.18, "label": 0,
         "reasoning": "Very low mean probability (0.12) suggests good water access across all neighbors."},
        {"mean_prob": 0.58, "std_prob": 0.12, "min_prob": 0.42, "max_prob": 0.75, "label": 1,
         "reasoning": "Mean above 0.5 threshold with majority of neighbors showing moderate-high poverty risk."},
        {"mean_prob": 0.45, "std_prob": 0.15, "min_prob": 0.25, "max_prob": 0.68, "label": 0,
         "reasoning": "Mean below 0.5 with high variance; despite some poor neighbors, overall area has adequate access."},
    ],
    "is_electr_poor": [
        {"mean_prob": 0.88, "std_prob": 0.04, "min_prob": 0.82, "max_prob": 0.93, "label": 1,
         "reasoning": "Extremely high electricity poverty probability, likely rural area without grid connection."},
        {"mean_prob": 0.08, "std_prob": 0.03, "min_prob": 0.04, "max_prob": 0.12, "label": 0,
         "reasoning": "Very low poverty rate indicates urban/peri-urban area with reliable electricity."},
        {"mean_prob": 0.62, "std_prob": 0.18, "min_prob": 0.35, "max_prob": 0.85, "label": 1,
         "reasoning": "Above threshold with significant neighbor showing high poverty, transitional area."},
        {"mean_prob": 0.48, "std_prob": 0.10, "min_prob": 0.35, "max_prob": 0.62, "label": 0,
         "reasoning": "Borderline but below 0.5, majority neighbors have acceptable electricity access."},
    ],
    "is_facility_poor": [
        {"mean_prob": 0.78, "std_prob": 0.08, "min_prob": 0.65, "max_prob": 0.88, "label": 1,
         "reasoning": "High facility poverty across neighbors indicates systematic lack of sanitation infrastructure."},
        {"mean_prob": 0.15, "std_prob": 0.05, "min_prob": 0.08, "max_prob": 0.22, "label": 0,
         "reasoning": "Low facility poverty suggests well-developed sanitation in the area."},
        {"mean_prob": 0.55, "std_prob": 0.20, "min_prob": 0.28, "max_prob": 0.82, "label": 1,
         "reasoning": "Above threshold despite variance; spatial clustering suggests local infrastructure gap."},
        {"mean_prob": 0.42, "std_prob": 0.12, "min_prob": 0.28, "max_prob": 0.58, "label": 0,
         "reasoning": "Below threshold with moderate variance, overall adequate facility access."},
    ],
    "is_tele_poor": [
        {"mean_prob": 0.85, "std_prob": 0.06, "min_prob": 0.76, "max_prob": 0.92, "label": 1,
         "reasoning": "Very high telecom poverty, likely remote area with no mobile coverage."},
        {"mean_prob": 0.10, "std_prob": 0.04, "min_prob": 0.05, "max_prob": 0.15, "label": 0,
         "reasoning": "Near-universal telecom access in urban area."},
        {"mean_prob": 0.52, "std_prob": 0.15, "min_prob": 0.32, "max_prob": 0.72, "label": 1,
         "reasoning": "Just above threshold, transitional coverage area."},
        {"mean_prob": 0.38, "std_prob": 0.18, "min_prob": 0.15, "max_prob": 0.62, "label": 0,
         "reasoning": "Below threshold, despite some poor-coverage neighbors, area has acceptable access."},
    ],
    "is_u5mr_poor": [
        {"mean_prob": 0.72, "std_prob": 0.08, "min_prob": 0.62, "max_prob": 0.82, "label": 1,
         "reasoning": "High child mortality risk across neighbors indicates healthcare-challenged region."},
        {"mean_prob": 0.18, "std_prob": 0.05, "min_prob": 0.12, "max_prob": 0.25, "label": 0,
         "reasoning": "Low mortality rates suggest good healthcare access and nutrition."},
        {"mean_prob": 0.58, "std_prob": 0.12, "min_prob": 0.42, "max_prob": 0.75, "label": 1,
         "reasoning": "Above threshold with elevated risk patterns in spatial neighbors."},
        {"mean_prob": 0.45, "std_prob": 0.10, "min_prob": 0.32, "max_prob": 0.58, "label": 0,
         "reasoning": "Below critical threshold, moderate risk but not classified as high-mortality area."},
    ],
}


def build_few_shot_prompt(task: str, user_prompt: str, n_shots: int = 3) -> str:
    """Build Few-Shot prompt with examples"""
    examples = FEW_SHOT_EXAMPLES.get(task, FEW_SHOT_EXAMPLES["is_water_poor"])[:n_shots]
    task_name = task.replace("is_", "").replace("_poor", " poverty")

    prompt = f"""You are an expert poverty prediction model. Your task is to predict {task_name} based on neighbor context.

Key principles:
1. Spatial autocorrelation: nearby households share similar poverty patterns
2. Mean probability is the primary indicator, but consider variance
3. When mean is close to 0.5, high variance suggests uncertainty
4. Consistent high/low values among neighbors increase prediction confidence

=== Examples ===

"""

    for i, ex in enumerate(examples, 1):
        prompt += f"""Example {i}:
- Mean probability: {ex['mean_prob']:.2f}
- Std deviation: {ex['std_prob']:.2f}
- Min probability: {ex['min_prob']:.2f}
- Max probability: {ex['max_prob']:.2f}
Reasoning: {ex['reasoning']}
Answer: {ex['label']}

"""

    prompt += f"""=== Your Task ===

{user_prompt}

Based on the neighbor context above, predict 0 (not poor) or 1 (poor):"""

    return prompt


def build_zero_shot_prompt(task: str, user_prompt: str) -> str:
    """Build Zero-Shot prompt (baseline)"""
    return f"""{user_prompt}

Answer with 0 or 1 only."""
