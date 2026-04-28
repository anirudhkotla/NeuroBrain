"""
src/output/mistral_summary.py
══════════════════════════════════════════════════════════════
Uses Mistral AI (mistral-small-latest) to generate a
personalised improvement summary based on the 6 brain scores.

Mistral SDK v2 syntax:
  from mistralai import Mistral
  with Mistral(api_key=...) as client:
      res = client.chat.complete(model=..., messages=[...])
      text = res.choices[0].message.content
"""

import os
from pathlib import Path

MISTRAL_MODEL = "mistral-small-latest"

REGION_LABELS = {
    "dmn"         : "Creativity / Novelty (DMN)",
    "ecn"         : "Feasibility / Logic (ECN)",
    "san"         : "Emotional Hook (SAN)",
    "striatum"    : "Dopamine / Want (Striatum)",
    "hippocampus" : "Memorability (Hippocampus)",
    "basal"       : "Repetitiveness (Basal Ganglia)",
}

MODALITY_TIPS = {
    "dmn"         : "Try making your idea more unexpected — use a distant analogy or combine two unrelated domains.",
    "ecn"         : "Add a concrete mechanism: who does what, how, and why it works.",
    "san"         : "Open with an emotional hook — a surprising stat, a relatable pain, or a bold claim.",
    "striatum"    : "Lead with desire: 'You've always wanted X — now it's possible.' Make the reward feel immediate.",
    "hippocampus" : "Wrap it in a story or give it a one-word name people can repeat.",
    "basal"       : "This idea scores high on repetitiveness — differentiate it from existing solutions explicitly.",
}


def generate_summary(scores: dict, idea_text: str = "") -> str:
    """
    Generate improvement suggestions using Mistral.

    Parameters
    ----------
    scores    : dict with keys dmn, ecn, san, striatum, hippocampus, basal (0-100)
    idea_text : optional — the original idea text for context

    Returns
    -------
    str — Mistral-generated improvement summary
    """
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key:
        return _fallback_summary(scores)

    # Build score table for prompt
    score_lines = "\n".join(
        f"  - {REGION_LABELS[k]}: {scores.get(k, 0):.1f}/100"
        for k in REGION_LABELS
    )

    weakest = min(
        [k for k in scores if k != "basal"],
        key=lambda k: scores.get(k, 0)
    )

    prompt = f"""You are a neuroscience-inspired idea coach.
A user submitted an idea and received these brain activation scores (0-100):

{score_lines}

{"Idea: " + idea_text if idea_text else ""}

The weakest region is: {REGION_LABELS[weakest]} ({scores.get(weakest,0):.1f}/100)
Note: For Basal Ganglia, a LOW score is good (means the idea is original).

Give 3 short, specific, actionable improvement suggestions.
Focus on the weakest scoring region first.
Be direct and practical. No fluff. Use plain language."""

    try:
        from mistralai import Mistral
        with Mistral(api_key=api_key) as client:
            res = client.chat.complete(
                model=MISTRAL_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            return res.choices[0].message.content.strip()
    except Exception as e:
        return f"[Mistral unavailable: {e}]\n\n" + _fallback_summary(scores)


def _fallback_summary(scores: dict) -> str:
    """Rule-based fallback when Mistral API is not available."""
    sorted_regions = sorted(
        [k for k in REGION_LABELS if k != "basal"],
        key=lambda k: scores.get(k, 0)
    )
    weakest = sorted_regions[:2]
    lines = ["Improvement suggestions (rule-based):"]
    for w in weakest:
        lines.append(f"\n• {REGION_LABELS[w]} ({scores.get(w,0):.1f}/100)")
        lines.append(f"  → {MODALITY_TIPS[w]}")
    if scores.get("basal", 0) > 70:
        lines.append(f"\n• {REGION_LABELS['basal']} is high — your idea may be too similar to existing ones.")
        lines.append(f"  → {MODALITY_TIPS['basal']}")
    return "\n".join(lines)


# ── Quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    test_scores = {
        "dmn": 87, "ecn": 74, "san": 81,
        "striatum": 78, "hippocampus": 72, "basal": 22,
    }
    test_idea = "An ML system that mimics brain networks to grade idea virality"
    print(generate_summary(test_scores, test_idea))
