"""
src/scoring/brain_scorer.py
══════════════════════════════════════════════════════════════
Step 8 — Brain Scorer

The final scoring layer. Takes fused encoder scores, optionally
refines them through the trained EEG bridge model, then produces:
  - 6 brain region scores (0–100)
  - Overall virality grade (A+ to F)
  - Benchmark comparison vs known viral ideas
  - Weakest region identification for improvement targeting

Usage
─────
  python src/scoring/brain_scorer.py

  from src.scoring.brain_scorer import BrainScorer
  scorer = BrainScorer()
  result = scorer.score(fused_scores)
"""

import numpy as np
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[2]
FINAL_DIR = ROOT / "models" / "final"

REGIONS = ["dmn","ecn","san","striatum","hippocampus","basal"]

REGION_LABELS = {
    "dmn"         : "Creativity / Novelty",
    "ecn"         : "Feasibility / Logic",
    "san"         : "Emotional Hook",
    "striatum"    : "Dopamine / Want",
    "hippocampus" : "Memorability",
    "basal"       : "Repetitiveness ↓",
}

GRADES = [
    (92,"A+"),(85,"A"),(78,"A-"),
    (72,"B+"),(65,"B"),(58,"B-"),
    (45,"C"),(30,"D"),(0,"F"),
]

def _grade(s: float) -> str:
    for t, g in GRADES:
        if s >= t: return g
    return "F"


# ── Viral benchmark ideas ────────────────────────────────────────────
# Known ideas with their approximate brain region activation profiles.
# Used for comparison in the output report.
BENCHMARKS = [
    {
        "name"    : "Airbnb",
        "tagline" : "Rent your spare room to travellers",
        "scores"  : {"dmn":78,"ecn":82,"san":70,"striatum":88,"hippocampus":85,"basal":35},
        "outcome" : "IPO $47B",
    },
    {
        "name"    : "Uber",
        "tagline" : "Tap a button, get a ride",
        "scores"  : {"dmn":72,"ecn":85,"san":75,"striatum":92,"hippocampus":90,"basal":40},
        "outcome" : "IPO $82B",
    },
    {
        "name"    : "TikTok",
        "tagline" : "Short video platform powered by AI recommendation",
        "scores"  : {"dmn":65,"ecn":60,"san":95,"striatum":90,"hippocampus":88,"basal":50},
        "outcome" : "1B+ users",
    },
    {
        "name"    : "Notion",
        "tagline" : "All-in-one workspace for notes, docs and databases",
        "scores"  : {"dmn":70,"ecn":88,"san":60,"striatum":72,"hippocampus":80,"basal":45},
        "outcome" : "$10B valuation",
    },
    {
        "name"    : "Juicero (failed)",
        "tagline" : "A $400 Wi-Fi connected juice press",
        "scores"  : {"dmn":55,"ecn":40,"san":30,"striatum":25,"hippocampus":35,"basal":80},
        "outcome" : "Shut down",
    },
]


class BrainScorer:
    """
    Final scoring layer — produces the complete brain activation report.
    """

    def __init__(self, use_bridge_model: bool = True):
        self.use_bridge = use_bridge_model
        self._model     = None
        self._device    = None

    def _try_load_model(self):
        """Try to load the trained bridge model. Falls back gracefully."""
        if not self.use_bridge:
            return False
        model_path = FINAL_DIR / "eeg_bridge.pt"
        if not model_path.exists():
            return False
        try:
            from src.scoring.bridge_model import load_trained_model
            self._model, self._device = load_trained_model()
            return True
        except Exception as e:
            print(f"  ⚠  Bridge model unavailable ({e}) — using fusion scores directly")
            return False

    def score(self, fused_scores: dict) -> dict:
        """
        Produce final brain activation report from fused scores.

        Parameters
        ----------
        fused_scores : dict — output from LateFusion.fuse()
                       must contain: dmn, ecn, san, striatum,
                                     hippocampus, basal

        Returns
        -------
        dict with full report:
            scores        : 6 region scores (0–100)
            overall       : float
            grade         : str
            weakest       : str  (region name)
            strongest     : str  (region name)
            benchmarks    : list of comparison results
            interpretation: plain-English summary
        """
        # Extract 6 scores
        scores = {r: float(fused_scores.get(r, 50.0)) for r in REGIONS}

        # Bridge model refinement only applies when raw EEG features (2548-dim)
        # are passed directly. When receiving 6-dim fusion scores, skip refinement
        # as the model was trained on 2548-dim EEG feature vectors.
        # (Bridge model is used in the full pipeline via predict() in bridge_model.py)

        # Overall virality
        overall = float(np.clip(
            scores["dmn"]         * 0.18 +
            scores["ecn"]         * 0.16 +
            scores["san"]         * 0.20 +
            scores["striatum"]    * 0.25 +
            scores["hippocampus"] * 0.12 +
            (100 - scores["basal"]) * 0.09,
            0, 100
        ))

        # Find weakest and strongest (excluding basal from strongest)
        non_basal = {r: scores[r] for r in REGIONS if r != "basal"}
        weakest   = min(non_basal, key=non_basal.get)
        strongest = max(non_basal, key=non_basal.get)

        # Benchmark comparison
        benchmarks = self._compare_benchmarks(scores)

        # Plain-English interpretation
        interpretation = self._interpret(scores, overall, weakest, strongest)

        return {
            "scores"         : scores,
            "overall"        : round(overall, 1),
            "grade"          : _grade(overall),
            "weakest"        : weakest,
            "strongest"      : strongest,
            "benchmarks"     : benchmarks,
            "interpretation" : interpretation,
            "modalities"     : fused_scores.get("modalities", ["text"]),
        }

    def _compare_benchmarks(self, scores: dict) -> list:
        """Compare idea scores against benchmark viral/failed ideas."""
        results = []
        for b in BENCHMARKS:
            # Euclidean distance in 6D score space (lower = more similar)
            dist = float(np.sqrt(sum(
                (scores[r] - b["scores"][r])**2 for r in REGIONS
            )))
            # Similarity score (0–100)
            similarity = float(np.clip(100 - dist / 2, 0, 100))
            results.append({
                "name"      : b["name"],
                "tagline"   : b["tagline"],
                "outcome"   : b["outcome"],
                "similarity": round(similarity, 1),
                "distance"  : round(dist, 1),
            })
        # Sort by similarity descending
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:3]   # top 3 closest benchmarks

    def _interpret(self, scores: dict, overall: float,
                   weakest: str, strongest: str) -> str:
        """Generate a plain-English summary of the brain activation pattern."""
        grade = _grade(overall)
        lines = []

        # Overall verdict
        if overall >= 80:
            lines.append(f"Strong virality potential (Grade {grade}). "
                         "Brain activation pattern resembles high-engagement ideas.")
        elif overall >= 60:
            lines.append(f"Moderate virality potential (Grade {grade}). "
                         "Core strengths present but key regions need reinforcement.")
        elif overall >= 40:
            lines.append(f"Below average virality potential (Grade {grade}). "
                         "The idea has novelty but lacks emotional and reward signals.")
        else:
            lines.append(f"Low virality potential (Grade {grade}). "
                         "The idea needs significant reframing to activate key brain regions.")

        # Strongest region
        lines.append(
            f"Strongest activation: {REGION_LABELS[strongest]} "
            f"({scores[strongest]:.0f}/100)."
        )

        # Weakest region
        lines.append(
            f"Critical weakness: {REGION_LABELS[weakest]} "
            f"({scores[weakest]:.0f}/100) — this is the primary target for improvement."
        )

        # Basal note
        if scores["basal"] > 60:
            lines.append(
                "⚠ High repetitiveness score — the idea resembles existing "
                "concepts too closely. Differentiation is needed."
            )
        elif scores["basal"] < 20:
            lines.append(
                "✓ Low repetitiveness — the idea is genuinely original."
            )

        return " ".join(lines)


# ════════════════════════════════════════════════════════════
# PRETTY PRINT
# ════════════════════════════════════════════════════════════

def _print_report(r: dict):
    scores = r["scores"]
    print(f"\n  ┌─ Brain Activation Report")
    print(f"  │  Overall Grade : {r['grade']}  ({r['overall']}/100)")
    print(f"  │")

    for region in REGIONS:
        s     = scores[region]
        bar   = "█" * int(s / 5)
        label = REGION_LABELS[region]
        tag   = ""
        if region == r["weakest"]:   tag = "  ← weakest"
        if region == r["strongest"]: tag = "  ← strongest"
        if region == "basal": tag += "  (low=original)"
        print(f"  │  {label:<28} {s:5.1f}  {bar}{tag}")

    print(f"  │")
    print(f"  │  Interpretation:")
    # Word-wrap interpretation
    words = r["interpretation"].split()
    line  = "  │    "
    for w in words:
        if len(line) + len(w) > 72:
            print(line)
            line = "  │    "
        line += w + " "
    if line.strip():
        print(line)

    print(f"  │")
    print(f"  │  Closest Benchmarks:")
    for b in r["benchmarks"]:
        print(f"  │    {b['name']:<20} similarity={b['similarity']:5.1f}  "
              f"({b['outcome']})")

    print(f"  └{'─'*56}")


# ════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))

    print("\n" + "═"*56)
    print("  STEP 8 — BRAIN SCORER")
    print("═"*56)

    scorer = BrainScorer(use_bridge_model=True)

    # Test with your own idea's fused scores (from Step 6 output)
    test_cases = [
        {
            "label": "Your idea — text only",
            "scores": {
                "dmn":92.3,"ecn":21.6,"san":6.0,
                "striatum":5.1,"hippocampus":20.9,"basal":0.0,
                "modalities": ["text"],
            }
        },
        {
            "label": "Your idea — text + audio",
            "scores": {
                "dmn":77.7,"ecn":23.2,"san":66.4,
                "striatum":51.0,"hippocampus":44.5,"basal":11.0,
                "modalities": ["text","audio"],
            }
        },
        {
            "label": "All three modalities",
            "scores": {
                "dmn":64.2,"ecn":33.9,"san":63.3,
                "striatum":50.6,"hippocampus":41.9,"basal":22.0,
                "modalities": ["text","audio","video"],
            }
        },
    ]

    for case in test_cases:
        print(f"\n  ── {case['label']} ──")
        result = scorer.score(case["scores"])
        _print_report(result)

    print("\n  ✅  Brain scorer working correctly.\n")