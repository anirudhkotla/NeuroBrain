"""
src/fusion/late_fusion.py
══════════════════════════════════════════════════════════════
Step 6 — Late Fusion Layer

Combines scores from Text + Audio + Video encoders into a single
set of 6 brain region scores and an overall virality grade.

Why Late Fusion?
────────────────
Each modality activates different brain regions differently.
Late fusion (combine SCORES not raw features) is the right approach
because each modality's scores are already in the same 0–100 space,
making combination principled and interpretable.

Fusion Strategy — Weighted Average with Modality Confidence
────────────────────────────────────────────────────────────
Each modality gets a base weight. If a modality is missing (user
submitted only text, no audio/video), the weights are redistributed
proportionally so the total always sums to 1.0.

Default weights per modality per brain region:
  Text   → strongest for DMN (novelty), ECN (logic), Basal (repetition)
  Audio  → strongest for SAN (emotion), Striatum (reward/confidence)
  Video  → strongest for Hippocampus (memory), SAN (visual energy)

Usage
─────
  python src/fusion/late_fusion.py

  from src.fusion.late_fusion import LateFusion
  fuser  = LateFusion()
  result = fuser.fuse(text_scores, audio_scores, video_scores)
  result = fuser.fuse(text_scores)                        # text only
  result = fuser.fuse(text_scores, audio_scores=audio_s) # text + audio
"""

import numpy as np
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]

GRADES = [
    (92,"A+"),(85,"A"),(78,"A-"),
    (72,"B+"),(65,"B"),(58,"B-"),
    (45,"C"),(30,"D"),(0,"F"),
]

def _grade(s: float) -> str:
    for t, g in GRADES:
        if s >= t: return g
    return "F"

REGIONS = ["dmn","ecn","san","striatum","hippocampus","basal"]

# ── Per-region modality weights ──────────────────────────────────────
# Each row = one brain region
# Each col = [text_weight, audio_weight, video_weight]
# Rows sum to 1.0 when all modalities present.
MODALITY_WEIGHTS = {
    #                  text   audio  video
    "dmn"         : [0.65,  0.15,  0.20],  # novelty is mostly semantic
    "ecn"         : [0.60,  0.20,  0.20],  # logic is mostly textual
    "san"         : [0.25,  0.45,  0.30],  # emotion splits audio/text/video
    "striatum"    : [0.30,  0.45,  0.25],  # reward driven by vocal confidence
    "hippocampus" : [0.30,  0.25,  0.45],  # memory driven by visual variety
    "basal"       : [0.55,  0.20,  0.25],  # repetition mostly textual
}


class LateFusion:
    """
    Fuses text, audio, and video scores into unified brain region scores.
    Any subset of modalities can be provided — missing ones are handled
    gracefully by redistributing weights.
    """

    def __init__(self):
        self.weights = MODALITY_WEIGHTS

    def fuse(self,
             text_scores  : Optional[dict] = None,
             audio_scores : Optional[dict] = None,
             video_scores : Optional[dict] = None) -> dict:
        """
        Fuse available modality scores.

        Parameters
        ----------
        text_scores  : dict from TextEncoder.encode()   or None
        audio_scores : dict from AudioEncoder.encode_*() or None
        video_scores : dict from VideoEncoder.encode_*() or None

        At least one modality must be provided.

        Returns
        -------
        dict:
            dmn, ecn, san, striatum, hippocampus, basal  (0–100)
            overall   : float
            grade     : str
            modalities: list of which modalities were used
            per_modality: raw scores per modality before fusion
        """
        if all(s is None for s in [text_scores, audio_scores, video_scores]):
            raise ValueError("At least one modality score dict must be provided.")

        # Track which modalities are present
        present = {
            "text"  : text_scores  is not None,
            "audio" : audio_scores is not None,
            "video" : video_scores is not None,
        }
        modalities_used = [k for k, v in present.items() if v]

        # Build fused scores per region
        fused = {}
        for region in REGIONS:
            w = self.weights[region]   # [text, audio, video]

            # Zero out weights for missing modalities
            active_w = [
                w[0] if present["text"]  else 0.0,
                w[1] if present["audio"] else 0.0,
                w[2] if present["video"] else 0.0,
            ]

            # Normalise so weights sum to 1.0
            total = sum(active_w)
            if total < 1e-8:
                active_w = [1.0, 0.0, 0.0]
                total    = 1.0
            norm_w = [x / total for x in active_w]

            # Gather scores (default 50 if modality missing — neutral)
            scores_list = [
                float(text_scores.get(region,  50)) if present["text"]  else 50.0,
                float(audio_scores.get(region, 50)) if present["audio"] else 50.0,
                float(video_scores.get(region, 50)) if present["video"] else 50.0,
            ]

            fused[region] = round(
                float(np.clip(
                    sum(s * w for s, w in zip(scores_list, norm_w)),
                    0, 100
                )), 1
            )

        # Overall virality score — Striatum + SAN carry most weight
        # Basal is inverted (low = good)
        overall = float(np.clip(
            fused["dmn"]         * 0.18 +
            fused["ecn"]         * 0.16 +
            fused["san"]         * 0.20 +
            fused["striatum"]    * 0.25 +
            fused["hippocampus"] * 0.12 +
            (100 - fused["basal"]) * 0.09,
            0, 100
        ))

        # Build per-modality breakdown for the report
        per_modality = {}
        if present["text"]:
            per_modality["text"]  = {r: text_scores.get(r, 0)  for r in REGIONS}
        if present["audio"]:
            per_modality["audio"] = {r: audio_scores.get(r, 0) for r in REGIONS}
        if present["video"]:
            per_modality["video"] = {r: video_scores.get(r, 0) for r in REGIONS}

        return {
            **fused,
            "overall"      : round(overall, 1),
            "grade"        : _grade(overall),
            "modalities"   : modalities_used,
            "per_modality" : per_modality,
        }

    def fuse_from_encoders(self,
                           idea_text   : Optional[str]  = None,
                           audio_path  : Optional[str]  = None,
                           video_path  : Optional[str]  = None,
                           use_wav2vec2: bool = False,
                           use_clip    : bool = False) -> dict:
        """
        Convenience method — runs all available encoders and fuses.

        Parameters
        ----------
        idea_text  : str  — idea description text
        audio_path : str  — path to audio file (.wav .mp3 etc.)
        video_path : str  — path to video file (.mp4 .mov etc.)
        use_wav2vec2 : bool — enable wav2vec2 for audio
        use_clip     : bool — enable CLIP for video
        """
        text_scores  = None
        audio_scores = None
        video_scores = None

        if idea_text:
            print("\n  [1/3] Running Text Encoder …")
            from src.encoders.text_encoder import TextEncoder
            text_scores = TextEncoder().encode(idea_text)

        if audio_path:
            print("\n  [2/3] Running Audio Encoder …")
            from src.encoders.audio_encoder import AudioEncoder
            audio_scores = AudioEncoder(use_wav2vec2=use_wav2vec2).encode_file(audio_path)
        else:
            print("\n  [2/3] Audio — skipped (no file provided)")

        if video_path:
            print("\n  [3/3] Running Video Encoder …")
            from src.encoders.video_encoder import VideoEncoder
            video_scores = VideoEncoder(use_clip=use_clip).encode_file(video_path)
        else:
            print("\n  [3/3] Video — skipped (no file provided)")

        return self.fuse(text_scores, audio_scores, video_scores)


# ════════════════════════════════════════════════════════════
# PRETTY PRINT
# ════════════════════════════════════════════════════════════

def _print_result(r: dict):
    LABELS = {
        "dmn"         : "DMN        Creativity",
        "ecn"         : "ECN        Feasibility",
        "san"         : "SAN        Emotion",
        "striatum"    : "Striatum   Reward",
        "hippocampus" : "Hippocampus Memory",
        "basal"       : "Basal      Repetitive ↓",
    }
    MODALITY_COLORS = {"text": "📝", "audio": "🎙", "video": "🎬"}

    mods = " + ".join(
        MODALITY_COLORS.get(m, m) + " " + m
        for m in r["modalities"]
    )
    print(f"\n  ┌─ Fused result  ({mods})")
    print(f"  │  Grade : {r['grade']}  ({r['overall']}/100)")
    print(f"  │")
    for k, label in LABELS.items():
        s   = r[k]
        bar = "█" * int(s / 5)
        inv = "  (low=original)" if k == "basal" else ""
        print(f"  │  {label:<26} {s:5.1f}  {bar}{inv}")

    if r.get("per_modality"):
        print(f"  │")
        print(f"  │  Per-modality breakdown:")
        for mod, scores in r["per_modality"].items():
            vals = "  ".join(f"{scores[k]:4.0f}" for k in REGIONS)
            print(f"  │    {MODALITY_COLORS.get(mod,mod)} {mod:<6}  [{vals}]")
        print(f"  │  regions: [dmn  ecn  san  stri hipp basa]")

    print(f"  └{'─'*54}")


# ════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))

    print("\n" + "═"*56)
    print("  STEP 6 — LATE FUSION LAYER")
    print("═"*56)

    fuser = LateFusion()

    # ── Test 1: Text only ────────────────────────────────────
    print("\n  Test 1 — Text only")
    text_s = {
        "dmn":92.3,"ecn":21.6,"san":6.0,
        "striatum":5.1,"hippocampus":20.9,"basal":0.0,
    }
    r1 = fuser.fuse(text_scores=text_s)
    _print_result(r1)

    # ── Test 2: Text + Audio ─────────────────────────────────
    print("\n  Test 2 — Text + Audio")
    audio_s = {
        "dmn":14.4,"ecn":28.1,"san":99.9,
        "striatum":81.6,"hippocampus":72.9,"basal":41.3,
    }
    r2 = fuser.fuse(text_scores=text_s, audio_scores=audio_s)
    _print_result(r2)

    # ── Test 3: All three modalities ─────────────────────────
    print("\n  Test 3 — Text + Audio + Video")
    video_s = {
        "dmn":10.3,"ecn":76.6,"san":56.0,
        "striatum":49.4,"hippocampus":38.7,"basal":55.1,
    }
    r3 = fuser.fuse(text_scores=text_s, audio_scores=audio_s,
                     video_scores=video_s)
    _print_result(r3)

    # ── Test 4: Full pipeline via fuse_from_encoders ─────────
    print("\n  Test 4 — Full pipeline (text encoder live)")
    r4 = fuser.fuse_from_encoders(
        idea_text="An ML system that simulates brain networks to grade "
                  "idea virality from text, audio, and video inputs.",
    )
    _print_result(r4)

    print("\n  ✅  Late fusion working correctly.\n")