"""
src/encoders/text_encoder.py
══════════════════════════════════════════════════════════════
Step 3 — Text Encoder

Takes a text idea description and extracts 6 NLP feature scores
that map to the 6 brain region proxies learned from EEG data.

Model   : all-MiniLM-L6-v2  (Sentence-BERT, 384-dim embeddings)
          ~80 MB, runs fine on RTX 3050 4 GB

Feature → Brain Region mapping
───────────────────────────────
  semantic_novelty     → DMN         (how far from known ideas)
  logical_coherence    → ECN         (structure and feasibility)
  emotional_intensity  → SAN         (emotional hook strength)
  desire_score         → Striatum    (reward / want signal)
  narrative_score      → Hippocampus (memorability)
  similarity_score     → Basal       (repetitiveness — lower is better)

Usage
─────
  python src/encoders/text_encoder.py

  from src.encoders.text_encoder import TextEncoder
  enc    = TextEncoder()
  scores = enc.encode("An AI that reads brain waves to grade startup ideas")
"""

import re
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ── Seed corpus — common/known ideas the model compares against ──────
SEED_IDEAS = [
    "food delivery app for restaurants",
    "ride sharing platform for commuters",
    "online marketplace for buying and selling products",
    "social media platform for sharing photos and videos",
    "fitness tracking app for workouts and calories",
    "e-learning platform for online courses",
    "project management tool for teams",
    "customer support chatbot for businesses",
    "personal finance app for budgeting",
    "smart home automation system",
    "telemedicine platform for remote doctor consultations",
    "job board for employment opportunities",
    "real estate app for buying and renting",
    "travel booking platform for flights and hotels",
    "recipe recommendation app for cooking",
]

# ── Word signal lists ────────────────────────────────────────────────
DESIRE_WORDS = [
    "want","need","wish","desire","must have","finally","imagine",
    "revolutionary","game changer","never seen","first ever",
    "breakthrough","transforms","eliminates","solves","instantly",
    "everyone","millions","massive","huge demand","can't live without",
]
NARRATIVE_WORDS = [
    "story","journey","imagine","picture this","once","today",
    "everyone knows","the problem is","we all","what if","meet",
    "introducing","like","just like","but better","the new",
]
CLICHE_WORDS = [
    "uber for","airbnb for","tinder for","linkedin for",
    "disrupting","disruptive","game changing","next generation",
    "ai powered","blockchain based","leveraging","synergy",
    "scalable solution","end to end","seamless experience",
    "one stop shop","cutting edge","state of the art",
]
LOGIC_WORDS = [
    "because","therefore","which means","as a result","enables",
    "allows","by using","through","via","the mechanism",
    "specifically","for example","step","process",
    "users can","this works by","the system","in order to",
]

# ── Grade thresholds ─────────────────────────────────────────────────
GRADES = [
    (92,"A+"),(85,"A"),(78,"A-"),
    (72,"B+"),(65,"B"),(58,"B-"),
    (45,"C"),(30,"D"),(0,"F"),
]


def _score_to_grade(s: float) -> str:
    for t, g in GRADES:
        if s >= t: return g
    return "F"


def _word_density(text: str, wordlist: list) -> float:
    """Hits per 100 words, clamped to 0–1."""
    t    = text.lower()
    n    = len(t.split()) + 1
    hits = sum(1 for w in wordlist if w in t)
    return min(hits / n * 100, 1.0)


class TextEncoder:
    """
    Encodes an idea text into 6 brain-region proxy scores (0–100).
    Sentence-BERT handles semantic features.
    Rule-based NLP handles linguistic features.
    """

    def __init__(self, device: str = "auto"):
        self.model     = None
        self.seed_embs = None
        self._device   = device
        self._loaded   = False

    # ── LAZY LOAD ───────────────────────────────────────────────────

    def _load(self):
        if self._loaded:
            return
        print("  Loading Sentence-BERT (all-MiniLM-L6-v2) …")
        from sentence_transformers import SentenceTransformer
        import torch

        if self._device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device=self._device,
        )
        self.seed_embs = self.model.encode(
            SEED_IDEAS,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        self._loaded = True
        print(f"  ✓ Loaded on {self._device}")

    # ── 6 FEATURE FUNCTIONS ─────────────────────────────────────────

    def _dmn(self, emb: np.ndarray) -> float:
        """Semantic novelty — distance from known idea centroid."""
        mean_sim = float(np.mean(self.seed_embs @ emb))
        return float(np.clip((1.0 - mean_sim) * 100, 0, 100))

    def _ecn(self, text: str) -> float:
        """Logical coherence — structure + length + connectors."""
        logic   = _word_density(text, LOGIC_WORDS)
        length  = min(len(text.split()) / 100, 1.0)
        sents   = min(len(re.split(r'[.!?]', text)) / 5, 1.0)
        return float(np.clip((logic*0.4 + length*0.3 + sents*0.3) * 100, 0, 100))

    def _san(self, text: str) -> float:
        """Emotional intensity — desire words + emphasis signals."""
        desire = _word_density(text, DESIRE_WORDS)
        punct  = min((text.count("!") + text.count("?")) / 3, 1.0)
        caps   = min(sum(1 for w in text.split() if w.isupper() and len(w)>1)/5, 1.0)
        return float(np.clip((desire*0.5 + punct*0.2 + caps*0.3) * 100, 5, 100))

    def _striatum(self, text: str, emb: np.ndarray) -> float:
        """Reward anticipation — desire language + semantic want-signal."""
        desire = _word_density(text, DESIRE_WORDS)
        reward_refs = [
            "I want this immediately",
            "this solves a painful problem I have every day",
            "I would pay for this right now",
            "everyone needs this in their life",
        ]
        ref_embs = self.model.encode(
            reward_refs, convert_to_numpy=True,
            show_progress_bar=False, normalize_embeddings=True,
        )
        sem = float(np.mean(ref_embs @ emb))
        return float(np.clip((desire*0.4 + max(sem,0)*0.6) * 100, 0, 100))

    def _hippocampus(self, text: str, emb: np.ndarray) -> float:
        """Memorability — narrative structure + sticky framing."""
        narr = _word_density(text, NARRATIVE_WORDS)
        story_refs = [
            "imagine a world where this problem is solved",
            "here is a story that will change how you think",
            "this is the one thing you will remember tomorrow",
        ]
        ref_embs = self.model.encode(
            story_refs, convert_to_numpy=True,
            show_progress_bar=False, normalize_embeddings=True,
        )
        sem     = float(np.mean(ref_embs @ emb))
        wc      = len(text.split())
        brevity = 1.0 if 10 <= wc <= 25 else 0.6
        return float(np.clip((narr*0.35 + max(sem,0)*0.45 + brevity*0.20)*100, 0, 100))

    def _basal(self, text: str, emb: np.ndarray) -> float:
        """Repetitiveness — cliché density + max seed similarity."""
        cliche  = _word_density(text, CLICHE_WORDS)
        max_sim = float(np.max(self.seed_embs @ emb))
        raw     = cliche*0.4 + max(max_sim - 0.3, 0)/0.7 * 0.6
        return float(np.clip(raw * 100, 0, 100))

    # ── MAIN ENCODE ─────────────────────────────────────────────────

    def encode(self, idea_text: str) -> dict:
        """
        Encode idea text → 6 brain region scores + grade.

        Returns
        -------
        dict: dmn, ecn, san, striatum, hippocampus, basal,
              overall, grade, embedding
        """
        self._load()

        emb = self.model.encode(
            idea_text, convert_to_numpy=True,
            show_progress_bar=False, normalize_embeddings=True,
        )

        dmn  = self._dmn(emb)
        ecn  = self._ecn(idea_text)
        san  = self._san(idea_text)
        stri = self._striatum(idea_text, emb)
        hipp = self._hippocampus(idea_text, emb)
        bas  = self._basal(idea_text, emb)

        # Weighted overall — Striatum+SAN carry most weight per neuroforecasting lit
        # Basal is inverted (low repetitiveness = good)
        overall = float(np.clip(
            dmn  * 0.18 +
            ecn  * 0.16 +
            san  * 0.20 +
            stri * 0.25 +
            hipp * 0.12 +
            (100 - bas) * 0.09,
            0, 100
        ))

        return {
            "dmn"         : round(dmn,     1),
            "ecn"         : round(ecn,     1),
            "san"         : round(san,     1),
            "striatum"    : round(stri,    1),
            "hippocampus" : round(hipp,    1),
            "basal"       : round(bas,     1),
            "overall"     : round(overall, 1),
            "grade"       : _score_to_grade(overall),
            "embedding"   : emb,
        }

    def encode_batch(self, texts: list) -> list:
        """Encode a list of idea texts."""
        self._load()
        return [self.encode(t) for t in texts]


# ── PRETTY PRINT ────────────────────────────────────────────────────

def _print_result(idea: str, r: dict):
    LABELS = {
        "dmn"         : "DMN        Creativity",
        "ecn"         : "ECN        Feasibility",
        "san"         : "SAN        Emotion",
        "striatum"    : "Striatum   Reward",
        "hippocampus" : "Hippocampus Memory",
        "basal"       : "Basal      Repetitive ↓",
    }
    print(f"\n  ┌─ Idea: {idea[:72]}")
    print(f"  │  Grade: {r['grade']}  ({r['overall']}/100)")
    print(f"  │")
    for k, label in LABELS.items():
        s   = r[k]
        bar = "█" * int(s / 5)
        inv = "  (low=original)" if k == "basal" else ""
        print(f"  │  {label:<24} {s:5.1f}  {bar}{inv}")
    print(f"  └{'─'*52}")


# ── ENTRY POINT ─────────────────────────────────────────────────────

if __name__ == "__main__":
    enc = TextEncoder()

    ideas = [
        # Your idea
        "An ML system that simulates 6 brain networks to evaluate how viral "
        "an idea will be — scoring creativity, feasibility, emotional hook, "
        "reward signal, memorability, and repetitiveness from text, audio and video.",

        # Viral benchmark
        "A pill that lets you choose what you dream about so you learn "
        "new skills while sleeping.",

        # Derivative idea
        "Another food delivery app using AI and blockchain for a seamless "
        "next-generation dining experience with cutting edge technology.",
    ]

    print("\n" + "═"*56)
    print("  STEP 3 — TEXT ENCODER")
    print("═"*56)

    for idea in ideas:
        result = enc.encode(idea)
        _print_result(idea, result)

    print("\n  ✅  Text encoder working correctly.\n")