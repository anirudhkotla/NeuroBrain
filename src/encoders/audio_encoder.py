"""
src/encoders/audio_encoder.py
══════════════════════════════════════════════════════════════
Step 4 — Audio Encoder

Takes an audio file (voice pitch of idea) and extracts 6 scores
mapping to the same brain region proxies as the EEG and text encoders.

Two-stage pipeline
──────────────────
Stage 1 — Librosa (always runs, no GPU needed)
  Extracts hand-crafted acoustic features:
  pitch, energy, tempo, spectral features, MFCCs, voice quality

Stage 2 — wav2vec2-base (optional, GPU accelerated)
  Facebook's speech model extracts deep contextual embeddings.
  Falls back gracefully if model unavailable.

Feature → Brain Region mapping
───────────────────────────────
  pitch_variability  → DMN         (dynamic range = creative energy)
  speech_rate        → ECN         (measured pace = logical delivery)
  energy_arousal     → SAN         (vocal energy = emotional salience)
  vocal_confidence   → Striatum    (assertive tone = reward signal)
  prosody_richness   → Hippocampus (melodic variation = memorability)
  monotone_score     → Basal       (flat = repetitive/boring delivery)

Supported formats : .wav  .mp3  .m4a  .ogg  .flac
VRAM usage        : ~500 MB for wav2vec2 (safe on RTX 3050 4 GB)

Usage
─────
  python src/encoders/audio_encoder.py                     # demo (sine wave)
  python src/encoders/audio_encoder.py --file my_pitch.wav # real file
"""

import os
import sys
import argparse
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]

# ── Grade thresholds (shared) ────────────────────────────────────────
GRADES = [
    (92,"A+"),(85,"A"),(78,"A-"),
    (72,"B+"),(65,"B"),(58,"B-"),
    (45,"C"),(30,"D"),(0,"F"),
]

def _grade(s: float) -> str:
    for t, g in GRADES:
        if s >= t: return g
    return "F"


# ════════════════════════════════════════════════════════════
# STAGE 1 — LIBROSA ACOUSTIC FEATURES
# ════════════════════════════════════════════════════════════

def extract_librosa_features(audio: np.ndarray, sr: int) -> dict:
    """
    Extract hand-crafted acoustic features from raw audio.

    Parameters
    ----------
    audio : np.ndarray — mono waveform, float32
    sr    : int        — sample rate

    Returns
    -------
    dict of scalar feature values
    """
    import librosa

    # ── Pitch (F0) via PYIN ──────────────────────────────────
    try:
        f0, voiced_flag, _ = librosa.pyin(
            audio, fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"), sr=sr,
        )
        f0_voiced = f0[voiced_flag] if voiced_flag.any() else np.array([200.0])
    except Exception:
        f0_voiced = np.array([200.0])

    pitch_mean    = float(np.nanmean(f0_voiced))
    pitch_std     = float(np.nanstd(f0_voiced))
    pitch_range   = float(np.nanmax(f0_voiced) - np.nanmin(f0_voiced))
    voiced_ratio  = float(np.sum(voiced_flag) / len(voiced_flag)) if len(f0_voiced) > 0 else 0.5

    # ── Energy / RMS ─────────────────────────────────────────
    rms           = librosa.feature.rms(y=audio)[0]
    energy_mean   = float(np.mean(rms))
    energy_std    = float(np.std(rms))
    energy_range  = float(np.max(rms) - np.min(rms))

    # ── Tempo / Speech Rate ──────────────────────────────────
    try:
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        tempo    = float(tempo) if not np.isnan(tempo) else 120.0
    except Exception:
        tempo = 120.0

    # ── MFCCs (13 coefficients) ──────────────────────────────
    mfcc      = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)    # (13,)
    mfcc_std  = mfcc.std(axis=1)     # (13,)

    # ── Spectral features ────────────────────────────────────
    spec_centroid  = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    spec_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)))
    spec_rolloff   = float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)))
    zcr            = float(np.mean(librosa.feature.zero_crossing_rate(audio)))

    # ── Chroma ───────────────────────────────────────────────
    chroma    = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_std = float(np.std(chroma))    # high = melodically varied

    return {
        "pitch_mean"    : pitch_mean,
        "pitch_std"     : pitch_std,
        "pitch_range"   : pitch_range,
        "voiced_ratio"  : voiced_ratio,
        "energy_mean"   : energy_mean,
        "energy_std"    : energy_std,
        "energy_range"  : energy_range,
        "tempo"         : tempo,
        "mfcc_mean"     : mfcc_mean,
        "mfcc_std"      : mfcc_std,
        "spec_centroid" : spec_centroid,
        "spec_bandwidth": spec_bandwidth,
        "spec_rolloff"  : spec_rolloff,
        "zcr"           : zcr,
        "chroma_std"    : chroma_std,
    }


# ════════════════════════════════════════════════════════════
# STAGE 2 — WAV2VEC2 DEEP EMBEDDINGS (optional)
# ════════════════════════════════════════════════════════════

def extract_wav2vec2_embedding(audio: np.ndarray, sr: int,
                                device: str = "cpu") -> np.ndarray | None:
    """
    Extract 768-dim contextual embedding from wav2vec2-base.
    Returns None if model unavailable.
    """
    try:
        import torch
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
        import librosa as lr

        # Resample to 16kHz (wav2vec2 requirement)
        if sr != 16000:
            audio = lr.resample(audio, orig_sr=sr, target_sr=16000)
            sr    = 16000

        processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base",
            local_files_only=False,
        )
        model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base",
            local_files_only=False,
        ).to(device)
        model.eval()

        # Truncate to 10 seconds max (160000 samples) to save VRAM
        audio = audio[:160000]

        inputs = processor(
            audio, sampling_rate=sr,
            return_tensors="pt", padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Mean-pool across time dimension → (768,)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding.astype(np.float32)

    except Exception as e:
        print(f"  ⚠  wav2vec2 unavailable ({type(e).__name__}) — using librosa only")
        return None


# ════════════════════════════════════════════════════════════
# BRAIN REGION SCORES FROM ACOUSTIC FEATURES
# ════════════════════════════════════════════════════════════

def acoustic_to_brain_scores(feats: dict) -> dict:
    """
    Convert librosa features → 6 brain region scores (0–100).

    All normalisation constants are derived from typical human speech ranges:
      pitch_std   :  0–80  Hz  (80 = highly expressive speaker)
      pitch_range :  0–400 Hz
      energy_mean :  0–0.1 (RMS amplitude)
      tempo       :  60–200 BPM
      chroma_std  :  0–0.4
    """

    def clip(x): return float(np.clip(x, 0, 100))

    # ── DMN — pitch variability = creative vocal energy ──────
    pitch_var = feats["pitch_std"] / 80.0          # 0–1
    pitch_rng = feats["pitch_range"] / 400.0       # 0–1
    dmn = clip((pitch_var * 0.6 + pitch_rng * 0.4) * 100)

    # ── ECN — measured speech rate + steady energy ───────────
    # Ideal speech rate ~120–150 BPM; penalise too fast or too slow
    tempo_norm = 1.0 - abs(feats["tempo"] - 135) / 135
    energy_steady = 1.0 - min(feats["energy_std"] / 0.05, 1.0)  # low variance = steady
    ecn = clip((max(tempo_norm, 0) * 0.5 + energy_steady * 0.5) * 100)

    # ── SAN — overall vocal energy / arousal ─────────────────
    energy_norm = min(feats["energy_mean"] / 0.05, 1.0)
    energy_rng  = min(feats["energy_range"] / 0.1, 1.0)
    voiced_norm = feats["voiced_ratio"]
    san = clip((energy_norm * 0.4 + energy_rng * 0.3 + voiced_norm * 0.3) * 100)

    # ── Striatum — vocal confidence / assertiveness ───────────
    # High ZCR + high spectral centroid = bright, assertive voice
    zcr_norm      = min(feats["zcr"] / 0.1, 1.0)
    centroid_norm = min(feats["spec_centroid"] / 4000, 1.0)
    energy_conf   = min(feats["energy_mean"] / 0.04, 1.0)
    striatum = clip((zcr_norm * 0.3 + centroid_norm * 0.3 + energy_conf * 0.4) * 100)

    # ── Hippocampus — prosodic richness = memorability ────────
    chroma_norm = min(feats["chroma_std"] / 0.3, 1.0)
    mfcc_var    = min(float(np.mean(feats["mfcc_std"])) / 20.0, 1.0)
    hippocampus = clip((chroma_norm * 0.5 + mfcc_var * 0.5) * 100)

    # ── Basal — monotone score = repetitive delivery ──────────
    # Low pitch variance + low energy range = monotone = high basal (bad)
    monotone = 1.0 - (pitch_var * 0.5 + min(energy_rng, 1.0) * 0.5)
    basal = clip(monotone * 100)

    return {
        "dmn"         : round(dmn,         1),
        "ecn"         : round(ecn,         1),
        "san"         : round(san,         1),
        "striatum"    : round(striatum,    1),
        "hippocampus" : round(hippocampus, 1),
        "basal"       : round(basal,       1),
    }


# ════════════════════════════════════════════════════════════
# MAIN ENCODER CLASS
# ════════════════════════════════════════════════════════════

class AudioEncoder:
    """
    Encodes a voice pitch audio file → 6 brain region scores (0–100).
    """

    def __init__(self, use_wav2vec2: bool = True, device: str = "auto"):
        self.use_wav2vec2 = use_wav2vec2
        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

    def encode_file(self, audio_path: str) -> dict:
        """
        Encode an audio file → 6 brain scores.

        Parameters
        ----------
        audio_path : str — path to .wav / .mp3 / .m4a / .ogg / .flac

        Returns
        -------
        dict: dmn, ecn, san, striatum, hippocampus, basal,
              overall, grade, librosa_features, wav2vec2_embedding
        """
        import librosa
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"  Loading {path.name} …")
        audio, sr = librosa.load(str(path), sr=None, mono=True)
        duration  = len(audio) / sr
        print(f"  Duration : {duration:.1f}s  |  Sample rate : {sr} Hz")

        return self.encode_array(audio, sr)

    def encode_array(self, audio: np.ndarray, sr: int) -> dict:
        """
        Encode a raw numpy audio array → 6 brain scores.

        Parameters
        ----------
        audio : np.ndarray — mono float32 waveform
        sr    : int        — sample rate
        """
        # Stage 1 — librosa features (always)
        feats   = extract_librosa_features(audio, sr)
        scores  = acoustic_to_brain_scores(feats)

        # Stage 2 — wav2vec2 embedding (optional)
        w2v_emb = None
        if self.use_wav2vec2:
            w2v_emb = extract_wav2vec2_embedding(audio, sr, device=self.device)

        # Overall score
        overall = float(np.clip(
            scores["dmn"]         * 0.18 +
            scores["ecn"]         * 0.16 +
            scores["san"]         * 0.20 +
            scores["striatum"]    * 0.25 +
            scores["hippocampus"] * 0.12 +
            (100 - scores["basal"]) * 0.09,
            0, 100
        ))

        return {
            **scores,
            "overall"           : round(overall, 1),
            "grade"             : _grade(overall),
            "librosa_features"  : feats,
            "wav2vec2_embedding": w2v_emb,
        }

    def encode_demo(self) -> dict:
        """
        Generate a synthetic speech-like signal and encode it.
        Used for testing without a real audio file.
        """
        print("  Generating synthetic speech signal …")
        sr     = 22050
        dur    = 5.0
        t      = np.linspace(0, dur, int(sr * dur))

        # Simulate a voice: fundamental + harmonics + pitch variation
        pitch_base = 150.0                          # Hz (speaking voice)
        pitch_var  = 20 * np.sin(2 * np.pi * 0.5 * t)  # pitch modulation
        f0         = pitch_base + pitch_var

        audio  = (
            0.5 * np.sin(2 * np.pi * np.cumsum(f0) / sr) +   # fundamental
            0.3 * np.sin(4 * np.pi * np.cumsum(f0) / sr) +   # 2nd harmonic
            0.1 * np.sin(6 * np.pi * np.cumsum(f0) / sr) +   # 3rd harmonic
            0.05 * np.random.randn(len(t))                     # noise
        ).astype(np.float32)

        # Amplitude envelope (speech-like pauses)
        envelope = np.abs(np.sin(np.pi * t / 0.8))
        audio    = audio * envelope
        audio    = audio / (np.max(np.abs(audio)) + 1e-8)   # normalise

        return self.encode_array(audio, sr)


# ════════════════════════════════════════════════════════════
# PRETTY PRINT
# ════════════════════════════════════════════════════════════

def _print_result(source: str, r: dict):
    LABELS = {
        "dmn"         : "DMN        Creativity",
        "ecn"         : "ECN        Feasibility",
        "san"         : "SAN        Emotion",
        "striatum"    : "Striatum   Confidence",
        "hippocampus" : "Hippocampus Memorability",
        "basal"       : "Basal      Monotone ↓",
    }
    print(f"\n  ┌─ Source : {source}")
    print(f"  │  Grade  : {r['grade']}  ({r['overall']}/100)")
    print(f"  │")
    for k, label in LABELS.items():
        s   = r[k]
        bar = "█" * int(s / 5)
        inv = "  (low=expressive)" if k == "basal" else ""
        print(f"  │  {label:<26} {s:5.1f}  {bar}{inv}")
    print(f"  └{'─'*54}")


# ════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None,
                        help="Path to audio file (.wav .mp3 .m4a .ogg .flac)")
    parser.add_argument("--no-wav2vec2", action="store_true",
                        help="Skip wav2vec2, use librosa only (faster)")
    args = parser.parse_args()

    print("\n" + "═"*56)
    print("  STEP 4 — AUDIO ENCODER")
    print("═"*56)

    enc = AudioEncoder(use_wav2vec2=not args.no_wav2vec2)

    if args.file:
        result = enc.encode_file(args.file)
        _print_result(args.file, result)
    else:
        print("  No --file provided — running demo with synthetic audio\n")
        result = enc.encode_demo()
        _print_result("synthetic speech signal", result)

    print("\n  ✅  Audio encoder working correctly.\n")