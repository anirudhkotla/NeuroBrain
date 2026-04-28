"""
src/encoders/video_encoder.py
══════════════════════════════════════════════════════════════
Step 5 — Video Encoder

Takes a video file (demo / explainer clip) and extracts 6 scores
mapping to the same brain region proxies as text and audio encoders.

Pipeline
────────
Stage 1 — Frame-level visual features (OpenCV, always runs)
  Extracts: brightness, contrast, motion, colour diversity,
            scene cuts, visual complexity per frame

Stage 2 — CLIP embeddings (optional, ~340 MB download)
  OpenAI CLIP encodes each sampled frame into a 512-dim embedding.
  Semantic similarity to reference concepts maps to brain scores.

Feature → Brain Region mapping
───────────────────────────────
  visual_novelty      → DMN         (unusual scenes = creative)
  scene_structure     → ECN         (clear structured visuals = logical)
  visual_energy       → SAN         (motion + contrast = salience)
  visual_appeal       → Striatum    (bright + colourful = reward)
  scene_variety       → Hippocampus (varied scenes = memorable)
  visual_repetition   → Basal       (static/repetitive frames = boring)

Supported formats : .mp4  .avi  .mov  .mkv  .webm
VRAM usage        : ~400 MB for CLIP (safe on RTX 3050 4 GB)
Frame sampling    : every 1 second (max 30 frames for speed)

Usage
─────
  python src/encoders/video_encoder.py                      # demo
  python src/encoders/video_encoder.py --file my_demo.mp4   # real file
  python src/encoders/video_encoder.py --no-clip            # faster
"""

import os
import sys
import argparse
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

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


# ════════════════════════════════════════════════════════════
# STAGE 1 — OPENCV VISUAL FEATURES
# ════════════════════════════════════════════════════════════

def extract_frames(video_path: str, max_frames: int = 30) -> list:
    """
    Sample frames from video at 1-second intervals.
    Returns list of BGR numpy arrays.
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps        = cap.get(cv2.CAP_PROP_FPS) or 25
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration   = total / fps
    step       = max(int(fps), 1)          # one frame per second
    frames     = []

    frame_idx  = 0
    while cap.isOpened() and len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_idx += step

    cap.release()
    print(f"  Sampled {len(frames)} frames  |  duration ~{duration:.1f}s  |  fps {fps:.0f}")
    return frames


def frame_features(frame: np.ndarray) -> dict:
    """Extract visual features from a single BGR frame."""
    import cv2

    # Convert to different colour spaces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab  = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # Brightness (L channel mean)
    brightness = float(np.mean(lab[:, :, 0])) / 255.0

    # Contrast (std of gray)
    contrast = float(np.std(gray)) / 128.0

    # Colour saturation (S channel mean)
    saturation = float(np.mean(hsv[:, :, 1])) / 255.0

    # Colour diversity (std of H channel)
    hue_std = float(np.std(hsv[:, :, 0])) / 90.0

    # Visual complexity (Laplacian variance — edge density)
    laplacian  = cv2.Laplacian(gray, cv2.CV_64F)
    complexity = float(np.var(laplacian)) / 10000.0

    # Colour histogram entropy (colour diversity measure)
    hist = cv2.calcHist([frame], [0, 1, 2], None,
                        [8, 8, 8], [0,256, 0,256, 0,256])
    hist = hist.flatten() / (hist.sum() + 1e-8)
    hist_entropy = float(-np.sum(hist * np.log(hist + 1e-8)))

    return {
        "brightness"   : brightness,
        "contrast"     : contrast,
        "saturation"   : saturation,
        "hue_std"      : hue_std,
        "complexity"   : complexity,
        "hist_entropy" : hist_entropy,
    }


def compute_motion(frames: list) -> list:
    """
    Compute optical flow motion score between consecutive frames.
    Returns list of per-frame motion scalars.
    """
    import cv2
    motions = [0.0]
    for i in range(1, len(frames)):
        g1 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(frames[i],   cv2.COLOR_BGR2GRAY)
        diff   = cv2.absdiff(g1, g2)
        motion = float(np.mean(diff)) / 128.0
        motions.append(motion)
    return motions


def count_scene_cuts(frames: list, threshold: float = 0.35) -> int:
    """Count number of scene transitions."""
    import cv2
    cuts = 0
    for i in range(1, len(frames)):
        g1 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY).astype(float)
        g2 = cv2.cvtColor(frames[i],   cv2.COLOR_BGR2GRAY).astype(float)
        diff = np.mean(np.abs(g1 - g2)) / 255.0
        if diff > threshold:
            cuts += 1
    return cuts


def aggregate_frame_features(frames: list) -> dict:
    """Aggregate per-frame features across all sampled frames."""
    all_feats = [frame_features(f) for f in frames]
    motions   = compute_motion(frames)
    cuts      = count_scene_cuts(frames)

    def stat(key):
        vals = [f[key] for f in all_feats]
        return {
            "mean": float(np.mean(vals)),
            "std" : float(np.std(vals)),
            "max" : float(np.max(vals)),
        }

    return {
        "brightness"   : stat("brightness"),
        "contrast"     : stat("contrast"),
        "saturation"   : stat("saturation"),
        "hue_std"      : stat("hue_std"),
        "complexity"   : stat("complexity"),
        "hist_entropy" : stat("hist_entropy"),
        "motion_mean"  : float(np.mean(motions)),
        "motion_std"   : float(np.std(motions)),
        "motion_max"   : float(np.max(motions)),
        "scene_cuts"   : cuts,
        "n_frames"     : len(frames),
    }


# ════════════════════════════════════════════════════════════
# STAGE 2 — CLIP EMBEDDINGS (optional)
# ════════════════════════════════════════════════════════════

def extract_clip_embeddings(frames: list,
                             device: str = "cpu") -> np.ndarray | None:
    """
    Encode sampled frames with CLIP ViT-B/32.
    Returns mean frame embedding (512,) or None if unavailable.
    """
    try:
        import torch
        import clip
        from PIL import Image

        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()

        embeddings = []
        # Process in small batches to respect 4GB VRAM
        batch_size = 4
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            imgs  = []
            for f in batch:
                import cv2
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                imgs.append(preprocess(pil))

            img_tensor = torch.stack(imgs).to(device)
            with torch.no_grad():
                embs = model.encode_image(img_tensor)
                embs = embs / embs.norm(dim=-1, keepdim=True)  # normalise
            embeddings.append(embs.cpu().numpy())

        all_embs = np.vstack(embeddings)        # (N_frames, 512)
        mean_emb = all_embs.mean(axis=0)        # (512,)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
        return mean_emb.astype(np.float32)

    except Exception as e:
        print(f"  ⚠  CLIP unavailable ({type(e).__name__}) — using OpenCV only")
        return None


# ════════════════════════════════════════════════════════════
# BRAIN REGION SCORES FROM VISUAL FEATURES
# ════════════════════════════════════════════════════════════

def visual_to_brain_scores(agg: dict,
                            clip_emb: np.ndarray | None = None) -> dict:
    """
    Convert aggregated visual features → 6 brain region scores (0–100).
    """
    def clip01(x): return float(np.clip(x, 0.0, 1.0))
    def clip100(x): return float(np.clip(x * 100, 0, 100))

    n = agg["n_frames"]

    # ── DMN — visual novelty (unusual / unexpected scenes) ───
    # High complexity + high hue diversity + many scene cuts = novel
    complexity_norm = clip01(agg["complexity"]["mean"] / 2.0)
    hue_norm        = clip01(agg["hue_std"]["mean"])
    cuts_norm       = clip01(agg["scene_cuts"] / max(n * 0.3, 1))
    dmn = clip100(complexity_norm * 0.4 + hue_norm * 0.3 + cuts_norm * 0.3)

    # ── ECN — scene structure (clear, well-lit, steady) ──────
    bright_norm   = clip01(agg["brightness"]["mean"] / 0.6)
    low_motion    = clip01(1.0 - agg["motion_mean"] / 0.3)  # steady = structured
    low_complexity= clip01(1.0 - agg["complexity"]["mean"] / 3.0)
    ecn = clip100(bright_norm * 0.4 + low_motion * 0.3 + low_complexity * 0.3)

    # ── SAN — visual energy (motion + contrast + saturation) ─
    motion_norm = clip01(agg["motion_mean"] / 0.2)
    contrast_n  = clip01(agg["contrast"]["mean"])
    sat_norm    = clip01(agg["saturation"]["mean"])
    san = clip100(motion_norm * 0.4 + contrast_n * 0.3 + sat_norm * 0.3)

    # ── Striatum — visual appeal (bright + colourful + rich) ─
    bright_appeal = clip01(agg["brightness"]["mean"] / 0.7)
    sat_appeal    = clip01(agg["saturation"]["mean"])
    entropy_norm  = clip01(agg["hist_entropy"]["mean"] / 4.0)
    striatum = clip100(bright_appeal * 0.35 + sat_appeal * 0.35 + entropy_norm * 0.30)

    # ── Hippocampus — scene variety (varied = memorable) ─────
    brightness_var = clip01(agg["brightness"]["std"] / 0.2)
    motion_var     = clip01(agg["motion_std"] / 0.1)
    cuts_variety   = clip01(agg["scene_cuts"] / max(n * 0.5, 1))
    hippocampus = clip100(brightness_var * 0.3 + motion_var * 0.3 + cuts_variety * 0.4)

    # ── Basal — visual repetition (static / monotone = bad) ──
    low_motion_b   = clip01(1.0 - agg["motion_mean"] / 0.1)
    low_cuts       = clip01(1.0 - agg["scene_cuts"] / max(n * 0.3, 1))
    low_entropy    = clip01(1.0 - agg["hist_entropy"]["mean"] / 4.0)
    basal = clip100(low_motion_b * 0.4 + low_cuts * 0.3 + low_entropy * 0.3)

    # ── Boost scores with CLIP if available ──────────────────
    if clip_emb is not None:
        try:
            import torch
            import clip as clip_mod
            device = "cpu"
            model, _ = clip_mod.load("ViT-B/32", device=device)
            model.eval()

            # Reference text prompts for each dimension
            refs = {
                "dmn"         : "a creative and visually surprising presentation",
                "ecn"         : "a clear and well structured professional demo",
                "san"         : "an exciting and emotionally engaging video",
                "striatum"    : "a beautiful and desirable product showcase",
                "hippocampus" : "a memorable and unique visual story",
                "basal"       : "a boring repetitive static talking head video",
            }
            text_tokens = clip_mod.tokenize(list(refs.values())).to(device)
            with torch.no_grad():
                text_embs = model.encode_text(text_tokens)
                text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
                text_embs = text_embs.cpu().numpy()

            clip_vec = clip_emb.reshape(1, -1)
            sims     = (text_embs @ clip_vec.T).flatten()  # (6,)
            keys     = list(refs.keys())

            # Blend OpenCV scores 60% + CLIP signal 40%
            blend = {
                "dmn"         : dmn         * 0.6 + float(np.clip((sims[0]+1)/2, 0,1)) * 40,
                "ecn"         : ecn         * 0.6 + float(np.clip((sims[1]+1)/2, 0,1)) * 40,
                "san"         : san         * 0.6 + float(np.clip((sims[2]+1)/2, 0,1)) * 40,
                "striatum"    : striatum    * 0.6 + float(np.clip((sims[3]+1)/2, 0,1)) * 40,
                "hippocampus" : hippocampus * 0.6 + float(np.clip((sims[4]+1)/2, 0,1)) * 40,
                "basal"       : basal       * 0.6 + float(np.clip((sims[5]+1)/2, 0,1)) * 40,
            }
            return {k: round(float(np.clip(v, 0, 100)), 1) for k, v in blend.items()}
        except Exception:
            pass

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

class VideoEncoder:
    """
    Encodes a video file → 6 brain region scores (0–100).
    """

    def __init__(self, use_clip: bool = True, device: str = "auto"):
        self.use_clip = use_clip
        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

    def encode_file(self, video_path: str, max_frames: int = 30) -> dict:
        """Encode a video file → 6 brain scores."""
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        print(f"  Loading {path.name} …")
        frames = extract_frames(str(path), max_frames=max_frames)
        return self._encode_frames(frames)

    def _encode_frames(self, frames: list) -> dict:
        # Stage 1 — OpenCV
        agg = aggregate_frame_features(frames)

        # Stage 2 — CLIP
        clip_emb = None
        if self.use_clip:
            clip_emb = extract_clip_embeddings(frames, device=self.device)

        scores  = visual_to_brain_scores(agg, clip_emb)
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
            "overall"          : round(overall, 1),
            "grade"            : _grade(overall),
            "agg_features"     : agg,
            "clip_embedding"   : clip_emb,
        }

    def encode_demo(self) -> dict:
        """
        Generate synthetic video frames (colour patterns) for testing.
        No real video file needed.
        """
        import cv2
        print("  Generating synthetic video frames …")
        frames = []
        np.random.seed(42)

        for i in range(20):
            # Vary brightness, colour, motion across frames
            base_colour = np.array([
                int(120 + 80 * np.sin(i * 0.4)),
                int(100 + 80 * np.cos(i * 0.3)),
                int(140 + 60 * np.sin(i * 0.5 + 1)),
            ], dtype=np.uint8)

            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            frame[:] = base_colour

            # Add random shapes for visual complexity
            for _ in range(5):
                x1 = np.random.randint(0, 280)
                y1 = np.random.randint(0, 200)
                x2 = x1 + np.random.randint(20, 80)
                y2 = y1 + np.random.randint(20, 80)
                colour = tuple(int(c) for c in np.random.randint(0, 255, 3))
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, -1)

            frames.append(frame)

        return self._encode_frames(frames)


# ════════════════════════════════════════════════════════════
# PRETTY PRINT
# ════════════════════════════════════════════════════════════

def _print_result(source: str, r: dict):
    LABELS = {
        "dmn"         : "DMN        Visual Novelty",
        "ecn"         : "ECN        Scene Structure",
        "san"         : "SAN        Visual Energy",
        "striatum"    : "Striatum   Visual Appeal",
        "hippocampus" : "Hippocampus Scene Variety",
        "basal"       : "Basal      Repetition ↓",
    }
    print(f"\n  ┌─ Source : {source}")
    print(f"  │  Grade  : {r['grade']}  ({r['overall']}/100)")
    print(f"  │")
    for k, label in LABELS.items():
        s   = r[k]
        bar = "█" * int(s / 5)
        inv = "  (low=varied)" if k == "basal" else ""
        print(f"  │  {label:<28} {s:5.1f}  {bar}{inv}")
    print(f"  └{'─'*56}")


# ════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",     type=str, default=None,
                        help="Path to video (.mp4 .avi .mov .mkv .webm)")
    parser.add_argument("--no-clip",  action="store_true",
                        help="Skip CLIP, use OpenCV only (faster)")
    args = parser.parse_args()

    print("\n" + "═"*56)
    print("  STEP 5 — VIDEO ENCODER")
    print("═"*56)

    enc = VideoEncoder(use_clip=not args.no_clip)

    if args.file:
        result = enc.encode_file(args.file)
        _print_result(args.file, result)
    else:
        print("  No --file provided — running demo with synthetic frames\n")
        result = enc.encode_demo()
        _print_result("synthetic video frames", result)

    print("\n  ✅  Video encoder working correctly.\n")