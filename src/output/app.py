"""
src/output/app.py
══════════════════════════════════════════════════════════════
Step 9 — Streamlit UI

Full brain activation analysis app with:
  - Text / Audio / Video input
  - Animated brain map (SVG, regions glow by score)
  - Per-region score breakdown
  - Benchmark comparison vs viral ideas
  - Mistral AI improvement summary

Run inside Docker:
  streamlit run src/output/app.py --server.port=8501 --server.address=0.0.0.0

Then open: http://localhost:8501
"""

import os
import sys
import tempfile
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import streamlit as st

# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Neuro Idea Grader",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background: #0e0e14; color: #e8e8f0; }
  .block-container { padding: 2rem 2rem 2rem 2rem; }

  .grade-badge {
    font-size: 3.5rem; font-weight: 700;
    text-align: center; padding: 0.5rem 1.5rem;
    border-radius: 16px; display: inline-block;
  }
  .grade-A  { background: #1a3a2a; color: #4ade80; border: 1px solid #4ade80; }
  .grade-B  { background: #1a2e3a; color: #60a5fa; border: 1px solid #60a5fa; }
  .grade-C  { background: #2e2a1a; color: #fbbf24; border: 1px solid #fbbf24; }
  .grade-D  { background: #2e1e1a; color: #f87171; border: 1px solid #f87171; }
  .grade-F  { background: #2e1a1a; color: #ef4444; border: 1px solid #ef4444; }

  .region-card {
    border-radius: 12px; padding: 12px 16px; margin: 6px 0;
    border: 1px solid rgba(255,255,255,0.08);
  }
  .score-label { font-size: 0.75rem; color: #9ca3af; margin-bottom: 4px; }
  .score-value { font-size: 1.4rem; font-weight: 600; }

  .benchmark-card {
    border-radius: 10px; padding: 10px 14px; margin: 6px 0;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
  }
  .mistral-box {
    background: rgba(139,92,246,0.08);
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 12px; padding: 16px 20px; margin-top: 16px;
  }
  .section-header {
    font-size: 0.8rem; font-weight: 600; letter-spacing: 0.1em;
    color: #6b7280; text-transform: uppercase; margin: 1.2rem 0 0.5rem;
  }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# BRAIN MAP SVG
# ════════════════════════════════════════════════════════════

def brain_map_svg(scores: dict) -> str:
    """
    Generate an SVG brain map where each region glows
    proportionally to its activation score.
    """
    def alpha(score, max_a=0.85, min_a=0.08):
        return min_a + (max_a - min_a) * (score / 100)

    def glow(score, color):
        a = alpha(score)
        return f"rgba({color},{a:.2f})"

    dmn  = scores.get("dmn", 50)
    ecn  = scores.get("ecn", 50)
    san  = scores.get("san", 50)
    stri = scores.get("striatum", 50)
    hipp = scores.get("hippocampus", 50)
    basa = scores.get("basal", 50)

    # Invert basal for display (high repetition = dim, not glowing)
    basa_display = 100 - basa

    svg = f"""
<svg viewBox="0 0 480 380" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:480px;background:#0e0e14;border-radius:16px">
  <defs>
    <radialGradient id="gDMN"  cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="rgba(167,139,250,{alpha(dmn):.2f})"/>
      <stop offset="100%" stop-color="rgba(167,139,250,0)"/>
    </radialGradient>
    <radialGradient id="gECN"  cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="rgba(52,211,153,{alpha(ecn):.2f})"/>
      <stop offset="100%" stop-color="rgba(52,211,153,0)"/>
    </radialGradient>
    <radialGradient id="gSAN"  cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="rgba(251,191,36,{alpha(san):.2f})"/>
      <stop offset="100%" stop-color="rgba(251,191,36,0)"/>
    </radialGradient>
    <radialGradient id="gSTRI" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="rgba(248,113,113,{alpha(stri):.2f})"/>
      <stop offset="100%" stop-color="rgba(248,113,113,0)"/>
    </radialGradient>
    <radialGradient id="gHIPP" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="rgba(96,165,250,{alpha(hipp):.2f})"/>
      <stop offset="100%" stop-color="rgba(96,165,250,0)"/>
    </radialGradient>
    <radialGradient id="gBASA" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="rgba(156,163,175,{alpha(basa_display):.2f})"/>
      <stop offset="100%" stop-color="rgba(156,163,175,0)"/>
    </radialGradient>
  </defs>

  <!-- Brain outline -->
  <ellipse cx="240" cy="185" rx="175" ry="148"
           fill="none" stroke="rgba(255,255,255,0.12)" stroke-width="1.5"/>
  <!-- Midline -->
  <line x1="240" y1="37" x2="240" y2="333"
        stroke="rgba(255,255,255,0.06)" stroke-width="1" stroke-dasharray="4 3"/>

  <!-- DMN glow — top centre (Default Mode, creativity) -->
  <ellipse cx="215" cy="95"  rx="70" ry="44" fill="url(#gDMN)"/>
  <ellipse cx="215" cy="95"  rx="52" ry="32"
           fill="none" stroke="rgba(167,139,250,0.5)" stroke-width="0.8"/>
  <text x="215" y="91"  text-anchor="middle"
        font-size="10" font-weight="600" fill="rgba(167,139,250,0.9)">DMN</text>
  <text x="215" y="104" text-anchor="middle"
        font-size="9"  fill="rgba(167,139,250,0.7)">{dmn:.0f}</text>

  <!-- ECN glow — top right (Executive, logic) -->
  <ellipse cx="328" cy="128" rx="58" ry="36" fill="url(#gECN)"/>
  <ellipse cx="328" cy="128" rx="44" ry="28"
           fill="none" stroke="rgba(52,211,153,0.5)" stroke-width="0.8"/>
  <text x="328" y="124" text-anchor="middle"
        font-size="10" font-weight="600" fill="rgba(52,211,153,0.9)">ECN</text>
  <text x="328" y="137" text-anchor="middle"
        font-size="9"  fill="rgba(52,211,153,0.7)">{ecn:.0f}</text>

  <!-- SAN glow — centre (Salience, emotion) -->
  <ellipse cx="240" cy="188" rx="52" ry="34" fill="url(#gSAN)"/>
  <ellipse cx="240" cy="188" rx="40" ry="26"
           fill="none" stroke="rgba(251,191,36,0.5)" stroke-width="0.8"/>
  <text x="240" y="184" text-anchor="middle"
        font-size="10" font-weight="600" fill="rgba(251,191,36,0.9)">SAN</text>
  <text x="240" y="197" text-anchor="middle"
        font-size="9"  fill="rgba(251,191,36,0.7)">{san:.0f}</text>

  <!-- Striatum glow — lower centre (Reward) -->
  <ellipse cx="215" cy="260" rx="62" ry="36" fill="url(#gSTRI)"/>
  <ellipse cx="215" cy="260" rx="48" ry="28"
           fill="none" stroke="rgba(248,113,113,0.5)" stroke-width="0.8"/>
  <text x="215" y="256" text-anchor="middle"
        font-size="10" font-weight="600" fill="rgba(248,113,113,0.9)">Striatum</text>
  <text x="215" y="269" text-anchor="middle"
        font-size="9"  fill="rgba(248,113,113,0.7)">{stri:.0f}</text>

  <!-- Hippocampus glow — lower right (Memory) -->
  <ellipse cx="328" cy="232" rx="56" ry="32" fill="url(#gHIPP)"/>
  <ellipse cx="328" cy="232" rx="44" ry="26"
           fill="none" stroke="rgba(96,165,250,0.5)" stroke-width="0.8"/>
  <text x="328" y="228" text-anchor="middle"
        font-size="10" font-weight="600" fill="rgba(96,165,250,0.9)">Hippocampus</text>
  <text x="328" y="241" text-anchor="middle"
        font-size="9"  fill="rgba(96,165,250,0.7)">{hipp:.0f}</text>

  <!-- Basal Ganglia glow — left (Habit/Repetition) -->
  <ellipse cx="148" cy="196" rx="56" ry="32" fill="url(#gBASA)"/>
  <ellipse cx="148" cy="196" rx="44" ry="26"
           fill="none" stroke="rgba(156,163,175,0.5)" stroke-width="0.8"/>
  <text x="148" y="192" text-anchor="middle"
        font-size="10" font-weight="600" fill="rgba(156,163,175,0.9)">Basal</text>
  <text x="148" y="205" text-anchor="middle"
        font-size="9"  fill="rgba(156,163,175,0.7)">{basa:.0f}</text>

  <!-- Labels -->
  <text x="65"  y="195" text-anchor="middle"
        font-size="9" fill="rgba(255,255,255,0.25)">LEFT</text>
  <text x="415" y="195" text-anchor="middle"
        font-size="9" fill="rgba(255,255,255,0.25)">RIGHT</text>
  <text x="240" y="360" text-anchor="middle"
        font-size="8" fill="rgba(255,255,255,0.2)">
    brighter glow = higher activation
  </text>
</svg>"""
    return svg


# ════════════════════════════════════════════════════════════
# GRADE BADGE HTML
# ════════════════════════════════════════════════════════════

def grade_badge(grade: str, overall: float) -> str:
    cls = "grade-A" if grade.startswith("A") else \
          "grade-B" if grade.startswith("B") else \
          "grade-C" if grade.startswith("C") else \
          "grade-D" if grade.startswith("D") else "grade-F"
    return f"""
<div style="text-align:center;padding:1rem 0">
  <div class="grade-badge {cls}">{grade}</div>
  <div style="font-size:1rem;color:#9ca3af;margin-top:6px">
    Overall Virality Score: <b style="color:#e8e8f0">{overall:.1f} / 100</b>
  </div>
</div>"""


# ════════════════════════════════════════════════════════════
# REGION SCORE BARS
# ════════════════════════════════════════════════════════════

REGION_CONFIG = {
    "dmn"         : ("🟣", "Creativity / Novelty",    "#a78bfa"),
    "ecn"         : ("🟢", "Feasibility / Logic",     "#34d399"),
    "san"         : ("🟡", "Emotional Hook",           "#fbbf24"),
    "striatum"    : ("🔴", "Dopamine / Want",          "#f87171"),
    "hippocampus" : ("🔵", "Memorability",             "#60a5fa"),
    "basal"       : ("⚪", "Repetitiveness",           "#9ca3af"),
}

def score_bars_html(scores: dict, weakest: str, strongest: str) -> str:
    html = ""
    for region, (icon, label, color) in REGION_CONFIG.items():
        s       = scores.get(region, 0)
        pct     = s
        tag     = ""
        if region == weakest:   tag = " 🎯 weakest"
        if region == strongest: tag = " ⭐ strongest"
        inv_note = " (low = original)" if region == "basal" else ""

        html += f"""
<div class="region-card" style="background:rgba({_hex_to_rgb(color)},0.06)">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
    <span style="font-size:0.85rem;color:#e8e8f0">{icon} {label}{inv_note}</span>
    <span class="score-value" style="color:{color}">{s:.1f}<span style="font-size:0.75rem;color:#6b7280">/100</span></span>
  </div>
  <div style="background:rgba(255,255,255,0.07);border-radius:4px;height:6px">
    <div style="width:{pct}%;background:{color};height:6px;border-radius:4px;
                transition:width 0.6s ease"></div>
  </div>
  {f'<div style="font-size:0.7rem;color:#6b7280;margin-top:3px">{tag}</div>' if tag else ''}
</div>"""
    return html


def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"{r},{g},{b}"


# ════════════════════════════════════════════════════════════
# MISTRAL IMPROVEMENT SUMMARY
# ════════════════════════════════════════════════════════════

def get_mistral_summary(scores: dict, idea_text: str,
                         weakest: str) -> str:
    api_key = os.getenv("MISTRAL_API_KEY","").strip()
    if not api_key:
        return ("_Mistral API key not set. Add `MISTRAL_API_KEY=...` to your "
                "`.env` file to enable AI-powered improvement suggestions._")

    score_lines = "\n".join(
        f"  - {REGION_CONFIG[r][1]}: {scores.get(r,0):.1f}/100"
        for r in REGION_CONFIG
    )
    prompt = f"""You are a neuroscience-inspired idea coach.

A user submitted an idea and received these brain activation scores (0–100):
{score_lines}

Idea: {idea_text[:500]}

The weakest brain region is: {REGION_CONFIG[weakest][1]} ({scores.get(weakest,0):.1f}/100)
Note: For Repetitiveness (Basal Ganglia), a LOW score is good (idea is original).

Give exactly 3 short, specific, actionable improvement suggestions.
Focus on the weakest region first.
Format as a numbered list. Be direct and practical. No fluff."""

    try:
        from mistralai import Mistral
        with Mistral(api_key=api_key) as client:
            res = client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role":"user","content":prompt}],
                stream=False,
            )
            return res.choices[0].message.content.strip()
    except Exception as e:
        return f"Mistral error: {e}"


# ════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════

def run_analysis(idea_text: str,
                 audio_path: str | None,
                 video_path: str | None) -> dict:
    """Run full pipeline and return result dict."""
    from src.fusion.late_fusion import LateFusion
    from src.scoring.brain_scorer import BrainScorer

    fuser  = LateFusion()
    scorer = BrainScorer(use_bridge_model=False)

    with st.spinner("Running encoders…"):
        fused = fuser.fuse_from_encoders(
            idea_text  = idea_text  or None,
            audio_path = audio_path or None,
            video_path = video_path or None,
            use_wav2vec2=False,
            use_clip    =False,
        )

    return scorer.score(fused)


def main():
    # ── Sidebar ─────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🧠 Neuro Idea Grader")
        st.markdown("_Neuroscience-inspired virality prediction_")
        st.divider()
        st.markdown("**How it works**")
        st.markdown("""
1. Submit your idea as text, audio, or video
2. Three encoders extract brain-relevant features
3. Late fusion combines all modalities
4. 6 brain regions are scored 0–100
5. Mistral AI generates improvement tips
        """)
        st.divider()
        st.markdown("**Brain Regions**")
        for r, (icon, label, color) in REGION_CONFIG.items():
            st.markdown(f"{icon} **{label}**")

    # ── Header ───────────────────────────────────────────────
    st.markdown("# 🧠 Neuro Idea Grader")
    st.markdown("*Submit your idea — see which brain regions it activates*")
    st.divider()

    # ── Input section ────────────────────────────────────────
    col_in1, col_in2 = st.columns([2, 1])

    with col_in1:
        st.markdown('<div class="section-header">📝 Idea Description (required)</div>',
                    unsafe_allow_html=True)
        idea_text = st.text_area(
            label="idea_text",
            placeholder="Describe your idea in 1–5 sentences. Be specific about "
                        "the problem, the solution, and who it's for.",
            height=140,
            label_visibility="collapsed",
        )

    with col_in2:
        st.markdown('<div class="section-header">🎙 Audio Pitch (optional)</div>',
                    unsafe_allow_html=True)
        audio_file = st.file_uploader(
            "Upload a voice recording of your pitch",
            type=["wav","mp3","m4a","ogg","flac"],
            label_visibility="collapsed",
        )
        st.markdown('<div class="section-header">🎬 Video Demo (optional)</div>',
                    unsafe_allow_html=True)
        video_file = st.file_uploader(
            "Upload a short demo or explainer video",
            type=["mp4","avi","mov","mkv","webm"],
            label_visibility="collapsed",
        )

    # ── Analyse button ───────────────────────────────────────
    st.divider()
    analyse_btn = st.button("🔬 Analyse Brain Activation", type="primary",
                             use_container_width=True,
                             disabled=not idea_text.strip())

    if not idea_text.strip():
        st.info("Enter an idea description above to get started.")
        return

    if not analyse_btn:
        return

    # ── Save uploaded files to temp ──────────────────────────
    audio_path = None
    video_path = None
    tmp_files  = []

    if audio_file:
        suffix = Path(audio_file.name).suffix
        tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(audio_file.read()); tmp.close()
        audio_path = tmp.name
        tmp_files.append(tmp.name)

    if video_file:
        suffix = Path(video_file.name).suffix
        tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(video_file.read()); tmp.close()
        video_path = tmp.name
        tmp_files.append(tmp.name)

    # ── Run analysis ─────────────────────────────────────────
    result = run_analysis(idea_text, audio_path, video_path)

    # Cleanup temp files
    for f in tmp_files:
        try: os.unlink(f)
        except: pass

    scores   = result["scores"]
    overall  = result["overall"]
    grade    = result["grade"]
    weakest  = result["weakest"]
    strongest= result["strongest"]

    # ── Results layout ───────────────────────────────────────
    st.divider()
    st.markdown("## Results")

    col_brain, col_scores, col_right = st.columns([1.2, 1.2, 1.0])

    # ── Left: Brain map ──────────────────────────────────────
    with col_brain:
        st.markdown('<div class="section-header">🧠 Brain Activation Map</div>',
                    unsafe_allow_html=True)
        st.markdown(brain_map_svg(scores), unsafe_allow_html=True)
        st.markdown(grade_badge(grade, overall), unsafe_allow_html=True)

    # ── Centre: Score bars ───────────────────────────────────
    with col_scores:
        st.markdown('<div class="section-header">📊 Region Breakdown</div>',
                    unsafe_allow_html=True)
        st.markdown(score_bars_html(scores, weakest, strongest),
                    unsafe_allow_html=True)

        # Modalities used
        mods = result.get("modalities", ["text"])
        mod_icons = {"text":"📝","audio":"🎙","video":"🎬"}
        mod_str   = " + ".join(mod_icons.get(m,m)+" "+m for m in mods)
        st.markdown(f"<div style='font-size:0.75rem;color:#6b7280;margin-top:8px'>"
                    f"Modalities: {mod_str}</div>", unsafe_allow_html=True)

    # ── Right: Benchmarks + Interpretation ───────────────────
    with col_right:
        st.markdown('<div class="section-header">🏆 Closest Benchmarks</div>',
                    unsafe_allow_html=True)
        for b in result["benchmarks"]:
            outcome_color = "#4ade80" if "B" in b["outcome"] or "$" in b["outcome"] else "#f87171"
            st.markdown(f"""
<div class="benchmark-card">
  <div style="font-size:0.85rem;font-weight:600;color:#e8e8f0">{b['name']}</div>
  <div style="font-size:0.72rem;color:#9ca3af;margin:2px 0">{b['tagline']}</div>
  <div style="display:flex;justify-content:space-between;margin-top:4px">
    <span style="font-size:0.72rem;color:{outcome_color}">{b['outcome']}</span>
    <span style="font-size:0.72rem;color:#6b7280">sim {b['similarity']:.0f}%</span>
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">💡 Interpretation</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
<div style="font-size:0.82rem;color:#d1d5db;line-height:1.6;
            background:rgba(255,255,255,0.03);border-radius:8px;padding:10px 12px">
{result['interpretation']}
</div>""", unsafe_allow_html=True)

    # ── Mistral improvement summary ──────────────────────────
    st.divider()
    st.markdown("### 💬 AI Improvement Summary")

    with st.spinner("Asking Mistral AI for improvement tips…"):
        summary = get_mistral_summary(scores, idea_text, weakest)

    st.markdown(f'<div class="mistral-box">{summary}</div>',
                unsafe_allow_html=True)

    # ── Raw scores expander ──────────────────────────────────
    with st.expander("🔢 Raw Scores (JSON)"):
        import json
        st.code(json.dumps({
            "scores"     : scores,
            "overall"    : overall,
            "grade"      : grade,
            "modalities" : result.get("modalities"),
            "benchmarks" : result["benchmarks"],
        }, indent=2))


if __name__ == "__main__":
    main()