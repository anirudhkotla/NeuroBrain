"""
src/eeg/preprocess.py
══════════════════════════════════════════════════════════════
Reads Kaggle emotions.csv + Zenodo CSVs.
Extracts 6 brain-region proxy scores per sample.
Saves .npy files to data/processed/.

Kaggle emotions.csv  (2132 rows × 2549 cols)
  Columns: mean_0..N, std_0..N, delta_0..N, theta_0..N,
           alpha_0..N, beta_0..N, gamma_0..N, label
  Label  : POSITIVE | NEUTRAL | NEGATIVE

Zenodo CSVs
  Various layouts — auto-detected

Emotiv EPOC+ 14-channel order (both datasets):
  idx: 0=AF3  1=F7  2=F3  3=FC5  4=T7  5=P7  6=O1
       7=O2   8=P8  9=T8  10=FC6 11=F4 12=F8  13=AF4
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parents[2]
K_DIR   = ROOT / "data" / "raw"  / "kaggle"
K2_DIR  = ROOT / "data" / "raw"  / "kaggle"   # second Kaggle dataset lives in same dir
OUT     = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# Emotiv channel indices
F3, F4, T7, T8, P7, P8 = 2, 11, 4, 9, 5, 8

VALENCE = {
    "POSITIVE":0.82,"NEUTRAL":0.50,"NEGATIVE":0.20,
    "positive":0.82,"neutral":0.50,"negative":0.20,
    "happy":0.85,"sad":0.22,"fear":0.18,"calm":0.55,
}
AROUSAL = {
    "POSITIVE":0.72,"NEUTRAL":0.45,"NEGATIVE":0.62,
    "positive":0.72,"neutral":0.45,"negative":0.62,
    "happy":0.80,"sad":0.35,"fear":0.78,"calm":0.28,
}


# ── PROXY CALCULATIONS ──────────────────────────────────────

def proxies_from_bands(row, n=14):
    def g(prefix, idx):
        c = f"{prefix}{idx}"
        return float(row[c]) if c in row.index else 0.0

    aF3 = g("alpha_",F3)+1e-10; aF4 = g("alpha_",F4)+1e-10
    bF3 = g("beta_", F3);       bF4 = g("beta_", F4)
    gF3 = g("gamma_",F3);       gF4 = g("gamma_",F4)
    tT7 = g("theta_",T7);       tT8 = g("theta_",T8)
    dP7 = g("delta_",P7);       dP8 = g("delta_",P8)
    tP7 = g("theta_",P7);       tP8 = g("theta_",P8)

    return dict(
        dmn  = float(np.log(aF4) - np.log(aF3)),
        ecn  = float(((bF3+bF4)/2) / ((aF3+aF4)/2+1e-10)),
        san  = float((gF3+gF4)/2),
        hipp = float((tT7+tT8)/2),
        basal= float((dP7+dP8+tP7+tP8)/4),
    )


def proxies_from_means(vals, n=14):
    f3v=vals[F3] if len(vals)>F3 else 0.0
    f4v=vals[F4] if len(vals)>F4 else 0.0
    t7v=vals[T7] if len(vals)>T7 else 0.0
    t8v=vals[T8] if len(vals)>T8 else 0.0
    p7v=vals[P7] if len(vals)>P7 else 0.0
    p8v=vals[P8] if len(vals)>P8 else 0.0
    return dict(
        dmn  = float(f4v-f3v),
        ecn  = float(abs(f3v+f4v)/(abs(t7v+t8v)+1e-10)),
        san  = float(np.mean(np.abs(vals))),
        hipp = float((t7v+t8v)/2),
        basal= float((p7v+p8v)/2),
    )


def striatum(val, aro): return float(0.5*val + 0.5*aro)


def _label_col(df):
    for c in ["label","emotion","class","category","condition","state"]:
        m = [x for x in df.columns if x.lower()==c]
        if m: return m[0]
    for c in df.columns:
        if df[c].dtype == object: return c
    return None


def _has(df, prefix):
    return any(c.startswith(prefix) for c in df.columns)


# ── KAGGLE ──────────────────────────────────────────────────

def process_kaggle():
    csv = K_DIR / "emotions.csv"
    if not csv.exists():
        print(f"  ⚠  Kaggle CSV not found — run dataset_loader.py --source kaggle")
        return None, None, None

    print(f"\n  ── Kaggle ──────────────────────────────")
    df = pd.read_csv(csv)
    print(f"  Shape  : {df.shape}")

    lc     = _label_col(df)
    labels = df[lc].astype(str).str.strip().str.upper()
    fcols  = [c for c in df.columns if c != lc]
    X      = df[fcols].select_dtypes(include=[np.number]).fillna(0).values.astype(np.float32)
    use_b  = _has(df,"delta_") and _has(df,"alpha_")

    print(f"  Label  : '{lc}'  →  {labels.value_counts().to_dict()}")
    print(f"  Feats  : {X.shape[1]}  |  band cols: {use_b}")

    rows = []
    for i in range(len(df)):
        row  = df.iloc[i]
        lbl  = labels.iloc[i]
        val_ = VALENCE.get(lbl, 0.5)
        aro_ = AROUSAL.get(lbl, 0.5)
        p    = proxies_from_bands(row) if use_b else proxies_from_means(X[i])
        p["striatum"] = striatum(val_, aro_)
        rows.append([p["dmn"],p["ecn"],p["san"],p["striatum"],p["hipp"],p["basal"]])

    y_v  = np.array([VALENCE.get(l,0.5) for l in labels], dtype=np.float32)
    y_a  = np.array([AROUSAL.get(l,0.5) for l in labels], dtype=np.float32)
    le   = LabelEncoder()
    y_c  = le.fit_transform(labels).astype(np.float32)

    print(f"  ✓ {len(rows)} samples")
    return X, np.array(rows,dtype=np.float32), np.column_stack([y_v,y_a,y_c]).astype(np.float32)


# ── ZENODO ──────────────────────────────────────────────────

def process_mental_state():
    """
    Kaggle Dataset 2: EEG Brainwave Mental State
    Labels: RELAXED | NEUTRAL | CONCENTRATING
    Same 14-channel Emotiv EPOC+ format as emotions.csv
    """
    # Try both possible filenames
    csv = K2_DIR / "mental_states.csv"
    if not csv.exists():
        csv = K2_DIR / "mental_state.csv"
    if not csv.exists():
        # Try any CSV that isn't emotions.csv
        others = [f for f in K2_DIR.glob("*.csv") if f.name != "emotions.csv"]
        if others:
            csv = others[0]
        else:
            print(f"  ⚠  Mental State CSV not found — run dataset_loader.py --source both")
            return None, None, None

    print(f"\n  ── Kaggle Mental State ─────────────────")
    df = pd.read_csv(csv)
    print(f"  File   : {csv.name}")
    print(f"  Shape  : {df.shape}")

    lc     = _label_col(df)
    if not lc:
        df["_lbl"] = "neutral"; lc = "_lbl"

    labels = df[lc].astype(str).str.strip().str.upper()

    # Map mental states to valence/arousal
    VALENCE_MS = {"RELAXED":0.70,"NEUTRAL":0.50,"CONCENTRATING":0.60}
    AROUSAL_MS = {"RELAXED":0.25,"NEUTRAL":0.45,"CONCENTRATING":0.80}

    fcols = [c for c in df.columns if c != lc]
    X     = df[fcols].select_dtypes(include=[np.number]).fillna(0).values.astype(np.float32)
    use_b = _has(df,"delta_") and _has(df,"alpha_")

    print(f"  Label  : '{lc}'  →  {labels.value_counts().to_dict()}")
    print(f"  Feats  : {X.shape[1]}  |  band cols: {use_b}")

    rows = []
    for i in range(len(df)):
        row  = df.iloc[i]
        lbl  = labels.iloc[i]
        val_ = VALENCE_MS.get(lbl, VALENCE.get(lbl.lower(), 0.5))
        aro_ = AROUSAL_MS.get(lbl, AROUSAL.get(lbl.lower(), 0.5))
        p    = proxies_from_bands(row) if use_b else proxies_from_means(X[i])
        p["striatum"] = striatum(val_, aro_)
        rows.append([p["dmn"],p["ecn"],p["san"],p["striatum"],p["hipp"],p["basal"]])

    y_v = np.array([VALENCE_MS.get(l, 0.5) for l in labels], dtype=np.float32)
    y_a = np.array([AROUSAL_MS.get(l, 0.5) for l in labels], dtype=np.float32)
    le  = LabelEncoder()
    y_c = le.fit_transform(labels).astype(np.float32)

    print(f"  ✓ {len(rows)} samples")
    return X, np.array(rows,dtype=np.float32), np.column_stack([y_v,y_a,y_c]).astype(np.float32)


# ── MERGE + SAVE ────────────────────────────────────────────

def build():
    print("\n" + "═"*56)
    print("  EEG PREPROCESSING PIPELINE")
    print("═"*56)

    parts = []
    for fn, name in [(process_kaggle,"kaggle_emotions"),(process_mental_state,"kaggle_mental")]:
        X, B, L = fn()
        if X is not None: parts.append((X,B,L,name))

    if not parts:
        print("\n  ❌  No data — run dataset_loader.py first"); return

    mx = max(r[0].shape[1] for r in parts)
    X  = np.vstack([np.pad(r[0],((0,0),(0,mx-r[0].shape[1]))) for r in parts])
    B  = np.vstack([r[1] for r in parts])
    L  = np.vstack([r[2] for r in parts])

    scaler = MinMaxScaler(feature_range=(0,100))
    Bs     = scaler.fit_transform(B).astype(np.float32)

    np.save(OUT/"X_features.npy",    X)
    np.save(OUT/"y_brain_scores.npy", Bs)
    np.save(OUT/"y_labels.npy",       L)
    (OUT/"meta.json").write_text(json.dumps({
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "sources": [r[3] for r in parts],
        "brain_region_order": ["dmn","ecn","san","striatum","hippocampus","basal"],
        "label_order": ["valence","arousal","emotion_class"],
    }, indent=2))

    print(f"\n  ✅  Saved to {OUT}/")
    print(f"     X_features.npy      {X.shape}")
    print(f"     y_brain_scores.npy  {Bs.shape}  (0–100)")
    print(f"     y_labels.npy        {L.shape}")

    NAMES = ["DMN (creativity)","ECN (feasibility)","SAN (emotion)",
             "Striatum (reward)","Hippocampus (memory)","Basal (pattern)"]
    idx = np.random.randint(len(Bs))
    print(f"\n  Sample row #{idx}:")
    for n,s in zip(NAMES,Bs[idx]):
        s = 0.0 if np.isnan(s) else float(s)
        print(f"    {n:<26} {s:5.1f}  {'█'*int(s/5)}")
    return X, Bs, L


# ── DEMO ────────────────────────────────────────────────────

def demo():
    print("═"*56+"\n  DEMO MODE\n"+"═"*56)
    np.random.seed(42)

    # Kaggle-style
    cols = ([f"mean_{i}"  for i in range(14)] +
            [f"delta_{i}" for i in range(14)] +
            [f"theta_{i}" for i in range(14)] +
            [f"alpha_{i}" for i in range(14)] +
            [f"beta_{i}"  for i in range(14)] +
            [f"gamma_{i}" for i in range(14)])
    df_k = pd.DataFrame(np.random.randn(120,len(cols))*5, columns=cols)
    df_k["label"] = np.random.choice(["POSITIVE","NEUTRAL","NEGATIVE"],120)
    df_k.to_csv(K_DIR/"emotions.csv", index=False)
    print(f"  Simulated Kaggle : {df_k.shape}")

    # Mental State style (same format, different labels)
    z_cols = ([f"delta_{i}" for i in range(14)] +
              [f"alpha_{i}" for i in range(14)] +
              [f"beta_{i}"  for i in range(14)])
    df_z = pd.DataFrame(np.random.randn(80,len(z_cols))*3, columns=z_cols)
    df_z["label"] = np.random.choice(["RELAXED","NEUTRAL","CONCENTRATING"],80)
    df_z.to_csv(K2_DIR/"mental_states.csv", index=False)
    print(f"  Simulated Mental State : {df_z.shape}\n")

    build()

    (K_DIR/"emotions.csv").unlink(missing_ok=True)
    (K2_DIR/"mental_states.csv").unlink(missing_ok=True)
    print("\n  ✅  Demo complete — pipeline verified\n")


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv: demo()
    else: build()