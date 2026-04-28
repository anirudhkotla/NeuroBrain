"""
src/eeg/dataset_loader.py
══════════════════════════════════════════════════════════════
Downloads TWO Kaggle EEG datasets — both by birdy654,
both from Emotiv EPOC+ (14 channels), same CSV format.
No Zenodo. No manual downloads. Just your Kaggle API key.

Dataset 1 — Feeling Emotions
  birdy654/eeg-brainwave-dataset-feeling-emotions
  Labels: POSITIVE | NEUTRAL | NEGATIVE
  ~2132 rows

Dataset 2 — Mental State
  birdy654/eeg-brainwave-dataset-mental-state
  Labels: RELAXED | NEUTRAL | CONCENTRATING
  ~1500 rows

Together: ~3600 samples, same 14-channel Emotiv layout,
same feature columns — merges cleanly with no alignment needed.

Usage (run inside Docker):
  python src/eeg/dataset_loader.py --source both
  python src/eeg/dataset_loader.py --inspect
"""

import os, sys, json, argparse
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[2]
KAGGLE_DIR = ROOT / "data" / "raw" / "kaggle"
KAGGLE_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    {
        "slug"  : "birdy654/eeg-brainwave-dataset-feeling-emotions",
        "file"  : "emotions.csv",
        "desc"  : "EEG Brainwave: Feeling Emotions  (POSITIVE/NEUTRAL/NEGATIVE)",
    },
    {
        "slug"  : "birdy654/eeg-brainwave-dataset-mental-state",
        "file"  : "mental_states.csv",
        "desc"  : "EEG Brainwave: Mental State  (RELAXED/NEUTRAL/CONCENTRATING)",
    },
]


def _section(t):
    print(f"\n{'═'*56}\n  {t}\n{'═'*56}")


def _write_kaggle_creds() -> bool:
    u = os.getenv("KAGGLE_USERNAME", "").strip()
    k = os.getenv("KAGGLE_KEY",      "").strip()
    if not u or not k:
        print("""
  ⚠  Kaggle credentials missing.
  Open .env and set:
    KAGGLE_USERNAME=your_username
    KAGGLE_KEY=your_api_key_string
  Then re-run.
        """)
        return False
    d = Path.home() / ".kaggle"
    d.mkdir(exist_ok=True)
    c = d / "kaggle.json"
    c.write_text(json.dumps({"username": u, "key": k}))
    c.chmod(0o600)
    print(f"  ✓ Kaggle credentials written → {c}")
    return True


def _get_api():
    try:
        import kaggle  # noqa
    except ImportError:
        os.system("pip install kaggle --quiet --break-system-packages")
    # Kaggle SDK v1.6+ renamed KaggleApiExtended → KaggleApi
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        from kaggle.api.kaggle_api_extended import KaggleApiExtended as KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def download_dataset(api, slug: str, filename: str, desc: str):
    _section(desc)
    dest = KAGGLE_DIR / filename
    if dest.exists():
        mb = dest.stat().st_size / 1024 / 1024
        print(f"  ✓ Already downloaded — {filename}  ({mb:.1f} MB)")
        return dest

    print(f"  Downloading from Kaggle: {slug} …")
    api.dataset_download_files(slug, path=str(KAGGLE_DIR), unzip=True, quiet=False)

    # Find the downloaded CSV — filename may vary
    csvs = sorted(KAGGLE_DIR.glob("*.csv"))
    # The latest downloaded CSV is the one we want
    new_csvs = [c for c in csvs if c.name == filename or
                (not (KAGGLE_DIR / filename).exists())]

    if dest.exists():
        print(f"\n  ✅  {filename}  ({dest.stat().st_size/1024/1024:.1f} MB)")
        return dest

    # Rename if Kaggle used a different filename
    remaining = [c for c in sorted(KAGGLE_DIR.glob("*.csv"))
                 if c.name not in [d["file"] for d in DATASETS]]
    if remaining:
        remaining[0].rename(dest)
        print(f"\n  ✅  {filename}  ({dest.stat().st_size/1024/1024:.1f} MB)")
        return dest

    print(f"  ⚠  Could not locate {filename} after download")
    print(f"     Files in {KAGGLE_DIR}:")
    for f in KAGGLE_DIR.glob("*"):
        print(f"       {f.name}")
    return None


def download_both():
    if not _write_kaggle_creds():
        return
    api = _get_api()
    for ds in DATASETS:
        download_dataset(api, ds["slug"], ds["file"], ds["desc"])
    inspect()


def inspect():
    import pandas as pd
    _section("DATASET INSPECTION")

    all_found = True
    total_rows = 0

    for ds in DATASETS:
        f = KAGGLE_DIR / ds["file"]
        if f.exists():
            df = pd.read_csv(f)
            total_rows += len(df)
            print(f"\n  📊  {ds['file']}")
            print(f"       Shape  : {df.shape}")
            print(f"       Cols   : {list(df.columns[:5])} … ({len(df.columns)} total)")
            lc = next((c for c in df.columns if c.lower() == "label"), None)
            if lc:
                print(f"       Labels : {df[lc].value_counts().to_dict()}")
        else:
            print(f"\n  ❌  {ds['file']} not found")
            print(f"       Run: python src/eeg/dataset_loader.py --source both")
            all_found = False

    if all_found:
        print(f"\n  ✅  Both datasets ready — {total_rows} total rows")
    print()


def _load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            l = line.strip()
            if l and not l.startswith("#") and "=" in l:
                k, v = l.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


if __name__ == "__main__":
    _load_env()
    p = argparse.ArgumentParser()
    p.add_argument("--source",  choices=["both"], default="both",
                   help="Download both Kaggle datasets")
    p.add_argument("--inspect", action="store_true")
    a = p.parse_args()

    if a.inspect:
        inspect(); sys.exit(0)

    download_both()