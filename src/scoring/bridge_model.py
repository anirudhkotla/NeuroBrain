"""
src/scoring/bridge_model.py
══════════════════════════════════════════════════════════════
Step 7 — EEG Bridge Model

Trains a neural network that learns to map idea feature vectors
(from the fusion layer) to EEG-derived brain region scores.

This is the "bridge" between the NLP/audio/video world and the
neuroscience world — trained on real EEG engagement data.

Architecture
────────────
Input  : fused feature vector (from fusion layer scores, 6-dim)
         OR raw EEG feature vector (2548-dim from preprocessing)
Hidden : 3 fully-connected layers with BatchNorm + Dropout
Output : 6 brain region scores (0–100)

Training data : data/processed/X_features.npy   (4611, 2548)
                data/processed/y_brain_scores.npy (4611, 6)

The model learns: "given these EEG signal patterns, what brain
region activation scores do they correspond to?" — then at
inference time we swap EEG features for NLP/audio/video features
that proxy the same brain states.

Usage
─────
  # Train
  python src/scoring/bridge_model.py --train

  # Evaluate only
  python src/scoring/bridge_model.py --eval

  # Demo (no real data needed)
  python src/scoring/bridge_model.py --demo
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[2]
PROCESSED  = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models" / "checkpoints"
FINAL_DIR  = ROOT / "models" / "final"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)

REGIONS = ["dmn","ecn","san","striatum","hippocampus","basal"]


# ════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ════════════════════════════════════════════════════════════

def build_model(input_dim: int, output_dim: int = 6):
    """
    Build the EEG bridge MLP.

    Architecture optimised for RTX 3050 4 GB:
    - Small hidden layers to stay within VRAM
    - BatchNorm for stable training on EEG data
    - Dropout to prevent overfitting on small dataset
    - Sigmoid output scaled to 0-100
    """
    import torch
    import torch.nn as nn

    class BridgeModel(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()

            # Encoder — compress input to latent space
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.1),
            )

            # 6 independent scoring heads — one per brain region
            # Each head specialises in one region's activation pattern
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(128, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid(),       # output 0–1, scaled to 0–100 later
                )
                for _ in range(output_dim)
            ])

        def forward(self, x):
            latent = self.encoder(x)
            scores = torch.cat(
                [head(latent) for head in self.heads], dim=1
            )               # (batch, 6)
            return scores * 100.0   # scale to 0–100

    return BridgeModel(input_dim, output_dim)


# ════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════

def load_data(processed_dir: Path):
    """Load preprocessed EEG features and brain scores."""
    X = np.load(processed_dir / "X_features.npy")
    y = np.load(processed_dir / "y_brain_scores.npy")
    print(f"  Loaded: X{X.shape}  y{y.shape}")
    return X.astype(np.float32), y.astype(np.float32)


def make_dataloaders(X, y, val_split=0.15, test_split=0.10,
                     batch_size=32, seed=42):
    """Split into train/val/test and create DataLoaders."""
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    np.random.seed(seed)
    n      = len(X)
    idx    = np.random.permutation(n)
    n_test = int(n * test_split)
    n_val  = int(n * val_split)

    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    def make_dl(indices, shuffle):
        ds = TensorDataset(
            torch.tensor(X[indices]),
            torch.tensor(y[indices]),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=True)

    return (make_dl(train_idx, True),
            make_dl(val_idx,   False),
            make_dl(test_idx,  False),
            len(train_idx), len(val_idx), len(test_idx))


# ════════════════════════════════════════════════════════════
# TRAINING
# ════════════════════════════════════════════════════════════

def train(epochs: int = 50, batch_size: int = 32,
          lr: float = 0.001, patience: int = 10):
    """
    Train the bridge model on EEG data.
    Uses mixed precision (fp16) to fit within 4 GB VRAM.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.cuda.amp import GradScaler, autocast

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")

    # Load data
    X, y = load_data(PROCESSED)

    # Normalise features (StandardScaler)
    from sklearn.preprocessing import StandardScaler
    import joblib
    scaler = StandardScaler()
    X      = scaler.fit_transform(X).astype(np.float32)
    joblib.dump(scaler, FINAL_DIR / "feature_scaler.pkl")
    print(f"  Scaler saved → {FINAL_DIR / 'feature_scaler.pkl'}")

    train_dl, val_dl, test_dl, n_tr, n_val, n_te = make_dataloaders(
        X, y, batch_size=batch_size
    )
    print(f"  Split  : train={n_tr}  val={n_val}  test={n_te}")

    # Build model
    model     = build_model(input_dim=X.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=True
    )
    criterion = nn.MSELoss()
    scaler_fp16 = GradScaler()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params : {n_params:,}")
    print(f"\n  {'Epoch':>5}  {'Train MSE':>10}  {'Val MSE':>10}  "
          f"{'Val MAE':>9}  {'LR':>8}")
    print("  " + "─"*50)

    best_val_loss = float("inf")
    no_improve    = 0
    history       = []

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            with autocast():
                pred = model(xb)
                loss = criterion(pred, yb)
            scaler_fp16.scale(loss).backward()
            scaler_fp16.step(optimizer)
            scaler_fp16.update()
            train_loss += loss.item() * len(xb)
        train_loss /= n_tr

        # ── Validate ───────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_mae  = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred     = model(xb)
                val_loss += criterion(pred, yb).item() * len(xb)
                val_mae  += torch.mean(torch.abs(pred - yb)).item() * len(xb)
        val_loss /= n_val
        val_mae  /= n_val

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch":epoch,"train_mse":train_loss,
                         "val_mse":val_loss,"val_mae":val_mae})

        # Print every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            print(f"  {epoch:>5}  {train_loss:>10.3f}  {val_loss:>10.3f}  "
                  f"{val_mae:>9.2f}  {current_lr:>8.6f}")

        # Early stopping + checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            torch.save({
                "epoch"       : epoch,
                "model_state" : model.state_dict(),
                "optimizer"   : optimizer.state_dict(),
                "val_loss"    : val_loss,
                "val_mae"     : val_mae,
                "input_dim"   : X.shape[1],
            }, FINAL_DIR / "eeg_bridge.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    # ── Test evaluation ────────────────────────────────────
    checkpoint = torch.load(FINAL_DIR / "eeg_bridge.pt",
                            map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(device)
            all_preds.append(model(xb).cpu().numpy())
            all_targets.append(yb.numpy())

    preds   = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    test_mae = float(np.mean(np.abs(preds - targets)))
    test_mse = float(np.mean((preds - targets)**2))

    print(f"\n  ── Test Results ───────────────────────────")
    print(f"  Test MAE  : {test_mae:.2f}  (avg error in score points)")
    print(f"  Test MSE  : {test_mse:.2f}")
    print(f"\n  Per-region MAE:")
    for i, region in enumerate(REGIONS):
        mae_r = float(np.mean(np.abs(preds[:,i] - targets[:,i])))
        print(f"    {region:<14} {mae_r:.2f}")

    # Save training history
    with open(FINAL_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  ✅  Model saved → {FINAL_DIR / 'eeg_bridge.pt'}")
    return model


# ════════════════════════════════════════════════════════════
# INFERENCE
# ════════════════════════════════════════════════════════════

def load_trained_model(device=None):
    """Load the trained bridge model from disk."""
    import torch
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt      = torch.load(FINAL_DIR / "eeg_bridge.pt", map_location=device)
    model     = build_model(input_dim=ckpt["input_dim"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, device


def predict(feature_vector: np.ndarray) -> dict:
    """
    Run inference on a feature vector.

    Parameters
    ----------
    feature_vector : np.ndarray — shape (2548,) raw EEG features
                                  OR (6,) fused encoder scores

    Returns
    -------
    dict: dmn, ecn, san, striatum, hippocampus, basal (all 0–100)
    """
    import torch
    import joblib

    model, device = load_trained_model()

    # Apply scaler if raw features
    scaler_path = FINAL_DIR / "feature_scaler.pkl"
    if scaler_path.exists() and len(feature_vector) > 6:
        scaler = joblib.load(scaler_path)
        feature_vector = scaler.transform(
            feature_vector.reshape(1, -1)
        ).flatten()

    x = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        scores = model(x).cpu().numpy().flatten()

    return {r: round(float(s), 1) for r, s in zip(REGIONS, scores)}


# ════════════════════════════════════════════════════════════
# DEMO (no real data needed)
# ════════════════════════════════════════════════════════════

def run_demo():
    """Train on synthetic data to verify the pipeline works."""
    import torch
    import torch.nn as nn
    import torch.optim as optim

    print("  Demo mode — synthetic data (100 samples, 50 features)")
    np.random.seed(42)
    n, d = 200, 50
    X = np.random.randn(n, d).astype(np.float32)
    # Synthetic target: each region is a linear combo of features + noise
    W = np.random.randn(d, 6).astype(np.float32)
    y = (X @ W * 10 + 50).astype(np.float32)
    y = np.clip(y, 0, 100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(input_dim=d).to(device)
    opt    = optim.Adam(model.parameters(), lr=0.01)
    loss_fn= nn.MSELoss()

    from torch.utils.data import TensorDataset, DataLoader
    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    dl = DataLoader(ds, batch_size=16, shuffle=True)

    print(f"  Training 10 epochs on {device} …")
    for epoch in range(1, 11):
        model.train()
        ep_loss = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        if epoch % 2 == 0:
            print(f"    Epoch {epoch:>2}  loss={ep_loss/len(dl):.2f}")

    # Test inference
    model.eval()
    test_x = torch.tensor(X[:1]).to(device)
    with torch.no_grad():
        pred = model(test_x).cpu().numpy().flatten()

    print(f"\n  Sample prediction (demo):")
    for region, score in zip(REGIONS, pred):
        bar = "█" * int(score / 5)
        print(f"    {region:<14} {score:5.1f}  {bar}")

    print("\n  ✅  Bridge model architecture verified.\n")


# ════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true",
                        help="Train on real EEG data")
    parser.add_argument("--eval",  action="store_true",
                        help="Evaluate saved model")
    parser.add_argument("--demo",  action="store_true",
                        help="Run with synthetic data (no EEG needed)")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr",         type=float, default=0.001)
    args = parser.parse_args()

    print("\n" + "═"*56)
    print("  STEP 7 — EEG BRIDGE MODEL")
    print("═"*56)

    if args.demo or (not args.train and not args.eval):
        run_demo()
    elif args.train:
        train(epochs=args.epochs,
              batch_size=args.batch_size,
              lr=args.lr)
    elif args.eval:
        if not (FINAL_DIR / "eeg_bridge.pt").exists():
            print("  ❌  No trained model found. Run --train first.")
            sys.exit(1)
        model, device = load_trained_model()
        print(f"  ✓ Model loaded from {FINAL_DIR / 'eeg_bridge.pt'}")
        print(f"  Run --train to train or import predict() for inference.")