"""
Tier 2: DINOv2 ViT-L/14 — per-frame spatial scene features.

Extracts patch tokens (excluding CLS and register tokens) per frame,
computes mean and std across patches → 2048-dim raw feature per frame.
PCA fitted on training sessions only → compress to 32 dims.

Usage:
  python tier2_dinov2.py --run 2025-03-17_10-51-58     (extract raw)
  python tier2_dinov2.py --all                          (extract all raw)
  python tier2_dinov2.py --pca --train-sessions s1 s2 s3 s4  (fit PCA + project all)

Outputs (per session, raw extraction):
  outputs/video_assessment/features/tier2_raw_{run_id}.npz
    keys: frame_idx, timestamp_ms, features  [N, 2048]

After PCA fit:
  outputs/video_assessment/features/tier2_{run_id}.parquet
    columns: frame_idx, timestamp_ms, pca_00 ... pca_31
  outputs/video_assessment/features/tier2_pca_model.npz
    PCA components, mean for later LOSO application
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import Dinov2Model

ROOT      = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/home/simran/allspark-data-exploration/Pittsburgh-pendulum-datalogs/extracted")
OUT_DIR   = ROOT / "outputs/video_assessment/features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE  = 64
PCA_DIMS    = 32
IMG_SIZE    = 224   # DINOv2 expects 224×224 (ViT-L patch_size=14 → 16×16 patches)
N_REGISTERS = 0   # facebook/dinov2-large has no register tokens

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_frame(frame_bgr: np.ndarray) -> torch.Tensor:
    """Replicate torchvision ImageNet normalization without torchvision."""
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    normed  = (resized.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
    # [H, W, C] → [C, H, W]
    return torch.from_numpy(normed.transpose(2, 0, 1))


def load_timestamps(ts_path: Path) -> dict:
    frame_ts = {}
    for line in ts_path.read_text().strip().split("\n"):
        parts = line.strip().split(":")
        if len(parts) == 2:
            try:
                frame_ts[int(parts[0].strip())] = int(parts[1].strip())
            except ValueError:
                pass
    return frame_ts


def extract_raw(run_id: str, model: nn.Module, device: str) -> bool:
    out_path = OUT_DIR / f"tier2_raw_{run_id}.npz"
    if out_path.exists():
        print(f"  [{run_id}] Raw already exists, skipping extraction.")
        return True

    run_dir   = DATA_ROOT / run_id
    video_dir = run_dir / "camera-video"
    mp4s      = sorted(video_dir.glob("output_*.mp4"))
    ts_files  = sorted(video_dir.glob("output_timestamps_session_*.txt"))

    if not mp4s or not ts_files:
        print(f"  [{run_id}] Missing video/timestamp files.")
        return False

    mp4      = mp4s[0]
    ts_file  = ts_files[0]
    frame_ts = load_timestamps(ts_file)
    print(f"  [{run_id}] {mp4.name} — {len(frame_ts)} timestamps")

    cap      = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {mp4}")

    all_feats     = []
    all_frame_idx = []
    all_ts        = []
    batch_imgs    = []
    batch_fidx    = []

    def flush_batch():
        if not batch_imgs:
            return
        batch_t = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            out = model(pixel_values=batch_t)
            # last_hidden_state: [B, 1+n_patches, 1024]  (token 0 = CLS, rest = patches)
            patches = out.last_hidden_state[:, 1:, :]   # [B, 256, 1024]
            mean_f  = patches.mean(dim=1).cpu().numpy()   # [B, 1024]
            std_f   = patches.std(dim=1).cpu().numpy()    # [B, 1024]
        feat = np.concatenate([mean_f, std_f], axis=1)   # [B, 2048]
        all_feats.append(feat)
        all_frame_idx.extend(batch_fidx)
        batch_imgs.clear()
        batch_fidx.clear()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        tensor = preprocess_frame(frame)
        batch_imgs.append(tensor)
        batch_fidx.append(frame_idx)
        if len(batch_imgs) >= BATCH_SIZE:
            flush_batch()
        frame_idx += 1

    flush_batch()
    cap.release()

    if not all_feats:
        print(f"  [{run_id}] No features extracted.")
        return False

    features    = np.vstack(all_feats)           # [N_frames, 2048]
    frame_idx_a = np.array(all_frame_idx)
    ts_a        = np.array([frame_ts.get(int(i), -1) for i in frame_idx_a])

    np.savez_compressed(out_path,
                        frame_idx=frame_idx_a,
                        timestamp_ms=ts_a,
                        features=features)
    print(f"    Saved raw: {features.shape} → {out_path.name}")
    return True


def fit_and_apply_pca(train_sessions: list, all_sessions: list):
    """Fit PCA on train_sessions, apply to all_sessions."""
    from sklearn.decomposition import PCA

    # Collect training features
    train_feats = []
    for run_id in train_sessions:
        npz_path = OUT_DIR / f"tier2_raw_{run_id}.npz"
        if not npz_path.exists():
            print(f"  WARNING: {npz_path.name} not found for PCA training, skipping.")
            continue
        data = np.load(npz_path)
        # Only use frames with valid timestamps for PCA
        valid = data["timestamp_ms"] > 0
        train_feats.append(data["features"][valid])

    if not train_feats:
        print("No training data for PCA.")
        return

    X_train = np.vstack(train_feats)
    print(f"  Fitting PCA({PCA_DIMS}) on {X_train.shape[0]:,} training frames...")
    pca = PCA(n_components=PCA_DIMS, random_state=42)
    pca.fit(X_train)
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")

    # Save PCA model
    np.savez(OUT_DIR / "tier2_pca_model.npz",
             components=pca.components_,
             mean=pca.mean_,
             explained_variance_ratio=pca.explained_variance_ratio_)
    print("  Saved tier2_pca_model.npz")

    # Apply to all sessions
    for run_id in all_sessions:
        npz_path = OUT_DIR / f"tier2_raw_{run_id}.npz"
        if not npz_path.exists():
            print(f"  [{run_id}] Raw not found, skipping PCA projection.")
            continue
        data = np.load(npz_path)
        proj = pca.transform(data["features"])   # [N, 32]
        cols = {f"pca_{i:02d}": proj[:, i] for i in range(PCA_DIMS)}
        df   = pd.DataFrame({
            "frame_idx":    data["frame_idx"],
            "timestamp_ms": data["timestamp_ms"],
            **cols,
        })
        out_path = OUT_DIR / f"tier2_{run_id}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  [{run_id}] Saved PCA-projected: {out_path.name}")


def get_video_sessions() -> list:
    params_path = ROOT / "outputs/video_assessment/dataset_params.json"
    if params_path.exists():
        params = json.loads(params_path.read_text())
        return params.get("full_video_sessions", params.get("video_sessions", []))
    return sorted(d.name for d in DATA_ROOT.iterdir()
                  if (d/"camera-video").exists() and (d/"datalayer").exists())


def main():
    parser = argparse.ArgumentParser(description="Tier 2: DINOv2 extraction")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run",  help="Single run ID (raw extraction)")
    group.add_argument("--all",  action="store_true", help="Extract all sessions")
    group.add_argument("--pca",  action="store_true", help="Fit PCA + project (run after --all)")
    parser.add_argument("--train-sessions", nargs="*",
                        help="Sessions to use for PCA fit (default: all but last)")
    args = parser.parse_args()

    print("=" * 60)
    print("Tier 2: DINOv2 ViT-L/14 Feature Extraction")
    print("=" * 60)

    if args.pca:
        sessions       = get_video_sessions()
        train_sessions = args.train_sessions or sessions[:-1]
        print(f"PCA fit on {len(train_sessions)} sessions, project {len(sessions)}")
        fit_and_apply_pca(train_sessions, sessions)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Loading DINOv2 ViT-L/14 (facebook/dinov2-large via transformers)...")
    try:
        model = Dinov2Model.from_pretrained("facebook/dinov2-large")
        model = model.to(device).eval()
    except Exception as e:
        print(f"FAILED to load DINOv2: {e}")
        raise

    if args.all:
        sessions = get_video_sessions()
        print(f"Processing {len(sessions)} sessions")
    else:
        sessions = [args.run]

    for run in sessions:
        extract_raw(run, model, device)

    print("\nTier 2 extraction complete.")
    print("Run with --pca to fit PCA and generate compressed parquet files.")


if __name__ == "__main__":
    main()
