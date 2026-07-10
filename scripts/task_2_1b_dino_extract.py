"""
task_2_1b_dino_extract.py — DINOv2 feature extraction from video frames.

Extracts DINOv2 CLS embeddings for the best video run identified by task_2_1a,
then tests whether visual features predict branch counts within the BALANCE state.

Usage:
  python scripts/task_2_1b_dino_extract.py [--run_id RUN_ID] [--subsample N]

Output:
  outputs/phase3/dino_features.parquet
  outputs/phase3/dino_branch_r2_summary.json
  outputs/phase3/fig_video_redundancy.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT   = Path("/home/simran/allspark-data-exploration/CPS-Debugger")
P3_OUT = ROOT / "outputs/phase3"
P3_OUT.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path("/home/simran/allspark-data-exploration/Pittsburgh-pendulum-datalogs")

PHYSICAL_COLS = [
    "dl_current_x_mean", "dl_current_x_std", "dl_current_x_delta",
    "dl_angle_mean", "dl_angle_std", "dl_angle_delta",
    "dl_velocity_mean", "dl_ang_vel_mean", "dl_ang_vel_std",
]


def find_mp4(run_id: str):
    """Find an MP4 file for the given run_id."""
    candidates = [
        DATA_ROOT / "2025" / f"{run_id}-videos" / "output_video.mp4",
        DATA_ROOT / "2025" / run_id / "camera-video" / "output_0.mp4",
        DATA_ROOT / "extracted" / run_id / "camera-video" / "output_0.mp4",
        DATA_ROOT / run_id / "output_video.mp4",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Search broadly
    for mp4 in DATA_ROOT.rglob("*.mp4"):
        if run_id in str(mp4):
            return mp4
    return None


def extract_frames(mp4_path: Path, subsample: int = 5):
    """Extract frames from video, return (frames_array, timestamps_approx)."""
    import cv2
    cap = cv2.VideoCapture(str(mp4_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video: {total} frames at {fps:.1f} fps "
          f"({total/fps:.1f}s) — subsampling every {subsample}")

    frames, frame_indices = [], []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % subsample == 0:
            # Resize to 224x224 for DINOv2
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (224, 224))
            frames.append(frame_resized)
            frame_indices.append(idx)
        idx += 1
    cap.release()
    print(f"  Extracted {len(frames)} frames (every {subsample}th)")
    return np.array(frames), np.array(frame_indices), fps


def extract_dino_features(frames: np.ndarray, batch_size: int = 32):
    """Extract DINOv2 CLS token embeddings."""
    import torch
    from transformers import AutoModel, AutoImageProcessor

    print("  Loading DINOv2...")
    model_name = "facebook/dinov2-vitb14"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = model.to(device).eval()
    print(f"  Model loaded, device={device}")

    all_feats = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        # Convert to PIL-like list
        from PIL import Image
        pil_batch = [Image.fromarray(f) for f in batch]
        inputs = processor(images=pil_batch, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        cls_tokens = out.last_hidden_state[:, 0, :].cpu().numpy()  # (B, 768)
        all_feats.append(cls_tokens)
        if i % (batch_size * 10) == 0:
            print(f"    Processed {i+len(batch)}/{len(frames)} frames")
    return np.vstack(all_feats)


def align_frames_to_windows(frame_indices, fps, datalayer_df, win_ms=10):
    """Map frame indices to 10ms window IDs by matching timestamps."""
    # Approximate: frame time in ms from start of video
    frame_ms = frame_indices / fps * 1000
    # Window assignment (relative to first window)
    first_win = datalayer_df["win"].min()
    win_us = win_ms * 1000
    # Use relative time
    frame_wins = (frame_ms / win_ms).astype(int) + first_win
    return frame_wins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id",   type=str, default=None)
    parser.add_argument("--subsample", type=int, default=5,
                        help="Use every Nth frame (default=5 = 6fps from 30fps)")
    args = parser.parse_args()

    # ── Determine run_id ─────────────────────────────────────────────────────
    if args.run_id is None:
        vid_inv = P3_OUT / "video_inventory.json"
        if vid_inv.exists():
            info = json.loads(vid_inv.read_text())
            args.run_id = info.get("best_dino_candidate")
        if args.run_id is None:
            print("ERROR: No run_id specified and no video_inventory.json found.")
            return

    print(f"Processing run: {args.run_id}")

    # ── Find MP4 ─────────────────────────────────────────────────────────────
    mp4 = find_mp4(args.run_id)
    if mp4 is None:
        print(f"ERROR: No MP4 found for {args.run_id}")
        return
    print(f"  MP4: {mp4}")

    # ── Extract frames ────────────────────────────────────────────────────────
    frames, frame_indices, fps = extract_frames(mp4, subsample=args.subsample)
    if len(frames) == 0:
        print("ERROR: No frames extracted.")
        return

    # ── DINOv2 embeddings ─────────────────────────────────────────────────────
    features = extract_dino_features(frames)  # (N_frames, 768)
    print(f"  DINOv2 features shape: {features.shape}")

    # ── Frame-to-frame similarity ─────────────────────────────────────────────
    norms  = np.linalg.norm(features, axis=1, keepdims=True)
    normed = features / (norms + 1e-8)
    sim    = (normed[:-1] * normed[1:]).sum(axis=1)  # cosine sim consecutive frames
    print(f"  Frame-to-frame cosine similarity: mean={sim.mean():.3f} "
          f"min={sim.min():.3f} p5={np.percentile(sim, 5):.3f}")

    # Save raw features
    feat_df = pd.DataFrame(features, columns=[f"dino_{i}" for i in range(features.shape[1])])
    feat_df["frame_idx"] = frame_indices
    feat_df.to_parquet(P3_OUT / "dino_features.parquet", index=False)
    print("  Saved dino_features.parquet")

    # ── Correlate with trace (if datalayer available) ─────────────────────────
    # Load aligned dataset for this run
    aligned_path = ROOT / "outputs/experiments/aligned_dataset.parquet"
    expanded_path = P3_OUT / "aligned_dataset_expanded.parquet"
    dataset_path = expanded_path if expanded_path.exists() else aligned_path

    if not dataset_path.exists():
        print("  No aligned dataset found — skipping trace correlation.")
        return

    df = pd.read_parquet(dataset_path)
    run_df = df[df["run"] == args.run_id]
    if len(run_df) == 0:
        print(f"  Run {args.run_id} not in dataset — skipping trace correlation.")
        return

    print(f"  Found {len(run_df)} windows for {args.run_id} in dataset")

    # Align frames to windows
    frame_wins = align_frames_to_windows(frame_indices, fps, run_df)
    feat_df["win"] = frame_wins

    # PCA on DINOv2 to reduce to 50 dims
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score

    print("  PCA on DINOv2 features (768 → 50)...")
    pca = PCA(n_components=50)
    feat_pca = pca.fit_transform(features)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")

    feat_df_pca = pd.DataFrame(feat_pca, columns=[f"pc_{i}" for i in range(50)])
    feat_df_pca["win"] = frame_wins

    # Merge with branch counts
    vocab_info = json.loads((ROOT / "outputs/experiments/vocab_info.json").read_text())
    vocab = vocab_info["vocab"][:50]  # top 50 branches

    merged = feat_df_pca.merge(run_df[["win"] + vocab], on="win", how="inner")
    if len(merged) < 10:
        print("  Too few matched windows for regression — skipping.")
        return

    X = merged[[f"pc_{i}" for i in range(50)]].values
    balance_mask = run_df.set_index("win").loc[
        merged["win"], "dl_state"
    ].values == 1
    print(f"  Matched windows: {len(merged)}, BALANCE: {balance_mask.sum()}")

    # R² per branch within BALANCE
    if balance_mask.sum() < 10:
        print("  Insufficient BALANCE windows in matched set.")
        r2_scores = []
    else:
        X_bal = X[balance_mask]
        r2_scores = []
        for branch in vocab:
            y = merged[branch].values[balance_mask]
            if y.std() < 1e-6:
                r2_scores.append(0.0)
                continue
            split = int(0.8 * len(X_bal))
            rf = RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=4)
            rf.fit(X_bal[:split], y[:split])
            pred = rf.predict(X_bal[split:])
            r2_scores.append(max(-1.0, r2_score(y[split:], pred)))

    mean_r2 = float(np.mean(r2_scores)) if r2_scores else 0.0
    frac_gt_03 = float(np.mean([r > 0.3 for r in r2_scores])) if r2_scores else 0.0
    print(f"\n  DINOv2→trace R² (within BALANCE): mean={mean_r2:.3f}, "
          f"frac>0.3={frac_gt_03:.2f}")

    # Physical baseline for comparison
    phys_cols = [c for c in PHYSICAL_COLS if c in run_df.columns]
    merged_phys = feat_df_pca.merge(run_df[["win"] + phys_cols + vocab], on="win", how="inner")
    if len(merged_phys) > 10:
        bm = run_df.set_index("win").loc[merged_phys["win"], "dl_state"].values == 1
        if bm.sum() > 10:
            r2_phys = []
            for branch in vocab:
                y = merged_phys[branch].values[bm]
                X_p = merged_phys[phys_cols].values[bm]
                if y.std() < 1e-6:
                    r2_phys.append(0.0)
                    continue
                split = int(0.8 * len(X_p))
                rf = RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=4)
                rf.fit(X_p[:split], y[:split])
                pred = rf.predict(X_p[split:])
                r2_phys.append(max(-1.0, r2_score(y[split:], pred)))
            phys_mean = float(np.mean(r2_phys))
            print(f"  Physical→trace R² (same windows): mean={phys_mean:.3f}")
        else:
            phys_mean = -1.0
    else:
        phys_mean = -1.0

    # Save summary
    out = {
        "run_id": args.run_id,
        "n_frames_extracted": int(len(frames)),
        "subsample_factor": args.subsample,
        "frame_sim_mean": float(sim.mean()),
        "frame_sim_p5":   float(np.percentile(sim, 5)),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "n_matched_windows": int(len(merged)),
        "n_balance_windows_matched": int(balance_mask.sum()) if len(r2_scores) else 0,
        "dino_r2_mean_within_balance": mean_r2,
        "dino_r2_frac_gt_03": frac_gt_03,
        "physical_r2_mean_same_windows": phys_mean,
        "is_video_additive": mean_r2 > phys_mean * 0.5,  # rough threshold
    }
    (P3_OUT / "dino_branch_r2_summary.json").write_text(json.dumps(out, indent=2))

    # ── Figure ────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(sim, bins=40, color="#2196F3", edgecolor="black", linewidth=0.3)
    axes[0].set_xlabel("Cosine similarity (frame N → frame N+1)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Frame redundancy: {args.run_id[:20]}")

    if r2_scores:
        axes[1].scatter(range(len(r2_scores)), sorted(r2_scores, reverse=True),
                        s=15, color="#4CAF50", label=f"DINOv2 (mean={mean_r2:.3f})")
        if phys_mean > 0:
            axes[1].axhline(phys_mean, color="red", linestyle="--",
                            label=f"Physical (mean={phys_mean:.3f})")
    axes[1].axhline(0.3, color="gray", linestyle=":", label="R²=0.3")
    axes[1].set_xlabel("Branch rank")
    axes[1].set_ylabel("R² (within BALANCE)")
    axes[1].set_title("DINOv2 → trace R²")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(P3_OUT / "fig_video_redundancy.png")
    plt.close(fig)

    print("Saved fig_video_redundancy.png")
    print(f"\nConclusion: video {'IS additive' if out['is_video_additive'] else 'is NOT additive'} "
          f"beyond physical sensors (DINOv2 R²={mean_r2:.3f} vs "
          f"physical R²={phys_mean:.3f})")
    print("Done.")


if __name__ == "__main__":
    main()
