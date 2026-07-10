"""
Tier 3: V-JEPA 2 ViT-L — temporal dynamics in latent space.

Processes 64-frame clips with stride 32 (50% overlap).
Each clip → mean-pool [8192, 1024] spatiotemporal tokens → [1024]-dim vector.
PCA fitted on training sessions only → compress to 32 dims.

Usage:
  python tier3_vjepa2.py --run 2025-03-17_10-51-58     (extract raw)
  python tier3_vjepa2.py --all                          (extract all)
  python tier3_vjepa2.py --pca --train-sessions s1 ...  (fit PCA)

Outputs:
  outputs/video_assessment/features/tier3_raw_{run_id}.npz
    keys: clip_center_ms, features [N_clips, 1024]
  outputs/video_assessment/features/tier3_{run_id}.parquet
    columns: clip_center_ms, pca_00 ... pca_31
  outputs/video_assessment/features/tier3_pca_model.npz
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT      = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/home/simran/allspark-data-exploration/Pittsburgh-pendulum-datalogs/extracted")
OUT_DIR   = ROOT / "outputs/video_assessment/features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLIP_FRAMES  = 64     # model expects 64 frames per clip
CLIP_STRIDE  = 32     # 50% overlap
IMG_SIZE     = 256    # fpc64-256 model
PCA_DIMS     = 32
MODEL_ID     = "facebook/vjepa2-vitl-fpc64-256"


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


def load_clip_frames(mp4_path: Path, start: int, n_frames: int = CLIP_FRAMES) -> np.ndarray | None:
    """Load exactly n_frames starting at start; returns [T, H, W, C] uint8 or None."""
    cap = cv2.VideoCapture(str(mp4_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(cv2.resize(frame, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
        frames.append(rgb)
    cap.release()
    return np.stack(frames) if len(frames) == n_frames else None


def load_vjepa2_model(device: str):
    """
    Load V-JEPA 2 ViT-L encoder using the local vjepa2 repo.
    Returns (encoder, expected_feat_dim) or (None, None) on failure.
    """
    vjepa2_dir = ROOT / "vjepa2"
    if not vjepa2_dir.exists():
        print("  vjepa2 repo not found at", vjepa2_dir)
        return None, None

    import sys
    sys.path.insert(0, str(vjepa2_dir))

    # vjepa2_vit_large already defaults to img_size=256, num_frames=64
    # It returns (encoder, predictor) — we only need the encoder.
    # Do NOT pass img_size= as it is already hardcoded in the hub function,
    # and passing it again causes "multiple values" error.
    try:
        result = torch.hub.load(
            str(vjepa2_dir),
            "vjepa2_vit_large",
            source="local",
            pretrained=True,
        )
        encoder = result[0] if isinstance(result, (tuple, list)) else result
        encoder = encoder.to(device).eval()
        print(f"  Loaded V-JEPA 2 ViT-L encoder via torch.hub (local repo)")
        return encoder, 1024
    except Exception as e:
        print(f"  torch.hub load failed: {e}")

    # Fallback: direct import from vjepa2 src + manual weight download
    try:
        from src.models.vision_transformer import vit_large  # noqa
        model = vit_large(
            img_size=IMG_SIZE,
            patch_size=16,
            num_frames=CLIP_FRAMES,
            tubelet_size=2,
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            use_rope=True,
        )
        # weights URL from ARCH_NAME_MAP["vit_large"] → "vitl" → vitl.pt
        weights_url  = "https://dl.fbaipublicfiles.com/vjepa2/vitl.pt"
        weights_path = ROOT / "outputs/video_assessment/vjepa2_vitl_weights.pth"
        if not weights_path.exists():
            print(f"  Downloading V-JEPA 2 weights (~1.2GB) from fbaipublicfiles...")
            import urllib.request
            urllib.request.urlretrieve(weights_url, str(weights_path))
        state = torch.load(weights_path, map_location="cpu")
        encoder_state = state.get("target_encoder", state.get("encoder", state))
        model.load_state_dict(encoder_state, strict=False)
        model = model.to(device).eval()
        print(f"  Loaded V-JEPA 2 ViT-L via direct import + weights")
        return model, 1024
    except Exception as e:
        print(f"  Direct import also failed: {e}")
        return None, None


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_clip(clip_np: np.ndarray) -> torch.Tensor:
    """
    clip_np: [T, H, W, C] uint8 RGB
    Returns: [1, C, T, H, W] float32 tensor normalized with ImageNet stats
    """
    clip_f = clip_np.astype(np.float32) / 255.0          # [T, H, W, C]
    clip_f = (clip_f - _IMAGENET_MEAN) / _IMAGENET_STD   # broadcast over spatial
    # [T, H, W, C] → [C, T, H, W] → [1, C, T, H, W]
    return torch.from_numpy(clip_f.transpose(3, 0, 1, 2)).unsqueeze(0)


def extract_raw_transformers(run_id: str) -> bool:
    """Load V-JEPA 2 via local vjepa2 repo and extract spatiotemporal features."""
    out_path = OUT_DIR / f"tier3_raw_{run_id}.npz"
    if out_path.exists():
        print(f"  [{run_id}] Raw already exists, skipping.")
        return True

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    model, feat_dim = load_vjepa2_model(device)
    if model is None:
        return False

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

    cap     = cv2.VideoCapture(str(mp4))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"  [{run_id}] {n_total} frames, stride={CLIP_STRIDE}")

    clip_starts  = list(range(0, n_total - CLIP_FRAMES + 1, CLIP_STRIDE))
    print(f"    {len(clip_starts)} clips to process on {device}...")

    all_features  = []
    all_center_ms = []

    for i, start in enumerate(clip_starts):
        clip_np = load_clip_frames(mp4, start)
        if clip_np is None:
            continue
        center_frame = start + CLIP_FRAMES // 2
        center_ms    = frame_ts.get(center_frame, -1)

        clip_t = preprocess_clip(clip_np).to(device)
        with torch.no_grad():
            try:
                out = model(clip_t)
                # out: [1, n_tokens, feat_dim] or [1, T', H', W', feat_dim]
                if out.dim() == 3:
                    feat = out[0].mean(dim=0).float().cpu().numpy()      # [feat_dim]
                else:
                    feat = out.mean(dim=(0,1,2,3)).float().cpu().numpy()  # [feat_dim]
            except Exception as e:
                print(f"    inference error at clip {i}: {e}")
                continue

        all_features.append(feat)
        all_center_ms.append(center_ms)
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(clip_starts)} clips")

    if not all_features:
        print(f"  [{run_id}] No features extracted.")
        return False

    features_arr  = np.stack(all_features)
    center_ms_arr = np.array(all_center_ms)

    np.savez_compressed(out_path,
                        clip_center_ms=center_ms_arr,
                        features=features_arr)
    print(f"  [{run_id}] Saved {features_arr.shape} → {out_path.name}")
    return True


def extract_raw_hub(run_id: str) -> bool:
    """Fallback: try loading via torch.hub if transformers fails."""
    out_path = OUT_DIR / f"tier3_raw_{run_id}.npz"
    if out_path.exists():
        return True

    try:
        import sys
        sys.path.insert(0, str(ROOT / "vjepa2"))  # if cloned locally
        from models.vision_transformer import vit_large
        model = vit_large(
            img_size=IMG_SIZE,
            num_frames=CLIP_FRAMES,
            patch_size=16,
        )
        # Load pretrained weights if available
        ckpt_path = ROOT / "vjepa2_vitl_fpc64_256.pth"
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state.get("encoder", state), strict=False)
    except Exception as e:
        print(f"  Hub/local V-JEPA2 load also failed: {e}")
        print(f"  Skipping Tier 3 for {run_id}.")
        return False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device).eval()

    run_dir   = DATA_ROOT / run_id
    mp4s      = sorted((run_dir / "camera-video").glob("output_*.mp4"))
    ts_files  = sorted((run_dir / "camera-video").glob("output_timestamps_session_*.txt"))
    if not mp4s:
        return False

    mp4      = mp4s[0]
    ts_file  = ts_files[0] if ts_files else None
    frame_ts = load_timestamps(ts_file) if ts_file else {}

    cap     = cv2.VideoCapture(str(mp4))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    all_features  = []
    all_center_ms = []

    for start in range(0, n_total - CLIP_FRAMES + 1, CLIP_STRIDE):
        clip_np = load_clip_frames(mp4, start)
        if clip_np is None:
            continue
        center_frame = start + CLIP_FRAMES // 2
        center_ms    = frame_ts.get(center_frame, -1)

        # [T, H, W, C] → [1, C, T, H, W] float16
        clip_t = torch.from_numpy(clip_np.astype(np.float32) / 255.0)
        clip_t = clip_t.permute(0, 3, 1, 2).unsqueeze(0).half().to(device)

        with torch.no_grad():
            out  = model(clip_t)
            feat = out.mean(dim=(1, 2, 3)).float().cpu().numpy()

        all_features.append(feat)
        all_center_ms.append(center_ms)

    if not all_features:
        return False

    np.savez_compressed(out_path,
                        clip_center_ms=np.array(all_center_ms),
                        features=np.stack(all_features))
    return True


def fit_and_apply_pca(train_sessions: list, all_sessions: list):
    from sklearn.decomposition import PCA

    train_feats = []
    for run_id in train_sessions:
        npz_path = OUT_DIR / f"tier3_raw_{run_id}.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path)
        valid = data["clip_center_ms"] > 0
        train_feats.append(data["features"][valid])

    if not train_feats:
        print("No training data for PCA.")
        return

    X_train = np.vstack(train_feats)
    print(f"  Fitting PCA({PCA_DIMS}) on {X_train.shape[0]:,} training clips...")
    pca = PCA(n_components=PCA_DIMS, random_state=42)
    pca.fit(X_train)
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")

    np.savez(OUT_DIR / "tier3_pca_model.npz",
             components=pca.components_,
             mean=pca.mean_,
             explained_variance_ratio=pca.explained_variance_ratio_)
    print("  Saved tier3_pca_model.npz")

    for run_id in all_sessions:
        npz_path = OUT_DIR / f"tier3_raw_{run_id}.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path)
        proj = pca.transform(data["features"])
        cols = {f"pca_{i:02d}": proj[:, i] for i in range(PCA_DIMS)}
        df   = pd.DataFrame({
            "clip_center_ms": data["clip_center_ms"],
            **cols,
        })
        out_path = OUT_DIR / f"tier3_{run_id}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  [{run_id}] Saved → {out_path.name}")


def get_video_sessions() -> list:
    params_path = ROOT / "outputs/video_assessment/dataset_params.json"
    if params_path.exists():
        params = json.loads(params_path.read_text())
        return params.get("full_video_sessions", params.get("video_sessions", []))
    return sorted(d.name for d in DATA_ROOT.iterdir()
                  if (d/"camera-video").exists() and (d/"datalayer").exists())


def main():
    parser = argparse.ArgumentParser(description="Tier 3: V-JEPA 2 extraction")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run",  help="Single run ID")
    group.add_argument("--all",  action="store_true")
    group.add_argument("--pca",  action="store_true")
    parser.add_argument("--train-sessions", nargs="*")
    args = parser.parse_args()

    print("=" * 60)
    print("Tier 3: V-JEPA 2 Feature Extraction")
    print("=" * 60)

    if args.pca:
        sessions       = get_video_sessions()
        train_sessions = args.train_sessions or sessions[:-1]
        fit_and_apply_pca(train_sessions, sessions)
        return

    sessions = get_video_sessions() if args.all else [args.run]

    # 30-minute timeout: try transformers first, then hub
    DEADLINE_S = 1800
    import time
    deadline = time.time() + DEADLINE_S

    for run in sessions:
        print(f"\nProcessing {run}...")
        if time.time() > deadline:
            print(f"  30-minute deadline reached. Skipping remaining sessions.")
            break

        success = extract_raw_transformers(run)
        if not success:
            print("  Transformers failed; trying hub/local fallback...")
            success = extract_raw_hub(run)
        if not success:
            print(f"  [{run}] Tier 3 SKIPPED — install v-jepa2 or use transformers>=4.46")

    print("\nTier 3 complete. Run with --pca to compress features.")


if __name__ == "__main__":
    main()
