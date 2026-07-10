"""
04_audio_preprocess.py — Extract and align audio features from Feb04 run.

The 2025-02-04_09-33-16 run has:
  - audio/output_*.csv: raw PCM audio samples (Timestamp, Audio Data)
  - processed_trace/processed_trace.csv: execution traces
  - syslog/syslog.csv: system metrics
  BUT NO datalayer (no physical cart/pendulum state sensors).

This script:
1. Extracts the Feb04 tarball (to /tmp or a specified dir)
2. Computes MFCC + spectral audio features from raw PCM
3. Aligns audio, syslog, and trace to 10ms windows
4. Saves audio_aligned_dataset.parquet for 04_audio_trace_prediction.py

Audio format: each CSV row = (timestamp_us, [PCM_sample_0 ... PCM_sample_N])
Estimated sample rate = total_samples_in_file / duration_seconds

Run: python3 scripts/04_audio_preprocess.py [--extract-dir /path/to/extract] [--win-ms 10]
"""

import argparse
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import spectrogram
from tqdm import tqdm

TARBALL = Path(
    "/home/simran/allspark-data-exploration/Pittsburgh-pendulum-datalogs/2025/"
    "2025-02-04_09-33-16.tar.gz"
)
DEFAULT_EXTRACT = Path(
    "/home/simran/allspark-data-exploration/Pittsburgh-pendulum-datalogs/extracted_feb04"
)
OUT_DIR = Path(
    "/home/simran/allspark-data-exploration/CPS-Debugger/outputs/experiments"
)
WIN_MS = 10
SAMPLE_RATE = 44100  # estimated; refined after loading


# ── Audio feature extraction ───────────────────────────────────────────────────
def compute_mfcc(signal: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    """Compute MFCCs from raw PCM using scipy (no librosa dependency).
    Returns n_mfcc coefficients."""
    # Pre-emphasis
    signal = np.append(signal[0], signal[1:] - 0.97 * signal[:-1]).astype(float)

    # Mel filter bank
    n_fft = 512
    n_mels = 40
    f_min, f_max = 0, sr // 2

    # STFT magnitude
    frame_len = n_fft
    hop_len = n_fft // 2
    if len(signal) < frame_len:
        signal = np.pad(signal, (0, frame_len - len(signal)))
    freqs, _, Sxx = spectrogram(signal, fs=sr, nperseg=frame_len, noverlap=frame_len - hop_len)
    Sxx = np.sqrt(Sxx.mean(axis=1) + 1e-10)  # averaged across time

    # Mel filterbank
    mel_min = 2595 * np.log10(1 + f_min / 700)
    mel_max = 2595 * np.log10(1 + f_max / 700)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bin_points = np.clip(bin_points, 0, len(freqs) - 1)

    fbank = np.zeros((n_mels, len(freqs)))
    for m in range(1, n_mels + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]
        if f_m > f_m_minus:
            fbank[m - 1, f_m_minus:f_m] = (
                np.arange(f_m_minus, f_m) - f_m_minus
            ) / (f_m - f_m_minus)
        if f_m_plus > f_m:
            fbank[m - 1, f_m:f_m_plus] = (
                f_m_plus - np.arange(f_m, f_m_plus)
            ) / (f_m_plus - f_m)

    # Apply filterbank
    mel_energy = np.dot(fbank, Sxx)
    mel_energy = np.where(mel_energy > 0, mel_energy, 1e-10)
    log_mel = np.log(mel_energy)

    # DCT for MFCCs
    n_mfcc_actual = min(n_mfcc, n_mels)
    mfccs = np.zeros(n_mfcc_actual)
    for i in range(n_mfcc_actual):
        mfccs[i] = np.sum(log_mel * np.cos(np.pi * i / n_mels * (np.arange(n_mels) + 0.5)))
    mfccs *= np.sqrt(2 / n_mels)

    return mfccs


def extract_audio_features(signal: np.ndarray, sr: int) -> dict:
    """Compute a feature vector from a PCM chunk."""
    sig = signal.astype(float)
    n = len(sig)
    if n == 0:
        return {}

    # RMS energy
    rms = float(np.sqrt(np.mean(sig ** 2)))

    # Zero crossing rate
    zcr = float(np.sum(np.abs(np.diff(np.sign(sig)))) / (2 * n))

    # Spectral features from FFT
    win = sig * np.hanning(n)
    fft_mag = np.abs(np.fft.rfft(win))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    eps = 1e-10
    total_power = fft_mag.sum() + eps
    spectral_centroid = float(np.sum(freqs * fft_mag) / total_power)
    spectral_rolloff = float(freqs[np.searchsorted(np.cumsum(fft_mag), 0.85 * total_power)])
    spectral_flatness = float(
        np.exp(np.mean(np.log(fft_mag + eps))) / (np.mean(fft_mag) + eps)
    )

    # MFCCs (simplified)
    mfccs = compute_mfcc(signal, sr, n_mfcc=13)

    features = {
        "audio_rms": rms,
        "audio_zcr": zcr,
        "audio_spectral_centroid": spectral_centroid,
        "audio_spectral_rolloff": spectral_rolloff,
        "audio_spectral_flatness": spectral_flatness,
    }
    for i, v in enumerate(mfccs):
        features[f"audio_mfcc_{i}"] = float(v)

    return features


def us_to_win(t_us, win_us: int):
    return (np.array(t_us, dtype=np.int64) // win_us).astype(np.int64)


def load_audio_csv(audio_csv: Path) -> tuple:
    """Load audio CSV → (timestamps_us, list_of_pcm_arrays)."""
    df = pd.read_csv(audio_csv)
    timestamps = df["Timestamp"].astype(np.int64).values
    pcm_chunks = []
    for val in df["Audio Data"].values:
        try:
            arr = np.array(json.loads(val), dtype=np.float32)
        except Exception:
            arr = np.zeros(1, dtype=np.float32)
        pcm_chunks.append(arr)
    return timestamps, pcm_chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-dir", type=Path, default=DEFAULT_EXTRACT)
    parser.add_argument("--win-ms", type=int, default=WIN_MS)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    win_us = args.win_ms * 1000
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Extract tarball ────────────────────────────────────────────────
    run_dir = args.extract_dir / "data" / "2025-02-04_09-33-16"
    if not run_dir.exists():
        print(f"Extracting {TARBALL} → {args.extract_dir}...")
        args.extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(TARBALL, "r:gz") as tar:
            # Extract only the non-binary files we need
            members_to_extract = [
                m for m in tar.getmembers()
                if any(
                    m.name.endswith(ext)
                    for ext in [".csv", ".json"]
                ) and not m.name.endswith(".jpg") and "/trace/" not in m.name
            ]
            print(f"  Extracting {len(members_to_extract)} CSV/JSON files...")
            tar.extractall(path=args.extract_dir, members=members_to_extract)
        print("  Extraction complete.")
    else:
        print(f"Found existing extraction at {run_dir}")

    # ── Step 2: Load audio CSVs ───────────────────────────────────────────────
    audio_dir = run_dir / "audio"
    audio_csvs = sorted(audio_dir.glob("output_*.csv"))
    print(f"Found {len(audio_csvs)} audio CSV files")

    if not audio_csvs:
        raise FileNotFoundError(f"No audio CSV files found in {audio_dir}")

    # Load all audio rows and estimate sample rate
    all_timestamps = []
    all_pcm = []
    for csv_path in tqdm(audio_csvs, desc="Loading audio"):
        ts, pcm = load_audio_csv(csv_path)
        all_timestamps.extend(ts.tolist())
        all_pcm.extend(pcm)

    all_timestamps = np.array(all_timestamps, dtype=np.int64)
    sort_idx = np.argsort(all_timestamps)
    all_timestamps = all_timestamps[sort_idx]
    all_pcm = [all_pcm[i] for i in sort_idx]

    # Estimate sample rate
    total_samples = sum(len(p) for p in all_pcm)
    duration_us = all_timestamps[-1] - all_timestamps[0]
    # Add samples in last chunk
    est_sr = int(total_samples / (duration_us / 1e6)) if duration_us > 0 else 44100
    print(f"Audio: {len(all_pcm):,} chunks, {total_samples:,} total samples, "
          f"estimated SR={est_sr} Hz")
    sr = est_sr

    # ── Step 3: Aggregate audio features to 10ms windows ─────────────────────
    wins = us_to_win(all_timestamps, win_us)
    win_ids = np.unique(wins)

    print(f"Extracting audio features for {len(win_ids):,} windows...")
    audio_rows = []
    t0 = time.time()
    for win_id in tqdm(win_ids, desc="Audio windows"):
        mask = wins == win_id
        # Concatenate all PCM chunks in this window
        chunks_in_win = [all_pcm[i] for i in np.where(mask)[0]]
        if not chunks_in_win:
            continue
        pcm_concat = np.concatenate(chunks_in_win)
        features = extract_audio_features(pcm_concat, sr)
        features["win"] = int(win_id)
        audio_rows.append(features)
    print(f"Done in {time.time()-t0:.1f}s")

    audio_df = pd.DataFrame(audio_rows).fillna(0.0)
    audio_df["win"] = audio_df["win"].astype(np.int64)
    audio_cols = [c for c in audio_df.columns if c != "win"]
    print(f"Audio features: {len(audio_cols)}")

    # ── Step 4: Load syslog ───────────────────────────────────────────────────
    syslog_path = run_dir / "syslog" / "syslog.csv"
    sl = pd.read_csv(syslog_path)
    sl["t_us"] = sl["Timestamp"].astype(np.int64)
    sl["win"] = us_to_win(sl["t_us"].values, win_us)
    sl_agg = sl.groupby("win").agg(
        sl_cpu=("CPU_Usage(%)", "mean"),
        sl_mem=("Memory_Usage(%)", "mean"),
        sl_load1=("Load_1m", "mean"),
    ).reset_index()
    sl_agg["win"] = sl_agg["win"].astype(np.int64)

    # ── Step 5: Load processed_trace ─────────────────────────────────────────
    trace_path = run_dir / "processed_trace" / "processed_trace.csv"
    print("Loading processed_trace...")
    tr = pd.read_csv(trace_path)
    tr["t_us"] = tr["timestamp"].astype(np.int64)
    tr["win"] = us_to_win(tr["t_us"].values, win_us)

    # Parse cf_table branch counts (reuse same logic as 00_preprocess.py)
    def parse_cf_row(cf_str):
        try:
            cf = json.loads(cf_str)
        except Exception:
            return {}
        result = {}
        for func_id, src_dict in cf.items():
            for src_pc, dst_dict in src_dict.items():
                for dst_pc, count in dst_dict.items():
                    key = f"{func_id}:{src_pc}:{dst_pc}"
                    result[key] = result.get(key, 0) + int(count)
        return result

    print(f"Parsing {len(tr):,} trace rows...")
    counts_rows = []
    for win_id, group in tqdm(tr.groupby("win"), desc="Trace windows"):
        wc = {}
        for cf_str in group["cf_table"].values:
            for k, v in parse_cf_row(cf_str).items():
                wc[k] = wc.get(k, 0) + v
        wc["win"] = int(win_id)
        counts_rows.append(wc)

    trace_df = pd.DataFrame(counts_rows).fillna(0.0)
    trace_df["win"] = trace_df["win"].astype(np.int64)
    branch_cols = [c for c in trace_df.columns if c != "win"]
    print(f"Unique branch keys: {len(branch_cols)}")

    # Build vocabulary for Feb04 run: top-50 by CV
    all_vals = trace_df[branch_cols].values
    presence_frac = (all_vals > 0).mean(axis=0)
    valid_branches = np.where(presence_frac >= 0.05)[0]
    means = all_vals[:, valid_branches].mean(axis=0)
    stds = all_vals[:, valid_branches].std(axis=0)
    cvs = stds / (means + 1e-9)
    top50_idx = valid_branches[np.argsort(cvs)[::-1][:50]]
    vocab_feb04 = [branch_cols[i] for i in top50_idx]

    # ── Step 6: Merge all modalities ─────────────────────────────────────────
    merged = audio_df.merge(sl_agg, on="win", how="inner")
    merged = merged.merge(trace_df[["win"] + vocab_feb04], on="win", how="inner")
    merged = merged.ffill().bfill().fillna(0.0)
    merged["run"] = "2025-02-04_09-33-16"
    merged = merged.sort_values("win").reset_index(drop=True)

    print(f"\nFeb04 aligned dataset: {len(merged):,} windows")
    print(f"Columns: {list(merged.columns[:10])}...")

    # Save
    merged.to_parquet(out / "audio_aligned_dataset.parquet", index=False)
    vocab_info_feb04 = {
        "vocab": vocab_feb04,
        "audio_cols": audio_cols,
        "syslog_cols": ["sl_cpu", "sl_mem", "sl_load1"],
        "win_ms": args.win_ms,
        "run": "2025-02-04_09-33-16",
        "estimated_sample_rate": int(sr),
    }
    (out / "vocab_info_feb04.json").write_text(json.dumps(vocab_info_feb04, indent=2))
    print(f"Saved: {out / 'audio_aligned_dataset.parquet'}")


if __name__ == "__main__":
    main()
