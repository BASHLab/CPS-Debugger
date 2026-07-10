"""GRPO on AR-modern decoder — Phase B.

Custom group-relative policy optimization loop (no TRL — ARModernModel is not
a HuggingFace model). Per step: state-stratified prompt batch → rollout K
samples per prompt → reward = exp(-α·NED) → group-relative advantage
A_i = (r_i − mean(r_group)) / std(r_group) → policy gradient
loss = -mean(A_i · sum_log_prob(y_i | x_i)) + β · KL(π || π_ref), where π_ref
is a frozen deepcopy of the un-GRPO'd checkpoint.

Optional --use-state-bonus adds a per-state reward bonus to s0/s2 prompts
(ablation: compare with default stratified-only).

Run under the `grpo` conda env (transformers 4.57 + peft 0.18 + torch 2.5.1).

────────────────────────────────────────────────────────────────────────────
File structure (built incrementally per Phase B plan):
  Piece 1: log-prob helper + frozen-reference setup + sanity test
  Piece 2: state-stratified prompt sampler over TraceDataset.fsm_states
  Piece 3: rollout + reward + group-relative advantage
  Piece 4: main loop (optimizer, logging, checkpoint)
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import ModelConfig, BASE_DIR, TOKEN_MAPPING
from data import TraceDataset, load_token_mapping
from ar_modern_model import ARModernModel
from metrics import normalized_edit_distance
from determinism import set_deterministic, DEFAULT_SEED


# ════════════════════════════════════════════════════════════════════════════
# PIECE 1 — log-prob helper + frozen-reference setup + sanity test
# ════════════════════════════════════════════════════════════════════════════
def strip_compile_prefix(sd: dict) -> dict:
    """Remove `_orig_mod.` prefix that torch.compile adds (CLAUDE.md gotcha)."""
    if any(k.startswith("_orig_mod.") for k in sd):
        return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    return sd


def sequence_log_prob(model: ARModernModel,
                     x0: torch.Tensor,           # (B, L) int — target tokens
                     sensor_emb: torch.Tensor,   # (B, ...) — sensor conditioning
                     pad_mask: torch.Tensor      # (B, L) float — 1 valid, 0 pad
                     ) -> torch.Tensor:
    """Sum_t log π(y_t | y_<t, x) per sequence. Returns shape (B,).

    Grad flow: caller controls — pass policy with grad ON, wrap reference in
    torch.no_grad(). Mirrors compute_loss's BOS-prepend / logits-slice exactly.
    """
    B, L = x0.shape
    bos = torch.full((B, 1), model.bos_id, dtype=x0.dtype, device=x0.device)
    input_ids = torch.cat([bos, x0], dim=1)                  # (B, L+1)
    logits = model.backbone(input_ids, sensor_emb)[:, :L, :] # (B, L, V)
    log_probs = F.log_softmax(logits, dim=-1)
    tok_lp = log_probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)  # (B, L)
    return (tok_lp * pad_mask).sum(dim=-1)                   # (B,)


def make_frozen_reference(policy: ARModernModel) -> ARModernModel:
    """Frozen deepcopy for the KL-penalty reference. Eval mode + no grad."""
    ref = copy.deepcopy(policy)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def _sanity_test_piece1():
    """Verify sequence_log_prob == -(loss × n_valid) from compute_loss.

    Must be run in eval mode — dropout in train mode makes the two backbone
    forward passes (one per helper) sample different masks and disagree.
    """
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=10, max_seq_len=8, d_sensor_emb=16)
    model = ARModernModel(cfg)
    model.eval()
    B, L = 3, 6
    x0 = torch.randint(0, cfg.vocab_size, (B, L))
    sensor_emb = torch.randn(B, cfg.d_sensor_emb)
    pad_mask = torch.ones(B, L); pad_mask[1, -2:] = 0.0
    with torch.no_grad():
        lp = sequence_log_prob(model, x0, sensor_emb, pad_mask)
        out = model.compute_loss(x0, sensor_emb, pad_mask)
    loss = out[0] if isinstance(out, tuple) else out
    expected = -loss.item() * pad_mask.sum().item()
    actual = lp.sum().item()
    assert abs(expected - actual) < 1e-3, f"mismatch: {expected:.4f} vs {actual:.4f}"
    ref = make_frozen_reference(model)
    assert all(not p.requires_grad for p in ref.parameters())
    print(f"[piece 1 sanity] OK: log_prob sum = {actual:.4f} ; ref frozen ✓")


# ════════════════════════════════════════════════════════════════════════════
# PIECE 2 — state-stratified prompt sampler over TraceDataset.fsm_states
# ════════════════════════════════════════════════════════════════════════════
class StratifiedPromptSampler:
    """Wraps a TraceDataset; yields mini-batches where each item is drawn by
    first picking an FSM state with given weights (default 40/20/40 for
    s0/s1/s2), then uniformly within that state's pool.

    Does NOT subclass DataLoader — GRPO needs full control over batch
    construction and per-prompt replication for K-sample rollouts.
    """

    def __init__(self, dataset: TraceDataset,
                 weights=(0.4, 0.2, 0.4), seed: int = DEFAULT_SEED):
        self.ds = dataset
        w = np.asarray(weights, dtype=np.float64)
        self.weights = w / w.sum()
        self.by_state = {0: [], 1: [], 2: []}
        for i, s in enumerate(dataset.fsm_states):
            self.by_state[int(s)].append(i)
        for s in (0, 1, 2):
            assert len(self.by_state[s]) > 0, f"empty state-{s} pool"
        self.rng = np.random.default_rng(seed)
        counts = {s: len(self.by_state[s]) for s in (0, 1, 2)}
        print(f"  StratifiedPromptSampler: pools s0={counts[0]} s1={counts[1]} "
              f"s2={counts[2]} ; weights={tuple(round(float(x), 2) for x in self.weights)}")

    def sample_indices(self, n: int) -> list[int]:
        idxs = []
        for _ in range(n):
            s = int(self.rng.choice(3, p=self.weights))
            pool = self.by_state[s]
            idxs.append(int(pool[self.rng.integers(0, len(pool))]))
        return idxs

    def collate(self, idxs: list[int]):
        """Returns (x0, sensor_emb, pad_mask, fsm_states, ref_lens).

        TraceDataset.__getitem__ yields {trace, trace_len, sensor_emb,
        fsm_state}. We use `trace` as x0 and derive `pad_mask` from
        `trace_len` (1 up to trace_len, 0 after — matches the convention
        used by ARModernModel.compute_loss).
        """
        batch = [self.ds[i] for i in idxs]
        x0 = torch.stack([b["trace"] for b in batch])             # (B, L)
        emb = torch.stack([b["sensor_emb"] for b in batch])
        B, L = x0.shape
        lens = torch.tensor([int(b["trace_len"]) for b in batch], dtype=torch.long)
        ar = torch.arange(L).unsqueeze(0).expand(B, L)
        pm = (ar < lens.unsqueeze(1)).float()                     # (B, L)
        states = torch.tensor([b["fsm_state"] for b in batch], dtype=torch.long)
        return x0, emb, pm, states, lens.clamp(min=1)


def _sanity_test_piece2():
    """Build a tiny fake state list and verify the sampler hits the target
    distribution within statistical tolerance."""
    class _FakeDS:
        fsm_states = [0]*1000 + [1]*1000 + [2]*1000
        def __getitem__(self, i): raise NotImplementedError
    sampler = StratifiedPromptSampler(_FakeDS(), weights=(0.4, 0.2, 0.4), seed=0)
    idxs = sampler.sample_indices(10_000)
    states = [0 if i < 1000 else (1 if i < 2000 else 2) for i in idxs]
    frac = {s: states.count(s) / len(states) for s in (0, 1, 2)}
    expected = {0: 0.4, 1: 0.2, 2: 0.4}
    for s in (0, 1, 2):
        assert abs(frac[s] - expected[s]) < 0.02, \
            f"state {s}: got {frac[s]:.3f}, expected {expected[s]:.3f}"
    print(f"[piece 2 sanity] OK: empirical fractions {frac} ≈ target {expected}")


# ════════════════════════════════════════════════════════════════════════════
# PIECE 3 — rollout + reward + group-relative advantage
# ════════════════════════════════════════════════════════════════════════════
def rollout_K_samples(policy: ARModernModel, sensor_emb: torch.Tensor,
                      K: int, seq_len: int, temperature: float = 1.0
                      ) -> torch.Tensor:
    """Generate K samples per prompt. Replicates emb K times along batch dim
    so policy.sample (which is @torch.no_grad) does B*K in one pass.
    Returns (B*K, seq_len) of token ids. Note: model.sample IS no_grad already,
    so caller doesn't need to wrap.
    """
    emb_kb = sensor_emb.repeat_interleave(K, dim=0)
    return policy.sample(emb_kb, seq_len=seq_len, temperature=temperature,
                         greedy=False)


def derive_sample_mask(samples: torch.Tensor, pad_id: int):
    """Build (mask, lens) for the sampled sequences: 1 up to (and excluding)
    the first pad_id occurrence, 0 after. Lens = sum of mask. Both shape (B,L)
    and (B,) respectively.

    Why cumulative-prefix rather than "where(tok != pad)": once the model
    emits a pad_id mid-sequence we treat that as end-of-trace; everything after
    is also "padding" for log-prob accounting. Matches how the eval crops
    sampled traces.
    """
    not_pad = (samples != pad_id).float()
    mask = (not_pad.cumprod(dim=1) > 0).float()
    lens = mask.sum(dim=1).long().clamp(min=1)
    return mask, lens


def compute_ned_rewards(samples: torch.Tensor, refs: torch.Tensor,
                        ref_lens: torch.Tensor, sample_lens: torch.Tensor,
                        alpha: float = 10.0) -> torch.Tensor:
    """exp(-α · NED(sample, ref)) per pair, shape (n,). NED computed on each
    pair cropped to its own length (consistent with evaluate.py)."""
    s_np = samples.detach().cpu().numpy()
    r_np = refs.detach().cpu().numpy()
    rs = np.empty(samples.shape[0], dtype=np.float32)
    for i in range(samples.shape[0]):
        s = s_np[i, :int(sample_lens[i].item())].tolist()
        r = r_np[i, :int(ref_lens[i].item())].tolist()
        ned = normalized_edit_distance(s, r)
        rs[i] = float(np.exp(-alpha * ned))
    return torch.from_numpy(rs)


def group_relative_advantage(rewards: torch.Tensor, K: int) -> torch.Tensor:
    """Within-group standardization: (r − mean) / (std + eps), grouped by K.
    Input (B*K,), output (B*K,). Removes prompt-difficulty as a confound
    (DeepSeek-R1's core GRPO trick)."""
    r = rewards.reshape(-1, K)
    adv = (r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, keepdim=True) + 1e-8)
    return adv.reshape(-1)


def _sanity_test_piece3():
    """Verify advantage standardization works."""
    rewards = torch.tensor([
        0.1, 0.5, 0.9,    # group 1: spread, mean 0.5
        0.4, 0.4, 0.4,    # group 2: constant — should yield ~0 advantages
        0.0, 1.0, 0.5,    # group 3
    ])
    adv = group_relative_advantage(rewards, K=3)
    # torch.std uses Bessel correction (n-1). Group 1 [0.1,0.5,0.9]: mean=0.5,
    # std=sqrt(0.32/2)=0.4 → adv[0] = (0.1-0.5)/0.4 = -1.0.
    assert abs(adv[0].item() - (-1.0)) < 0.01, f"got {adv[0].item():.4f}"
    # group 2: zero std → near-zero advantages (eps prevents div-by-zero)
    assert abs(adv[3].item()) < 1e-3
    # group 3 [0,1,0.5]: mean=0.5, std=sqrt(0.5/2)=0.5 → (0-0.5)/0.5 = -1.0
    assert abs(adv[6].item() - (-1.0)) < 0.01, f"got {adv[6].item():.4f}"
    print(f"[piece 3 sanity] OK: advantages standardized within-group ✓")


# ════════════════════════════════════════════════════════════════════════════
# PIECE 4 — main GRPO training loop
# ════════════════════════════════════════════════════════════════════════════
def train_grpo(args):
    set_deterministic(args.seed)
    device = torch.device(args.device)

    # Token vocab + model config. load_token_mapping returns
    # (vocab_size, triples, triple_to_id); unpack the int (not len(tuple)).
    vocab_size, _, _ = load_token_mapping(Path(args.vocab))
    mcfg = ModelConfig(
        vocab_size=vocab_size,
        max_seq_len=args.max_seq_len,
        d_sensor_emb=args.d_sensor_emb,
    )

    # Load policy from CE-trained Phase B checkpoint (un-GRPO'd reference)
    ckpt = torch.load(args.policy_ckpt, map_location=device, weights_only=False)
    sd = strip_compile_prefix(ckpt["model"])
    policy = ARModernModel(mcfg).to(device)
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  policy load: missing={len(missing)} unexpected={len(unexpected)}")
    ref = make_frozen_reference(policy).to(device)
    print(f"  policy + frozen reference loaded from {args.policy_ckpt}")

    # Dataset + stratified sampler. TraceDataset signature:
    # (sessions, max_len, mask_id, emb_dir=...). mask_id = vocab_size (the
    # pad/mask convention used everywhere in this codebase).
    train_ds = TraceDataset(
        sessions=args.train_sessions,
        max_len=args.max_seq_len,
        mask_id=vocab_size,
        emb_dir=Path(args.emb_dir),
    )
    sampler = StratifiedPromptSampler(
        train_ds, weights=tuple(args.state_weights), seed=args.seed,
    )

    opt = torch.optim.AdamW(
        policy.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95),
    )

    K, B = args.K, args.batch_prompts
    state_bonus = args.state_bonus if args.use_state_bonus else 0.0
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "grpo_log.jsonl"
    log_f = open(log_path, "a")
    t0 = time.time()
    print(f"\nGRPO: K={K} prompts/step={B} α={args.alpha} β={args.beta} "
          f"lr={args.lr} steps={args.num_steps} state_bonus={state_bonus}")

    best_reward_ema = None
    for step in range(1, args.num_steps + 1):
        # ── rollout ──
        idxs = sampler.sample_indices(B)
        x0_b, emb_b, pm_b, states_b, ref_lens_b = sampler.collate(idxs)
        x0_b = x0_b.to(device); emb_b = emb_b.to(device); pm_b = pm_b.to(device)
        ref_lens_b = ref_lens_b.to(device)

        samples = rollout_K_samples(
            policy, emb_b, K=K, seq_len=args.max_seq_len,
            temperature=args.temperature,
        )                                                    # (B*K, L)
        sample_mask, sample_lens = derive_sample_mask(samples, policy.pad_id)

        # ── reward ──
        refs_kb = x0_b.repeat_interleave(K, dim=0)
        ref_lens_kb = ref_lens_b.repeat_interleave(K, dim=0)
        emb_kb = emb_b.repeat_interleave(K, dim=0)
        rewards = compute_ned_rewards(
            samples, refs_kb, ref_lens_kb, sample_lens, alpha=args.alpha,
        ).to(device)
        if state_bonus != 0.0:
            states_kb = states_b.repeat_interleave(K).to(device)
            rewards = rewards + state_bonus * ((states_kb == 0) | (states_kb == 2)).float()

        advantage = group_relative_advantage(rewards, K).to(device)

        # ── policy gradient + KL ──
        logp_pi = sequence_log_prob(policy, samples, emb_kb, sample_mask)
        with torch.no_grad():
            logp_ref = sequence_log_prob(ref, samples, emb_kb, sample_mask)

        n_tok = sample_mask.sum(dim=1).clamp(min=1)
        # Both PG and KL length-normalized per-token, so short and long
        # sequences contribute equally to the gradient (matches DeepSeek-R1
        # GRPO; without this, long traces dominate per-step updates).
        logp_pi_pt = logp_pi / n_tok
        logp_ref_pt = logp_ref / n_tok
        pg_loss = -(advantage.detach() * logp_pi_pt).mean()
        kl_per_seq = logp_pi_pt - logp_ref_pt
        kl_term = args.beta * kl_per_seq.mean()
        loss = pg_loss + kl_term

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        opt.step()

        if step % args.log_every == 0 or step == 1:
            r_mean, r_std = rewards.mean().item(), rewards.std().item()
            ms = (time.time() - t0) * 1000 / step
            best_reward_ema = r_mean if best_reward_ema is None \
                else 0.95 * best_reward_ema + 0.05 * r_mean
            entry = {
                "step": step, "loss": loss.item(),
                "pg_loss": pg_loss.item(), "kl_term": kl_term.item(),
                "reward_mean": r_mean, "reward_std": r_std,
                "reward_ema": best_reward_ema,
                "logp_pi": logp_pi.mean().item(),
                "logp_ref": logp_ref.mean().item(),
                "kl_per_seq": kl_per_seq.mean().item(),
                "ms_per_step": ms,
            }
            print(f"step {step:>5d} | loss {loss.item():+.4f} | "
                  f"pg {pg_loss.item():+.4f} kl {kl_term.item():+.4f} | "
                  f"r {r_mean:.4f}±{r_std:.4f} ema {best_reward_ema:.4f} | "
                  f"{ms:.0f} ms/step")
            log_f.write(json.dumps(entry) + "\n"); log_f.flush()

        if step % args.save_every == 0:
            torch.save({
                "model": policy.state_dict(), "step": step,
                "model_config": mcfg.__dict__, "grpo_args": vars(args),
                "objective": "autoregressive_modern",
            }, out_dir / f"grpo_step_{step}.pt")

    final = out_dir / "grpo_best.pt"
    torch.save({"model": policy.state_dict(), "step": args.num_steps,
                "model_config": mcfg.__dict__, "grpo_args": vars(args),
                "objective": "autoregressive_modern"}, final)
    log_f.close()
    print(f"\nGRPO done. Final ckpt: {final}")


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity-only", action="store_true",
                    help="Run the three piece-level sanity tests and exit.")
    ap.add_argument("--policy-ckpt",
                    help="Phase B CE-trained AR-modern ckpt (also the frozen ref).")
    ap.add_argument("--emb-dir")
    ap.add_argument("--train-sessions", nargs="+")
    ap.add_argument("--output-dir")
    ap.add_argument("--vocab", default=str(TOKEN_MAPPING),
                    help="Token mapping JSON (default: pruned_token_mapping.json).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--d-sensor-emb", type=int, default=272,
                    help="With threshold features: 256+16=272.")
    # GRPO hparams
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--batch-prompts", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--num-steps", type=int, default=800)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=200)
    # Stratification
    ap.add_argument("--state-weights", nargs=3, type=float, default=[0.4, 0.2, 0.4])
    ap.add_argument("--use-state-bonus", action="store_true",
                    help="Add per-state bonus to reward for s0/s2 prompts.")
    ap.add_argument("--state-bonus", type=float, default=0.5)
    args = ap.parse_args()

    if args.sanity_only:
        _sanity_test_piece1()
        _sanity_test_piece2()
        _sanity_test_piece3()
        return

    # required args for actual training
    for required in ("policy_ckpt", "emb_dir", "train_sessions", "output_dir"):
        if getattr(args, required) is None:
            ap.error(f"--{required.replace('_','-')} is required when not --sanity-only")
    train_grpo(args)


if __name__ == "__main__":
    main()
