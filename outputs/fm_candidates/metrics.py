"""Shared metric helpers for generator-objective evaluation.

All functions operate on lists of token-id sequences (Python ints). Stay in
token space — do not detokenize; this corpus uses a domain-specific vocab.

Metric priorities for the BL-2 comparison.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np


# ── Sequence-level distance ──────────────────────────────────────────────

def normalized_edit_distance(pred: Sequence[int], target: Sequence[int]) -> float:
    """Levenshtein distance / max(len(pred), len(target))."""
    n, m = len(pred), len(target)
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return 1.0
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            if pred[i - 1] == target[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[m] / max(n, m)


def rouge_l_f1(pred: Sequence[int], target: Sequence[int]) -> Dict[str, float]:
    """ROUGE-L via LCS. Diagnostic; near-redundant with NED on this corpus."""
    n, m = len(pred), len(target)
    if n == 0 and m == 0:
        return {"p": 1.0, "r": 1.0, "f1": 1.0}
    if n == 0 or m == 0:
        return {"p": 0.0, "r": 0.0, "f1": 0.0}
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if pred[i - 1] == target[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
        for j in range(m + 1):
            curr[j] = 0
    lcs = prev[m]
    p = lcs / n
    r = lcs / m
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {"p": p, "r": r, "f1": f1}


# ── Distributional / corpus-level ────────────────────────────────────────

def token_distribution_kl(gen_tokens: List[int], ref_tokens: List[int],
                          vocab_size: int) -> float:
    """KL(ref || gen) over unigram token distributions, with Laplace smoothing."""
    ref_counts = np.zeros(vocab_size, dtype=np.float64)
    gen_counts = np.zeros(vocab_size, dtype=np.float64)
    for t in ref_tokens:
        if 0 <= t < vocab_size:
            ref_counts[t] += 1
    for t in gen_tokens:
        if 0 <= t < vocab_size:
            gen_counts[t] += 1
    ref_dist = (ref_counts + 1) / (ref_counts.sum() + vocab_size)
    gen_dist = (gen_counts + 1) / (gen_counts.sum() + vocab_size)
    return float(np.sum(ref_dist * np.log(ref_dist / gen_dist)))


def distinct_n(sequences: List[List[int]], n: int) -> float:
    """Distinct-n (Li et al., NAACL 2016): unique n-grams / total n-grams."""
    total = 0
    unique = set()
    for seq in sequences:
        if len(seq) < n:
            continue
        for i in range(len(seq) - n + 1):
            unique.add(tuple(seq[i:i + n]))
            total += 1
    return (len(unique) / total) if total > 0 else 0.0


def compute_crystal_bleu(
    generated: List[List[int]],
    references: List[List[int]],
    corpus_for_trivial: List[List[int]],
    k: int = 500,
) -> Dict[str, float]:
    """CrystalBLEU (Eghbali & Pradel, ASE 2022).

    Strips the top-k most frequent n-grams from the corpus before scoring so
    BLEU reflects non-trivial matches. Returns both raw and crystal BLEU.
    """
    from crystalbleu import corpus_bleu
    from nltk.util import ngrams as make_ngrams

    trivial_ngrams: Counter = Counter()
    for seq in corpus_for_trivial:
        str_seq = [str(t) for t in seq]
        for nn in range(1, 5):
            trivial_ngrams.update(make_ngrams(str_seq, nn))
    trivially_shared = Counter(dict(trivial_ngrams.most_common(k)))

    hyps = [[str(t) for t in seq] for seq in generated]
    refs = [[[str(t) for t in seq]] for seq in references]

    bleu = corpus_bleu(refs, hyps)
    crystal = corpus_bleu(refs, hyps, ignoring=trivially_shared)
    return {"bleu": float(bleu), "crystal_bleu": float(crystal)}


def compute_self_bleu(generated: List[List[int]], n_sample: int = 500,
                      seed: int = 0) -> float:
    """Self-BLEU (Zhu et al., 2018) — intra-set similarity; lower = more diverse."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    if len(generated) < 2:
        return 0.0

    rng = np.random.default_rng(seed)
    if len(generated) > n_sample:
        indices = rng.choice(len(generated), n_sample, replace=False)
        subset = [generated[i] for i in indices]
    else:
        subset = generated

    smoothing = SmoothingFunction().method1
    scores = []
    for i, hyp in enumerate(subset):
        refs = [subset[j] for j in range(len(subset)) if j != i]
        if len(refs) > 20:
            ref_idx = rng.choice(len(refs), 20, replace=False)
            refs = [refs[j] for j in ref_idx]
        str_hyp = [str(t) for t in hyp]
        str_refs = [[str(t) for t in r] for r in refs]
        scores.append(sentence_bleu(str_refs, str_hyp,
                                    smoothing_function=smoothing))
    return float(np.mean(scores))


def compute_mauve(
    generated: List[List[int]],
    references: List[List[int]],
    max_len: int = 1024,
    n_sample: int = 500,
    seed: int = 0,
) -> float:
    """MAUVE (Pillutla et al., NeurIPS 2021) — distributional match of gen vs ref.

    Requires len(generated) == len(references); paired samples go into the
    same k-means quantizer so mismatched lengths would mix unrelated gens/refs.
    """
    import mauve

    if len(generated) != len(references):
        raise ValueError(f"MAUVE expects aligned pairs; got "
                         f"{len(generated)} gen vs {len(references)} ref")

    rng = np.random.default_rng(seed)
    if len(generated) > n_sample:
        idx = rng.choice(len(generated), n_sample, replace=False)
        gen_sub = [generated[i] for i in idx]
        ref_sub = [references[i] for i in idx]
    else:
        gen_sub = generated
        ref_sub = references

    gen_text = [" ".join(str(t) for t in seq[:max_len]) for seq in gen_sub]
    ref_text = [" ".join(str(t) for t in seq[:max_len]) for seq in ref_sub]

    try:
        result = mauve.compute_mauve(
            p_text=ref_text, q_text=gen_text,
            device_id=-1, max_text_length=max_len, verbose=False,
        )
        return float(result.mauve)
    except Exception as e:
        print(f"  MAUVE computation failed: {e}")
        return -1.0


# ── Vendi score (Friedman & Dieng, TMLR 2023) ───────────────────────────

def compute_vendi_score(sequences: List[List[int]], n_sample: int = 300,
                        seed: int = 0) -> float:
    """Effective number of distinct samples under a NED-similarity kernel.

    VS(X) = exp(H(λ / n)) where λ are eigenvalues of K, K[i,j] = 1 − NED(s_i, s_j).
    Range [1, n]: 1 = all identical, n = all distinct. Domain-appropriate
    (no pretrained featurizer); handles 98%-self-similar corpora sensibly
    because both gen and ref give small Vendi — the ratio is what's meaningful.

    O(n² · L²) in edit distance; we subsample to keep tractable.
    """
    if len(sequences) < 2:
        return 1.0

    rng = np.random.default_rng(seed)
    if len(sequences) > n_sample:
        idx = rng.choice(len(sequences), n_sample, replace=False)
        subset = [sequences[i] for i in idx]
    else:
        subset = list(sequences)

    n = len(subset)
    K = np.ones((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            sim = 1.0 - normalized_edit_distance(subset[i], subset[j])
            K[i, j] = K[j, i] = sim

    eigvals = np.linalg.eigvalsh(K / n)
    # Non-PSD safety: clip, renormalize so they sum to 1 before entropy.
    eigvals = np.clip(eigvals, 0.0, None)
    s = eigvals.sum()
    if s <= 0:
        return 1.0
    eigvals = eigvals / s
    H = -np.sum(eigvals * np.log(eigvals + 1e-12))
    return float(np.exp(H))


# ── Per-FSM-state breakouts of headline metrics ──────────────────────────

def compute_per_state_metrics(
    generated: List[List[int]],
    references: List[List[int]],
    fsm_states: Sequence[int],
    vendi_n: int = 200,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """NED / exact match / Vendi ratio, broken out per FSM state.

    Why FSM state: the pendulum controller's behavior is *defined* by its
    state machine. A worst-case per-state value is a drop in the model's
    ability to follow a specific control-flow branch — something the mean
    washes out because one state (`pendulum_state == 1`) is ~97% of ticks.

    Returns `{state_id (str): {n, ned_mean, exact_match, vendi_generated,
    vendi_reference, vendi_ratio_gen_over_ref}}`. Callers decide which per-
    state value matters (worst = max NED / min exact / min Vendi ratio); we
    only report the raw table.
    """
    if len(generated) != len(references) or len(generated) != len(fsm_states):
        raise ValueError("generated, references, fsm_states must be aligned")

    by_state: Dict[int, List[int]] = {}
    for i, s in enumerate(fsm_states):
        by_state.setdefault(int(s), []).append(i)

    per_state: Dict[str, Dict[str, float]] = {}
    for state, idx_list in sorted(by_state.items()):
        gen_s = [generated[i] for i in idx_list]
        ref_s = [references[i] for i in idx_list]

        neds = [normalized_edit_distance(g, r) for g, r in zip(gen_s, ref_s)]
        ned_mean = float(np.mean(neds)) if neds else 0.0
        n_exact = sum(1 for g, r in zip(gen_s, ref_s) if g == r)
        exact = n_exact / max(len(gen_s), 1)

        vs_n = min(len(gen_s), vendi_n)
        vs_gen = compute_vendi_score(gen_s, n_sample=vs_n, seed=seed)
        vs_ref = compute_vendi_score(ref_s, n_sample=vs_n, seed=seed)
        ratio = (vs_gen / vs_ref) if vs_ref > 0 else 0.0

        per_state[str(state)] = {
            "n": len(idx_list),
            "ned_mean": ned_mean,
            "exact_match": exact,
            "vendi_generated": vs_gen,
            "vendi_reference": vs_ref,
            "vendi_ratio_gen_over_ref": ratio,
        }
    return per_state


# ── Convenience: compute the full generator-comparison metric pack ───────

def compute_generation_metrics(
    generated: List[List[int]],
    references: List[List[int]],
    vocab_size: int,
    crystal_k: int = 500,
    self_bleu_n: int = 200,
    mauve_n: int = 500,
    vendi_n: int = 300,
    fsm_states: Optional[Sequence[int]] = None,
    seed: int = 0,
) -> Dict[str, float]:
    """Run the full metric pack on (generated, references).

    `fsm_states` (per-sample int state id) enables conditional n-gram JS,
    grouped by exact FSM state. If omitted, the conditional block is skipped.

    Returns a flat dict suitable for JSON dump. Used by both model eval and
    trivial-baseline eval so numbers are directly comparable.
    """
    out: Dict[str, float] = {}

    # Sequence-level precision (per-sample → mean/median)
    ned = [normalized_edit_distance(g, r) for g, r in zip(generated, references)]
    out["ned_mean"] = float(np.mean(ned)) if ned else 0.0
    out["ned_median"] = float(np.median(ned)) if ned else 0.0

    rouge_f1 = [rouge_l_f1(g, r)["f1"] for g, r in zip(generated, references)]
    out["rouge_l_f1_mean"] = float(np.mean(rouge_f1)) if rouge_f1 else 0.0
    out["rouge_l_f1_median"] = float(np.median(rouge_f1)) if rouge_f1 else 0.0

    n_exact = sum(1 for g, r in zip(generated, references) if g == r)
    out["exact_match"] = n_exact / max(len(generated), 1)

    # Per-position token accuracy under self-generation
    total_correct, total_tokens = 0, 0
    for g, r in zip(generated, references):
        for gt, rt in zip(g, r):
            if gt == rt:
                total_correct += 1
            total_tokens += 1
    out["token_acc_generation"] = total_correct / max(total_tokens, 1)

    # CrystalBLEU + raw BLEU (precision above trivial-overlap floor)
    bleu = compute_crystal_bleu(generated, references, references, k=crystal_k)
    out["bleu"] = bleu["bleu"]
    out["crystal_bleu"] = bleu["crystal_bleu"]

    # MAUVE was dropped (GPT-2 features pretrained on English don't transfer
    # to our 648 pruned domain tokens, and it was the source of a 6h
    # eval-timeout cascade due to HF tokenizer fork-deadlock with DataLoader
    # workers). Vendi (NED kernel) covers the distributional-match role.

    # Distinct-n — NLP-classic output variety
    out["distinct_1"] = distinct_n(generated, 1)
    out["distinct_2"] = distinct_n(generated, 2)
    out["distinct_1_reference"] = distinct_n(references, 1)
    out["distinct_2_reference"] = distinct_n(references, 2)

    # Self-BLEU — intra-set similarity (mode-collapse diagnostic)
    sb_n = min(len(generated), self_bleu_n)
    out["self_bleu_generated"] = compute_self_bleu(generated, n_sample=sb_n, seed=seed)
    out["self_bleu_reference"] = compute_self_bleu(references, n_sample=sb_n, seed=seed)

    # Unigram KL
    flat_gen = [t for seq in generated for t in seq]
    flat_ref = [t for seq in references for t in seq]
    out["token_kl_divergence"] = token_distribution_kl(flat_gen, flat_ref, vocab_size)

    # Vendi score — effective-diversity with NED-similarity kernel.
    # Ratio gen/ref is the interpretable quantity on a 98%-self-similar corpus.
    vs_n = min(len(generated), vendi_n)
    vs_gen = compute_vendi_score(generated, n_sample=vs_n, seed=seed)
    vs_ref = compute_vendi_score(references, n_sample=vs_n, seed=seed)
    out["vendi_generated"] = vs_gen
    out["vendi_reference"] = vs_ref
    out["vendi_ratio_gen_over_ref"] = (vs_gen / vs_ref) if vs_ref > 0 else 0.0

    # Per-FSM-state NED / exact match / Vendi ratio. Dominant state (~97% of
    # ticks) drives the marginal numbers; per-state surfaces where models fail
    # on transient states.
    if fsm_states is not None and len(fsm_states) == len(generated):
        out["per_state"] = compute_per_state_metrics(
            generated, references, fsm_states,
            vendi_n=min(vendi_n, 200), seed=seed,
        )

    return out
