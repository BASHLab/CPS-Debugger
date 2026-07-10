# Research prompt — stabilizing a long-window self-supervised time-series encoder

Paste the section below into Claude.ai (or another strong model). It is
self-contained; no repository access is needed.

---

## Role

You are an ML research advisor specializing in self-supervised representation
learning for time series. I need a prioritized, concrete set of recommendations
for a pretraining problem, ranked by expected payoff under tight constraints.
Favor specific recipe/hyperparameter changes and named methods (with paper
references) over generic advice.

## Setting

I am building a sensor encoder for a "trace foundation model." The downstream
task: given a multivariate sensor stream from an inverted-pendulum controller
(7 channels at 1 kHz: FSM state, an iteration counter, target/current cart
position, velocity, pole angle, angular velocity), autoregressively predict the
controller's per-tick WASM control-flow execution trace (which basic blocks /
branch targets execute each tick). A frozen sensor encoder produces per-tick
conditioning embeddings; a separate AR decoder generates the trace.

The hard part of the downstream task is **branch targets that depend on
long-horizon accumulator state**. The controller's C source has counters/timers
that gate branches at thresholds up to ~4000 ticks (e.g. a setpoint-reference
counter fires at 4000, a stand-up counter at 1500, a setpoint-advance counter at
1000, a "good range" gain switch at 500). The exact threshold-crossing tick has
no clean sensor proxy, so the encoder must *represent the count itself*.

## Encoder architecture

- Input window: 4000 ticks × 7 channels (recently lengthened from 500).
- Conv frontend (4 Conv1d layers, total stride 40) → ~98 time patches/channel.
- Axial transformer: 12 layers, d_model=512, alternating time-axis and
  channel-axis attention over a (98 patches × 7 channels) grid (~686 tokens).
- Perceiver pool: K=4 learned query tokens → 4 "dense" tokens at d=256 (the
  decoder's conditioning input). ~41M params total.
- The decoder either takes the K=4 pool (mean-pooled to a prefix) or
  cross-attends to the full ~686 axial tokens (no-Perceiver). The K=4 pool is a
  ~27× squeeze of the 4000×7 input; empirically the no-Perceiver coupling is
  much better downstream, so the pool is a real information bottleneck.

## Pretraining objective and the failure mode

The encoder is pretrained with a **data2vec-style latent-prediction objective**:
an EMA-teacher copy of the encoder produces target features for masked patches,
the student predicts them, loss = smooth_L1 on the masked positions. Masking is
multi-span over the flattened patch grid (mask ratio ~0.35, span scaled to the
patch count). 30K steps, single GPU, batch 16 × grad-accum 8.

**Two problems:**

1. **Moving-target collapse / best-at-warmup.** Validation loss is *lowest at
   step ~1000* (essentially warmup), then rises ~20% and plateaus. Best-val
   selection therefore returns a near-warmup checkpoint. The EMA teacher starts
   nearly identical to the student (a trivially easy prediction target) and the
   target hardens as it drifts, so val_loss rises even if representations are
   still changing. Crucially, **downstream trace-prediction quality also
   degrades monotonically with more pretraining steps** (step-1000 > 10K > 40K
   on a cheap linear-probe decoder), so this is not only a val_loss artifact.
   A variant with an added supervised class-balanced contrastive auxiliary
   (positives = same FSM state) does NOT show this — its best-val lands at 30K,
   apparently because the contrastive term provides a non-moving target.

2. **Uncertain long-horizon retention.** FSM state is ~99.9% linearly decodable
   from the encoder features — but FSM state is literally one of the 7 input
   channels, so that is a weak signal. I am separately probing whether the
   4000-tick representation retains a long-horizon accumulator
   ("ticks-since-last-FSM-transition"), banded by horizon, vs a 500-tick encoder
   that physically cannot see past 500 ticks. (Result pending.)

## Hard constraint (applies to ALL items below)

The encoder is frozen and run per-tick at inference, so capacity costs latency.
Treat **~41M parameters, single GPU, ~30K pretraining steps** as a hard budget
for the *primary* recommendation in every item. You may flag **at most two**
higher-capacity options across the whole answer, and only if there is strong
published evidence they specifically help accumulator/counter representation;
mark each clearly as "exceeds budget" with the rough cost.

## What I want from you

Give a prioritized plan (highest expected ROI first), each item with the
concrete change and the method/paper it draws on:

1. **Stabilize the data2vec EMA target for long sequences.** Address the
   moving-target collapse specifically: teacher-target normalization
   (instance/layer norm of targets as in data2vec 2.0), EMA momentum schedule
   (annealing the decay), predictor depth/width, mask ratio and span design for
   long patch grids, and whether to stop-gradient / sharpen the target. Which of
   these most directly prevents "best-val at warmup, degrade after"?

2. **Should I switch objective?** Compare, for *preserving long-horizon
   accumulator state* in a 4000-step 1D multivariate series: data2vec 2.0,
   masked autoencoding for time series (e.g. SimMTM, Ti-MAE, TimeMAE, PatchTST),
   contrastive (TS2Vec, TF-C), and JEPA variants. You are NOT restricted to that
   list — you may bring in adjacent/recent (2024-2026) objectives where clearly
   superior for our two failures, e.g. HuBERT-style discrete-codebook targets,
   MoCo-v3 dual-view contrastive, CPC / predictive-coding, or hybrid
   input-reconstruction + latent-target losses. But keep it a short, ranked
   shortlist (not a survey), and score every candidate explicitly on (a)
   robustness to the moving-target collapse and (b) long-horizon
   counter/integrator retention.

3. **Architecture for long-horizon retention.** The K=4 Perceiver bottleneck
   and pure attention may not represent a 4000-tick integrator well. Evaluate
   alternatives: state-space / Mamba (incl. Mamba-2 / delta-rule) or
   linear-attention / linear-recurrent memory (RWKV, RetNet), dilated/causal
   conv stacks, hierarchical or multi-scale patching (fine-short + coarse-long),
   learned register/memory tokens, and explicit recurrent/accumulator inductive
   biases. **Out of scope: external-retrieval / RAG-style memory** — the entire
   4000-tick window already fits in context, so retrieval does not address the
   in-window retention question.

4. **Probing protocol.** How would you rigorously test whether a learned
   representation encodes a monotonic counter / time-since-event, and attribute
   downstream failures to "info not in representation" vs "decoder can't read
   it"? Suggest targets and controls beyond the ticks-since-transition probe.

For each recommendation, state the expected effect, the rough implementation
cost, and any risk. End with a single recommended next experiment.
