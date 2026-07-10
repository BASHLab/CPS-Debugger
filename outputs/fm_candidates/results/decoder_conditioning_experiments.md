# Decoder-conditioning experiments on top of the frozen JEPA encoder

This document compares three decoder-only interventions tested on the
frozen ViT-Small JEPA encoder, motivated by the per-state diagnostic
chain
(`results/diagnostics_summary.md`,
 `results/followup_diagnostics_summary.md`,
 `results/wave3_diagnostics_summary.md`).

The encoder was held fixed across all three; only the AR-modern
decoder's training setup or output head changed. Each variant was
evaluated on the deterministic `100K` test subset across the same 5
folds used for BL-2 and the baseline BL-7.

Numbers below are 5-fold median NED [min-max].

## Reference rows

| Variant | state-0 (SWINGUP, 17.6%) | state-1 (BALANCE, 79.4%) | state-2 (RESET, 3.0%) | all |
|---|---|---|---|---|
| Chronos-2 + AR-modern (BL-2) | 0.0639 [0.0561-0.0657] | **0.0132** [0.0104-0.0138] | 0.1042 [0.0713-0.1884] | **0.0251** [0.0203-0.0276] |
| ViT-Small JEPA + AR-modern (frozen-encoder baseline) | 0.0749 [0.0648-0.0835] | 0.0265 [0.0240-0.0285] | 0.0845 [0.0693-0.1448] | 0.0379 [0.0334-0.0386] |

The JEPA encoder lowers state-2 NED by 0.020 over Chronos-2 but
raises state-1 by 0.013 and aggregate by 0.013. The three experiments
below target that trade-off.

## Method 1: State-balanced training-batch sampler

Replace the default uniform shuffle of the training set with a
`WeightedRandomSampler` weighted by inverse FSM-state frequency. At
fold-0 natural rates (state-0: 1.76 %, state-1: 97.9 %, state-2:
0.35 %), the sampler produces approximately 33 %/33 %/33 % batches.
Encoder and decoder architectures are unchanged.

Implementation: `--state-balanced` flag in `ar_modern_train.py`.

Fold-0 result (one fold only; folds 1-4 not run after fold-0
verdict):

| state-0 | state-1 | state-2 | all |
|---|---|---|---|
| 0.0618 | **0.0407** | 0.0785 | 0.0455 |

State-1 collapsed from 0.0265 (frozen-encoder baseline) to 0.0407 —
worse than both BL-2 (0.0132) and the baseline. State-0 and state-2
improved modestly but the BALANCE majority regression makes the
overall trade-off the opposite of what the goal required. Not pursued
further.

## Method 2: Factored opcode / operand decoder head

Replace the joint 648-way softmax with three conditional heads:
opcode (30 + 2 specials), source-PC | opcode (344 + 2 specials,
masked to the legal-set for the chosen opcode), and target-PC |
(opcode, source-PC) (480 + 2 specials, masked similarly). Loss is
the sum of three masked cross-entropies. At sampling time the three
heads are sampled in order with legal-triple masking, so every
emitted token corresponds to a legal `(opcode, op1, op2)` triple in
the vocabulary.

Motivation: the Wave-3 diagnostic found 99.9 % of state-1 errors are
operand-2-only, with 7 distinct target-PC values covering 80 % of
all state-1 errors. Giving target-PC prediction a dedicated head with
explicit conditioning was hypothesised to absorb that error mass.

Implementation: `ar_modern_factored_model.py` + `--use-factored-head`
flag.

Fold-0 result (only fold run):

| state-0 | state-1 | state-2 | all |
|---|---|---|---|
| 0.0767 | 0.0252 | **0.1051** | 0.0367 |

state-2 NED jumped to 0.1051 (worse than the baseline's 0.0845 and
above the kill threshold of 0.095). Plausible reading: with only
~3,000 state-2 tokens in the 100K test set, the conditional
op2-given-`(opcode, op1)` head is severely under-trained on
state-2-frequent prefixes, and the legal-set masking forces
predictions into a small support that the model has not seen enough
of. The aggregate gain on state-1 (0.0252 vs 0.0265 baseline) is
real but does not justify the state-2 regression. Not pursued
further.

## Method 3: Derived sensor-window features as a decoder side input

Pre-compute 16 derived features per tick from the raw 7-channel
500-tick sensor window:

* PLC threshold magnitudes: `|theta - pi|`, `|theta|`, `|theta_d|`,
  `|v|`, `|target_x - current_x|`
* RESET-trigger features: `|x|/MAX_X_SAFE`, `sign(x)`
* libm-input shapes: `cos(theta)`, `sin(theta)`,
  `log(1 - 0.9999 * |x|/MAX_X_DISPL)`,
  `log(1 - 0.9999 * |v|/V_MAX)`
* 5-tick lagged copies of the PLC threshold features

The PLC-related features were z-scored against train-set statistics
and clipped to ±5 sigma; bounded features (cos/sin, ratios, signs)
were used as-is. The 16-vector is broadcast across the K=4 frozen
encoder tokens and concatenated, producing per-tick sensor input of
shape `(N, 4, 272)`. The decoder is unchanged except for the
sensor-emb projection dimension (256 → 272).

Motivation: the diagnostic chain localised the operand-2 errors to
PLC edge-detector branches whose decisions are boolean threshold
crossings of continuous sensor signals; the encoder smooths these.
Making them explicit lets the decoder bypass the encoder's smoothing
for the specific signals the controller branches on.

Implementation: `precompute_thresh_features.py` (writes
`sensor_embeddings_bl7_thresh/`).

5-fold median result:

| state-0 | state-1 | state-2 | all |
|---|---|---|---|
| **0.0552** [0.0511-0.0602] | 0.0172 [0.0164-0.0208] | **0.0738** [0.0552-0.1372] | 0.0269 [0.0242-0.0288] |

Against BL-2: state-0 better by 0.009, state-2 better by 0.030,
state-1 worse by 0.004, aggregate worse by 0.002. Against the
frozen-encoder baseline: every metric better
(state-0 -0.020, state-1 -0.009, state-2 -0.011, aggregate -0.011).

Kept; promoted to the canonical JEPA variant on the leaderboard.

## Outcome

Of the three: **Method 3 wins on every per-state target**. It
maintains the original JEPA encoder's state-2 advantage over BL-2
and closes most of the state-1 majority gap. Methods 1 and 2 hit
their kill criteria (state-1 collapse and state-2 regression
respectively) and are not pursued further.
