"""Alignment-based separation of fault divergence from reconstruction error.

The composition localizer compares the reconstruction against the executed trace
POSITIONALLY. On normal operation the reconstruction has mean NED 0.058 but only
0.527 positional agreement, and the two numbers can only differ that much
through alignment: the reconstruction skips or adds a short run of edges, which
edit distance charges a few operations and positional comparison charges at
every subsequent index. A reconstruction indel at position 4 therefore makes the
"first positional mismatch" a reconstruction artifact rather than the fault,
which is why the naive rule reaches only 37% and why the rule that recovers 60%
had to consult a replay of the shipped binary to tell the two apart.

Aligning first removes the confound without any privileged artifact. Under an
edit-distance alignment, a reconstruction that skipped a block shows up as
indels, while a flipped branch shows up as a SUBSTITUTION at an aligned
position: the controller reached the same point and took a different edge.

Substitutions also occur in normal operation wherever the reconstruction is
simply wrong. Those are learnable: calibrate on a normal session from a
DIFFERENT recording, record which substitutions the reconstruction makes when no
fault is present, and at test time attribute only substitutions that the
calibration never produced. Nothing here reads the shipped binary's trace.
"""
from collections import defaultdict


def align(ref, obs):
    """Levenshtein alignment of ref (reconstruction) to obs (executed).

    Returns a list of (op, i, j) with op in {"match", "sub", "del", "ins"};
    i indexes ref, j indexes obs. "del" is an edge the reconstruction emitted
    that the machine did not execute, "ins" one the machine executed that the
    reconstruction missed.
    """
    n, m = len(ref), len(obs)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        di, dp = d[i], d[i - 1]
        for j in range(1, m + 1):
            c = 0 if ri == obs[j - 1] else 1
            di[j] = min(dp[j] + 1, di[j - 1] + 1, dp[j - 1] + c)
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            c = 0 if ref[i - 1] == obs[j - 1] else 1
            if d[i][j] == d[i - 1][j - 1] + c:
                ops.append(("match" if c == 0 else "sub", i - 1, j - 1))
                i -= 1; j -= 1
                continue
        if i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", i - 1, j)); i -= 1
        else:
            ops.append(("ins", i, j - 1)); j -= 1
    ops.reverse()
    return ops


def substitutions(ref, obs):
    """(ref_token, obs_token) pairs the alignment calls substitutions, in order,
    each with the index into obs where it occurs."""
    return [(ref[i], obs[j], j) for op, i, j in align(ref, obs) if op == "sub"]


class NormalProfile:
    """Substitutions the reconstruction makes on fault-free operation.

    Keyed by (controller state, reconstruction token, observed token). A
    substitution seen during calibration is a reconstruction error and must not
    be attributed to a fault; one never seen is the candidate."""

    def __init__(self, counts=None, min_count=1):
        self.counts = defaultdict(int, counts or {})
        self.min_count = min_count

    def observe(self, state, ref_tok, obs_tok):
        self.counts[(state, ref_tok, obs_tok)] += 1

    def is_known_error(self, state, ref_tok, obs_tok):
        if self.counts.get((state, ref_tok, obs_tok), 0) >= self.min_count:
            return True
        # A substitution unseen in this state but common in another is still a
        # reconstruction quirk rather than evidence of a fault; requiring the
        # state to match would make the rule fire on ordinary state coverage
        # gaps in the calibration recording.
        return any(k[1] == ref_tok and k[2] == obs_tok and v >= self.min_count
                   for k, v in self.counts.items())

    def to_json(self):
        return {f"{s}|{a}|{b}": c for (s, a, b), c in self.counts.items()}

    @staticmethod
    def from_json(d, min_count=1):
        c = {}
        for k, v in d.items():
            s, a, b = k.split("|")
            c[(int(s), int(a), int(b))] = v
        return NormalProfile(c, min_count)


def first_fault_substitution(ref, obs, state, profile):
    """Index into obs of the first substitution the calibration never produced,
    or None. This is the deployable divergence criterion."""
    for a, b, j in substitutions(ref, obs):
        if not profile.is_known_error(state, a, b):
            return j
    return None
