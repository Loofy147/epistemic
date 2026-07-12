"""
epistem — exact multi-party theoretical consensus on theory manifolds.

Three primitives:
  embed(corpus)               TF-IDF + SVD -> normalised profiles, zero network
  lp_consensus(profiles, W)   exact LP, global optimum, one HiGHS call
  stress(profiles)            vectorised adversarial battery + exact greedy worst-case

Single file by design: the previous multi-module package was lost to a sandbox
reset before it was copied out. Consolidating to one file until there's a
reason (size, reuse across unrelated projects) to split it again.
"""
from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.optimize import linprog
from scipy.stats import pearsonr
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

__all__ = [
    "embed", "lp_consensus", "stress", "q_worst", "q_best", "fragility",
    "ConsensusResult", "StressReport", "isomorphism",
    "sensitivity", "SensitivityResult",
    "innovate", "InnovationResult",
    "load_cartridge", "run_cartridge", "main",
]


# ───────────────────────────── embedding ─────────────────────────────────

def embed(
    corpus: List[Tuple[str, str]],
    n_dims: int = 8,
    lo: float = 0.15,
    hi: float = 0.85,
    ngram_range: Tuple[int, int] = (1, 2),
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """
    TF-IDF + TruncatedSVD -> profiles scaled to [lo, hi] per dimension.

    Raises ValueError if the corpus is too small to produce n_dims genuine
    (non-degenerate) components. An earlier version zero-padded short
    corpora instead; every theory tied at the same padded value, and that
    tie could win argmin() in q_worst/fragility, misreporting a fake
    dimension as a theory's real bottleneck. Padding was removed rather
    than patched: there's no non-arbitrary way to decide what a "padded"
    dimension should mean to a weight vector the caller wrote assuming
    n_dims real ones, so the honest fix is to refuse and say why, not guess.
    """
    names = [c[0] for c in corpus]
    texts = [c[1] for c in corpus]
    n = len(texts)
    if n < n_dims + 1:
        raise ValueError(
            f"embed() got {n} texts but n_dims={n_dims} -- TruncatedSVD needs "
            f"at least n_dims+1 texts to produce that many genuine components. "
            f"Either add more theories ({n_dims + 1 - n} more needed) or call "
            f"embed(corpus, n_dims={max(n - 1, 1)})."
        )
    k = min(n_dims, n - 1, 50)

    X = TfidfVectorizer(ngram_range=ngram_range, min_df=1,
                         sublinear_tf=True).fit_transform(texts)
    proj = TruncatedSVD(n_components=k, random_state=random_state).fit_transform(X)

    col_min, col_max = proj.min(0), proj.max(0)
    rng = np.where(col_max - col_min < 1e-9, 1.0, col_max - col_min)
    scaled = lo + (hi - lo) * (proj - col_min) / rng

    result = {names[i]: scaled[i] for i in range(n)}
    dupes = _find_collisions(result)
    if dupes:
        import warnings
        anagram_groups = [g for g in dupes if _is_anagram_group(g, dict(corpus))]
        msg = (
            f"embed() produced identical profiles for {dupes} -- these theories "
            f"are now mathematically indistinguishable to lp_consensus/stress. "
            f"This happens when input texts are short and share little "
            f"distinguishing vocabulary beyond common words; TruncatedSVD has "
            f"no signal left to separate them. Fix: write longer, more specific "
            f"descriptions (30+ words with concrete distinguishing details), "
            f"not shorter ones."
        )
        if anagram_groups:
            msg += (
                f" Note: {anagram_groups} are word-for-word rearrangements of "
                f"each other (identical word multiset, different order) -- this "
                f"is a different situation than coincidental SVD collision above: "
                f"no amount of length will fix it, ngram_range gives partial "
                f"protection but the shared unigram signal still dominates at "
                f"typical n_dims. If these are meant to be distinct theories, "
                f"they need genuinely different wording, not reordered wording."
            )
        warnings.warn(msg, stacklevel=2)
    return result


def _is_anagram_group(names: List[str], text_by_name: Dict[str, str]) -> bool:
    """True if every text in this group of colliding names is an exact
    word-multiset permutation of the others (same words, same counts,
    different order). Distinguishes "these are literally reordered
    duplicates" from "these are genuinely different short texts that
    happened to collapse under dimensionality reduction" -- the two
    situations warrant different fixes (deduplicate vs. write more
    specific text), and the plain "identical profiles" warning alone
    doesn't tell a caller which one they're looking at.

    Tokenizes on whitespace with surrounding punctuation stripped per
    token (not just lowercased) -- reversing word order moves sentence
    punctuation onto a different word (e.g. a trailing "." ends up glued
    to whatever word is now last), which otherwise makes a genuine
    word-for-word reversal register as a false negative here.
    """
    from collections import Counter

    def _tokens(text: str) -> Counter:
        return Counter(w for w in (
            t.strip(".,;:!?\"'()[]") for t in text.lower().split()
        ) if w)

    bags = [_tokens(text_by_name[n]) for n in names]
    return all(b == bags[0] for b in bags[1:]) and bool(bags[0])


def _find_collisions(profiles: Dict[str, np.ndarray], atol: float = 1e-6) -> List[List[str]]:
    """Group theory names whose profiles are numerically identical."""
    names = [n for n in profiles if not n.startswith("__")]
    groups: Dict[Tuple[float, ...], List[str]] = {}
    for n in names:
        key = tuple(np.round(profiles[n], int(-np.log10(atol))))
        groups.setdefault(key, []).append(n)
    return [g for g in groups.values() if len(g) > 1]


# ───────────────────────── exact adversarial ─────────────────────────────

def q_worst(v: np.ndarray) -> float:
    """
    Exact worst-case score over all weight vectors in the simplex.
    min_{w in Delta} w.v = min_i(v_i) -- the LP places all mass on the
    smallest coordinate. O(D), no search, no restarts.
    """
    return float(np.min(np.clip(v, 0, 1)))


def q_best(v: np.ndarray) -> float:
    return float(np.max(np.clip(v, 0, 1)))


def fragility(v: np.ndarray) -> float:
    return q_best(v) - q_worst(v)


# ────────────────────────────── consensus ────────────────────────────────

@dataclass
class ConsensusResult:
    consensus_Q: float
    v_opt: np.ndarray
    mixture: Dict[str, float]
    party_scores: Dict[str, float]
    tension: float
    deadlock: bool

    @property
    def dominant(self) -> Tuple[str, float]:
        if not self.mixture:
            return ("none", 0.0)
        return max(self.mixture.items(), key=lambda x: x[1])

    def summary(self, name: str = "") -> str:
        lines = [f"{name or 'Consensus'}: Q={self.consensus_Q:.4f} "
                 f"tension={self.tension:.4f} "
                 f"{'DEADLOCK' if self.deadlock else 'resolved'}"]
        for p, s in sorted(self.party_scores.items(), key=lambda x: -x[1]):
            lines.append(f"  {p:20s} {s:.4f}")
        lines.append("mixture:")
        for t, l in sorted(self.mixture.items(), key=lambda x: -x[1]):
            lines.append(f"  {l:.4f}  {t}")
        return "\n".join(lines)


def lp_consensus(
    profiles: Dict[str, np.ndarray],
    weight_matrix: np.ndarray,
    party_names: Optional[List[str]] = None,
    deadlock_threshold: float = 0.08,
) -> ConsensusResult:
    """
    v* = sum_i lambda_i v_i  maximising  min_j W_j . v*
    s.t. sum(lambda)=1, lambda>=0.

    Exact LP on the theory manifold (convex hull of actual theories), not the
    full [0,1]^D hypercube -- the hypercube version is degenerate: v=[1,...,1]
    trivially satisfies every non-negative weight vector summing to 1.
    """
    names = [n for n in profiles if not n.startswith("__")]
    T = np.array([profiles[n] for n in names]).T
    N, nw = len(names), weight_matrix.shape[0]

    if party_names is None:
        party_names = [f"Party{i}" for i in range(nw)]
    if len(party_names) != nw:
        raise ValueError(
            f"party_names has {len(party_names)} entries but weight_matrix "
            f"has {nw} rows -- they must match 1:1."
        )

    c = np.zeros(N + 1); c[-1] = -1.0
    Au = np.zeros((nw, N + 1))
    Au[:, :N] = -(weight_matrix @ T)
    Au[:, -1] = 1.0
    Ae = np.zeros((1, N + 1)); Ae[0, :N] = 1.0

    res = linprog(c, A_ub=Au, b_ub=np.zeros(nw),
                  A_eq=Ae, b_eq=np.array([1.0]),
                  bounds=[(0, None)] * N + [(0, None)], method="highs")

    if not res.success:
        import warnings
        warnings.warn(
            f"lp_consensus solver did not converge (status={res.status}: "
            f"{res.message}). Falling back to a uniform mixture of all "
            f"input theories -- treat consensus_Q from this result with "
            f"suspicion, it is not the verified LP optimum.",
            RuntimeWarning, stacklevel=2,
        )
        v_fb = T @ (np.ones(N) / N)
        return ConsensusResult(float(np.min(weight_matrix @ v_fb)), v_fb,
                                {}, {}, 0.0, False)

    lam = np.clip(res.x[:N], 0, None)
    lam = lam / lam.sum() if lam.sum() > 0 else lam
    v_opt = T @ lam
    t_opt = float(res.x[-1])

    mixture = {names[i]: float(lam[i]) for i in range(N) if lam[i] > 0.01}
    party_scores = {party_names[j]: float(np.dot(weight_matrix[j], v_opt))
                     for j in range(nw)}
    tension = float(np.std(list(party_scores.values())))

    return ConsensusResult(t_opt, v_opt, mixture, party_scores, tension,
                            tension > deadlock_threshold)


# ─────────────────────────── stress testing ──────────────────────────────

@dataclass
class StressReport:
    results: Dict[str, dict]

    @property
    def most_robust(self) -> Tuple[str, float]:
        b = min(self.results.items(), key=lambda x: x[1]["fragility"])
        return (b[0], b[1]["fragility"])

    @property
    def most_fragile(self) -> Tuple[str, float]:
        b = max(self.results.items(), key=lambda x: x[1]["fragility"])
        return (b[0], b[1]["fragility"])

    def table(self, dim_labels: Optional[List[str]] = None) -> str:
        lines = [f"{'name':24s} {'worst':>7} {'mean':>7} {'frag':>7}  bottleneck"]
        for n, d in sorted(self.results.items(), key=lambda x: x[1]["fragility"]):
            btn = (dim_labels[d["btn_dim"]]
                   if dim_labels and d["btn_dim"] < len(dim_labels)
                   else f"dim{d['btn_dim']}")
            lines.append(f"{n:24s} {d['worst']:>7.4f} {d['mean']:>7.4f} "
                         f"{d['fragility']:>7.4f}  {btn}")
        return "\n".join(lines)


def stress(
    profiles: Dict[str, np.ndarray],
    weight_matrix: Optional[np.ndarray] = None,
    n_scenarios: int = 1000,
    alpha: float = 0.4,
    seed: Optional[int] = None,
) -> StressReport:
    """One matrix multiply covering n_scenarios adversarial weight vectors,
    plus exact greedy worst/best per theory."""
    names = [n for n in profiles if not n.startswith("__")]
    V = np.array([np.clip(profiles[n], 0, 1) for n in names])
    D = V.shape[1]

    rng = np.random.default_rng(seed)
    adv = rng.dirichlet(np.ones(D) * alpha, n_scenarios)
    if weight_matrix is not None:
        adv = np.vstack([weight_matrix, adv])

    scores = adv @ V.T
    worst_v, mean_v = scores.min(0), scores.mean(0)

    results = {
        names[i]: {
            "worst": float(worst_v[i]),
            "mean": float(mean_v[i]),
            "worst_exact": q_worst(V[i]),
            "best_exact": q_best(V[i]),
            "fragility": fragility(V[i]),
            "btn_dim": int(np.argmin(V[i])),
        }
        for i in range(len(names))
    }
    return StressReport(results)


# ─────────────────────────── isomorphism ─────────────────────────────────

def isomorphism(v_a: np.ndarray, v_b: np.ndarray) -> Tuple[float, float]:
    """
    Pearson r and p-value between two consensus profiles.
    scipy's pearsonr returns (nan, nan) with a ConstantInputWarning if
    either vector has zero variance (e.g. a single-theory consensus, or a
    degenerate all-equal mixture) -- checked explicitly and confirmed
    reproducible; guarded here rather than left to surface as a silent NaN.

    Naming note: this is Pearson correlation, not a mathematical
    isomorphism (a structure-preserving bijection) -- r=1.0 means two
    profiles covary linearly across dimensions, not that they're
    structurally equivalent. Kept as-is here to avoid a breaking rename
    of a public function; worth knowing if anything downstream reads
    more into the score than "these two profiles move together."
    """
    if np.std(v_a) < 1e-9 or np.std(v_b) < 1e-9:
        return 0.0, 1.0
    r, p = pearsonr(v_a, v_b)
    return float(r), float(p)


# ─────────────────────────── sensitivity ──────────────────────────────────

def _shift_weight(w: np.ndarray, dim_idx: int, delta: float) -> np.ndarray:
    """
    Move `delta` of one party's weight onto dim_idx, taking it
    proportionally from the other dimensions, renormalized to sum to 1.
    """
    w = w.copy().astype(float)
    other = np.ones(len(w), dtype=bool)
    other[dim_idx] = False
    new_target = float(np.clip(w[dim_idx] + delta, 0, 1))
    remaining = 1.0 - new_target
    other_sum = w[other].sum()
    if other_sum > 1e-12:
        w[other] = w[other] * (remaining / other_sum)
    else:
        w[other] = remaining / other.sum()
    w[dim_idx] = new_target
    return w


@dataclass
class SensitivityResult:
    party: str
    dim_label: str
    deltas: List[float]
    consensus_Q: List[float]
    dominant: List[Tuple[str, float]]

    @property
    def flips(self) -> bool:
        """Does the dominant theory change anywhere across the swept range?"""
        return len({d[0] for d in self.dominant}) > 1

    def table(self) -> str:
        lines = [f"sensitivity: {self.party} on {self.dim_label}",
                 f"{'delta':>7} {'Q':>8}  dominant"]
        for d, q, dom in zip(self.deltas, self.consensus_Q, self.dominant):
            lines.append(f"{d:>+7.2f} {q:>8.4f}  {dom[0]} ({dom[1]:.3f})")
        return "\n".join(lines)


def sensitivity(
    profiles: Dict[str, np.ndarray],
    weight_matrix: np.ndarray,
    party_idx: int,
    dim_idx: int,
    deltas: Optional[List[float]] = None,
    party_names: Optional[List[str]] = None,
    dim_labels: Optional[List[str]] = None,
) -> SensitivityResult:
    """
    Re-solve the EXACT lp_consensus at each of several perturbations to one
    party's weight on one dimension. Not an approximation or a derivative
    estimate -- the literal exact LP answer at each perturbed point, since
    lp_consensus is cheap enough (~2ms) to just call repeatedly.

    Answers a different question than stress(): stress asks "how bad could
    this get under an adversarial or random weighting"; sensitivity asks
    "given the REAL weights we actually have, how far would this specific
    party need to shift on this specific dimension before the recommended
    mixture changes."
    """
    if deltas is None:
        deltas = [-0.10, -0.05, 0.0, 0.05, 0.10]
    if party_names is None:
        party_names = [f"Party{i}" for i in range(weight_matrix.shape[0])]
    label = (dim_labels[dim_idx] if dim_labels and dim_idx < len(dim_labels)
             else f"dim{dim_idx}")

    qs, doms = [], []
    for d in deltas:
        W2 = weight_matrix.copy()
        W2[party_idx] = _shift_weight(weight_matrix[party_idx], dim_idx, d)
        r = lp_consensus(profiles, W2, party_names)
        qs.append(r.consensus_Q)
        doms.append(r.dominant)

    return SensitivityResult(party_names[party_idx], label, list(deltas), qs, doms)


# ─────────────────────────── innovation ───────────────────────────────────

@dataclass
class InnovationResult:
    consensus_Q: float          # best achievable within the bounded radius
    baseline_Q: float           # the plain lp_consensus() score, for comparison
    v_new: np.ndarray           # the synthetic profile that achieves it
    radius: float
    headroom: float             # consensus_Q - baseline_Q

    def summary(self) -> str:
        return (f"innovation headroom: {self.headroom:+.4f} "
                f"(baseline={self.baseline_Q:.4f} -> {self.consensus_Q:.4f}, "
                f"radius={self.radius:.2f})\n"
                f"synthetic profile: {np.round(self.v_new, 3)}")


def innovate(
    profiles: Dict[str, np.ndarray],
    weight_matrix: np.ndarray,
    radius: float = 0.10,
    center: Optional[np.ndarray] = None,
) -> InnovationResult:
    """
    lp_consensus() finds the exact best point WITHIN the convex hull of
    existing theories -- a genuinely new option (a newly drafted policy, a
    reformulated drug candidate, a job description nobody has actually
    written yet) isn't limited to that hull. Search a bounded neighborhood
    of radius `radius` (L-infinity ball, per dimension, in [0,1]-space)
    around the consensus-optimal point instead, asking: how much better
    could a genuinely novel proposal do, and what would it need to look like.

    Unconstrained search over this question reintroduces the exact
    degenerate bug lp_consensus's manifold constraint exists to prevent
    (an unbounded search always "wants" v=[1,...,1], regardless of whether
    anything resembling that is achievable in reality). Bounding the
    departure by `radius` keeps the answer meaningful: it answers "how much
    could realistic drafting effort buy you," not "what if anything were
    possible." Still solved exactly -- max_v min_j(W_j.v) subject to box
    constraints is still a linear program, just with v itself as the
    decision variable instead of a convex combination of existing theories.

    radius=0 reproduces lp_consensus's answer exactly, since the box
    collapses to the single point v_opt.
    """
    baseline = lp_consensus(profiles, weight_matrix)
    if center is None:
        center = baseline.v_opt
    D = len(center)
    nw = weight_matrix.shape[0]

    lo = np.clip(center - radius, 0, 1)
    hi = np.clip(center + radius, 0, 1)

    c = np.zeros(D + 1); c[-1] = -1.0
    Au = np.zeros((nw, D + 1))
    Au[:, :D] = -weight_matrix
    Au[:, -1] = 1.0
    bounds = [(lo[i], hi[i]) for i in range(D)] + [(0, None)]

    res = linprog(c, A_ub=Au, b_ub=np.zeros(nw),
                  bounds=bounds, method="highs")

    if not res.success:
        return InnovationResult(baseline.consensus_Q, baseline.consensus_Q,
                                 center, radius, 0.0)

    v_new = res.x[:D]
    q_new = float(res.x[-1])
    return InnovationResult(q_new, baseline.consensus_Q, v_new, radius,
                             q_new - baseline.consensus_Q)


# ────────────────────────────── cartridges ────────────────────────────────

def load_cartridge(path: str) -> dict:
    """
    Load a consensus problem from a JSON file:

        {"name": "...",
         "dim_labels": ["...", ...],           # optional
         "corpus":  [{"name": "...", "text": "..."}, ...],
         "weights": {"PartyName": [w0, w1, ...], ...}}

    Validates eagerly so failures happen at load time with an actionable
    message, not as a silent artifact deep in embed()'s output:
      - all weight vectors the same length and summing to ~1
      - dim_labels length matches the weight vector length, if given
      - corpus has enough theories for that many dimensions (see embed())
    """
    import json
    with open(path) as f:
        raw = json.load(f)

    missing = {"name", "corpus", "weights"} - raw.keys()
    if missing:
        raise ValueError(f"cartridge missing required keys: {missing}")

    corpus = [(item["name"], item["text"]) for item in raw["corpus"]]
    weights = {k: np.array(v, dtype=float) for k, v in raw["weights"].items()}
    dim_labels = raw.get("dim_labels")

    lens = {len(v) for v in weights.values()}
    if len(lens) != 1:
        raise ValueError(f"weight vectors have inconsistent lengths: {lens}")
    n_dims = lens.pop()

    if dim_labels and len(dim_labels) != n_dims:
        raise ValueError(
            f"dim_labels has {len(dim_labels)} entries but weight vectors "
            f"have {n_dims} -- they must match."
        )
    for party, w in weights.items():
        s = float(w.sum())
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"weight vector for '{party}' sums to {s:.4f}, not 1.0")
    if len(corpus) < n_dims + 1:
        raise ValueError(
            f"cartridge has {len(corpus)} theories but weight vectors imply "
            f"{n_dims} dimensions -- embed() needs at least {n_dims + 1} "
            f"theories to produce that many genuine dimensions. Add "
            f"{n_dims + 1 - len(corpus)} more theories or shorten the weight vectors."
        )

    return {"name": raw["name"], "corpus": corpus, "weights": weights,
            "dim_labels": dim_labels, "n_dims": n_dims}


def run_cartridge(source, n_scenarios: int = 1000, seed: int = 42) -> dict:
    """
    One-call pipeline: load (if given a path) -> embed -> lp_consensus -> stress.
    `source` is either a path string or an already-loaded cartridge dict
    (as returned by load_cartridge).
    """
    cart = load_cartridge(source) if isinstance(source, str) else source
    np.random.seed(seed)
    profiles = embed(cart["corpus"], n_dims=cart["n_dims"])
    party_names = list(cart["weights"].keys())
    W = np.array(list(cart["weights"].values()))
    consensus = lp_consensus(profiles, W, party_names)
    stress_r = stress(profiles, weight_matrix=W, n_scenarios=n_scenarios, seed=seed)
    return {"name": cart["name"], "profiles": profiles, "consensus": consensus,
            "stress": stress_r, "dim_labels": cart.get("dim_labels"),
            "party_names": party_names, "weight_matrix": W}


def _cli_report(result: dict) -> str:
    return "\n".join([
        f"=== {result['name']} ===",
        result["consensus"].summary(),
        "",
        result["stress"].table(result.get("dim_labels")),
    ])


def main(argv: Optional[List[str]] = None) -> int:
    import sys as _sys
    argv = _sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: python -m epistem <cartridge.json> [more.json ...]")
        return 1
    for path in argv:
        try:
            print(_cli_report(run_cartridge(path)))
            print()
        except (ValueError, FileNotFoundError, KeyError) as e:
            print(f"error loading {path}: {e}")
            return 1
    return 0


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main())
