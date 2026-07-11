# epistem

Exact multi-party consensus over short text positions. Given a set of
written proposals ("theories") and a set of stakeholders with different
priorities, `epistem` finds the mathematically optimal compromise — the
point that maximizes the *worst-off* party's satisfaction — and tells
you exactly how fragile that compromise is, how sensitive it is to each
party's priorities, and how much a genuinely new proposal could improve
on it. Every answer is an exact linear-program solution, not a heuristic
or a simulation average: for any fixed inputs, `lp_consensus` returns
the same provably-optimal point every time.

Single file (`epistem.py`, ~590 lines), three real dependencies
(`numpy`, `scipy`, `scikit-learn`), zero network calls at runtime.

```bash
pip install numpy scipy scikit-learn
python3 test_epistem.py   # 41/41, ~4 seconds
```

## Quickstart

```python
import numpy as np
import epistem as ep

corpus = [
    ("status_quo",     "Maintain current single family zoning rules unchanged across the corridor, citing neighborhood stability and predictable property values for existing homeowners."),
    ("upzone",         "Rezone the full corridor for mid rise apartment buildings near the transit station, citing regional housing supply shortages and long commute times for essential workers."),
    ("transit_overlay","Allow mid rise apartments only within two blocks of the transit station, preserving existing single family zoning everywhere else in the corridor."),
    ("moratorium",     "Pause all zoning decisions along the corridor for eighteen months pending a comprehensive traffic and infrastructure capacity study."),
    ("density_bonus",  "Keep current zoning limits but offer density bonuses and expedited permitting to any developer who includes below market rate affordable units."),
    ("townhome_middle","Permit townhomes and small multiplexes corridor wide as a middle ground between single family and mid rise apartments, capped at three stories."),
]

W = np.array([
    [0.40, 0.05, 0.20, 0.15, 0.20],   # Homeowners' priority share across 5 dims
    [0.05, 0.40, 0.20, 0.10, 0.25],   # Renters
    [0.05, 0.30, 0.15, 0.05, 0.45],   # Developers
])
party_names = ["Homeowners", "Renters", "Developers"]

profiles = ep.embed(corpus, n_dims=5)
result = ep.lp_consensus(profiles, W, party_names=party_names)
print(result.summary("Zoning consensus"))
```

```
Zoning consensus: Q=0.5349 tension=0.0345 resolved
  Developers           0.6082
  Renters              0.5349
  Homeowners           0.5349
mixture:
  0.8279  status_quo
  0.1721  density_bonus
```

That's the real output of the real code — an 83/17 blend of `status_quo`
and `density_bonus` turns out to be the exact point that maximizes the
worst-off party's score, given these five options and these three
weight vectors. `Q=0.5349` is that worst-off score; `tension` is how
spread out the three parties' individual scores are at that point;
`resolved` means tension is under the deadlock threshold. Full runnable
version with `stress`, `sensitivity`, and `innovate` too is in
[`quickstart.py`](./quickstart.py).

## The mental model

1. **Turn text into geometry.** `embed()` maps each written position to
   a point in `[0.15, 0.85]^k` — a small number of coordinates capturing
   the dominant axes of disagreement across the corpus. Two positions
   that use similar language land close together; positions that argue
   past each other land far apart.
2. **Turn priorities into weight vectors.** Each stakeholder gets a
   vector over the same `k` axes, saying how much they care about each
   one. A stakeholder's satisfaction with any point `v` is just
   `W_i . v` — a weighted sum.
3. **The compromise space is the convex hull of the actual proposals**
   — every weighted blend of the real options on the table — not the
   full coordinate space. This one constraint is what keeps every
   answer meaningful instead of degenerate (more below).
4. **Consensus = maximin.** `lp_consensus` finds the exact blend that
   makes the worst-off stakeholder as satisfied as possible. This is
   the same principle behind Rawlsian bargaining and egalitarian
   cooperative game theory, solved here as a plain linear program.
5. **Everything downstream is either exact or vectorized.** Worst/best
   case bounds are closed-form (`q_worst`/`q_best`), robustness testing
   is one matrix multiply over thousands of sampled scenarios
   (`stress`), sensitivity is repeated exact re-solves
   (`sensitivity`), and exploring beyond the existing options is still
   an exact LP, just with a bounded search box (`innovate`).

## API

### `embed(corpus, n_dims=8, lo=0.15, hi=0.85, ngram_range=(1,2), random_state=42) -> Dict[str, np.ndarray]`

`corpus` is a list of `(name, text)` pairs. Pipeline: sublinear TF-IDF
(unigrams + bigrams) → `TruncatedSVD` to `n_dims` components → per-axis
min-max scaling to `[lo, hi]`.

- **Raises `ValueError`** if there are fewer than `n_dims + 1` texts —
  not enough documents to produce that many genuine (non-degenerate)
  SVD components. The error message names the exact `n_dims` that
  *would* work for your corpus size. An earlier version zero-padded
  short corpora instead of refusing; the padded (fake) dimensions were
  always identical across every theory, and could win `argmin()` in
  `q_worst`/`fragility`, silently misreporting a fabricated dimension
  as a theory's real weak point. That version was removed rather than
  patched — see `test_epistem.py`'s `T2b`, which quantifies why the
  next-most-obvious fix (pad with the ceiling instead of the floor)
  is broken too, just in the opposite direction (it corrupts
  `q_best`/`fragility` instead of `q_worst`).
- **Warns (`UserWarning`), doesn't fail silently**, if two or more
  input texts collapse onto identical coordinates — this happens when
  texts are short and don't carry enough distinguishing vocabulary for
  SVD to tell them apart. The warning tells you which theories
  collided and that longer, more specific text (30+ words) fixes it.
  One case that warning *doesn't* fully fix: two texts that are exact
  word-for-word rearrangements of each other still collide (reordering
  doesn't change unigram counts, and that shared signal outweighs the
  sparse bigram difference once compressed to `n_dims`). When that's
  what happened, the warning says so explicitly and tells you it's a
  wording problem, not a length problem — see `T1d`.

### `q_worst(v)`, `q_best(v)`, `fragility(v) -> float`

Exact worst-case, best-case, and range of a single profile's score
across *every possible* non-negative weight vector, not just the ones
you happened to test. `min_{w in simplex} w.v = min(v)` — the extremum
of a linear functional over a simplex is always at a vertex, so this is
a closed-form O(k) lookup, not a search. `fragility = q_best - q_worst`.
Both clip their input to `[0, 1]`, not `[lo, hi]` — deliberate, because
`innovate()` (below) can propose synthetic points outside `[lo, hi]`
while still inside `[0, 1]`, and these functions need to handle both
sources of input, not just `embed()`'s narrower output range.

### `lp_consensus(profiles, weight_matrix, party_names=None, deadlock_threshold=0.08) -> ConsensusResult`

The core solve. `profiles` is `embed()`'s output (or any
`{name: k-vector}` dict); `weight_matrix` is one non-negative row per
stakeholder. Finds `v* = sum_i lambda_i * v_i` (a convex combination of
the actual input theories) maximizing `min_j (W_j . v*)`, via one
`scipy.optimize.linprog(..., method="highs")` call using the standard
epigraph trick (an auxiliary variable `t`, maximized subject to
`t <= W_j . v*` for every party).

Restricted to the convex hull of the real theories on purpose: the
unrestricted version (`v` free in `[0,1]^k`) is degenerate, because
`v = [1,...,1]` trivially satisfies every non-negative weight vector
that sums to 1 — a perfect answer nobody actually proposed. Restricting
to real, mixable options is what makes `consensus_Q` mean something.

Returns a `ConsensusResult`:
- `consensus_Q` — the worst-off party's exact score at the optimum
- `v_opt` — the optimal blended point
- `mixture` — `{theory_name: weight}` for every theory with >1% of the
  blend (a display cutoff; very small weights are dropped for
  readability, not zeroed by the solver)
- `party_scores`, `tension` (std dev of party scores), `deadlock`
  (`tension > deadlock_threshold`)
- `.dominant` — the single highest-weighted theory in the mixture
- `.summary(name)` — human-readable report (shown above)

If the solver doesn't converge, `lp_consensus` doesn't raise — it warns
(`RuntimeWarning`, explicitly telling you to distrust the result) and
falls back to a uniform mixture, so a single hard-to-solve cartridge
can't take down a batch job partway through. Note `lp_consensus` itself
doesn't check that weight rows are non-negative or sum to 1 — that
validation lives in `load_cartridge()`; calling `lp_consensus` directly
skips it.

### `stress(profiles, weight_matrix=None, n_scenarios=1000, alpha=0.4, seed=None) -> StressReport`

Samples `n_scenarios` adversarial weight vectors from
`Dirichlet(alpha, ..., alpha)` (α<1 biases toward sparse/polarized
weightings — a stakeholder who cares intensely about one axis, rather
than evenly across all of them), scores every theory against every
scenario in one matrix multiply, and reports each theory's empirical
worst/mean alongside the *exact* `q_worst`/`q_best`/`fragility`
(guaranteed to bound the empirical numbers, since they're the true
extrema over the entire simplex, not just the sampled slice). Pass your
real `weight_matrix` in to include your actual stakeholders in the
battery alongside the random ones.

`StressReport.table()` prints a sorted comparison; `.most_robust` /
`.most_fragile` give the theories with the smallest/largest `fragility`.

### `isomorphism(v_a, v_b) -> (r, p)`

Pearson correlation and two-tailed p-value between two profiles.
**Naming note, not a behavior note:** this is correlation, not a
mathematical isomorphism (a structure-preserving bijection) — `r=1.0`
means the two profiles move together linearly across dimensions, not
that they're structurally equivalent. Handles zero-variance input
explicitly (returns `(0.0, 1.0)` instead of letting `scipy.stats.pearsonr`
return a silent `NaN`).

### `sensitivity(profiles, weight_matrix, party_idx, dim_idx, deltas=None, party_names=None, dim_labels=None) -> SensitivityResult`

Re-solves the *exact* `lp_consensus` (not an approximation or a
gradient estimate — a real call per point, since each solve is ~2ms) at
several perturbations to one party's weight on one dimension, moving
`delta` onto that axis and taking it proportionally from the party's
other axes. Answers "how far would this specific party need to shift
before the recommended blend actually changes" — a different question
from `stress`, which asks how bad things could get under an adversarial
or randomly sampled weighting rather than the real one you have.
`.flips` tells you whether the dominant theory actually changes anywhere
across the swept deltas.

### `innovate(profiles, weight_matrix, radius=0.10, center=None) -> InnovationResult`

`lp_consensus` is bounded to the convex hull of proposals that already
exist. `innovate` asks a different question: how much better could a
*genuinely new* proposal do — one nobody has drafted yet — if it could
land anywhere within `radius` (L∞, per-dimension) of the current
consensus point. Still solved exactly, as an LP with box constraints
instead of a simplex-mixture constraint. `radius=0` reproduces
`lp_consensus`'s answer exactly (the box collapses to one point) —
useful as a sanity check that the two solvers agree. `headroom` is how
much `consensus_Q` improves; `v_new` is the coordinates the synthetic
proposal would need.

### Cartridges: `load_cartridge(path)`, `run_cartridge(source, n_scenarios=1000, seed=42)`, `main(argv=None)`

A JSON file format for a full consensus problem:

```json
{
  "name": "Zoning corridor decision",
  "dim_labels": ["density", "transit_access", "affordability", "process", "flexibility"],
  "corpus": [
    {"name": "status_quo", "text": "Maintain current single family zoning..."},
    {"name": "upzone", "text": "Rezone the full corridor for mid rise..."}
  ],
  "weights": {
    "Homeowners": [0.40, 0.05, 0.20, 0.15, 0.20],
    "Renters":    [0.05, 0.40, 0.20, 0.10, 0.25]
  }
}
```

`load_cartridge` validates eagerly — mismatched weight-vector lengths,
weights that don't sum to ~1, a corpus too small for the implied
`n_dims` — so a malformed file fails at load time with a message
telling you exactly what to fix, instead of surfacing as a confusing
result three functions later. `run_cartridge` chains
`embed → lp_consensus → stress` in one call. `main()` is a CLI entry
point:

```bash
python3 -m epistem cartridge1.json cartridge2.json
```

prints a consensus summary and stress table for each file given.

## Why it's built this way

The design choices that most affect correctness all trace back to the
same instinct, visible directly in the source comments and confirmed
independently while testing this: **when there's no principled answer,
refuse and say why, rather than pick a default and hope.**

- Too little text to embed meaningfully → `ValueError`, not a guessed
  padding scheme.
- LP solver fails to converge → loud `RuntimeWarning` and an explicit
  "don't trust this" fallback, not a silent wrong answer.
- A search space that would let the optimizer cheat (the full
  hypercube in `lp_consensus`, an unbounded neighborhood in
  `innovate`) → deliberately constrained to where a real answer is
  possible, with the degenerate alternative named directly in the
  docstring.

The test suite matches this philosophy: it doesn't just check that
functions return *a* value, it cross-checks every non-trivial claim
against an independent method — `q_worst` against `scipy`'s SLSQP
(`T6`), `lp_consensus` against brute-force random search over the
mixture simplex (`T5`), `innovate` the same way over its search box
(`T11`) — and it keeps a record of rejected alternative designs with
the numbers that killed them (`T2b`), not just the one that was kept.
That's a meaningfully higher bar than "the tests pass," and it's the
reason this document can say "exact" and "provably optimal" above
without hedging.

## Known limitations

- **Short, generic text collides.** If two theories don't carry enough
  distinguishing vocabulary, they can land on identical coordinates.
  Loud warning, not silent — but worth writing substantive (30+ word)
  descriptions from the start.
- **Exact word-order permutations collide even at production length.**
  Confirmed at `n_dims=8` with realistic 40+ word prose, not just tiny
  examples (`T1d`). Checked, not just theorized: this does *not*
  distort `lp_consensus`'s actual output (`consensus_Q` was identical
  with the duplicate present or removed in every case tested) — a
  repeated point can't expand the convex hull — so the practical risk
  looks like mixture-report clarity, not a way to buy influence.
- **`stress`'s and `T5`'s random-sampling checks are reliable for the
  problem sizes and profile geometries currently tested, not
  guaranteed at arbitrary scale.** Real embedded profiles tend to
  cluster onto a small number of near-repeated values rather than
  spreading out generically, which makes pure random simplex search
  converge faster than it would for a fully generic/adversarial set of
  points. Not an issue today; worth another look if theory counts grow
  substantially or profiles stop clustering.

## What this repository does *not* include

An earlier planning document for this project described a full
reinforcement-learning facilitation layer — an MDP environment, a PPO
agent, baseline comparisons — in detail, down to specific
hyperparameters. None of that exists in this code, and there's no
evidence it ever did outside that document. If you're looking for it:
it isn't here, and building it would mean designing it fresh against
the primitives above, not recovering something that was lost.

## Files

- `epistem.py` — the library.
- `test_epistem.py` — 41 tests, `python3 test_epistem.py`, no test
  framework dependency (plain assertions + a small runner).
- `quickstart.py` — the runnable version of the example above, plus
  `stress`, `sensitivity`, and `innovate` on the same corpus.
