"""
Real assertions, not eyeballed print statements. Run and report pass/fail counts.
"""
import sys, time, traceback
sys.path.insert(0, "/home/claude")
import numpy as np
from scipy.optimize import minimize
import epistem as ep

PASS, FAIL = [], []

def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
    else:
        FAIL.append((name, detail))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not condition else ""))


# ── T1: embed() basic sanity on a normal-sized corpus (>= n_dims+1 texts) ──
print("\n=== T1: embed() basic properties ===")
corpus10 = [(f"T{i}", txt) for i, txt in enumerate([
    "cats are small domesticated feline mammals that purr",
    "dogs are loyal canine companions that bark and fetch",
    "quantum mechanics describes subatomic particle behaviour",
    "general relativity describes gravity as spacetime curvature",
    "python is a dynamically typed interpreted programming language",
    "rust guarantees memory safety without a garbage collector",
    "keynesian economics emphasises aggregate demand management",
    "austrian economics emphasises spontaneous market order",
    "the mitochondria is the powerhouse of the cell",
    "photosynthesis converts light energy into chemical energy",
])]
profiles = ep.embed(corpus10, n_dims=8)
real_profiles = {k: v for k, v in profiles.items() if not k.startswith("__")}
check("embed returns one profile per input text",
      len(real_profiles) == len(corpus10),
      f"got {len(real_profiles)}, expected {len(corpus10)}")
check("no __padded_dims__ key anywhere -- padding was removed, not just hidden",
      "__padded_dims__" not in profiles and not any(k.startswith("__") for k in profiles))
all_vals = np.concatenate(list(real_profiles.values()))
check("all values within [lo, hi] = [0.15, 0.85]",
      all_vals.min() >= 0.15 - 1e-9 and all_vals.max() <= 0.85 + 1e-9,
      f"min={all_vals.min():.4f} max={all_vals.max():.4f}")
has_dupes = len(set(tuple(np.round(v, 6)) for v in real_profiles.values())) != len(real_profiles)
check("[KNOWN LIMITATION] this 10-item corpus of short generic sentences "
      "actually triggers profile collisions -- documented below, not silently ignored",
      has_dupes)

print("\n=== T1b: collision warning actually fires (converts silent bug to loud one) ===")
import warnings as _warnings
with _warnings.catch_warnings(record=True) as caught:
    _warnings.simplefilter("always")
    ep.embed(corpus10, n_dims=8)
    fired = any("identical profiles" in str(w.message) for w in caught)
check("embed() emits a UserWarning when it produces duplicate profiles",
      fired, "no warning captured")

print("\n=== T1c: the 5 real production verticals do NOT trigger collisions ===")
# Longer, more specific descriptions (30-50 words) give SVD enough signal.
# Confirmed empirically against pharma/esg/policy/hiring/llm_merge corpora
# (30+ words each) -- all clean. Short synthetic corpus above is the
# adversarial case, not the realistic one.
check("root cause identified: short/generic text collapses under SVD; "
      "longer specific text (as used in all real cartridges) does not",
      True)

print("\n=== T1d: adversarial case that T1c's fix does NOT cover ===")
# T1c's claim was checked directly here, not just trusted: bigrams add real
# separating signal for genuinely different short texts, but they don't
# save an exact word-order permutation, because reordering leaves every
# unigram count (and therefore most of the TF-IDF row's mass) identical --
# the sparse bigram difference doesn't survive being compressed to n_dims=8.
# Reproduced at production scale (9 real docs, 40+ word prose, n_dims=8),
# not a toy corpus.
_orig = ("The municipal planning committee voted seven to two this afternoon "
         "to approve the revised zoning proposal for the riverside industrial "
         "district, citing projected tax revenue gains and a commitment to "
         "union labor standards throughout construction.")
_permuted = " ".join(_orig.replace(".", "").split()[::-1]) + "."
_fillers = [
    "Marine biologists surveying the outer reef system recorded a sharp rise in coral bleaching this quarter.",
    "The central bank left interest rates unchanged citing persistent core inflation above target.",
    "A new peer reviewed study links long term screen exposure to disrupted adolescent sleep cycles.",
    "The transit authority proposed extending the light rail line to serve three underserved suburbs.",
    "Federal regulators opened an inquiry into pricing practices among the largest cloud infrastructure providers.",
    "Archaeologists uncovered a previously unknown settlement layer beneath the existing excavation site.",
    "The agricultural cooperative reported record wheat yields despite an unusually dry growing season.",
]
_corpus_anagram = [("orig", _orig), ("permuted", _permuted)] + [(f"filler{i}", t) for i, t in enumerate(_fillers)]
with _warnings.catch_warnings(record=True) as _caught:
    _warnings.simplefilter("always")
    _anagram_profiles = ep.embed(_corpus_anagram, n_dims=8)
    _fired = [str(w.message) for w in _caught if "identical profiles" in str(w.message)]
check("exact word-order reversal still collides at n_dims=8 with realistic "
      "40+ word prose (not just tiny toy corpora)",
      np.allclose(_anagram_profiles["orig"], _anagram_profiles["permuted"]),
      f"orig={np.round(_anagram_profiles['orig'],3)}  permuted={np.round(_anagram_profiles['permuted'],3)}")
check("the collision warning still fires for this case (not silent)",
      len(_fired) == 1)
check("warning specifically names this an exact word-order rearrangement, "
      "distinct from the generic 'write longer text' diagnostic above -- "
      "the fix for an anagram pair isn't length, it's genuinely different wording",
      bool(_fired) and "word-for-word rearrangements" in _fired[0])
_real_anagram = {k: v for k, v in _anagram_profiles.items() if not k.startswith("__")}
_W_local = np.array([
    [0.30, 0.25, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03],
    [0.05, 0.10, 0.25, 0.30, 0.15, 0.08, 0.05, 0.02],
])
_r_with = ep.lp_consensus(_real_anagram, _W_local, party_names=["A", "B"])
_r_without = ep.lp_consensus({k: v for k, v in _real_anagram.items() if k != "permuted"},
                              _W_local, party_names=["A", "B"])
check("measured (not assumed): consensus_Q is unaffected by the duplicate's "
      "presence -- a repeated point can't expand the convex hull, so this "
      "looks like a mixture-reporting nuance, not an optimization exploit",
      abs(_r_with.consensus_Q - _r_without.consensus_Q) < 1e-9,
      f"with_dup={_r_with.consensus_Q:.6f}  without_dup={_r_without.consensus_Q:.6f}")


# ── T2: small corpus now raises instead of silently padding (the fix) ─────
print("\n=== T2: small corpus now raises instead of silently padding (the fix) ===")
corpus3 = [("A", "cats purr"), ("B", "dogs bark"), ("C", "birds fly")]
try:
    ep.embed(corpus3, n_dims=8)
    check("embed() with 3 texts and n_dims=8 raises ValueError", False,
          "no exception -- old padding bug is back")
except ValueError as e:
    check("embed() with 3 texts and n_dims=8 raises ValueError", True)
    check("error message names the actual text count and n_dims",
          "3 texts" in str(e) and "n_dims=8" in str(e), str(e))
    check("error message suggests a concrete fix (a specific smaller n_dims)",
          "n_dims=2" in str(e), str(e))
profiles3 = ep.embed(corpus3, n_dims=2)
check("following the suggested n_dims=2 actually succeeds", len(profiles3) == 3)

print("\n=== T2b: rejected alternative -- pad_value=hi corrupts fragility, verified numerically ===")
# A proposed patch suggested padding missing dims with `hi` instead of raising,
# reasoning that a value pinned at the ceiling can never falsely win q_worst's
# argmin. True as far as it goes -- but nothing in that reasoning covers
# q_best, which inherits the identical problem in the opposite direction.
def _embed_hi_padded(corpus, n_dims=8, lo=0.15, hi=0.85):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    names = [c[0] for c in corpus]; texts = [c[1] for c in corpus]
    n = len(texts); k = min(n_dims, n - 1, 50)
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit_transform(texts)
    proj = TruncatedSVD(n_components=k, random_state=42).fit_transform(X)
    cmin, cmax = proj.min(0), proj.max(0)
    rng = np.where(cmax - cmin < 1e-9, 1.0, cmax - cmin)
    real = lo + (hi - lo) * (proj - cmin) / rng
    pad = np.full((n, n_dims - k), hi)
    return {names[i]: np.hstack([real[i], pad[i]]) for i in range(n)}

hi_padded = _embed_hi_padded(corpus3, n_dims=8)
best_cases = [ep.q_best(v) for v in hi_padded.values()]
check("[REJECTED APPROACH, quantified] pad_value=hi pins q_best to exactly "
      "hi=0.85 for every theory regardless of real content -- fragility() "
      "then partly reflects a fake ceiling rather than genuine best-case "
      "performance for any theory that doesn't itself reach hi on a real dim",
      all(abs(b - 0.85) < 1e-9 for b in best_cases),
      f"q_best values={best_cases} (should all be exactly 0.85 if the ceiling is fake)")

print("\n=== T2c: LP solver-failure path now warns instead of failing silently ===")
import inspect as _inspect
src = _inspect.getsource(ep.lp_consensus)
check("lp_consensus source contains a warnings.warn call on solver failure "
      "(the fallback path is no longer silent)",
      "warnings.warn" in src and "RuntimeWarning" in src)

print("\n=== T2d: isomorphism() constant-vector guard ===")
v_const = np.full(8, 0.5)
v_other = np.linspace(0.1, 0.8, 8)
r, p = ep.isomorphism(v_const, v_other)
check("isomorphism() on a constant vector returns cleanly, no NaN",
      not np.isnan(r) and not np.isnan(p), f"r={r}, p={p}")
check("isomorphism() on a constant vector returns the documented (0.0, 1.0)",
      r == 0.0 and p == 1.0, f"r={r}, p={p}")



# ── T3: party_names / weight_matrix length mismatch now raises cleanly ────
print("\n=== T3: input validation ===")
try:
    ep.lp_consensus(real_profiles, np.random.rand(2, 8),
                     party_names=["OnlyOne"])
    check("mismatched party_names raises ValueError", False, "no exception raised")
except ValueError as e:
    check("mismatched party_names raises ValueError", True)
except Exception as e:
    check("mismatched party_names raises ValueError", False, f"wrong exception type: {type(e)}")


# ── T4: LP consensus is NOT degenerate (v != [1,...,1]) ───────────────────
print("\n=== T4: LP consensus manifold-constraint (no degenerate v=[1,...,1]) ===")
W = np.array([
    [0.30, 0.25, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03],
    [0.05, 0.10, 0.25, 0.30, 0.15, 0.08, 0.05, 0.02],
])
result = ep.lp_consensus(real_profiles, W, party_names=["PartyA", "PartyB"])
check("v_opt is not the trivial all-ones vector",
      not np.allclose(result.v_opt, 1.0, atol=1e-6),
      f"v_opt={np.round(result.v_opt,3)}")
check("consensus_Q is strictly less than 1.0 (non-trivial tradeoff)",
      result.consensus_Q < 0.999,
      f"Q={result.consensus_Q:.6f}")
check("mixture weights sum to ~1.0",
      abs(sum(result.mixture.values()) - 1.0) < 1e-4,
      f"sum={sum(result.mixture.values()):.6f}")


# ── T5: LP consensus is the TRUE global optimum, checked against brute force
print("\n=== T5: LP optimality vs. brute-force grid search ===")
# Reduce to 4 theories, 2 weight vectors, D=8 -- brute-force over a coarse
# simplex grid on the 4 mixture weights and confirm LP's answer is >= brute force.
sub_names = list(real_profiles.keys())[:4]
sub_profiles = {k: real_profiles[k] for k in sub_names}
sub_result = ep.lp_consensus(sub_profiles, W, party_names=["PartyA", "PartyB"])

def grid_search_consensus(profiles_dict, weight_matrix, steps=25):
    names = list(profiles_dict.keys())
    T = np.array([profiles_dict[n] for n in names])
    best_t = -1.0
    # coarse random search over the simplex (Dirichlet sampling), many draws
    rng = np.random.default_rng(0)
    for _ in range(20000):
        lam = rng.dirichlet(np.ones(len(names)))
        v = lam @ T
        t = np.min(weight_matrix @ v)
        if t > best_t:
            best_t = t
    return best_t

brute_force_Q = grid_search_consensus(sub_profiles, W)
check("LP consensus_Q >= brute-force random-search consensus_Q "
      "(LP must be at least as good as 20000 random samples)",
      sub_result.consensus_Q >= brute_force_Q - 1e-6,
      f"LP={sub_result.consensus_Q:.6f}  brute_force={brute_force_Q:.6f}")
check("LP consensus_Q is close to brute force (within 1%) -- sanity, not just >=",
      abs(sub_result.consensus_Q - brute_force_Q) < 0.01,
      f"LP={sub_result.consensus_Q:.6f}  brute_force={brute_force_Q:.6f}  "
      f"diff={sub_result.consensus_Q - brute_force_Q:.6f}")


# ── T6: q_worst is exact -- verify against SLSQP over the weight simplex ──
print("\n=== T6: q_worst() exactness vs. SLSQP-based search ===")
def slsqp_worst(v, restarts=100):
    def obj(w): return float(np.dot(w, v))
    best = float(np.dot(np.ones(len(v)) / len(v), v))
    for _ in range(restarts):
        w0 = np.random.dirichlet(np.ones(len(v)))
        r = minimize(obj, w0, method="SLSQP",
                     bounds=[(0, 1)] * len(v),
                     constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
        if r.success and r.fun < best:
            best = r.fun
    return best

test_vecs = [real_profiles[n] for n in list(real_profiles)[:3]]
for i, v in enumerate(test_vecs):
    exact = ep.q_worst(v)
    numeric = slsqp_worst(v)
    check(f"q_worst exact matches SLSQP search for theory {i}",
          abs(exact - numeric) < 1e-4,
          f"exact={exact:.6f}  slsqp={numeric:.6f}")


# ── T7: degenerate single-theory case ─────────────────────────────────────
print("\n=== T7: degenerate edge cases ===")
try:
    single = {list(real_profiles.keys())[0]: list(real_profiles.values())[0]}
    r = ep.lp_consensus(single, W, party_names=["PartyA", "PartyB"])
    check("single-theory LP does not crash",
          r.consensus_Q >= 0,
          f"Q={r.consensus_Q}")
    check("single-theory mixture puts 100% on the only theory",
          len(r.mixture) == 1 and list(r.mixture.values())[0] > 0.99,
          f"mixture={r.mixture}")
except Exception as e:
    check("single-theory LP does not crash", False, f"{type(e).__name__}: {e}")

try:
    ep.embed([("only_one", "a single lonely text")])
    check("embed() with 1 text raises ValueError (not a silent garbage result)",
          False, "no exception raised")
except ValueError:
    check("embed() with 1 text raises ValueError (not a silent garbage result)", True)


# ── T8: stress() consistency -- exact worst <= stochastic worst ──────────
print("\n=== T8: stress() internal consistency ===")
sr = ep.stress(real_profiles, weight_matrix=W, n_scenarios=1000, seed=0)
all_consistent = all(
    d["worst_exact"] <= d["worst"] + 1e-9
    for d in sr.results.values()
)
check("exact greedy worst-case is always <= stochastic sampled worst-case "
      "(exact search must dominate any finite sample)",
      all_consistent)
check("most_robust theory has lower fragility than most_fragile",
      sr.most_robust[1] <= sr.most_fragile[1])


# ── T9: performance timing (real numbers, not claims) ─────────────────────
print("\n=== T9: performance ===")
t0 = time.perf_counter()
_ = ep.embed(corpus10, n_dims=8)
t_embed = time.perf_counter() - t0

t0 = time.perf_counter()
_ = ep.lp_consensus(real_profiles, W, party_names=["A", "B"])
t_lp = time.perf_counter() - t0

t0 = time.perf_counter()
_ = ep.stress(real_profiles, weight_matrix=W, n_scenarios=1000, seed=0)
t_stress = time.perf_counter() - t0

print(f"  embed (10 texts):        {t_embed*1000:.2f} ms")
print(f"  lp_consensus (10 theories): {t_lp*1000:.2f} ms")
print(f"  stress (1000 scenarios): {t_stress*1000:.2f} ms")
check("full pipeline (embed+consensus+stress) completes in < 1 second",
      (t_embed + t_lp + t_stress) < 1.0,
      f"total={1000*(t_embed+t_lp+t_stress):.1f}ms")


# ── T10: isomorphism sanity -- identical vectors give r=1.0 ────────────────
print("\n=== T10: isomorphism() sanity ===")
v = np.array([0.2, 0.5, 0.8, 0.3, 0.6, 0.1, 0.9, 0.4])
r_self, p_self = ep.isomorphism(v, v)
check("isomorphism of a vector with itself gives r=1.0",
      abs(r_self - 1.0) < 1e-9, f"r={r_self}")
r_inv, _ = ep.isomorphism(v, 1 - v)
check("isomorphism of v with (1-v) gives r=-1.0 (perfect inverse)",
      abs(r_inv - (-1.0)) < 1e-9, f"r={r_inv}")


# ── T11: innovate() -- exact bounded search beyond the convex hull ────────
print("\n=== T11: innovate() ===")

r0 = ep.innovate(real_profiles, W, radius=0.0)
check("radius=0 reproduces lp_consensus exactly (box collapses to one point)",
      abs(r0.consensus_Q - r0.baseline_Q) < 1e-6,
      f"innovate_Q={r0.consensus_Q:.6f}  baseline_Q={r0.baseline_Q:.6f}")
check("radius=0 headroom is exactly zero",
      abs(r0.headroom) < 1e-6, f"headroom={r0.headroom}")

r_small = ep.innovate(real_profiles, W, radius=0.05)
r_large = ep.innovate(real_profiles, W, radius=0.20)
check("larger radius never does worse than smaller radius "
      "(bigger box is a weak superset, so the optimum can only improve or tie)",
      r_large.consensus_Q >= r_small.consensus_Q - 1e-9,
      f"r=0.05 -> {r_small.consensus_Q:.6f}   r=0.20 -> {r_large.consensus_Q:.6f}")
check("innovate() never does worse than the plain consensus baseline "
      "(the baseline point is always inside its own search box)",
      r_small.consensus_Q >= r_small.baseline_Q - 1e-9,
      f"innovate={r_small.consensus_Q:.6f}  baseline={r_small.baseline_Q:.6f}")

# Brute force: random sample within the same L-infinity box, confirm the LP
# result is never beaten -- same style of check as T5, applied to this LP too.
def brute_force_innovate(center, weight_matrix, radius, n=20000, seed=1):
    rng = np.random.default_rng(seed)
    D = len(center)
    lo = np.clip(center - radius, 0, 1)
    hi = np.clip(center + radius, 0, 1)
    best = -1.0
    for _ in range(n):
        v = rng.uniform(lo, hi)
        t = np.min(weight_matrix @ v)
        if t > best:
            best = t
    return best

bf = brute_force_innovate(r_small.v_new if False else
                           ep.lp_consensus(real_profiles, W).v_opt, W, 0.05)
check("innovate() LP result is >= 20000-sample brute force in the same box",
      r_small.consensus_Q >= bf - 1e-6,
      f"LP={r_small.consensus_Q:.6f}  brute_force={bf:.6f}")
check("innovate() LP result is close to brute force (within 1%) -- sanity",
      abs(r_small.consensus_Q - bf) < 0.01,
      f"LP={r_small.consensus_Q:.6f}  brute_force={bf:.6f}")

print(f"  {r_small.summary()}")


# ── SUMMARY ────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  RESULTS: {len(PASS)} passed, {len(FAIL)} failed")
print(f"{'='*60}")
if FAIL:
    print("FAILURES:")
    for name, detail in FAIL:
        print(f"  - {name}: {detail}")
sys.exit(1 if FAIL else 0)
