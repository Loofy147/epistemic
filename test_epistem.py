import numpy as np
import pytest
import warnings
import time
from scipy.optimize import minimize
from epistem import (
    embed, lp_consensus, stress, q_worst, q_best, fragility,
    isomorphism, ConsensusResult, StressReport, PPOAgent
)

# T1: embed Sanity
def test_t1_embed_sanity():
    corpus = [
        ("cat", "Domestic cats are small carnivorous mammals."),
        ("dog", "Dogs are domesticated descendants of wolves."),
        ("bird", "Birds are a group of warm-blooded vertebrates."),
        ("science", "Science is a systematic enterprise that builds knowledge."),
        ("physics", "Physics is the natural science that studies matter."),
        ("biology", "Biology is the scientific study of life."),
        ("economics", "Economics is the social science that studies production."),
        ("finance", "Finance is the study of money and investments."),
        ("math", "Mathematics includes the study of such topics as quantity."),
        ("history", "History is the systematic study and documentation of the past.")
    ]
    n_dims = 8
    profiles = embed(corpus, n_dims=n_dims)

    # Exactly one profile per input, excluding metadata
    names = [c[0] for c in corpus]
    assert len([k for k in profiles if k in names]) == len(names)
    assert "__padded_dims__" not in profiles

    # Boundaries [0.15, 0.85]
    for name in names:
        assert np.all(profiles[name] >= 0.15 - 1e-9)
        assert np.all(profiles[name] <= 0.85 + 1e-9)

# T1b/c: Profile Collisions
def test_t1bc_profile_collisions():
    corpus = [
        ("a", "the"),
        ("b", "the")
    ]
    with pytest.warns(UserWarning, match="identical profiles"):
        embed(corpus, n_dims=4)

# T2: Padding Bug
def test_t2_padding_bug():
    corpus = [
        ("t1", "The quick brown fox jumps over the lazy dog."),
        ("t2", "A fast dark fox leaps above a tired hound."),
        ("t3", "Swift onyx vulpine hops over an inactive canine.")
    ]
    n_dims = 8
    profiles = embed(corpus, n_dims=n_dims, pad_value=0.85)

    assert "__padded_dims__" in profiles
    padded_dims = profiles["__padded_dims__"]
    assert len(padded_dims) > 0

    for name in ["t1", "t2", "t3"]:
        # Verify padded dimensions are set to pad_value (0.85)
        assert np.allclose(profiles[name][padded_dims], 0.85)

# T3: Input Validation
def test_t3_input_validation():
    profiles = {"t1": np.random.rand(8), "t2": np.random.rand(8)}
    W = np.random.rand(2, 8)
    # 2 rows in W, but only 1 party name
    with pytest.raises(ValueError, match="party_names has 1 entries but weight_matrix has 2 rows"):
        lp_consensus(profiles, W, party_names=["Party1"])

# T4: LP Non-Degeneracy
def test_t4_lp_non_degeneracy():
    profiles = {
        "A": np.array([0.8, 0.2, 0.5, 0.5]),
        "B": np.array([0.2, 0.8, 0.5, 0.5])
    }
    W = np.eye(4) # Each dimension is a stakeholder
    res = lp_consensus(profiles, W)

    # Not trivial degenerate vector
    assert not np.allclose(res.v_opt, 0.85)
    assert not np.allclose(res.v_opt, 0.15)

    # Consensus score < 1
    assert res.consensus_Q < 1.0

    # Sum of weights is 1
    assert np.isclose(sum(res.mixture.values()), 1.0, atol=1e-4)

# T5: LP Optimality
def test_t5_lp_optimality():
    rng = np.random.default_rng(42)
    n_theories = 5
    n_dims = 8
    profiles = {f"T{i}": rng.uniform(0.15, 0.85, n_dims) for i in range(n_theories)}
    W = rng.dirichlet(np.ones(n_dims), size=3)

    res = lp_consensus(profiles, W)

    # Brute force random search
    n_samples = 2000
    best_brute = -1.0
    T_mat = np.array([profiles[f"T{i}"] for i in range(n_theories)]).T
    for _ in range(n_samples):
        lam = rng.dirichlet(np.ones(n_theories))
        v = T_mat @ lam
        q = np.min(W @ v)
        if q > best_brute:
            best_brute = q

    assert res.consensus_Q >= best_brute - 1e-7
    # Ensure brute force is close to LP (within 5% given sample size)
    assert best_brute >= res.consensus_Q * 0.95

# T6: q_worst Exactness
def test_t6_q_worst_exactness():
    rng = np.random.default_rng(42)
    v = rng.uniform(0.15, 0.85, 8)

    q_ana = q_worst(v)

    # Numerical SLSQP
    def obj(w):
        return np.dot(w, v)

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = [(0, 1) for _ in range(8)]

    res = minimize(obj, x0=np.ones(8)/8, method='SLSQP', bounds=bounds, constraints=cons)

    assert np.isclose(q_ana, res.fun, atol=1e-7)

# T7: Boundary Sizes
def test_t7_boundary_sizes():
    # Single candidate theory in lp_consensus
    profiles = {"T1": np.array([0.5, 0.6])}
    W = np.array([[1.0, 0.0], [0.0, 1.0]])
    res = lp_consensus(profiles, W)
    assert np.isclose(res.mixture["T1"], 1.0)

    # Single-text error in embed
    with pytest.raises(ValueError, match="needs at least 2 texts"):
        embed([("t1", "text")], n_dims=8)

# T8: Stress Consistency
def test_t8_stress_consistency():
    profiles = {
        "robust": np.array([0.5, 0.5, 0.5]),
        "fragile": np.array([0.15, 0.85, 0.5])
    }
    report = stress(profiles, seed=42)

    # Exact worst-case <= stochastic worst-case
    for name in ["robust", "fragile"]:
        assert report.results[name]["worst_exact"] <= report.results[name]["worst"] + 1e-9

    assert report.results["robust"]["fragility"] < report.results["fragile"]["fragility"]

# T9: Execution Budget
def test_t9_execution_budget():
    corpus = [("T"+str(i), "Some distinct text for theory " + str(i)) for i in range(20)]

    start = time.perf_counter()
    profiles = embed(corpus, n_dims=8)
    W = np.random.rand(5, 8)
    lp_consensus(profiles, W)
    stress(profiles, n_scenarios=1000)
    end = time.perf_counter()

    duration = end - start
    assert duration < 1.0

# T10: Isomorphism
def test_t10_isomorphism():
    v = np.array([0.2, 0.4, 0.6, 0.8])

    # Identity
    r_id, _ = isomorphism(v, v)
    assert np.isclose(r_id, 1.0)

    # Inverse (complement)
    # v_inv = 1 - v
    # pearson(v, 1-v) should be -1
    r_inv, _ = isomorphism(v, 1.0 - v)
    assert np.isclose(r_inv, -1.0)

if __name__ == "__main__":
    pytest.main([__file__])

# T11: PPO Agent Concordance
def test_t11_ppo_agent_concordance():
    import torch
    state_dim = 16
    action_dim = 4
    agent = PPOAgent(state_dim, action_dim)

    # Check network layers
    assert len(list(agent.actor.children())) == 5 # 3 Linear + 2 Tanh
    assert len(list(agent.critic.children())) == 5

    # Select action
    state = np.random.rand(state_dim).astype(np.float32)
    action, log_prob = agent.select_action(state)
    assert action.shape == (action_dim,)
    assert isinstance(log_prob, torch.Tensor)

    # Update step
    states = torch.randn(10, state_dim)
    actions = torch.randn(10, action_dim)
    old_log_probs = torch.randn(10)
    rewards = torch.randn(10)
    dones = torch.zeros(10)

    loss = agent.update(states, actions, old_log_probs, rewards, dones)
    assert isinstance(loss, float)
