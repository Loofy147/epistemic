from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.optimize import linprog
from scipy.stats import pearsonr
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import warnings
import torch
import torch.nn as nn
import torch.optim as optim

__all__ = [
    "embed", "lp_consensus", "stress", "q_worst", "q_best", "fragility",
    "ConsensusResult", "StressReport", "isomorphism", "PPOAgent",
]


# ───────────────────────────── embedding ─────────────────────────────────

def embed(
    corpus: List[Tuple[str, str]],
    n_dims: int = 8,
    lo: float = 0.15,
    hi: float = 0.85,
    ngram_range: Tuple[int, int] = (1, 2),
    random_state: int = 42,
    pad_value: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    TF-IDF + TruncatedSVD -> profiles scaled to [lo, hi] per dimension.

    If len(corpus) <= n_dims, SVD produces k < n_dims real components.
    The remaining dimensions are padded with `pad_value`.

    By default, `pad_value` is set to `hi`. This ensures that uninformative
    padded dimensions never falsely win the argmin() bottleneck in downstream
    adversarial calculations (like q_worst/fragility).
    """
    names = [c[0] for c in corpus]
    texts = [c[1] for c in corpus]
    n = len(texts)
    if n < 2:
        raise ValueError(f"embed() needs at least 2 texts, got {n}")

    # Robust default: pad with `hi` to prevent false bottlenecks
    if pad_value is None:
        pad_value = hi

    X = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=1,
        sublinear_tf=True
    ).fit_transform(texts)

    n_features = X.shape[1]
    k = min(n_dims, n - 1, n_features - 1)

    if k >= 1:
        proj = TruncatedSVD(n_components=k, random_state=random_state).fit_transform(X)
        col_min, col_max = proj.min(0), proj.max(0)
        rng = np.where(col_max - col_min < 1e-9, 1.0, col_max - col_min)
        scaled_genuine = lo + (hi - lo) * (proj - col_min) / rng

        if k < n_dims:
            padding = np.full((n, n_dims - k), pad_value)
            scaled = np.hstack([scaled_genuine, padding])
        else:
            scaled = scaled_genuine
    else:
        scaled = np.full((n, n_dims), pad_value)
        k = 0

    result = {names[i]: scaled[i] for i in range(n)}
    if k < n_dims:
        result["__padded_dims__"] = np.arange(k, n_dims)

    dupes = _find_collisions(result)
    if dupes:
        warnings.warn(
            f"embed() produced identical profiles for {dupes} -- these theories "
            f"are now mathematically indistinguishable to lp_consensus/stress. "
            f"This happens when input texts are short and share little "
            f"distinguishing vocabulary beyond common words; TruncatedSVD has "
            f"no signal left to separate them. Fix: write longer, more specific "
            f"descriptions (30+ words with concrete distinguishing details), "
            f"not shorter ones.",
            UserWarning,
            stacklevel=2,
        )
    return result


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
    """Exact worst-case score over all weight vectors in the simplex."""
    return float(np.min(np.clip(v, 0, 1)))


def q_best(v: np.ndarray) -> float:
    """Exact best-case score over all weight vectors in the simplex."""
    return float(np.max(np.clip(v, 0, 1)))


def fragility(v: np.ndarray) -> float:
    """The vulnerability margin between the best and worst-case scenario outcomes."""
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
    """v* = sum_i lambda_i v_i maximising min_j W_j . v*"""
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

    c = np.zeros(N + 1)
    c[-1] = -1.0
    Au = np.zeros((nw, N + 1))
    Au[:, :N] = -(weight_matrix @ T)
    Au[:, -1] = 1.0
    Ae = np.zeros((1, N + 1))
    Ae[0, :N] = 1.0

    res = linprog(c, A_ub=Au, b_ub=np.zeros(nw),
                  A_eq=Ae, b_eq=np.array([1.0]),
                  bounds=[(0, None)] * N + [(0, None)], method="highs")

    if not res.success:
        warnings.warn(
            f"lp_consensus solver failed to converge (status: {res.status}, message: {res.message}). "
            f"Falling back to a uniform mixture of all input theories.",
            RuntimeWarning,
            stacklevel=2
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
    """One matrix multiply covering n_scenarios adversarial weight vectors."""
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
    """Pearson r and p-value between two profiles. Safeguarded against constant inputs."""
    if np.std(v_a) < 1e-9 or np.std(v_b) < 1e-9:
        return 0.0, 1.0
    r, p = pearsonr(v_a, v_b)
    return float(r), float(p)


# ────────────────────────────── PPO AGENT ─────────────────────────────────

class PPOAgent:
    """
    Continuous Proximal Policy Optimization (PPO) facilitator.
    Implements Actor-Critic with Gaussian policy and targeted hyperparameters.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        c1: float = 0.5,
        c2: float = 0.01,
        grad_clip: float = 0.5,
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.c1 = c1
        self.c2 = c2
        self.grad_clip = grad_clip

        # Actor-Critic Networks: Two layers with Tanh activation
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Continuous action: mean (actor output) and trainable log_std
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.optimizer = optim.Adam([
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()},
            {'params': [self.log_std]}
        ], lr=lr)

    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        state_t = torch.FloatTensor(state)
        with torch.no_grad():
            mu = self.actor(state_t)
            std = torch.exp(self.log_std)
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            action_log_prob = dist.log_prob(action).sum(dim=-1)
        return action.numpy(), action_log_prob

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor
    ):
        # Calculate returns and advantages
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)

        returns = torch.stack(returns).detach()
        values = self.critic(states).squeeze()
        advantages = (returns - values).detach()
        # Normalise advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy update
        mu = self.actor(states)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mu, std)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

        loss_actor = -torch.min(surr1, surr2).mean()
        loss_critic = self.c1 * nn.MSELoss()(self.critic(states).squeeze(), returns)
        loss_entropy = -self.c2 * entropy.mean()

        total_loss = loss_actor + loss_critic + loss_entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_([*self.actor.parameters(), *self.critic.parameters(), self.log_std], self.grad_clip)
        self.optimizer.step()

        return total_loss.item()
