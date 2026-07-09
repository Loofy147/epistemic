"""
epistem_rl — Standalone Reinforcement Learning environment and PPO training pipeline
for dynamic, multi-party consensus facilitation.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import epistem as ep

# ───────────────────────── Gym Environment ─────────────────────────────

class EpistemConsensusEnv:
    """
    A continuous state/action environment for consensus optimization.
    The agent modifies a 'Working Draft' to reconcile drifting stakeholders.
    """
    def __init__(self, n_static_theories=5, n_stakeholders=3, n_dims=6, max_steps=50):
        self.n_static = n_static_theories
        self.n_stakeholders = n_stakeholders
        self.D = n_dims
        self.max_steps = max_steps
        self.action_low = -0.05
        self.action_high = 0.05

        # Static theories (N*D) + Working Draft (D) + Stakeholder weights (M*D)
        self.obs_dim = (self.n_static + 1 + self.n_stakeholders) * self.D
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.static_profiles = np.random.uniform(0.15, 0.85, (self.n_static, self.D))
        self.working_draft = np.mean(self.static_profiles, axis=0)
        self.stakeholder_coords = np.random.uniform(0.15, 0.85, (self.n_stakeholders, self.D))
        self._normalize_weights()
        return self._get_obs()

    def _normalize_weights(self):
        self.W_norm = self.stakeholder_coords / self.stakeholder_coords.sum(axis=1, keepdims=True)

    def _get_obs(self):
        return np.concatenate([
            self.static_profiles.flatten(),
            self.working_draft.flatten(),
            self.W_norm.flatten()
        ]).astype(np.float32)

    def step(self, action):
        self.current_step += 1
        clipped_action = np.clip(action, self.action_low, self.action_high)
        self.working_draft = np.clip(self.working_draft + clipped_action, 0.15, 0.85)

        # Simulate priority drift (stochastic environmental perturbation)
        drift = np.random.normal(0, 0.01, self.stakeholder_coords.shape)
        self.stakeholder_coords = np.clip(self.stakeholder_coords + drift, 0.15, 0.85)
        self._normalize_weights()

        profiles = {f"Theory_{i}": self.static_profiles[i] for i in range(self.n_static)}
        profiles["Working_Draft"] = self.working_draft

        # Run maximin linear programming consensus
        res = ep.lp_consensus(profiles, self.W_norm)

        # Evaluate draft fragility via scenario stress-testing
        draft_stress = ep.stress({"Working_Draft": self.working_draft}, weight_matrix=self.W_norm, n_scenarios=50)
        draft_fragility = draft_stress.results["Working_Draft"]["fragility"]

        # Reward formulation matching specification
        reward = res.consensus_Q
        draft_weight = res.mixture.get("Working_Draft", 0.0)
        reward += draft_weight * 0.5
        reward -= 0.2 * res.tension
        reward -= 0.1 * draft_fragility

        obs = self._get_obs()
        info = {
            "consensus_Q": res.consensus_Q,
            "tension": res.tension,
            "draft_mixture_weight": draft_weight,
            "draft_fragility": draft_fragility,
            "resolved": not res.deadlock
        }
        return obs, float(reward), self.current_step >= self.max_steps, info


# ─────────────────────── PPO Actor-Critic Model ─────────────────────────

class ActorCritic(nn.Module):
    """Deep Neural Network parameterizing the continuous Gaussian Policy."""
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # Actor network with Tanh activations
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic network providing expected baseline values
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def get_action_and_value(self, state, action=None):
        mean = self.actor(state)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(state)

        return action, log_prob, entropy, value


# ─────────────────────────── Training Loop ─────────────────────────────

def train_ppo(env, num_episodes=100, lr=3e-4, gamma=0.99, clip_eps=0.2, ppo_epochs=4):
    ac = ActorCritic(env.obs_dim, env.D)
    optimizer = optim.Adam(ac.parameters(), lr=lr)

    print(f"Initializing PPO training loop over {num_episodes} episodes...")
    print("-" * 80)

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
        done = False

        # Rollout Phase
        while not done:
            state_t = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                action, log_prob, _, value = ac.get_action_and_value(state_t)

            np_action = np.clip(action.cpu().numpy(), env.action_low, env.action_high)
            next_state, reward, done, info = env.step(np_action)

            states.append(state_t)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            dones.append(done)
            state = next_state

        # Convert rollouts to tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        log_probs = torch.stack(log_probs)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        values = torch.cat(values).squeeze()
        dones = torch.tensor(dones, dtype=torch.float32)

        # Discounted Returns & Advantages calculation
        returns = torch.zeros_like(rewards)
        discounted_sum = 0
        for t in reversed(range(len(rewards))):
            discounted_sum = rewards[t] + gamma * discounted_sum * (1 - dones[t].item())
            returns[t] = discounted_sum

        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO Update Phase
        for _ in range(ppo_epochs):
            _, new_log_probs, entropy, new_values = ac.get_action_and_value(states, actions)
            new_values = new_values.squeeze()

            ratios = torch.exp(new_log_probs - log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(new_values, returns)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
            optimizer.step()

        if ep % 20 == 0 or ep == 1:
            print(f"Episode {ep:3d} | Total Episode Reward: {sum(rewards):7.4f} | Consensus Q: {info['consensus_Q']:.4f} | Final Draft Weight: {info['draft_mixture_weight']:.4f}")

    return ac


if __name__ == "__main__":
    env = EpistemConsensusEnv(n_static_theories=5, n_stakeholders=3, n_dims=6, max_steps=40)
    trained_policy = train_ppo(env, num_episodes=5) # Reduced episodes for quick verification
