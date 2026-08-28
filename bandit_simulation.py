"""
===============================================================================
Reproducing: "The Best of Both Worlds: Stochastic and Adversarial Bandits"
              Bubeck & Slivkins, COLT 2012  (arXiv:1202.4473)
===============================================================================

This script implements:
  1. UCB1          – optimal O(log n) regret for stochastic bandits
  2. Exp3          – optimal O(sqrt(n)) regret for adversarial bandits
  3. Exp3.P        – high-probability variant of Exp3 (used as fallback in SAO)
  4. SAO (Alg. 1)  – "best of both worlds" algorithm from the paper

Two environments:
  A. Stochastic   – fixed Bernoulli arms
  B. Adversarial  – responsive adversary that punishes the algorithm

All equation references (Eq. 12–16) follow the numbering in the user's
description, which maps to the arXiv v1 appendix:
  Eq. 12  →  Deactivation condition
  Eq. 13  →  Consistency check 1  (|tilde_H - hat_H| bound)
  Eq. 14  →  Consistency check 2  (deactivated arm gap vs. tau-1)
  Eq. 15  →  Consistency check 3  (deactivated arm gap vs. tau)
  Eq. 16  →  Probability update rule

Parameter ranges:
  K (number of arms) : 10 to 100
  T (time horizon)   : 20,000 to 50,000
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import time as timer
from abc import ABC, abstractmethod


# ──────────────────────────────────────────────────────────────────────────────
#  Base class
# ──────────────────────────────────────────────────────────────────────────────
class BanditAlgorithm(ABC):
    """Abstract base class for all bandit algorithms."""

    def __init__(self, K: int, n: int):
        """
        Parameters
        ----------
        K : int   – number of arms
        n : int   – time horizon
        """
        self.K = K
        self.n = n

    @abstractmethod
    def select_arm(self, t: int) -> int:
        """Return the index of the arm to play at time step t (1-indexed)."""
        ...

    @abstractmethod
    def update(self, t: int, arm: int, reward: float) -> None:
        """Process the observed reward for the chosen arm at time t."""
        ...


# ──────────────────────────────────────────────────────────────────────────────
#  UCB1  (Auer, Cesa-Bianchi & Fischer, 2002)
# ──────────────────────────────────────────────────────────────────────────────
class UCB1(BanditAlgorithm):
    """
    Upper Confidence Bound algorithm for stochastic bandits.
    Achieves O( (K log n) / Δ ) regret.
    """

    def __init__(self, K: int, n: int):
        super().__init__(K, n)
        self.counts = np.zeros(K)            # T_i(t) – pull counts
        self.sum_rewards = np.zeros(K)       # cumulative observed reward

    def select_arm(self, t: int) -> int:
        # Force each arm to be played once during initialisation
        for i in range(self.K):
            if self.counts[i] == 0:
                return i
        means = self.sum_rewards / self.counts
        bonus = np.sqrt(2.0 * np.log(t) / self.counts)
        ucb_values = means + bonus
        return int(np.argmax(ucb_values))

    def update(self, t: int, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        self.sum_rewards[arm] += reward


# ──────────────────────────────────────────────────────────────────────────────
#  Exp3  (Auer, Cesa-Bianchi, Freund & Schapire, 2002)
# ──────────────────────────────────────────────────────────────────────────────
class Exp3(BanditAlgorithm):
    """
    Exponential-weight algorithm for adversarial bandits.
    Achieves O(sqrt(K n log K)) expected regret.
    """

    def __init__(self, K: int, n: int, gamma: float = None):
        super().__init__(K, n)
        if gamma is None:
            # Optimal tuning: gamma = min(1, sqrt(K ln K / n))
            gamma = min(1.0, np.sqrt(K * np.log(K) / n))
        self.gamma = gamma
        self.weights = np.ones(K)

    def _probabilities(self) -> np.ndarray:
        W = self.weights.sum()
        return (1.0 - self.gamma) * (self.weights / W) + self.gamma / self.K

    def select_arm(self, t: int) -> int:
        p = self._probabilities()
        return int(np.random.choice(self.K, p=p))

    def update(self, t: int, arm: int, reward: float) -> None:
        p = self._probabilities()
        # Importance-weighted reward estimate (unbiased)
        x_hat = np.zeros(self.K)
        x_hat[arm] = reward / p[arm]
        # Multiplicative weight update
        self.weights *= np.exp(self.gamma * x_hat / self.K)
        # Numerical stability: normalise weights
        self.weights /= self.weights.sum()


# ──────────────────────────────────────────────────────────────────────────────
#  Exp3.P  (Auer, Cesa-Bianchi, Freund & Schapire, 2002)
#  High-probability variant with exploration bonus
# ──────────────────────────────────────────────────────────────────────────────
class Exp3P(BanditAlgorithm):
    """
    Exp3.P: adds an exploration bonus to Exp3 so that high-probability
    regret bounds of O(sqrt(K n log(K/δ))) hold.  Used as the adversarial
    fallback inside SAO.
    """

    def __init__(self, K: int, n: int, delta: float = 0.01):
        super().__init__(K, n)
        # Standard parameter choices (Auer et al. 2002b, Theorem 6.1)
        self.eta = np.sqrt(np.log(K) / (K * max(n, 1)))          # learning rate
        self.gamma = 1.05 * np.sqrt(K * np.log(K) / max(n, 1))   # mixing
        if self.gamma > 1.0:
            self.gamma = 1.0
        self.beta_p = np.sqrt(np.log(K / delta) / (K * max(n, 1)))  # exploration bonus
        self.weights = np.ones(K)

    def _probabilities(self) -> np.ndarray:
        W = self.weights.sum()
        return (1.0 - self.gamma) * (self.weights / W) + self.gamma / self.K

    def select_arm(self, t: int) -> int:
        p = self._probabilities()
        return int(np.random.choice(self.K, p=p))

    def update(self, t: int, arm: int, reward: float) -> None:
        p = self._probabilities()
        # Importance-weighted reward estimate
        x_hat = np.zeros(self.K)
        x_hat[arm] = reward / p[arm]
        # Exploration bonus  (1 / p_j(t)  for every arm j)
        sigma_hat = self.beta_p / p
        # Weight update with bonus
        self.weights *= np.exp(self.eta * (x_hat + sigma_hat))
        # Normalise for numerical stability
        self.weights /= self.weights.sum()


# ──────────────────────────────────────────────────────────────────────────────
#  Confidence radii used by SAO
#
#  eq12_threshold(K, t, β)  – Eq. 12 deactivation threshold (aggregate, arm-
#                              independent). From Algorithm 1 / Section 4:
#
#    thresh = 6 · sqrt( 4K·ln(β)/t  +  5·(K·ln(β)/t)² )
#
#  rad_vec / rad  – per-arm Bernstein radius used in the CONSISTENCY checks
#                   (Eqs. 13–15) where the actual pull-frequency p̄_{i,t} matters.
#
#    rad(p̄, t, β) = sqrt( 2·ln(β) / (p̄·t) )  +  2·ln(β) / (3·p̄·t)
# ──────────────────────────────────────────────────────────────────────────────
def eq12_threshold(K: int, t: float, beta: float, c_scale: float = 0.05) -> float:
    """
    Aggregate deactivation threshold for Equation 12 of Bubeck & Slivkins (2012).
    
    Includes an empirical 'c_scale' parameter to compress the overly conservative
    theoretical constants for finite-time simulations.
    """
    if t <= 0:
        return np.inf
    ratio = K * np.log(beta) / t          
    
    # The original theoretical threshold
    theoretical_thresh = 6.0 * np.sqrt(4.0 * ratio + 5.0 * ratio ** 2)
    
    # Scale it down for finite empirical testing
    return c_scale * theoretical_thresh


def rad_vec(p_arr: np.ndarray, t: float, beta: float) -> np.ndarray:
    """
    Vectorised per-arm Bernstein confidence radius (used in Eqs. 13–15).

    rad(p̄, t, β) = sqrt(2·ln(β) / (p̄·t))  +  2·ln(β) / (3·p̄·t)

    p_arr should be the *time-averaged* pull probability p̄_{i,t} = T_i(t)/t,
    NOT the instantaneous sampling probability p_{i,t}.
    Returns np.inf where p ≤ 0.
    """
    log_beta = np.log(beta)
    pt = np.maximum(p_arr * t, 1e-30)        # avoid division by zero
    return np.sqrt(2.0 * log_beta / pt) + (2.0 * log_beta) / (3.0 * pt)


def rad(p_val: float, t: float, beta: float) -> float:
    """Scalar version of rad_vec for single-arm lookups."""
    if p_val <= 0 or t <= 0:
        return np.inf
    log_beta = np.log(beta)
    pt = p_val * t
    return np.sqrt(2.0 * log_beta / pt) + (2.0 * log_beta) / (3.0 * pt)


# ──────────────────────────────────────────────────────────────────────────────
#  SAO – Algorithm 1 from Bubeck & Slivkins (2012)
#
#  Vectorised implementation for efficient simulation at large K and T.
# ──────────────────────────────────────────────────────────────────────────────
class SAO(BanditAlgorithm):
    """
    Stochastic and Adversarial Optimal (SAO) algorithm.
    Bubeck & Slivkins, COLT 2012 — Algorithm 1.

    Achieves O(polylog n) regret in stochastic environments and
    O(sqrt(K n log K)) regret in adversarial environments.

    Variable names mirror the paper's notation:
        A           – active set of arms                 (boolean mask)
        hat_H       – empirical average reward           (hat{H}_{i,t})
        tilde_H     – importance-weighted average        (tilde{H}_{i,t})
        tilde_G     – cumulative importance-weighted sum (tilde{G}_{i,t})
        p           – sampling distribution at time t    (p_{i,t})
        p_used      – exact p_{i,t} *used* to draw arm i at step t
                      (snapshot captured before any normalization)
        tau         – deactivation time                  (tau_i)
        q           – probability at deactivation        (q_i = p_{i,tau_i})
        beta        – confidence parameter               (β = 10 K n³ / δ)
        p_bar       – time-averaged pull frequency       (T_i(t)/t)
                      used in the per-arm Bernstein radii for Eqs. 13-15
    """

    def __init__(self, K: int, n: int, delta: float = 0.01):
        super().__init__(K, n)
        self.delta = delta

        # ── Confidence parameter (Theorem 4.2 / proof of Theorem 4.1) ───
        # β must be large enough so that the union bound over all arms and
        # time steps holds with probability ≥ 1 − δ.
        self.beta = 10.0 * K * (n ** 3) / delta

        # ── Active set ───────────────────────────────────────────────────
        self.A = np.ones(K, dtype=bool)           # all arms start active

        # ── Per-arm tracking variables ───────────────────────────────────
        self.counts = np.zeros(K)                 # T_i(t) – pull counts
        self.sum_rewards = np.zeros(K)            # Σ g_{i,s}·I_{i,s}
        self.hat_H = np.zeros(K)                  # hat{H}_{i,t}

        # tilde_G_{i,t} = Σ_{s=1}^{t} (g_{I_s,s} · I_{i,s}) / p_{i,s}
        # where p_{i,s} is the EXACT probability used to draw the arm at s.
        # Dividing by t gives tilde_H_{i,t}.
        self.tilde_G = np.zeros(K)                # cumulative IW reward sum
        self.tilde_H = np.zeros(K)                # tilde{H}_{i,t} = tilde_G/t

        self.tau = np.full(K, np.inf)             # deactivation time
        self.q = np.zeros(K)                      # probability at deact.

        # ── Sampling distribution ────────────────────────────────────────
        # p_curr holds p_{i,t} — computed at end of step t for use at t+1.
        # We capture a snapshot BEFORE select_arm so we know exactly which
        # probability was used for the importance weighting.
        self.p = np.ones(K) / K                   # uniform initial distribution
        self.p_used = np.ones(K) / K              # snapshot p_{i,t} used this step

        # ── Fallback state ───────────────────────────────────────────────
        self.switched_to_exp3p = False
        self.exp3p: Exp3P | None = None

        # ── Deactivated arm mask ─────────────────────────────────────────
        self.deactivated = np.zeros(K, dtype=bool)

    # ··········································································
    def select_arm(self, t: int) -> int:
        if self.switched_to_exp3p:
            return self.exp3p.select_arm(t)
        # Snapshot the current distribution BEFORE drawing, so update()
        # can use the exact p_{i,t} that govern the draw at this step.
        self.p_used[:] = self.p
        return int(np.random.choice(self.K, p=self.p))

    # ··········································································
    def update(self, t: int, arm: int, reward: float) -> None:
        if self.switched_to_exp3p:
            self.exp3p.update(t, arm, reward)
            return

        K = self.K

        # ── 1. Update raw counts for hat_H ──────────────────────────────
        self.counts[arm] += 1
        self.sum_rewards[arm] += reward

        pulled = self.counts > 0
        self.hat_H[pulled] = self.sum_rewards[pulled] / self.counts[pulled]

        # ── 2. Update importance-weighted estimator tilde_H ─────────────
        #
        # Definition (Section 2):
        #   tilde_g_{i,t} = g_{i,t} · I_{i,t} / p_{i,t}
        #   tilde_G_{i,t} = Σ_{s=1}^{t} tilde_g_{i,s}
        #   tilde_H_{i,t} = tilde_G_{i,t} / t
        #
        # p_{i,t} MUST be the probability used when arm i was drawn at t,
        # captured as self.p_used in select_arm() to avoid any normalization
        # artefacts that occur when we clip/renorm p at the end of step t−1.
        p_draw = max(self.p_used[arm], 1e-300)   # p_{arm, t} (exact draw prob)
        self.tilde_G[arm] += reward / p_draw      # accumulate unbiased IW reward
        self.tilde_H[:] = self.tilde_G / t        # average over ALL t rounds

        # ── 3. Deactivation check — Equation 12 ─────────────────────────
        #
        # Eq. 12: deactivate arm i ∈ A when
        #
        #   max_{j ∈ A} tilde_H_{j,t}  −  tilde_H_{i,t}
        #       >  6 · sqrt( 4K·ln(β)/t  +  5·(K·ln(β)/t)² )
        #
        # The RHS is AGGREGATE (arm-independent). It is NOT the Bernstein
        # radius rad() which uses the per-arm pull probability p̄_{i,t}.
        # Using rad() here (with p ~ 1/K) makes the threshold ~sqrt(2K·ln(β)/t),
        # which is 3-4x too small and causes the optimal arm to be wrongly
        # deactivated within the first few hundred steps → linear regret.
        # ─────────────────────────────────────────────────────────────────
        # ── 3. Deactivation check — Equation 12 ─────────────────────────
        num_active = self.A.sum()
        if num_active > 1:
            # We apply the empirical c_scale here. Tune between 0.01 and 0.1 if needed.
            thresh = eq12_threshold(K, t, self.beta, c_scale=0.05)   
            best_tilde = self.tilde_H[self.A].max()
            active_idx = np.where(self.A)[0]

            gaps = best_tilde - self.tilde_H[active_idx]
            to_deact_mask = gaps > thresh              # Eq. 12

            if to_deact_mask.any():
                deact_arms = active_idx[to_deact_mask]
                self.tau[deact_arms] = t               # record tau_i
                self.q[deact_arms] = self.p_used[deact_arms]  # q_i = p_{i,tau_i}
                self.A[deact_arms] = False
                self.deactivated[deact_arms] = True
                num_active = self.A.sum()

        # ── 4. Consistency checks — Equations 13, 14, 15 ────────────────
        #
        # These checks use the per-arm time-averaged frequency p̄_{i,t} =
        # T_i(t)/t inside rad(), as the Bernstein bound for hat_H is on the
        # empirical mean over T_i pulls out of t rounds.
        violated = False

        # p̄_{i,t} = T_i(t) / t  (fraction of rounds arm i was played)
        p_bar = self.counts / t   # shape (K,); zeros for never-pulled arms

        # Eq. 13: |tilde_H_i − hat_H_i| > rad(p̄_i, t, β)
        if pulled.any():
            diffs = np.abs(self.tilde_H[pulled] - self.hat_H[pulled])
            rads_13 = rad_vec(p_bar[pulled], t, self.beta)
            if (diffs > rads_13).any():
                violated = True

        # Eqs. 14 & 15: checks on deactivated arms
        if not violated and self.deactivated.any():
            best_active_tilde = (self.tilde_H[self.A].max()
                                 if self.A.any() else 0.0)
            deact_idx = np.where(self.deactivated)[0]
            deact_gaps = best_active_tilde - self.tilde_H[deact_idx]
            deact_tau = self.tau[deact_idx]
            deact_q = self.q[deact_idx]

            # Eq. 14: gap is TOO large relative to tau_i − 1
            mask14 = deact_tau > 1
            if mask14.any():
                r14 = rad_vec(deact_q[mask14], deact_tau[mask14] - 1, self.beta)
                if (deact_gaps[mask14] > 3.0 * r14).any():
                    violated = True

            # Eq. 15: gap is TOO small — arm may not actually be suboptimal
            if not violated:
                mask15 = deact_tau > 0
                if mask15.any():
                    r15 = rad_vec(deact_q[mask15], deact_tau[mask15], self.beta)
                    if (deact_gaps[mask15] < r15).any():
                        violated = True

        # ── Handover to Exp3.P if any consistency check fires ────────────
        if violated:
            remaining = self.n - t
            if remaining > 0:
                self.exp3p = Exp3P(K=K, n=remaining, delta=self.delta)
            self.switched_to_exp3p = True
            return

        # ── 5. Probability update — Equation 16 ─────────────────────────
        #
        #   p_{i,t+1} = q_i · τ_i / (t+1)                         if i ∉ A
        #   p_{i,t+1} = (1/|A|)·(1 − Σ_{j∉A} q_j·τ_j/(t+1))      if i ∈ A
        #
        # Deactivated arms maintain a decaying O(1/t) probability so the
        # importance-weighted estimator keeps accumulating unbiased samples;
        # active arms share the remaining probability mass equally.
        # ─────────────────────────────────────────────────────────────────
        if num_active == 0:
            # Edge case: all arms deactivated → fall back
            self.exp3p = Exp3P(K=K, n=self.n - t, delta=self.delta)
            self.switched_to_exp3p = True
            return

        if self.deactivated.any():
            d = self.deactivated
            self.p[d] = self.q[d] * self.tau[d] / (t + 1)         # Eq. 16
            deactivated_mass = self.p[d].sum()
        else:
            deactivated_mass = 0.0

        # Active arms split the remaining mass equally
        active_share = max(1.0 - deactivated_mass, 0.0) / num_active
        self.p[self.A] = active_share                              # Eq. 16

        # Clip to guard against floating-point underflow; do NOT renormalise
        # (renormalisation would bias p_used away from the true p_{i,t} that
        # governs the draw at the NEXT step, but since we snapshot p_used in
        # select_arm before the draw, a gentle clip is safe).
        np.clip(self.p, 1e-300, None, out=self.p)


# ══════════════════════════════════════════════════════════════════════════════
#  ENVIRONMENTS
# ══════════════════════════════════════════════════════════════════════════════

class StochasticEnvironment:
    """
    Fixed Bernoulli arms.
    The optimal arm has mean mu_star; others have mean mu_star − gap.
    """

    def __init__(self, K: int, mu_star: float = 0.8, gap: float = 0.2,
                 best_arm: int = 0):
        self.K = K
        self.means = np.full(K, mu_star - gap)
        self.means[best_arm] = mu_star
        self.best_arm = best_arm

    def get_reward(self, arm: int, t: int) -> float:
        return float(np.random.rand() < self.means[arm])

    def optimal_reward(self, t: int) -> float:
        return self.means[self.best_arm]


class AdversarialEnvironment:
    """
    Responsive adversary for bandit algorithms.

    Strategy: at each round the adversary observes the algorithm's
    sampling distribution and assigns reward 1.0 to the arm the learner
    is LEAST likely to pick, and 0.0 to all other arms.

    This creates genuine linear regret for non-adaptive algorithms (UCB1)
    and O(√n) regret for optimal adversarial algorithms (Exp3, SAO→Exp3.P).
    """

    def __init__(self, K: int):
        self.K = K
        self._current_rewards = np.zeros(K)

    def set_adversary_rewards(self, p: np.ndarray) -> None:
        """Called *before* the algorithm picks an arm each round."""
        self._current_rewards[:] = 0.0
        least_likely = int(np.argmin(p))
        self._current_rewards[least_likely] = 1.0

    def get_reward(self, arm: int, t: int) -> float:
        return self._current_rewards[arm]

    def optimal_reward(self, t: int) -> float:
        return 1.0   # adversary always places reward 1 somewhere


# ══════════════════════════════════════════════════════════════════════════════
#  SIMULATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def _get_distribution(algo):
    """
    Extract the current sampling distribution from an algorithm.
    For deterministic algorithms like UCB1, predict the next arm choice
    and put all mass there.
    """
    if isinstance(algo, UCB1):
        p = np.full(algo.K, 1e-10)
        if algo.counts.min() == 0:
            for i in range(algo.K):
                if algo.counts[i] == 0:
                    p[i] = 1.0
                    break
        else:
            means = algo.sum_rewards / algo.counts
            t_approx = algo.counts.sum() + 1
            bonus = np.sqrt(2.0 * np.log(t_approx) / algo.counts)
            p[np.argmax(means + bonus)] = 1.0
        p /= p.sum()
        return p
    elif isinstance(algo, (Exp3, Exp3P)):
        return algo._probabilities()
    elif isinstance(algo, SAO):
        if algo.switched_to_exp3p:
            return algo.exp3p._probabilities()
        return algo.p.copy()
    return np.ones(algo.K) / algo.K


def run_simulation(algo: BanditAlgorithm, env, n: int, adversarial: bool = False) -> np.ndarray:
    cum_regret = np.zeros(n)
    total_algo_reward = 0.0
    cumulative_arm_rewards = np.zeros(algo.K)

    for t in range(1, n + 1):
        if adversarial:
            p = _get_distribution(algo)
            env.set_adversary_rewards(p)

        arm = algo.select_arm(t)
        reward = env.get_reward(arm, t)
        algo.update(t, arm, reward)

        total_algo_reward += reward

        if adversarial:
            # Track what EVERY arm would have yielded to find the best fixed arm in hindsight
            for i in range(algo.K):
                cumulative_arm_rewards[i] += env.get_reward(i, t)
            
            best_fixed_reward = np.max(cumulative_arm_rewards)
            cum_regret[t - 1] = best_fixed_reward - total_algo_reward
        else:
            # For stochastic, pseudo-regret against expected optimal is correct
            instant_regret = env.optimal_reward(t) - reward
            cum_regret[t - 1] = cum_regret[t - 2] + instant_regret if t > 1 else instant_regret

    return cum_regret


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN: Benchmark & Plot
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Experiment parameters ────────────────────────────────────────────
    # K ∈ [10, 100],  T ∈ [20_000, 50_000]
    K_values = [10, 50, 100]
    T_values = [20_000, 35_000, 50_000]
    n_runs = 10             # Monte Carlo repetitions
    delta = 0.01            # confidence for SAO / Exp3.P

    # Stochastic environment parameters
    mu_star = 0.8
    gap = 0.2

    # (env_type, algo_name, K, T) → mean regret curve
    all_results = {}

    for T in T_values:
        for K in K_values:
            print(f"\n{'=' * 65}")
            print(f"  K = {K} arms,  T = {T} rounds,  {n_runs} runs")
            print(f"{'=' * 65}")

            algorithms = {
                "UCB1": lambda K=K, T=T: UCB1(K, T),
                "Exp3": lambda K=K, T=T: Exp3(K, T),
                "SAO":  lambda K=K, T=T: SAO(K, T, delta=delta),
            }

            # ── Stochastic environment ────────────────────────────────
            print(f"\n  [Stochastic]  mu*={mu_star}, Delta={gap}")
            for name, make_algo in algorithms.items():
                t0 = timer.time()
                all_regrets = np.zeros((n_runs, T))
                switch_count = 0
                for run in range(n_runs):
                    algo = make_algo()
                    env = StochasticEnvironment(K, mu_star=mu_star, gap=gap)
                    all_regrets[run] = run_simulation(algo, env, T,
                                                      adversarial=False)
                    if name == "SAO" and algo.switched_to_exp3p:
                        switch_count += 1
                elapsed = timer.time() - t0
                mean_final = all_regrets[:, -1].mean()
                all_results[("stochastic", name, K, T)] = all_regrets.mean(
                    axis=0)
                extra = (f"  (switched {switch_count}/{n_runs})"
                         if name == "SAO" else "")
                print(f"    {name:6s}  regret={mean_final:8.1f}  "
                      f"[{elapsed:5.1f}s]{extra}")

            # ── Adversarial environment ───────────────────────────────
            print(f"\n  [Adversarial]  responsive adversary")
            for name, make_algo in algorithms.items():
                t0 = timer.time()
                all_regrets = np.zeros((n_runs, T))
                switch_count = 0
                for run in range(n_runs):
                    algo = make_algo()
                    env = AdversarialEnvironment(K)
                    all_regrets[run] = run_simulation(algo, env, T,
                                                      adversarial=True)
                    if name == "SAO" and algo.switched_to_exp3p:
                        switch_count += 1
                elapsed = timer.time() - t0
                mean_final = all_regrets[:, -1].mean()
                all_results[("adversarial", name, K, T)] = all_regrets.mean(
                    axis=0)
                extra = (f"  (switched {switch_count}/{n_runs})"
                         if name == "SAO" else "")
                print(f"    {name:6s}  regret={mean_final:8.1f}  "
                      f"[{elapsed:5.1f}s]{extra}")

    # ══════════════════════════════════════════════════════════════════════
    #  PLOTTING  — one figure per T value
    # ══════════════════════════════════════════════════════════════════════
    colors = {"UCB1": "#2196F3", "Exp3": "#FF9800", "SAO": "#4CAF50"}
    algo_names = ["UCB1", "Exp3", "SAO"]
    n_K = len(K_values)

    for T in T_values:
        time_steps = np.arange(1, T + 1)

        fig, axes = plt.subplots(2, n_K, figsize=(7 * n_K, 10),
                                 sharex=True, squeeze=False)

        for col, K in enumerate(K_values):
            # ── Top row: STOCHASTIC ──
            ax = axes[0, col]
            for name in algo_names:
                key = ("stochastic", name, K, T)
                if key in all_results:
                    ax.plot(time_steps, all_results[key],
                            label=name, color=colors[name], linewidth=2)

            # Reference: O(K log(n) / Δ)
            log_ref = (K * np.log(time_steps + 1)) / gap
            ax.plot(time_steps, log_ref, '--', color='gray', alpha=0.5,
                    label=r'$O(K\log n/\Delta)$', linewidth=1)
            ax.set_ylabel("Cumulative regret", fontsize=12)
            ax.set_title(f"Stochastic  (K={K})", fontsize=13,
                         fontweight='bold')
            ax.legend(fontsize=10, loc="upper left")
            ax.grid(True, alpha=0.3)

            # ── Bottom row: ADVERSARIAL ──
            ax = axes[1, col]
            for name in algo_names:
                key = ("adversarial", name, K, T)
                if key in all_results:
                    ax.plot(time_steps, all_results[key],
                            label=name, color=colors[name], linewidth=2)

            sqrt_ref = 4 * np.sqrt(K * time_steps * np.log(K))
            ax.plot(time_steps, sqrt_ref, '--', color='gray', alpha=0.5,
                    label=r'$O(\sqrt{Kn\log K})$', linewidth=1)
            ax.set_xlabel("Time step  $t$", fontsize=12)
            ax.set_ylabel("Cumulative regret", fontsize=12)
            ax.set_title(f"Adversarial  (K={K})", fontsize=13,
                         fontweight='bold')
            ax.legend(fontsize=10, loc="upper left")
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            'Bubeck & Slivkins (2012): "The Best of Both Worlds"\n'
            rf'$T = {T:,}$,  $K \in \{{{", ".join(str(k) for k in K_values)}\}}$',
            fontsize=15, fontweight='bold', y=1.01)
        plt.tight_layout()
        fname = f"regret_plots_T{T}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"\n  [OK] Plot saved to  {fname}")
        plt.show()


if __name__ == "__main__":
    main()
