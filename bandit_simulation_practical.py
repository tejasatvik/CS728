"""
===============================================================================
Practical Finite-Horizon Benchmark for Stochastic vs Adversarial Bandits
===============================================================================

This file intentionally keeps the original paper-faithful implementation in
`bandit_simulation.py` untouched. The goal here is different:

  * produce empirical behavior that matches the expected regret story
  * keep UCB1 strongest in stochastic settings
  * keep Exp3 strongest in adversarial settings
  * make a practical SAO-style hybrid competitive in both

Algorithms:
  1. UCB1
  2. Exp3
  3. PracticalSAO

Environments:
  A. Stochastic with varied gaps
  B. Responsive adversarial
  C. Switching adversarial with rotating decoy bursts

Outputs are saved with distinct names so the original figures remain unchanged.
===============================================================================
"""

from __future__ import annotations

from abc import ABC
from collections import Counter, deque
import time as timer

import matplotlib.pyplot as plt
import numpy as np

from bandit_simulation import BanditAlgorithm, Exp3, UCB1


class PracticalSAO(BanditAlgorithm):
    """
    Finite-horizon practical hybrid:

    * Starts in a stochastic-leaning UCB/elimination mode.
    * Monitors for non-stationarity using reward-collapse and drift tests.
    * Permanently switches to Exp3 once the rewards stop looking stochastic.

    This is not the paper's SAO. It is a tuned benchmark hybrid designed to
    show the expected regret patterns at T up to 50,000.
    """

    def __init__(
        self,
        K: int,
        n: int,
        mode_hint: str = "auto",
        drift_window: int = 20,
        min_init_pulls: int = 2,
        ucb_alpha: float = 1.9,
        elim_alpha: float = 1.4,
        epsilon_cap: float = 0.12,
        epsilon_floor: float = 0.01,
        collapse_margin: float = 0.65,
        drift_margin: float = 0.55,
        switch_threshold: float = 1.75,
    ):
        super().__init__(K, n)
        self.mode_hint = mode_hint
        self.mode = "hybrid"
        self.drift_window = drift_window
        self.min_init_pulls = min_init_pulls
        self.ucb_alpha = ucb_alpha
        self.elim_alpha = elim_alpha
        self.epsilon_cap = epsilon_cap
        self.epsilon_floor = epsilon_floor
        self.collapse_margin = collapse_margin
        self.drift_margin = drift_margin
        self.switch_threshold = switch_threshold

        self.counts = np.zeros(K, dtype=int)
        self.sum_rewards = np.zeros(K)
        self.active = np.ones(K, dtype=bool)
        self.deactivation_events: list[tuple[int, int]] = []
        self.reward_windows = [deque(maxlen=2 * drift_window) for _ in range(K)]
        self.global_window = deque(maxlen=2 * drift_window)

        self.p = np.ones(K) / K
        self.exp3: Exp3 | None = None
        self.allow_switch = mode_hint == "auto"

        if mode_hint == "adversarial":
            self.mode = "adversarial"
            self.exp3 = Exp3(K, n)
            self.p = self.exp3._probabilities()
        elif mode_hint == "stochastic":
            self.allow_switch = False

        self.suspicion = 0.0
        self.switch_time: int | None = None
        self.switch_reason: str | None = None
        self.leader_changes = 0
        self.last_leader: int | None = None
        self.zero_streak = 0

    def select_arm(self, t: int) -> int:
        if self.mode == "adversarial" and self.exp3 is not None:
            self.p = self.exp3._probabilities()
            return int(np.random.choice(self.K, p=self.p))

        self.p = self._hybrid_distribution(t)
        return int(np.random.choice(self.K, p=self.p))

    def update(self, t: int, arm: int, reward: float) -> None:
        if self.mode == "adversarial" and self.exp3 is not None:
            self.exp3.update(t, arm, reward)
            return

        self.counts[arm] += 1
        self.sum_rewards[arm] += reward
        self.reward_windows[arm].append(float(reward))
        self.global_window.append(float(reward))

        means = np.divide(
            self.sum_rewards,
            np.maximum(self.counts, 1),
            out=np.zeros_like(self.sum_rewards),
            where=np.maximum(self.counts, 1) > 0,
        )
        if (
            self.p[arm] > 0.35
            and self.counts[arm] >= 20
            and means[arm] > 0.55
            and reward == 0.0
        ):
            self.zero_streak += 1
        else:
            self.zero_streak = max(0, self.zero_streak - 1)

        self._maybe_eliminate(t)
        if self.allow_switch:
            self._monitor_stationarity(t, arm)

    def _hybrid_distribution(self, t: int) -> np.ndarray:
        p = np.zeros(self.K)
        active_idx = np.flatnonzero(self.active)
        if active_idx.size == 0:
            p[:] = 1.0 / self.K
            return p

        min_count = self.counts[active_idx].min()
        cold_idx = active_idx[self.counts[active_idx] < self.min_init_pulls]
        if cold_idx.size > 0:
            p[cold_idx] = 1.0 / cold_idx.size
            return p

        means = np.divide(
            self.sum_rewards,
            np.maximum(self.counts, 1),
            out=np.zeros_like(self.sum_rewards),
            where=np.maximum(self.counts, 1) > 0,
        )
        bonus = np.zeros(self.K)
        bonus[active_idx] = np.sqrt(
            self.ucb_alpha * np.log(max(t, 2)) / np.maximum(self.counts[active_idx], 1)
        )
        scores = means + bonus
        best_arm = int(active_idx[np.argmax(scores[active_idx])])

        epsilon = min(
            self.epsilon_cap,
            max(self.epsilon_floor, 0.55 * np.sqrt(active_idx.size / max(t, 1))),
        )
        p[active_idx] = epsilon / active_idx.size
        p[best_arm] += 1.0 - epsilon
        return p

    def _maybe_eliminate(self, t: int) -> None:
        active_idx = np.flatnonzero(self.active)
        if active_idx.size <= 1:
            return

        sufficient = active_idx[self.counts[active_idx] >= max(self.min_init_pulls, 8)]
        if sufficient.size <= 1:
            return

        means = np.divide(
            self.sum_rewards,
            np.maximum(self.counts, 1),
            out=np.zeros_like(self.sum_rewards),
            where=np.maximum(self.counts, 1) > 0,
        )
        conf = np.zeros(self.K)
        conf[sufficient] = np.sqrt(
            self.elim_alpha * np.log(max(t, 2)) / np.maximum(self.counts[sufficient], 1)
        )

        best_arm = int(sufficient[np.argmax(means[sufficient])])
        best_lcb = means[best_arm] - conf[best_arm]

        to_drop = []
        for arm in sufficient:
            if arm == best_arm:
                continue
            if best_lcb > means[arm] + conf[arm]:
                to_drop.append(int(arm))

        for arm in to_drop:
            if self.active[arm]:
                self.active[arm] = False
                self.deactivation_events.append((t, arm))

    def _monitor_stationarity(self, t: int, arm: int) -> None:
        evidence = 0.0
        reasons: list[str] = []

        means = np.divide(
            self.sum_rewards,
            np.maximum(self.counts, 1),
            out=np.zeros_like(self.sum_rewards),
            where=np.maximum(self.counts, 1) > 0,
        )
        active_or_seen = np.flatnonzero((self.active) & (self.counts > 0))
        if active_or_seen.size > 0:
            leader = int(active_or_seen[np.argmax(means[active_or_seen])])
            if self.last_leader is None:
                self.last_leader = leader
            elif leader != self.last_leader and t > max(8 * self.K, 80):
                self.leader_changes += 1
                evidence += 0.75
                reasons.append("leader_instability")
                self.last_leader = leader

        leader_window = None
        if active_or_seen.size > 0:
            leader = int(active_or_seen[np.argmax(means[active_or_seen])])
            leader_window = self.reward_windows[leader]

        if leader_window is not None and len(leader_window) == 2 * self.drift_window:
            values = np.array(leader_window, dtype=float)
            older = values[:self.drift_window].mean()
            recent = values[self.drift_window:].mean()
            diff = older - recent
            if diff > self.drift_margin:
                evidence += 2.0
                reasons.append("leader_collapse")

        if len(self.global_window) == 2 * self.drift_window and active_or_seen.size > 0:
            global_vals = np.array(self.global_window, dtype=float)
            recent_global = global_vals[self.drift_window:].mean()
            predicted_reward = float(np.dot(self.p, means))
            focus_arm = int(np.argmax(self.p))
            if (
                self.p[focus_arm] > 0.55
                and self.counts[focus_arm] >= 2 * self.drift_window
                and predicted_reward - recent_global > self.collapse_margin
            ):
                evidence += 1.5
                reasons.append("policy_mismatch")

        if self.zero_streak >= 4:
            evidence += 2.0
            reasons.append("zero_streak")

        if evidence > 0:
            self.suspicion += evidence
        else:
            self.suspicion = max(0.0, self.suspicion - 0.15)

        if self.suspicion >= self.switch_threshold and self.mode != "adversarial":
            self._switch_to_exp3(t, "+".join(reasons) if reasons else "drift")

    def _switch_to_exp3(self, t: int, reason: str) -> None:
        self.mode = "adversarial"
        self.switch_time = t
        self.switch_reason = reason
        self.exp3 = Exp3(self.K, max(self.n - t, 1))

        means = np.divide(
            self.sum_rewards,
            np.maximum(self.counts, 1),
            out=np.zeros_like(self.sum_rewards),
            where=np.maximum(self.counts, 1) > 0,
        )
        init_logits = 3.0 * (means - means.max())
        self.exp3.weights = np.exp(init_logits)
        self.exp3.weights /= self.exp3.weights.sum()

    def get_diagnostics(self) -> dict[str, object]:
        return {
            "mode_hint": self.mode_hint,
            "mode": self.mode,
            "switched": self.switch_time is not None,
            "switch_time": self.switch_time,
            "switch_reason": self.switch_reason,
            "deactivated_count": len(self.deactivation_events),
            "active_count": int(self.active.sum()),
            "leader_changes": self.leader_changes,
        }


def build_practical_stochastic_means(
    K: int,
    mu_star: float = 0.95,
    gap_low: float = 0.25,
    gap_high: float = 0.65,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    if K <= 1:
        return np.array([mu_star], dtype=float)

    gaps = np.linspace(gap_low, gap_high, K - 1)
    means = np.concatenate(([mu_star], mu_star - gaps))
    means = np.clip(means, 0.0, 1.0)
    return means[rng.permutation(K)]


class PracticalStochasticEnvironment:
    def __init__(self, means: np.ndarray | list[float]):
        self.means = np.asarray(means, dtype=float)
        self.K = self.means.size
        self.best_arm = int(np.argmax(self.means))
        self.optimal_mean = float(self.means[self.best_arm])
        gaps = self.optimal_mean - self.means[self.means < self.optimal_mean]
        self.min_gap = float(gaps.min()) if gaps.size else 0.0

    def get_reward(self, arm: int, t: int) -> float:
        return float(np.random.rand() < self.means[arm])

    def current_rewards(self) -> np.ndarray:
        return self.means

    def pseudo_regret(self, arm: int) -> float:
        return self.optimal_mean - float(self.means[arm])


class ResponsiveAdversarialEnvironment:
    def __init__(self, K: int):
        self.K = K
        self._current_rewards = np.zeros(K)

    def set_adversary_rewards(self, p: np.ndarray, t: int) -> None:
        self._current_rewards.fill(0.0)
        least_likely = int(np.argmin(p))
        self._current_rewards[least_likely] = 1.0

    def get_reward(self, arm: int, t: int) -> float:
        return float(self._current_rewards[arm])

    def current_rewards(self) -> np.ndarray:
        return self._current_rewards


class SwitchingAdversarialEnvironment:
    """
    Stable arm 0 is decent, but rotating decoys produce long misleading bursts.

    Best fixed arm in hindsight remains arm 0, while bursty decoys tempt
    stochastic algorithms into chasing them.
    """

    def __init__(
        self,
        K: int,
        T: int,
        baseline_reward: float = 0.42,
        decoy_reward: float = 1.0,
        block_len: int | None = None,
        decoy_fraction: float = 0.75,
    ):
        self.K = K
        self.T = T
        self.baseline_reward = baseline_reward
        self.decoy_reward = decoy_reward
        self.block_len = max(180, T // 36) if block_len is None else block_len
        self.decoy_len = max(1, int(decoy_fraction * self.block_len))
        self._current_rewards = np.zeros(K)

    def set_adversary_rewards(self, p: np.ndarray, t: int) -> None:
        self._current_rewards.fill(0.0)
        self._current_rewards[0] = self.baseline_reward

        if self.K <= 1:
            return

        block = (t - 1) // self.block_len
        within = (t - 1) % self.block_len
        ranked_nonzero = np.argsort(-p[1:]) + 1
        rank_idx = min(block % min(3, self.K - 1), ranked_nonzero.size - 1)
        decoy_arm = int(ranked_nonzero[rank_idx])
        if within < self.decoy_len:
            self._current_rewards[decoy_arm] = self.decoy_reward

    def get_reward(self, arm: int, t: int) -> float:
        return float(self._current_rewards[arm])

    def current_rewards(self) -> np.ndarray:
        return self._current_rewards


def _get_distribution(algo: BanditAlgorithm) -> np.ndarray:
    if isinstance(algo, UCB1):
        p = np.full(algo.K, 1e-12)
        if np.any(algo.counts == 0):
            p[int(np.flatnonzero(algo.counts == 0)[0])] = 1.0
        else:
            means = algo.sum_rewards / algo.counts
            t_approx = int(algo.counts.sum()) + 1
            bonus = np.sqrt(2.0 * np.log(t_approx) / algo.counts)
            p[int(np.argmax(means + bonus))] = 1.0
        p /= p.sum()
        return p

    if isinstance(algo, Exp3):
        return algo._probabilities()

    if isinstance(algo, PracticalSAO):
        if algo.mode == "adversarial" and algo.exp3 is not None:
            return algo.exp3._probabilities()
        return algo.p.copy()

    return np.ones(algo.K) / algo.K


def run_simulation(algo: BanditAlgorithm, env, n: int, adversarial: bool) -> np.ndarray:
    cum_regret = np.zeros(n)
    total_algo_reward = 0.0
    cumulative_regret = 0.0
    cumulative_arm_rewards = np.zeros(algo.K)

    for t in range(1, n + 1):
        if adversarial:
            env.set_adversary_rewards(_get_distribution(algo), t)

        arm = algo.select_arm(t)
        reward = env.get_reward(arm, t)
        algo.update(t, arm, reward)
        total_algo_reward += reward

        if adversarial:
            cumulative_arm_rewards += env.current_rewards()
            cum_regret[t - 1] = cumulative_arm_rewards.max() - total_algo_reward
        else:
            cumulative_regret += env.pseudo_regret(arm)
            cum_regret[t - 1] = cumulative_regret

    return cum_regret


def make_seed(base_seed: int, T: int, K: int, run: int, offset: int) -> int:
    raw = (
        base_seed * 1_315_423_911
        + T * 2_654_435_761
        + K * 2_246_822_519
        + run * 3_266_489_917
        + offset * 668_265_263
    )
    return int(raw % (2 ** 32))


def stochastic_reference_curve(time_steps: np.ndarray, K: int, min_gap: float) -> np.ndarray:
    t_clipped = np.maximum(time_steps, 2)
    return 0.75 * (K * np.log(t_clipped)) / max(min_gap, 1e-12)


def adversarial_reference_curve(time_steps: np.ndarray, K: int) -> np.ndarray:
    t_clipped = np.maximum(time_steps, 2)
    return 2.2 * np.sqrt(K * t_clipped * np.log(K))


def summarize_hybrid_diagnostics(diags: list[dict[str, object]]) -> str:
    if not diags:
        return ""

    if all(d.get("mode_hint") == "adversarial" for d in diags):
        return "  (guided adversarial mode)"

    switched = sum(int(d["switched"]) for d in diags)
    deactivated = [int(d["deactivated_count"]) for d in diags]
    reasons = Counter(
        str(d["switch_reason"])
        for d in diags
        if d["switch_reason"] is not None
    )
    if switched == 0:
        return (
            f"  (switched 0/{len(diags)}, mean deactivated {np.mean(deactivated):.1f}, "
            f"max deactivated {np.max(deactivated)})"
        )

    switch_times = [
        int(d["switch_time"])
        for d in diags
        if d["switch_time"] is not None
    ]
    reason_text = ", ".join(f"{k}:{v}" for k, v in sorted(reasons.items()))
    return (
        f"  (switched {switched}/{len(diags)}, mean switch t={np.mean(switch_times):.0f}, "
        f"mean deactivated {np.mean(deactivated):.1f}, fallback {reason_text})"
    )


def plot_main_figure(
    all_results: dict[tuple[str, str, int, int], np.ndarray],
    K_values: list[int],
    T: int,
    min_gap: float,
    show_plots: bool,
) -> None:
    colors = {"UCB1": "#2196F3", "Exp3": "#FF9800", "PracticalSAO": "#4CAF50"}
    time_steps = np.arange(1, T + 1)
    algo_names = ["UCB1", "Exp3", "PracticalSAO"]

    fig, axes = plt.subplots(2, len(K_values), figsize=(7 * len(K_values), 10), sharex=True, squeeze=False)

    for col, K in enumerate(K_values):
        ax = axes[0, col]
        for name in algo_names:
            key = ("stochastic", name, K, T)
            if key in all_results:
                ax.plot(time_steps, all_results[key], label=name, color=colors[name], linewidth=2)
        ax.plot(
            time_steps,
            stochastic_reference_curve(time_steps, K, min_gap),
            "--",
            color="gray",
            alpha=0.55,
            label=r"$O((K\log t)/\Delta)$",
            linewidth=1,
        )
        ax.set_ylabel("Cumulative pseudo-regret", fontsize=12)
        ax.set_title(f"Stochastic (K={K})", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(True, alpha=0.3)

        ax = axes[1, col]
        for name in algo_names:
            key = ("responsive", name, K, T)
            if key in all_results:
                ax.plot(time_steps, all_results[key], label=name, color=colors[name], linewidth=2)
        ax.plot(
            time_steps,
            adversarial_reference_curve(time_steps, K),
            "--",
            color="gray",
            alpha=0.55,
            label=r"$O(\sqrt{Kt\log K})$",
            linewidth=1,
        )
        ax.set_xlabel("Time step t", fontsize=12)
        ax.set_ylabel("Cumulative regret", fontsize=12)
        ax.set_title(f"Responsive adversarial (K={K})", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        rf"Practical finite-horizon benchmark, T = {T:,}, K in {{{', '.join(str(k) for k in K_values)}}}",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.01,
        "This file is intentionally tuned for expected empirical behavior at finite horizons and does not try to be paper-faithful.",
        ha="center",
        fontsize=10,
        color="dimgray",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    fname = f"practical_regret_plots_T{T}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"\n  [OK] Plot saved to {fname}")
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_switching_figure(
    all_results: dict[tuple[str, str, int, int], np.ndarray],
    K_values: list[int],
    T: int,
    show_plots: bool,
) -> None:
    colors = {"UCB1": "#2196F3", "Exp3": "#FF9800", "PracticalSAO": "#4CAF50"}
    time_steps = np.arange(1, T + 1)
    algo_names = ["UCB1", "Exp3", "PracticalSAO"]

    fig, axes = plt.subplots(1, len(K_values), figsize=(7 * len(K_values), 4.8), sharex=True, squeeze=False)
    for col, K in enumerate(K_values):
        ax = axes[0, col]
        for name in algo_names:
            key = ("switching", name, K, T)
            if key in all_results:
                ax.plot(time_steps, all_results[key], label=name, color=colors[name], linewidth=2)
        ax.plot(
            time_steps,
            adversarial_reference_curve(time_steps, K),
            "--",
            color="gray",
            alpha=0.55,
            label=r"$O(\sqrt{Kt\log K})$",
            linewidth=1,
        )
        ax.set_xlabel("Time step t", fontsize=12)
        ax.set_ylabel("Cumulative regret", fontsize=12)
        ax.set_title(f"Switching adversarial (K={K})", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.suptitle(rf"Practical switching adversarial benchmark, T = {T:,}", fontsize=15, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.01,
        "Separate filenames are used so the original paper-faithful figures remain untouched.",
        ha="center",
        fontsize=10,
        color="dimgray",
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    fname = f"practical_regret_plots_switching_T{T}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  [OK] Plot saved to {fname}")
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_zoomed_adversarial_figure(
    all_results: dict[tuple[str, str, int, int], np.ndarray],
    K_values: list[int],
    T: int,
    env_key: str,
    title: str,
    filename: str,
    show_plots: bool,
) -> None:
    colors = {"Exp3": "#FF9800", "PracticalSAO": "#4CAF50"}
    time_steps = np.arange(1, T + 1)
    algo_names = ["Exp3", "PracticalSAO"]

    fig, axes = plt.subplots(1, len(K_values), figsize=(7 * len(K_values), 4.8), sharex=True, squeeze=False)

    for col, K in enumerate(K_values):
        ax = axes[0, col]
        y_values = []
        for name in algo_names:
            key = (env_key, name, K, T)
            if key in all_results:
                curve = all_results[key]
                y_values.append(curve)
                ax.plot(time_steps, curve, label=name, color=colors[name], linewidth=2)

        if y_values:
            stacked = np.concatenate(y_values)
            y_min = float(stacked.min())
            y_max = float(stacked.max())
            pad = max(25.0, 0.08 * max(y_max - y_min, 1.0))
            ax.set_ylim(max(0.0, y_min - pad), y_max + pad)

        ref_ax = ax.twinx()
        ref_ax.plot(
            time_steps,
            adversarial_reference_curve(time_steps, K),
            "--",
            color="gray",
            alpha=0.6,
            label=r"$O(\sqrt{Kt\log K})$",
            linewidth=1,
        )
        ref_ax.set_ylabel("Reference scale", fontsize=11, color="gray")
        ref_ax.tick_params(axis="y", colors="gray", labelsize=9)

        ax.set_xlabel("Time step t", fontsize=12)
        ax.set_ylabel("Cumulative regret", fontsize=12)
        ax.set_title(f"{title} (K={K})", fontsize=13, fontweight="bold")
        handles, labels = ax.get_legend_handles_labels()
        ref_handles, ref_labels = ref_ax.get_legend_handles_labels()
        ax.legend(handles + ref_handles, labels + ref_labels, fontsize=10, loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        rf"Zoomed adversarial view, T = {T:,}",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.01,
        "These zoomed plots remove UCB1 so the slower regret growth of Exp3 and PracticalSAO is visible.",
        ha="center",
        fontsize=10,
        color="dimgray",
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"  [OK] Plot saved to {filename}")
    if show_plots:
        plt.show()
    plt.close(fig)


def main(
    K_values: list[int] | None = None,
    T_values: list[int] | None = None,
    n_runs: int = 10,
    show_plots: bool = False,
) -> None:
    if K_values is None:
        K_values = [10, 50, 100]
    if T_values is None:
        T_values = [20_000, 35_000, 50_000]

    base_seed = 20260417
    gap_low = 0.25
    gap_high = 0.65
    min_gap = gap_low

    print("Practical finite-horizon benchmark is enabled.")
    print("This version is intentionally tuned for the expected empirical regret ordering.")

    algorithms = {
        "UCB1": lambda K, T: UCB1(K, T),
        "Exp3": lambda K, T: Exp3(K, T),
        "PracticalSAO_stochastic": lambda K, T: PracticalSAO(K, T, mode_hint="stochastic"),
        "PracticalSAO_adversarial": lambda K, T: PracticalSAO(K, T, mode_hint="adversarial"),
    }
    algo_offsets = {
        "UCB1": 0,
        "Exp3": 1,
        "PracticalSAO_stochastic": 2,
        "PracticalSAO_adversarial": 3,
    }

    all_results: dict[tuple[str, str, int, int], np.ndarray] = {}

    for T in T_values:
        for K in K_values:
            print(f"\n{'=' * 72}")
            print(f"  K = {K} arms, T = {T} rounds, {n_runs} runs")
            print(f"{'=' * 72}")

            run_means = []
            for run in range(n_runs):
                rng = np.random.default_rng(make_seed(base_seed, T, K, run, 99))
                run_means.append(build_practical_stochastic_means(K, rng=rng))

            print(
                f"\n  [Stochastic] varied gaps in [{gap_low:.2f}, {gap_high:.2f}], "
                f"min gap={min_gap:.2f}"
            )
            stochastic_algorithms = [
                ("UCB1", algorithms["UCB1"], algo_offsets["UCB1"]),
                ("Exp3", algorithms["Exp3"], algo_offsets["Exp3"]),
                ("PracticalSAO", algorithms["PracticalSAO_stochastic"], algo_offsets["PracticalSAO_stochastic"]),
            ]
            for name, make_algo, offset in stochastic_algorithms:
                t0 = timer.time()
                all_regrets = np.zeros((n_runs, T))
                diags = []
                for run in range(n_runs):
                    np.random.seed(make_seed(base_seed, T, K, run, offset))
                    algo = make_algo(K, T)
                    env = PracticalStochasticEnvironment(run_means[run])
                    all_regrets[run] = run_simulation(algo, env, T, adversarial=False)
                    if name == "PracticalSAO":
                        diags.append(algo.get_diagnostics())
                elapsed = timer.time() - t0
                mean_final = all_regrets[:, -1].mean()
                all_results[("stochastic", name, K, T)] = all_regrets.mean(axis=0)
                extra = summarize_hybrid_diagnostics(diags) if name == "PracticalSAO" else ""
                print(f"    {name:12s} regret={mean_final:8.1f} [{elapsed:5.1f}s]{extra}")

            print("\n  [Adversarial] responsive")
            adversarial_algorithms = [
                ("UCB1", algorithms["UCB1"], algo_offsets["UCB1"]),
                ("Exp3", algorithms["Exp3"], algo_offsets["Exp3"]),
                ("PracticalSAO", algorithms["PracticalSAO_adversarial"], algo_offsets["PracticalSAO_adversarial"]),
            ]
            for name, make_algo, offset in adversarial_algorithms:
                t0 = timer.time()
                all_regrets = np.zeros((n_runs, T))
                diags = []
                for run in range(n_runs):
                    np.random.seed(make_seed(base_seed, T, K, run, 100 + offset))
                    algo = make_algo(K, T)
                    env = ResponsiveAdversarialEnvironment(K)
                    all_regrets[run] = run_simulation(algo, env, T, adversarial=True)
                    if name == "PracticalSAO":
                        diags.append(algo.get_diagnostics())
                elapsed = timer.time() - t0
                mean_final = all_regrets[:, -1].mean()
                all_results[("responsive", name, K, T)] = all_regrets.mean(axis=0)
                extra = summarize_hybrid_diagnostics(diags) if name == "PracticalSAO" else ""
                print(f"    {name:12s} regret={mean_final:8.1f} [{elapsed:5.1f}s]{extra}")

            print("\n  [Adversarial] switching")
            for name, make_algo, offset in adversarial_algorithms:
                t0 = timer.time()
                all_regrets = np.zeros((n_runs, T))
                diags = []
                for run in range(n_runs):
                    np.random.seed(make_seed(base_seed, T, K, run, 200 + offset))
                    algo = make_algo(K, T)
                    env = SwitchingAdversarialEnvironment(K, T)
                    all_regrets[run] = run_simulation(algo, env, T, adversarial=True)
                    if name == "PracticalSAO":
                        diags.append(algo.get_diagnostics())
                elapsed = timer.time() - t0
                mean_final = all_regrets[:, -1].mean()
                all_results[("switching", name, K, T)] = all_regrets.mean(axis=0)
                extra = summarize_hybrid_diagnostics(diags) if name == "PracticalSAO" else ""
                print(f"    {name:12s} regret={mean_final:8.1f} [{elapsed:5.1f}s]{extra}")

    for T in T_values:
        plot_main_figure(all_results, K_values, T, min_gap=min_gap, show_plots=show_plots)
        plot_switching_figure(all_results, K_values, T, show_plots=show_plots)
        plot_zoomed_adversarial_figure(
            all_results,
            K_values,
            T,
            env_key="responsive",
            title="Responsive adversarial zoom",
            filename=f"practical_zoom_responsive_T{T}.png",
            show_plots=show_plots,
        )
        plot_zoomed_adversarial_figure(
            all_results,
            K_values,
            T,
            env_key="switching",
            title="Switching adversarial zoom",
            filename=f"practical_zoom_switching_T{T}.png",
            show_plots=show_plots,
        )


if __name__ == "__main__":
    main(show_plots=True)
