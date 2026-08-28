# Stochastic and Adversarial Bandits

This repository contains the implementation of the SAO (Stochastic and Adversarial Optimal) algorithm, based on "The Best of Both Worlds: Stochastic and Adversarial Bandits" by Sébastien Bubeck & Aleksandrs Slivkins (COLT 2012)[cite: 2, 4]. 

## Overview
Classic bandit algorithms face a strict tradeoff: UCB1 provides optimal O(log n) regret in stochastic environments but fails with O(n) linear regret against adversaries[cite: 2, 4]. Conversely, Exp3 provides robust O(sqrt(n)) regret against adversaries but is suboptimal in benign stochastic settings[cite: 2, 4]. The SAO algorithm bridges this gap, achieving near-optimal theoretical regret in both environments up to polylogarithmic factors[cite: 4].

## Implemented Algorithms
* **UCB1**: Upper Confidence Bound algorithm optimized for stochastic bandits[cite: 2].
* **Exp3 & Exp3.P**: Exponential-weight algorithms for adversarial bandits, with Exp3.P acting as a high-probability fallback[cite: 2].
* **SAO (Algorithm 1)**: The paper-faithful implementation that uses consistency checks to detect adversarial manipulation[cite: 2].
* **Practical SAO**: A finite-horizon hybrid algorithm tuned for time steps between T = 20,000 and 50,000[cite: 3, 4]. 

## SAO Consistency Framework
The SAO algorithm maintains an active set of arms and monitors environmental stationarity by comparing two estimators:
* **Importance-Weighted Estimator**: Estimates reward by boosting observations from rarely played arms[cite: 4].
* **Empirical Estimator**: A simple running average of observed rewards[cite: 4].

In a stochastic environment, these estimators converge to the same true mean[cite: 4]. If an adversary manipulates the rewards, the estimators diverge, triggering a consistency violation and forcing a permanent safety fallback to Exp3.P[cite: 4].

## Practical Finite-Horizon Enhancements
The theoretical asymptotic thresholds (Eq. 12) in the original SAO algorithm are too conservative for finite horizons, meaning the algorithm rarely deactivates suboptimal arms at T = 20,000 to 50,000[cite: 4]. To bypass these theoretical bottlenecks, the `practical_hybrid.py` implementation introduces:
* Aggressive arm elimination for clearly suboptimal arms[cite: 3].
* Drift detection heuristics, including leader instability and reward-collapse tests, to monitor for non-stationarity[cite: 3].
* A permanent switch to Exp3 once rewards stop exhibiting stochastic behavior[cite: 3].

## Repository Structure
* `bandit_simulation.py`: Contains the strict, paper-faithful implementation of UCB1, Exp3, Exp3.P, and SAO algorithms[cite: 2, 3].
* `bandit_simulation_practical.py`: Contains the Practical SAO implementation benchmarked across stochastic, responsive adversarial, and switching adversarial environments[cite: 3].
* `CS728_best_of_both_worlds.pdf`: Course presentation slides detailing the theoretical proofs, bottlenecks, and logic behind the consistency checks[cite: 4].

## Usage
Run the paper-faithful simulation:
```bash
python bandit_simulation.py
```
Run the finite-horizon practical benchmark:
```bash
python practical_hybrid.py
```
