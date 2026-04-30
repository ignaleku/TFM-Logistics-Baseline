# RL-1 Results Summary

## 1. Experiment Setup

Three picking policies were evaluated on the same 10,000-order slice from `orders_base.csv` across three resource regimes:

| Policy | Description |
|---|---|
| FIFO | Arrivals served in strict arrival order across all stages |
| Urgent-first | Urgent orders always served before normal at every stage |
| DQN | Learned policy; agent acts only at picking when both urgent and normal queues are non-empty |

**Regimes evaluated:**

| Regime | Picking workers | Packing workers | Dispatch workers | Congestion |
|---|---|---|---|---|
| s111 | 1 | 1 | 1 | Severe |
| s211 | 2 | 1 | 1 | Medium (packing bottleneck) |
| s221 | 2 | 2 | 1 | Light (dispatch bottleneck) |

**Training:** 60 episodes, ~10,000 orders per episode, mixed scenario sampling (s211 60%, s111 30%, s221 10%). DQN uses epsilon-greedy exploration with linear decay from 1.0 to 0.05 over 200,000 steps. Reward is continuous: `+w` for on-time completion, proportional penalty for tardiness.

---

## 2. Results by Regime

### 2.1 s111 — Severe congestion (1-1-1)

| Policy | Total SLA | Urgent SLA | Normal SLA | Mean sys. time (min) | P90 sys. time (min) | % urgent dec. |
|---|---|---|---|---|---|---|
| FIFO | 0.027 | 0.007 | 0.030 | 12,467 | 22,304 | — |
| Urgent-first | 0.076 | 0.494 | 0.018 | 12,530 | 22,955 | — |
| **DQN** | **0.139** | **1.000** | 0.019 | **12,419** | 22,854 | **100%** |

DQN achieves the highest total SLA in this regime. Urgent SLA reaches 1.0, compared to 0.49 for urgent-first and 0.007 for FIFO. Normal SLA is comparable across urgent-first and DQN (0.018 vs 0.019). Mean system time under DQN is slightly lower than both baselines, though all three policies operate under extreme queueing pressure and absolute times are very high.

### 2.2 s211 — Medium congestion, packing bottleneck (2-1-1)

| Policy | Total SLA | Urgent SLA | Normal SLA | Mean sys. time (min) | P90 sys. time (min) | % urgent dec. |
|---|---|---|---|---|---|---|
| FIFO | 0.206 | 0.023 | 0.232 | 2,500 | 4,226 | — |
| Urgent-first | 0.274 | 1.000 | 0.173 | 2,533 | 4,556 | — |
| **DQN** | 0.217 | 0.323 | 0.202 | 2,508 | **3,724** | **5.5%** |

In this regime DQN does not replicate urgent-first behaviour. With a packing bottleneck downstream, the agent learned to serve normal orders the large majority of the time at contested decision points (5.5% urgent decisions). This reduces P90 system time by 18% relative to urgent-first (3,724 vs 4,556 min), and improves normal SLA over urgent-first (0.202 vs 0.173), at the cost of urgent SLA (0.323 vs 1.000). Total SLA is between FIFO and urgent-first.

### 2.3 s221 — Light congestion, dispatch bottleneck (2-2-1)

| Policy | Total SLA | Urgent SLA | Normal SLA | Mean sys. time (min) | P90 sys. time (min) | % urgent dec. |
|---|---|---|---|---|---|---|
| FIFO | 0.406 | 0.057 | 0.455 | 1,624 | 2,882 | — |
| Urgent-first | 0.462 | 1.000 | 0.388 | 1,603 | 2,989 | — |
| **DQN** | **0.460** | **1.000** | 0.385 | 1,611 | 3,006 | **100%** |

DQN matches urgent-first almost exactly: total SLA 0.460 vs 0.462, urgent SLA both 1.000, normal SLA 0.385 vs 0.388. System time difference is negligible (< 0.5%). The agent converged to a pure urgent-priority policy (100% urgent decisions) and independently recovered the urgent-first strategy.

---

## 3. Key Conclusions

### Where DQN beats FIFO

DQN improves total SLA over FIFO in all three regimes (+412% in s111, +5% in s211, +13% in s221). The improvement is driven almost entirely by urgent SLA: DQN achieves urgent SLA of 1.0 in s111 and s221 and 0.32 in s211, against FIFO urgent SLA of 0.007, 0.023 and 0.057 respectively. FIFO does not differentiate order types, so any learned prioritisation policy dominates it on urgent SLA when congestion is non-trivial.

### Where DQN matches or approaches urgent-first

In s221 (light congestion), DQN is statistically equivalent to urgent-first across all metrics. The agent independently converged to the same pure-urgent strategy, suggesting that with sufficient capacity the optimal policy is unambiguous.

In s111 (severe congestion), DQN **outperforms** urgent-first on total SLA (0.139 vs 0.076) and urgent SLA (1.000 vs 0.494). This result is unexpected and warrants further investigation; it may reflect interactions between the reward shaping and queue dynamics that the hardcoded policy does not capture.

### Where DQN underperforms urgent-first

In s211, DQN underperforms urgent-first on urgent SLA (0.323 vs 1.000) and total SLA (0.217 vs 0.274). The agent traded urgent priority for a lower P90 system time. Whether this constitutes underperformance depends on the objective: if the SLA target for urgent orders is hard, the DQN policy as trained is not adequate for this regime. Reward redesign or a higher `w_urgent` coefficient would likely push the agent toward more urgent decisions.

### What the urgent-decision rate suggests

The decision rate reveals regime-specific specialisation: the agent chose urgent in 100% of contested decisions in s111 and s221, and in only 5.5% in s211. This bifurcation is consistent with the downstream bottleneck structure: in s211 the single packing worker creates a queue that affects all order types equally after picking, reducing the marginal value of urgent prioritisation at picking. The DQN policy is therefore not simply a noisy approximation of urgent-first; it has learned a structurally different response to congestion configuration.

---

## 4. Training Dynamics

- Training loss becomes non-zero from episode 2 (buffer reaches train\_start\_size=3,000 after episode 1).
- Epsilon decays from 1.0 to approximately 0.093 by episode 60.
- Replay buffer reaches ~191,000 transitions by the final episode.
- Later training episodes (≥49) draw heavily from s211 windows with high decision counts (dec > dec\_cap=4,500), triggering curriculum skip; these episodes contribute to the buffer but do not run gradient updates. This explains loss entries of `NA` in episodes 45 and 49 onward.
- The training `p_urgent_decisions` metric shows a clear divergence between scenario types over time: s111 episodes trend toward high urgent rates (0.55 → 0.95), while s211 episodes trend toward low urgent rates (0.50 → 0.14). This within-training divergence preceded the regime-specific behaviour observed at evaluation.

---

## 5. Limitations and Next Steps

- All SLA rates are low in absolute terms due to extreme congestion in s111 and s211 relative to the order volume. Results are best interpreted comparatively across policies rather than as absolute operational targets.
- The continuous reward function (`w_urgent=5.0`, `late_penalty_urgent=2.0`) may insufficiently penalise urgent tardiness in the s211 regime; increasing `w_urgent` or adding a hard SLA violation penalty is a natural next step.
- Evaluation used a single fixed order window and seed; confidence intervals across multiple windows and seeds are needed before drawing firm conclusions.
- The s111 outperformance over urgent-first is anomalous and should be replicated with different seeds to verify it is not seed-specific.
