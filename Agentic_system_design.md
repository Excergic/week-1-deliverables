# Agentic System Design for High-Frequency Trading

## Business Problem

The core challenge in this high-frequency trading system is avoiding adverse selection. The system must intelligently decide **when not to trade**, when to enter positions aggressively, and which assets currently offer the best opportunities.

The central decision is:  
Given a ranked list of trading opportunities, which ones should we act on aggressively, which ones should we hedge, and which ones should we ignore entirely?

## Layer 1 — Data

Our competitive edge comes from high-resolution microstructure data rather than generic price information.

Key data sources include:
- **Order book data** (Level 1 and Level 2): best bid/ask prices, bid-ask spread, depth at each level, and order imbalance (buy vs sell pressure).
- **Trade prints (tick data)**: last traded price, trade size, and buyer/seller-initiated direction.
- **Derived microstructure signals**: short-term returns (1s, 5s, 30s), volatility bursts, order flow imbalance (OFI), and queue position (when market making).
- **Cross-asset signals**: price movements in correlated assets, lead-lag relationships, and divergences such as ETF vs. underlying constituents.
- **Latency and execution data**: time delay between signal generation and execution, fill probability, and estimated slippage.

From an experienced trader's perspective, the most critical features are:
- Order Flow Imbalance (OFI) — detecting aggressive buying or selling pressure in real time.
- Spread and overall liquidity — determining how costly it is to enter or exit a position.
- Short-term momentum or reversion patterns driven by microstructure noise.
- Cross-asset divergences — temporary mispricings relative to highly correlated instruments.
- Current volatility regime — whether the market is stable or experiencing chaos.

**Common data quality challenges in live markets:**
- Non-stationarity: patterns that worked yesterday can suddenly stop working due to regime shifts.
- Fat tails: rare events often dominate profit and loss.
- Latency mismatch: features are calculated at time T but execution occurs at T + Δ, making signals stale.
- Selection bias: we only observe outcomes from the trades we actually executed.
- Label noise: it is often unclear whether a losing trade resulted from a bad signal or poor timing.

**Critical data gap:**  
We currently lack complete true outcomes for every decision — actual PnL after a few seconds, confirmation of adverse selection, counterfactuals (what would have happened if we had not traded), and the real effectiveness of hedges. Building robust logging to capture this information is essential.

## Layer 2 — Prediction

The model should not focus primarily on predicting raw returns. Instead, it must address the real risk we face in high-frequency trading.

**Primary output:**  
The probability of adverse selection occurring in the next 5 seconds — that is, the likelihood that the price will move sharply against our position immediately after entry.

**Secondary output (optional):**  
Expected return conditional on no adverse selection occurring.

This framing is important because the cost of a single bad trade is significantly higher than the typical small profits in HFT. The system’s first priority is to avoid expensive mistakes (“landmines”), and only then to optimize for profit.

**Loss Function**  
Standard binary cross-entropy treats all prediction errors equally, which does not align with business reality. A false negative (entering a trade we should have avoided) can cost thousands of dollars, while a false positive (skipping a good trade) represents only an opportunity cost.

We therefore use a cost-sensitive loss function that penalizes false negatives much more heavily (for example, 5–10× the weight of false positives). In practice, the system should prefer missing ten good trades over taking one genuinely bad trade.

**Definition of success**  
Model performance is measured in business terms, not just ML metrics. The model is considered good enough when it reduces adverse selection losses by 30–50% while preserving at least 60–70% of the originally profitable trading opportunities.

The practical workflow is two-stage:
1. Filter out high-risk trades using the probability model.
2. Rank and select the best opportunities from the remaining low-risk set.

## Layer 3 — Decision

Model outputs are converted into actionable decisions using expected value (EV) calculations.

For each opportunity, we calculate the EV of different actions (aggressive trade, hedge, or do nothing) based on the predicted probability of adverse selection (`P_bad`).

**Decision Policy:**

- **P_bad < 0.60** → Tier 1: Aggressive trade (very low risk, high expected upside)
- **0.60 ≤ P_bad < 0.75** → Tier 2: Trade with hedge (moderate risk, reduced exposure)
- **0.75 ≤ P_bad < 0.90** → Tier 3: Hedge only (high risk, no directional exposure)
- **P_bad ≥ 0.90** → Tier 4: Do nothing (negative or zero EV)

These thresholds are derived from realistic profit/loss assumptions and execution costs, not from an arbitrary 0.5 cutoff. Using a naive 0.5 threshold would lead to over-trading and substantial accumulated losses.

The decision policy is deterministic and fully EV-driven. Any LLM usage in this layer is limited to generating explanations, logs, or summaries for human review. The LLM does not make trading decisions or override policy thresholds.

## Layer 4 — Action

At every time step (every few milliseconds to seconds), the system performs the following:

1. Receives `P_bad` (and optional expected return) for each asset.
2. Computes expected value for each possible action.
3. Ranks opportunities by EV.
4. Applies portfolio-level risk constraints (capital limits, correlation controls, maximum exposure).
5. Executes the selected trades via the automated low-latency engine.

**Tiered Execution Logic:**

- **Tier 1 — Aggressive Buy**  
  Trigger: `P_bad < 0.60`  
  The system places aggressive orders on the highest-EV assets within risk limits. Execution is fully automated with sub-100ms latency. Capital is allocated proportionally to EV rather than equally.

- **Tier 2 — Hedge + Buy**  
  Trigger: `0.60 ≤ P_bad < 0.75`  
  The system enters the primary trade and simultaneously hedges using a correlated asset. Both legs must execute nearly simultaneously for the hedge to be effective.

- **Tier 3 — Hedge Only**  
  Trigger: `0.75 ≤ P_bad < 0.90`  
  No directional position is taken. The system only exploits relative value by going long the undervalued asset and short the overvalued correlated asset.

- **Tier 4 — Do Nothing**  
  Trigger: `P_bad ≥ 0.90`  
  No action is taken. The decision is logged for learning.

A separate **portfolio risk engine** (deterministic, non-ML) sits between decision and execution. It enforces capital limits, prevents excessive concentration in correlated assets, and caps tail risk to ensure individual trades do not create dangerous portfolio-level exposures.

The LLM’s role is strictly supportive: generating trade summaries, risk reports, and post-trade explanations. It does not decide trades, allocate capital, or override risk controls.

## Layer 5 — Feedback (The Agentic Loop)

This layer transforms the system from a static model into a self-improving trading agent.

Every trade decision is recorded in a detailed logging schema that captures:
- Model outputs at decision time (`P_bad`, model version)
- Assigned tier and action taken
- Execution details (entry price, position size, slippage, costs)
- Short-term outcome (PnL after 5 seconds, adverse selection label)
- Portfolio context (total exposure, correlation clusters, volatility regime)

This data enables **drift-based retraining** instead of calendar-based updates. The system continuously monitors for degradation signals such as collapsing win rates, probability calibration drift, or declining average EV. Retraining is triggered when these signals appear or during major market events, using recent data (typically the last 7–14 days of executed trades with known outcomes).

Clear ownership is defined across all layers to prevent gaps:
- **Prediction model** → Quant Research / ML team
- **Decision policy** → Quant Trading Lead
- **Execution infrastructure** → Trading Systems Engineering
- **Logging and data quality** → Data Engineering
- **Retraining and updates** → Joint ownership (Quant + Trading)

This closed feedback loop is what makes the system truly agentic: every action generates high-quality outcome data, which improves the model, which leads to better decisions over time.

---

**Summary**  
This design creates an adaptive, risk-aware high-frequency trading system that prioritizes avoiding costly adverse selection while remaining opportunistic in favorable conditions. The combination of microstructure-aware prediction, EV-based decision policy, strict risk controls, and continuous feedback ensures robustness in live markets.