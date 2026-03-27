# t-SNE vs K-Means: What Transferred vs What Cracked

**Domain:** SaaS (Customer usage & behavioral data)  
**Analysis:** What thinking from the K-Means document still applies to t-SNE, and what fundamentally breaks or needs to be replaced.

## Transfer Analysis Table

| #  | What transferred from K-Means<br>(still works the same way or very similarly) | What cracked or had to be replaced<br>(breaks or needs fundamental change) |
|----|-------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| 1  | Unsupervised framing shift: no labels needed; discover structure instead of predicting | Output type: discrete cluster assignments → continuous 2D/3D coordinates for visualization only |
| 2  | Hypothesis about data shape: you are still betting on an assumed structure in the data | Hypothesis strength: strong geometric assumption (spherical blobs) → weak, local-neighborhood-only assumption (non-linear manifold) |
| 3  | Need for human judgment on key hyperparameter (K vs perplexity) using business context | Hyperparameter interpretation: K = number of groups → perplexity = effective neighborhood size (scale of "local") |
| 4  | Evaluation without ground truth: rely on internal metrics + business validation | Evaluation focus: cluster tightness/separation (WCSS/silhouette) → local similarity preservation (KL divergence) + visual inspection |
| 5  | Mathematical quality ≠ business utility (Framework #3): pretty output doesn't guarantee value | Global interpretability: distances/centers meaningful → global distances often meaningless or distorted |
| 6  | Agentic pipeline of decisions: feature prep → run algo → interpret → policy → act → feedback | Reusability for new data: can apply learned centers to new points → no stable mapping; must re-run whole algorithm |
| 7  | Feedback loop requirement: track whether discovered structure predicts real SaaS outcomes (retention, expansion) | Determinism/stability: same input → same result (up to random init) → stochastic; multiple runs can look different |
| 8  | Need to align algo objective with business objective (loss ≠ agent goal) | Optimization goal: minimize within-cluster variance → minimize KL divergence between similarity distributions |
| 9  | Business validation of output before acting (are segments actionable in SaaS?) | Purpose of output: partitioning for assignment/rules → exploration & hypothesis generation via visual map |
| 10 | Expected value thinking on downstream decisions (Framework #6): weigh uncertainty of structure | Density preservation: roughly preserved in clusters → often distorted or equalized in the plot |
| 11 | Reality checks and failure mode awareness (over-interpreting math as truth) | Out-of-sample projection: easy with cluster centers → not directly possible without re-embedding |
| 12 | Injection of domain knowledge at interpretation stage (SaaS usage patterns) | Cluster definition: clear boundaries and centers → no centers, boundaries can be illusory due to local focus |
| 13 | Scaling/normalization of features before running the algorithm | Scalability with dataset size: works well on moderate data → computationally heavy (pairwise similarities) on large SaaS logs |
| 14 | Use as Layer 2 in the 5-layer stack: representation feeds decisions | Role in pipeline: can be final for segmentation → primarily intermediate for visualization; often paired with clustering afterward |
| 15 | Non-convex optimization requires care (local minima possible) | Trust in visual clusters: can run K-Means on the plot reliably → risky; t-SNE clusters are often exaggerated or artifactual |

## Summary

### What transferred strongly (core unsupervised thinking stays intact)
- The philosophical framing shift (unsupervised = discover structure when labels don't exist)
- Heavy reliance on human judgment for hyperparameters and interpretation using SaaS business context
- Mathematical quality ≠ business utility (Framework #3)
- Need for strong feedback loops and expected value thinking on downstream decisions
- Agentic pipeline structure and reality checks around over-interpreting output
- Layer 2 position in the 5-layer intelligent system stack

### What cracked or had to be replaced (major mindset shifts)
- t-SNE is **not** a clustering algorithm — it is a visualization and exploration tool
- Shift from global partitioning (centers + discrete assignments) to local neighborhood preservation
- Loss of reproducibility and easy out-of-sample projection
- Global distances in the plot are often unreliable or distorted
- Output is primarily for hypothesis generation, not direct production rules
- Higher sensitivity to hyperparameters (especially perplexity) and stochastic behavior

**Key takeaway for SaaS teams:**  
Use t-SNE **early** in the exploration phase to discover non-linear behavioral manifolds in high-dimensional usage data. Then apply K-Means-style thinking (or actual K-Means) when you need stable, deployable segments for product decisions, targeting, or automation.

---
