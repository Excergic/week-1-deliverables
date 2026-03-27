# K-Means → PCA: What Transfers vs What Cracks

This document captures the transfer of thinking from K-Means (clustering) to PCA (dimensionality reduction), highlighting what remains consistent and what fundamentally changes.

---

## What Transferred vs What Cracked

| What Transferred from K-Means | What Cracked or Had to Be Replaced |
|------------------------------|------------------------------------|
| Unsupervised framing: no labels, no ground truth | “Correct answer” intuition → replaced with “usefulness depends on interpretation” |
| Every algorithm is a hypothesis about data shape | Hypothesis type changed: clusters → continuous directions |
| Internal metric ≠ business value | WCSS → replaced with variance explained (same trap, new form) |
| Need for human interpretation | Interpretation became harder (clusters are intuitive, components are abstract) |
| Hyperparameters are human decisions | K → replaced with number of components (but less obvious how to choose) |
| Evaluation without ground truth | Silhouette/elbow → replaced with explained variance (weaker signal of usefulness) |
| Model output is not the system | Cluster → action mapping → replaced with score → threshold → action |
| Need for feature preprocessing (scaling) | Became more critical — PCA is extremely sensitive to scale |
| Alignment problem (loss vs objective) | Became worse — variance often less aligned with business goals than clustering |
| Feedback loop necessity | Harder to detect failure — no clear “bad clustering” signal |
| Sensitivity to data assumptions | Different assumptions — K-Means: shape, PCA: variance structure |
| Overfitting equivalent exists | Changed form — not too many clusters, but too many meaningless components |
| Agent cannot make business decisions | Same — but now also cannot interpret components |
| Output requires validation before action | Validation harder — “does this axis mean anything?” vs “are these clusters real?” |
| Pipeline thinking (data → model → decision → action) | Representation step became more abstract and indirect |

---

### What Stayed the Same
- No labels → no ground truth  
- Metrics can mislead  
- Human judgment is required  
- Model ≠ system  
- Feedback loops are essential  

---

### What Fundamentally Changed
- Discrete grouping → continuous representation  
- Intuitive outputs → abstract components  
- Iterative optimization → direct mathematical solution  
- Visible failures → silent failures  

---

### The Core Insight

> K-Means simplifies by grouping.  
> PCA simplifies by compressing.

---

### Final Mental Model

- **K-Means:** “Which bucket does this belong to?”  
- **PCA:** “Where does this lie on a spectrum?”  

---


**What stayed the same:**  
> This is not about the model. It’s about what you do with it.

**What changed:**  
> The model now gives you a different kind of thing to think with.