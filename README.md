# VHHP: Structural-Prior-Augmented Hessian Pruning for Large Language Models

> **Repository naming note.** This repository is referred to as **VHHP**.  
> In the current manuscript, the framework is named **Contrast-Hessian Hybrid Pruning (CHHP)**. The implementation and equations described below follow the current manuscript.

VHHP is a post-training pruning framework for large language models built on top of **SparseLLM** and **SparseGPT**.

The central idea is simple: the standard SparseGPT / Optimal Brain Surgeon (OBS) importance score is a strong second-order measure of the local reconstruction cost of pruning a weight, but it evaluates each weight largely in isolation. VHHP augments that score with two structural priors computed from information already available during Hessian-based pruning:

1. **Contrast Manifold** — emphasizes weights that are dominant within their local row/neuron and suppresses background weights.
2. **Feature Uniqueness** — penalizes weights connected to highly correlated, redundant input channels.

These two priors are fused multiplicatively with the OBS score into a **Champion Score** used for mask selection. The standard OBS compensation/recovery step is then applied unchanged.

No backpropagation or retraining is required.

---

## Overview

For a layer with weight matrix \(W\) and calibration activations \(X\), SparseGPT constructs the damped empirical Hessian

\[
H = XX^\top + \lambda I.
\]

The standard OBS / SparseGPT importance score is

\[
S^{\text{base}}_{ki}
=
\frac{w_{ki}^{2}}
     {[H^{-1}]_{ii}}.
\]

This score captures second-order sensitivity, but it does not explicitly distinguish:

- a weight that is dominant relative to the other weights in its row from one that is merely background magnitude, or
- a weight connected to a unique input feature from one connected to a highly redundant feature.

VHHP adds both signals before pruning.

---

## Method

### 1. OBS Base Score

The starting point is the SparseGPT / OBS score:

\[
S^{\text{base}}_{ki}
=
\frac{w_{ki}^{2}}
     {[H^{-1}]_{ii}}.
\]

This estimates the local reconstruction cost of removing \(w_{ki}\) under the quadratic approximation used by OBS.

---

### 2. Contrast Manifold

Weights are normalized within each output row:

\[
\bar{w}_{ki}
=
\frac{|w_{ki}|}
     {\max_j |w_{kj}| + \varepsilon}.
\]

The normalized magnitude is then sharpened with an odd-power contrast function:

\[
V_{ki}
=
\bar{w}_{ki}^{\,2n+1}.
\]

The purpose of this transformation is to preserve row-dominant weights while strongly suppressing weights whose magnitude is small relative to the strongest weight in the same row.

For example, with \(n=3\), the exponent is \(7\). A normalized weight of \(0.5\) becomes

\[
0.5^7 \approx 0.0078,
\]

while the row maximum remains \(1\).

---

### 3. Feature Uniqueness

The calibration Hessian also contains cross-channel information.

First, normalize it into a correlation matrix:

\[
C_{ij}
=
\frac{H_{ij}}
     {\sqrt{H_{ii}H_{jj}}}.
\]

The accumulated redundancy of input channel \(i\) is

\[
R_i
=
\sum_j |C_{ij}|.
\]

The corresponding Feature Uniqueness score is

\[
U_i
=
\frac{1}{\log(1 + R_i)}.
\]

A channel that is strongly correlated with many other channels receives a smaller uniqueness value, while a more independent channel receives a larger survival prior.

This is a correlation-based redundancy heuristic; it is not an information-theoretic entropy or mutual-information estimate.

---

### 4. Champion Score

The final pruning score combines the three signals multiplicatively:

\[
S^{\text{final}}_{ki}
=
S^{\text{base}}_{ki}
\left(
V_{ki} U_i
\right)^\alpha.
\]

where:

- \(S^{\text{base}}\) is the exact OBS-derived second-order score,
- \(V\) is the Contrast Manifold factor,
- \(U\) is the Feature Uniqueness factor,
- \(\alpha\) controls the strength of the structural priors.

Typical settings described in the manuscript are:

- \(\alpha = 1.0\): full structural fusion,
- \(\alpha = 0.35\): conservative / tie-breaking regime,
- \(\alpha = 0\): recovers the plain OBS/SparseGPT score.

The multiplicative form acts like an **AND gate**: a weight should be important according to curvature, local magnitude contrast, and feature uniqueness in order to receive a high final score.

---

## Two-Stage Pruning

VHHP deliberately separates **mask selection** from **weight recovery**.

### Stage 1 — Structural mask selection

The Champion Score is computed using the original frozen pretrained weights.

The top-\((1-p)\) fraction of scores is retained, producing a binary mask \(M\), where \(p\) is the target sparsity.

### Stage 2 — OBS recovery

Once the mask is fixed, the original SparseGPT / OBS compensation step is used without changing its mathematics:

\[
\delta w
=
-
\frac{w_{ki}}
     {[H^{-1}]_{ii}}
H^{-1}_{:,i}.
\]

This lets VHHP alter **which weights are selected for pruning** while preserving the standard second-order recovery mechanism.

---

## Integration with SparseLLM

This repository is forked from **SparseLLM**.

SparseLLM provides a global multi-layer ADMM orchestration framework. In this project, VHHP is used as the **local pruning solver** inside that outer optimization loop.

Conceptually:

```text
Calibration activations
        |
        v
Hessian construction
H = X X^T + lambda I
        |
        v
Hessian inversion
        |
        +-----------------------------+
        |                             |
        v                             v
Attention sublayers             MLP sublayers
SparseGPT / OBS score           VHHP Champion Score
        |                             |
        +-------------+---------------+
                      |
                      v
                 Global mask
                      |
                      v
              OBS prune + compensate
                      |
                      v
              SparseLLM ADMM loop
```

The current best-performing configuration applies the structural VHHP/CHHP score to **MLP sublayers** (`fc1`, `fc2`) and retains vanilla SparseGPT / OBS scoring for **multi-head attention** projections.

---

## Why MLP-Only?

The current single-matrix formulation works best on MLP sublayers.

For an MLP block, row-wise weight contrast and input-channel redundancy can be treated locally with reasonable effectiveness.

Multi-head attention is different. Query and key projections interact jointly inside the softmax:

\[
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right),
\]

so independently scoring \(W_Q\) and \(W_K\) ignores an important cross-matrix dependency.

For this reason, the current recommended configuration is:

```text
MHA: SparseGPT / OBS
MLP: VHHP / CHHP Champion Score
```

The manuscript proposes coupled QK/VO masking, head-level contrast, per-head entropy/uniqueness, and an adaptive contrast schedule as future extensions. These extensions are not part of the current evaluated implementation.

---

## Complexity

For each layer, the additional VHHP operations are:

- \(O(d_{\text{in}}^2)\) for the correlation matrix and redundancy sums,
- \(O(d_{\text{out}}d_{\text{in}})\) for the Contrast scores.

These terms are dominated by the Hessian inversion / Cholesky work already required by SparseGPT.

Therefore, VHHP has the same **asymptotic complexity class** as the underlying SparseGPT solver.

---

## Numerical Stability

High contrast exponents can produce very small values.

For stable FP16 execution, the score can be evaluated in log space:

\[
\log S^{\text{final}}_{ki}
=
\log S^{\text{base}}_{ki}
+
\alpha
\left[
(2n+1)\log \bar{w}_{ki}
+
\log U_i
\right].
\]

This preserves ranking while avoiding numerical underflow.

---

## Code Parameterization Note

The theoretical notation in the manuscript uses

\[
V_{ki} = \bar{w}_{ki}^{2n+1}.
\]

Some implementation paths expose the contrast exponent directly through a code parameter named `ncontrast`.

These parameterizations are related, but **`ncontrast` should not automatically be interpreted as the manuscript variable `n`**. Check the implementation path being used before reproducing an experiment.

---

## Dependencies

The project inherits the SparseLLM software stack. The base repository reports the following tested versions:

- Python 3.10.14
- PyTorch 2.4.1 with CUDA 12.4
- Transformers 4.45.1
- Datasets 3.0.1
- NumPy 2.1.1
- pandas 2.2.3
- huggingface_hub 0.25.1
- wandb 0.18.2

Install the repository dependencies according to your local CUDA/PyTorch environment.

---

## Usage

The repository retains the SparseLLM-style OPT entry point.

A basic OPT pruning run follows the original SparseLLM interface:

```bash
python opt_main.py \
    --model facebook/opt-125m \
    --dataset c4 \
    --sparsity 0.8
```

The modified local solver implements the VHHP/CHHP scoring path inside the pruning engine.

### Main inherited arguments

- `--model`: Hugging Face model identifier.
- `--dataset`: calibration/evaluation dataset, such as `c4`, `wikitext2`, or `ptb`.
- `--sparsity`: target fraction of weights to prune.

### VHHP-specific configuration

The current manuscript describes the following method-level parameters:

- `n` / contrast exponent: controls Contrast Manifold sharpness.
- `ncontrast`: exponent parameter exposed by some code paths.
- `alpha`: blend strength for the structural priors.
- damping: `percdamp = 0.01 * mean(diag(H))`.
- numerical stabilizer: approximately \(10^{-8}\) to \(10^{-9}\).
- OBS block size: 128 columns.

> **Important:** the paper documents the code-level quantity `ncontrast`, but it does not fully specify the command-line spelling of every modified-repository option. The exact CLI examples should be synchronized with the repository's `argparse` definitions before release.

---

## Experimental Setup in the Manuscript

The current evaluation uses OPT models:

| Model | Calibration dataset | Calibration samples | Evaluated sparsity |
|---|---|---:|---|
| OPT-125M | C4 | 128 | 50%–95% |
| OPT-1.3B | C4 | 30 | 50%, 60%, 70%, 80%, 90% |
| OPT-2.7B | C4 | 30 | 50%, 60%, 70%, 80%, 90% |
| OPT-6.7B | C4 | 10 | 40%–80% |

Perplexity is evaluated on **WikiText-2** and **C4** with sequence length 2048.

The base SparseLLM repository also contains LLaMA entry points, but the current VHHP/CHHP results reported in the manuscript are for the OPT family.

---

## Representative Results

Under the matched OPT-125M experiment reported in the manuscript:

- target sparsity: **80%**
- calibration: **C4**
- calibration samples: **128**
- seed: **0**
- attention: vanilla SparseGPT
- MLP: VHHP/CHHP Champion Score

| Method | WikiText-2 PPL ↓ | C4 PPL ↓ |
|---|---:|---:|
| Magnitude pruning | 4859.41 | 2444.93 |
| SparseGPT / OBS | 1686.32 | 857.79 |
| Wanda | 1183.86 | 600.80 |
| SparseGPT-attn + VHHP/CHHP-MLP | **819.92** | **469.52** |

In that matched run, the hybrid MLP configuration improves perplexity by approximately:

- **51.3%** relative to SparseGPT/OBS on WikiText-2,
- **45.3%** relative to SparseGPT/OBS on C4,
- **30.7%** relative to Wanda on WikiText-2,
- **21.9%** relative to Wanda on C4.

The manuscript also reports strong improvements for OPT-125M across the moderate/high sparsity regime and an MLP-only improvement in a pilot OPT-6.7B experiment.

---

## Important Experimental Caveats

The current results should be interpreted with the scope of the manuscript in mind:

1. **MLP vs. attention**  
   The current structural score improves MLP pruning but does not improve MHA pruning. The recommended implementation therefore falls back to SparseGPT for attention.

2. **Scale dependence**  
   Gains do not transfer uniformly to billion-parameter models at high sparsity. Performance deteriorates beyond roughly the 70% sparsity region in the reported billion-scale experiments.

3. **Extreme sparsity**  
   Fixed contrast settings become fragile near 95% sparsity.

4. **Exponent selection**  
   Several reported “best-\(n\)” results were selected using the same evaluation set used for reporting perplexity. The manuscript proposes a held-out calibration reconstruction criterion for deployment-time selection, but that gap has not yet been fully evaluated.

5. **Single-run measurements**  
   Reported perplexities are currently based on single runs rather than variance estimates across multiple calibration draws/seeds.

6. **Baseline variation**  
   The manuscript notes that nominally similar OPT-125M / 80% / C4 baseline measurements differ across some reported experiments. Results should therefore be interpreted according to their exact experimental protocol.

---

## Held-Out Selection of the Contrast Exponent

For deployment, the manuscript proposes choosing the contrast order using only calibration data.

Split calibration activations into fitting and validation subsets:

\[
X = X_{\text{fit}} \cup X_{\text{val}}.
\]

For each candidate contrast order \(n\):

1. build the Hessian using \(X_{\text{fit}}\),
2. prune the layer,
3. apply OBS correction,
4. compute held-out reconstruction error

\[
E(n)
=
\left\|
WX_{\text{val}}
-
\hat{W}^{(n)}X_{\text{val}}
\right\|_F^2.
\]

Then choose

\[
n^\star
=
\arg\min_n E(n).
\]

This avoids selecting the exponent directly from downstream test perplexity.

---

## Repository Lineage

This implementation is built on top of:

- **SparseLLM** — global ADMM-based multi-layer pruning orchestration.
- **SparseGPT** — second-order OBS-style local pruning and weight compensation.
- **Wanda** — included in the lineage of the original SparseLLM repository and used as a comparison baseline.

VHHP modifies the local mask-selection criterion while retaining the surrounding SparseLLM/SparseGPT machinery.

---

## Citation

If you use this implementation, please cite the VHHP/CHHP work and the upstream SparseLLM project.

### VHHP / CHHP

```bibtex
@misc{sehili2026chhp,
  title  = {Contrast-Hessian Hybrid Pruning: A Structural-Prior-Augmented Second-Order Framework for Post-Training Compression of Large Language Models},
  author = {Sehili, Chams-Eddine and Albourm, Amar and Boutellaa, Elhocine and Namane, Rachid and Flitti, Farid and Belhaouari, Samir Brahim},
  year   = {2026},
  note   = {Preprint submitted to AI Open}
}
```

### SparseLLM

```bibtex
@inproceedings{bai2024sparsellm,
  title     = {SparseLLM: Towards Global Pruning of Pre-trained Language Models},
  author    = {Bai, Guangji and Li, Yijiang and Ling, Chen and Kim, Kibaek and Zhao, Liang},
  booktitle = {The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year      = {2024}
}
```

---

## Acknowledgements

This repository is forked from **SparseLLM** and uses the SparseGPT/OBS pruning machinery as its second-order recovery backbone.

We thank the authors of SparseLLM, SparseGPT, Wanda, and the broader open-source model-compression community for making their implementations available.

---

## Current Scope

VHHP/CHHP should currently be understood as a **structural-prior extension to second-order pruning for MLP sublayers**, not as a universal replacement for SparseGPT across every Transformer component.

The main research result is that structural information already present in the Hessian can improve mask selection without retraining and without changing the asymptotic complexity of the SparseGPT recovery pipeline.
