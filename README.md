# VHHP: Structural-Prior-Augmented Hessian Pruning for Large Language Models

This repository contains the implementation of our Hessian-based post-training pruning framework for large language models. The codebase is built on top of **SparseLLM** and uses the **SparseGPT / Optimal Brain Surgeon (OBS)** recovery mechanism.

> **Naming note:** the current manuscript refers to the method as **Contrast-Hessian Hybrid Pruning (CHHP)**. This repository is named **VHHP**.

VHHP improves the SparseGPT pruning score by adding two lightweight structural priors computed from the same calibration Hessian:

- **Contrast Manifold**: emphasizes row-dominant weights and suppresses weak background weights.
- **Feature Uniqueness**: penalizes weights connected to highly correlated, redundant input channels.

The two priors are fused with the OBS score into a **Champion Score** used for mask selection. The standard OBS compensation step is then applied unchanged.

No retraining or backpropagation is required.

---

## Method

For a weight matrix \(W\) and calibration activations \(X\), the damped empirical Hessian is

$$
H = XX^\top + \lambda I.
$$

SparseGPT uses the OBS importance score

$$
S^{\mathrm{base}}_{ki}
=
\frac{w_{ki}^{2}}
{[H^{-1}]_{ii}}.
$$

VHHP augments this score with two structural terms.

### Contrast Manifold

Weights are normalized within each row:

$$
\bar{w}_{ki}
=
\frac{|w_{ki}|}
{\max_j |w_{kj}| + \varepsilon}.
$$

The normalized values are sharpened using

$$
V_{ki}
=
\bar{w}_{ki}^{\,2n+1}.
$$

This preserves dominant weights while strongly suppressing weights that are small relative to the maximum magnitude in the same row.

### Feature Uniqueness

The Hessian is normalized into a correlation matrix:

$$
C_{ij}
=
\frac{H_{ij}}
{\sqrt{H_{ii}H_{jj}}}.
$$

The redundancy of channel \(i\) is

$$
R_i = \sum_j |C_{ij}|,
$$

and the Feature Uniqueness score is

$$
U_i =
\frac{1}{\log(1 + R_i)}.
$$

Highly correlated input channels receive a smaller uniqueness score.

### Champion Score

The final pruning score is

$$
S^{\mathrm{final}}_{ki}
=
S^{\mathrm{base}}_{ki}
\left(V_{ki} U_i\right)^\alpha.
$$

where:

- \(S^{\mathrm{base}}\): SparseGPT / OBS second-order importance.
- \(V_{ki}\): Contrast Manifold score.
- \(U_i\): Feature Uniqueness score.
- \(\alpha\): structural-prior blending factor.

The mask is selected using the Champion Score, after which the original OBS recovery update is applied without modification.

---

## MLP-Aware Hybrid Pruning

The current best-performing configuration applies VHHP to the **MLP sublayers** while keeping standard SparseGPT scoring for **multi-head attention**.

```text
Transformer Layer
│
├── Attention
│   ├── q_proj  ── SparseGPT / OBS
│   ├── k_proj  ── SparseGPT / OBS
│   ├── v_proj  ── SparseGPT / OBS
│   └── out_proj ─ SparseGPT / OBS
│
└── MLP
    ├── fc1 ── VHHP Champion Score
    └── fc2 ── VHHP Champion Score
```

This distinction is important because query and key projections are coupled inside the attention softmax, while the current VHHP score operates on one weight matrix at a time.

---

## SparseLLM Integration

VHHP is implemented as a modified local pruning solver inside the **SparseLLM ADMM framework**.

The overall pipeline is:

```text
Calibration data
      │
      ▼
Construct Hessian
      │
      ▼
Compute pruning scores
      │
      ├── Attention → SparseGPT / OBS
      │
      └── MLP       → VHHP Champion Score
      │
      ▼
Select pruning mask
      │
      ▼
OBS compensation
      │
      ▼
SparseLLM ADMM coordination
```

The structural terms add only

- \(O(d_{\mathrm{in}}^2)\) work for Hessian correlations, and
- \(O(d_{\mathrm{out}}d_{\mathrm{in}})\) work for Contrast scores.

These operations do not change the asymptotic complexity of the SparseGPT pruning step.

---

## Dependencies

The repository inherits the SparseLLM software stack.

Tested upstream versions include:

- Python 3.10.14
- PyTorch 2.4.1
- CUDA 12.4
- Transformers 4.45.1
- Datasets 3.0.1
- NumPy 2.1.1
- pandas 2.2.3
- huggingface_hub 0.25.1
- wandb 0.18.2

---

## Usage

The repository keeps the SparseLLM-style OPT entry point.

### Example: OPT-125M

```bash
python opt_main.py \
    --model facebook/opt-125m \
    --dataset c4 \
    --sparsity 0.8
```

Main inherited arguments:

- `--model`: Hugging Face model identifier.
- `--dataset`: calibration/evaluation dataset.
- `--sparsity`: target sparsity level.

The modified pruning implementation contains the VHHP/CHHP scoring path used for MLP pruning.

### Contrast Parameter

The implementation exposes the contrast strength through the code-level parameter `ncontrast`.

The manuscript expresses the Contrast Manifold as

$$
V_{ki} = \bar{w}_{ki}^{\,2n+1}.
$$

The paper and code use slightly different indexing conventions for this exponent, so reproduction should follow the parameterization implemented in the repository.

---

## Experimental Setup

The manuscript evaluates the method on the OPT family.

| Model | Calibration | Samples | Sparsity |
|---|---|---:|---|
| OPT-125M | C4 | 128 | 50%–95% |
| OPT-1.3B | C4 | 30 | 50%–90% |
| OPT-2.7B | C4 | 30 | 50%–90% |
| OPT-6.7B | C4 | 10 | 40%–80% |

Perplexity is evaluated on **WikiText-2** and **C4** with sequence length 2048.

---

## Representative Results

Matched comparison on **OPT-125M**, **80% sparsity**, **C4 calibration**, **128 calibration samples**, and **seed 0**:

| Method | WikiText-2 PPL ↓ | C4 PPL ↓ |
|---|---:|---:|
| Magnitude | 4859.41 | 2444.93 |
| SparseGPT / OBS | 1686.32 | 857.79 |
| Wanda | 1183.86 | 600.80 |
| SparseGPT-attn + VHHP/CHHP-MLP | **819.92** | **469.52** |

Under this protocol, the hybrid VHHP configuration improves over SparseGPT / OBS by approximately:

- **51.3%** on WikiText-2.
- **45.3%** on C4.

It also improves over Wanda by approximately:

- **30.7%** on WikiText-2.
- **21.9%** on C4.

---

## Current Scope

The current implementation is most effective when the structural score is applied to **MLP sublayers**.

The manuscript reports that:

- MLP pruning benefits from the added structural priors.
- Applying the same single-matrix score directly to attention does not provide the same benefit.
- Performance becomes scale- and sparsity-dependent for billion-parameter models beyond roughly 70% sparsity.
- Extreme sparsity can require more careful selection of the contrast exponent.

Future extensions proposed in the manuscript include coupled QK/VO masking, head-level contrast, per-head uniqueness, and adaptive contrast scheduling.

---

## Citation

If you use this code in your research, please cite our work:

```bibtex
@misc{sehili2026chhp,
  title  = {Contrast-Hessian Hybrid Pruning: A Structural-Prior-Augmented Second-Order Framework for Post-Training Compression of Large Language Models},
  author = {Sehili, Chams-Eddine and Albourm, Amar and Boutellaa, Elhocine and Namane, Rachid and Flitti, Farid and Belhaouari, Samir Brahim},
  year   = {2026},
  note   = {Preprint submitted to AI Open}
}
```

Please also cite SparseLLM:

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

This repository is forked from **SparseLLM** and builds on the **SparseGPT / OBS** pruning framework.

We thank the authors of SparseLLM, SparseGPT, Wanda, and the broader open-source model-compression community for making their work publicly available.
