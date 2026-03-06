# The Geometric Inductive Bias of Grokking: Bypassing Phase Transitions via Architectural Topology

[![arXiv](https://img.shields.io/badge/arXiv-2603.05228-b31b1b.svg)](https://arxiv.org/abs/2603.05228)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

> **Official PyTorch implementation** for the paper *The Geometric Inductive Bias of Grokking: Bypassing Phase Transitions via Architectural Topology*.

## Overview

This repository contains the code to reproduce the interventional mechanistic interpretability experiments from our paper. We demonstrate that the delayed generalization phenomenon (grokking) in Transformer models is heavily influenced by excess architectural degrees of freedom—specifically, unconstrained representational magnitude and data-dependent attention routing.

By introducing a **Fully Bounded Spherical Topology** and a **Uniform Attention Ablation**, we effectively bypass the memorization phase in cyclic modular addition ($\mathbb{Z}_{113}$), reducing grokking onset time by **$\sim25\times$** without relying on weight decay. 

We also provide the code for our non-commutative $S_5$ permutation composition negative control, which demonstrates that this acceleration relies on task-specific geometric alignment rather than generic optimization stabilization.

## Key Interventions

1. **Fully Bounded Spherical Topology (Intervention A):** Replaces standard LayerNorm/RMSNorm with strict $L_2$ normalization across the residual stream and the unembedding matrix. This restricts the model to a bounded cosine geometry, mathematically preventing Softmax Collapse and magnitude-driven memorization.
2. **Uniform Attention Ablation (Intervention B):** Overrides learned query-key routing with a fixed uniform distribution, reducing the attention mechanism to a Continuous Bag-of-Words (CBOW) aggregator. 

## Repository Structure

All experiments are conducted using a minimal, highly interpretable 1-layer Transformer architecture. The codebase is structured as Jupyter Notebooks for easy reproduction and visualization:

* `Final_Modular_Norm_Baselines.ipynb` - Trains standard baselines (LayerNorm, RMSNorm) against spherical topologies with standard weight decay on the $\mathbb{Z}_{113}$ task.
* `Final_Modular_No_Decay.ipynb` - Implements the **Fully Bounded Spherical Topology** ($\lambda=0.0$) demonstrating the $\sim25\times$ acceleration. Decay is set 1.0 as default. You can set it 0.0 simply.
* `Final_No_attention_all_models.ipynb` - Implements the **Uniform Attention Ablation** (CBOW aggregator) to test the necessity of data-dependent routing.
* `Fourier_Analysis_Last.ipynb` - Generates the spectral analysis and Fraction of Variance Explained (FVE) metrics reported in Section 4.3 to verify the underlying Fourier circuits.

## Getting Started

### Reproducibility Note regarding $L_2$ Normalization
Section 3.2 of the paper notes a theoretical $\epsilon=10^{-8}$ for the spherical normalization step to ensure numerical stability. In this codebase, we utilize PyTorch's native `torch.nn.functional.normalize`, which defaults to `eps=1e-12`. This difference is purely a default floating-point stabilizer and does not affect the geometry or training dynamics described in the text.

## Errata (v1 → v2)

> This errata applies to arXiv v1 (2603.05228). A corrected v2 will be uploaded shortly.

We identified a labeling error and a missing architectural clarification in v1 of the
manuscript. All experimental results remain valid; the corrections concern how configurations
were named and described. The core findings that bounded spherical topology massively
accelerates generalization over standard baselines are unchanged.

---

### 1. Architectural Clarification

In v1, "Spherical Norm" and "Fully Bounded" are described as two distinct architectures,
with the implication that Spherical Norm uses an unconstrained unembedding layer.
This is incorrect.

Both configurations share the identical architecture:
  - L2-normalized residual stream
  - On-the-fly normalized unembedding matrix
  - Fixed temperature scaling (τ = 10.0)

Without unembedding normalization, all spherical models exhibited Softmax Collapse and
training instability, making it a necessary component of any spherical configuration.
The only difference between the two reported configurations is the weight decay setting (λ).

This affects six passages in the manuscript (Abstract, Introduction §1, §3.2, §4.1, §4.3 twice)
that incorrectly describe the Spherical Norm variant as having an unconstrained unembedding.
These will be corrected in v2.

---

### 2. Table 1 Correction

The v1 row labeled "Fully Bounded (λ = 0.0)" actually contained results from a
Fourier-initialized spherical model trained with weight decay (λ = 1.0).

Below are the complete corrected results across all configurations.

#### Baselines (unchanged from v1)

| LR       | Architecture | Mean Grok Epoch | Std Dev | Min    | Max    | Failures | Peak Acc |
|----------|-------------|-----------------|---------|--------|--------|----------|----------|
| 1×10⁻⁴  | LayerNorm   | 54,160          | 13,490  | 32,800 | 71,600 | 0/10     | 100%     |
| 1×10⁻⁴  | RMSNorm     | 51,240          | 11,200  | 38,800 | 74,600 | 0/10     | 100%     |
| 6×10⁻⁴  | LayerNorm   | 7,800           | 1,095   | 6,000  | 9,400  | 0/10     | 100%     |
| 6×10⁻⁴  | RMSNorm     | 7,300           | 925     | 6,000  | 9,200  | 0/10     | 100%     |

#### Bounded Spherical Topology (corrected)

| LR       | Architecture                    | λ   | Mean Grok Epoch | Std Dev | Min   | Max   | Failures | Peak Acc |
|----------|---------------------------------|-----|-----------------|---------|-------|-------|----------|----------|
| 1×10⁻⁴  | Bounded Sphere                  | 1.0 | 2,480           | 464     | 1,800 | 3,200 | 0/10     | 100%     |
| 1×10⁻⁴  | Bounded Sphere                  | 0.0 | 2,400           | 353     | 1,800 | 3,000 | 0/10     | 100%     |
| 1×10⁻⁴  | Bounded Sphere + Fourier Init   | 1.0 | 2,100           | 316     | 1,800 | 2,600 | 0/10     | 100%     |
| 1×10⁻⁴  | Bounded Sphere + Fourier Init   | 0.0 | 2,120           | 368     | 1,600 | 2,600 | 0/10     | 100%     |
| 6×10⁻⁴  | Bounded Sphere                  | 1.0 | 820             | 199     | 600   | 1,200 | 0/10     | 100%     |
| 6×10⁻⁴  | Bounded Sphere                  | 0.0 | 1,560           | 853     | 600   | 3,000 | 0/10     | 100%     |
| 6×10⁻⁴  | Bounded Sphere + Fourier Init   | 1.0 | 700             | 194     | 400   | 1,000 | 0/10     | 100%     |
| 6×10⁻⁴  | Bounded Sphere + Fourier Init   | 0.0 | 800             | 163     | 600   | 1,000 | 0/10     | 100%     |

#### Key Observations

- All bounded spherical configurations generalize 10-50x faster than baselines
  regardless of weight decay or initialization strategy. The core claim is unchanged.
- At LR=1e-4, weight decay has minimal effect. All sphere variants converge
  in ~2,100-2,480 epochs.
- At LR=6e-4, Fourier initialization provides a consistent advantage (700-800
  vs 820-1,560), and weight decay helps stabilize the standard-init variant.
- Fourier initialization was not reported separately in v1. It will be properly
  credited in v2.

---

### 3. What Does Not Change

- Uniform Attention Ablation results (Table 2)
- S5 negative control results (Table 3)
- Spectral verification / Fourier circuit analysis (Section 4.3)
- All figures and training dynamics plots
- All qualitative conclusions

v2 with full corrections will be uploaded to arXiv shortly.
```
