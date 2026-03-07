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

# Errata (v1 → v2)

> **Note:** This errata applies to arXiv v1 (2603.05228). A corrected v2 will be uploaded shortly.

We identified a labeling error and a missing architectural clarification in v1 of the manuscript. All core findings are unchanged; the corrections concern how configurations were named, described, and reported.

## 1. Architectural Clarification

In v1, "Spherical Norm" and "Fully Bounded" are described as two distinct architectures, with the implication that Spherical Norm uses an unconstrained unembedding layer. This is incorrect. 

Both configurations share the identical bounded architecture:
* L2-normalized residual stream
* On-the-fly normalized unembedding matrix
* Fixed temperature scaling (τ = 10.0)

Without unembedding normalization, all spherical models exhibited Softmax Collapse and training instability, making it a necessary component of any spherical configuration. The only difference between the two configurations is the weight decay setting (λ). 

In v2, both are renamed to **Bounded Sphere** and distinguished solely by λ.

## 2. Table 1 Correction

The v1 row labeled "Fully Bounded (λ = 0.0)" contained results from a different experimental run. All spherical results have been rerun from scratch with verified weight decay settings. v2 also reports a new **Fourier Init** variant, where the first 10 embedding dimensions are deterministically initialized with cosine/sine values at five key frequencies.

### Baselines (unchanged)

| LR | Architecture | Mean Grok Epoch | Std Dev | Min | Max | Failures | Peak Acc |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1×10⁻⁴ | LayerNorm | 54,160 | 13,490 | 32,800 | 71,600 | 0/10 | 100% |
| 1×10⁻⁴ | RMSNorm | 51,240 | 11,200 | 38,800 | 74,600 | 0/10 | 100% |
| 6×10⁻⁴ | LayerNorm | 7,800 | 1,095 | 6,000 | 9,400 | 0/10 | 100% |
| 6×10⁻⁴ | RMSNorm | 7,300 | 925 | 6,000 | 9,200 | 0/10 | 100% |

### Bounded Sphere (corrected)

| LR | Architecture | Mean Grok Epoch | Std Dev | Min | Max | Failures | Peak Acc |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1×10⁻⁴ | Bounded Sphere (λ=1.0) | 2,400 | 353 | 1,800 | 3,000 | 0/10 | 100% |
| 1×10⁻⁴ | Bounded Sphere (λ=0.0) | 2,480 | 464 | 1,800 | 3,200 | 0/10 | 100% |
| 1×10⁻⁴ | Bounded Sphere + Fourier Init (λ=1.0) | 2,120 | 368 | 1,600 | 2,600 | 0/10 | 100% |
| 1×10⁻⁴ | Bounded Sphere + Fourier Init (λ=0.0) | 2,100 | 316 | 1,800 | 2,600 | 0/10 | 100% |
| 6×10⁻⁴ | Bounded Sphere (λ=1.0) | 1,560 | 853 | 600 | 3,000 | 0/10 | 100% |
| 6×10⁻⁴ | Bounded Sphere (λ=0.0) | 820 | 199 | 600 | 1,200 | 0/10 | 100% |
| 6×10⁻⁴ | Bounded Sphere + Fourier Init (λ=1.0) | 800 | 163 | 600 | 1,000 | 0/10 | 100% |
| 6×10⁻⁴ | Bounded Sphere + Fourier Init (λ=0.0) | 711 | 203 | 400 | 1,000 | 0/10 | 100% |

### Key observations:
* All bounded sphere configurations generalize 10–50× faster than baselines. The core claim is unchanged.
* At LR=1×10⁻⁴, weight decay has minimal effect — all sphere variants converge in ~2,100–2,480 epochs.
* At LR=6×10⁻⁴, removing weight decay helps the standard-init variant substantially (820 vs 1,560, std 199 vs 853).
* Fourier initialization provides a consistent modest advantage, with the best overall result at 711 epochs (6×10⁻⁴, λ=0.0).

## 3. Text Corrections

Six passages in v1 incorrectly described the Spherical Norm variant as having an unconstrained unembedding (Abstract, Introduction, §3.2, §4.1, §4.3 ×2). All have been corrected in v2 to reflect the shared bounded architecture. 

The naming throughout has been updated: "Spherical Norm (λ=1.0)" and "Fully Bounded (λ=0.0)" are now both called **Bounded Sphere**, distinguished only by λ.

## 4. What Does Not Change

* Uniform Attention Ablation results (Table 2)
* S5 negative control results (Table 3)
* Spectral verification / Fourier circuit analysis
* All qualitative conclusions
* **Core finding:** bounded spherical topology massively accelerates generalization over standard baselines.
