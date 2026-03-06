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

## Errata (v1)
We identified a labeling error and a missing architectural clarification in v1 of the manuscript. All experimental results remain valid; the corrections concern how configurations were named and described. The core findings — that bounded spherical topology massively accelerates generalization over standard baselines — are unchanged.

---

### 1. Architectural Clarification

In v1, "Spherical Norm" and "Fully Bounded" are described as two distinct architectures, with the implication that Spherical Norm uses an **unconstrained** unembedding layer. This is incorrect.

**Both configurations share the identical architecture:**
- L2-normalized residual stream
- On-the-fly normalized unembedding matrix
- Fixed temperature scaling (τ = 10.0)

Without unembedding normalization, all spherical models exhibited Softmax Collapse and training instability, making it a necessary component of any spherical configuration. The **only difference** between the two reported configurations is the weight decay setting (λ).

This affects six passages in the manuscript (Abstract, Introduction §1, §3.2, §4.1, §4.3 twice) that incorrectly describe the Spherical Norm variant as having an unconstrained unembedding. These will be corrected in v2.

