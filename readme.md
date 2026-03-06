# The Geometric Inductive Bias of Grokking: Bypassing Phase Transitions via Architectural Topology

[![arXiv](https://img.shields.io/badge/arXiv-Pending-b31b1b.svg)]([https://arxiv.org/](https://arxiv.org/abs/2603.05228))
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

**Naming clarification for Spherical Norm vs Fully Bounded configurations:**

In the current manuscript, the "Fully Bounded Spherical Topology" is described as introducing unembedding normalization as an additional architectural step beyond the "Spherical Norm" variant. In practice, **both configurations share the identical bounded architecture**: L2-normalized residual stream + on-the-fly normalized unembedding matrix + fixed temperature scaling (τ = 10.0). Without unembedding normalization, all spherical models exhibited Softmax Collapse and training instability, making it a necessary component of any spherical configuration.

The **only difference** between the two reported configurations is the weight decay setting:
- **Spherical Norm (λ = 1.0):** Bounded spherical architecture **with** weight decay
- **Fully Bounded (λ = 0.0):** Bounded spherical architecture **without** weight decay

This will be corrected in v2 of the paper. The experimental results, training logs, and all conclusions remain unchanged.
