# Geometric Grokking — experiment and analysis code

Code bundle prepared 2026-10-30 for the camera-ready revision. It contains 16 recovered rebuttal notebooks, the six original repository notebooks, a legacy reporting notebook, and the current figure/ablation analysis scripts with their verified input histories.

## Start here

The notebooks are research snapshots, not a single sequential pipeline. Run the relevant notebook for an experiment, not every notebook in this directory. Overlapping versions can reuse the same Drive output directory and skip existing runs. Inspect each configuration list and select a fresh output directory when rerunning a changed configuration. Folder and notebook names alone are not evidence of the weight-decay or head setting; use the run metadata.

| Experiment | Notebook in `notebooks/rebuttal/` |
| --- | --- |
| Addition residual/head/WD ablation | `Addition_Rebuttal_Ablations_v1_Final.ipynb` and `Addition_Rebuttal_Ablations_v1__LN_bounded_unbounded.ipynb` |
| Addition robustness extensions | `Addition_Rebuttal_Ablations_with_Robustness_Sweep_last2_added.ipynb` (earlier sweep retained for provenance) |
| Addition learned-attention/spherical LR sweep | `Addition_Rebuttal_Matched_LR_Sweep.ipynb` |
| Matched uniform-attention LR sweep | `Uniform_Attention_Matched_LR_Sweep_log5.ipynb` |
| Five-epoch train/test onset measurements | `Q1_5epoch_Onset_Zoom.ipynb` |
| Norms, attention and Fourier diagnostics | `Q1_Temporal_Mechanism_and_Topology.ipynb` |
| S5 head controls | `S5_Sweep_Final_Colab_v7_head_controls_final.ipynb` and `S5_Sweep_Final_Colab_v7_head_controls_10seed.ipynb` |
| S5 swap/position and extended controls | `S5_Sweep_Final_Colab_v6_posthocfix.ipynb` |
| S5 spherical bounded uniform-attention control | `S5_Uniform_Attention_Decoupled_Head.ipynb` |
| A5 four-way comparison, WD=1 | `A5_FourWay_Final_PaperAligned_Colab.ipynb` |
| A5 four-way comparison, WD=0 | `A5_FourWay_Final_PaperAligned_No_WD.ipynb` |
| Historical aggregation helpers | `S5_Sweep_Results_table.ipynb`, `full_results_rebuttal.ipynb` |

`notebooks/original/` contains the earlier public experiment implementations. These are retained as historical code, not asserted to reproduce the revised paper without configuration changes. `notebooks/legacy_reporting/Final_Tables.ipynb` is the old reporting notebook involved in the path/label audit; do not use it as the authoritative revised Table 1 generator.

## Running notebooks

Open a notebook in Google Colab and run its setup cells. Most training notebooks mount your own Google Drive and use `/content/drive/MyDrive/...` paths. Review the output-root variables before launching. A GPU is recommended for training. Local execution requires adapting the Colab mount cells and paths. `requirements.txt` lists the imported packages; it is not a recovered, version-pinned training environment. Install a PyTorch build appropriate to your GPU.

The five-epoch onset notebook originally uses a runtime-relative `SAVE_ROOT`. Change it to a persistent directory for new runs. Its original full per-epoch histories were not recovered; the included onset CSV and saved text were extracted from the saved notebook outputs.

## Regenerate verified results without training

From the bundle root, with NumPy and Matplotlib installed:

```bash
python analysis/verify_ablation.py
python analysis/generate_figures.py
python analysis/plot_uniform_summary.py
```

The ablation script verifies 31 histories across eight conditions and writes per-run and summary files. The figure script verifies 33 histories and regenerates Figure 1 plus the addition LR comparison. The uniform summary plot uses explicitly recorded rebuttal aggregate means, not reconstructed trajectories. Its plots are written to `analysis/camera_ready_review/`; the other outputs are written next to their scripts.

## Results and interpretation

- `analysis/*logs.json` contains the input histories used by the ablation/figure scripts, with source provenance.
- `results/group_controls/` contains the recovered S5/A5 summaries.
- `results/temporal/` contains the temporal and swap-control CSVs used in the revision.
- `results/onset_5epoch/` contains the recoverable five-epoch onset summaries.
- Test-99 is the first logged checkpoint at or above 99% accuracy. The completion gap is Test-99 minus Train-99. It is distinct from the start of the test rise.
- Residual pre-normalization norms measure the residual after adding the update, not the update alone. Uniform attention retains positional information unless positions are explicitly removed.
- A5 WD=0 has 17 available summary rows in this recovered snapshot; do not relabel it as a verified 20-run result.
- Notebook variants and reporting helpers retain their original code/comments. The verified analysis scripts and input metadata take precedence over historical naming or narrative claims.

## Provenance and validation

Notebook outputs, execution counts and notebook/cell metadata were cleared for a compact source release. Experiment code cells were preserved. `notebook_sources.json` maps the rebuttal notebooks to their Drive source IDs. These source IDs do not grant access to private Drive files; the included code can be read without them. `MANIFEST_SHA256.csv` records file hashes.

The package was checked for valid notebook JSON, Python cell syntax (excluding notebook magic/shell lines), ZIP integrity, and regeneration of the local ablation/figure outputs. Training was not rerun. Full training checkpoints and all raw histories for every rebuttal experiment are not included. This bundle is not a claim that every result can already be reproduced end to end with one command.
