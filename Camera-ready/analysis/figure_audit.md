# Figure sources and captions

## Figure 1 replacement

Generated directly from nine rebuttal JSON logs with explicit per-run metadata (three seeds per condition), rather than relabeling the old PNG. Source URLs are preserved in figure_source_logs.json and verified_onsets.csv. No source Drive files or manuscript files were modified.

Caption: Training and test accuracy on modular addition at learning rate 6e-4. Columns show LayerNorm with a raw output head and WD=1, spherical residuals with a bounded output head and WD=1, and the same spherical configuration with WD=0. Colors denote seeds 1–3; dashed and solid lines denote training and test accuracy, respectively. The top row uses a common 0–9,000 epoch axis; the bottom row shows the same runs over the first 2,000 epochs. Accuracy was logged every 50 epochs. Curves end at their final recorded checkpoint; no continuation, smoothing, or extrapolation is applied. Test-99 denotes the first logged checkpoint with test accuracy at least 99%; annotations show mean ± sample standard deviation across all three seeds. The means are 7,233 ± 1,097, 883 ± 416, and 1,750 ± 1,386 epochs, respectively. These threshold-completion times are distinct from test-rise onset.

Sources under grokking_rebuttal_addition:
- diag_layernorm_raw_lr_6e-4_long
- diag_sphere_bounded_lr_6e-4
- diag_sphere_bounded_lr_6e-4_wd0_long

The LayerNorm and WD=0 sphere runs have maximum budgets of 60,000 epochs, and the WD=1 sphere runs have a maximum budget of 5,000. All nine stopped with reason stable_test_100, before their budgets. This plot does not establish the outcome of an unshown WD=0 LayerNorm control or isolate the head effect. The complete factorial belongs in a separate table. It also does not replace the separate five-epoch-resolution onset analysis.

## Learning-rate control

Caption: Learning-rate sensitivity of the first logged Test-99 epoch for LayerNorm/raw and sphere/bounded, both with WD=1. Each condition has three paired seeds at each learning rate. Faint points show individual runs; large markers and error bars show the arithmetic mean ± sample standard deviation. The vertical axis is logarithmic, and learning-rate settings are displayed as discrete categories. Accuracy was logged every 50 epochs. All 24 runs reached Test-99. At LR=1e-3, means are 3,917 and 683 epochs (5.73-fold reduction).

Source: grokking_rebuttal_addition_lr_sweep_v1. These are a separate experiment series; its LR=6e-4 values (6,900 and 900) must not be substituted into the ablation series (7,233 and 883).

## Legacy audit

Examined all 60 JSON files across grokking_decay_6e-4_rerun, grokking_no_decay_6e-4_rerun, and grokking_no_decay_sphere_decay_added_6e-4. Each folder currently contains ten Sphere and ten Fourier+Sphere logs. All 20 parsed JSON histories in the decay_rerun folder exactly match their filename counterparts in the sphere_decay_added folder. This establishes duplicate log content, not historical optimizer settings. The old JSON histories do not contain WD metadata; neither folder names nor the current training notebook alone prove the setting used historically.

Recomputing first test accuracy >95% gives:
- decay_rerun and sphere_decay_added: Sphere 820; Fourier+Sphere 700.
- no_decay_rerun: Sphere 1,560; Fourier+Sphere 800.

The pasted Final_Tables saved output contains 711 for one Fourier row; the current complete ten-file folder gives 700. The origin of that saved-output discrepancy is unresolved and is not treated as a newly established reporting error. No conclusions about the legacy 1e-4 labels are made here.

The two previously disclosed reporting corrections remain the basis for replacing Figure 1: wrong WD source/label and >95% versus 100% threshold reporting, plus the separate S5 raw/bounded head mismatch. New plots use explicit rebuttal metadata and reproduce the rebuttal values.

## Reproduction

Install numpy and matplotlib in your Python environment, then run `python generate_figures.py` alongside figure_source_logs.json. It validates monotonic epochs, bounded accuracies, matching train/test array lengths, and recomputes Train-99 and Test-99 for all 33 runs against their saved summaries. PDF and SVG are vector outputs; PNG is 300 dpi. Source arrays and Drive provenance remain in figure_source_logs.json.
