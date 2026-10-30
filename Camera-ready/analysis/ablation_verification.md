# Verification of addition factorial

All eight cells reproduced from 31 raw JSON histories. Test-99 and Train-99 were recomputed using first logged accuracy >=0.99; gap was recomputed as their difference. Mean and sample standard deviation match the rebuttal after rounding. Success counts are 0/3, 0/3, 0/3, 3/3 for WD=0 and 3/3, 10/10, 3/3, 3/3 for WD=1 (LayerNorm/raw, LayerNorm/bounded, sphere/raw, sphere/bounded order).

Checks include chronological epochs, equal history lengths, accuracies in [0,1], distinct seeds per cell, metadata LR=0.0006, residual type, bounded-head flag, weight decay, saved versus recomputed first threshold times, and peak test accuracy. Provenance URLs and per-run budgets are in ablation_verified_per_run.csv. The 22 newly fetched runs are in ablation_additional_logs.json; the nine previously verified runs are in figure_source_logs.json. Reproduce with verify_ablation.py using the Python standard library.

Maximum budgets differ: sphere/bounded/WD=1 has 5,000; LayerNorm/bounded/WD=1 has 30,000; others have 60,000. All failures exhaust their budget (last epoch 59,999); all successes reach threshold within their budget. Log spacing is 50 epochs, with 49-epoch final interval on failures. Do not describe the grid as using identical maximum budgets.

This audit verifies recorded results and the explicit metadata, not the historical optimizer implementation, training split, initialization or per-step compute. Those require the generating notebook/configuration. It is not a rerun or an independent statistical significance test. The LayerNorm+sphere hybrid is not one of the eight factorial cells and has not been verified in this audit.

ablation_table1.tex is a replacement table fragment, not a standalone document. The original manuscript has not been changed in this verification turn.
