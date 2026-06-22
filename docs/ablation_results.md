# Waterfall Ablation Results

## Purpose

This document supports the optional deliverable: replacing the estimated middle
bars of the presentation waterfall with measured values.

The source of truth is the notebook variable `ablation_results`, generated in
`Phase 1b - Feature-group ablation`.

## Method

- Split: one fixed stratified 80/20 validation split, `random_state=42`.
- Metric: ROC-AUC on the validation split.
- Model: the same application-only baseline LightGBM settings for every stage.
- Comparison: cumulative feature groups, added in the same order as the
  waterfall slide.
- Scope: the ablation is explanatory. The final submission model remains the
  5-fold LightGBM from Phase 5.
- Compatibility cleanup: after aggregation, the notebook coerces any remaining
  object columns to numeric because some pandas versions keep boolean one-hot
  min/max aggregates as object dtype.

## Feature Groups

| Stage | Added feature group | Feature prefix rule |
|---|---|---|
| Application only | application columns | no satellite-table prefix |
| + Bureau history | bureau and bureau balance | `BURO_` |
| + Previous applications | previous Home Credit applications | `PREV_` |
| + POS & Credit Card | POS cash and credit-card balances | `POS_`, `CC_` |
| + Installments (temporal) | installment payment behaviour | `INSTAL_` |

## Results Table

Values are copied from the notebook's `ablation_results` output. AUC and delta
values are kept to 5 decimals in this document.

| Stage | Prefixes added | Features | Validation AUC | Delta |
|---|---:|---:|---:|---:|
| Application only | application columns | 258 | 0.76925 | 0.00000 |
| + Bureau history | `BURO_` | 320 | 0.77449 | +0.00524 |
| + Previous applications | `PREV_` | 506 | 0.78037 | +0.00588 |
| + POS & Credit Card | `POS_ + CC_` | 665 | 0.78666 | +0.00628 |
| + Installments (temporal) | `INSTAL_` | 691 | 0.78725 | +0.00059 |

## Interpretation

The largest measured gains come from the satellite credit-history tables. Bureau
history adds 0.00524 AUC, previous Home Credit applications add 0.00588, and POS
plus credit-card balances add 0.00628. Installment payment behaviour still helps,
but only slightly on this fixed split (+0.00059), which is why the presentation
now frames it as a small final lift rather than a large jump.

## Presentation Sync

The presentation `WAVES` constants have been updated with these measured values.
The deck displays them rounded to 3 decimals, while this document keeps 5
decimals. The final result slide is unchanged because the final 5-fold model and
leaderboard evidence are separate from this fixed-split ablation.

## Local Run Note

The official Kaggle competition download returned `403 Forbidden` in this local
checkout, so the CSVs were downloaded from the public Kaggle dataset mirror
`megancrenshaw/home-credit-default-risk`. The filenames and table contents match
the Home Credit competition data used by the notebook.
