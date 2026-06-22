# Home Credit Default Risk

Machine Learning II final project, IE University (Dr. Jaume Manero).

**Team:** Kate Orefice, Matteo Mainetti, Anthony El Feghaly

Predict the probability that a loan applicant will default, using alternative data,
so that capable borrowers with thin credit histories are not turned away.

## Result

| Stage | AUC |
|---|---|
| Baseline (application table only) | 0.769 |
| Final LightGBM, 5-fold out-of-fold | **0.79139** |
| XGBoost (comparison) | 0.79170 |
| Kaggle public leaderboard | **0.79244** |
| Kaggle private leaderboard | 0.79076 |

The out-of-fold estimate, the public score, and the private score all agree within
0.002. That is the evidence the model is not overfit and the cross-validation is honest.

## What the model does

Binary classification on the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/)
dataset. Seven relational tables are aggregated into one row per applicant (~675
features), then a LightGBM is trained with stratified 5-fold cross-validation. Metric
is AUC-ROC, because at an 8% default rate accuracy is meaningless.

## Repository structure

```
.
├── notebook/        Jupyter notebook, all 7 project phases, runs top to bottom on Kaggle
├── presentation/    HTML slide deck + speaker script and Q&A
├── report/          Interpretability report (SHAP analysis of two extreme applicants)
├── submission/      Final submission.csv scored on Kaggle
├── assets/          SHAP plots, feature importance, leaderboard screenshots
└── docs/            Original project brief + ablation documentation
```

## The pipeline (7 phases)

1. **Dimensionality reduction** — aggregate 7 tables, prune by feature importance
2. **Imbalance and probability** — 8% defaults, optimise AUC not accuracy
3. **Time-series lite** — days-past-due and payment trends over time
4. **Feature engineering** — income/credit ratios, the 365243 sentinel fix, NaN flags
5. **Battle of the GBMs** — LightGBM vs XGBoost, 5-fold cross-validation
6. **Submission** — per-applicant default probability
7. **Interpretability** — SHAP force plots, no black boxes in lending

## How to reproduce

1. Open the [competition](https://www.kaggle.com/competitions/home-credit-default-risk/),
   click Late Submission to join.
2. New Kaggle Notebook, import `notebook/home_credit_default_risk.ipynb`.
3. Add the competition dataset as input. It mounts at
   `/kaggle/input/home-credit-default-risk/` (the `PATH` in the setup cell).
4. Run All (about 30 to 45 minutes). It writes `submission.csv`.
5. Submit that file to the competition for a score around 0.79.

## Key design decisions

- **No resampling.** SMOTE and undersampling distort the 8% base rate. We optimised
  AUC directly with stratified folds; the final LightGBM also uses `is_unbalance=True`
  as a model weighting hint, not row duplication.
- **LightGBM over XGBoost.** Tied on AUC, faster on this wide, sparse matrix.
- **Feature-importance pruning, not PCA.** PCA destroys interpretability, which we
  needed for the SHAP phase.
- **No Bayesian Network.** A full network over 675 one-hot features is intractable and
  would not improve ranking. The boosting model is the probabilistic model.
- **Numeric cleanup after aggregation.** Some pandas versions keep boolean one-hot
  min/max aggregates as object columns, so the notebook coerces any remaining
  object columns to numeric before LightGBM.

## Waterfall ablation

The optional waterfall deliverable is implemented in the notebook as
`Phase 1b - Feature-group ablation`. It measures cumulative AUC on one fixed
stratified 80/20 validation split, using the same baseline LightGBM settings for
each stage:

| Stage | Added feature group | Features | Validation AUC | Delta |
|---|---|---:|---:|---:|
| Application only | application columns | 258 | 0.76925 | 0.00000 |
| + Bureau history | `BURO_` | 320 | 0.77449 | +0.00524 |
| + Previous applications | `PREV_` | 506 | 0.78037 | +0.00588 |
| + POS & Credit Card | `POS_` + `CC_` | 665 | 0.78666 | +0.00628 |
| + Installments (temporal) | `INSTAL_` | 691 | 0.78725 | +0.00059 |

The notebook output `ablation_results` is the source of truth for the waterfall.
The presentation rounds these values to 3 decimals; `docs/ablation_results.md`
keeps 5 decimals.

## Dominant feature (from SHAP)

`EXT_SOURCE_2`, an external credit score, decides both extreme cases: at 0.01 it sends
an applicant to an 81% default probability, at 0.78 it sends another to 0.2%.

## Deliverables checklist

- [x] Jupyter notebook, all phases
- [x] Leaderboard screenshot (`assets/kaggle_submission.png`)
- [x] Interpretability report
- [x] Presentation (deck + speaker script)
- [x] Optional: waterfall ablation workflow added (`ablation_results` in notebook,
      documented in `docs/ablation_results.md`)

## Links

- Competition: https://www.kaggle.com/competitions/home-credit-default-risk/
- Kaggle notebook share link: unavailable in this checkout
