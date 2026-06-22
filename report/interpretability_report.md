# Interpretability Report
### Home Credit Default Risk · Deliverable 3
**Group members:** Kate Orefice, Matteo Mainetti, Anthony El Feghaly

---

## Why interpretability matters here

In consumer lending a model cannot be a black box. Regulators require that any
automated credit decision can be explained back to the applicant ("adverse action
reasons"), and a loan officer has to be able to defend why someone was scored as
high risk. So alongside the AUC number, we explain *how* the model reaches a
decision for an individual.

We use **SHAP (SHapley Additive exPlanations)**. SHAP assigns every feature a
signed contribution to a single prediction: positive values push the predicted
probability of default **up**, negative values push it **down**. The contributions
sum from a baseline (the average prediction across all applicants, here a log-odds
of about -2.43) to the applicant's final score. We ran `shap.TreeExplainer` on the
final LightGBM model and produced a force plot for the two most extreme applicants.

---

## Global picture (summary plot)

Across the sampled applicants, the model's strongest drivers are, in order:

1. **EXT_SOURCE_2, EXT_SOURCE_3, EXT_SOURCE_1** — normalized scores from external
   credit data. These three dominate every other feature. A high score pushes
   strongly toward repayment.
2. **PREV_CNT_PAYMENT_MEAN** — average term length of the applicant's previous
   Home Credit loans (a satellite-table feature).
3. **PAYMENT_RATE** — annuity relative to credit, i.e. the debt-service burden.
4. **AMT_ANNUITY, AMT_GOODS_PRICE, DAYS_EMPLOYED** — loan size and job stability.
5. **INSTAL_DPD_MEAN** — average days-past-due on previous installments. A history
   of late payment pushes risk up.

The takeaway: the model leans on external credit signals, prior payment behaviour,
debt burden, and employment. Nothing exotic, all of it defensible to a regulator.

---

## Case 1 — High-risk applicant

**Predicted probability of default: 0.814** (log-odds f(x) = 1.48, well above the
~0.08 average).

Features pushing the score **toward default** (red / positive SHAP):

- **EXT_SOURCE_2 = 0.010** — effectively the lowest possible external credit score.
  This single feature is the largest contributor to the high prediction.
- **EXT_SOURCE_3 = 0.408** — also low, reinforcing the weak external-credit signal.

With both external scores weak, almost nothing pulls the prediction back down, so
the applicant lands at an 81% probability of default.

**Plain reading:** a borrower whose external credit signals are poor across the
board. A loan officer would decline, and the explainable reasons given to the
applicant would be the external credit scores.

---

## Case 2 — Low-risk applicant

**Predicted probability of default: 0.002** (log-odds f(x) = -6.24, far below
average).

Features pushing the score **toward repayment** (blue / negative SHAP):

- **EXT_SOURCE_2 = 0.783** — a strong external credit score, the dominant reason
  this applicant is scored as safe. Supporting blue contributions from the other
  external scores and a clean profile push the prediction down further.

**Plain reading:** a financially established applicant with strong external credit.
This is exactly the kind of capable borrower the project set out to approve with
confidence.

---

## What this buys us

Notice the symmetry: **EXT_SOURCE_2 is the deciding feature in both cases.** At 0.01
it sends one applicant to an 81% default probability; at 0.78 it sends another to
0.2%. Same feature, opposite value, opposite outcome. That is the model behaving
exactly as a sensible credit model should, and every score breaks down into named,
defensible factors rather than an opaque number.

**Caveat:** SHAP explains *association inside the model*, not causation. A high
contribution means the model learned to rely on that feature given the training
data; it does not prove the feature causes default. The model also inherits any
bias in the historical data, which is why human review and fairness checks sit on
top of a score like this in practice.
