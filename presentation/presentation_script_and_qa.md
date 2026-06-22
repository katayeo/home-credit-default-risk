# Presentation Script & Q&A
### Home Credit Default Risk · for the class 13/14 presentation

Target: roughly 8 to 10 minutes of talking, then questions. Aim for one breath per
slide, do not read the slide out loud, say the story around it. Slides advance with
the arrow keys.

---

## Slide-by-slide script

**1 · Title.**
"Our project is Home Credit Default Risk. The goal is to predict whether a loan
applicant will repay, using alternative data, so that people with no credit history
are not automatically rejected. We score risk and we got an AUC of 0.79."

**2 · The problem.**
"The business problem is financial inclusion. The data problem is that only 8% of
applicants default. That means accuracy is useless, a model that approves everyone
is already 92% accurate and helps nobody. So we rank probability of default and
score it with AUC, where 1.0 is perfect and 0.5 is a coin flip."

**3 · The data.**
"This is what made it advanced. It is not one table, it is seven related tables:
the main application plus credit bureau records, previous loans, card balances, and
payment history. The real work was aggregating those six satellite tables down to
one row per applicant. That left us with around 675 features."

**4 · The pipeline.**
"We ran the seven phases from the brief: reduce dimensionality, handle the
imbalance, build temporal features, engineer features, compare gradient boosting
models, generate the submission, and explain it with SHAP. Each phase added
measurable value, which is what the next slide shows."

**5 · Baseline.**
"We started deliberately simple. One model on just the application table, no credit
history. That already hit 0.769, which tells us the basic financials carry real
signal. But we were ignoring the entire payment-history story."

**6 · The idea.**
"So version two brought in the history plus a few specific tricks. We aggregated the
six tables. We fixed a known data trap, the value 365243 that acts as fake infinity
in the employment column. We flagged missing values as their own signal. And we
built debt-to-income ratios, which are exactly what a human credit officer looks
at."

**7 · The waterfall (the key slide, slow down here).**
"This is the heart of it. Each bar is one change in the pipeline, isolated, so you
can see what it actually bought us. We start at 0.769 with the application table.
Adding bureau history, then previous applications, then card and point-of-sale data,
then the installment payment behaviour, each step adds value. We finish at 0.791.
The headline: the lift came from the data and the features, not from anything fancy
in the model."

**8 · Battle of the GBMs.**
"We compared LightGBM and XGBoost on the same features. They finished within
0.0003 of each other, basically tied. We shipped LightGBM because it is faster on
this wide sparse data. The five cross-validation folds all land between 0.787 and
0.798, which means the score is stable, no single fold is carrying it."

**9 · Final result.**
"Final out-of-fold AUC is 0.791, which sits in the strong public-solution range for
this competition. We did not chase a fragile leaderboard spike. This is a clean,
defensible result built on honest cross-validation." [point to leaderboard
screenshot]

**10 · Interpretability.**
"In lending a black box is not allowed. SHAP breaks a single decision into the
features that pushed it. On the left, our low-risk applicant: strong external credit
scores and stable employment push toward repay. On the right, the high-risk one:
weak credit signals and a heavy debt load push toward default. Every score is
explainable."

**11 · Lessons.**
"Three lessons. Features beat algorithms, almost all our gain came from data work.
Trust the right metric, accuracy would have lied to us. And small data bugs do big
damage, one bad value and one wrong metric setting each cost us real performance.
Next step toward top-5% would be stacking multiple models."

**12 · Close.**
"So: 0.769 to 0.791, a clean lift, fully explainable. Happy to take questions."

---

## Q&A bank (likely questions, with crisp answers)

**Why AUC and not accuracy?**
"Because the classes are 8/92. Accuracy rewards predicting the majority. AUC
measures how well we rank a random defaulter above a random non-defaulter, which is
exactly the lending task."

**Why LightGBM over XGBoost if they tied?**
"Speed and memory. On 675 mostly-sparse features LightGBM trains noticeably faster
for the same score. We kept XGBoost as a cross-check, not a tie-breaker."

**The brief mentions Bayesian Networks and PCA. Did you use them?**
Be honest: "We used feature-importance pruning rather than PCA, because tree models
keep features interpretable, which we needed for the SHAP phase. For the imbalance
phase we treated the gradient boosting model as our probabilistic model and
optimised AUC directly, rather than building a full Bayesian Network over hundreds
of one-hot features, which would not have improved ranking. We can add a small
Bayesian Network demo if you want it for completeness."

**How did you handle the class imbalance?**
"We did not resample. We let the model output calibrated probabilities and judged it
on AUC and the precision-recall curve, both of which are designed for imbalance.
The cross-validation folds were stratified to preserve the 8% rate."

**What is the 365243 thing?**
"In the employment-days column, 365243 is a placeholder the dataset uses for
unknown or not-applicable, roughly a thousand years. Left as a number it poisons the
model, so we replaced it with missing."

**Why not top-5%?**
"Top-5% needs heavy stacking and weeks of feature engineering. We prioritised a
correct, well-validated, fully explainable pipeline over a fragile leaderboard
chase. The realistic path higher is blending LightGBM, XGBoost, and CatBoost plus
more bureau-balance trend features."

**Is SHAP causal?**
"No. SHAP explains what the model relied on given the data, not what causes default.
It is for transparency, not causal claims."

**Out-of-fold AUC versus the leaderboard score?**
"Out-of-fold is our own honest estimate, every training row scored by a model that
never saw it. The leaderboard score is on Kaggle's hidden test set. They should be
close, and ours are."

**How did you avoid data leakage?**
"We aggregated the satellite tables before splitting, but all the feature values are
historical relative to the application, and the cross-validation never lets a
training fold see its validation rows. The tight, even fold scores are the evidence
there is no leak."

---

## ADDENDUM: expanded 20-slide deck (depth + appendix)

The deck now has 17 presentation slides plus a 3-slide appendix (18 to 20) that you
only flip to if asked. Full running order:

1 Title · 2 Problem · 3 Data · 4 Pipeline · 5 Baseline · 6 The idea ·
**7 Aggregation** · 8 Waterfall · 9 Feature importance · **10 Validation** ·
**11 Imbalance** · 12 Battle of GBMs · 13 Final result · 14 SHAP global ·
15 SHAP cases · 16 Lessons · 17 Close · [18 Hyperparameters · 19 Design decisions ·
20 Gap to top-5%]

### Script for the new slides

**7 · Aggregation.**
"Quick on the hardest mechanical part. One applicant appears many times in these
tables, say three past loans in the bureau. The model needs one row per person, so
we collapse each column with mean, max, min and sum. Three loan amounts become a
mean, a max, a sum. Do that across all six tables and you get the 675 features."

**10 · Validation.**
"How do we know 0.79 is real and not luck? Five-fold stratified cross-validation.
We split into five parts, each kept at the 8% default rate, train on four and
predict the fifth, and rotate. So every applicant is scored by a model that never
saw them. The five folds all land between 0.787 and 0.798, and the leaderboard came
back 0.792. That agreement is how we know there is no leakage."

**11 · Imbalance.**
"On the imbalance, we chose not to resample, because oversampling distorts the real
8% rate and the probability calibration. We optimised AUC directly. And here is a
real lesson: our first run set is_unbalance to true, it broke early stopping, the
model quit after three rounds and scored 0.72. We removed it, set the metric to AUC,
and went straight back to 0.79. The fix was worse than the problem."

**18 to 20 · Appendix (only if asked).**
Hyperparameters: the full LightGBM config. Design decisions: why no Bayesian Network
or PCA, why no resampling, why LightGBM. Gap to top-5%: winners used stacking and
hundreds more features to reach ~0.805; we prioritised a correct, explainable
pipeline.

### Suggested split across three presenters (~20 min)

- **Presenter 1, the setup (slides 1 to 6, ~6 min).** Problem, data, the pipeline,
  baseline, the idea. Sets up the whole story.
- **Presenter 2, the method (slides 7 to 11, ~7 min).** Aggregation, the waterfall,
  feature importance, validation, imbalance. The technical core. Owns appendix
  questions on hyperparameters and design decisions.
- **Presenter 3, results and meaning (slides 12 to 17, ~7 min).** GBM comparison,
  final result, the two SHAP slides, lessons, close. Owns the appendix question on
  the gap to top-5%.

Hand-offs should be one sentence: "I will hand over to [name] to walk through the
method." Practice the two transitions, they are where group talks usually stumble.
