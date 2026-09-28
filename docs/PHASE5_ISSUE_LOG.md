# Phase 5 Issue Log

**Project** SECOM feature-selection stability study (`iipp_task2`)
**Prepared** 2026-09-25, at handover of the project from Phase 5 onward
**Status** For review. **No change has been made to `01`–`04`, to the splits, or to any stored result.**

## Purpose and standing

The Locked Research Protocol (Phases 0–4, 4R, 4R.1) forbids unilateral changes to the Research
Questions, the feature-selection methods, the models, the metrics or the validation strategy. This
log is the documented alternative: every issue found while reading the code is recorded here with
its evidence, its effect on the affected Research Question, a recommendation, and an explicit note
on whether acting on it would amend the protocol or merely correct an implementation defect.

Nothing in section A or B will be implemented until you sign it off item by item.

## Evidence base

All numbers below come from one experiment run and can be regenerated from it.

| Artefact | SHA-256 (first 16) |
|---|---|
| `experiments/20260925_022325_results.json` | `d260e62a2bda04c8` |
| `experiments/20260925_022325_selections.json` | `3901e967433da9e1` |
| `experiments/20260925_022325_stability.json` | `03c330990e1f8929` |
| `experiments/20260925_022325_feature_frequency.json` | `27e62c9b62342a61` |
| `experiments/20260925_022325_summary.json` | `4523c1ac42ce68de` |

Dataset: 1567 wafers, 104 Fail, positive rate 0.0664. Run grid: 5 selectors × 3 imbalance
conditions × 3 models × 50 folds = 2250 evaluation rows, 750 selection records.

Corrected stability figures quoted below are produced by `05_visualize_result.py`, which recomputes
them read-only from the same artefacts and writes them to
`experiments/graph/20260925_022325/tables/stability_recomputed.csv`. It never modifies an input.

## Severity scale

| Level | Meaning |
|---|---|
| **P1** | A Research Question cannot be answered, or a reported number is wrong |
| **P2** | A protocol clause is not satisfied as written |
| **P3** | A result is defensible but weaker than the protocol intends |
| **P4** | Housekeeping, no effect on findings |

---

# A. Protocol deviations — your decision required

## A1 · Protected test set is 10%, not the locked 20% — P2

**Claim.** Phase 4 locks a 20% protected test set for final confirmatory evaluation. The split on
disk is 10%.

**Evidence.**

- `03_protected_split.py:24` — `TEST_SIZE = 0.10`, used at line 89 and recorded at line 103.
- `splits/random/split_config.json` — `"test_size": 0.1`, `n_dev: 1410`, `n_test: 157`.
- The 157 held-out wafers contain **10 Fail**. A 20% split would hold ~313 wafers and ~21 Fail.

**Why it matters.** Confirmatory PR-AUC on 10 positives has a confidence interval wide enough to
cover almost any effect the development phase might report. Any final claim of the form "the reduced
feature set holds up on unseen data" rests on those 10 wafers.

**Options.**

1. **Re-split at 20%.** `03_protected_split.py:79` raises `FileExistsError` if a split exists, so
   this is a deliberate act: delete `splits/random/`, set `TEST_SIZE = 0.20`, rerun `03`. Every
   development result must then be regenerated, because the development set changes from 1410 to
   ~1254 rows. Cost: one full `04` rerun.
2. **Keep 10% and amend the protocol on record**, stating that the confirmatory set holds 10
   positives and that the confirmatory claim is correspondingly weak.

**Recommendation.** Option 1, and do it before any further experiment, because everything
downstream is invalidated by a later re-split. If the schedule cannot absorb the rerun, option 2 is
defensible only if the thinness is stated in the results chapter, not buried.

**Decision required.** Re-split, or amend the protocol.

## A2 · The protected test set has never been evaluated — P1

**Claim.** The confirmatory evaluation step does not exist in code.

**Evidence.** `04_track_evaluation.py:305` builds `test_df`, and no later line reads it. The script
ends at the development summary. Grep confirms `test_df` appears exactly once.

**Why it matters.** RQ1 and RQ3 currently rest on 50 development folds only. The Phase 3 chain ends
at *decision reliability*, which the protocol ties to a confirmatory evaluation.

**Recommendation.** Add a separate confirmatory script (`06_confirmatory_test.py`) rather than
extending `04`, so the development experiment stays untouched and the protected set is read in
exactly one place. It should: build the Final Stable Feature Set from development data only (see
C1), refit the chosen configurations on the whole development set, evaluate once on the protected
test set, and stop. One pass, no tuning, no repetition.

**Decision required.** Approve the new script, and fix A1 and C1 first, since both feed it.

## A3 · The temporal robustness track has never been run — P2

**Claim.** The temporal split exists but nothing consumes it.

**Evidence.** `splits/temporal/split_config.json` — cutoff `2008-10-04T23:18:00`, dev 1303 rows /
89 Fail, test 264 rows / 15 Fail. `04_track_evaluation.py:303` loads `DEV_RANDOM_PATH` only; the
temporal constants at lines 44–46 are defined and never used.

**Why it matters.** The protocol names random-stratified and temporal evaluation as complementary
tracks. With one track missing there is no evidence on whether the stable variable set survives a
shift in time, which is the claim a fab would actually care about.

**Recommendation.** Run the existing `04` grid a second time against the temporal development set,
writing to a separate `run_id`, and report the two tracks side by side. No code change beyond
parameterising which split is loaded.

**Decision required.** Approve running the second track, and confirm it is reported as complementary
evidence rather than as a second independent test.

---

# B. Measurement defects — approval needed, these change published numbers

## B1 · Stability is computed over the wrong variable universe — P1, primary RQ

**Claim.** Nogueira and Kuncheva are computed against all 590 raw variables, but only ~292 can ever
be selected. Variables that cannot be chosen contribute no variance and inflate the index.

**Evidence.**

- `04_track_evaluation.py:312` — `feature_universe = X.columns.tolist()`, taken before
  `preprocess_fold` runs, so p = 590.
- Per fold, preprocessing leaves 271.6 variables on average (min 267, max 277); the union across all
  folds is **292**.
- Recomputed on the 292-variable pool:

| condition | Nogueira as reported by `04` | corrected | inflation |
|---|---|---|---|
| all features (any imbalance) | 0.970 | **0.766** | 0.204 |
| Boruta + SMOTE | 0.968 | **0.773** | 0.195 |
| L1 + SMOTE | 0.588 | **0.399** | 0.189 |
| L1 + class weight | 0.592 | **0.463** | 0.129 |
| L1 + none | 0.598 | **0.555** | 0.043 |
| mutual information + SMOTE | 0.947 | 0.945 | 0.002 |

Mean inflation 0.087, maximum 0.204. The distortion is largest exactly where the subsets are
largest, so it does not cancel out — it reorders the selectors.

**Why it matters.** This is the primary Research Question's headline number.

**Recommendation.** Compute the universe after preprocessing. Two defensible definitions: the union
of per-fold surviving pools (292, used above), or the intersection. Report which one and why. Keep
the raw-universe figure in an appendix so the correction is visible rather than silent.

**Decision required.** Approve the fix in `04`, and choose union or intersection.

## B2 · Selection frequency uses the wrong denominator — P1

**Claim.** Frequency divides by 50 folds regardless of whether the variable survived preprocessing
in that fold.

**Evidence.** `04_track_evaluation.py:275–284` — `freq = selection_frequency(Z)` over all 50 rows,
with `Z` built on the 590-variable universe.

**Why it matters.** A variable dropped by the missing-rate or correlation filter in some folds was
never a candidate there, yet those folds count against it. Its frequency is understated, which
propagates into the top-k explanation sets that RQ5 depends on.

**Recommendation.** Denominator = number of folds in which the variable survived preprocessing.
Record the surviving pool per fold in `selections` so this is computable after the fact.

**Decision required.** Approve, together with B1 — they share the same root cause.

## B3 · Every thresholded metric is fixed at 0.5 — P1

**Claim.** Recall, precision, F1 and MCC describe the decision threshold, not the models.

**Evidence.** `04_track_evaluation.py:357` — `y_pred = (y_proba >= 0.5).astype(int)`, on a 6.64%
positive rate. Share of folds with recall exactly zero:

| imbalance | logistic | random forest | XGBoost |
|---|---|---|---|
| none | 55.2% | **89.2%** | 75.6% |
| class weight | 0.0% | **97.6%** | 37.2% |
| SMOTE | 0.4% | 57.6% | 19.6% |

Overall, **48%** of all 2250 rows report recall = 0. Random forest under class weighting is blind in
97.6% of folds while its PR-AUC is 0.198, among the best in the run.

**Why it matters.** RQ4 asks how imbalance-aware learning affects minority-class prediction. Read at
0.5, the answer is dominated by the threshold. PR-AUC, the locked primary metric, is unaffected.

**Recommendation.** Keep PR-AUC as primary, unchanged. Add a threshold chosen inside the training
fold — F1-max or recall at a fixed precision — and report the thresholded metrics at that operating
point. Selecting the threshold on the training fold only keeps the no-leakage rule intact. If you
prefer not to add an operating-point rule, the alternative is to drop the thresholded columns from
the claims entirely and say why.

**Decision required.** Approve an in-fold threshold rule, or approve dropping the thresholded
metrics from the reported findings.

## B4 · `(mutual information, class weight)` is not a real experimental arm — P3

**Claim.** The cell duplicates `(mutual information, none)` exactly at the selection stage.

**Evidence.** `04_track_evaluation.py:136–138` — the `mi` branch ignores the `cw` variable built at
line 130. Verified: the 50 selected subsets for `(mi, none)` and `(mi, weight)` are byte-identical,
and both report Nogueira 0.367 and mean Jaccard 0.262.

**Why it matters.** RQ4 asks how imbalance handling alters *selection* stability. For mutual
information the answer is structurally zero, not empirically zero, and the figure currently invites
the reader to treat it as a finding. The downstream model still differs, so the RQ1 cell remains
valid.

**Recommendation.** No code change needed. State in the results that mutual information has no
class-weight mechanism, so that cell is a no-op by construction. Optionally skip it to save compute.

**Decision required.** Confirm the wording; no protocol change.

## B5 · Algorithm-induced stability is not measured — P3

**Claim.** The per-fold seed is frozen, so only data-induced stability varies.

**Evidence.** `04_track_evaluation.py:327–328` — `# seed = RANDOM_SEED + fold_id` is commented out
and `seed = RANDOM_SEED` is active. SMOTE, Random Forest, Boruta and XGBoost all receive the same
seed in every fold.

**Why it matters.** Phase 4 distinguishes four stability kinds. Only data-induced stability is
currently observable. This is a sound choice for RQ2 as worded — it isolates resampling — but it
must be stated, and the taxonomy promises the other three.

**Recommendation.** Keep the frozen seed for the RQ2 headline. Add a small side experiment that
holds the fold fixed and varies the algorithm seed, to quantify algorithm-induced stability
separately. It does not need the full grid.

**Decision required.** Approve the side experiment, or accept that the taxonomy is reported with one
of four kinds measured.

## B6 · Kuncheva is unavailable for 9 of 15 conditions — P3

**Claim.** The fixed-k comparison the protocol promises exists for only the two fixed-size selectors.

**Evidence.** `experiments/20260925_022325_stability.json` — `kuncheva` is `NaN` in 9 of 15 cells:
every `none`, `l1` and `boruta` row, in all three imbalance conditions. Only `mi` and `xgb`, both
pinned at k = 20, produce a value. `04_track_evaluation.py:233–239` asserts equal subset sizes, and
line 271 guards with `same_size`.

**Why it matters.** Cross-selector stability comparison currently rests on Nogueira alone. Jaccard
is size-biased and favours the selectors that keep more variables.

**Recommendation.** Add a matched-k arm: force every selector to its top-k for a common k (20 is the
natural choice, since two selectors already use it) purely for the stability comparison, leaving the
native-size runs untouched for RQ1 and RQ3. This adds a comparison, it does not replace one.

**Decision required.** Approve the matched-k arm and the value of k.

## B7 · Preprocessing contributes unattributed instability — P3

**Claim.** Part of what is reported as selector instability is produced by the preprocessing filters.

**Evidence.** The all-features reference scores **0.766**, not 1.0, on the corrected universe. It
selects nothing — it keeps whatever survives — so the entire 0.234 shortfall comes from the per-fold
filters. `utils.py:correlation_filter` drops the second member of each correlated pair in descending
correlation order, so which twin survives can change with the fold.

**Why it matters.** Every selector inherits this floor. A selector scoring 0.583 is not 0.583 of the
way to perfect reproducibility; it is below a ceiling of 0.766 that preprocessing already imposed.

**Recommendation.** Report the all-features row explicitly as the preprocessing-induced floor, and
add it to the Phase 4 stability taxonomy as a fifth kind. `05_visualize_result.py` already plots it
and its caption already says so.

**Decision required.** Confirm the taxonomy gains a fifth kind; no code change.

---

# C. Missing deliverables

## C1 · No rule for constructing the Final Stable Feature Set — P1

**Claim.** The protocol requires a stable feature set built from development data only. No code
constructs one.

**Evidence.** `04_track_evaluation.py` writes per-fold subsets and frequencies and stops. There is
no consensus threshold, no aggregation rule, no saved final set anywhere in the repository.

**Why it matters.** It is the object RQ5 compares against SHAP, the input to the confirmatory
evaluation (A2), and the artefact a process engineer would actually receive. Without it the Phase 3
chain stops at prediction reliability and never reaches decision reliability.

**Recommendation.** Define one rule and fix it before looking at any result, so the choice cannot be
tuned to the outcome. The conventional choice is a selection-frequency threshold π — a variable
enters the final set when it is selected in at least π of the folds, with π commonly 0.6 to 0.8.
State π, state the selector and imbalance condition the final set is built from, and state the
resulting set size before evaluating it.

**Decision required.** Choose π and the source condition.

## C2 · RQ5 has no data at all — P1

**Claim.** No SHAP values, coefficients or permutation importances are persisted, and `shap` is not
installed.

**Evidence.** `pip list` in `.venv` shows no `shap`. No script imports it. `04` discards each fitted
model after scoring. `05_visualize_result.py` therefore answers RQ5 with a labelled proxy —
selection frequency and cross-method overlap of top-10 sets — and says so in the figure title and
caption.

**Why it matters.** RQ5 asks whether stable variables agree with SHAP importance across repeated
fits. That question is currently unanswerable.

**Recommendation.** Install `shap`. In `04`, for each fold and each fitted model, compute SHAP on
the validation fold and persist the mean absolute value per variable. Then measure importance-rank
stability (Spearman across folds, and top-k Jaccard) and compare it with selection stability. Note
the cost: SHAP on Random Forest over 50 folds is the expensive part; `TreeExplainer` keeps it
tractable, and Logistic Regression can use coefficients directly.

The protocol's rule that SHAP is post-hoc and must not build the stable feature set stays intact —
these values are only compared against the set, never used to construct it.

**Decision required.** Approve adding SHAP persistence to `04`, and confirm the explainer choice per
model.

## C3 · No hyperparameter tuning, and no recorded decision not to tune — P3

**Claim.** Model parameters are hard-coded; no search of any kind exists.

**Evidence.** `04_track_evaluation.py:68–73` fixes `MODEL_PARAMS`. No `GridSearchCV`,
`RandomizedSearchCV` or `cross_val_score` appears in `01`–`04`; the only match in the repository is
inside a docstring in `utils.py:202`.

**Why it matters.** The protocol says tuning must not be fitted on validation or test data, which
implies tuning was anticipated. A reviewer will ask whether the comparison is confounded by
parameters that happen to suit one selector.

**Recommendation.** Either declare "no tuning, fixed published defaults, identical across all arms"
as a protocol decision — which is clean and defensible for a stability study — or add nested CV
inside the training fold. Declaring no tuning costs nothing and is my recommendation.

**Decision required.** Declare no tuning, or approve nested CV.

## C4 · No statistical comparison inside the experiment — P3

**Claim.** `04` reports means and standard deviations only.

**Evidence.** The `summary` artefact holds `pr_auc_mean` and `pr_auc_std`; no interval, no test.

**Why it matters.** RQ1 asks whether performance is *preserved*, which is a claim about a difference
being small, and RQ3 ranks configurations. Both need uncertainty.

**Recommendation.** `05_visualize_result.py` already computes paired per-fold differences against
the all-feature reference with 95% percentile bootstrap intervals, seeded at 42 with 5000 resamples,
and writes them to
`experiments/graph/<run_id>/tables/rq1_02_change_against_all_features.csv`. Adopt that as the
reporting standard, and note that folds from repeated CV overlap, so the intervals are indicative
rather than exact.

**Decision required.** Confirm the bootstrap interval as the reporting standard.

## C5 · The monitoring-complexity end of the Phase 3 chain has no metric — P3

**Claim.** The framework ends at *monitoring complexity and decision support*; nothing measures it.

**Evidence.** All current outputs stop at prediction and stability.

**Recommendation.** Make it concrete and cheap: report the size of the final stable set as the
monitoring load, alongside the PR-AUC retained and the stability achieved. That is a defensible
proxy — a fab monitors variables, so the count is the cost — and it needs no new experiment.

**Decision required.** Confirm the proxy, or name a different one.

---

# D. Housekeeping

## D1 · `requirements.txt` cannot rebuild the environment — P4

`Boruta 0.4.3` is imported by `04` and absent from the file. `scipy` is installed and unlisted.
`shap` will be needed for C2. Conversely `lightgbm` and `catboost` are listed and never imported.

**Recommendation.** Pin what is actually imported, drop what is not. Add `shap` when C2 is approved.

## D2 · Minor code defects in `04` — P4

None affect results.

- Lines 318 and 324 construct `RepeatedStratifiedKFold` twice with identical arguments.
- Line 86 has an unbalanced parenthesis inside an f-string message.
- Unused imports: `plt`, `save_figure`, `VotingFeatureSelector`, `find_temporal_cutoff_candidates`,
  `train_test_split`, `classification_report`, `confusion_matrix`, `KNeighborsClassifier`,
  `SelectFromModel`, and the unused constant `VAL_SIZE`.
- `XGBClassifier` may be `None` if the import fails at line 30, with no guard in `select_features`
  or `build_model`.
- `kuncheva_index` carries a Thai-language assertion message; English is better for a paper
  appendix.

---

# E. Proposed order of work

Dependencies, not preference. Each step assumes the one above is signed off.

1. **A1** — settle the protected split size. Everything downstream depends on the development set.
2. **B1 + B2** — fix the stability universe and the frequency denominator. These change the primary
   RQ2 numbers, so they must land before anything is written up.
3. **B3** — decide the threshold policy, which fixes what RQ4 is allowed to claim.
4. **C1** — fix π and build the Final Stable Feature Set.
5. **C2** — add SHAP persistence, then answer RQ5.
6. **A3 + B5 + B6** — the complementary arms: temporal track, algorithm-seed side experiment,
   matched-k stability comparison.
7. **A2** — the confirmatory evaluation on the protected test set. Last, and run once.
8. **D1 + D2 + B4 + B7 + C3 + C4 + C5** — housekeeping and reporting decisions, any time.

Steps 1–2 require a full rerun of `04`. Step 5 requires another. Budget two full grid runs.

# F. What has not been touched

`01_secom_data_audit.py`, `02_temporal_audit.py`, `03_protected_split.py`,
`04_track_evaluation.py`, `utils.py`, every file under `splits/`, and every artefact under
`experiments/` are unmodified.

The only file added during this review is `05_visualize_result.py`, which reads the JSON artefacts
and writes figures, tables, captions and a manifest under `experiments/graph/<run_id>/`. It computes
the corrected stability figures quoted in B1 and B7 without altering any stored result, reports both
the corrected and the original universe side by side so the difference is auditable, and is
deterministic on rerun.
