# Does the "distillation understates AL's benefit" finding generalize as a theorem?

RESULTS.md §4 reports a single empirical comparison: training/evaluating on
VarWISE's own predicted labels (smoother, because it over-predicts rare
classes) shows a smaller measured active-learning advantage (72.4% saving)
than evaluating on independent SIMBAD truth (85.7%). That is one archive, one
comparison — "suggestive, not established," as flagged at the time.

This note asks whether a general mathematical statement sits behind that one
data point, derives one, and tests it on controlled synthetic data where the
truth is known by construction rather than inferred from a single real-world
pair.

**Bottom line up front:** the core mechanism is confirmed cleanly. The
specific quantitative prediction is confirmed only weakly, and the most
likely reason is simulation noise (3 seeds) rather than a flaw in the
mechanism — but that is not proven, only argued.

---

## 1. The formal claim

**Setup.** Random sampling needs roughly `k_min / π_c` labels to collect
`k_min` examples of class `c` (a standard multinomial coupon-collector bound,
provable via Chernoff tail bounds on a binomial). Active learning, once it
has located each class's region of feature space, needs roughly `O(K)` labels
total, independent of `π_c` — it can query members of a rare class directly.

**Aggregate consequence.** To reach a fixed macro-performance target, every
class must individually reach competence, so random sampling's total budget
is bottlenecked by whichever class is rarest:

```
N_random(π) ≈ k_min · max_c(1/π_c) = k_min / π_min
```

**The provable part.** `π ↦ max_c(1/π_c)` is a pointwise supremum of convex
functions (each `1/π_c` is convex), hence convex; it is also symmetric under
permuting classes. A function that is both convex and symmetric on the
simplex is **Schur-convex** — this is a standard result (the
convex-plus-symmetric sufficient condition for Schur-convexity), not a new
theorem being asserted here.

**The corollary.** Schur-convexity means: if the true class distribution `π`
majorizes a smoothed distribution `π'` (formally, `π` is "more unequal"),
then `N_random(π) ≥ N_random(π')`. Since VarWISE's own predictions inflate
rare-class share relative to SIMBAD truth (`cv` predicted 7.51% vs true
0.14%) — a textbook majorization-decreasing smoothing — the theory predicts
exactly what was observed: smaller measured savings against the smoothed
target.

**What is idealized, not proven.** Two assumptions carry the whole argument:
(i) random scales as `Θ(1/π_min)`, and (ii) active learning scales as `O(K)`,
independent of `π_min`. Neither holds exactly for a real gradient-boosted or
logistic classifier.

Assumption (i) is elementary probability (the expected wait time for a
negative-binomial/coupon-collector process), not attributed to any specific
paper. Assumption (ii) and the overall claim that active learning
dramatically outperforms random sampling for rare-category discovery is an
*established empirical phenomenon* in the rare-category-detection literature
— He & Carbonell 2008, "Rare Class Discovery Based on Active Learning"
(ISAIM), and Pelleg & Moore 2004, "Active Learning for Anomaly and
Rare-Category Detection" (NeurIPS), both demonstrate large empirical gains
(Pelleg & Moore: their method can "quickly zoom in on an anomaly set
containing a few tens of points in a dataset of hundreds of thousands").

**Citation check performed 17 August 2026** (both papers fetched and
confirmed to exist under the cited title/authors/venue): neither paper was
found to state the specific `Θ(1/π_min)` vs `O(K)` asymptotic scaling law,
or a majorization/Schur-convexity framing, in those terms. That specific
derivation is original to this note, built from standard, individually
well-known components (a coupon-collector bound and a textbook
convex-plus-symmetric Schur-convexity criterion) rather than drawn from the
cited papers directly. The citations above should be read as "closest prior
art for the empirical phenomenon this mechanism is trying to explain," not
as "this scaling law is already proven there." No existing paper was found
connecting the mechanism specifically to *distillation-target evaluation
bias* in label-efficiency studies — see the novelty note at the end.

---

## 2. Free consistency check (real archive, 2 data points)

Track B's rarest class is `cv` at 0.14%; Track A's is `cep` at 0.70%
(`1/π_min` ~5× larger for B). The theory predicts bigger AL savings for B —
observed: 85.7% (B) vs 72.4% (A). Direction matches. Two points, not a
regression; already fact-checked as part of RESULTS.md.

---

## 3. Synthetic validation

`scripts/synthetic_majorization_test.py` builds a **literal majorization
chain by construction**, rather than relying on one real-world pair:

```
π(t) = (1-t)·π_0 + t·uniform,   t ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
π_0  = [0.50, 0.25, 0.10, 0.08, 0.05, 0.02]
```

Linear interpolation toward the uniform vector is a standard
majorization-decreasing path: `π(t1)` majorizes `π(t2)` for every `t1 < t2`,
guaranteed by construction, not something the simulation needs to establish.

For each `t`, synthetic 6-class data is drawn with **exactly** those class
proportions (`sklearn.datasets.make_classification(weights=π(t))`), and the
real `common/active_learning.py` loop is run — `margin_score` (the best
strategy in the real study) vs `random_score` — 3 seeds, batch size 30, 25
rounds (14 → 768 labels), `LogisticRegression` for speed.

### 3a. The raw mechanism — clean confirmation

Random sampling's own recovery of the rarest class's F1, at a fixed
300-label checkpoint (not the curve's final ~768-label endpoint — this is
mid-curve, where the coupon-collector effect should be most visible), as
`π_min` varies:

| `π_min` | random's rarest-class F1 |
|---|---|
| 0.0200 | 0.399 |
| 0.0493 | 0.504 |
| 0.0787 | 0.602 |
| 0.1080 | 0.692 |
| 0.1373 | 0.719 |
| 0.1667 | 0.818 |

**Monotonically increasing, every step, no exceptions.** This is exactly the
coupon-collector prediction and it holds cleanly under construction where the
prevalence relationship is known exactly, not inferred.

### 3b. The downstream "savings" prediction — weak, noisy support

Two ways of measuring "AL's advantage over random" as a function of `π_min`:

**Aggregate macro-F1 savings at a fixed 300-label budget:**

| `t` | `π_min` | savings |
|---|---|---|
| 0.0 | 0.0200 | +40.9% |
| 0.2 | 0.0493 | +47.4% |
| 0.4 | 0.0787 | +31.3% |
| 0.6 | 0.1080 | +28.3% |
| 0.8 | 0.1373 | +18.5% |
| **1.0** | **0.1667** | **+49.5%** |

Spearman ρ(`π_min`, savings) = **−0.086** — essentially zero, and the most
balanced case (`t=1`) shows the *highest* savings, opposite the naive
prediction.

**Rarest-class-specific savings** (labels for random vs margin to reach the
same rarest-class F1 target — the direct analog of the real archive's
per-class methodology, rather than an aggregate mixing in five other
classes):

| `t` | `π_min` | savings |
|---|---|---|
| 0.0 | 0.0200 | +55.1% |
| 0.2 | 0.0493 | +75.1% |
| 0.4 | 0.0787 | +62.6% |
| 0.6 | 0.1080 | +36.3% |
| 0.8 | 0.1373 | +49.4% |
| 1.0 | 0.1667 | +59.0% |

Spearman ρ = **−0.314** — right sign, far weaker than the real archive's
−0.857, and not monotonic.

### Why the targeted metric is still weak

The aggregate metric is confounded by construction: six different `t` values
are six *different* six-class problems, not the same problem relabeled — at
`t=1` every class sits at `π=0.167`, individually easy to sample, but the
macro-F1 target now depends on five other classes whose own difficulty is not
held fixed by the idealized model. The rarest-class-specific metric removes
that confound and improves the correlation (−0.086 → −0.314), but doesn't
close the gap to −0.857.

**The most likely explanation is simulation noise, not a wrong mechanism.**
Seed-to-seed standard deviation in the rarest class's F1 is 0.05–0.08 across
the sweep — comparable in size to the between-`t` differences the correlation
is trying to resolve, with only 3 seeds per point. That is a real limitation
of this test, not evidence the theory is wrong; it is also not proof the
theory is right at this quantitative level. More seeds (10+) and more `t`
values would be needed to tell the two apart.

---

## 4. Honest verdict

| claim | status |
|---|---|
| Random sampling's rarest-class recovery degrades as `π_min` shrinks | **Confirmed cleanly**, monotonic, no exceptions, both in the real archive and in controlled synthetic data |
| `N_random(π) ≈ k_min/π_min` is Schur-convex, so majorization-decreasing smoothing shrinks measured random-sampling deficiency | **Provable as stated** (standard convex-plus-symmetric argument); not independently in question |
| Real classifiers' AL advantage over random, aggregated to a macro-performance target, tracks `π_min` via this mechanism | **Weakly supported.** Right direction on the real archive (2 points) and on synthetic per-class savings (ρ=−0.314, 6 points), but far short of a tight quantitative match, and the aggregate-metric version showed no relationship at all (ρ=−0.086) |

This is evidence *for* the conjectured mechanism under controlled conditions,
not a validated theorem. What remains to close the gap: more seeds per
scenario (the noise diagnosis above is the natural next test — if ρ sharpens
toward −0.857 with 10+ seeds, the mechanism is quantitatively right and this
was noise; if it stays near −0.3, the idealized `O(K)` active-learning
assumption is doing too much work and needs revision), and a harder
separability regime to check the result isn't an artifact of the specific
`make_classification` recipe used here.

## 5. Novelty

The building blocks — coupon-collector bounds, Schur-convexity, rare-category
active learning — are all standard and pre-existing (searched again 17 August
2026: no paper found connecting majorization/Schur-convexity to active-
learning label-efficiency, nor to distillation-target evaluation bias
specifically). The reframing is the contribution, and per the analysis above
it currently supports a mechanism claim, not a validated quantitative
theorem. Presented honestly, this is a small theory note with a clearly
flagged gap, not a finished result.
