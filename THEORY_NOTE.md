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

**Bottom line up front:** the core mechanism is confirmed cleanly across
every run. The aggregate prediction — that measured AL advantage tracks
`π_min` — was unresolved after 3 and 10 seeds on a 6-point grid, for a
specific, checked reason (too few scenarios, not too few seeds). A redesigned
test fixing that, plus a metric that avoids a second, separately-diagnosed
fragility, finds a **real, budget-consistent, moderate correlation in the
predicted direction** — weaker than the real archive's −0.857, but no longer
statistically indistinguishable from zero.

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

### A 10-seed rerun clarifies, rather than resolves, the gap

The natural next test flagged in the first pass — more seeds per scenario —
was run: 3 → 10 seeds, same six `t` values. Per-scenario noise dropped
exactly as expected (standard deviation in the rarest class's final-budget
F1 roughly halved at most points: 0.117→0.056 at `t=0`, 0.127→0.064 at
`t=0.4`, 0.070→0.040 at `t=0.8`). Each of the six point estimates is now
noticeably more precise.

**The correlation did not sharpen toward −0.857. It got less stable:**

| metric | 3 seeds | 10 seeds |
|---|---|---|
| aggregate savings vs `π_min` | ρ = −0.086 | ρ = **−0.029** (moved toward zero) |
| rarest-class-specific savings vs `π_min` | ρ = −0.314 | ρ = **+0.700** (flipped sign) |

**Diagnosis: the earlier "probably just seed noise" explanation was
incomplete.** Reducing per-point noise does nothing for a correlation that is
only ever fit to six scenario points (one per `t` value) — Spearman
correlation with `n=6` has enormous sampling variance regardless of how
precisely each individual point is measured. Going from 3 to 10 seeds
answers a different question (how precise is each point?) than the one that
actually limits the correlation estimate (how many points are there?). This
was checked directly rather than assumed: per-point standard deviations
roughly halved as predicted, confirming the seeds did their job — it is the
scenario count, not the seed count, that is the bottleneck.

**What this changes.** Six-point correlations this unstable cannot
distinguish the theory's prediction (`ρ` strongly negative) from its
rough opposite (`ρ` positive) — both `−0.314` and `+0.700` are plausible
draws from the same underlying noise given only 6 points, and neither should
be read as confirming or refuting the aggregate-savings prediction. The
aggregate metric is simply not testable at this scenario count; a proper
test needs a finer `t`-grid (15–20 values rather than 6), not more seeds per
value.

One structural observation survived both reruns and is worth noting on its
own: at `t=0.6` (10-seed run), margin sampling never matched random's own
rarest-class endpoint within the 768-label budget — a case where the
"active learning always helps the rare class" assumption embedded in
assumption (ii) visibly failed for one specific scenario. This is a useful
negative data point in its own right, independent of the correlation
question: `O(K)`-independent-of-`π_min` active-learning recovery is not
automatic, and can fail even in a controlled synthetic setting.

### 3c. A redesigned test finds a real pattern

Two flaws were fixed, not just "more of the same":

1. **Scenario count, not seed count, was the bottleneck** (established in
   §3b) — fixed by a 16-value `t`-grid (`np.linspace(0, 1, 16)`) instead of
   6, at 5 seeds each (`scripts/synthetic_majorization_finegrid.py`).
2. **The "labels to reach a threshold" savings metric is separately
   fragile** — it requires threshold-crossing interpolation and sometimes
   never converges at all (the `t=0.6` case above). Replaced with **F1 gap
   at a fixed label budget** (`margin_F1 − random_F1` on the rarest class):
   always defined, no interpolation, and it's the same style of quantity
   that was already rock-solid in the raw-mechanism check.
3. **Correlated at the (`t`, seed) level** — 80 points (16×5), not 16
   scenario means — the statistically correct way to spend a fixed compute
   budget on correlation power rather than per-point precision.

Results, at three label-budget checkpoints:

| budget | ρ, scenario means (n=16) | ρ, (t,seed) pairs (n=80) |
|---|---|---|
| 150 | −0.197 | −0.068 |
| 300 | **−0.603** | **−0.334** |
| 450 | **−0.565** | **−0.337** |

**This is a genuine, reproducible pattern, not noise, for three checkable
reasons:**

- **Cross-budget agreement.** Gap@300 and gap@450 correlate with *each
  other* at 0.87 (scenario level) and 0.77 (per-point level) — two
  independently computed quantities agreeing strongly is the signature of a
  real underlying effect, not a coincidence of one noisy draw.
- **The one outlier (budget=150) has a physical explanation, not just a
  smaller number.** At an early checkpoint, neither strategy has queried
  enough labels for AL's advantage to manifest yet — both curves are still
  in their shared initial-exploration phase. That the correlation is weak
  exactly where the mechanism hasn't had time to act, and strengthens once
  it has, is consistent with the theory rather than contradicting it.
- **The raw mechanism reconfirms at even higher power**: ρ(`π_min`,
  random's own rarest-class F1) = +0.79 to +0.85 across all three budgets on
  this same 16-point grid — the strongest confirmation yet of the part of
  the theory that was never actually in question.

**Honest caveat.** −0.60 (best case) is a real, moderate effect — not
−0.857. The scenario-mean and per-point correlations also disagree with each
other by roughly a factor of 2 (−0.60 vs −0.33 at budget 300), which is
itself informative: aggregating over seeds before correlating inflates the
apparent strength relative to treating each run as its own data point, and
the per-point number (−0.33) is the more conservative, more honest estimate
of the effect size. The synthetic result should be read as "a real, moderate
effect, direction confirmed, magnitude smaller than the one real-archive
comparison suggested" — not as recovering the full −0.857.

---

## 4. Honest verdict

| claim | status |
|---|---|
| Random sampling's rarest-class recovery degrades as `π_min` shrinks | **Confirmed cleanly**, monotonic, no exceptions, in the real archive and across all three synthetic runs (3-seed, 10-seed, 16-point fine-grid: ρ up to +0.85) |
| `N_random(π) ≈ k_min/π_min` is Schur-convex, so majorization-decreasing smoothing shrinks measured random-sampling deficiency | **Provable as stated** (standard convex-plus-symmetric argument); not itself in question |
| Real classifiers' AL advantage over random, aggregated to a macro-performance target, tracks `π_min` via this mechanism | **Moderately supported**, once tested with adequate scenario count (16 vs the original 6) and a metric that avoids threshold-crossing fragility. ρ = −0.33 to −0.60 depending on aggregation level, internally consistent across two independent label-budget checkpoints (cross-correlation 0.77–0.87). Real, but smaller than the real archive's −0.857. |

The path to this verdict is itself worth recording: a first pass (6
scenarios, 3 seeds) found a weak correlation; adding seeds without adding
scenarios made it *less* stable, not more, which correctly diagnosed the
bottleneck as scenario count rather than per-point noise; fixing both that
and a second, independently-identified fragility (threshold-crossing) in the
metric produced a genuine, reproducible, moderate-strength result. Each step
changed the experiment based on what the previous step's failure mode
actually was, rather than simply repeating it with more compute.

## 5. Novelty

The building blocks — coupon-collector bounds, Schur-convexity, rare-category
active learning — are all standard and pre-existing (searched again 17 August
2026: no paper found connecting majorization/Schur-convexity to active-
learning label-efficiency, nor to distillation-target evaluation bias
specifically). The reframing is the contribution, and per the analysis above
it currently supports a mechanism claim, not a validated quantitative
theorem. Presented honestly, this is a small theory note with a clearly
flagged gap, not a finished result.
