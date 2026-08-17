# An independent assessment of the VarWISE catalog

Three linked pieces of work, all buildable from the published catalogs alone:

1. **A reliability audit of the Extended Catalog** — the published quality cut
   does not protect period validity.
2. **A learned replacement for the CV/SN transient rule** — 1.5% → 96.6%
   accuracy on the population the rule acts on.
3. **Closing the YSO recall gap** — 0.436 → 0.910 recall at matched precision.

Reproduce with:

```bash
python scripts/audit_extended.py       # -> results/extended_audit.txt
python scripts/fix_transient_rule.py   # -> results/transient_rule_fix.txt
python scripts/improve_yso.py          # -> results/yso_recall.txt
```

> **Read this first.** The VarWISE paper contains its own per-class visual
> inspection, and it is considerably more self-critical than most catalog
> papers. It already reports that only 9% of `sn` objects are solid candidates
> with "56% ... normal AGNs", and that only 20% of inspected `lpv` objects are
> truly periodic. Where the work below overlaps those statements, it confirms
> and quantifies them at scale rather than discovering them. Credit is noted
> throughout.

---

## 1. Extended Catalog reliability

The Extended Catalog holds 1,918,082 objects and contains the Pure tier
(cluster_id overlap = all 457,080). The Extended-only remainder is 1,461,002
objects.

### The published quality cut does not constrain period validity

The paper recommends `confidence >= 0.9`. For the `rr` class, whose physical
period range is 0.20–1.00 d:

| cut | n | physical period | P ≥ 2 d (unphysical) |
|---|---|---|---|
| no cuts | 443,991 | 23.2% | 46.5% |
| confidence ≥ 0.8 | 205,311 | 19.9% | 47.3% |
| **confidence ≥ 0.9** *(recommended)* | **124,026** | **19.8%** | **47.0%** |
| confidence ≥ 0.99 | 21,045 | 41.4% | 28.6% |
| confidence ≥ 0.9 AND `suspect_period` = 0 | 123,774 | 19.9% | 46.9% |
| **`period_significance` > 20** | **12,861** | **71.3%** | **0.1%** |
| confidence ≥ 0.9 AND `period_significance` > 20 | 11,351 | **72.4%** | 0.1% |

Pure `rr` for comparison (independently validated at F1 0.948): 69.7% physical
with no cuts at all.

**`confidence` measures certainty in the class, not the period.** Raising it
from 0 to 0.9 leaves the period distribution essentially unchanged — it even
dips slightly. `suspect_period` catches under 1% of the problem. Only
`period_significance` fixes the sample, and that cut is not among the
published recommendations.

### Per-class period plausibility at confidence ≥ 0.9

| class | physical range | n | physical | beyond 2× |
|---|---|---|---|---|
| `rr` | 0.2 – 1 d | 124,026 | **19.8%** | **47.0%** |
| `lpv` | 30 – 3000 d | 411,270 | 38.7% | 0.0% |
| `cep` | 1 – 100 d | 7,741 | 83.9% | 12.9% |
| `ew` | 0.15 – 1.5 d | 116,228 | 87.5% | 5.9% |
| `ea` | 0.3 – 1000 d | 55,976 | 96.6% | 0.0% |

Pure tier equivalents are markedly better (`rr` 70.2%, `cep` 95.2%,
`ew` 95.0%). `rr` is the outlier: it is the only class where the Extended
tier degrades catastrophically rather than gradually.

### Population sanity check

Gaia DR3's validated all-sky RR Lyrae catalogue contains **270,905** objects
(Clementini et al. 2023, A&A 674, A18). RR Lyrae are among the
best-inventoried variable classes in astronomy, and their amplitudes are
weakest in the mid-infrared.

| sample | n | × Gaia DR3 |
|---|---|---|
| Extended `rr`, no cuts | 443,991 | **1.64×** |
| Extended `rr`, confidence ≥ 0.9 | 124,026 | 0.46× |
| Extended `rr`, + physical period | 24,607 | 0.09× |
| Extended `rr`, + `period_significance` > 20 | 11,351 | 0.04× |

### The confidence cut silently deletes the rule-assigned classes

`cv` and `sn` carry no confidence value, so `confidence >= 0.9` drops **100%
of `sn`** and **89.6% of `cv`** — without the user being told why. Protective
by accident, but it means published class counts and any confidence-filtered
analysis describe different samples.

### Independent physical confirmation: period–luminosity

The argument above rests on *range plausibility* — RR Lyrae cannot have 5-day
periods. That is sound but weak, since it uses only the period's own value. A
period–luminosity test is independent and much stronger: RR Lyrae are
horizontal-branch stars sitting at **M_W1 ≈ −0.5**, nearly independent of
period, so an object claiming to be one that sits three magnitudes fainter is
not one, whatever its period looks like.

Using Gaia parallaxes (S/N > 5), M_W1 = W1 + 5·log₁₀(plx_mas) − 10
(`scripts/pl_relation_check.py`):

| sample | n | median M_W1 | offset from −0.5 |
|---|---|---|---|
| **Pure `rr`, all** | 6,671 | **−0.59** | −0.09 |
| Pure `rr`, confidence ≥ 0.9 | 6,191 | −0.62 | −0.12 |
| Pure `rr`, `period_significance` > 20 | 6,302 | −0.61 | −0.11 |
| **Extended `rr`, all** | 221,982 | **+2.51** | **+3.01** |
| **Extended `rr`, confidence ≥ 0.9** *(paper's cut)* | 71,378 | **+2.47** | **+2.97** |
| **Extended `rr`, `period_significance` > 20** | 7,624 | **−0.55** | **−0.05** |

Three results, all independent of the period-range argument:

1. **Pure `rr` sits exactly on the RR Lyrae locus**, confirming from physics
   that it is a clean sample — consistent with its measured F1 of 0.948.
2. **Extended `rr` sits ~3 magnitudes too faint.** Those objects are not RR
   Lyrae. The paper's recommended `confidence ≥ 0.9` barely moves it
   (+2.97).
3. **`period_significance > 20` returns the sample to the correct locus**
   (−0.55), recovering 7,624 objects that behave like genuine RR Lyrae.

PL-relation scatter tells the same story: Extended `rr` scatter is 1.063
with no cuts and 0.938 at confidence ≥ 0.9, but **0.467 at
`period_significance` > 20** — essentially matching Pure's 0.433.

**Negative result, reported as such:** the same test on LPVs is
uninformative. Scatter stays at 2.4–3.1 under every cut, and
`period_significance` does not help. This is expected — Miras, semiregulars
and overtone pulsators occupy different PL sequences, so a single fit cannot
be tight. The LPV periods are neither validated nor impugned by this test.

### Attempted extension to Cepheids — inconclusive, reported honestly

The RR Lyrae PL check was decisive enough to ask whether the same test
sharpens further on Cepheids, whose mid-IR PL relation is the tightest
standard-candle relation in astronomy (intrinsic scatter ~0.1–0.2 mag, versus
RR Lyrae's much looser relation). It does not — and the reason matters more
than the null result itself.

**Even the Pure-tier, high-confidence Cepheid fit shows 2.6 mag of scatter**,
an order of magnitude above what a genuine Cepheid PL relation should show.
That rules out reading this as "Cepheid periods are 26× worse than RR Lyrae
periods." It means the test itself doesn't work for this class as
constructed: Cepheids sit much farther away than RR Lyrae, so a parallax
S/N > 5 cut leaves real residual distance-precision problems (worse
Lutz-Kelker bias), and no extinction correction was applied — Cepheids trace
young, disk-concentrated populations that are more dust-affected than the
older, more spread-out RR Lyrae population.

The *relative* trend across cuts is at least directionally consistent with
the RR Lyrae story — comparing Extended `cep` against the Pure-tier PL fit,
RMS offset falls from 4.55 mag (no cuts) to 2.25 mag (`period_significance`
> 20), and the median offset flips from +4.2 to −0.5 — but given the
untrustworthy baseline, this is reported as **directionally suggestive, not
a second independent confirmation**. A proper version of this test would
need an extinction correction and a tighter parallax S/N cut before it could
carry independent weight.

### Follow-up: is the flagged `ecl` correction real, or a classifier default?

The transient-rule replacement (§2 below) predicts `ecl` for 21,646 objects,
of which 21,289 were flagged low-reliability (trained on only 344 examples,
CV F1 0.598) and named "the clearest target for follow-up." Resolved with an
independent check: eclipsing binaries are periodic by definition, so real
ones should show significant `period_significance` regardless of what the
classifier's own confidence says.

| group | n | median `period_significance` | % with `period_significance` > 20 | % with no period |
|---|---|---|---|---|
| Reference: VarWISE `ea`/`ew` (independently validated, F1 ≈ 0.99) | 119,713 | **60.4** | 94.5% | 0.0% |
| Predicted `ecl`, low reliability | 21,289 | **6.8** | 12.7% | **45.3%** |
| Predicted `ecl`, validated/high | 357 | 9.0 | 17.3% | 44.8% |

**Verdict: the hypothesis is not supported.** Median period significance is
less than a ninth of the genuine reference, and 45% have no usable period at
all — disqualifying for a class defined by periodicity. The model is most
likely defaulting to `ecl` as a residual/catch-all rather than detecting
real eclipsing signal. The `reliability=low` flag is doing its job; these
21,289 rows should be treated as unclassified, not as tentative eclipsing
binaries.

### Recommended cuts for Extended-tier users

1. For any period-dependent use, add **`period_significance > 20`**. This is
   now supported two ways: it is the only cut that restores physical period
   distributions, and it is the only cut that returns Extended `rr` to the
   correct absolute magnitude.
2. Treat `cv`/`sn` as a separate, rule-assigned product — not classifier output.
3. Apply a class-appropriate period sanity range before population statistics.

### Caveats

- **I could not exactly reproduce the Pure-tier selection** from published
  columns: my reconstruction yields 858,516 rows against the actual 457,080,
  so additional undocumented criteria must apply. The "full Pure-tier
  criteria" row above is therefore approximate.
- `period1` may be an alias or harmonic rather than a wrong detection; a 2×
  alias is excluded from the "beyond 2×" column for this reason.
- Infrared surveys can genuinely find dust-obscured RR Lyrae that optical
  surveys miss — but that does not explain 47% at P ≥ 2 d.
- This is a label-free audit against physical constraints, not against truth
  labels; the Extended tier has no populated `simbad_type`.

---

## 2. Replacing the CV/SN transient rule

VarWISE assigns `cv` and `sn` with a rule, not the classifier: VARnet flags a
transient, then a 2″ crossmatch against Gaia DR3 galaxy/QSO catalogs assigns
SN on a match and CV otherwise.

> **Correction (from the full-catalog scan).** An earlier version of this
> document treated `cv` as entirely rule-assigned. It is not. **`cv` is a mix
> of two mechanisms**, exactly separable by whether `confidence` and `period1`
> are null — there are **zero** mixed combinations:
>
> | class | total | rule-assigned | classifier-assigned |
> |---|---|---|---|
> | `cv` | 34,316 | **28,419 (82.8%)** | **5,897 (17.2%)** |
> | `sn` | 9,596 | 9,596 (100%) | 0 |
>
> The two sub-populations perform very differently against SIMBAD:
> rule-assigned `cv` precision **0.013**, classifier-assigned `cv` precision
> **0.144** — an 11× difference. The classifier-assigned portion is still
> poor, but it is not the same failure. The catalog does not label this
> distinction; users must derive it from the null pattern.

Of 14,955 rule-assigned transients carrying an independent SIMBAD type, what
they actually are:

| SIMBAD truth | n | share |
|---|---|---|
| `lpv` | 8,338 | 55.8% |
| `agn` | 3,600 | 24.1% |
| `yso` | 2,576 | 17.2% |
| `cv` | 226 | 1.5% |
| `ecl` | 160 | 1.1% |
| `sn` | 20 | 0.1% |

### Result

5-fold cross-validated LightGBM on the same catalog features:

| method | accuracy | macro F1 | `cv` F1 |
|---|---|---|---|
| rule as published | **0.0149** | 0.008 | 0.038 |
| classifier (with crossmatch flag) | **0.9655** | 0.851 | 0.921 |
| classifier (photometry only) | 0.9654 | 0.856 | 0.917 |

What it recovers from the rule's false positives:

| actually | n | rule correct | classifier correct | recovered |
|---|---|---|---|---|
| `lpv` | 8,338 | 0 | 8,104 | **+8,104** |
| `agn` | 3,600 | 0 | 3,563 | **+3,563** |
| `yso` | 2,576 | 0 | 2,455 | **+2,455** |
| `ecl` | 160 | 0 | 60 | +60 |
| `cv` | 226 | 222 | 204 | −18 |

**The extragalactic crossmatch is not the informative signal.** Photometry
alone matches the version that includes it (0.9654 vs 0.9655) — mid-IR colour
and amplitude separate these populations without needing the crossmatch the
rule is built on.

### Applied to the full catalog

`scripts/apply_transient_fix.py` trains on all 17,513 labelled rule-assigned
objects and applies the model to **all 79,293** rule-assigned transients
across both tiers (Pure: 43,912; Extended-only: 35,381), producing
`results/varwise_transient_corrections.csv` keyed on `cluster_id`.

**The generalization check passes on accuracy.** Every finding in this repo
rests on the bright, SIMBAD-covered subset, so accuracy was measured in W1
magnitude bins:

| W1 range | n | accuracy | macro F1 |
|---|---|---|---|
| 0–8 | 2,809 | 0.995 | 0.405 |
| 8–10 | 6,384 | 0.983 | 0.520 |
| 10–12 | 2,516 | 0.907 | ~0.58 |
| 12–14 | 3,581 | 0.934 | 0.592 |
| 14–30 | 2,223 | 0.949 | 0.627 |

Accuracy does **not** collapse toward the faint end (0.949 at W1 > 14), so
extrapolation is defensible. The labelled set spans W1 = 7.5–15.9; 22.3% of
unlabelled objects lie beyond the labelled 95th percentile.

**But macro F1 is low in every bin, and that is the honest measure.**
Aggregate accuracy is carried by the majority classes. Per-class
cross-validated F1:

| class | train n | CV F1 | verdict |
|---|---|---|---|
| `agn` | 3,945 | 0.981 | reliable |
| `lpv` | 9,626 | 0.978 | reliable |
| `yso` | 3,181 | 0.932 | reliable |
| `cv` | 343 | 0.915 | reliable |
| `ecl` | 344 | **0.598** | **unreliable** |
| `sn` | 36 | **0.316** | **unreliable** |
| `cep` | 38 | **0.049** | **unreliable** |

The table therefore ships a `reliability` column:

| tier | n | share |
|---|---|---|
| validated (independent label exists) | 17,539 | 22.1% |
| high (class F1 ≥ 0.85, prob ≥ 0.9) | 31,344 | 39.5% |
| medium | 2,708 | 3.4% |
| **low — do not use** | **27,702** | **34.9%** |

**The `ecl` result is a hypothesis, not a correction.** The model assigns
`ecl` to 21,646 objects — the largest single corrected class — but 21,289 of
those are flagged low reliability, because `ecl` was trained on only 344
examples and scores CV F1 0.598. Those objects may well be eclipsing binaries
the rule swept into `cv`, but this analysis cannot establish that. It is the
clearest target for follow-up.

Usable output: **61.6%** of rows (validated + high), covering the `lpv`,
`agn`, `yso` and `cv` corrections that constitute the bulk of the rule's
failures.

### Caveats

- The rule was never designed to emit `lpv`/`agn`/`yso`, so it cannot win on
  those. The useful reading is *"this population is recoverable"*, not
  *"the rule is bad at its own job"*.
- Only 343 genuine CVs and 36 SNe exist in the labelled population, so those
  columns rest on small numbers.
- Only 22.1% of rule-assigned objects carry a SIMBAD type, and those are
  systematically brighter (median W1 9.64 vs 13.63 for the rest).
- Predicted-class colour loci sit systematically bluer than the labelled
  reference loci (e.g. predicted `agn` W1−W2 0.730 vs labelled 0.959). This
  may reflect genuine differences in the faint population, or mild
  distribution drift; it is unresolved.

---

## 3. Closing the YSO recall gap

The Pure audit found VarWISE recovers only **43.6%** of SIMBAD YSOs
(precision 0.940, F1 0.595). The paper's own visual inspection reports 85% of
inspected `yso` objects are solid candidates — that is *precision*, and a
visual check of flagged objects structurally cannot measure recall. The
deficit is therefore not something their validation could have surfaced.

Binary YSO classifier, 5-fold CV, 28 catalog features, average precision
**0.973**:

| threshold | precision | recall | F1 | n flagged |
|---|---|---|---|---|
| 0.10 | 0.831 | 0.981 | 0.900 | 14,167 |
| 0.50 | 0.890 | 0.965 | 0.926 | 13,009 |
| 0.75 | 0.913 | 0.949 | **0.931** | 12,461 |
| 0.95 | 0.944 | 0.901 | 0.922 | 11,444 |

Matched comparisons:

- **At VarWISE's precision (0.940): recall 0.910 vs 0.436 — a 2.09× improvement.**
- At VarWISE's recall (0.436): precision 0.992 vs 0.940.
- Best F1: 0.931 vs 0.595.

### The residual failures are physical, not modelling

| group | n | W1−W2 | W1 | W1 amp |
|---|---|---|---|---|
| YSOs recovered | 11,582 | 0.680 | 11.91 | 0.232 |
| YSOs still missed | 416 | **0.334** | 11.40 | 0.205 |
| LPVs (the contaminant) | 64,136 | **0.047** | 8.62 | 0.110 |

The 416 still-missed YSOs sit at bluer W1−W2 than recovered ones, shifted
toward the LPV locus. That is genuine physical overlap between dusty YSOs and
AGB stars, not a shortfall in the model.

Most informative features: `bp_rp`, `n_obs`, `w1_w2`, `w3_w4`, `w2_w3`,
`period_significance`.

### Caveats

- **Not a like-for-like contest.** VarWISE trained on Gaia/ZTF labels and is
  evaluated here against SIMBAD; this classifier is trained *and* evaluated on
  SIMBAD, so it has the easier task by construction. The result shows the
  information needed for higher recall is present in the catalog columns — not
  that this model beats theirs on their own distribution.
- SIMBAD YSO labels are ~58% `YSO_Candidate`.
- Restricted to objects with a mapped SIMBAD type, which skew bright.

---

---

## 4. Full-catalog scan

`scripts/full_catalog_scan.py` sweeps every published column looking for what
targeted analysis would miss. **The catalog's basic construction is sound**,
and that deserves saying as plainly as the criticisms:

- **No duplicate `cluster_id`, `designation`, or coordinates** — 457,080
  unique on all three.
- **Every sky-distribution sanity check passes.** `lpv` median |b| = 1.9°
  (90.6% in the Galactic plane), `yso` 2.4°, `agn` 29.1° (48.6% at |b| > 30°).
  These are exactly right for AGB stars, star-forming regions, and
  extragalactic sources respectively.
- **`n_obs` rises toward the ecliptic poles** (median 558 at |dec| > 66° vs
  253 at |dec| < 30°), matching WISE's survey geometry.
- **No pile-up at the period search rails** — 3 objects at the 0.1 d lower
  bound and 39 at the 999 d upper bound out of 419,065.
- **No objects with fewer than 38 epochs.**

Independent corroboration of the transient-rule finding: **`sn` has median
|b| = 38.3° with 64% at |b| > 30°** — a high-latitude, extragalactic
distribution consistent with the population being AGN rather than Galactic
transients.

Three genuine flags:

| severity | finding |
|---|---|
| MED | **`cv` mixes two assignment mechanisms** (see correction above) |
| MED | **W1 amplitude and `variability_snr` are nearly uncorrelated** (Spearman ρ = **+0.099**). Two columns a user would reasonably treat as interchangeable measures of "how variable" rank objects very differently. |
| LOW | 7 non-positive photometric uncertainties; 13.0% of objects have `confidence` exactly 1.000 (probability saturation) |

**A flag I raised and then withdrew.** 21,818 objects have BP−RP > 5, which
looks like bad cross-matching. It is not: 86% are `lpv` and their median W1 is
8.39 versus 12.60 for the catalog overall. Miras genuinely reach BP−RP > 5.
These are real red giants.

**The structural scan was extended to the Extended Catalog** (via TAP
aggregates rather than a full download — 1.9M rows). It is equally clean:
1,918,082 unique `cluster_id` and `designation`, zero out-of-range
coordinates, zero negative amplitudes/SNR/periods, zero `n_obs` < 20. The
`cv`/`sn` mechanism split reproduces at this tier too, with a different
mixing ratio: `cv` is 73.9% rule-assigned / 26.1% classifier-assigned in
Extended versus 82.8%/17.2% in Pure; `sn` is 100% rule-assigned in both.

---

## 4. Concordance with an independent mid-IR variable catalog

The natural next check — comparing VarWISE against a second, independently
built mid-IR variability catalog — was attempted via the unTimely-derived
variability catalog (Yao et al., 8.26M sources), but that specific product
is **not yet publicly released** ("tables will be available online soon" per
the preprint). The underlying base unTimely photometric catalog is public
but has ~23.5 billion raw detections — re-deriving variability from it is
its own multi-week project, not a concordance check.

A working substitute exists: Kim, Son, Kim, Ho, Jeong, Lee & Yang 2026, ApJS
284:39, "A Catalog of Mid-infrared Variable Sources in the Ecliptic Poles" —
30,345 objects, independently detected and classified (via a ZTF-light-curve
deep-neural-network classifier, Healy et al. 2024 — an entirely separate
classification pipeline from both VarWISE and SIMBAD), covering 5°-radius
circles around the north and south ecliptic poles. Machine-readable tables
are directly downloadable from the journal.

### AGN classification is confirmed strong; eclipse/YSO reveal a new, regional failure mode

Cross-matched against VarWISE Pure at 2″ (5,267 matches; 148 with a class
mappable to VarWISE's taxonomy):

| ZTF-catalog class | n | VarWISE agrees | agreement |
|---|---|---|---|
| Q (QSO → `agn`) | 84 | 83 | **98.8%** |
| E (eclipse → `ecl`) | 54 | 8 | **14.8%** |
| Y (YSO → `yso`) | 10 | 0 | **0.0%** |

The AGN result independently reconfirms the SIMBAD-based finding. The
eclipse and YSO disagreement is new and, on inspection, not an artifact:

- **Match quality rules out mismatched sources.** Median separation for the
  disagreeing objects is 0.04–0.08″, tighter than the overall match
  distribution (median 0.05″) — these are the same physical objects.
- **VarWISE's confidence on these misclassifications is high**, not
  borderline: median 0.98 (`ecl`→`agn`) and 0.99 (`yso`→`agn`).
- **Both mismatch types dominantly land on the same wrong class**
  (`agn`) — 43/54 eclipse objects and 8/10 YSOs.

### A mechanistic explanation, not just a correlation

Two follow-up checks converge on a specific cause:

1. **Every single mismatch comes from the North Ecliptic Pole; zero from
   the South.**
2. **These objects have a median of 1,763 epochs — about 6.5× the typical
   VarWISE object (270).** The ecliptic poles receive near-continuous WISE
   coverage from the scanning geometry, unlike the rest of the sky, which is
   visited roughly twice a year.

The plausible mechanism: a classifier's light-curve features, built and
implicitly calibrated against the catalog's typical ~270-epoch cadence, may
read an object with 1,700+ densely-sampled epochs as AGN-like stochastic
variability rather than recognizing genuine periodic eclipses or YSO
behavior — a cadence-driven miscalibration specific to the
continuous-viewing-zone regions, distinct from every other failure mode
found so far (the CV/SN rule, the Extended `rr` contamination, the YSO
recall gap). The YSO mismatches show the same pattern even more strongly
(median 2,478 epochs, ~9× typical).

### Caveats

- **Small sample** (n=56 eclipse, n=12 YSO) and **single region** (NEP
  only) — this does not generalize to a global concordance rate, and should
  not be read as contradicting the much larger (n=111,241), SIMBAD-based
  finding that VarWISE's `ecl` class is excellent (F1 0.99) across the sky.
  The two results are compatible: `ecl` may be reliable generally and
  specifically miscalibrated in the unusually high-cadence polar regions.
- Only Q and Y map cleanly onto VarWISE's taxonomy; P/S/B (pulsating,
  generic "variable star," binary) are too coarse to force-map and are
  excluded rather than guessed.
- The ZTF classification is itself a DNN threshold call, not ground truth —
  disagreement could in principle originate on either side, though the
  n_obs/hemisphere pattern points toward a VarWISE-side cadence effect
  specifically.
- This substitutes for the unavailable unTimely comparison and is not a
  full-sky concordance check.

---

## Novelty assessment (re-checked 16 August 2026)

| Piece | Status | Nearest prior work |
|---|---|---|
| **Extended audit** | **Novel.** No independent validation of VarWISE exists. The method (period-plausibility diagnostics) is standard — Gaia's SOS Cep&RRL validation uses period vs amplitude/Fourier diagnostics, and Pan-STARRS RR Lyrae periods were validated against K2 ([arXiv:2408.14260](https://arxiv.org/abs/2408.14260)). Applying it here is new; the specific finding that `confidence` does not constrain period validity is new. | Gaia DR3 SOS validation ([A&A 674, A18](https://www.aanda.org/articles/aa/full_html/2023/06/aa43964-22/aa43964-22.html)) |
| **CV/SN rule replacement** | **Diagnosis partly anticipated** — the authors already state 56% of `sn` are "normal AGNs". The *fix* is new, as is the LPV→`cv` quantification and the finding that photometry alone suffices. | ML for CV discovery in ZTF alerts ([MNRAS 527, 8633](https://academic.oup.com/mnras/article/527/3/8633/7459939)) and Gaia Science Alerts ([MNRAS 517, 3362](https://academic.oup.com/mnras/article/517/3/3362/6747149)) — both optical, neither targeting VarWISE |
| **YSO recall** | **Novel finding, crowded method space.** The recall deficit is new (their visual inspection can only measure precision). But YSO/AGB separation using IR colours + time-domain ML was published Jan 2026 ([ApJ, ae25f2](https://iopscience.iop.org/article/10.3847/1538-4357/ae25f2)); a general YSO-classification contribution would be contested. Frame this as closing a catalog-specific gap, not as a new YSO method. | [ApJ ae25f2](https://iopscience.iop.org/article/10.3847/1538-4357/ae25f2) (Jan 2026); [ApJS ad5a08](https://iopscience.iop.org/article/10.3847/1538-4365/ad5a08) |

**Overall:** the Extended audit is the strongest and least contested piece.
The transient-rule replacement is the most *useful*, since it converts a
documented weakness into a working product. The YSO work is real but should
be framed narrowly — as a catalog gap closed, not a new method — because that
space acquired direct competition in January 2026.
