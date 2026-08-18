# An Independent Assessment of the VarWISE Catalog

**Draft — not submitted.** Prepared for author feedback prior to any
submission decision. Candidate venues: PASP or AJ short note (as drafted,
~3,800 words); could be trimmed to Research Notes of the AAS (RNAAS, 1,000
word limit) by keeping only §3–4 and the corrected-label release.

---

## Abstract

VarWISE (Paz et al. 2026, ApJS 284, 41) is a recently published catalog of
457,080 (Pure) and 1,918,082 (Extended) infrared-variable objects drawn from
the NEOWISE single-exposure archive, classified into nine variability types
by a combination of an XGBoost classifier and a rule-based transient
assignment. We present the first independent validation of this catalog,
using literature cross-matches (SIMBAD) and a second, independently
classified catalog covering the ecliptic poles. We find the XGBoost
classifier validates well against labels it was not trained on (macro
F1 = 0.879 across six classes), with young stellar object (YSO) recall
(0.436) as its principal weakness. We show that the catalog's `cv` and `sn`
labels are produced by a separate rule, not the classifier, and that this
rule over-predicts by 38.5× and 96.5× respectively when checked against
independent labels, driven by long-period variables and active galactic
nuclei (AGN) that satisfy the rule's Local-Group/extragalactic transient
criteria without being genuine cataclysmic variables or supernovae. We
demonstrate that the Extended Catalog's `rr` (RR Lyrae) class is severely
contaminated — a raw count 1.64× the entire validated Gaia DR3 RR Lyrae
population, confirmed independently via a period–luminosity test that finds
these objects 3.0 mag offset from the expected locus — and that VarWISE's
own recommended quality cut (`confidence` ≥ 0.9) does not address this,
while a cut on `period_significance` does. A cross-match against an
independently classified mid-infrared variable catalog at the ecliptic
poles reveals a further, previously undocumented failure mode: eclipsing
binaries and YSOs are misclassified as AGN specifically in the North
Ecliptic Pole region, where continuous WISE viewing produces 6–9× the
catalog's typical epoch count. A Gaia parallax test — a physically
independent check unrelated to photometry or cross-classification —
corroborates this: while VarWISE's `agn` class is astrometrically clean in
aggregate, the small subset where SIMBAD positively disagrees shows a
100–150× excess of significant parallax detections, concentrated among
young stellar and pre-main-sequence SIMBAD types. We release a value-added,
reliability-tiered correction table for all 79,293 `cv`/`sn`-classified
objects across both catalog tiers, recovering the correct class for the
majority of the XGBoost-confounded population at 96.6% cross-validated
accuracy.

---

## 1. Introduction

VarWISE (Paz et al. 2026) is a substantial and welcome addition to the
infrared time-domain literature: a decade of NEOWISE single-exposure
photometry, processed with a purpose-built variability detector (VARnet)
and classified into nine astrophysical types. As with any large,
machine-learning-classified catalog, its labels carry two distinct sources
of uncertainty — the training data's own limitations, and the classifier's
generalization to the full survey — neither of which the discovery paper
can fully characterize using only its own held-out validation split. VarWISE
reports a macro-averaged F1 of 0.95 on a validation set drawn from the same
Gaia (Rimoldini et al. 2023) and ZTF (Chen et al. 2020) sources used for
training; this is the appropriate in-sample check, but it cannot detect
failure modes that correlate with how the training set was assembled, nor
can it substitute for validation against genuinely independent labels.

We conduct such an independent assessment here, motivated by three
questions a prospective user of the catalog would reasonably ask: (1) does
the classifier's reported performance hold up against labels it never saw;
(2) are the catalog's lower-quality "Extended" tier and its two
rule-rather-than-classifier-assigned classes (`cv`, `sn`) as reliable as the
headline statistics suggest; and (3) do these classifications agree with
an entirely separate classification effort applied to overlapping data. We
find informative, and in places unexpected, answers to all three.

Section 2 describes the data and cross-match methodology. Section 3
validates the XGBoost classifier. Section 4 diagnoses the `cv`/`sn`
transient-assignment rule. Section 5 assesses the Extended Catalog's
reliability, centered on a period–luminosity test of the `rr` class.
Section 6 presents an independent concordance check against a second
mid-infrared variability catalog. Section 7 describes a corrected-label
data product we are releasing alongside this note. Section 8 discusses
implications and gives specific, actionable recommendations for catalog
users.

All quantitative claims in this note were independently re-derived from
source and machine-verified prior to submission (213 automated consistency
checks; see Data Availability).

## 2. Data and Methods

### 2.1 VarWISE

We use the VarWISE Pure Catalog (457,080 objects, the highest-confidence
tier) and, where noted, the Extended Catalog (1,918,082 objects),
downloaded via the IRSA TAP service (tables `varwisepure`, `varwiseext`;
DOI 10.26131/IRSA656). The downloaded Pure Catalog was verified against a
live re-query: fresh row counts match exactly, a 60-object stratified
sample matches the live source cell-for-cell across 36 columns (2,160
comparisons, zero mismatches), and all internal declination/right-ascension
tiling boundaries used during download show no duplication or omission.

### 2.2 Independent labels

**SIMBAD.** 229,365 Pure Catalog objects (50.2%) carry an independent
`simbad_type` cross-match, populated by VarWISE itself but not used in its
training. We map SIMBAD types onto VarWISE's nine-class taxonomy where an
unambiguous correspondence exists (e.g., `RRLyrae` → `rr`,
`ClassicalCep`/`Type2Cep` → `cep`), merging VarWISE's `ea`/`ew` distinction
into a single `ecl` class, since SIMBAD's `EclBin` does not separate them.
Ambiguous SIMBAD types (`C*`, `SB*`, `Variable*`, and similar) are dropped
rather than force-mapped.

**Ecliptic-poles catalog.** For an entirely independent classification
check (Section 6), we cross-match against Kim, Son, Kim, Ho, Jeong, Lee &
Yang (2026, ApJS 284, 39), a catalog of 30,345 mid-infrared variable
sources at the north and south ecliptic poles, classified via a deep neural
network trained on ZTF light curves (Healy et al. 2024) — a classification
pipeline sharing no methodology, training data, or authorship with VarWISE.

### 2.3 Statistical methods

Period–luminosity tests (Section 5) use Gaia DR3 parallaxes with
signal-to-noise > 5, absolute magnitude
$M_{W1} = W1 + 5\log_{10}(\varpi_{\rm mas}) - 10$, and an iteratively
sigma-clipped linear fit (3σ, 5 iterations) to characterize scatter about
the fitted relation. Cross-matches use a 2″ radius throughout, matching
VarWISE's own stated convention. Classifier comparisons use 5-fold
cross-validated LightGBM (Section 4, 7) or XGBoost (Section 3), with
class-balanced weighting where classes are imbalanced.

## 3. Validation of the XGBoost Classifier

VarWISE's `vartype` column is populated by two distinct mechanisms,
distinguishable in the published data by whether the `confidence` column is
populated: the six classes `agn`, `cep`, `ea`/`ew`, `lpv`, `rr`, and `yso`
come from the XGBoost classifier (confidence always populated); `cv` and
`sn` come from a separate rule (confidence never populated for these
classes; see Section 4). Conflating the two — as an earlier version of our
own analysis initially did — understates the classifier's real performance
by attributing the rule's failures to it.

Restricting to the 205,374 Pure Catalog objects where both the SIMBAD truth
and the VarWISE prediction fall among the six classifier-only classes, we
find:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `ecl` | 0.996 | 0.985 | 0.990 | 111,081 |
| `lpv` | 0.942 | 0.989 | 0.965 | 55,794 |
| `rr` | 0.934 | 0.963 | 0.948 | 12,095 |
| `agn` | 0.844 | 0.995 | 0.913 | 15,017 |
| `cep` | 0.803 | 0.935 | 0.864 | 1,996 |
| `yso` | 0.940 | 0.436 | 0.595 | 9,391 |
| **Macro average** | **0.910** | **0.884** | **0.879** | 205,374 |

A macro F1 of 0.879 against labels the classifier never saw, compared with
0.95 on its own Gaia/ZTF validation split, represents ordinary
out-of-sample degradation, not a failure of the classifier. Five of six
classes achieve F1 ≥ 0.86.

**YSO recall is the one substantive weakness.** At 0.436, the classifier
recovers fewer than half of independently confirmed YSOs, while its
precision on the class it does flag remains high (0.940) — when the
classifier calls an object a YSO, it is very likely correct; it simply
misses the majority. We verified that this deficit is recoverable from
information already present in the published catalog columns: a
cross-validated binary classifier trained on the same 28 catalog-derived
features (colors, amplitude, periodicity) achieves 0.910 recall at matched
0.940 precision — a 2.1× improvement — with the best-F1 operating point
reaching F1 = 0.931 versus VarWISE's 0.595. We emphasize this is not a
like-for-like methodological comparison (our classifier is trained and
evaluated on the same SIMBAD label source, whereas VarWISE trains on
Gaia/ZTF and is evaluated here against SIMBAD), so it demonstrates that the
recall is recoverable in principle, not that our specific model would
outperform VarWISE's on its own training distribution. We nonetheless flag
it as evidence that a straightforward recall-oriented adjustment (e.g., a
lower per-class decision threshold for `yso`) is likely available to the
catalog maintainers at low cost.

## 4. The Transient Assignment Rule (`cv`, `sn`)

VarWISE's own paper states its approach to cataclysmic variables (CVs) and
supernovae (SNe) plainly:

> "Given that WISE observes very few stellar phenomena outside our own
> Local Group, if we can identify a transient event with a known galaxy,
> we can sensibly assign it the class of SN. Conversely, if we find that a
> transient event lies within our Local Group, it is likely to be some
> sort of CV-related event."

Concretely: objects VARnet flags as transient are cross-matched against
Gaia DR3 galaxy/QSO catalogs within 2″; a match yields `sn`, no match
yields `cv`. This is a rule, not a trained classifier, and — notably — it
is applied inconsistently within the `cv` label itself: 82.8% of `cv`
classifications (28,419 of 34,316 in the Pure Catalog) are rule-assigned
(confidence and period both null), while the remaining 17.2% (5,897) carry
a populated confidence and period, indicating classifier involvement we
were unable to fully reconcile with the paper's stated methodology. `sn` is
100% rule-assigned.

### 4.1 Validation against SIMBAD

Restricting to the two rule-populated classes and their SIMBAD-covered
subset:

| Class | SIMBAD n | VarWISE n | Over-prediction | Precision |
|---|---|---|---|---|
| `cv` | 301 | 11,576 | 38.5× | 0.019 |
| `sn` | 35 | 3,379 | 96.5× | 0.002 |

The false positives are systematic and traceable directly to the rule's
own logic: 8,291 SIMBAD long-period variables (LPVs) are assigned `cv`
(bright, slowly evolving Galactic giants that VARnet flags as transient but
which are not extragalactic), and 3,275 SIMBAD AGN are assigned `sn`
(genuinely extragalactic and variable, matching the rule's SN criterion).
Photometrically, the LPV-contaminated `cv` population is ≈6 mag brighter
and has ≈4× lower W1 amplitude than the 222 genuine SIMBAD-confirmed `cv`
objects recovered by the same rule — these are ordinary red giants, not
borderline cataclysmic variables. We note VarWISE's own visual inspection
of a separate sample independently reports a compatible finding for `sn`
(only 9% of 212 inspected objects judged solid candidates, with 56%
resembling normal AGN); our result extends this to a quantified,
literature-cross-matched estimate and to the `cv` class, which the original
inspection did not separately flag at this severity.

Selection bias affects the exact precision values above: only 37.3% of
`cv`-classified objects carry a SIMBAD type, and those that do are on
average 3.8 mag brighter than those that do not (real CVs are intrinsically
faint; the LPV contaminants are bright). The precision figures above should
therefore be read as characterizing the bright, spectroscopically- or
literature-confirmed end of the population, not as catalog-wide rates.

### 4.2 A learned replacement recovers the population

Given the population the rule acts on is, in the large majority,
misassigned, we ask whether it is recoverable. Training a class-balanced,
5-fold cross-validated LightGBM classifier on the 14,955 rule-assigned
objects carrying a SIMBAD label, using the same 28 catalog features (with
and without the extragalactic cross-match flag the rule itself is built
on):

| Method | Accuracy | Macro F1 |
|---|---|---|
| Rule as published | 0.015 | 0.008 |
| Learned classifier (28 features + cross-match flag) | 0.966 | 0.851 |
| Learned classifier (photometry only) | 0.965 | 0.856 |

The classifier recovers 8,104 of 8,338 true LPVs, 3,563 of 3,600 true AGN,
and 2,455 of 2,576 true YSOs from the rule's false positives. Notably, the
extragalactic cross-match flag the rule is built around contributes
essentially nothing beyond what ordinary mid-infrared photometry already
provides (0.965 vs 0.966 accuracy) — the informative signal is color and
amplitude, not the crossmatch.

## 5. Extended Catalog Reliability

### 5.1 A population-level implausibility

The Extended Catalog assigns `rr` (RR Lyrae) to 443,991 objects — 1.64×
the entire validated Gaia DR3 all-sky RR Lyrae population (270,905;
Clementini et al. 2023). RR Lyrae are among the best-characterized variable
classes in the literature, and mid-infrared amplitudes are their weakest
across any commonly used band, making this raw excess implausible on its
face.

### 5.2 Period-range and period–luminosity confirmation

Real RR Lyrae periods are confined to 0.2–1.0 d. In the Extended Catalog,
only 19.8–23.2% of `rr` objects fall in this range depending on the exact
quality cut applied, with 45–47% at periods beyond 2 d — physically
impossible for this class. Critically, **VarWISE's own recommended quality
cut, `confidence` ≥ 0.9, does not address this**: the physical-period
fraction is 19.8% at this cut, essentially unchanged from the uncut
sample's 23.2%.

An independent, physically motivated confirmation follows from the
period–luminosity relation: RR Lyrae are horizontal-branch stars at
$M_{W1} \approx -0.5$, nearly independent of period.

| Sample | n | Median $M_{W1}$ | Offset |
|---|---|---|---|
| Pure `rr`, all | 6,671 | −0.59 | −0.09 |
| Extended `rr`, all | 221,982 | +2.51 | **+3.01** |
| Extended `rr`, confidence ≥ 0.9 | 71,378 | +2.47 | **+2.97** |
| Extended `rr`, `period_significance` > 20 | 7,624 | −0.55 | −0.05 |

Pure-tier `rr` sits exactly on the expected locus, independently confirming
its reliability (consistent with its F1 = 0.948 in Section 3). Extended-tier
`rr` sits three magnitudes off — objects that are not RR Lyrae, by any
reasonable interpretation — and the recommended confidence cut leaves this
essentially unchanged. A cut on `period_significance` > 20, not among the
paper's recommendations, restores both the period-range fraction (72.4%
physical) and the absolute magnitude (−0.55, matching the expected −0.5)
to values consistent with a genuine RR Lyrae population.

We attempted the analogous period–luminosity test on Cepheids, whose
relation has substantially tighter intrinsic scatter (~0.1–0.2 mag) than
RR Lyrae's, in the hope of a sharper diagnostic. The test proved
inconclusive: even the Pure-tier, high-confidence subsample shows 2.6 mag
of scatter about the fitted relation — an order of magnitude above the
expected intrinsic scatter — most plausibly attributable to uncorrected
interstellar extinction and residual Lutz–Kelker bias at the larger typical
distances of Cepheids relative to RR Lyrae, rather than to period
unreliability. We report this as inconclusive rather than as a second
confirmation.

### 5.3 Recommendation

For any period-dependent use of the Extended Catalog, we recommend a cut on
`period_significance` (> 20 is sufficient for `rr`) in addition to, not
instead of, the published `confidence` recommendation. We further note
that `confidence` ≥ 0.9 removes 100% of `sn` and 89.6% of `cv`
classifications without documentation, since these carry no confidence
value (Section 4) — users applying this cut should be aware they are also,
silently, filtering by classification mechanism.

## 6. An Independent Concordance Check

Sections 3–5 validate VarWISE against literature cross-matches drawn from
the same general body of prior knowledge (SIMBAD, Gaia) that informed the
catalog's own training set to varying degrees. As a check less susceptible
to shared systematic biases, we cross-match against Kim et al. (2026), an
independently detected and classified mid-infrared variable catalog
covering 5°-radius fields at the north and south ecliptic poles (NEP, SEP),
using a wholly separate classification pipeline (a ZTF-light-curve deep
neural network; Healy et al. 2024).

At a 2″ match radius, 5,267 of 30,345 objects (17.4%) match the VarWISE
Pure Catalog. Restricting to the subset of ZTF-catalog classes with an
unambiguous VarWISE-taxonomy correspondence:

| ZTF class | n | VarWISE agrees | Agreement |
|---|---|---|---|
| QSO → `agn` | 84 | 83 | 98.8% |
| Eclipse → `ecl` | 54 | 8 | 14.8% |
| YSO → `yso` | 10 | 0 | 0.0% |

The AGN result independently reconfirms Section 3. The eclipse and YSO
results are unexpected given `ecl`'s excellent sky-wide SIMBAD-validated
reliability (F1 = 0.990, Section 3) and warrant scrutiny before being taken
at face value.

We rule out a matching artifact directly: median positional separation for
the disagreeing objects (0.04–0.08″) is tighter than the overall match
distribution (0.05″), and VarWISE's classification confidence on these
specific misclassifications is high (median 0.98–0.99), not borderline.
Both mismatch types disproportionately land on the same incorrect class,
`agn` (43 of 54 eclipse objects; 8 of 10 YSOs).

Two further checks converge on a specific, plausible mechanism. First,
**every one of the 51 combined mismatches originates in the North Ecliptic
Pole**; none originate in the South. Second, these specific objects carry a
median of 1,763–2,478 epochs — 6.5–9.2× the catalog-wide median of 270.
The ecliptic poles receive near-continuous WISE coverage from the survey's
scanning geometry, unlike the rest of the sky, which is revisited roughly
twice yearly. We hypothesize that classifier features calibrated against
the catalog's typical ~270-epoch cadence may read an object sampled at
1,700+ epochs as exhibiting AGN-like stochastic variability rather than
recognizing genuine periodic eclipsing or YSO behavior — a cadence-driven
miscalibration specific to continuous-viewing-zone regions, distinct from
both failure modes described above.

We emphasize the modest scale of this result (n = 54 and n = 10) and its
restriction to a single, atypical sky region; it should not be read as
contradicting the much larger, sky-wide finding that `ecl` is VarWISE's
most reliable class. The two are compatible: a class can be reliable in
general and specifically miscalibrated in an observationally unusual
region.

### 6.1 An astrometric cross-check

As a further, physically independent test unrelated to photometry,
periods, or classification methodology, we examine Gaia parallax for the
`agn` class. Genuine AGN are at cosmological distances and should show a
measured parallax consistent with zero; we quantify this via the fraction
of each class exceeding a parallax significance threshold
($\varpi/\sigma_\varpi > k$), first validating the test on Galactic classes
(which show high significant-parallax rates, 17.5–95.7% at $k=3$, as
expected) before applying it to `agn`.

In aggregate, `agn` is astrometrically well-behaved: 2.4% exceed $k=3$
(median S/N $-0.07$), modestly above the idealized Gaussian expectation
(0.135%) but consistent with ordinary astrometric systematics near
crowded/blended fields, and far below any Galactic class — a result we
report as reassuring for the classifier's aggregate behavior. However, this
rate is not uniform: objects independently confirmed as AGN by SIMBAD show
a significant-parallax rate of 0.1% ($k=5$), while objects where SIMBAD
positively identifies something *other* than AGN show 15.0% — a
100–150$\times$ excess. The SIMBAD types among these astrometrically
confirmed contaminants (310 objects at $k>10$, median VarWISE confidence
0.971) are dominated by young stellar and pre-main-sequence classifications
(`OrionV*`, `YSO`, `TTauri*`), providing a third, methodologically
independent line of evidence for the YSO/`agn` confusion described in
Sections 3 and 6, now confirmed via distance rather than color, period, or
cross-catalog classification agreement.

## 7. A Corrected-Label Data Product

Motivated by Section 4.2, we release a value-added table
(`varwise_transient_corrections.csv`) providing corrected classifications
for all 79,293 rule-assigned `cv`/`sn` objects across both catalog tiers
(43,912 Pure, 35,381 Extended-only), keyed on the catalog's `cluster_id`.

Before applying the model trained on the SIMBAD-covered (and therefore
systematically brighter) subset to the full, mostly fainter population, we
verified generalization directly: cross-validated accuracy by W1 magnitude
bin remains high and does not degrade toward the faint end (0.995 at
W1 < 8 down to 0.949 at W1 > 14). Per-class reliability, however, varies
sharply with training-sample size (`agn`, `lpv`, `yso`, `cv`: cross-validated
F1 ≥ 0.85 on hundreds to thousands of examples; `ecl`, `sn`, `cep`: F1
0.05–0.60 on 36–344 examples), and the released table carries an explicit
`reliability` column (`validated` / `high` / `medium` / `low`) rather than
presenting all corrections as equally trustworthy. 61.6% of rows fall in
the `validated` or `high` tiers, covering the bulk of the LPV/AGN/YSO
recoveries; 34.9% are flagged `low` and should not be used for population
statistics without further verification. In particular, the model assigns
`ecl` to 21,646 objects — its largest single corrected class — of which
21,289 are `low` reliability; a direct follow-up check found these objects'
median period significance (6.8) far below a genuine eclipsing-binary
reference population (60.4), with 45.3% carrying no usable period at all,
indicating this specific correction is not supported and should be treated
as unclassified rather than as a tentative eclipsing binary.

## 8. Discussion and Recommendations

VarWISE's headline classifier performs close to its own reported standard
when checked against genuinely independent labels, with young stellar
object recall as its clearest area for improvement. The catalog's weaker
points are concentrated and specific rather than diffuse: a
rule-not-classifier transient assignment that fails in a direction
predictable from its own stated logic, an Extended-tier reliability gap
that the paper's own recommended quality cut does not close, and a
cadence-correlated regional miscalibration visible only under
cross-catalog comparison. We offer the following concrete recommendations
to catalog users:

1. Treat `cv` and `sn` as products of a rule, not the XGBoost classifier;
   consider using the corrected-label table released alongside this note,
   respecting its reliability tiering.
2. For any period-dependent Extended-tier analysis, add a cut on
   `period_significance` in addition to the published `confidence`
   recommendation.
3. Exercise added caution with `ecl`/`yso` classifications in the immediate
   vicinity of the ecliptic poles, where anomalously high epoch counts may
   correlate with degraded classification reliability.
4. YSO recall can likely be improved at low engineering cost via a
   lower per-class decision threshold, given the necessary discriminating
   information is already present in existing feature columns.

We stress throughout that this is an assessment of specific, addressable
failure modes in an otherwise substantial and useful catalog, not a
wholesale critique. We are sharing these findings with the VarWISE authors
directly and would welcome their assessment before any formal submission.

## Data Availability

All code, intermediate data, and the corrected-label table are available at
[github.com/rajul-kk/varwise-label-efficiency](https://github.com/rajul-kk/varwise-label-efficiency).
Every quantitative claim in this note is independently re-derived from
source data and machine-verified (`scripts/factcheck.py`; 213 checks, 0
failures at the time of writing) rather than transcribed by hand. This
research made use of the NASA/IPAC Infrared Science Archive, funded by NASA
and operated by Caltech; of data from the European Space Agency mission
Gaia, processed by the Gaia Data Processing and Analysis Consortium; and of
the SIMBAD database, operated at CDS, Strasbourg.

## References (informal, to be converted to full bibliography before submission)

- Chen, X., et al. 2020, ApJS, 249, 18 (ZTF Source Classification Project)
- Clementini, G., et al. 2023, A&A, 674, A18 (Gaia DR3 RR Lyrae)
- Healy, B. F., et al. 2024, ApJS, 272, 14 (ZTF DNN variable classifier)
- Kim, M., Son, S., Kim, S., Ho, L. C., Jeong, W.-S., Lee, B., & Yang, Y.
  2026, ApJS, 284, 39 (Ecliptic-poles mid-IR variable catalog)
- Paz, M., Kirkpatrick, J. D., Uttamchandani, R., Raen, T., & Cutri, R. M.
  2026, ApJS, 284, 41 (VarWISE)
- Rimoldini, L., et al. 2023, A&A, 674, A14 (Gaia DR3 variability)

---

### Draft notes (remove before submission)

- Word count: ~3,800. Fits PASP/AJ note format comfortably; would need
  cutting to ~1,000 words for RNAAS (drop Sections 3, 6, and most of 7's
  discussion; RNAAS also does not permit an abstract with citations, so
  those would need paraphrasing).
- Author list and acknowledgments section intentionally left open pending
  outreach to the VarWISE authors (see `scratchpad/email_to_varwise_authors.md`
  in a prior session) — this note explicitly proposes sharing findings with
  them before submission, and their response may materially change what
  gets submitted, if anything.
- Figures not yet selected for this draft; `results/curve_overall_track_b_xgb.png`-style
  plots are not relevant here (that's the AL study), but the repo's
  `results/curve_overall_track_a.png` is likewise not applicable. New
  figures would be needed: (a) the RR Lyrae PL diagram (Pure vs Extended,
  with cuts), (b) the confusion matrix from Section 3, (c) possibly a sky
  map showing the NEP/SEP mismatch concentration from Section 6.
- All numbers in this draft trace to `CATALOG_ASSESSMENT.md`,
  `NOVELTY.md`, and `RESULTS.md` §6 in this repository, all of which are
  covered by the 213-check verification suite.
