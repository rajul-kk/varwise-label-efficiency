# Search audit and recommended next directions

A survey of what is already claimed in the NEOWISE mid-infrared variability
space (as of **16 August 2026**), what remains open, and what is actually
buildable with the data currently reachable.

---

## 1. Competitive landscape — what is already taken

The mid-IR variability space became crowded in the last ~12 months. Three
full-scale variability catalogs now exist from essentially the same WISE
imaging:

| Work | Date | Scope | Method |
|---|---|---|---|
| **unTimely mid-IR variables**, [arXiv:2511.22071](https://arxiv.org/abs/2511.22071) | Nov 2025 | **8,256,042** W1 + 7,147,661 W2 variables, full sky | Bayesian Gaussian mixture / Dirichlet process on unWISE coadds, plus dedicated outlier detection |
| **Mid-IR Variables in the Ecliptic Poles**, [arXiv:2604.05332](https://arxiv.org/abs/2604.05332), ApJS | Apr 2026 | 2,764 NEP + 27,581 SEP | Variability probability + W1/W2 correlation; Gaia proper motion + color-color classification |
| **VarWISE**, [arXiv:2605.19059](https://arxiv.org/abs/2605.19059), ApJS 284:41 | May 2026 | 457,080 Pure / 1,918,082 Extended, **classified** | VARnet detection + XGBoost classification |

**Implication.** "Detect mid-IR variables" is thoroughly claimed. VarWISE's
distinguishing feature is that it *classifies* and assigns periods — so
anything new should build on the classification layer, not the detection
layer.

### Directions that are closed

| Direction | Why it is closed |
|---|---|
| **Anomaly / novelty detection on WISE light curves** | unTimely already runs outlier detection and reports eruptive YSOs, extreme AGN, rare transients. Also: OCSVM novelty detection in AllWISE ([arXiv:1706.06389](https://arxiv.org/abs/1706.06389)) with spectroscopic follow-up ([arXiv:2008.10658](https://arxiv.org/abs/2008.10658)), and Fink's anomaly pipeline ([arXiv:2603.29511](https://arxiv.org/abs/2603.29511)) which found AM CVn systems, UX Ori stars, 33 SNe and 9 dwarf novae. **Do not pursue.** |
| **Active learning for variable-star classification, as a general claim** | Richards et al. 2011 ([arXiv:1106.2832](https://arxiv.org/abs/1106.2832)); El-Kholy & Hayman 2026 ([arXiv:2602.23666](https://arxiv.org/abs/2602.23666)). Already covered by this repo's existing framing. |
| **Gaia DR4 cross-match** | DR4 is expected **December 2026**. Not available. |
| **LSST / Rubin cross-match** | DP1 (Jun 2025) and early DP2 (Jul 2026) are restricted to data-rights holders; alerts stream to brokers only. Public release follows a 2-year proprietary period. **Not accessible.** |

---

## 2. Data availability constraints (checked directly against IRSA)

| Resource | Status |
|---|---|
| `varwisepure` (457,080 rows) | ✅ On IRSA TAP, fully queryable, downloaded |
| `varwiseext` (1,918,082 rows) | ✅ On IRSA TAP, fully queryable |
| **Associations table** (light curves) | ❌ **Not on TAP.** Only `varwisepure` and `varwiseext` are exposed. Light curves must come from the IRSA DOI bulk download, or be rebuilt by positional cross-match against `neowiser_p1bs_psd` (heavy). |
| `simbad_type` in Extended Catalog | ❌ **Empty — 0 populated rows.** Independent labels for the Extended tier require running your own SIMBAD cross-match. |
| Parallax S/N > 5 in Pure | 143,234 objects |
| LPVs with both period and good parallax | 13,035 objects |

---

## 3. Recommendations, ranked

### 🥇 Tier 1 — strongest, and buildable today

#### R1. Audit the Extended Catalog. The evidence is already in hand.

The Pure Catalog audit found the classifier sound (macro-F1 0.879) and the
CV/SN rule broken. The **Extended Catalog looks far worse**, and one class in
particular fails a basic plausibility check:

**VarWISE Extended assigns `rr` (RR Lyrae) to 443,991 objects.** Gaia DR3's
validated all-sky RR Lyrae catalogue contains **270,905**. VarWISE claims
**1.6× the entire known population** — from an infrared band where RR Lyrae
amplitudes are at their weakest.

Period distribution confirms the problem. Real RR Lyrae are confined to
~0.2–1.0 d:

| Period range | Pure `rr` | Extended `rr` |
|---|---|---|
| 0.20 – 1.00 d (**physical**) | **69.7%** | **23.2%** |
| 1.05 – 2.00 d (2× alias) | 25.9% | 25.6% |
| **2 – 10 d (impossible)** | 0.5% | **41.8%** |
| 10 – 100 d | 0.0% | 4.2% |
| > 100 d | 0.02% | 0.5% |

Supporting evidence: Extended `rr` has mean confidence 0.756 (Pure: 0.982),
mean period significance 7.9 (Pure: 42.5), mean period 5.40 d (Pure: 0.86 d).

**Why this is a good project:** requires only TAP queries; the headline check
is a period cut anyone can verify; it extends work already done; and it is
directly useful — the Extended Catalog is the tier most likely to be used
naively for population studies. Deliverable: a per-class reliability
assessment of the Extended tier plus recommended quality cuts.

- **Novelty:** high — no independent validation of VarWISE exists at all.
- **Effort:** low (days).
- **Risk:** low. Caveat to handle: `period1` may be an alias/harmonic, and IR
  can genuinely find dust-obscured RR Lyrae that optical surveys miss — but
  neither explains 42% at 2–10 d, nor 1.6× the known population.

#### R2. Replace the CV/SN transient rule with a learned classifier.

The audit showed the rule over-predicts `cv` by 38.5× and `sn` by 96.5×,
failing exactly as its definition predicts (bright Miras → CV, AGN → SN). A
classifier trained on color, amplitude, period, and Galactic latitude should
beat a two-way extragalactic crossmatch easily. The failure modes are already
diagnosed, so the target is well defined.

- **Novelty:** moderate–high. Motivated directly by a finding nobody else has.
- **Effort:** low–moderate. Labels exist (SIMBAD CataclyV*/Nova, plus
  transient catalogs).
- **Risk:** low. Main limit is only 301 SIMBAD-confirmed CVs and 35 SNe —
  may need external CV catalogues (e.g. Catalina, AAVSO VSX) to build a
  training set.

#### R3. Fix the YSO recall deficit.

The classifier recovers only **43.6%** of SIMBAD YSOs (precision 0.940 — when
it says YSO it is right, it just misses most). YSOs also scatter into `cv`
(21.2%) and `lpv` (23.8%). Since YSOs are a headline science case for mid-IR
variability, a targeted recall improvement is directly valuable.

- **Novelty:** moderate. YSO classification is well-studied, but the specific
  VarWISE deficit is a new, quantified finding.
- **Effort:** moderate.
- **Risk:** moderate — YSO/LPV confusion is genuinely hard, both being red and
  variable.

---

### 🥈 Tier 2 — good, but more work or more risk

#### R4. VarWISE vs unTimely concordance study.

Two independent full-sky mid-IR variability catalogs built from the same
underlying WISE imaging by different methods (single-exposure + VARnet vs
coadd + Bayesian GMM). **Do they agree on which objects vary?** Where they
disagree, which is right? unTimely reports 8.26M variables against VarWISE's
1.92M — a 4× discrepancy demanding explanation.

- **Novelty:** high — nobody has compared them; unTimely variables is 9 months
  old, VarWISE 3 months.
- **Effort:** moderate–high (large cross-match, and unTimely's variability
  definition differs from VarWISE's).
- **Risk:** moderate. The catalogs use different detection thresholds, so
  disagreement may be definitional rather than substantive — that needs
  careful framing to be a result rather than a tautology.

#### R5. Rebuild the light-curve features and redo the label-efficiency study properly.

The current study uses 28 catalog-derived features because VarWISE's 31
light-curve features (Fourier coefficients, Stetson indices, χ² statistics)
are not in the published catalog. Rebuilding them from `neowiser_p1bs_psd`
would turn the approximation into a genuine reproduction and let the AL result
be stated against VarWISE's actual feature space.

- **Novelty:** low on its own — this is rigor, not discovery.
- **Effort:** high (positional cross-match against a billion-row
  single-exposure table).
- **Risk:** low scientifically, high on compute.
- **Verdict:** worth doing only if the AL work is submitted for publication.

#### R6. Mid-IR period–luminosity relations.

13,035 LPVs have both a period and parallax S/N > 5. Mid-IR PL relations are
tighter than optical (less extinction sensitivity).

- **Novelty:** low. Mira PL relations in W1/W2 are well-established.
- **Verdict:** only interesting if framed as *calibrating VarWISE's periods*
  against a known relation — which doubles as another audit, since periods
  that scatter off the PL relation are likely wrong. **That framing makes it a
  Tier 1 companion to R1.**

---

### 🥉 Tier 3 — avoid

- **Anomaly detection on NEOWISE** — comprehensively taken (see §1).
- **Anything requiring Gaia DR4 or LSST** — not available to you.
- **A general "active learning works in astronomy" paper** — already
  established; the existing repo's framing is the defensible version.

---

## 4. Suggested sequencing

1. **R1 (Extended audit)** — fastest path to a standalone, defensible result.
   Combine with the R6 framing (period–luminosity as an independent period
   check) for a stronger paper.
2. **R2 (CV/SN classifier)** — natural follow-on; turns a criticism into a
   contribution.
3. **R4 (unTimely concordance)** — highest ceiling, but scope it carefully.

R1 + R2 together form a coherent single paper: *an independent assessment of
the VarWISE catalog, with a corrected transient classifier*. That is a more
citable unit than either alone, and it is entirely within reach of data
already downloaded.

---

## 5. Honest caveat on the whole area

Three mid-IR variability catalogs appeared within twelve months, and VarWISE
has unusually high visibility. Follow-up work is likely to appear quickly, so
**re-run the novelty check immediately before committing to any of these** —
particularly R4, which is the most obvious idea for someone else to have.
