# Novelty audit

Consolidated, re-checked 18 August 2026. Every claim below was searched
fresh this pass, not carried over from earlier checks. Verdicts are ranked
by strength, not by order of discovery.

| # | Finding | Verdict | Basis |
|---|---|---|---|
| 1 | Extended-tier `rr` sits 3 mag off the RR Lyrae locus; period-range and period-luminosity checks agree | **Novel** | No independent validation of VarWISE exists anywhere (searched this pass: no hits). Two independent methods converging is what makes it solid, not just new. |
| 2 | `ecl`/`yso` scatter into `agn` exclusively at the North Ecliptic Pole, tracking a 6–9× epoch-count excess | **Novel** | No prior cross-match of VarWISE against any second classification pipeline found. One close call this pass — see §3 below — resolved in the finding's favor with primary data, not search results. |
| 3 | `cv`/`sn` transient-assignment rule over-predicts 38–96×; corrected-label table shipped | **Novel finding, partly anticipated by the authors** | VarWISE's own paper already flags ~56% of `sn` as "normal AGNs" from visual inspection. The *quantification*, the *mechanism* (two sub-populations, not one), and the *fix* (96.6% accuracy classifier) are new; the qualitative direction was not. |
| 4 | 86% active-learning label saving on VarWISE/NEOWISE, concentrated in rare classes, survives a composition-matched control | **Replication in a new archive, with one added rigor step** | Richards et al. 2011 already established AL works for variable-star classification. The composition-matched control (separating rebalancing from informativeness) is less commonly done and is the actual contribution here. |
| 5 | Distillation targets (training on a classifier's own predictions) understate measured AL benefit; formalized via majorization/Schur-convexity | **Novel framing on standard math; empirically moderate** | Label-bias propagation from catalog labels is an established topic generally (galaxy-morphology de-biasing work). This specific connection to AL label-efficiency measurement is new. Confirmed on real data (2 points, both estimators) and on redesigned synthetic data (ρ = −0.33 to −0.60, moderate not strong). |
| 6 | Hard per-class label quotas underperform random sampling | **Replication of a negative result** | Same finding already recorded in the `chandra-toolkit` companion project. Useful to reconfirm, not new on its own. |
| 7 | `neowiser_p1bs_psd` is too large (~10¹¹ rows, confirmed by direct query) to rebuild VarWISE's light-curve features via TAP | Not a finding — a **feasibility result** | Closes a question rather than opening one; recorded so it isn't re-attempted without checking first. |

## The one claim that needed real scrutiny this pass

A search this session surfaced a snippet claiming VarWISE has "zero density…
at the north and south ecliptic poles" — which, if true, would have
invalidated finding #2 (5,267 real crossmatches exist in that exact region).
Checked directly against the locally verified catalog rather than trusting
the search summary:

| region | declination band | object count |
|---|---|---|
| Ecliptic pole (where finding #2 sits) | 64°–72° | **7,979** |
| True celestial pole | > 89.5° | **2** |

The search snippet conflated the *celestial* pole (a genuine, tiny WISE
scan-pattern gap, unrelated to this analysis) with the *ecliptic* pole
(where the analysis and the comparison catalog both sit, and where density
is substantial). Finding #2 stands; primary data resolved this, not a
second search.

## What's *not* claimed as novel

- Every acquisition function used (margin, uncertainty, quota, prototype) is
  standard, off-the-shelf active learning.
- The coupon-collector and Schur-convexity building blocks behind finding #5
  are textbook, not new mathematics.
- The diagnostic *methods* behind findings #1–3 (period-luminosity checks,
  cross-validated classifiers, cross-catalog concordance) are standard
  astronomical practice — the novelty is in *where they're pointed* (a
  three-month-old catalog nobody had audited), not in the methods
  themselves.

## Strength ranking, for anyone deciding what to lead with

**Strongest → weakest:** #1 (two independent methods agree) → #3 (fix
shipped, generalization checked, reliability-tiered) → #2 (real but thin
sample, single region) → #4 (solid replication) → #5 (real but moderate
effect size, honestly reported after two redesigns) → #6 (useful
confirmation) → #7 (negative/feasibility result).
