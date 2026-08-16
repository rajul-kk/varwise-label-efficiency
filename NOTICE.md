# Data acknowledgment and terms

This repository's analysis is built entirely on data hosted by the NASA/IPAC
Infrared Science Archive (IRSA). Per IRSA's data use terms
(https://irsa.ipac.caltech.edu/data_use_terms.html):

> This research has made use of the NASA/IPAC Infrared Science Archive, which
> is funded by the National Aeronautics and Space Administration and operated
> by the California Institute of Technology.

## Source catalog

VarWISE Pure and Extended Catalogs — Paz, M., Kirkpatrick, J. D.,
Uttamchandani, R., Raen, T., & Cutri, R. M. 2026, ApJS, 284, 41
([arXiv:2605.19059](https://arxiv.org/abs/2605.19059),
[doi:10.3847/1538-4365/ae562f](https://doi.org/10.3847/1538-4365/ae562f)).
Catalog DOI: [10.26131/IRSA656](https://doi.org/10.26131/IRSA656).

Both catalogs are public data products served via IRSA's TAP service
(`varwisepure`, `varwiseext`). Downloaded data is byte-for-byte verified
against the live source (`scripts/verify_download_integrity.py`,
`results/download_integrity_check.txt`).

## SIMBAD

Independent labels used throughout this repository's validation and
active-learning tracks are drawn from the `simbad_type` column, itself a
cross-match to the SIMBAD astronomical database, operated at CDS,
Strasbourg, France.

## Gaia

Parallaxes and photometry (`gmag`, `bpmag`, `rpmag`, `plx`, `e_plx`) are Gaia
DR3 values as cross-matched into the VarWISE catalog. This publication makes
use of data from the European Space Agency (ESA) mission Gaia
(https://www.cosmos.esa.int/gaia), processed by the Gaia Data Processing and
Analysis Consortium (DPAC,
https://www.cosmos.esa.int/web/gaia/dpac/consortium).

## What this repository adds and how it may be used

The code, analysis, figures, and prose in this repository are original work
by the repository author and are covered by the MIT license in `LICENSE`.

The derived data products in `results/` (in particular
`varwise_transient_corrections.csv`, a value-added table of corrected
classifications for VarWISE's rule-assigned `cv`/`sn` objects) are built
directly from IRSA-hosted VarWISE data and therefore carry through IRSA's
attribution requirement above — cite both the VarWISE paper and this
repository if you use them. They are independent, unofficial corrections;
they have not been reviewed or endorsed by the VarWISE authors.
