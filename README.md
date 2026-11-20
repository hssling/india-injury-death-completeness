# Completeness of police-recorded injury deaths in India, by state and sex

Code and derived data for a manuscript on the completeness of police-recorded injury deaths in India (Siddalingaiah H S;
under review). Archive (all versions): https://doi.org/10.5281/zenodo.24199145; v1.2.0: https://doi.org/10.5281/zenodo.24391356; v1.1.0: https://doi.org/10.5281/zenodo.24322112; v1.0.0:
https://doi.org/10.5281/zenodo.24199146.

Version 1.2.0 (2026-11-20) adds the decision consequences of incomplete recording (`src/15_policy_decisions.py`, from the
posterior draws): a fitness-for-use category and correction multiplier for each state, the share of a budget that police-based
allocation would send to the wrong states (dissimilarity index), and how often police counts misjudge a 10% reduction
target, nationally and by state (`outputs/policy.json`, `outputs/policy_by_state.csv`). Earlier estimates are unchanged.

Version 1.1.0 (2026-11-17) added unrecorded deaths per year (national, state, sex), the bias that changing completeness
introduces into trends, an external comparison with GBD 2023 and civil registration (`src/14_implications.py`, posterior
draws in `outputs/bayes_draws_*.npz`), and a state-by-round term in the sex-ratio model. Earlier estimates are unchanged.

## What this does

Deaths reported in the household rosters of the National Family Health Survey (NFHS-4, 2015–16; NFHS-5, 2019–21), with
the item "was the death due to an accident, violence, poisoning, homicide or suicide?", are compared with police (NCRB)
counts of injury deaths (accidental deaths excluding sudden deaths, deaths in pregnancy and epidemics; suicides; murders).
A Bayesian measurement model allows for household reporting error, Sample Registration System under-coverage and survey
design error, and estimates the completeness of police recording by state and round, its female-to-male ratio, and the
civil-registration premium of injury deaths.

## Reproduce

```bash
pip install -r requirements.txt -r requirements-bayes.txt
bash run_all.sh      # police counts, calibration, 10 model variants, sex ratio, implications, policy, figures, numbers, verify.py
```

`verify.py` recomputes the key quantities from `data/` and `outputs/`, checks every worded claim against its number, and
(without the manuscript) answers the text checks from `results/expected_claims.json`. `verify.py --selftest` plants a wrong
number and requires a new failure. The CI workflow refits the main model and the sex-ratio model and requires agreement with
the shipped summaries within 5% on the log scale.

## Data

| Folder | Content | Source |
|---|---|---|
| data/adsi_state_year_sex.csv | NCRB accidental deaths by cause, state, year, sex, 2014–2024 | NCRB, Accidental Deaths and Suicides in India |
| data/ncrb_suicides_state_year.csv | NCRB suicides by state, year, sex | NCRB ADSI |
| data/cii_murders_state_year.csv, data/cii_murder_victims_sex.csv | Murder cases 2014–2019; victims by sex 2016, 2019 | NCRB, Crime in India 2016, 2019 (tables 2A.1, 2A.3) |
| data/crs_srs_state_year.csv | SRS death rate and estimated deaths by state | ORGI, Civil Registration System reports, statement 2 |
| data/state_population_units.csv | Population, 35 units, 2014–2024 | Technical Group on Population Projections (2019) |
| data/plfs_inmigration_2020_21.csv | Inter-state in-migrant share by state (aggregates) | PLFS 2020–21 unit-level data |
| data/gbd_2023_extract.csv | All-cause (2019) and injury (2020) death rates | GBD 2023 results (IHME) |
| outputs/ | Survey state aggregates, police windows, model summaries, sex ratios, registration models | this analysis |

**Microdata (not included).** NFHS household recodes (IAHR74, IAHR7E) are free from https://dhsprogram.com after registration;
PLFS unit-level data from https://microdata.gov.in. Neither may be redistributed. Set `NFHS_DIR` and `PLFS_ZIP` (or
`data_local/paths.json`) and run steps 01, 01b, 02, 05, 07, 07b and 09 to regenerate the survey aggregates shipped in
`outputs/`.

The state map (Figure 1) is shipped rendered: the state geometry is derived from district boundaries whose licence does
not allow redistribution.

## Licence

Code: MIT. Derived data, outputs and supplement: CC BY 4.0. GBD extract: © IHME (cite as in LICENSE).
