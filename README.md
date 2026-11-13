# Completeness of police-recorded injury deaths in India, by state and sex

Code and derived data for: *How many injury deaths do police record in India? Household death rosters as an independent
count, by state and sex* (Siddalingaiah H S; submitted). Archive: (Zenodo DOI to be added on release).

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
bash run_all.sh      # police counts, calibration, 10 model variants, sex ratio, figures, numbers, verify.py
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
