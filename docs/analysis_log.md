# Kill check: NFHS household death rosters vs NCRB (2026-11-13)

Scripts: `src/01_extract_roster.py` (extraction), `src/02_kill_check.py` (checks).
Outputs: `outputs/killcheck_national.csv`, `outputs/killcheck_state.csv`.
Unit-level NFHS data are kept in `data_local/` and must not be redistributed.

## Data

| Round | Households | Deaths reported | Deaths in window | Injury-flagged in window | Window |
|---|---|---|---|---|---|
| NFHS-5 (2019–21) | 636,699 | 81,340 | 63,711 | 6,012 (don't know 131) | Jan 2017 – Feb 2020 (COVID excluded) |
| NFHS-4 (2015–16) | 601,509 | 74,945 | 63,231 | 6,199 (don't know 202) | Jan 2013 – interview |

- Up to 5 death slots per household, and no household reported more than 5, so there is no truncation.
- The injury flag is "accident, violence, poisoning, homicide or suicide". The injury share of deaths is 9.1% (NFHS-5) and 10.0% (NFHS-4), matching Panda & Mishra 2021 (10.3%).

## Pre-specified kill rules

| Rule | Threshold | Result | Verdict |
|---|---|---|---|
| K1: roster completeness of all deaths vs SRS crude death rate | kill if < 60% | 132% (NFHS-5), 126% (NFHS-4) | PASS, with a calibration issue (see below) |
| K2: states with ≥100 flagged deaths, both rounds | kill if < 20 | 24 | PASS |
| K3: state-rounds with raw NCRB/survey > 1.2 | kill if most | 0% (maximum 1.19, Chhattisgarh NFHS-5) | PASS |

**Verdict: GO.**

## Headline signal (raw, before modelling)

- **National ratio.** NCRB accidental deaths plus suicides, divided by survey injury deaths: 0.55 (2017–19) and 0.47 (2013–16).
  - After scaling the survey to the SRS crude death rate: 0.73 and 0.59.
  - The NCRB side excludes homicide (about 2/100k), which biases the ratio slightly down.
- **State spread is wide** (NFHS-4 / NFHS-5):

  | State | Ratio |
  |---|---|
  | Bihar | 0.11 / 0.21 |
  | Jammu & Kashmir | 0.18 / 0.17 |
  | Uttar Pradesh | 0.25 / 0.27 |
  | West Bengal | 0.25 / 0.35 |
  | Telangana | – / 0.27 |
  | Maharashtra | 0.72 / 0.97 |
  | Madhya Pradesh | 0.61 / 1.00 |
  | Chhattisgarh | 1.08 / 1.19 |

  Police capture of household-reported injury deaths appears to vary roughly 5–10-fold between states.
- **Registration.** First look at NFHS-5 (sh91), weighted and unadjusted: deaths flagged as injury were registered 78.7% of the time, against 69.8% for other deaths. This is a crude premium of +9 percentage points.
  - It needs adjustment for age, sex, residence, wealth and state, because injury deaths are concentrated in young adult males, who register more.
  - Saikia et al. (Bull WHO 2023, PMC9874366) only hypothesise that police involvement in accidental deaths raises registration. Singh et al. (BMC Public Health 2025) do not split registration by cause. **Neither tabulates sh95.**

## Issue to resolve in the design: the roster gives MORE deaths than SRS

- **Size.** The survey crude death rate is 8.0 unweighted (8.4 weighted) per 1,000, against SRS 6.0–6.3. It is the same in both rounds, so it is not a weighting artifact.
- **Recall profile.** Deaths by months before interview (NFHS-5, pre-COVID households, 6-month bins): 8,891 / 7,214 / 6,924 / 6,825 / 6,889 / 6,237. The most recent half-year is about 25% higher than the other bins, which are flat. This points to forward telescoping into recent months, or recall decay of older deaths. The remaining excess over SRS is level.
- **Candidate explanations:**
  - SRS under-registration of deaths;
  - duplicate reporting of one death by two households (split joint families, "usual member" ambiguity);
  - the denominator (de jure members at interview);
  - telescoping.
- **Consequence.** Raw survey rates overstate the true rate if the excess is roster over-reporting, and correct (and SRS low) if it is SRS under-registration. NCRB completeness therefore lies between the raw and the SRS-calibrated ratio (national 0.55–0.73 for 2017–19).
- **Plan.** The model must carry roster calibration explicitly:
  - k_s as a parameter, with an SRS-anchored prior that allows SRS incompleteness;
  - sensitivity runs using months 7–36 only (drops the telescoping bin), raw versus calibrated rates, and adding Crime-in-India homicides.
  - The design assumption to test is that the over-reporting factor is the same for injury and non-injury deaths. Injury share by recall bin is the diagnostic.

## To verify before writing

- SRS national CDR series used here (2013 7.0 … 2019 6.0) is from memory. Replace it with SRS Bulletin values, and add state CDRs (SRS Statistical Reports, or CRS "level of registration" tables).
- Crime in India murder + culpable homicide not amounting to murder (CHNATM), by state 2013–2019, are not on disk.
- Residence vs place-of-occurrence mismatch: Delhi, Chandigarh, Goa and the hill states with tourist traffic need flagging.
- NFHS-4 Telangana and Andhra Pradesh are separate (97 and 109 flagged deaths); the NaN cells in the ratio table are only the ≥100 filter. Telangana NFHS-5 has 514 flagged deaths out of 3,342 (15%, against about 9% nationally). Check this before trusting it: possible oversampling or a real excess.

## Progress 2026-11-13 (after GO)

### SRS state death rates (src/03_crs_srs_rates.py → data/crs_srs_state_year.csv)

- **Source:** CRS annual reports 2013–2020 from dc.crsorgi.gov.in (saved in data_raw/crs/ with SHA256SUMS).
- **Table used:** "Statement 2: Estimated events and Level of Registration", parsed for 2013–2016, 2018 and 2019. In 2017 and 2020 the statement has no extractable text.
- **Checks:**
  - every row satisfies est_deaths = population × DR;
  - the units sum to the India row (0.995–1.002; 0.970 in 2013, when Andhra Pradesh and Telangana are split synthetically).
- **SRS year mapping:** CRS 2019 prints SRS **2018** rates (footnote), identical in all 37 rows to CRS 2018. National values match the SRS series (2013 7.0, 2014 6.7, 2015 6.5, 2016 6.4, 2018 6.2).
- **Approximations:**
  - SRS 2017 = mean of 2016 and 2018 by state;
  - SRS 2019 = 2018 × 6.0/6.2.
- **Bonus:** CRS level of death registration by state (lor_deaths) is in the same file.

### State roster calibration (src/04_state_calibration.py → outputs/state_calibration.csv)

- **k_s = survey CDR / SRS CDR:** median 1.21 (NFHS-4) and 1.28 (NFHS-5); range 0.96–1.69.
- **High k_s states:** Bihar 1.6, West Bengal 1.4–1.7, Andhra Pradesh and Telangana about 1.5, Manipur 1.7. These are states where SRS itself is suspected low (Bihar SRS CDR 5.8), so k_s mixes roster over-reporting with SRS under-coverage.
- **National NCRB completeness, state-calibrated:** 0.74 (2017–19) and 0.59 (2013–16). This is close to the national calibration.
- **Calibrated completeness above 1:** Maharashtra 1.36 and Haryana 1.27 (NFHS-5). These are in-migration and highway states where police record non-resident deaths. Full SRS calibration therefore overcorrects.
- **Design consequence:** raw and calibrated figures bracket the truth. The model must treat SRS as imperfect too, using:
  - a weakly informative prior on SRS coverage;
  - a place-of-occurrence term (net in-migration or a highway-density covariate).

### Registration premium (src/05_registration_premium.py → outputs/registration_premium*.csv)

- **Sample:** NFHS-5 deaths Jan 2017 – Feb 2020: n = 63,107, of which 6,044 are injury.
- **Prevalence ratio** (modified Poisson; survey weights; PSU-clustered SEs):

  | Model | PR (95% CI) |
  |---|---|
  | Crude | 1.125 (1.104–1.147) |
  | + age, sex, residence, wealth, year | 1.125 |
  | + state fixed effects | **1.092 (1.071–1.112)** |

- **Standardized registration:** injury 76.6% vs other 70.1%, so **+6.4 pp**.
- **Larger where baseline registration is low:**
  - rural PR 1.10 vs urban 1.05;
  - age 0–4 PR 1.22 and 5–14 PR 1.24 vs adults 1.05–1.07;
  - 15–29 PR 1.12.
- **Interpretation:** consistent with a police/medico-legal pathway (inquest, post-mortem, compensation or insurance claims needing a death certificate). This is the hypothesis Saikia et al. (Bull WHO 2023) raised but did not test.

### Data-quality flag: injury flag among neonates

- 7.1% (NFHS-5) and 6.5% (NFHS-4) of neonatal deaths are flagged as injury, about 380 per round. That is implausible and likely reflects birth complications reported as "accident".
- Infants 1–11 months: 9–11%. Ages 1–4: 12–14%. Ages 15–29: 36–37%, close to SRS COD.
- **Main analysis will exclude deaths under 1 year,** with a sensitivity run including them. Rerun 05 with age ≥1 for the main estimate.

### Update: murders added and a J&K bug fixed (2026-11-13)

- **Murders.** Crime in India Table 2A.1 murder cases, from the user-saved CII 2016 (2014–16) and CII 2019 Vol I (2017–19); `src/06_cii_murders.py` → `data/cii_murders_state_year.csv`.
  - State sums equal the printed All-India totals in all six years.
  - Culpable homicide not amounting to murder (about 3,000 a year) is not included.
  - West Bengal 2019 = 2018 (CII footnote).
- **Bug fixed.** The suicide file names the state "JAMMU & KASHMIR", but the panel uses "JAMMU & KASHMIR (incl. LADAKH)". J&K suicides were therefore silently summed as 0.
  - J&K ratio corrected: 0.17–0.18 → 0.22.
  - 02_kill_check.py now asserts that every component (accidental, suicide, murder, population) is present for all 35 units × years.
- **Revised headline**, NCRB accidental + suicide + murder vs survey injury deaths:

  | Period | Raw | SRS state-calibrated |
  |---|---|---|
  | 2017–19 | 0.58 | 0.78 |
  | 2013–16 | 0.50 | 0.63 |

  - K3 is now 2% (one state-round above 1.2: Chhattisgarh NFHS-5, 1.24).
- **Registration premium, main sample** (deaths at age ≥1, n = 53,881): adjusted PR **1.071 (1.050–1.092)**, standardized +5.2 pp (78.0% vs 72.8%). All ages: PR 1.092, +6.4 pp.

## Bayesian measurement model (src/08_bayes_model.py), 2026-11-13

**Model.** The structure and priors are in the script docstring. Key points:
- The model is anchored (non-centred): log λ = log(NCRB/PY) − log C − noise, and log μ = log SRS − log c − noise.
- A free parameterisation failed (R-hat ~10^6), because the Poisson counts of about 10^5 make a razor-thin ridge in λ × C.

**NCRB definition, fixed before the final runs.**
- Injury-only: ADSI other-causes Total − Sudden Deaths − Deaths of Women during Pregnancy, plus forces of nature excluding epidemic and starvation, plus suicides, plus CII murders.
- NFHS-4 is compared with NCRB 2016 only, because ADSI 2014–15 used a different classification.

**Convergence.** All nine variants have R-hat ≤ 1.010 and ESS ≥ 462. Divergences: 0–2.

### National completeness C (posterior median, 95% CrI)

| Variant | 2013–16 (NFHS-4) | 2017–19 (NFHS-5) |
|---|---|---|
| **main** | **0.64 (0.51–0.80)** | **0.72 (0.58–0.91)** |
| SRS complete (c = 1) | 0.72 (0.58–0.87) | 0.81 (0.66–0.99) |
| roster exact (k = 1, raw) | 0.56 (0.46–0.69) | 0.61 (0.50–0.74) |
| ρ prior sd 0.25 | 0.63 (0.39–1.02) | 0.72 (0.44–1.15) |
| drop last 6 months (recall) | 0.61 | 0.70 |
| injury at all ages | 0.56 | 0.63 |
| units ≥ 30 deaths | 0.63 | 0.72 |
| NCRB excl. 'causes not known' | 0.62 | 0.70 |
| NFHS-4 vs NCRB 2014–16 | 0.63 | 0.72 |

- The national level depends on the identifying assumption: the bracket is 0.61–0.81.
- **State ranking is invariant:** Spearman correlation with the main model is ≥ 0.974 in every variant.

### States, 2017–19 (main)

- **Lowest:**

  | State | C (95% CrI) |
  |---|---|
  | J&K | 0.27 (0.20–0.39) |
  | Nagaland | 0.30 |
  | Manipur | 0.34 |
  | Assam | 0.36 |
  | Uttar Pradesh | 0.39 (0.30–0.52) |
  | Uttarakhand | 0.40 |
  | Bihar | 0.44 (0.32–0.60) |
  | Arunachal Pradesh | 0.45 |
  | Telangana | 0.46 |

  P(C < 0.5) ≥ 0.68 for each of these.
- **C > 1:** Delhi 1.72, Chhattisgarh 1.24, Haryana 1.22, Puducherry 1.19, Madhya Pradesh 1.17, Goa, Rajasthan, Maharashtra, Sikkim and Karnataka at about 1.0–1.1.
  - Delhi, Haryana, Goa, Puducherry and Maharashtra are in-migration, transit or tourist states.
  - NCRB counts by place of occurrence, while the survey counts usual residents. The estimand is therefore "police-recorded deaths per true resident injury death".
  - **Next:** model C with a net-migration or highway covariate to separate recording from place of occurrence.
- **Change NFHS-4 → NFHS-5:**
  - Rose (P > 0.95): Bihar ×1.93, Manipur 1.65, Haryana 1.55, Kerala 1.51, Madhya Pradesh 1.49, Odisha 1.46, West Bengal 1.40, Jharkhand 1.38, Rajasthan 1.37.
  - Fell: Telangana ×0.62, Assam ×0.68.
  - Telangana's survey injury deaths jumped from 97 to 514, with a 15% injury share in NFHS-5. Check this survey artefact before interpreting.

### External benchmark

- GBD 2023 injury death rate, India 2020: 64.4/100k.
- Survey raw (age ≥ 1): 64.2/100k.
- Model "true" rate: about 55–60/100k.
- So GBD sides with the raw roster on injuries, which suggests the roster's all-cause excess over SRS sits mainly in non-injury deaths. That would mean ρ < 1 and a main C that is slightly high. This is reported as a bracket, not a correction.

## Telangana, migration, sex (2026-11-13)

**Telangana NFHS-5 injury share (20%).**
- Uniform across all 31 districts (13–27%) and all years, and not concentrated in a few PSUs.
- Its age pattern differs from the national one: ages 30–59 are 32% vs a national median of 14.5%.
- GBD 2023 also gives Telangana the highest injury death rate in India (118/100k vs 64).
- So it is probably largely real. The NFHS-4 → NFHS-5 completeness fall for Telangana (×0.62) is probably the small NFHS-4 Telangana sample (1,032 deaths), not a recording change. Report it with this caveat.
- Infant injury false flags are national, not state-specific: median 8% of infant deaths are flagged in both rounds. This justifies excluding deaths under 1 year everywhere.

**Place of occurrence: VARIANT=migration** (`src/09_migration_covariate.py`; PLFS 2020-21 Block 7.1).
- In-migrant definition: last usual place of residence in another state, moved before April 2020. National share 3.0%; Delhi 22%, DNH & DD 31%; NE states and Bihar about 0.
- β = 0.30 (0.14–0.45) on log C per SD of log in-migrant share.
- Migration-adjusted ("recording") C, 2017–19:
  - Delhi 1.73 → 1.04, Haryana 1.23 → 0.99, Puducherry 1.35 → 0.79, Goa 1.15 → 0.87.
  - Robust low (both versions): J&K 0.30, Uttarakhand 0.33, Uttar Pradesh 0.42, Arunachal 0.44, Telangana 0.45, West Bengal 0.55, Andhra Pradesh 0.56.
  - Zero-migration states rise (Bihar 0.67, Nagaland 0.48, Mizoram 1.37). This is partly confounding (migration correlates with administrative capacity), so treat the adjusted values as upper bounds there.
  - Madhya Pradesh 1.18, Chhattisgarh 1.26 and Rajasthan 1.13 stay above 1. These are freight-transit states (through-traffic deaths) or states with roster under-reporting.

**Sex: female/male completeness ratio** (`src/01b_extract_member_sex.py`, `src/06b_cii_victims_sex.py`, `src/10_sex_ratio.py`).
- Inputs:
  - Murder victims by sex: CII Table 2A.3, 2016 and 2019. Sums are exact against the printed All-India totals.
  - De jure members by sex: hv102/hv104. They sum to hv012 in 99.98% (NFHS-5) and 100% (NFHS-4) of households.
- **Pooled female/male ratio of police completeness: 0.77 (0.69–0.88) in 2017–19; 0.60 (0.54–0.68) in 2013–16.**
  - Raw national: 0.76 and 0.60.
  - Between-state SD (log) 0.21. R-hat 1.001, no divergences.
- k, c and ρ cancel if they are sex-invariant, so this result does NOT depend on the bracketing assumptions.
- **Lowest states (2017–19):** Jharkhand 0.55, Haryana 0.62, J&K 0.63, Punjab 0.63, Kerala 0.69, Telangana 0.70, Andhra Pradesh 0.70, West Bengal 0.71, Bihar 0.71, Uttar Pradesh 0.73. P(R < 1) = 1.00 for 13 states.
- Female injury deaths are never also flagged as maternal (0 of 799 at ages 15–49 in NFHS-5; 0 of 880 in NFHS-4), so maternal misreporting does not explain the gap.
- Remaining threat: if households over-report female injury deaths relative to male (sex-specific ρ), R would be biased down. That seems unlikely: concealment of female suicide or dowry deaths would push the other way.
