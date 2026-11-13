#!/usr/bin/env bash
# Reproduce every reported number from the shipped tables (no microdata needed), then verify.
set -euo pipefail
run() { echo ">> $*"; python "$@"; }
if [ -d data_local ] && [ -f data_local/nfhs5_deaths.parquet ]; then   # optional: steps that need NFHS/PLFS unit-level data
  run src/02_kill_check.py; run src/05_registration_premium.py; SAMPLE=all python src/05_registration_premium.py
  for v in main recall7 allages; do VARIANT=$v python src/07_survey_estimates.py; done
  run src/07b_roster_quality.py
fi
run src/02b_ncrb_window.py
run src/04_state_calibration.py
for v in main srs_complete roster_exact rho_wide recall7 allages min30 ncrb_strict ncrb_2014_16 migration; do
  VARIANT=$v python src/08_bayes_model.py
done
run src/10_sex_ratio.py
run src/11_figures.py
run src/12_numbers.py
run verify.py
run verify.py --selftest
