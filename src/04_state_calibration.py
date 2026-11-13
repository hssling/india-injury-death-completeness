"""State roster calibration: survey crude death rate vs SRS death rate, and SRS-calibrated NCRB completeness.

SRS state death rates come from CRS Statement 2 (03_crs_srs_rates.py). CRS year -> SRS year: 2013-2016 same year;
CRS 2018 and 2019 both print SRS 2018 (CRS 2019 footnote; identical in all 37 rows).
Gaps: SRS 2017 = mean(2016, 2018) by state; SRS 2019 = SRS 2018 x (6.0/6.2), the national SRS 2019/2018 ratio
(SRS Bulletin 2019, national 6.0). Both are approximations and are flagged in outputs.

k_s = survey CDR_s / SRS CDR_s (roster reporting factor). Calibrated survey injury rate = raw / k_s.
NCRB completeness (calibrated) = NCRB (accidental + suicide) rate / calibrated survey injury rate.
Assumes the roster over/under-reporting factor is the same for injury and other deaths (tested later).
"""
import pathlib
import numpy as np, pandas as pd

P = pathlib.Path(__file__).resolve().parents[1]
crs = pd.read_csv(P / "data/crs_srs_state_year.csv")
crs = crs[crs.crs_year != 2019].assign(srs_year=lambda d: d.crs_year)          # 2019 duplicates SRS 2018
U = {"A & N Islands": "A & N ISLANDS", "Delhi": "DELHI (UT)", "D & N Haveli": "DNH & DD", "Daman & Diu": "DNH & DD",
     "Jammu & Kashmir": "JAMMU & KASHMIR (incl. LADAKH)"}
crs["unit"] = crs.state.map(lambda s: U.get(s, s.upper()))
crs = crs[crs.unit != "INDIA"]
# population-weighted DR for merged units
srs = crs.groupby(["unit", "srs_year"]).apply(lambda g: np.average(g.srs_dr, weights=g.pop_000), include_groups=False).unstack()
srs[2017] = (srs[2016] + srs[2018]) / 2
import sys
sys.path.insert(0, str(P / "src"))
from common import SRS_INDIA_2019   # noqa: E402
ind18 = pd.read_csv(P / "data/crs_srs_state_year.csv").query("crs_year == 2018 and state == 'India'").srs_dr.iloc[0]
srs[2019] = srs[2018] * SRS_INDIA_2019 / ind18
srs = srs[sorted(srs.columns)]
W = {"nfhs5": {2017: 1, 2018: 1, 2019: 1, 2020: 2 / 12}, "nfhs4": {2013: 1, 2014: 1, 2015: 1, 2016: 0.5}}  # approx person-time shares
srs[2020] = srs[2019]                    # Jan-Feb 2020 only (pre-COVID); SRS 2020 not used
k = pd.read_csv(P / "outputs/killcheck_state.csv")
out = []
for r, w in W.items():
    yrs = [y for y in w if y in srs.columns]
    s_cdr = sum(srs[y] * w[y] for y in yrs) / sum(w[y] for y in yrs)
    g = k[k["round"] == r].set_index("unit")
    g["survey_cdr"] = 1000 * g.deaths_w / g.pt
    g["srs_cdr"] = s_cdr.reindex(g.index)
    g["k_s"] = g.survey_cdr / g.srs_cdr
    g["survey_inj_calib"] = g.survey_rate / g.k_s
    g["ncrb_completeness_calib"] = g.ncrb_rate / g.survey_inj_calib
    out.append(g.reset_index())
O = pd.concat(out, ignore_index=True)
O.to_csv(P / "outputs/state_calibration.csv", index=False)
big = O[O.inj_unw >= 100]
print(big.pivot_table(index="unit", columns="round", values=["k_s", "ratio_raw", "ncrb_completeness_calib"]).round(2).to_string())
print("\nk_s summary by round (states with >=100 flagged deaths):")
print(big.groupby("round").k_s.describe().round(2).to_string())
for r in W:
    g = O[O["round"] == r]
    cal = (g.ncrb_rate * g.pt).sum() / (g.survey_inj_calib * g.pt).sum()   # person-time-weighted national
    print(f"{r}: national NCRB completeness, state-calibrated = {cal:.2f}")
print("corr(k_s, raw ratio) across big state-rounds:", round(np.corrcoef(big.k_s, big.ratio_raw)[0, 1], 2))
