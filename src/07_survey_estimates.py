"""Design-based survey estimates by state x round: all-cause and injury death rates with linearised SEs.

Rate R = sum_i w_i y_i / sum_i w_i x_i over households (y = deaths in window, x = de jure members x years exposed).
Variance: Taylor linearisation, PSUs (hv021) within strata (hv022), with-replacement approximation:
  z_i = w_i (y_i - R x_i) / X;  var(R) = sum_h n_h/(n_h-1) sum_c (z_hc - zbar_h)^2.
Variants (VARIANT env):
  main      windows NFHS-5 Jan 2017-Feb 2020, NFHS-4 Jan 2013-interview; injury deaths at age >= 1 year
            (neonatal/infant injury flags implausible), all-cause all ages
  recall7   as main, but deaths and exposure in the 6 months before interview are dropped (telescoping check)
  allages   injury deaths at all ages
Output: outputs/survey_estimates_<variant>.csv (state x round), plus national rows (unit = INDIA).
"""
import os, pathlib, re, zipfile
import numpy as np, pandas as pd

P = pathlib.Path(__file__).resolve().parents[1]
D = P / "data_local"; O = P / "outputs"
VARIANT = os.environ.get("VARIANT", "main")
import sys
sys.path.insert(0, str(P / "src"))
from common import state_labels, unit, ratio_se, nfhs, cmc   # noqa: E402
ROUNDS = {"nfhs5": dict(zip=nfhs("nfhs5")[0], do=nfhs("nfhs5")[1], w0=cmc(2017, 1), w1=cmc(2020, 3)),
          "nfhs4": dict(zip=nfhs("nfhs4")[0], do=nfhs("nfhs4")[1], w0=cmc(2013, 1), w1=cmc(2017, 1))}


rows = []
for r, c in ROUNDS.items():
    lab = state_labels(c["zip"], c["do"])
    hh = pd.read_parquet(D / f"{r}_households.parquet")
    de = pd.read_parquet(D / f"{r}_deaths.parquet")
    hh["unit"] = hh.hv024.map(lab).map(unit)
    end = np.minimum(hh.hv008 - (6 if VARIANT == "recall7" else 0), c["w1"])
    hh["x"] = hh.hv012 * np.clip(end - c["w0"], 0, None) / 12
    de = de[de.yr < 9000].copy()
    de["dcmc"] = cmc(de.yr, de.mo.where(de.mo <= 12, 6))
    de = de.merge(pd.DataFrame({"hhid": hh.hhid, "end": end}), on="hhid")
    de = de[(de.dcmc >= c["w0"]) & (de.dcmc < de.end)]
    age_y = np.where(de.age_u == 3, de.age_n, np.where(de.age_u.isin([1, 2]), 0, np.nan))
    age_y = np.where(age_y >= 98, np.nan, age_y)
    inj = (de.inj == 1) & ((age_y >= 1) if VARIANT != "allages" else True)
    hh = hh.set_index("hhid")
    hh["y_all"] = de.groupby("hhid").size().reindex(hh.index, fill_value=0)
    hh["y_inj"] = de[inj].groupby("hhid").size().reindex(hh.index, fill_value=0)
    hh["w"] = hh.hv005 / 1e6
    hh["psu"] = hh.hv021.astype(int); hh["h"] = hh.hv022.astype(int) * 100 + hh.hv024.astype(int)
    for u, g in list(hh.groupby("unit")) + [("INDIA", hh)]:
        Ra, sa = ratio_se(g.y_all, g.x, g.w, g.psu, g.h)
        Ri, si = ratio_se(g.y_inj, g.x, g.w, g.psu, g.h)
        rows.append(dict(round=r, unit=u, n_hh=len(g), n_psu=g.psu.nunique(), deaths_all=int(g.y_all.sum()), deaths_inj=int(g.y_inj.sum()),
                         cdr=1000 * Ra, cdr_se=1000 * sa, inj_rate=1e5 * Ri, inj_se=1e5 * si,
                         log_cdr_se=sa / Ra, log_inj_se=si / Ri if Ri > 0 else np.nan))
S = pd.DataFrame(rows)
S.to_csv(O / f"survey_estimates_{VARIANT}.csv", index=False)
print(S[S.unit == "INDIA"].round(3).to_string(index=False))
big = S[(S.unit != "INDIA") & (S.deaths_inj >= 100)]
print(f"{VARIANT}: state-rounds with >=100 injury deaths: {len(big)}; median log-SE injury {big.log_inj_se.median():.3f}, all-cause {big.log_cdr_se.median():.3f}")
