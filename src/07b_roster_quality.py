"""Roster data-quality diagnostics from the NFHS death rosters -> outputs/roster_quality.json (aggregates only).

infant_flag_pct_<round>      share of infant deaths flagged as injury (implausible; motivates the age >= 1 restriction)
age15_29_flag_pct_<round>    share of deaths at ages 15-29 flagged as injury
telangana_*                  NFHS-5 Telangana injury share overall and range across its districts
national_inj_share_allyrs_pct NFHS-5 injury share of all reported deaths (all years)
"""
import json, pathlib, sys
import numpy as np, pandas as pd

P = pathlib.Path(__file__).resolve().parents[1]; D = P / "data_local"; O = P / "outputs"
sys.path.insert(0, str(P / "src"))
from common import state_labels, unit, nfhs   # noqa: E402
N = {}
for r in ("nfhs5", "nfhs4"):
    z, do, _ = nfhs(r)
    de = pd.read_parquet(D / f"{r}_deaths.parquet"); de = de[(de.yr < 9000) & de.inj.isin([0, 1])]
    de["u"] = de.hv024.map(state_labels(z, do)).map(unit)
    infant = (de.age_u.isin([1, 2])) | ((de.age_u == 3) & (de.age_n == 0))
    N[f"infant_flag_pct_{r}"] = f"{100 * (de.inj[infant] == 1).mean():.1f}"
    a = np.where(de.age_u == 3, de.age_n, np.nan)
    N[f"age15_29_flag_pct_{r}"] = f"{100 * (de.inj[(a >= 15) & (a <= 29)] == 1).mean():.0f}"
    if r == "nfhs5":
        t = de[de.u == "TELANGANA"]
        N["telangana_inj_share_pct"] = f"{100 * (t.inj == 1).mean():.0f}"
        g = t.groupby("shdist").inj.apply(lambda v: (v == 1).mean())
        N["telangana_district_min_pct"] = f"{100 * g.min():.0f}"; N["telangana_district_max_pct"] = f"{100 * g.max():.0f}"
        N["telangana_n_districts"] = str(g.size)
        N["national_inj_share_allyrs_pct"] = f"{100 * (de.inj == 1).mean():.1f}"
json.dump(N, open(O / "roster_quality.json", "w", encoding="utf-8"), indent=1)
print(N)
