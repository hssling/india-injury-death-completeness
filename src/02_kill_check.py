"""Kill check for the NFHS death-roster vs NCRB idea (rules pre-specified in the opportunity scan of 12 November 2026).

KILL if any of:
  K1  roster completeness of total deaths < ~60% nationally (survey CDR / SRS CDR)
  K2  fewer than ~20 states with >= 100 injury-flagged deaths across both rounds
  K3  raw NCRB/NFHS ratio > 1.2 in most large states (police would then record MORE than households report)

Survey quantities are weighted (hv005); exposure = weighted de jure members x person-time in the window, so the
weight scale cancels in rates. Windows avoid COVID: NFHS-5 deaths Jan 2017 - Feb 2020; NFHS-4 Jan 2013 - interview.
Person-time per household = months from window start to min(interview, window end).
Output: outputs/killcheck_*.csv and printed summary.
"""
import os, pathlib, re, zipfile
import numpy as np, pandas as pd

P = pathlib.Path(__file__).resolve().parents[1]
M = P.parent
D = P / "data_local"; O = P / "outputs"; O.mkdir(exist_ok=True)
SRS_CDR = {2013: 7.0, 2014: 6.7, 2015: 6.5, 2016: 6.4, 2017: 6.3, 2018: 6.2, 2019: 6.0}   # SRS bulletins, India, per 1,000 (to verify)


from common import state_labels, unit, cmc, nfhs   # noqa: E402


ROUNDS = {"nfhs5": dict(zip=nfhs("nfhs5")[0], do=nfhs("nfhs5")[1], w0=cmc(2017, 1), w1=cmc(2020, 3)),
          "nfhs4": dict(zip=nfhs("nfhs4")[0], do=nfhs("nfhs4")[1], w0=cmc(2013, 1), w1=cmc(2017, 1))}
rows, srows = [], []
for r, c in ROUNDS.items():
    lab = state_labels(c["zip"], c["do"])
    hh = pd.read_parquet(D / f"{r}_households.parquet")
    de = pd.read_parquet(D / f"{r}_deaths.parquet")
    hh["unit"] = hh.hv024.map(lab).map(unit)
    de["unit"] = de.hv024.map(lab).map(unit)
    assert hh.unit.notna().all()
    w = hh.hv005 / 1e6
    end = np.minimum(hh.hv008, c["w1"])
    pt = w * hh.hv012 * np.clip(end - c["w0"], 0, None) / 12           # weighted person-years
    hh["pt"] = pt
    # deaths: valid month/year; month unknown -> mid-year
    yr = de.yr.where(de.yr < 9000); mo = de.mo.where(de.mo <= 12, 6)
    de["dcmc"] = cmc(yr, mo)
    de = de.merge(hh[["hhid"]].assign(end=end.values), on="hhid")
    inwin = (de.dcmc >= c["w0"]) & (de.dcmc < de.end)
    dw = de[inwin].copy(); dw["w"] = dw.hv005 / 1e6
    dw["injury"] = (dw.inj == 1).astype(int)
    print(f"\n=== {r}: deaths in window {inwin.sum():,} of {len(de):,} (year unknown {yr.isna().sum():,}); "
          f"injury flag yes {int(dw.injury.sum()):,}, don't know {int((dw.inj == 8).sum())}, missing {int(dw.inj.isna().sum())}")
    print(" year-of-death distribution:", de.yr.where(de.yr < 9000).value_counts().sort_index().to_dict())
    cdr = 1000 * dw.w.sum() / hh.pt.sum()
    inj = 1e5 * (dw.w * dw.injury).sum() / hh.pt.sum()
    yrs = range(c["w0"] // 12 + 1900, (c["w1"] - 1) // 12 + 1900 + 1)
    srs = np.mean([SRS_CDR[y] for y in yrs if y in SRS_CDR])
    print(f" survey CDR {cdr:.2f}/1000 vs SRS {srs:.2f} -> roster completeness {cdr / srs:.0%}; injury share "
          f"{(dw.w * dw.injury).sum() / dw.w.sum():.1%}; injury death rate {inj:.1f}/100k")
    srows.append(dict(round=r, deaths_window=int(inwin.sum()), cdr=cdr, srs_cdr=srs, completeness=cdr / srs,
                      injury_share=(dw.w * dw.injury).sum() / dw.w.sum(), injury_rate_100k=inj))
    g = dw.groupby("unit").apply(lambda x: pd.Series({"inj_unw": x.injury.sum(), "deaths_unw": len(x),
                                                       "inj_w": (x.w * x.injury).sum(), "deaths_w": x.w.sum()}), include_groups=False)
    g["pt"] = hh.groupby("unit").pt.sum()
    g["round"] = r; g["years"] = f"{min(yrs)}-{max(yrs)}"
    rows.append(g.reset_index())

S = pd.DataFrame(srows); S.to_csv(O / "killcheck_national.csv", index=False)
G = pd.concat(rows, ignore_index=True)

# ---- NCRB injury deaths per unit x round: computed from data/ by 02b_ncrb_window.py (run it first)
NW = pd.read_csv(O / "ncrb_window.csv")
ncrb_rate = {}
for r, x in NW.groupby("round"):
    x = x.set_index("unit")
    ncrb_rate[r] = 1e5 * x.ncrb / x.person_years
    nat = 1e5 * x.ncrb.sum() / x.person_years.sum()
    s = S.set_index("round").loc[r]
    print(f"{r}: NCRB acc+suicide+murder {nat:.1f}/100k vs survey injury {s.injury_rate_100k:.1f}/100k -> raw ratio {nat / s.injury_rate_100k:.2f};"
          f" after roster-completeness correction {nat / (s.injury_rate_100k / s.completeness):.2f}")
G["survey_rate"] = 1e5 * G.inj_w / G.pt
G["ncrb_rate"] = [ncrb_rate[r].get(u, np.nan) for r, u in zip(G["round"], G.unit)]
G["ratio_raw"] = G.ncrb_rate / G.survey_rate
G.to_csv(O / "killcheck_state.csv", index=False)

# ---- verdicts
tot = G.groupby("unit")[["inj_unw", "deaths_unw"]].sum().sort_values("inj_unw", ascending=False)
k2 = int((tot.inj_unw >= 100).sum())
big = G[G.inj_unw >= 100]
k3 = (big.ratio_raw > 1.2).mean()
print("\nInjury-flagged deaths in window by state (both rounds):\n", tot.head(36).to_string())
print("\nState ratios NCRB/survey (states with >=100 flagged deaths in that round):")
print(big.pivot_table(index="unit", columns="round", values="ratio_raw").round(2).to_string())
print(f"\nK1 completeness: {', '.join(f'{r} {v:.0%}' for r, v in zip(S['round'], S.completeness))}  (kill if < 60%)")
print(f"K2 states with >=100 flagged deaths (both rounds): {k2}  (kill if < 20)")
print(f"K3 share of state-rounds with raw NCRB/survey > 1.2: {k3:.0%}  (kill if most)")
