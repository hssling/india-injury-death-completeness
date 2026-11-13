"""Are injury deaths more likely to be registered with the civil authority? (NFHS-5, sh91 x sh95)

Deaths Jan 2017 - Feb 2020 (pre-COVID), injury flag yes/no (don't know excluded), registration yes/no.
Modified Poisson (log link) with survey weights (normalised hv005) and PSU-clustered robust SEs; prevalence ratio (PR)
for injury vs other deaths:
  M0 crude; M1 + age group, sex; M2 + residence, wealth quintile, year of death; M3 + state fixed effects.
Adjusted risk difference by marginal standardisation from M3. Heterogeneity: PR by sex, by age group, by residence.
"""
import pathlib, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
P = pathlib.Path(__file__).resolve().parents[1]
d = pd.read_parquet(P / "data_local/nfhs5_deaths.parquet")
d = d[(d.yr < 9000)].copy()
d["dcmc"] = (d.yr - 1900) * 12 + d.mo.where(d.mo <= 12, 6)
d = d[(d.dcmc >= (2017 - 1900) * 12 + 1) & (d.dcmc < (2020 - 1900) * 12 + 3) & d.inj.isin([0, 1]) & d.reg.isin([0, 1])
      & d.sex.isin([1, 2])].copy()
age = np.where(d.age_u == 3, d.age_n, np.where(d.age_u.isin([1, 2]), 0, np.nan))
age = np.where(age >= 98, np.nan, age)
d["agegrp"] = pd.cut(age, [-1, 4, 14, 29, 44, 59, 74, 200], labels=["0-4", "5-14", "15-29", "30-44", "45-59", "60-74", "75+"])
d["age"] = age
d = d[d.agegrp.notna()].copy()
# main analysis excludes deaths under 1 year (implausible 7% injury flag among neonates); SAMPLE=all keeps them
import os
SAMPLE = os.environ.get("SAMPLE", "age1plus")
if SAMPLE == "age1plus":
    d = d[d.age >= 1].copy()
    d["agegrp"] = d.agegrp.cat.rename_categories({"0-4": "1-4"})
d["w"] = d.hv005 / d.hv005.mean()
d["female"] = (d.sex == 2).astype(int)
d["rural"] = (d.hv025 == 2).astype(int)
d["injury"] = d.inj.astype(int)
d["reg"] = d.reg.astype(int)
d["dyear"] = d.yr.astype(int).clip(upper=2019)          # Jan-Feb 2020 pooled with 2019
print(f"deaths analysed: {len(d):,}; injury {d.injury.sum():,}; weighted registration injury "
      f"{np.average(d.reg[d.injury == 1], weights=d.w[d.injury == 1]):.3f} vs other {np.average(d.reg[d.injury == 0], weights=d.w[d.injury == 0]):.3f}")


def fit(f, data=d):
    m = smf.glm(f, data=data, family=sm.families.Poisson(), var_weights=data.w)
    return m.fit(cov_type="cluster", cov_kwds={"groups": data.hv021.astype(int).values + 10 ** 6 * data.hv024.astype(int).values})


F = {"M0 crude": "reg ~ injury",
     "M1 + age, sex": "reg ~ injury + C(agegrp) + female",
     "M2 + residence, wealth, year": "reg ~ injury + C(agegrp) + female + rural + C(hv270) + C(dyear)",
     "M3 + state FE": "reg ~ injury + C(agegrp) + female + rural + C(hv270) + C(dyear) + C(hv024)"}
rows = []
for k, f in F.items():
    r = fit(f)
    b, se = r.params["injury"], r.bse["injury"]
    rows.append(dict(model=k, PR=np.exp(b), lo=np.exp(b - 1.96 * se), hi=np.exp(b + 1.96 * se), p=r.pvalues["injury"]))
    if k.startswith("M3"):
        m3 = r
R = pd.DataFrame(rows)
print(R.round(3).to_string(index=False))
# marginal standardisation: predicted registration if all deaths were injury vs if none
p1 = np.average(m3.predict(d.assign(injury=1)), weights=d.w)
p0 = np.average(m3.predict(d.assign(injury=0)), weights=d.w)
print(f"standardised registration: injury {p1:.3f} vs other {p0:.3f}; difference {100 * (p1 - p0):.1f} pp")
import json
json.dump({"n_deaths": int(len(d)), "n_injury": int(d.injury.sum()),
           "reg_injury_crude": float(np.average(d.reg[d.injury == 1], weights=d.w[d.injury == 1])),
           "reg_other_crude": float(np.average(d.reg[d.injury == 0], weights=d.w[d.injury == 0])),
           "std_injury": float(p1), "std_other": float(p0), "std_diff_pp": float(100 * (p1 - p0))},
          open(P / f"outputs/registration_summary_{SAMPLE}.json", "w"), indent=1)
# heterogeneity (M3 within strata)
het = []
for name, col in [("sex", "female"), ("residence", "rural"), ("age", "agegrp")]:
    for lev, g in d.groupby(col, observed=True):
        if g.injury.sum() < 100:
            continue
        f = F["M3 + state FE"].replace(" + female", "" if col == "female" else " + female").replace(" + rural", "" if col == "rural" else " + rural")
        if col == "agegrp":
            f = f.replace(" + C(agegrp)", "")
        r = fit(f, g)
        b, se = r.params["injury"], r.bse["injury"]
        het.append(dict(by=name, level=lev, n_inj=int(g.injury.sum()), reg_other=np.average(g.reg[g.injury == 0], weights=g.w[g.injury == 0]),
                        PR=np.exp(b), lo=np.exp(b - 1.96 * se), hi=np.exp(b + 1.96 * se)))
H = pd.DataFrame(het)
print(H.round(3).to_string(index=False))
R.to_csv(P / f"outputs/registration_premium_{SAMPLE}.csv", index=False)
H.to_csv(P / f"outputs/registration_premium_by_group_{SAMPLE}.csv", index=False)
