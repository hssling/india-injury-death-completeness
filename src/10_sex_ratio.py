"""Female-to-male ratio of police completeness, by state and round.

R_s = (NCRB_F / NCRB_M) / (SURVEY_F / SURVEY_M), with survey = injury death rates (deaths at age >= 1 per de jure
person-year of that sex). Household reporting (k), SRS coverage (c) and injury-specific reporting (rho) cancel if they do
not differ by sex, so R is identified without the bracketing assumptions needed for the level of C.
NCRB injury deaths by sex: other-causes Total - Sudden Deaths - Deaths of Women during Pregnancy (female) + forces of
nature (excl. epidemic, starvation) + suicides + murder cases split by the state's victim sex share (CII Table 2A.3:
2019 shares for 2017-19, 2016 shares for 2016; national share where a state has no victims).
Survey SEs by linearisation for each sex; the SE of the log ratio assumes independence between sexes (conservative:
the same PSUs give positive covariance, which would shrink the true SE).
Hierarchical model: y_sr ~ N(theta_r + w_s, se_sr), w_s ~ N(0, tau). Output: outputs/sex_ratio_*.csv
"""
import pathlib, sys
import numpy as np, pandas as pd
import jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

numpyro.set_host_device_count(4)
P = pathlib.Path(__file__).resolve().parents[1]; M = P.parent; O = P / "outputs"; D = P / "data_local"
sys.path.insert(0, str(P / "src"))
from common import state_labels, unit, cmc, ratio_se, nfhs   # noqa: E402

ROUNDS = {"nfhs5": dict(zip=nfhs("nfhs5")[0], do=nfhs("nfhs5")[1], w0=cmc(2017, 1), w1=cmc(2020, 3), ny=[2017, 2018, 2019]),
          "nfhs4": dict(zip=nfhs("nfhs4")[0], do=nfhs("nfhs4")[1], w0=cmc(2013, 1), w1=cmc(2017, 1), ny=[2016])}

# ---------------- survey, by sex
if (D / "nfhs5_member_sex.parquet").exists():          # NFHS unit-level data (not redistributable)
    rows = []
    for r, c in ROUNDS.items():
        lab = state_labels(c["zip"], c["do"])
        hh = pd.read_parquet(D / f"{r}_households.parquet").merge(pd.read_parquet(D / f"{r}_member_sex.parquet"), on="hhid")
        de = pd.read_parquet(D / f"{r}_deaths.parquet")
        hh["unit"] = hh.hv024.map(lab).map(unit)
        end = np.minimum(hh.hv008, c["w1"]); yrs = np.clip(end - c["w0"], 0, None) / 12
        de = de[de.yr < 9000].copy(); de["dcmc"] = cmc(de.yr, de.mo.where(de.mo <= 12, 6))
        de = de.merge(pd.DataFrame({"hhid": hh.hhid, "end": end}), on="hhid")
        de = de[(de.dcmc >= c["w0"]) & (de.dcmc < de.end)]
        age = np.where(de.age_u == 3, de.age_n, np.where(de.age_u.isin([1, 2]), 0, np.nan))
        de = de[(de.inj == 1) & (age >= 1) & (age < 98)]
        hh = hh.set_index("hhid"); hh["w"] = hh.hv005 / 1e6
        hh["psu"] = hh.hv021.astype(int); hh["h"] = hh.hv022.astype(int) * 100 + hh.hv024.astype(int)
        for sx, col in [("male", "n_male"), ("female", "n_female")]:
            hh["x"] = hh[col] * yrs.values
            hh["y"] = de[de.sex == (1 if sx == "male" else 2)].groupby("hhid").size().reindex(hh.index, fill_value=0)
            for u, g in list(hh.groupby("unit")) + [("INDIA", hh)]:
                R, se = ratio_se(g.y, g.x, g.w, g.psu, g.h)
                rows.append(dict(round=r, unit=u, sex=sx, deaths=int(g.y.sum()), rate=1e5 * R, log_se=se / R if R > 0 else np.nan))
    S = pd.DataFrame(rows)
    S.to_csv(O / "survey_by_sex.csv", index=False)
else:                                                      # public repository: shipped aggregates
    S = pd.read_csv(O / "survey_by_sex.csv")

# ---------------- NCRB injury deaths by sex
adsi = pd.read_csv(P / "data/adsi_state_year_sex.csv")
adsi = adsi[adsi.sex.isin(["male", "female"])]
oc = adsi[adsi.table == "other_causes"].pivot_table(index=["unit", "year", "sex"], columns="cause", values="n", aggfunc="sum").fillna(0)
acc = oc["Total"] - oc["Sudden Deaths (Total)"] - oc["Deaths of Women during Pregnancy (Total)"]
fon = adsi[(adsi.table == "forces_of_nature") & ~adsi.cause.isin(["Epidemic", "Starvation due to Natural Calamity"])].groupby(["unit", "year", "sex"]).n.sum()
sui = pd.read_csv(P / "data/ncrb_suicides_state_year.csv")
sui = sui.melt(id_vars=["unit", "year"], value_vars=["male", "female"], var_name="sex", value_name="n").groupby(["unit", "year", "sex"]).n.sum()
mur = pd.read_csv(P / "data/cii_murders_state_year.csv")
vs = pd.read_csv(P / "data/cii_murder_victims_sex.csv")
nat_share = {y: g.male.sum() / (g.male + g.female).sum() for y, g in vs.groupby("year")}
vs["male_share"] = vs.male_share.fillna(vs.year.map(nat_share))
mur["vy"] = np.where(mur.year >= 2017, 2019, 2016)
mur = mur.merge(vs[["unit", "year", "male_share"]].rename(columns={"year": "vy"}), on=["unit", "vy"], how="left")
mur["male_share"] = mur.male_share.fillna(mur.vy.map(nat_share))
mm = pd.concat([mur.assign(sex="male", n=mur.murders * mur.male_share), mur.assign(sex="female", n=mur.murders * (1 - mur.male_share))])
mm = mm.groupby(["unit", "year", "sex"]).n.sum()
N = pd.DataFrame({"acc": acc, "fon": fon, "sui": sui, "mur": mm}).dropna(subset=["acc", "sui", "mur"])
N["fon"] = N.fon.fillna(0)
N["ncrb"] = N.acc + N.fon + N.sui + N.mur
N = N.reset_index()
nrows = []
for r, c in ROUNDS.items():
    x = N[N.year.isin(c["ny"])].groupby(["unit", "sex"]).ncrb.sum().reset_index().assign(round=r)
    assert x.unit.nunique() == 35, r
    nrows.append(x)
NC = pd.concat(nrows)
NC = pd.concat([NC, NC.groupby(["round", "sex"]).ncrb.sum().reset_index().assign(unit="INDIA")])
for r in ROUNDS:
    t = NC[(NC["round"] == r) & (NC.unit == "INDIA")].set_index("sex").ncrb
    print(f"{r}: NCRB injury deaths male {t['male']:,.0f} female {t['female']:,.0f} (female share {t['female'] / t.sum():.3f})")

# ---------------- ratio and hierarchical model
W = S.pivot_table(index=["round", "unit"], columns="sex", values=["rate", "log_se", "deaths"])
W.columns = [f"{a}_{b}" for a, b in W.columns]
W = W.reset_index().merge(NC.pivot_table(index=["round", "unit"], columns="sex", values="ncrb").add_prefix("ncrb_").reset_index(), on=["round", "unit"])
W["y"] = np.log(W.ncrb_female / W.ncrb_male) - np.log(W.rate_female / W.rate_male)
W["se"] = np.sqrt(W.log_se_female ** 2 + W.log_se_male ** 2 + 1 / W.ncrb_female + 1 / W.ncrb_male)
nat = W[W.unit == "INDIA"].copy(); nat["ratio_raw"] = np.exp(nat.y)
print(nat[["round", "deaths_female", "deaths_male", "rate_female", "rate_male", "ratio_raw", "se"]].round(3).to_string(index=False))
d = W[(W.unit != "INDIA") & (W.deaths_female >= 10) & (W.deaths_male >= 10)].reset_index(drop=True)
units = sorted(d.unit.unique()); si = d.unit.map({u: i for i, u in enumerate(units)}).values
ri = (d["round"] == "nfhs5").astype(int).values


def model(y, se):
    th = numpyro.sample("theta", dist.Normal(0, 1).expand([2]))
    tau = numpyro.sample("tau", dist.HalfNormal(0.5))
    w = tau * numpyro.sample("zw", dist.Normal(0, 1).expand([len(units)]))
    mu = numpyro.deterministic("mu", th[ri] + w[si])
    numpyro.sample("obs", dist.Normal(mu, se), obs=y)


mc = MCMC(NUTS(model, target_accept_prob=0.95), num_warmup=1000, num_samples=2000, num_chains=4, progress_bar=False)
mc.run(jax.random.PRNGKey(7), y=jnp.array(d.y.values), se=jnp.array(d.se.values))
f = {k: np.asarray(v) for k, v in mc.get_samples().items()}
from numpyro.diagnostics import summary
s = summary(mc.get_samples(group_by_chain=True))
print("max R-hat", round(max(float(np.max(v["r_hat"])) for v in s.values()), 3), "divergences", int(mc.get_extra_fields()["diverging"].sum()))
th = np.exp(f["theta"])
for j, r in enumerate(["nfhs4", "nfhs5"]):
    print(f"{r}: pooled female/male completeness ratio {np.median(th[:, j]):.2f} ({np.percentile(th[:, j], 2.5):.2f}-{np.percentile(th[:, j], 97.5):.2f})")
print(f"tau (between-state SD, log) {np.median(f['tau']):.2f} ({np.percentile(f['tau'], 2.5):.2f}-{np.percentile(f['tau'], 97.5):.2f})")
mu = np.exp(f["mu"])
d["R_med"], d["R_lo"], d["R_hi"] = np.median(mu, 0), np.percentile(mu, 2.5, 0), np.percentile(mu, 97.5, 0)
d["P_R_below_1"] = (mu < 1).mean(0)
d.to_csv(O / "sex_ratio_state.csv", index=False)
nat.to_csv(O / "sex_ratio_national_raw.csv", index=False)
pd.DataFrame({"round": ["nfhs4", "nfhs5"], "R_med": np.median(th, 0), "R_lo": np.percentile(th, 2.5, 0),
              "R_hi": np.percentile(th, 97.5, 0)}).to_csv(O / "sex_ratio_pooled.csv", index=False)
print(d[d["round"] == "nfhs5"].sort_values("R_med")[["unit", "deaths_female", "deaths_male", "R_med", "R_lo", "R_hi", "P_R_below_1"]].round(2).to_string(index=False))
