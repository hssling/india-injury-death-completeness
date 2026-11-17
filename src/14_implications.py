"""What the completeness estimates imply for users of police injury statistics (JECH revision, 2026-11-17).

1. Unrecorded injury deaths per year, 2017-2019: true resident injury deaths (posterior, main model) minus police deaths,
   national and by state; bracketed by the survey-exact and SRS-complete variants.
2. Split by sex: with national police deaths by sex (N_f, N_m), national true deaths T and the female-to-male completeness
   ratio R (national, unpooled: (N_f/N_m)/(survey_f/survey_m), lognormal with its SE), men's completeness is
   C_m = (N_f/R + N_m)/T and women's C_f = R*C_m.
3. Trend bias: change in the police injury death rate between rounds against the change in the model's true rate; the
   ratio of the two is the change in completeness. By state: states where police and true trends point in opposite
   directions with posterior probability > 0.9.
4. External convergence: GBD 2023 injury death rate by state (2020) as a third, modelled count, compared by rank
   correlation with the model's true resident rate and with the police rate; police/GBD nationally is a GBD-implied
   completeness. Ecological correlate: the Civil
   Registration System level of death registration (CRS 2018-2019 reports).
Outputs: outputs/implications.json, outputs/unrecorded_by_state.csv, outputs/trend_by_state.csv, outputs/convergence_state.csv
"""
import json, pathlib, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr

P = pathlib.Path(__file__).resolve().parents[1]; O = P / "outputs"
sys.path.insert(0, str(P / "src"))
from common import unit   # noqa: E402

YEARS = 3                                                   # 2017-2019 (NFHS-5 round): person-years span 3 calendar years
rng = np.random.default_rng(2026)
J = {}
q = lambda x: [float(np.median(x)), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def load(v):
    z = np.load(O / f"bayes_draws_{v}.npz", allow_pickle=False)    # written by 08_bayes_model.py (plain arrays, no pickle)
    return {k: z[k] for k in z.files}


def totals(z, r):
    m = z["round"] == r
    lam = np.exp(z["log_lam"][:, m]); py = z["person_years"][m]; nc = z["ncrb"][m]
    true = lam * py / 1e5                                     # draws x states: true resident injury deaths in the window
    return true, nc, py, z["unit"][m]


# ---------------- 1. unrecorded deaths, national and by state (2017-2019)
Z = load("main")
true5, nc5, py5, u5 = totals(Z, "nfhs5")
T = true5.sum(1) / YEARS; Npol = nc5.sum() / YEARS
J["police_per_year"] = float(Npol)
J["true_per_year"] = q(T)
J["unrecorded_per_year"] = q(T - Npol)
J["unrecorded_share"] = q(1 - Npol / T)
for v in ("roster_exact", "srs_complete"):
    t, n, _, _ = totals(load(v), "nfhs5")
    J[f"unrecorded_per_year_{v}"] = float(np.median(t.sum(1) / YEARS - n.sum() / YEARS))
miss = (true5 - nc5) / YEARS
S = pd.DataFrame({"unit": u5, "police_per_year": nc5 / YEARS, "true_med": np.median(true5, 0) / YEARS,
                  "unrecorded_med": np.median(miss, 0), "unrecorded_lo": np.percentile(miss, 2.5, 0),
                  "unrecorded_hi": np.percentile(miss, 97.5, 0), "P_unrecorded_pos": (miss > 0).mean(0)})
S = S.sort_values("unrecorded_med", ascending=False)
S.to_csv(O / "unrecorded_by_state.csv", index=False)
pos = S[S.unrecorded_med > 0]
J["n_states_net_unrecorded"] = int(len(pos))
J["top5_states_unrecorded"] = [[r.unit, float(r.unrecorded_med), float(r.unrecorded_lo), float(r.unrecorded_hi)] for r in S.head(5).itertuples()]
J["top5_share_of_net_unrecorded"] = float(S.head(5).unrecorded_med.sum() / S.unrecorded_med.sum())
J["up_share_of_net_unrecorded"] = float(S.set_index("unit").loc["UTTAR PRADESH", "unrecorded_med"] / S.unrecorded_med.sum())

# ---------------- 2. split by sex
sx = pd.read_csv(O / "sex_ratio_national_raw.csv").set_index("round").loc["nfhs5"]
fshare = sx.ncrb_female / (sx.ncrb_female + sx.ncrb_male)
Nf, Nm = Npol * fshare, Npol * (1 - fshare)                 # police deaths per year by sex, scaled to the modelled states
R = np.exp(rng.normal(sx.y, sx.se, size=len(T)))
Cm = (Nf / R + Nm) / T; Cf = R * Cm
J["female_share_police"] = float(fshare)
J["R_national_raw"] = q(R)
J["C_men"] = q(Cm); J["C_women"] = q(Cf)
J["unrecorded_women_per_year"] = q(Nf / Cf - Nf); J["unrecorded_men_per_year"] = q(Nm / Cm - Nm)
J["female_share_unrecorded"] = q((Nf / Cf - Nf) / (T - Npol))
J["female_share_true"] = q((Nf / Cf) / T)
# tipping point: female/male ratio of the injury-specific survey reporting factor that would explain the whole gap
pool = pd.read_csv(O / "sex_ratio_pooled.csv").set_index("round").loc["nfhs5"]
J["tip_rho_ratio_null"] = float(1 / pool.R_med); J["tip_rho_ratio_ci"] = float(1 / pool.R_hi)

# ---------------- 3. trend bias between rounds (same states in both rounds)
true4, nc4, py4, u4 = totals(Z, "nfhs4")
assert list(u4) == list(u5) or set(u4) == set(u5)
o4 = {u: i for i, u in enumerate(u4)}; idx = np.array([o4[u] for u in u5])
true4, nc4, py4 = true4[:, idx], nc4[idx], py4[idx]
true_ratio = (true5.sum(1) / py5.sum()) / (true4.sum(1) / py4.sum())
pol_ratio = (nc5.sum() / py5.sum()) / (nc4.sum() / py4.sum())
J["police_rate_ratio"] = float(pol_ratio); J["true_rate_ratio"] = q(true_ratio)
J["completeness_ratio"] = q(pol_ratio / true_ratio)
J["P_true_decline"] = float((true_ratio < 1).mean())
J["masked_share_of_decline"] = float((pol_ratio - np.median(true_ratio)) / (1 - np.median(true_ratio)))
vr = {}
for f in sorted(O.glob("bayes_national_*.csv")):
    n = pd.read_csv(f).set_index("round")
    vr[f.stem.replace("bayes_national_", "")] = float(n.loc["nfhs5", "true_inj_rate_med"] / n.loc["nfhs4", "true_inj_rate_med"])
J["true_rate_ratio_variants"] = vr
tr = (true5 / py5) / (true4 / py4); pr = (nc5 / py5) / (nc4 / py4)
TR = pd.DataFrame({"unit": u5, "police_ratio": pr, "true_ratio_med": np.median(tr, 0), "true_lo": np.percentile(tr, 2.5, 0),
                   "true_hi": np.percentile(tr, 97.5, 0), "P_true_decline": (tr < 1).mean(0)})
TR["opposite"] = ((TR.police_ratio > 1) & (TR.P_true_decline > 0.9)) | ((TR.police_ratio < 1) & (TR.P_true_decline < 0.1))
TR.sort_values("police_ratio").to_csv(O / "trend_by_state.csv", index=False)
J["n_states_trend"] = int(len(TR))
J["n_opposite"] = int(TR.opposite.sum())
J["opposite_states"] = [[r.unit, float(r.police_ratio), float(r.true_ratio_med)] for r in TR[TR.opposite].itertuples()]
J["n_police_rise"] = int((TR.police_ratio > 1).sum())
J["n_true_decline_090"] = int((TR.P_true_decline > 0.9).sum())

# ---------------- 4. convergence with GBD, and civil registration
C = pd.read_csv(O / "bayes_completeness_main.csv").query("round == 'nfhs5'").set_index("unit")
nw = pd.read_csv(O / "ncrb_window.csv").query("round == 'nfhs5'").set_index("unit")
g = pd.read_csv(P / "data/gbd_2023_extract.csv").query("cause_name == 'Injuries' and year == 2020")
g = g[~g.location_name.isin(["India", "Other Union Territories"])].assign(unit=lambda x: x.location_name.map(unit)).set_index("unit").val
crs = pd.read_csv(P / "data/crs_srs_state_year.csv").query("crs_year in [2018, 2019] and state != 'India'")
lor = crs.assign(unit=crs.state.map(unit)).groupby("unit").lor_deaths.mean()
mig = pd.read_csv(P / "data/plfs_inmigration_2020_21.csv").set_index("unit").inmig_share
V = pd.DataFrame({"C_med": C.C_med, "deaths_inj": C.deaths_inj, "police_rate": 1e5 * nw.ncrb / nw.person_years,
                  "gbd_rate": g, "lor_deaths": lor, "inmig_share": mig}).dropna(subset=["C_med"])
# The true resident rate from the model is compared with GBD directly. (Correlating C with police/GBD would share the
# police rate in both numerators and inflate the agreement: mathematical coupling.)
V["true_rate"] = pd.Series(np.median(true5 / py5 * 1e5, 0), index=u5)
V["C_gbd"] = V.police_rate / V.gbd_rate
V.to_csv(O / "convergence_state.csv")
gg = V.dropna(subset=["gbd_rate"])
J["n_gbd"] = int(len(gg))
J["rho_true_gbd"] = float(spearmanr(gg.true_rate, gg.gbd_rate)[0]); J["p_true_gbd"] = float(spearmanr(gg.true_rate, gg.gbd_rate)[1])
J["rho_police_gbd"] = float(spearmanr(gg.police_rate, gg.gbd_rate)[0])
J["C_gbd_india"] = float((nc5.sum() / py5.sum() * 1e5) / pd.read_csv(P / "data/gbd_2023_extract.csv")
                         .query("cause_name == 'Injuries' and year == 2020 and location_name == 'India'").val.iloc[0])
J["true_over_gbd_median_state"] = float((gg.true_rate / gg.gbd_rate).median())
ll = V.dropna(subset=["lor_deaths"])
J["n_lor"] = int(len(ll)); J["rho_lor"] = float(spearmanr(ll.C_med, ll.lor_deaths)[0]); J["p_lor"] = float(spearmanr(ll.C_med, ll.lor_deaths)[1])
mm = V.dropna(subset=["inmig_share"])
J["rho_mig"] = float(spearmanr(mm.C_med, mm.inmig_share)[0])
(O / "implications.json").write_text(json.dumps(J, indent=1), encoding="utf-8")
for k, v in J.items():
    print(k, v if not isinstance(v, float) else round(v, 3))
