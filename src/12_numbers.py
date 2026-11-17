"""Every number quoted in the manuscript, computed from outputs/ and data/ (never typed). Writes outputs/numbers.json.

Keys are referenced as {key} in manuscript/manuscript_template.md and filled by 13_build_manuscript.py.
Formatting: f2 = 2 decimals, ci = "a (b–c)", pct = percent with 1 decimal, n = thousands separator.
"""
import json, pathlib, sys
import numpy as np, pandas as pd

P = pathlib.Path(__file__).resolve().parents[1]; M = P.parent; O = P / "outputs"; D = P / "data_local"
sys.path.insert(0, str(P / "src"))
from common import state_labels, unit, SRS_INDIA_2019   # noqa: E402
N = {}
f2 = lambda x: f"{x:.2f}"
ci = lambda m, l, h: f"{m:.2f} ({l:.2f}–{h:.2f})"
n0 = lambda x: f"{int(round(x)):,}"
nice = lambda u: {"JAMMU & KASHMIR (incl. LADAKH)": "Jammu and Kashmir (including Ladakh)", "DELHI (UT)": "Delhi",
                  "A & N ISLANDS": "Andaman and Nicobar Islands", "DNH & DD": "Dadra and Nagar Haveli and Daman and Diu"}.get(u, u.title())

# ---- survey
S = pd.read_csv(O / "survey_estimates_main.csv"); Sa = pd.read_csv(O / "survey_estimates_allages.csv")
R7 = pd.read_csv(O / "survey_estimates_recall7.csv")
kc = pd.read_csv(O / "killcheck_national.csv").set_index("round")
for r in ("nfhs4", "nfhs5"):
    s = S[(S["round"] == r) & (S.unit == "INDIA")].iloc[0]; sa = Sa[(Sa["round"] == r) & (Sa.unit == "INDIA")].iloc[0]
    r7 = R7[(R7["round"] == r) & (R7.unit == "INDIA")].iloc[0]
    N[f"hh_{r}"] = n0(s.n_hh); N[f"psu_{r}"] = n0(s.n_psu)
    N[f"deaths_{r}"] = n0(s.deaths_all); N[f"inj_deaths_{r}"] = n0(s.deaths_inj); N[f"inj_deaths_allages_{r}"] = n0(sa.deaths_inj)
    N[f"cdr_{r}"] = f2(s.cdr); N[f"inj_rate_{r}"] = f"{s.inj_rate:.1f}"; N[f"inj_rate_allages_{r}"] = f"{sa.inj_rate:.1f}"
    N[f"srs_cdr_{r}"] = f2(kc.loc[r, "srs_cdr"]); N[f"roster_srs_pct_{r}"] = f"{100 * kc.loc[r, 'completeness']:.0f}"
    N[f"cdr_recall7_{r}"] = f2(r7.cdr); N[f"inj_rate_recall7_{r}"] = f"{r7.inj_rate:.1f}"
    N[f"inj_share_pct_{r}"] = f"{100 * sa.deaths_inj / s.deaths_all:.1f}"
N["hh_total"] = n0(S[S.unit == "INDIA"].n_hh.sum()); N["deaths_total"] = n0(S[S.unit == "INDIA"].deaths_all.sum())
N["inj_deaths_total"] = n0(S[S.unit == "INDIA"].deaths_inj.sum())

# ---- roster data quality (07b_roster_quality.py; aggregates of the NFHS rosters)
N.update(json.loads((O / "roster_quality.json").read_text(encoding="utf-8")))

# ---- NCRB
NW = pd.read_csv(O / "ncrb_window.csv")
for r in ("nfhs4", "nfhs5"):
    x = NW[NW["round"] == r]
    a, b = x.years.iloc[0].split("-")
    N[f"ncrb_total_{r}"] = n0(x.ncrb.sum()); N[f"ncrb_years_{r}"] = a if a == b else f"{a}–{b}"
    N[f"ncrb_rate_{r}"] = f"{1e5 * x.ncrb.sum() / x.person_years.sum():.1f}"
adsi = pd.read_csv(P / "data/adsi_state_year_sex.csv")
adsi = adsi[(adsi.sex == "total") & (adsi.table == "other_causes")]
sd = adsi[adsi.cause == "Sudden Deaths (Total)"].groupby("year").n.sum(); tot = adsi[adsi.cause == "Total"].groupby("year").n.sum()
N["sudden_2016"] = n0(sd[2016]); N["sudden_2024"] = n0(sd[2024]); N["sudden_2019"] = n0(sd[2019])
N["sudden_share_2016_pct"] = f"{100 * sd[2016] / tot[2016]:.0f}"; N["sudden_share_2024_pct"] = f"{100 * sd[2024] / tot[2024]:.1f}"
mur = pd.read_csv(P / "data/cii_murders_state_year.csv")
N["murders_2019"] = n0(mur[mur.year == 2019].murders.sum())

# ---- SRS / GBD
crs = pd.read_csv(P / "data/crs_srs_state_year.csv")
N["srs_india_2019"] = f"{SRS_INDIA_2019:.1f}"
gbd = pd.read_csv(P / "data/gbd_2023_extract.csv")
g = gbd[(gbd.cause_name == "All causes") & (gbd.year == 2019) & (gbd.location_name == "India")].val.iloc[0]
N["gbd_cdr_2019"] = f2(g / 100)
N["srs_cov_prior"] = f"{SRS_INDIA_2019 / (g / 100):.2f}"        # SRS/GBD all-cause, India 2019 = prior centre in 08 (0.90)
gi = gbd[(gbd.cause_name == "Injuries") & (gbd.year == 2020)].set_index("location_name").val
N["gbd_inj_india_2020"] = f"{gi['India']:.1f}"; N["gbd_inj_telangana_2020"] = f"{gi['Telangana']:.0f}"

# ---- model: national, variants, parameters
V = ["main", "srs_complete", "roster_exact", "rho_wide", "recall7", "allages", "min30", "ncrb_strict", "ncrb_2014_16", "migration"]
nat = {v: pd.read_csv(O / f"bayes_national_{v}.csv").set_index("round") for v in V}
for r in ("nfhs4", "nfhs5"):
    m = nat["main"].loc[r]
    N[f"C_nat_{r}"] = ci(m.C_med, m.C_lo, m.C_hi); N[f"C_nat_med_{r}"] = f2(m.C_med)
    N[f"C_srs_complete_{r}"] = f2(nat["srs_complete"].loc[r, "C_med"]); N[f"C_roster_exact_{r}"] = f2(nat["roster_exact"].loc[r, "C_med"])
    meds = [nat[v].loc[r, "C_med"] for v in V if v != "rho_wide"]
    N[f"C_nat_min_{r}"] = f2(min(meds)); N[f"C_nat_max_{r}"] = f2(max(meds))
    N[f"true_inj_rate_{r}"] = f"{nat['main'].loc[r, 'true_inj_rate_med']:.0f}"
diag = [open(O / f"bayes_diag_{v}.txt").read() for v in V]
import re
N["max_rhat"] = f"{max(float(re.search(r'R-hat ([0-9.]+)', d).group(1)) for d in diag):.3f}"
N["min_ess"] = n0(min(float(re.search(r'ESS ([0-9.]+)', d).group(1)) for d in diag))
N["max_div"] = str(max(int(re.search(r'divergences (\d+)', d).group(1)) for d in diag))
N["n_variants"] = str(len(V))
pm = pd.read_csv(O / "bayes_params_main.csv").set_index("param")
for r in ("nfhs4", "nfhs5"):
    N[f"k_{r}"] = ci(pm.loc[f"kap_{r}", "exp_med"], pm.loc[f"kap_{r}", "exp_lo"], pm.loc[f"kap_{r}", "exp_hi"])
N["srs_cov"] = ci(pm.loc["gam", "exp_med"], pm.loc["gam", "exp_lo"], pm.loc["gam", "exp_hi"])
N["sig_u"] = ci(pm.loc["sig_u", "med"], pm.loc["sig_u", "lo"], pm.loc["sig_u", "hi"])
pg = pd.read_csv(O / "bayes_params_migration.csv").set_index("param")
N["beta_mig"] = ci(pg.loc["beta_mig", "med"], pg.loc["beta_mig", "lo"], pg.loc["beta_mig", "hi"])

# ---- states
C = pd.read_csv(O / "bayes_completeness_main.csv"); c5 = C[C["round"] == "nfhs5"].set_index("unit")
N["n_units"] = str(c5.shape[0]); N["n_units_both"] = str(C.groupby("unit").size().eq(2).sum())
lo5 = c5.sort_values("C_med")
N["lowest_states"] = "; ".join(f"{nice(u)} {ci(r.C_med, r.C_lo, r.C_hi)}" for u, r in lo5.head(6).iterrows())
N["n_p_below_half_090"] = str(int((c5["P_C_below_0.5"] > 0.9).sum()))
above = lo5[lo5.C_med > 1.1].sort_values("C_med", ascending=False)
N["above_one_states"] = ", ".join(f"{nice(u)} {r.C_med:.2f}" for u, r in above.iterrows())
big = c5[c5.deaths_inj >= 100]
N["ratio_hi_lo_big"] = f"{big.C_med.max() / big.C_med.min():.1f}"
N["min_big"] = f"{nice(big.C_med.idxmin())} ({big.C_med.min():.2f})"; N["max_big"] = f"{nice(big.C_med.idxmax())} ({big.C_med.max():.2f})"
rc = []
for v in V[1:]:
    x = pd.read_csv(O / f"bayes_completeness_{v}.csv"); x = x[x["round"] == "nfhs5"].set_index("unit").C_med
    j = c5.index.intersection(x.index); rc.append(c5.C_med[j].rank().corr(x[j].rank()))
N["min_rank_corr"] = f"{min(rc):.2f}"
G = pd.read_csv(O / "bayes_completeness_migration.csv"); g5 = G[G["round"] == "nfhs5"].set_index("unit")
for u, k in [("DELHI (UT)", "delhi"), ("HARYANA", "haryana"), ("BIHAR", "bihar")]:
    N[f"{k}_C"] = f2(c5.loc[u, "C_med"]); N[f"{k}_Crec"] = f2(g5.loc[u, "Crec_med"])
both_low = [u for u in c5.index if c5.loc[u, "C_med"] < 0.6 and g5.loc[u, "Crec_med"] < 0.6]
N["robust_low_states"] = ", ".join(nice(u) for u in c5.loc[both_low].sort_values("C_med").index)
N["n_robust_low"] = str(len(both_low))
mg = pd.read_csv(P / "data/plfs_inmigration_2020_21.csv").set_index("unit")
N["inmig_delhi_pct"] = f"{100 * mg.loc['DELHI (UT)', 'inmig_share']:.0f}"
ch = pd.read_csv(O / "bayes_change_main.csv")
up = ch[ch.P_increase > 0.975].sort_values("ratio_med", ascending=False); dn = ch[ch.P_increase < 0.025]
N["n_increase"] = str(len(up)); N["n_decrease"] = str(len(dn))
N["increase_states"] = ", ".join(f"{nice(r.unit)} ×{r.ratio_med:.2f}" for r in up.itertuples())
N["decrease_states"] = ", ".join(f"{nice(r.unit)} ×{r.ratio_med:.2f}" for r in dn.itertuples())

# ---- sex
sp = pd.read_csv(O / "sex_ratio_pooled.csv").set_index("round")
for r in ("nfhs4", "nfhs5"):
    N[f"R_sex_{r}"] = ci(sp.loc[r, "R_med"], sp.loc[r, "R_lo"], sp.loc[r, "R_hi"])
sr = pd.read_csv(O / "sex_ratio_state.csv"); s5 = sr[sr["round"] == "nfhs5"].sort_values("R_med")
N["n_states_sex"] = str(len(s5)); N["n_sex_P975"] = str(int((s5.P_R_below_1 > 0.975).sum()))
N["sex_lowest"] = "; ".join(f"{nice(r.unit)} {r.R_med:.2f}" for r in s5.head(4).itertuples())
sn = pd.read_csv(O / "sex_ratio_national_raw.csv").set_index("round")
for r in ("nfhs4", "nfhs5"):
    N[f"inj_rate_f_{r}"] = f"{sn.loc[r, 'rate_female']:.1f}"; N[f"inj_rate_m_{r}"] = f"{sn.loc[r, 'rate_male']:.1f}"

# ---- registration
for tag, f in [("", "registration_premium_age1plus.csv"), ("_all", "registration_premium_all.csv")]:
    R = pd.read_csv(O / f).set_index("model")
    N[f"pr_crude{tag}"] = ci(R.loc["M0 crude", "PR"], R.loc["M0 crude", "lo"], R.loc["M0 crude", "hi"])
    N[f"pr_adj{tag}"] = ci(R.loc["M3 + state FE", "PR"], R.loc["M3 + state FE", "lo"], R.loc["M3 + state FE", "hi"])
rs = json.load(open(O / "registration_summary_age1plus.json"))
N["reg_n"] = n0(rs["n_deaths"]); N["reg_n_inj"] = n0(rs["n_injury"])
N["reg_inj_pct"] = f"{100 * rs['reg_injury_crude']:.1f}"; N["reg_oth_pct"] = f"{100 * rs['reg_other_crude']:.1f}"
N["reg_std_inj_pct"] = f"{100 * rs['std_injury']:.1f}"; N["reg_std_oth_pct"] = f"{100 * rs['std_other']:.1f}"
N["reg_std_diff"] = f"{rs['std_diff_pp']:.1f}"
H = pd.read_csv(O / "registration_premium_by_group_age1plus.csv")
hv = lambda by, lv: H[(H.by == by) & (H.level.astype(str) == lv)].iloc[0]
N["pr_rural"] = f2(hv("residence", "1").PR); N["pr_urban"] = f2(hv("residence", "0").PR)
for nm, (by, lv) in {"rural": ("residence", "1"), "urban": ("residence", "0"), "female": ("sex", "1"), "male": ("sex", "0")}.items():
    h = hv(by, lv); N[f"pr_{nm}_ci"] = ci(h.PR, h.lo, h.hi)
N["pr_5_14"] = f2(hv("age", "5-14").PR); N["pr_15_29"] = f2(hv("age", "15-29").PR)

# ---- implications (14_implications.py): unrecorded deaths, sex split, trend bias, external comparison
I = json.loads((O / "implications.json").read_text(encoding="utf-8"))
k0 = lambda x: f"{1000 * round(x / 1000):,}"                    # round to the nearest thousand
kci = lambda v: f"{k0(v[0])} ({k0(v[1])}–{k0(v[2])})"
N["police_per_year"] = k0(I["police_per_year"]); N["true_per_year"] = kci(I["true_per_year"])
N["unrec_per_year"] = kci(I["unrecorded_per_year"]); N["unrec_med_k"] = f"{round(I['unrecorded_per_year'][0] / 1000):.0f}"
N["unrec_roster_exact"] = k0(I["unrecorded_per_year_roster_exact"]); N["unrec_srs_complete"] = k0(I["unrecorded_per_year_srs_complete"])
N["unrec_lo_k"] = f"{round(I['unrecorded_per_year_srs_complete'] / 1000):.0f}"; N["unrec_hi_k"] = f"{round(I['unrecorded_per_year_roster_exact'] / 1000):.0f}"
N["n_states_net_unrec"] = str(I["n_states_net_unrecorded"])
N["top5_unrec"] = "; ".join(f"{nice(u)} {k0(m)} ({k0(lo)}–{k0(hi)})" for u, m, lo, hi in I["top5_states_unrecorded"])
N["top5_unrec_share_pct"] = f"{100 * I['top5_share_of_net_unrecorded']:.0f}"; N["up_unrec_share_pct"] = f"{100 * I['up_share_of_net_unrecorded']:.0f}"
N["up_unrec"] = k0(I["top5_states_unrecorded"][0][1]) if I["top5_states_unrecorded"][0][0] == "UTTAR PRADESH" else "NA"
N["C_men"] = ci(*I["C_men"]); N["C_women"] = ci(*I["C_women"])
N["unrec_women"] = kci(I["unrecorded_women_per_year"]); N["unrec_men"] = kci(I["unrecorded_men_per_year"])
N["female_share_police_pct"] = f"{100 * I['female_share_police']:.0f}"
N["female_share_true_pct"] = f"{100 * I['female_share_true'][0]:.0f}"
N["female_share_unrec_pct"] = f"{100 * I['female_share_unrecorded'][0]:.0f}"
N["R_national_raw"] = ci(*I["R_national_raw"])
N["tip_rho_null"] = f"{I['tip_rho_ratio_null']:.2f}"; N["tip_rho_ci"] = f"{I['tip_rho_ratio_ci']:.2f}"
N["tip_rho_null_pct"] = f"{100 * (I['tip_rho_ratio_null'] - 1):.0f}"; N["tip_rho_ci_pct"] = f"{100 * (I['tip_rho_ratio_ci'] - 1):.0f}"
N["police_change_pct"] = f"{100 * (1 - I['police_rate_ratio']):.0f}"
tr = I["true_rate_ratio"]
N["true_change_pct"] = f"{100 * (1 - tr[0]):.0f}% ({100 * (1 - tr[2]):.0f}–{100 * (1 - tr[1]):.0f}%)"
N["true_change_med_pct"] = f"{100 * (1 - tr[0]):.0f}"
N["completeness_change"] = ci(*I["completeness_ratio"])
N["masked_pct"] = f"{100 * I['masked_share_of_decline']:.0f}"
N["completeness_pct"] = f"{100 * (I['completeness_ratio'][0] - 1):.0f}"
N["decline_ratio"] = f"{(1 - tr[0]) / (1 - I['police_rate_ratio']):.1f}"
N["data_statement"] = ("Code, derived state-level data, model outputs and the online supplement are openly available at Zenodo "
                       "(https://doi.org/10.5281/zenodo.24199145) and at https://github.com/hssling/india-injury-death-completeness. "
                       "NFHS unit-level data are available on request from the Demographic and Health Surveys Program (https://dhsprogram.com) "
                       "and PLFS unit-level data from the Ministry of Statistics and Programme Implementation (https://microdata.gov.in); "
                       "neither may be redistributed.")
vv = [v for k, v in I["true_rate_ratio_variants"].items()]
N["true_change_var_lo"] = f"{100 * (1 - max(vv)):.0f}"; N["true_change_var_hi"] = f"{100 * (1 - min(vv)):.0f}"
N["n_states_trend"] = str(I["n_states_trend"]); N["n_opposite"] = str(I["n_opposite"])
N["n_police_rise"] = str(I["n_police_rise"]); N["n_true_decline_090"] = str(I["n_true_decline_090"])
sg = lambda x: f"{'+' if x >= 1 else '−'}{abs(100 * (x - 1)):.0f}%"
N["opposite_states"] = "; ".join(f"{nice(u)} (police {sg(p)}, true {sg(t)})" for u, p, t in sorted(I["opposite_states"], key=lambda r: r[1] - r[2]))
N["n_gbd"] = str(I["n_gbd"]); N["rho_true_gbd"] = f"{I['rho_true_gbd']:.2f}"; N["rho_police_gbd"] = f"{I['rho_police_gbd']:.2f}"
N["C_gbd_india"] = f2(I["C_gbd_india"]); N["rho_lor"] = f"{I['rho_lor']:.2f}"; N["n_lor"] = str(I["n_lor"])
N["p_lor"] = "<0.001" if I["p_lor"] < 0.001 else f"{I['p_lor']:.3f}"; N["p_true_gbd"] = "<0.001" if I["p_true_gbd"] < 0.001 else f"{I['p_true_gbd']:.3f}"

json.dump(N, open(O / "numbers.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"{len(N)} numbers written"); [print(f"  {k}: {v}") for k, v in N.items()]
