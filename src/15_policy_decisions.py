"""Decision consequences of incomplete police injury statistics (Public Health version, 2026-11-20).

Three questions a programme manager would ask of the posterior draws of 08_bayes_model.py:

1. Fitness for use. Which states' police counts can be used as recorded? Tiers from the posterior of completeness C
   (main model, 2017-2019): over-recording P(C > 1) >= 0.8 (deaths of non-residents counted at place of occurrence);
   adequate P(C >= 0.8) >= 0.8; unfit P(C < 0.5) >= 0.8 (an independent count is needed); otherwise 'correct before use'.
   A separate trend flag: the 95% credible interval of the completeness change between rounds excludes 1.
   Correction multiplier = 1 / C (posterior median and 95% CrI).
2. Allocation. If a fixed national budget were shared among states in proportion to police-recorded injury deaths rather
   than true resident injury deaths, the share misdirected is the dissimilarity index D = 0.5 * sum_s |p_s - t_s|
   (p, t: state shares). Shares do not depend on the national level of completeness, so D is computed for the main model
   and for the two extreme identifying assumptions. Also: states whose police share is < 80% of their true share.
3. Target judgement. The National Suicide Prevention Strategy seeks a 10% reduction by 2030. Using the change between
   rounds as a test case, a state is judged to have 'achieved a 10% reduction' if the rate ratio <= 0.9. Police verdict:
   police rate ratio (2017-2019 vs 2016); true verdict: model draws (2017-2019 vs 2013-2016 round). Posterior of the
   number of states misjudged, split into missed achievements and false achievements. Also the number of states whose
   completeness changed by more than 10% (posterior median), i.e. more than the size of the target.
Outputs: outputs/policy.json, outputs/policy_by_state.csv
"""
import json, pathlib
import numpy as np, pandas as pd

P = pathlib.Path(__file__).resolve().parents[1]; O = P / "outputs"
q = lambda x: [float(np.median(x)), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
TARGET = 0.9          # rate ratio corresponding to a 10% reduction
J = {}


def load(v):
    z = np.load(O / f"bayes_draws_{v}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def rnd(z, r):
    m = z["round"] == r
    return dict(C=np.exp(z["log_C"][:, m]), lam=np.exp(z["log_lam"][:, m]), py=z["person_years"][m],
                nc=z["ncrb"][m], unit=z["unit"][m])


Z = load("main")
r5, r4 = rnd(Z, "nfhs5"), rnd(Z, "nfhs4")
o4 = {u: i for i, u in enumerate(r4["unit"])}
assert set(o4) == set(r5["unit"]), "rounds cover different states"
idx = np.array([o4[u] for u in r5["unit"]])
for k in ("C", "lam"):
    r4[k] = r4[k][:, idx]
for k in ("py", "nc", "unit"):
    r4[k] = r4[k][idx]
U = r5["unit"]; C = r5["C"]; pop = r5["py"] / r5["py"].sum()

# ---------------- 1. fitness for use
p_over, p_adeq, p_unfit = (C > 1).mean(0), (C >= 0.8).mean(0), (C < 0.5).mean(0)
tier = np.where(p_over >= 0.8, "over-recording", np.where(p_adeq >= 0.8, "adequate",
                np.where(p_unfit >= 0.8, "unfit", "correct before use")))
cc = r5["C"] / r4["C"]                                         # completeness change between rounds, draws x states
trend_flag = (np.percentile(cc, 2.5, 0) > 1) | (np.percentile(cc, 97.5, 0) < 1)
mult = 1 / C
S = pd.DataFrame({"unit": U, "C_med": np.median(C, 0), "C_lo": np.percentile(C, 2.5, 0), "C_hi": np.percentile(C, 97.5, 0),
                  "P_C_ge_0.8": p_adeq, "P_C_lt_0.5": p_unfit, "P_C_gt_1": p_over, "tier": tier,
                  "mult_med": np.median(mult, 0), "mult_lo": np.percentile(mult, 2.5, 0), "mult_hi": np.percentile(mult, 97.5, 0),
                  "cc_med": np.median(cc, 0), "cc_lo": np.percentile(cc, 2.5, 0), "cc_hi": np.percentile(cc, 97.5, 0),
                  "trend_unstable": trend_flag, "pop_share": pop, "deaths_inj": None})
cm = pd.read_csv(O / "bayes_completeness_main.csv").query("round == 'nfhs5'").set_index("unit").deaths_inj
S["deaths_inj"] = S.unit.map(cm)
order = ["unfit", "correct before use", "adequate", "over-recording"]
J["tiers"] = {t: {"n": int((tier == t).sum()), "pop_share": float(pop[tier == t].sum()),
                  "states": sorted(U[tier == t].tolist())} for t in order}
J["n_units"] = int(len(U))
J["n_trend_unstable"] = int(trend_flag.sum())
J["pop_share_not_usable"] = float(pop[(tier == "unfit") | (tier == "correct before use")].sum())

# ---------------- 2. allocation
def dissim(z):
    r = rnd(z, "nfhs5")
    t = r["lam"] * r["py"]                                     # true resident injury deaths (draws x states, scale-free)
    ts = t / t.sum(1, keepdims=True); ps = r["nc"] / r["nc"].sum()
    return 0.5 * np.abs(ps - ts).sum(1), ps / ts, r["unit"]


D, share_ratio, Ud = dissim(Z)
assert list(Ud) == list(U)
J["dissimilarity"] = q(D)
for v in ("roster_exact", "srs_complete", "migration"):
    J[f"dissimilarity_{v}"] = float(np.median(dissim(load(v))[0]))
sr = np.median(share_ratio, 0)
S["share_ratio_med"] = sr; S["share_ratio_lo"] = np.percentile(share_ratio, 2.5, 0); S["share_ratio_hi"] = np.percentile(share_ratio, 97.5, 0)
under = (share_ratio < 0.8).mean(0) >= 0.9
J["n_under_allocated_090"] = int(under.sum())
J["under_allocated_states"] = [[u, float(s)] for u, s in sorted(zip(U[under], sr[under]), key=lambda x: x[1])]
J["pop_share_under_allocated"] = float(pop[under].sum())
big = pd.Series(sr, index=U)
J["share_ratio_up"] = float(big["UTTAR PRADESH"])

# ---------------- 3. target judgement (test case: change between survey rounds)
pr = (r5["nc"] / r5["py"]) / (r4["nc"] / r4["py"])            # police rate ratio, 2017-2019 vs 2016 (fixed)
tr = (r5["lam"]) / (r4["lam"])                                 # true rate ratio, draws x states
pol_met = pr <= TARGET
true_met = tr <= TARGET
mis = true_met != pol_met                                      # draws x states
J["n_states_target"] = int(len(U))
J["n_police_met"] = int(pol_met.sum())
J["n_true_met"] = q(true_met.sum(1))
J["n_misjudged"] = q(mis.sum(1))
J["n_missed_achievement"] = q((true_met & ~pol_met).sum(1))
J["n_false_achievement"] = q((~true_met & pol_met).sum(1))
J["n_misjudged_p090"] = int((mis.mean(0) > 0.9).sum())
J["misjudged_p090_states"] = sorted(U[mis.mean(0) > 0.9].tolist())
hi = mis.mean(0) > 0.9
J["n_missed_p090"] = int((hi & ~pol_met).sum()); J["n_false_p090"] = int((hi & pol_met).sum())
J["missed_p090_states"] = sorted(U[hi & ~pol_met].tolist()); J["false_p090_states"] = sorted(U[hi & pol_met].tolist())
S["police_ratio"] = pr; S["P_true_met"] = true_met.mean(0); S["P_misjudged"] = mis.mean(0)
# national
nat_t = (r5["lam"] * r5["py"]).sum(1) / r5["py"].sum() / ((r4["lam"] * r4["py"]).sum(1) / r4["py"].sum())
nat_p = (r5["nc"].sum() / r5["py"].sum()) / (r4["nc"].sum() / r4["py"].sum())
J["national_police_ratio"] = float(nat_p); J["national_P_true_met"] = float((nat_t <= TARGET).mean())
# completeness change larger than the target
J["n_cc_gt_10pct"] = int((np.abs(np.median(cc, 0) - 1) > 0.10).sum())
J["n_cc_gt_10pct_P090"] = int((((cc > 1.1).mean(0) > 0.9) | ((cc < 1 / 1.1).mean(0) > 0.9)).sum())
J["cc_iqr_states"] = [float(np.percentile(np.median(cc, 0), 25)), float(np.percentile(np.median(cc, 0), 75))]

S.sort_values("C_med").to_csv(O / "policy_by_state.csv", index=False)
(O / "policy.json").write_text(json.dumps(J, indent=1), encoding="utf-8")
for k, v in J.items():
    print(k, v if not isinstance(v, float) else round(v, 3))
