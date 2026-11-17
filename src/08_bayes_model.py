"""Bayesian measurement model: completeness of police (NCRB) injury-death recording by state, NFHS-4 and NFHS-5.

For state s, round r (units with >= MIN_INJ survey injury deaths):
  true all-cause death rate mu_sr, true resident injury death rate lam_sr (per 100k)
  survey all-cause   log CDR_hat ~ N(log mu + log k_s + kap_r, se_A)         k: household roster reporting factor
  SRS                log SRS     ~ N(log mu + log c_s, se_S)                 c: SRS coverage of deaths
  survey injury      log INJ_hat ~ N(log lam + log k_s + kap_r + log rho, se_I) rho: injury-specific reporting
  police             NCRB_sr     ~ Poisson(PY_sr * lam_sr/1e5 * C_sr)         C: police completeness (estimand)
Priors:
  log c_s = gam + sig_c z_s,   gam ~ N(log 0.90, 0.05)   (GBD 2023 all-cause / SRS, India 2019: 6.64 vs 6.0)
  log k_s = sig_k z'_s ; kap_r ~ N(0, 0.5) round levels ; sig_k, sig_c ~ HalfNormal(0.2)
  log rho ~ N(0, 0.1)          (injury deaths reported like other deaths; recall check: no differential telescoping)
  log C_sr = a_r + u_s + v_sr ; a_r ~ N(log 0.6, 1); u_s ~ N(0, sig_u); v_sr ~ N(0, sig_v); sig_u ~ HN(0.5), sig_v ~ HN(0.25)
Only the ratio k/c is identified state by state; the split follows the priors. Variants (VARIANT env) bracket it:
  main | srs_complete (c = 1: all survey/SRS gaps are roster over-reporting) | roster_exact (k = 1: raw survey) |
  rho_wide (sd 0.25) | recall7 | allages | min30 (units with >= 30 survey injury deaths) |
  ncrb_strict (NCRB excludes Causes Not Known) | ncrb_2014_16 (NFHS-4 vs NCRB 2014-16, mixed ADSI classification)
Outputs: outputs/bayes_completeness_<variant>.csv, outputs/bayes_national_<variant>.csv, outputs/bayes_diag_<variant>.txt
"""
import os, pathlib
import numpy as np, pandas as pd
import jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

numpyro.set_host_device_count(4)
P = pathlib.Path(__file__).resolve().parents[1]; O = P / "outputs"
VARIANT = os.environ.get("VARIANT", "main")
DATA = {"recall7": "recall7", "allages": "allages"}.get(VARIANT, "main")
MIN_INJ = 30 if VARIANT == "min30" else 10
QUICK = os.environ.get("QUICK") == "1"

S = pd.read_csv(O / f"survey_estimates_{DATA}.csv"); S = S[S.unit != "INDIA"]
N = pd.read_csv(O / ("ncrb_window" + {"ncrb_strict": "_strict", "ncrb_2014_16": "_2014_16"}.get(VARIANT, "") + ".csv"))
srs = pd.read_csv(O / "state_calibration.csv")[["round", "unit", "srs_cdr"]]
d = S.merge(N[["unit", "round", "ncrb", "person_years"]], on=["unit", "round"]).merge(srs, on=["unit", "round"])
d = d[(d.deaths_inj >= MIN_INJ) & d.srs_cdr.notna()].reset_index(drop=True)
units = sorted(d.unit.unique()); rounds = ["nfhs4", "nfhs5"]
si = d.unit.map({u: i for i, u in enumerate(units)}).values
ri = d["round"].map({r: i for i, r in enumerate(rounds)}).values
pop_m = d.person_years / d["round"].map({"nfhs4": 3, "nfhs5": 3}) / 1e6
se_S = np.where(pop_m > 10, 0.03, 0.08)          # SRS sampling error on log CDR (large vs small states)
print(f"{VARIANT}: {len(d)} state-rounds, {len(units)} units (min survey injury deaths {MIN_INJ})")
# migration variant: standardised log inter-state in-migrant share (PLFS 2020-21, 09_migration_covariate.py)
MIG = pd.read_csv(P / "data/plfs_inmigration_2020_21.csv").set_index("unit").inmig_share
xm = np.log(MIG.reindex(units).values + 0.005)
assert not np.isnan(xm).any()
xm = jnp.array((xm - xm.mean()) / xm.std())


def model(y_cdr, se_A, y_srs, y_inj, se_I, ncrb, py):
    ns, nr = len(units), len(rounds)
    if VARIANT == "srs_complete":
        log_c = jnp.zeros(ns)
    else:
        gam = numpyro.sample("gam", dist.Normal(jnp.log(0.90), 0.05))
        sig_c = numpyro.sample("sig_c", dist.HalfNormal(0.2))
        log_c = numpyro.deterministic("log_c", gam + sig_c * numpyro.sample("zc", dist.Normal(0, 1).expand([ns])))
    if VARIANT == "roster_exact":
        log_k = jnp.zeros(len(si))
    else:
        kap = numpyro.sample("kap", dist.Normal(0, 0.5).expand([nr]))
        sig_k = numpyro.sample("sig_k", dist.HalfNormal(0.2))
        log_k = kap[ri] + sig_k * numpyro.sample("zk", dist.Normal(0, 1).expand([ns]))[si]
    log_rho = numpyro.sample("log_rho", dist.Normal(0, 0.25 if VARIANT == "rho_wide" else 0.1))
    a = numpyro.sample("a", dist.Normal(jnp.log(0.6), 1.0).expand([nr]))
    sig_u = numpyro.sample("sig_u", dist.HalfNormal(0.5))
    sig_v = numpyro.sample("sig_v", dist.HalfNormal(0.25))
    u = sig_u * numpyro.sample("zu", dist.Normal(0, 1).expand([ns]))
    v = sig_v * numpyro.sample("zv", dist.Normal(0, 1).expand([len(si)]))
    if VARIANT == "migration":
        # place of occurrence: police count deaths where they occur (incl. non-household and non-resident people);
        # beta = change in log C per SD of log in-migrant share. 'log_C_rec' = C at mean migration (recording component)
        beta = numpyro.sample("beta_mig", dist.Normal(0, 0.5))
        numpyro.deterministic("log_C_rec", a[ri] + u[si] + v)
        log_C = numpyro.deterministic("log_C", a[ri] + beta * xm[si] + u[si] + v)
    else:
        log_C = numpyro.deterministic("log_C", a[ri] + u[si] + v)
    # Anchored (non-centred) parameterisation. The police count and the SRS rate pin log(lam*C) and log(mu*c) almost
    # exactly; sampling lam and mu freely leaves a razor-thin ridge NUTS cannot traverse. Equivalent model:
    #   NCRB ~ Poisson(PY lam C) ~= N on the log scale with sd 1/sqrt(NCRB)  ->  log lam = log(NCRB/PY) - log C - e1*sd_N
    #   log SRS ~ N(log mu + log c, se_S)                                      ->  log mu  = log SRS - log c - e2*se_S
    sd_N = 1 / jnp.sqrt(ncrb)
    log_lam = numpyro.deterministic("log_lam", jnp.log(ncrb / py * 1e5) - log_C - sd_N * numpyro.sample("e1", dist.Normal(0, 1).expand([len(si)])))
    log_mu = y_srs - log_c[si] - se_S * numpyro.sample("e2", dist.Normal(0, 1).expand([len(si)]))
    numpyro.sample("obs_cdr", dist.Normal(log_mu + log_k, se_A), obs=y_cdr)
    numpyro.sample("obs_inj", dist.Normal(log_lam + log_k + log_rho, se_I), obs=y_inj)


args = dict(y_cdr=jnp.log(d.cdr.values), se_A=jnp.array(d.log_cdr_se.values), y_srs=jnp.log(d.srs_cdr.values),
            y_inj=jnp.log(d.inj_rate.values), se_I=jnp.array(d.log_inj_se.values), ncrb=jnp.array(d.ncrb.values),
            py=jnp.array(d.person_years.values))
mc = MCMC(NUTS(model, target_accept_prob=0.95), num_warmup=500 if QUICK else 1500, num_samples=500 if QUICK else 2000,
          num_chains=4, progress_bar=False)
mc.run(jax.random.PRNGKey(2026), **args)
smp = mc.get_samples(group_by_chain=True)
flat = {k: np.asarray(v).reshape(-1, *v.shape[2:]) for k, v in smp.items()}

# diagnostics
from numpyro.diagnostics import summary
summ = summary({k: v for k, v in smp.items() if k in ("a", "sig_u", "sig_v", "log_rho", "kap", "sig_k", "gam", "sig_c", "log_C")})
rhat = max(float(np.nanmax(s["r_hat"])) for s in summ.values())
ess = min(float(np.nanmin(s["n_eff"])) for s in summ.values())
div = int(np.asarray(mc.get_extra_fields()["diverging"]).sum())
diag = f"{VARIANT}: max R-hat {rhat:.3f}; min bulk ESS {ess:.0f}; divergences {div}"
print(diag); (O / f"bayes_diag_{VARIANT}.txt").write_text(diag + "\n")

C = np.exp(flat["log_C"])
# posterior draws for downstream quantities (14_implications.py: unrecorded deaths, trend bias)
np.savez_compressed(O / f"bayes_draws_{VARIANT}.npz", log_C=flat["log_C"].astype("float32"),
                    log_lam=flat["log_lam"].astype("float32"), unit=d.unit.values.astype(str), round=d["round"].values.astype(str),
                    ncrb=d.ncrb.values, person_years=d.person_years.values,
                    **({"log_C_rec": flat["log_C_rec"].astype("float32")} if "log_C_rec" in flat else {}))
q = lambda x, p: np.percentile(x, p, axis=0)
out = d[["unit", "round", "deaths_inj", "ncrb", "inj_rate", "cdr", "srs_cdr"]].copy()
out["C_med"], out["C_lo"], out["C_hi"] = q(C, 50), q(C, 2.5), q(C, 97.5)
out["P_C_below_0.5"] = (C < 0.5).mean(0)
if "log_C_rec" in flat:
    Cr = np.exp(flat["log_C_rec"])
    out["Crec_med"], out["Crec_lo"], out["Crec_hi"] = q(Cr, 50), q(Cr, 2.5), q(Cr, 97.5)
out.to_csv(O / f"bayes_completeness_{VARIANT}.csv", index=False)
# national completeness: total NCRB / total implied true deaths (posterior)
lam = np.exp(flat["log_lam"]); py = d.person_years.values
nat = []
for j, r in enumerate(rounds):
    m = ri == j
    tot = (lam[:, m] * py[m] / 1e5).sum(1)
    Cn = d.ncrb.values[m].sum() / tot
    nat.append(dict(round=r, C_med=np.median(Cn), C_lo=np.percentile(Cn, 2.5), C_hi=np.percentile(Cn, 97.5),
                    true_inj_rate_med=np.median(tot / py[m].sum() * 1e5)))
# change NFHS-4 -> NFHS-5 for units in both
both = [u for u in units if ((d.unit == u) & (ri == 0)).any() and ((d.unit == u) & (ri == 1)).any()]
chg = []
for u in both:
    i4 = np.where((d.unit == u) & (ri == 0))[0][0]; i5 = np.where((d.unit == u) & (ri == 1))[0][0]
    x = C[:, i5] / C[:, i4]
    chg.append(dict(unit=u, ratio_med=np.median(x), lo=np.percentile(x, 2.5), hi=np.percentile(x, 97.5), P_increase=(x > 1).mean()))
pd.DataFrame(chg).to_csv(O / f"bayes_change_{VARIANT}.csv", index=False)
NAT = pd.DataFrame(nat); NAT.to_csv(O / f"bayes_national_{VARIANT}.csv", index=False)
print(NAT.round(3).to_string(index=False))
w = out.pivot_table(index="unit", columns="round", values="C_med").round(2)
print(w.sort_values("nfhs5").to_string())
for k in ("sig_u", "sig_v", "sig_k", "sig_c", "gam", "log_rho", "beta_mig"):
    if k in flat:
        print(f"{k}: {np.median(flat[k]):.3f} ({np.percentile(flat[k], 2.5):.3f}-{np.percentile(flat[k], 97.5):.3f})")
if "kap" in flat:
    print("k by round (exp kap):", np.round(np.exp(np.median(flat["kap"], 0)), 3))
# parameter summary (scalars; vectors by round): median and 95% CrI, natural and exp scale
par = []
for k, v in flat.items():
    if k in ("sig_u", "sig_v", "sig_k", "sig_c", "gam", "log_rho", "beta_mig") or k in ("kap", "a"):
        arr = v.reshape(v.shape[0], -1)
        for j in range(arr.shape[1]):
            nm = k if arr.shape[1] == 1 else f"{k}_{rounds[j]}"
            x = arr[:, j]
            par.append(dict(param=nm, med=np.median(x), lo=np.percentile(x, 2.5), hi=np.percentile(x, 97.5),
                            exp_med=np.exp(np.median(x)), exp_lo=np.exp(np.percentile(x, 2.5)), exp_hi=np.exp(np.percentile(x, 97.5))))
pd.DataFrame(par).to_csv(O / f"bayes_params_{VARIANT}.csv", index=False)
