"""Independent checks of the manuscript against data and outputs.

Each check recomputes a quantity without the pipeline code where possible (raw extracts, published tables) or tests that a
worded claim agrees with its number.
  python verify.py              run all checks (manuscript text if present, else the frozen claims in results/)
  python verify.py --selftest   plant a wrong number and require a NEW failure (a verifier that cannot fail is worthless)
  python verify.py --freeze     record every question asked of the manuscript text, with its answer, in
                                results/expected_claims.json (the public repository ships this instead of the manuscript)
Checks that need NFHS unit-level data or the Crime in India PDF are skipped (reported as SKIP) when those are absent.
"""
import json, pathlib, re, sys
import numpy as np, pandas as pd

P = pathlib.Path(__file__).resolve().parent
O = P / "outputs"; D = P / "data_local"
N = json.loads((O / "numbers.json").read_text(encoding="utf-8"))
MSF = P / "manuscript/01_manuscript.md"; FROZEN = P / "results/expected_claims.json"
cmc = lambda y, m: (y - 1900) * 12 + m


class Text:
    """Questions the checks ask of the manuscript. Live: answered from the text (and recorded for --freeze).
    Frozen: answered from results/expected_claims.json; an unrecorded question fails."""

    def __init__(self, ms=None, frozen=None):
        self.ms, self.frozen, self.log = ms, frozen, {}

    def _ans(self, key, fn):
        if self.ms is None:
            return self.frozen.get(key)
        v = fn(); self.log[key] = v; return v

    def intext(self, s):
        pat = (r"(?<![\d.])" if s[:1].isdigit() else "") + re.escape(s) + (r"(?![\d])" if s[-1:].isdigit() else "")
        return self._ans("in:" + s, lambda: re.search(pat, self.ms) is not None) is True

    def has(self, phrase):
        return self._ans("has:" + phrase, lambda: phrase in self.ms) is True

    def first_lowest(self):
        f = lambda: re.search(r"lowest estimates in 2017–2019 were for ([A-Za-z ()]+?) \d", self.ms).group(1).strip()
        return self._ans("first_lowest", f) or ""

    def abstract_words(self):
        end = "## Key messages" if self.ms is not None and "## Key messages" in self.ms else "## Introduction"
        f = lambda: len(re.sub(r"[#*]", " ", self.ms.split("## Abstract")[1].split(end)[0]).split()) - 4
        v = self._ans("abstract_words", f)
        return 9999 if v is None else v

    def clean(self):
        return self._ans("clean", lambda: "{" not in self.ms and "[@" not in self.ms) is True


def run(T):
    res = []

    def chk(name, ok, detail=""):
        res.append((name, bool(ok), detail))
    intext = T.intext

    # 1-3 survey, recomputed directly from the extracted rosters (NFHS unit-level data; skipped if absent)
    if (D / "nfhs5_deaths.parquet").exists():
        hh = pd.read_parquet(D / "nfhs5_households.parquet"); de = pd.read_parquet(D / "nfhs5_deaths.parquet")
        end = np.minimum(hh.hv008, cmc(2020, 3)); x = hh.hv012 * np.clip(end - cmc(2017, 1), 0, None) / 12
        de = de[de.yr < 9000].copy(); de["d"] = cmc(de.yr, de.mo.where(de.mo <= 12, 6))
        de = de.merge(pd.DataFrame({"hhid": hh.hhid, "end": end}), on="hhid"); de = de[(de.d >= cmc(2017, 1)) & (de.d < de.end)]
        age = np.where(de.age_u == 3, de.age_n, np.where(de.age_u.isin([1, 2]), 0, np.nan))
        inj = de[(de.inj == 1) & (age >= 1) & (age < 98)]
        chk("NFHS-5 deaths in window", f"{len(de):,}" == N["deaths_nfhs5"] and intext(N["deaths_nfhs5"]), f"{len(de):,}")
        chk("NFHS-5 injury deaths age>=1", f"{len(inj):,}" == N["inj_deaths_nfhs5"] and intext(N["inj_deaths_nfhs5"]), f"{len(inj):,}")
        cdr = 1000 * de.hv005.sum() / (hh.hv005 * x).sum()
        chk("NFHS-5 survey CDR", f"{cdr:.2f}" == N["cdr_nfhs5"] and intext(N["cdr_nfhs5"]), f"{cdr:.2f}")
        ir = 1e5 * inj.hv005.sum() / (hh.hv005 * x).sum()
        chk("NFHS-5 injury rate age>=1", f"{ir:.1f}" == N["inj_rate_nfhs5"], f"{ir:.1f}")
        n4 = len(pd.read_parquet(D / "nfhs4_households.parquet"))
        chk("households total", f"{len(hh) + n4:,}" == N["hh_total"] and intext(N["hh_total"]))
    else:
        chk("SKIP survey recomputation (NFHS unit-level data not present)", True)
        s = pd.read_csv(O / "survey_estimates_main.csv").query("unit == 'INDIA'").set_index("round")
        chk("survey estimates file = quoted numbers", f"{int(s.loc['nfhs5', 'deaths_all']):,}" == N["deaths_nfhs5"]
            and f"{s.loc['nfhs5', 'cdr']:.2f}" == N["cdr_nfhs5"] and intext(N["deaths_nfhs5"]) and intext(N["cdr_nfhs5"]))

    # 4-6 police counts recomputed from the published tables in data/
    a = pd.read_csv(P / "data/adsi_state_year_sex.csv"); a = a[a.sex == "total"]
    yrs = [2017, 2018, 2019]
    oc = a[(a.table == "other_causes") & a.year.isin(yrs)].groupby("cause").n.sum()
    fon = a[(a.table == "forces_of_nature") & a.year.isin(yrs) & ~a.cause.isin(["Epidemic", "Starvation due to Natural Calamity"])].n.sum()
    sui = pd.read_csv(P / "data/ncrb_suicides_state_year.csv"); sui = sui[sui.year.isin(yrs)].total.sum()
    mur = pd.read_csv(P / "data/cii_murders_state_year.csv"); mur = mur[mur.year.isin(yrs)].murders.sum()
    tot = oc["Total"] - oc["Sudden Deaths (Total)"] - oc["Deaths of Women during Pregnancy (Total)"] + fon + sui + mur
    chk("NCRB injury deaths 2017-19", f"{int(tot):,}" == N["ncrb_total_nfhs5"] and intext(N["ncrb_total_nfhs5"]), f"{int(tot):,}")
    sd = a[(a.table == "other_causes") & (a.cause == "Sudden Deaths (Total)")].groupby("year").n.sum()
    chk("sudden deaths 2016/2024 in text", f"{int(sd[2016]):,}" == N["sudden_2016"] and f"{int(sd[2024]):,}" == N["sudden_2024"]
        and intext(N["sudden_2016"]) and intext(N["sudden_2024"]))
    chk("CII 2019 murders sum = 28,918 (printed All-India total)",
        int(pd.read_csv(P / "data/cii_murders_state_year.csv").query("year == 2019").murders.sum()) == 28918)
    pdf = P / "data_raw/cii/Crime_in_India_2019_Volume_I_7dfa44948c.pdf"
    if pdf.exists():
        import fitz
        lines = fitz.open(pdf)[198].get_text().split("\n")
        i = [j for j, l in enumerate(lines) if l.strip().startswith("TOTAL ALL INDIA")][0]
        chk("CII 2019 PDF printed all-India murders = parsed", int(lines[i + 3]) == int(N["murders_2019"].replace(",", "")))

    # 7-12 model results and the claims built on them
    nat = {v: pd.read_csv(O / f"bayes_national_{v}.csv").set_index("round") for v in ("main", "srs_complete", "roster_exact")}
    m5 = nat["main"].loc["nfhs5"]
    chk("national C 2017-19 in text", f"{m5.C_med:.2f} ({m5.C_lo:.2f}–{m5.C_hi:.2f})" == N["C_nat_nfhs5"] and intext(N["C_nat_nfhs5"]))
    lo, hi = nat["roster_exact"].loc["nfhs5", "C_med"], nat["srs_complete"].loc["nfhs5", "C_med"]
    chk("bracket ordered roster_exact < main < srs_complete", lo < m5.C_med < hi)
    chk("'one fifth and two fifths' = 1 - bracket", abs((1 - hi) - 0.2) < 0.05 and abs((1 - lo) - 0.4) < 0.05
        and T.has("one fifth to two fifths"), f"{1 - hi:.2f}-{1 - lo:.2f}")
    chk("'three in four' ~ main C", 0.68 <= m5.C_med <= 0.80 and T.has("three in four"))
    sp = pd.read_csv(O / "sex_ratio_pooled.csv").set_index("round").loc["nfhs5"]
    chk("sex ratio CrI excludes 1 ('less completely recorded')", sp.R_hi < 1 and T.has("less completely recorded"))
    chk("'about a quarter less completely' ~ 1 - R", 0.18 <= 1 - sp.R_med <= 0.32 and T.has("a quarter less completely"), f"{1 - sp.R_med:.2f}")
    rg = pd.read_csv(O / "registration_premium_age1plus.csv").set_index("model").loc["M3 + state FE"]
    chk("registration PR CI excludes 1 ('more often registered')", rg.lo > 1 and T.has("more often registered"))
    pg = pd.read_csv(O / "bayes_params_migration.csv").set_index("param").loc["beta_mig"]
    chk("migration coefficient CI excludes 0 ('rose with the share')", pg.lo > 0 and T.has("rose with the share of in-migrants"))
    c5 = pd.read_csv(O / "bayes_completeness_main.csv").query("round == 'nfhs5'").set_index("unit")
    big = c5[c5.deaths_inj >= 100]
    chk("fold range among large states", f"{big.C_med.max() / big.C_med.min():.1f}" == N["ratio_hi_lo_big"]
        and T.has(f"varied {N['ratio_hi_lo_big']}-fold"))
    first = T.first_lowest()
    chk("first-named lowest state has the minimum C", bool(first) and first.lower().startswith(c5.C_med.idxmin().split(" ")[0].lower()), first)
    rc = []
    for v in ("srs_complete", "roster_exact", "rho_wide", "recall7", "allages", "min30", "ncrb_strict", "ncrb_2014_16", "migration"):
        x = pd.read_csv(O / f"bayes_completeness_{v}.csv").query("round == 'nfhs5'").set_index("unit").C_med
        j = c5.index.intersection(x.index); rc.append(np.corrcoef(c5.C_med[j].rank(), x[j].rank())[0, 1])
    chk("rank correlation >= stated minimum", min(rc) >= float(N["min_rank_corr"]) - 0.005, f"{min(rc):.3f}")

    # 13-16 methods text agrees with code
    code = (P / "src/08_bayes_model.py").read_text(encoding="utf-8")
    chk("draws/warm-up/chains in text = code", "num_warmup=500 if QUICK else 1500" in code and "num_samples=500 if QUICK else 2000" in code
        and "num_chains=4" in code and T.has("four chains of 2000 draws after 1500 warm-up"))
    chk("SRS coverage prior centre in code = SRS/GBD in text", f"jnp.log({N['srs_cov_prior']})" in code and intext(N["srs_cov_prior"]))
    chk("injury-specific prior sd 0.1 / wide 0.25 in code", 'dist.Normal(0, 0.25 if VARIANT == "rho_wide" else 0.1)' in code)
    diag = [(O / f.name).read_text() for f in sorted(O.glob("bayes_diag_*.txt"))]
    rh = max(float(re.search(r"R-hat ([0-9.]+)", d).group(1)) for d in diag)
    chk("all model variants converged (R-hat <= 1.01)", rh <= 1.0105 and intext(N["max_rhat"]), f"{rh:.3f}")

    # 17-20 survey-side claims and text hygiene
    chk("female survey injury rate fell between rounds", float(N["inj_rate_f_nfhs5"]) < float(N["inj_rate_f_nfhs4"]) and T.has("survey injury rate among women also fell"))
    r7, r0 = float(N["inj_rate_recall7_nfhs5"]), float(N["inj_rate_nfhs5"])
    chk("'left the injury rate unchanged' (recall check, <2%)", abs(r7 / r0 - 1) < 0.02 and T.has("left the injury rate unchanged"))
    chk("no unfilled placeholder or citation marker", T.clean())
    chk("abstract <= 250 words", T.abstract_words() <= 250)

    # 21-30 JECH additions (2026-11-17): unrecorded deaths, sex split, trends, external comparison; recomputed from draws
    z = np.load(O / "bayes_draws_main.npz", allow_pickle=False)
    m5, m4 = z["round"] == "nfhs5", z["round"] == "nfhs4"
    lam = np.exp(z["log_lam"].astype(float)); py = z["person_years"]; nc = z["ncrb"]
    tot5 = (lam[:, m5] * py[m5] / 1e5).sum(1) / 3; pol5 = nc[m5].sum() / 3
    un = np.median(tot5 - pol5)
    chk("unrecorded deaths/year recomputed from draws", f"{1000 * round(un / 1000):,}" == N["unrec_per_year"].split(" ")[0]
        and intext(N["unrec_per_year"]), f"{un:,.0f}")
    chk("police deaths/year = NCRB 2017-19 modelled states / 3", f"{1000 * round(pol5 / 1000):,}" == N["police_per_year"] and intext(N["police_per_year"]))
    o4 = {u: i for i, u in enumerate(z["unit"][m4])}; ix = np.array([o4[u] for u in z["unit"][m5]])
    l4 = lam[:, m4][:, ix]; p4 = py[m4][ix]; n4 = nc[m4][ix]
    trr = ((lam[:, m5] * py[m5]).sum(1) / py[m5].sum()) / ((l4 * p4).sum(1) / p4.sum())
    prr = (nc[m5].sum() / py[m5].sum()) / (n4.sum() / p4.sum())
    chk("police vs true change recomputed", f"{100 * (1 - prr):.0f}" == N["police_change_pct"]
        and f"{100 * (1 - np.median(trr)):.0f}" == N["true_change_med_pct"] and intext(N["true_change_pct"]))
    masked = (prr - np.median(trr)) / (1 - np.median(trr))
    chk("'about three quarters of a real decline' ~ masked share", 0.65 <= masked <= 0.85 and T.has("about three quarters of a real decline")
        and np.percentile(trr, 97.5) < 1, f"{masked:.2f}")
    cr = prr / np.median(trr)
    chk("completeness rise exceeds the 10% NSPS target ('more than the 10% reduction')", cr - 1 > 0.10 and T.has("more than the 10% reduction"), f"{cr:.3f}")
    chk("decline ratio in text", f"{(1 - np.median(trr)) / (1 - prr):.1f}" == N["decline_ratio"] and T.has(f"{N['decline_ratio']} times as large"))
    tb = pd.read_csv(O / "trend_by_state.csv").set_index("unit").loc["BIHAR"]
    chk("Bihar: police rose, true fell (P>0.9)", tb.police_ratio > 1 and tb.P_true_decline > 0.9 and T.has("Bihar police counts rose while true injury mortality fell"))
    I = json.loads((O / "implications.json").read_text(encoding="utf-8"))
    chk("GBD: true rates agree more closely than police rates", I["rho_true_gbd"] > I["rho_police_gbd"] and T.has("agreed more closely with GBD estimates than police rates did"))
    chk("GBD-implied national completeness inside the bracket ('inside our range')", lo <= I["C_gbd_india"] <= hi and T.has("inside our range"), f"{I['C_gbd_india']:.2f}")
    chk("women's completeness below men's; women over-represented among unrecorded", I["C_women"][0] < I["C_men"][0]
        and I["female_share_unrecorded"][0] > I["female_share_true"][0] > I["female_share_police"])
    sx_code = (P / "src/10_sex_ratio.py").read_text(encoding="utf-8")
    chk("sex-ratio model has state-by-round term (Methods)", "tau_x" in sx_code and T.has("a state effect and a state-by-round deviation"))
    return res


def report(res):
    for n, ok, d in res:
        print(("PASS " if ok else "FAIL ") + n + (f"  [{d}]" if d else ""))
    nf = sum(not ok for _, ok, _ in res)
    print(f"{len(res) - nf}/{len(res)} checks passed")
    return nf


def selftest(make_T, mutate):
    base = {n for n, ok, _ in run(make_T()) if not ok}
    mutate()
    new = {n for n, ok, _ in run(make_T()) if not ok} - base
    print("self-test:", "PASS (mutation detected by: " + ", ".join(sorted(new)) + ")" if new else "FAIL (mutation not detected)")
    return 0 if new else 1


if MSF.exists():
    MS = MSF.read_text(encoding="utf-8")
    if "--freeze" in sys.argv:
        T = Text(MS); res = run(T)
        assert report(res) == 0, "freeze only from a fully passing run"
        FROZEN.parent.mkdir(exist_ok=True)
        FROZEN.write_text(json.dumps(T.log, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"froze {len(T.log)} manuscript answers -> results/expected_claims.json")
        sys.exit(0)
    if "--selftest" in sys.argv:
        box = {"ms": MS}
        sys.exit(selftest(lambda: Text(box["ms"]), lambda: box.update(ms=MS.replace(N["C_nat_nfhs5"], "0.82 (0.58–0.91)"))))
    sys.exit(1 if report(run(Text(MS))) else 0)

# public repository: no manuscript; questions are answered from the frozen claims
FR = json.loads(FROZEN.read_text(encoding="utf-8"))
if "--selftest" in sys.argv:
    sys.exit(selftest(lambda: Text(frozen=FR), lambda: N.update(C_nat_nfhs5="0.82 (0.58–0.91)")))
sys.exit(1 if report(run(Text(frozen=FR))) else 0)
