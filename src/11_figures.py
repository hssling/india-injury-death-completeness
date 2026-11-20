"""Figures (300-dpi PNG + vector PDF in figures/).

Fig 1  Map: police completeness C (posterior median), 2017-19, main model; log scale diverging at 1.
Fig 2  Forest: C by state with 95% CrI (2017-19, main), migration-adjusted C, and 2013-16 (NFHS-4 round) C.
Fig 3  Female-to-male ratio of completeness by state (2017-19) with pooled national ratios for both rounds.
Fig 4  (a) National C under each identifying assumption / sensitivity analysis, both rounds;
       (b) registration premium (prevalence ratio, injury vs other deaths) overall and by subgroup.
Palette and styling as in the heat-recording paper (validated reference palette; diverging blue<->red, grey midpoint).
State geometry: derived from district boundaries whose licence does not allow redistribution (heat project, local).
"""
import pathlib, sys
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import LinearSegmentedColormap, LogNorm

P = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P / "src"))
from common import unit   # noqa: E402
O = P / "outputs"; F = P / "figures"; F.mkdir(exist_ok=True)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
DIV = LinearSegmentedColormap.from_list("div", ["#9e1f1f", "#e34948", "#f2a3a2", "#f0efec", "#86b6ef", "#2a78d6", "#104281"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300})


def nice(u):
    s = {"JAMMU & KASHMIR (incl. LADAKH)": "Jammu & Kashmir + Ladakh", "DELHI (UT)": "Delhi", "A & N ISLANDS": "Andaman & Nicobar",
         "DNH & DD": "DNH & Daman-Diu"}.get(u)
    return s or u.title().replace(" And ", " and ")


def save(fig, name):
    fig.savefig(F / f"{name}.png", bbox_inches="tight"); fig.savefig(F / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig); print("wrote", name)


main = pd.read_csv(O / "bayes_completeness_main.csv")
mig = pd.read_csv(O / "bayes_completeness_migration.csv")
m5 = main[main["round"] == "nfhs5"].set_index("unit")
m4 = main[main["round"] == "nfhs4"].set_index("unit")
g5 = mig[mig["round"] == "nfhs5"].set_index("unit")

# ---------------- Fig 1 map
geo = P.parent / "meta_heat_deaths_ascertainment_india/data/states_2011_complete.geojson"
if geo.exists():
    st = gpd.read_file(geo); st["u"] = st.unit.map(unit)
    st = st.merge(m5[["C_med", "deaths_inj"]], left_on="u", right_index=True, how="left")
    assert set(st.u[st.C_med.isna()]) == {"CHANDIGARH", "LAKSHADWEEP"}, st.u[st.C_med.isna()].tolist()
    fig, ax = plt.subplots(figsize=(6.2, 6.6))
    norm = LogNorm(vmin=0.2, vmax=5)
    lo = st[st.deaths_inj < 30]; hi = st[st.deaths_inj >= 30]
    hi.plot(column="C_med", cmap=DIV, norm=norm, ax=ax, edgecolor="white", linewidth=0.4)
    lo.plot(column="C_med", cmap=DIV, norm=norm, ax=ax, edgecolor="white", linewidth=0.4, hatch="////", alpha=0.85)
    st[st.C_med.isna()].plot(ax=ax, color="#d9d8d2", edgecolor="white", linewidth=0.4)
    for r in st[st.deaths_inj >= 100].itertuples():
        pt = r.geometry.representative_point()
        ax.annotate(f"{r.C_med:.2f}", (pt.x, pt.y), ha="center", va="center", fontsize=6.5, color=INK,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    ax.set_axis_off()
    sm = plt.cm.ScalarMappable(cmap=DIV, norm=norm)
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.035, pad=0.01, ticks=[0.2, 0.3, 0.5, 1, 2, 3, 5])
    cb.ax.set_xticklabels(["0.2", "0.3", "0.5", "1", "2", "3", "5"])
    cb.set_label("Police-recorded injury deaths per resident injury death (log scale; 1 = complete)", color=INK2)
    cb.outline.set_visible(False)
    ax.text(0.0, 0.0, "Hatched: <30 injury deaths in the survey. Grey: not estimated.\nValues shown for states with ≥100 survey injury deaths.\n"
            "Values above 1 include deaths of non-residents (police count by place of occurrence).",
            transform=ax.transAxes, fontsize=6.5, color=INK2, va="bottom")
    save(fig, "fig1_map_completeness")

# ---------------- Fig 2 forest
order = m5.sort_values("C_med").index.tolist()
fig, ax = plt.subplots(figsize=(6.4, 7.6))
y = np.arange(len(order))
ax.axvline(1, color=MUTED, lw=0.8, zorder=0)
ax.hlines(y, m5.loc[order, "C_lo"], m5.loc[order, "C_hi"], color=CAT[0], lw=2, zorder=2)
ax.plot(m5.loc[order, "C_med"], y, "o", color=CAT[0], ms=5, mec="white", mew=0.6, zorder=3, label="2017–19, main model (95% CrI)")
ax.plot(g5.reindex(order).C_rec if "C_rec" in g5 else g5.reindex(order).Crec_med, y - 0.28, "D", color=CAT[1], ms=3.6, mec="white",
        mew=0.5, zorder=3, label="2017–19, adjusted for in-migration")
ax.plot(m4.reindex(order).C_med, y + 0.28, "s", color=CAT[2], ms=3.6, mfc="white", mew=1, zorder=3, label="2013–16, main model")
ax.set_xscale("log"); ax.set_xlim(0.12, 6)
ax.set_xticks([0.2, 0.3, 0.5, 1, 2, 3, 5]); ax.set_xticklabels(["0.2", "0.3", "0.5", "1", "2", "3", "5"])
ax.set_yticks(y); ax.set_yticklabels([nice(u) for u in order])
ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.set_xlabel("Police-recorded injury deaths per resident injury death (log scale)")
ax.legend(frameon=False, fontsize=7, loc="upper left")
for i, u in enumerate(order):
    ax.text(4.3, i, f"{int(m5.loc[u, 'deaths_inj']):,}", va="center", fontsize=6.5, color=INK2)
ax.text(4.3, len(order) - 0.3, "Survey\ninjury\ndeaths", fontsize=6.5, color=INK2, va="bottom")
save(fig, "fig2_forest_completeness")

# ---------------- Fig 3 sex ratio
sx = pd.read_csv(O / "sex_ratio_state.csv"); sx = sx[sx["round"] == "nfhs5"].sort_values("R_med").reset_index(drop=True)
pool = pd.read_csv(O / "sex_ratio_pooled.csv").set_index("round")
fig, ax = plt.subplots(figsize=(5.6, 6.4))
yy = np.arange(len(sx)) + 2
ax.axvline(1, color=MUTED, lw=0.8, zorder=0)
ax.hlines(yy, sx.R_lo, sx.R_hi, color=CAT[4], lw=1.8)
ax.plot(sx.R_med, yy, "o", color=CAT[4], ms=4.5, mec="white", mew=0.6)
for j, (r, lab) in enumerate([("nfhs5", "India, 2017–19 (pooled)"), ("nfhs4", "India, 2013–16 (pooled)")]):
    yv = 0.6 - j * 0.9
    ax.hlines(yv, pool.loc[r, "R_lo"], pool.loc[r, "R_hi"], color=INK, lw=2)
    ax.plot(pool.loc[r, "R_med"], yv, "D", color=INK, ms=6, mec="white", mew=0.6)
ax.set_yticks(list(yy) + [0.6, -0.3])
ax.set_yticklabels([nice(u) for u in sx.unit] + ["India, 2017–19 (pooled)", "India, 2013–16 (pooled)"])
ax.axhline(1.4, color=GRID, lw=0.8)
ax.set_xscale("log"); ax.set_xticks([0.4, 0.5, 0.7, 1, 1.4, 2]); ax.set_xticklabels(["0.4", "0.5", "0.7", "1", "1.4", "2"])
ax.set_xlim(0.38, 2)
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.set_xlabel("Completeness for women relative to men\n(log scale; below 1 = women's injury deaths less often recorded)")
save(fig, "fig3_sex_ratio")

# ---------------- Fig 4 sensitivity + registration
V = [("main", "Main model"), ("srs_complete", "SRS complete (survey–SRS gap = household over-reporting)"),
     ("roster_exact", "Survey exact (raw survey rates)"), ("rho_wide", "Injury-specific reporting prior SD 0.25"),
     ("recall7", "Drop deaths in last 6 months (recall)"), ("allages", "Injury deaths at all ages"),
     ("min30", "States with ≥30 survey injury deaths"), ("ncrb_strict", "NCRB excl. 'causes not known'"),
     ("ncrb_2014_16", "2013–16 round vs NCRB 2014–16"), ("migration", "With in-migration covariate")]
fig, (b, a) = plt.subplots(1, 2, figsize=(8.6, 3.9), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 1.45})
# panel A (left) = registration, panel B (right) = sensitivity: the text cites the registration panel first (JECH)
yv = np.arange(len(V))[::-1]
for j, (r, col, mk, lab) in enumerate([("nfhs5", CAT[0], "o", "2017–19"), ("nfhs4", CAT[2], "s", "2013–16")]):
    for i, (v, _) in enumerate(V):
        n = pd.read_csv(O / f"bayes_national_{v}.csv").set_index("round").loc[r]
        off = 0.17 if j == 0 else -0.17
        a.hlines(yv[i] + off, n.C_lo, n.C_hi, color=col, lw=1.6)
        a.plot(n.C_med, yv[i] + off, mk, color=col, ms=4.5, mec="white", mew=0.6, label=lab if i == 0 else None)
a.set_yticks(yv); a.set_yticklabels([l for _, l in V], fontsize=6.8)
a.set_xlabel("National completeness\n(posterior median, 95% CrI)")
a.xaxis.grid(True, color=GRID, lw=0.5); a.set_axisbelow(True); a.set_xlim(0.3, 1.3)
a.legend(frameon=False, fontsize=7, loc="lower right")
a.text(-0.02, 1.02, "B", transform=a.transAxes, fontweight="bold", fontsize=9)
R = pd.read_csv(O / "registration_premium_age1plus.csv")
H = pd.read_csv(O / "registration_premium_by_group_age1plus.csv")
rows = [("All deaths (adjusted)", R.iloc[-1].PR, R.iloc[-1].lo, R.iloc[-1].hi)]
LAB = {("sex", "0"): "Men", ("sex", "1"): "Women", ("residence", "0"): "Urban", ("residence", "1"): "Rural"}
for h in H.itertuples():
    lab = LAB.get((h.by, str(h.level)), f"Age {h.level}" if h.by == "age" else f"{h.by} {h.level}")
    rows.append((lab, h.PR, h.lo, h.hi))
yb = np.arange(len(rows))[::-1]
b.axvline(1, color=MUTED, lw=0.8)
for i, (lab, pr, lo, hi) in enumerate(rows):
    c = INK if i == 0 else CAT[3]
    b.hlines(yb[i], lo, hi, color=c, lw=1.8); b.plot(pr, yb[i], "D" if i == 0 else "o", color=c, ms=5 if i == 0 else 4, mec="white", mew=0.6)
b.set_yticks(yb); b.set_yticklabels([r[0] for r in rows], fontsize=6.8)
b.set_xlabel("Registration, injury vs other deaths\n(prevalence ratio, 95% CI)")
b.xaxis.grid(True, color=GRID, lw=0.5); b.set_axisbelow(True)
b.text(-0.02, 1.02, "A", transform=b.transAxes, fontweight="bold", fontsize=9)
save(fig, "fig4_sensitivity_registration")

# ---------------- Fig 5 trend bias: police change vs completeness-corrected change, 2013-16 round -> 2017-19 round
# (14_implications.py). Points on the diagonal: police trend = true trend; above it: police overstate the change.
tb = pd.read_csv(O / "trend_by_state.csv")
imp = __import__("json").loads((O / "implications.json").read_text(encoding="utf-8"))
fig, ax = plt.subplots(figsize=(5.2, 4.8))
xl, yl = (0.7, 1.5), (0.4, 2.0)
ax.plot((0.4, 2), (0.4, 2), color=MUTED, lw=0.8, zorder=0)
ax.axhline(1, color=GRID, lw=0.8, zorder=0); ax.axvline(1, color=GRID, lw=0.8, zorder=0)
col = np.where(tb.opposite, CAT[1], CAT[0])
ax.vlines(tb.police_ratio, tb.true_lo, tb.true_hi, color=col, lw=1.0, alpha=0.55)
ax.scatter(tb.police_ratio, tb.true_ratio_med, s=18, c=col, edgecolor="white", linewidth=0.5, zorder=3)
for r in tb.itertuples():
    if r.opposite or abs(np.log(r.police_ratio / r.true_ratio_med)) > 0.45:
        ax.annotate(nice(r.unit), (r.police_ratio, r.true_ratio_med), xytext=(3, 2), textcoords="offset points", fontsize=6, color=INK2)
pr, tr = imp["police_rate_ratio"], imp["true_rate_ratio"]
ax.vlines(pr, tr[1], tr[2], color=INK, lw=2); ax.plot(pr, tr[0], "D", color=INK, ms=6.5, mec="white", mew=0.6, zorder=4)
ax.annotate("India", (pr, tr[0]), xytext=(6, -10), textcoords="offset points", fontsize=7, fontweight="bold", color=INK)
ax.set_xscale("log"); ax.set_yscale("log")
for a_ in (ax.xaxis, ax.yaxis):
    a_.set_major_locator(matplotlib.ticker.FixedLocator([0.5, 0.7, 0.85, 1, 1.2, 1.4, 2]))
    a_.set_major_formatter(matplotlib.ticker.FixedFormatter(["0.5", "0.7", "0.85", "1", "1.2", "1.4", "2"]))
    a_.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlim(*xl); ax.set_ylim(*yl)
ax.set_xlabel("Change in police-recorded injury death rate\n(2017–19 ÷ 2016; log scale)")
ax.set_ylabel("Change in true resident injury death rate\n(model; 2017–19 ÷ 2013–16 round; log scale)")
ax.text(0.71, 1.85, "Above the diagonal: police\nunderstate a rise", fontsize=6.5, color=MUTED, va="top")
ax.text(1.49, 0.42, "Below: police understate\na fall (or show a false rise)", fontsize=6.5, color=MUTED, ha="right")
save(fig, "fig5_trend_bias")
