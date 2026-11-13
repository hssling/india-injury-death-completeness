"""Police (NCRB) injury deaths and person-years per state x survey round, from data/ only.

Injury deaths match the NFHS item (accident, violence, poisoning, homicide, suicide):
  ADSI other-causes 'Total' - Sudden Deaths (Total) - Deaths of Women during Pregnancy (Total)
  + forces of nature excluding Epidemic and Starvation + suicides + murder cases (Crime in India table 2A.1).
ADSI reclassified accidental deaths in 2016 (poisoning, animal bites, overdose, spurious liquor, suffocation and sudden
deaths became categories), so the NFHS-4 round is compared with NCRB 2016; NFHS-5 with 2017-2019.
Outputs (model inputs):
  outputs/ncrb_window.csv           main definition
  outputs/ncrb_window_strict.csv    also excludes 'Causes Not Known'
  outputs/ncrb_window_2014_16.csv   NFHS-4 compared with NCRB 2014-2016 (mixed classification)
Every component must be present for all 35 units x years (a unit-name mismatch once zeroed J&K suicides silently).
"""
import pathlib
import pandas as pd

P = pathlib.Path(__file__).resolve().parents[1]; D = P / "data"; O = P / "outputs"; O.mkdir(exist_ok=True)
adsi = pd.read_csv(D / "adsi_state_year_sex.csv"); adsi = adsi[adsi.sex == "total"]
oc = adsi[adsi.table == "other_causes"].pivot_table(index=["unit", "year"], columns="cause", values="n", aggfunc="sum").fillna(0)
fon = adsi[(adsi.table == "forces_of_nature") & ~adsi.cause.isin(["Epidemic", "Starvation due to Natural Calamity"])]
fon = fon.groupby(["unit", "year"]).n.sum()
sui = pd.read_csv(D / "ncrb_suicides_state_year.csv").set_index(["unit", "year"]).total
mur = pd.read_csv(D / "cii_murders_state_year.csv").set_index(["unit", "year"]).murders
pop = pd.read_csv(D / "state_population_units.csv").set_index(["unit", "year"]).population
for tag, noninj, y4 in [("", [], [2016]), ("_strict", ["Causes Not Known"], [2016]), ("_2014_16", [], [2014, 2015, 2016])]:
    acc = (oc["Total"] - oc[["Sudden Deaths (Total)", "Deaths of Women during Pregnancy (Total)"] + noninj].sum(axis=1)).add(fon, fill_value=0)
    assert (acc > 0).all()
    n = pd.DataFrame({"acc": acc, "sui": sui, "mur": mur}).join(pop.rename("pop"), how="left")
    n = n[n.index.get_level_values("year") <= 2019]
    assert n[["acc", "sui", "mur", "pop"]].notna().all().all(), n[n.isna().any(axis=1)]
    assert n.index.get_level_values("unit").nunique() == 35
    win = []
    for r, yrs in [("nfhs5", [2017, 2018, 2019]), ("nfhs4", y4)]:
        x = n.loc[n.index.get_level_values("year").isin(yrs)].groupby("unit").sum()
        win.append(x.assign(round=r, years=f"{min(yrs)}-{max(yrs)}", ncrb=x.acc + x.sui + x.mur)
                   .rename(columns={"pop": "person_years"}).reset_index())
    W = pd.concat(win)
    W.to_csv(O / f"ncrb_window{tag}.csv", index=False)
    print(f"ncrb_window{tag}.csv: " + "; ".join(f"{r} {g.ncrb.sum():,.0f} deaths, {1e5 * g.ncrb.sum() / g.person_years.sum():.1f}/100k"
                                               for r, g in W.groupby("round")))
