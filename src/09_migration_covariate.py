"""State in-migration from PLFS 2020-21 (first visit, Block 7.1), as a place-of-occurrence covariate.

Inter-state in-migrant = present place differs from last usual place of residence (b7pt1_q3 = 1), last UPR in another
state (b7pt1_q6 in 5, 6), and moved before April 2020 (b7pt1_q4 != 1; excludes COVID-era return migration).
Weights: MULT/100 if NSS == NSC else MULT/200, divided by NO_QTR (PLFS README; as in the heat project's plfs_io.py).
Output: data/plfs_inmigration_2020_21.csv with in-migrant share of all persons and of males aged 15-59.
PLFS unit-level data are used only for these state aggregates and are not redistributed.
"""
import pathlib, sys, zipfile
import numpy as np, pandas as pd

P = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P / "src"))
from common import unit   # noqa: E402
from common import plfs_2020_21_zip   # noqa: E402
RAW = pathlib.Path(plfs_2020_21_zip())
CODES = {"28": "ANDHRA PRADESH", "12": "ARUNACHAL PRADESH", "18": "ASSAM", "10": "BIHAR", "22": "CHHATTISGARH", "30": "GOA",
         "24": "GUJARAT", "06": "HARYANA", "02": "HIMACHAL PRADESH", "20": "JHARKHAND", "29": "KARNATAKA", "32": "KERALA",
         "23": "MADHYA PRADESH", "27": "MAHARASHTRA", "14": "MANIPUR", "17": "MEGHALAYA", "15": "MIZORAM", "13": "NAGALAND",
         "21": "ODISHA", "03": "PUNJAB", "08": "RAJASTHAN", "11": "SIKKIM", "33": "TAMIL NADU", "36": "TELANGANA", "16": "TRIPURA",
         "09": "UTTAR PRADESH", "05": "UTTARAKHAND", "19": "WEST BENGAL", "35": "A & N ISLANDS", "04": "CHANDIGARH",
         "07": "DELHI", "31": "LAKSHADWEEP", "34": "PUDUCHERRY", "01": "JAMMU & KASHMIR", "37": "LADAKH",
         "25": "DADRA & NAGAR HAVELI", "26": "DAMAN & DIU"}       # same codes as plfs_io.UNITS in the heat project
z = zipfile.ZipFile(RAW)
cols = ["state_perv1", "b4q5_perv1", "b4q6_perv1", "b7pt1_q3_perv1", "b7pt1_q4_perv1", "b7pt1_q6_perv1",
        "mult_perv1", "NSS_perv1", "NSC_perv1", "no_qtr_perv1"]
d = pd.read_csv(z.open("CSV_Unit_level_data_PLFS_July2020_June2021/perv1.csv"), usecols=cols, dtype=str)
d.columns = [c.replace("_perv1", "").lower() for c in d.columns]
num = lambda c: pd.to_numeric(d[c], errors="coerce")
d["unit"] = d.state.str.zfill(2).map(CODES).map(unit)
assert d.unit.notna().all(), d.state[d.unit.isna()].unique()
w = np.where(num("nss") == num("nsc"), num("mult") / 100, num("mult") / 200) / num("no_qtr")
mig = (num("b7pt1_q3") == 1) & num("b7pt1_q6").isin([5, 6]) & (num("b7pt1_q4") != 1)
male_wa = (num("b4q5") == 1) & num("b4q6").between(15, 59)
d = d.assign(w=w, mig=mig.astype(int), mwa=male_wa.astype(int))
g = d.groupby("unit").apply(lambda x: pd.Series({
    "persons_w": x.w.sum(), "inmig_share": np.average(x.mig, weights=x.w),
    "inmig_share_male15_59": np.average(x.mig[x.mwa == 1], weights=x.w[x.mwa == 1]) if x.mwa.sum() else np.nan,
    "n_persons": len(x)}), include_groups=False)
nat = np.average(d.mig, weights=d.w)
print(f"PLFS 2020-21 persons {len(d):,}; national inter-state in-migrant share {nat:.3%} (weighted pop {d.w.sum() / 1e9:.2f} bn)")
print(g.sort_values("inmig_share", ascending=False)[["inmig_share", "inmig_share_male15_59", "n_persons"]].round(4).to_string())
assert len(g) == 35
g.to_csv(P / "data/plfs_inmigration_2020_21.csv")
