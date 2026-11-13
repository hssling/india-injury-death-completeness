"""Murder victims by sex and state, Crime in India Table 2A.3 (total victims, child + adult), 2016 and 2019.

CII 2019 Vol I p.207: per state 8 numbers = adult victims (M, F, Trans, Total) then total victims (M, F, Trans, Total).
CII 2016 p.144: per state 12 numbers = 45-60 (M,F,T), 60+ (M,F,T), adult (M,F,T), total victims (M,F,T).
Checks: state sums equal the printed All-India total victims (2019: M 21,277, F 8,641; 2016: M 22,608, F 8,888).
Output: data/cii_murder_victims_sex.csv (unit, year, male, female, total, male_share). Used to split murder cases by sex
(2019 shares for 2017-19; 2016 shares for 2016).
"""
import pathlib, re
import fitz, pandas as pd

P = pathlib.Path(__file__).resolve().parents[1]
C = P / "data_raw/cii"
U = {"A&N ISLANDS": "A & N ISLANDS", "A & N ISLANDS": "A & N ISLANDS", "DELHI": "DELHI (UT)", "DELHI UT": "DELHI (UT)",
     "D&N HAVELI": "DNH & DD", "DAMAN & DIU": "DNH & DD", "JAMMU & KASHMIR": "JAMMU & KASHMIR (incl. LADAKH)"}
SRC = [("Crime_in_India_2019_Volume_I_7dfa44948c.pdf", 207, 2019, 8, (4, 5), (21277, 8641)),
       ("1653886924_Crime in India - 2016 Complete PDF 291117.pdf", 144, 2016, 12, (9, 10), (22608, 8888))]
rows = []
for f, page, yr, n, (im, iff), (TM, TF) in SRC:
    L = [x.strip() for x in fitz.open(C / f)[page - 1].get_text().split("\n") if x.strip()]
    got = []
    for i, l in enumerate(L):
        name = re.sub(r"^\d+\s+", "", l)
        if re.fullmatch(r"[A-Za-z&. ]{3,}", name) and not re.match(r"(TOTAL|STATES|UNION|State|Adult|Victims|Percentage|S\.|No|TABLE|Total)", name):
            v = L[i + 1:i + 1 + n]
            if len(v) == n and all(re.fullmatch(r"\d+", x) for x in v):
                nm = name.upper().strip()
                got.append(dict(year=yr, state=nm, unit=U.get(nm, nm), male=int(v[im]), female=int(v[iff])))
    g = pd.DataFrame(got)
    assert g.male.sum() == TM and g.female.sum() == TF, (yr, g.male.sum(), g.female.sum())
    print(f"{yr}: {len(g)} states/UTs; male {g.male.sum():,} female {g.female.sum():,} = printed All-India")
    rows.append(g)
D = pd.concat(rows).groupby(["unit", "year"], as_index=False)[["male", "female"]].sum()
D["male_share"] = D.male / (D.male + D.female)
assert D.unit.nunique() == 35
D.to_csv(P / "data/cii_murder_victims_sex.csv", index=False)
print(D.groupby("year").male_share.describe().round(3).to_string())
