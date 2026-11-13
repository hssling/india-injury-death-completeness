"""Murder cases by state, 2014-2019, from Crime in India Table 2A.1 (user-saved PDFs in data_raw/cii/).

CII 2016 Table 2A.1 (PDF p.138): 2014, 2015, 2016.   CII 2019 Vol I Table 2A.1 (PDF p.199): 2017, 2018, 2019.
Note (CII 2019): West Bengal 2019 = 2018 data (non-receipt). Culpable homicide not amounting to murder (~3,000/yr
nationally) is not included. Cases, not victims (victims are close; Table 2A.3).
Checks: units sum to the printed TOTAL ALL INDIA for every year.
Output: data/cii_murders_state_year.csv (unit names harmonised to the ADSI 35-unit panel).
"""
import pathlib, re
import fitz, pandas as pd

P = pathlib.Path(__file__).resolve().parents[1]
C = P / "data_raw/cii"
SRC = [("1653886924_Crime in India - 2016 Complete PDF 291117.pdf", 138, [2014, 2015, 2016]),
       ("Crime_in_India_2019_Volume_I_7dfa44948c.pdf", 199, [2017, 2018, 2019])]
U = {"A&N ISLANDS": "A & N ISLANDS", "A & N ISLANDS": "A & N ISLANDS", "DELHI": "DELHI (UT)", "DELHI UT": "DELHI (UT)",
     "D&N HAVELI": "DNH & DD", "DAMAN & DIU": "DNH & DD", "JAMMU & KASHMIR": "JAMMU & KASHMIR (incl. LADAKH)"}
rows = []
for f, page, yrs in SRC:
    t = fitz.open(C / f)[page - 1].get_text()
    assert re.search(r"TABLE 2A\.1", t), (f, page)
    L = [x.strip() for x in t.split("\n") if x.strip()]
    tot = None
    for i, l in enumerate(L):
        if re.fullmatch(r"[A-Za-z&. ]{3,}", l) and not re.match(r"(TOTAL|STATES|UNION|State|Murder|Rate|Mid|Percentage|S\.|No|Note|TABLE)", l):
            v = L[i + 1:i + 4]
            if all(re.fullmatch(r"\d+", x) for x in v):
                name = l.upper().strip()
                for y, x in zip(yrs, v):
                    rows.append(dict(year=y, state=name, unit=U.get(name, name), murders=int(x)))
        if l.startswith("TOTAL ALL INDIA"):
            tot = [int(x) for x in L[i + 1:i + 4]]
    got = pd.DataFrame(rows)
    for y, T in zip(yrs, tot):
        s = got[got.year == y].murders.sum()
        assert s == T, (f, y, s, T)
        print(f"{y}: {got[got.year == y].state.nunique()} units, sum {s:,} = printed All-India {T:,}")
D = pd.DataFrame(rows).groupby(["unit", "year"], as_index=False).murders.sum()
assert D.unit.nunique() == 35, sorted(D.unit.unique())
D.to_csv(P / "data/cii_murders_state_year.csv", index=False)
print(D.pivot(index="unit", columns="year", values="murders").loc[["BIHAR", "UTTAR PRADESH", "WEST BENGAL", "DNH & DD"]])
