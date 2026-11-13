"""Parse 'Estimated events and Level of Registration' (CRS annual reports, Statement 2) for 2013-2020.

Each row gives mid-year population ('000), SRS birth rate, estimated births, SRS death rate, estimated deaths and level
of registration (births, deaths, %). The SRS rates are what the Registrar General uses for that year (the footnote
says which SRS year); we keep the footnote text for audit.
Output: data/crs_srs_state_year.csv (one row per state x CRS year), with sums checked against the India row.
"""
import pathlib, re
import fitz, pandas as pd

P = pathlib.Path(__file__).resolve().parents[1]
RAW = P / "data_raw/crs"; OUT = P / "data"; OUT.mkdir(exist_ok=True)
NUM = r"-?[\d,]+(?:\.\d+)?|NA|N\.A\.|-"


def page_lines(doc):
    """The statement page (title 'Statement N: Estimated events and Level of Registration') and the page after it."""
    for i, p in enumerate(doc):
        t = p.get_text()
        if re.search(r"Statement\s*\d+\s*:\s*Estimated\s+events\s+and\s+Level\s+of\s+Registration", t, re.I):
            yield i, t
            if i + 1 < len(doc):
                yield i + 1, doc[i + 1].get_text()
            return


rows, notes = [], []
for f in sorted(RAW.glob("crs_*.pdf")):
    yr = int(f.stem.split("_")[1])
    doc = fitz.open(f)
    pages = list(page_lines(doc))
    if not pages:
        print(yr, "statement not found"); continue
    # take consecutive pages starting at the first hit (table may continue)
    txt = "\n".join(t for _, t in pages[:3])
    head = re.search(r"Statement[^\n]*\n", txt)
    lines = [l.strip() for l in txt.split("\n") if l.strip()]
    i = 0
    got = {}
    while i < len(lines):
        l = lines[i]
        name = re.sub(r"^\d+\.\s*", "", l)
        if re.fullmatch(r"[A-Za-z&.()\- ]{3,}", name) and not re.search(r"Statement|States|Population|Registration|Estimated|SRS|Births|Deaths|Level|Sl|No|Mid Year|Union Territories|UTs", name):
            vals = []
            j = i + 1
            while j < len(lines) and len(vals) < 7 and re.fullmatch(NUM, lines[j].replace(" ", "")):
                vals.append(lines[j].replace(" ", "")); j += 1
            if len(vals) == 7 and name not in got:
                got[name] = vals
                i = j; continue
        i += 1
    foot = re.findall(r"\*[^\n]*SRS[^\n]*(?:\n[^\n]*){0,2}", txt)
    notes.append((yr, foot[:2], [p + 1 for p, _ in pages[:3]]))
    for name, v in got.items():
        def num(x):
            return None if x in ("NA", "N.A.", "-") else float(x.replace(",", ""))
        rows.append(dict(crs_year=yr, state=name, pop_000=num(v[0]), srs_br=num(v[1]), est_births=num(v[2]),
                         srs_dr=num(v[3]), est_deaths=num(v[4]), lor_births=num(v[5]), lor_deaths=num(v[6])))
    print(yr, "pages", [p + 1 for p, _ in pages[:3]], "rows", len(got))

D = pd.DataFrame(rows)
for yr, foot, pg in notes:
    print(yr, "footnote:", " | ".join(re.sub(r"\s+", " ", x)[:200] for x in foot))
# checks: estimated deaths ~= pop x DR; states sum ~= India
D["chk_est"] = (D.pop_000 * D.srs_dr - D.est_deaths).abs() / D.est_deaths
bad = D[D.chk_est > 0.02]
print("rows where est_deaths != pop x DR (>2%):", len(bad))
if len(bad): print(bad[["crs_year", "state", "pop_000", "srs_dr", "est_deaths"]].to_string())
for yr, g in D.groupby("crs_year"):
    ind = g[g.state.str.upper() == "INDIA"]
    rest = g[g.state.str.upper() != "INDIA"]
    if len(ind):
        print(yr, "India est deaths", int(ind.est_deaths.iloc[0]), "sum of units", int(rest.est_deaths.sum()),
              f"({rest.est_deaths.sum() / ind.est_deaths.iloc[0]:.3f})", "units", len(rest), "India DR", ind.srs_dr.iloc[0])
D.drop(columns="chk_est").to_csv(OUT / "crs_srs_state_year.csv", index=False)
