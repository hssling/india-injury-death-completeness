"""Extract the NFHS-4/5 household death rosters (one row per reported death) into data_local/.

Unit-level NFHS data are used under the DHS Program terms and are NOT redistributed: outputs in data_local/ stay
off any public repository. The 4-5 GB .DTA files are unpacked temporarily to a scratch folder on C: (D: lacks space),
read with only the needed columns, and deleted.

Usage: python 01_extract_roster.py <scratch_dir>
"""
import pathlib, sys, zipfile, re, shutil, time
import pandas as pd, pyreadstat

P = pathlib.Path(__file__).resolve().parents[1]
OUT = P / "data_local"; OUT.mkdir(exist_ok=True)
SCR = pathlib.Path(sys.argv[1]); SCR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(P / "src"))
from common import nfhs   # noqa: E402  (microdata location from data_local/paths.json or NFHS_DIR)

ROUNDS = {
    "nfhs5": dict(zip=pathlib.Path(nfhs("nfhs5")[0]), dta=nfhs("nfhs5")[2], do=nfhs("nfhs5")[1],
                  any="sh87", n="sh88",
                  slot={"sex": "sh90", "reg": "sh91", "age_u": "sh92u", "age_n": "sh92n", "mo": "sh93m", "yr": "sh93y",
                        "mat": "sh94", "inj": "sh95"}),
    "nfhs4": dict(zip=pathlib.Path(nfhs("nfhs4")[0]), dta=nfhs("nfhs4")[2], do=nfhs("nfhs4")[1],
                  any="sh70", n="sh71",
                  slot={"sex": "sh73", "age_u": "sh74u", "age_n": "sh74n", "mo": "sh75m", "yr": "sh75y",
                        "mat": "sh77", "inj": "sh76"}),
}
HH = ["hhid", "hv001", "hv002", "hv005", "hv006", "hv007", "hv008", "hv012", "hv021", "hv022", "hv024", "hv025", "hv270"]


def labels(z, do):
    t = zipfile.ZipFile(z).read(do).decode("latin1")
    return {n for n, _ in re.findall(r'label variable (\S+)\s+"([^"]*)"', t)}


for rnd, c in ROUNDS.items():
    t0 = time.time()
    have = labels(c["zip"], c["do"])
    nslot = max(int(m.group(1)) for v in have if (m := re.match(rf"{c['slot']['inj']}_(\d+)$", v)))
    dist = [d for d in ("shdist", "shdistri", "sdist") if d in have][:1]
    cols = HH + dist + [c["any"], c["n"]] + [f"{v}_{k}" for v in c["slot"].values() for k in range(1, nslot + 1)]
    miss = [x for x in cols if x not in have]
    assert not miss, (rnd, miss)
    dta = SCR / c["dta"]
    if not dta.exists():
        print(rnd, "unpacking", c["dta"]); zipfile.ZipFile(c["zip"]).extract(c["dta"], SCR)
    hh, meta = pyreadstat.read_dta(str(dta), usecols=cols, apply_value_formats=False)
    print(rnd, "households", len(hh), "slots", nslot, f"{time.time() - t0:.0f}s")
    hh.to_parquet(OUT / f"{rnd}_households.parquet")      # household frame (denominators)
    # value labels for the slot variables (codes only are stored)
    vl = {v: meta.variable_value_labels.get(f"{v}_1", {}) for v in c["slot"].values()}
    pd.Series({k: str(v) for k, v in vl.items()}).to_json(OUT / f"{rnd}_value_labels.json")
    # long file: one row per death slot that is filled
    rows = []
    for k in range(1, nslot + 1):
        d = hh[HH + dist].copy()
        for name, v in c["slot"].items():
            d[name] = hh[f"{v}_{k}"]
        d["slot"] = k
        rows.append(d[d["yr"].notna() | d["inj"].notna() | d["sex"].notna()])
    deaths = pd.concat(rows, ignore_index=True)
    deaths["n_deaths_hh"] = deaths.hhid.map(hh.set_index("hhid")[c["n"]])
    deaths.to_parquet(OUT / f"{rnd}_deaths.parquet")
    print(rnd, "death rows", len(deaths), "households reporting >slots deaths:",
          int((hh[c["n"]] > nslot).sum()), "max", hh[c["n"]].max())
    dta.unlink()
shutil.rmtree(SCR, ignore_errors=True)
