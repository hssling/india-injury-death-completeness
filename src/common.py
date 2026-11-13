"""Shared helpers: NFHS state labels, unit harmonisation to the 35-unit ADSI panel, CMC dates."""
import re, zipfile
import numpy as np, pandas as pd


def state_labels(zipf, do):
    t = zipfile.ZipFile(zipf).read(do).decode("latin1")
    blk = t[t.find("label define HV024"):]
    blk = blk[:blk.find(";")]
    return {int(k): v for k, v in re.findall(r'(\d+)\s+"([^"]+)"', blk)}


def unit(name):
    s = re.sub(r"[^A-Z]", "", name.upper().replace("&", "AND"))
    for pat, u in [("ANDAMAN", "A & N ISLANDS"), ("ANISLANDS", "A & N ISLANDS"), ("DELHI", "DELHI (UT)"),
                   ("DADRA", "DNH & DD"), ("DAMAN", "DNH & DD"), ("DNH", "DNH & DD"), ("JAMMU", "JAMMU & KASHMIR (incl. LADAKH)"),
                   ("LADAKH", "JAMMU & KASHMIR (incl. LADAKH)")]:
        if pat in s:
            return u
    return name.upper().strip()


def cmc(y, m):
    return (y - 1900) * 12 + m


def ratio_se(y, x, w, psu, strat):
    X = (w * x).sum(); R = (w * y).sum() / X
    z = pd.DataFrame({"z": w * (y - R * x) / X, "psu": psu, "h": strat}).groupby(["h", "psu"]).z.sum().reset_index()
    v = 0.0
    for _, g in z.groupby("h"):
        n = len(g)
        if n > 1:
            v += n / (n - 1) * ((g.z - g.z.mean()) ** 2).sum()
    return R, np.sqrt(v)


# National SRS crude death rate, India, 2019 (SRS Bulletin vol. 54 no. 1, Oct 2021; confirmed by two secondary sources,
# primary PDF not opened). SRS 2018 national rate (6.2) is read from CRS 2018/2019 statement 2 (crs_srs_state_year.csv).
SRS_INDIA_2019 = 6.0


# Locations of the unit-level microdata (NOT redistributable; obtain from dhsprogram.com and microdata.gov.in).
# Set NFHS_DIR / PLFS_ZIP environment variables, or write data_local/paths.json {"nfhs_dir": ..., "plfs_2020_21_zip": ...}.
import json as _json, os as _os, pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parents[1]


def _paths():
    f = _ROOT / "data_local" / "paths.json"
    return _json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def nfhs(r):
    """(zip, .DO, .DTA) of the NFHS household recode for round 'nfhs5' or 'nfhs4'."""
    base = _os.environ.get("NFHS_DIR") or _paths().get("nfhs_dir") or str(_ROOT / "data_local" / "nfhs")
    return {"nfhs5": (f"{base}/NFHS 5 microdata/IAHR7EDT.zip", "IAHR7EFL.DO", "IAHR7EFL.DTA"),
            "nfhs4": (f"{base}/NFHS 4 microdata/IAHR74DT.zip", "IAHR74FL.DO", "IAHR74FL.DTA")}[r]


def plfs_2020_21_zip():
    return _os.environ.get("PLFS_ZIP") or _paths().get("plfs_2020_21_zip") or str(_ROOT / "data_local" / "plfs" / "CSV_Unit_level_data_PLFS_July2020_June2021.zip")
