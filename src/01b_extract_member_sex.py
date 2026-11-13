"""Per-household counts of de jure (usual resident) males and females, NFHS-4/5 household recode (hv102_xx, hv104_xx).

Needed for sex-specific person-years. Check: n_male + n_female (+ unknown sex) equals hv012 (number of de jure members).
Usage: python 01b_extract_member_sex.py <scratch_dir>   (DTA unpacked temporarily on C:, then deleted)
Output: data_local/<round>_member_sex.parquet (hhid, n_male, n_female)
"""
import pathlib, sys, zipfile, re
import numpy as np, pandas as pd, pyreadstat

P = pathlib.Path(__file__).resolve().parents[1]
SCR = pathlib.Path(sys.argv[1]); SCR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(P / "src"))
from common import nfhs   # noqa: E402
R = {r: (nfhs(r)[0], nfhs(r)[2], nfhs(r)[1]) for r in ("nfhs5", "nfhs4")}
for r, (z, dta, do) in R.items():
    t = zipfile.ZipFile(z).read(do).decode("latin1")
    names = set(re.findall(r'label variable (\S+)\s+"', t))
    ns = max(int(m.group(1)) for v in names if (m := re.match(r"hv104_(\d+)$", v)))
    cols = ["hhid", "hv012"] + [f"hv102_{k:02d}" for k in range(1, ns + 1)] + [f"hv104_{k:02d}" for k in range(1, ns + 1)]
    f = SCR / dta
    if not f.exists():
        zipfile.ZipFile(z).extract(dta, SCR)
    d, _ = pyreadstat.read_dta(str(f), usecols=cols, apply_value_formats=False)
    res = np.column_stack([d[f"hv102_{k:02d}"].values == 1 for k in range(1, ns + 1)])
    sex = np.column_stack([d[f"hv104_{k:02d}"].values for k in range(1, ns + 1)])
    out = pd.DataFrame({"hhid": d.hhid, "n_male": (res & (sex == 1)).sum(1), "n_female": (res & (sex == 2)).sum(1)})
    gap = d.hv012.values - out.n_male - out.n_female
    print(f"{r}: households {len(out):,}; de jure members check: exact {np.mean(gap == 0):.4%}, max gap {gap.max()}, min {gap.min()}")
    out.to_parquet(P / "data_local" / f"{r}_member_sex.parquet")
    f.unlink()
