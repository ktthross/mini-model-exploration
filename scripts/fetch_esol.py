"""Download the ESOL (Delaney) dataset to data/esol.csv. Idempotent; stdlib only.

TDC does not ship ESOL (only Solubility_AqSolDB), so this pulls the MoleculeNet copy.
Run: uv run scripts/fetch_esol.py
"""
import csv
import urllib.request
from pathlib import Path

URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
OUT = Path(__file__).resolve().parent.parent / "data" / "esol.csv"

if OUT.exists():
    print(f"already present: {OUT}")
else:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")  # atomic: never leave a partial file
    urllib.request.urlretrieve(URL, tmp)
    tmp.replace(OUT)
    print(f"saved: {OUT}")

with OUT.open(newline="") as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 1128 and "smiles" in rows[0], "unexpected ESOL contents"
