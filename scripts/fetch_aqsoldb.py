"""Download the nine unmerged AqSolDB source sets (Sorkun et al. 2019) to data/aqsoldb_sources/. Idempotent; stdlib only.

Pulled from the AqSolDB repo at a pinned commit. We merge replicates ourselves, so AqSolDB's curated
one-value-per-compound CSV is not used.
Run: uv run scripts/fetch_aqsoldb.py
"""
import csv
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "aqsoldb_sources"
URL = "https://raw.githubusercontent.com/mcsorkun/AqSolDB/8e02b548fd9a78778ff89a5aa9a460d1a289cc3a/data/dataset-{}.csv"

rows = []
for source in "ABCDEFGHI":
    out = OUT / f"dataset-{source}.csv"
    if out.exists():
        print(f"already present: {out}")
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")  # atomic: never leave a partial file
        urllib.request.urlretrieve(URL.format(source), tmp)
        tmp.replace(out)
        print(f"saved: {out}")
    with out.open(newline="") as f:
        rows += list(csv.DictReader(f))
assert len(rows) == 19795 and {"InChIKey", "SMILES", "Solubility"} <= rows[0].keys(), "unexpected AqSolDB source contents"
