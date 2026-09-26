"""
fetch_structures.py
===================
Venue-independent structural asset fetch for the journal-grade overhaul.
Downloads HIV-1/HIV-2 protease crystal structures from RCSB and the
chemical-component definitions for the four ritonavir-boosted inhibitors.

Sources:
  * Structures:  https://files.rcsb.org/download/{ID}.pdb  (fallback: .cif)
  * Ligands:     https://data.rcsb.org/rest/v1/core/chemcomp/{ID}

IDs:
  * 3S45  HIV-2 PR (ROD) + Amprenavir, 1.51 A   -- anchor (pocket source)
  * 3EBZ  HIV-2 PR + Darunavir
  * 2AQU  HIV-2 PR + Saquinavir
  * 4LL3  HIV-2 PR + Atazanavir
  * 3OXC  HIV-2 PR + Lopinavir
  * 1MUI  HIV-1 PR + Saquinavir (indinavir-era benchmark)
  * 1HSG  HIV-1 PR + Indinavir (used by make_fig_structure.py, Fig. 2B)

Ligand chemical components:
  * 017  Darunavir   * DR7  Atazanavir   * ROC  Saquinavir   * AB1  Lopinavir
"""
from __future__ import annotations

import json
import ssl
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
STRUCT = HERE / "structures"
LIG_DIR = STRUCT / "ligands"
STRUCT.mkdir(exist_ok=True)
LIG_DIR.mkdir(exist_ok=True)

STRUCT_IDS = ["3S45", "3EBZ", "2AQU", "4LL3", "3OXC", "1MUI", "1HSG"]
LIG_IDS = {"017": "darunavir", "DR7": "atazanavir",
           "ROC": "saquinavir", "AB1": "lopinavir"}
UA = {"User-Agent": "Mozilla/5.0 (HIV-research asset pipeline)"}
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return r.read()


def fetch_ok(identifier: str, kind: str) -> None:
    print(f"OK   {kind:8s} {identifier}")


def fetch_fail(identifier: str, kind: str, err: str) -> None:
    print(f"FAIL {kind:8s} {identifier}  ({err})")


def get_structure(pdb_id: str) -> None:
    pdb = STRUCT / f"{pdb_id}.pdb"
    cif = STRUCT / f"{pdb_id}.cif"
    if pdb.exists() and pdb.stat().st_size > 0:
        fetch_ok(pdb_id, "pdb(cached)")
        return
    try:
        data = fetch(f"https://files.rcsb.org/download/{pdb_id}.pdb")
        pdb.write_bytes(data)
        fetch_ok(pdb_id, "pdb")
    except Exception as e1:
        try:
            data = fetch(f"https://files.rcsb.org/download/{pdb_id}.cif")
            cif.write_bytes(data)
            fetch_ok(pdb_id, "cif")
        except Exception as e2:
            fetch_fail(pdb_id, "both", f"{e1} | {e2}")


def get_chemcomp(cc_id: str, drug: str) -> None:
    out = LIG_DIR / f"chemcomp_{cc_id}_{drug}.json"
    if out.exists() and out.stat().st_size > 0:
        fetch_ok(f"{cc_id} ({drug})", "chemcomp(cached)")
        return
    try:
        data = fetch(f"https://data.rcsb.org/rest/v1/core/chemcomp/{cc_id}")
        payload = json.loads(data)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        fetch_ok(f"{cc_id} ({drug})", "chemcomp")
    except Exception as e:
        fetch_fail(f"{cc_id} ({drug})", "chemcomp", str(e))


def main() -> None:
    print("=== structures ===")
    for pdb_id in STRUCT_IDS:
        get_structure(pdb_id)
    print("=== chemical components ===")
    for cc_id, drug in LIG_IDS.items():
        get_chemcomp(cc_id, drug)
    print("=== summary ===")
    for f in sorted(STRUCT.glob("*")):
        print(f"{f.stat().st_size:>10,} B  {f.relative_to(HERE)}")


if __name__ == "__main__":
    main()