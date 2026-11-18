#!/usr/bin/env python3
"""T3: the computation behind 'Calculate Radii' (calculate_valve_diameters.main_calc_diameter).

Runs on a copy of the synthetic example (dataset/example) in a temporary folder; the
repository is not touched. Needs numpy, pandas, scipy, matplotlib and trimesh, i.e. the
Python that Calculate Radii runs in:

  python tools/test_calculate_valve_diameters.py

Checks the summary.csv contract (exact header, positive areas, R_target, approach), that
width, depth and AV diameter build A_fit with the template shapes, and the warning for a
Doppler CSV with the opposite sign convention.
"""
from __future__ import annotations

import csv
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "addons"))
sys.path.insert(0, str(REPO / "tools"))
import calculate_valve_diameters as cvd  # noqa: E402
import valve_geometry as vgc  # noqa: E402
from test_valve_area_check import A4_AV, A4_MV, FAILURES, check, close  # noqa: E402  (T1 reference outlines)

EXAMPLE = REPO / "dataset/example"
HEADER = ["path", "bpm_mv [1/min]", "bpm_av [1/min]", "approach", "A_MV [m^2]", "A_AV [m^2]", "R_target [-]",
          "width_MV [m]", "depth_MV [m]", "d_AV [m]"]
# A_fit of the example from an earlier run (the tracked dataset/example/calc_valve_diameters_outputs/summary.csv).
A_MV, A_AV = 5.022664e-05, 4.113095e-05
REL = 1e-4


def copy_case(tmp: Path, name: str) -> Path:
    case = tmp / name
    case.mkdir()
    shutil.copytree(EXAMPLE / "STL", case / "STL", ignore=shutil.ignore_patterns("rotated"))
    for f in ("inputPython.txt", "synthetic_AV.csv", "synthetic_MV.csv"):
        shutil.copy(EXAMPLE / f, case / f)
    return case


def flip_sign(csv_path: Path) -> None:
    """Negate the velocity column of a Doppler CSV ('t; v', decimal comma)."""
    out = []
    for line in csv_path.read_text().splitlines():
        t, v = line.split(";")
        v = v.strip()
        out.append(f"{t}; {v[1:] if v.startswith('-') else '-' + v}")
    csv_path.write_text("\n".join(out) + "\n")


def run(case: Path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, fit = cvd.main_calc_diameter(case / "STL", 400, approach=4, R_target=1.3,
                                        shapes={"MV": A4_MV, "AV": A4_AV})
    with (case / "calc_valve_diameters_outputs" / "summary.csv").open(newline="") as fh:
        rows = list(csv.reader(fh))
    return fit, rows, [str(w.message) for w in caught]


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        print("-- T3 synthetic example, A4, R_target 1.3")
        fit, rows, msgs = run(copy_case(tmp, "case"))
        check(rows[0] == HEADER, f"summary.csv header exactly as agreed: {','.join(rows[0])}")
        check(len(rows) == 2 and all(c.isascii() for c in rows[0]), "one data row, ASCII header")
        row = dict(zip(rows[0], rows[1]))
        a_mv, a_av = float(row["A_MV [m^2]"]), float(row["A_AV [m^2]"])
        check(close(a_mv, A_MV, REL) and close(a_av, A_AV, REL), f"A_MV {a_mv:.6e}, A_AV {a_av:.6e} m^2: positive magnitudes")
        check(row["approach"] == "4" and float(row["R_target [-]"]) == 1.3, "approach 4 and R_target 1.3 written")
        w, d, dav = float(row["width_MV [m]"]), float(row["depth_MV [m]"]), float(row["d_AV [m]"])
        check(close(w / d, 1.3, REL) and close(A4_MV.fill * w * d, A_MV, REL), f"MV {w * 1e3:.3f} x {d * 1e3:.3f} mm builds A_MV")
        check(close(A4_AV.area * (dav / 2) ** 2, A_AV, REL), f"AV diameter {dav * 1e3:.3f} mm builds A_AV")
        check((fit.A_MV, fit.A_AV, fit.sizes) == (a_mv, a_av, vgc.ValveSizes(w, d, dav)),
              "named result equals the summary.csv row")
        check(not any("sign" in m for m in msgs), "no sign warning for the example")

        print("-- T3 AV Doppler CSV with the opposite sign convention")
        case = copy_case(tmp, "flipped")
        flip_sign(case / "synthetic_AV.csv")
        _, rows, msgs = run(case)
        row = dict(zip(rows[0], rows[1]))
        check(any("AV" in m and "sign" in m for m in msgs), "warning about the AV sign: " + " | ".join(msgs))
        check(float(row["A_AV [m^2]"]) > 0 and float(row["A_MV [m^2]"]) > 0, "areas still written as positive magnitudes")

    print()
    print(f"{len(FAILURES)} failure(s)" if FAILURES else "all checks passed")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
