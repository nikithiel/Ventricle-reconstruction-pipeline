#!/usr/bin/env python3
"""Tests for addons/valve_geometry_check.py, runnable without Blender (standard library only).

  python tools/test_valve_area_check.py

1. Template areas read from GVR-Pipeline.blend (tools/valve_area_diagnosis.py).
2. The Blender front end run_valve_area_check() against a minimal bpy mock. The scene holds
   the exported example geometry (dataset/example/export: ventricle_*, atrium, aorta) and the
   A4 porous zones from the .blend, placed exactly like scale_rotate_translate_object() does.

The expected values describe the CURRENT pipeline: the MV disc is 81.6 % of
pi * r_long * r_small because the MV template is D-shaped. After the pipeline fix (area
coefficient instead of pi in calculate_valve_diameters.py) the MV expectations here change.
"""
from __future__ import annotations

import math
import re
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "addons"))
sys.path.insert(0, str(REPO / "tools"))
import valve_geometry_check as vgc  # noqa: E402
import valve_area_diagnosis as diag  # noqa: E402

SESSION = REPO / "dataset/example/Session_Logs/session_2026-10-06_13-45-54.json"
EXPORT = REPO / "dataset/example/export"
FAILURES: list = []


def check(cond: bool, what: str) -> None:
    print(("PASS  " if cond else "FAIL  ") + what)
    if not cond:
        FAILURES.append(what)


def close(a, b, rel=2e-3) -> bool:
    return a is not None and abs(a - b) <= rel * abs(b)


def disc_note(text: str, name: str):
    """(height mm, share %) from the atrium/aorta note, or (None, None)."""
    m = re.search(rf"'{name}' (?:bottom disc ([0-9.]+) mm (?:above|below) the valve plane|disc in the valve plane)"
                  rf"[^:]*: [0-9.]+ mm2 \(([0-9.]+) % of the ventricle disc\)", text)
    if not m:
        return None, None
    return (float(m.group(1)) if m.group(1) else 0.0), float(m.group(2))


# -----------------------------------------------------------------------------
# minimal bpy mock
# -----------------------------------------------------------------------------
class _Mat:
    def __matmul__(self, co):
        return tuple(co)


class _G:
    def __init__(self, group, weight=1.0):
        self.group, self.weight = group, weight


class _V:
    def __init__(self, i, co, groups):
        self.index, self.co, self.groups = i, co, groups


class _P:
    def __init__(self, verts):
        self.vertices = verts


class _VG:
    def __init__(self, name, index):
        self.name, self.index = name, index


class _Dict(dict):
    def get(self, name, default=None):
        return dict.get(self, name, default)


class _Objects(_Dict):
    def __iter__(self):
        return iter(self.values())


class _Obj:
    def __init__(self, name, verts, faces, vgroups=None):
        self.name, self.type, self.parent, self.matrix_world = name, "MESH", None, _Mat()
        vgroups = vgroups or {}
        self.vertex_groups = _Dict({n: _VG(n, k) for k, n in enumerate(vgroups)})
        per_vertex = {}
        for k, members in enumerate(vgroups.values()):
            for i in members:
                per_vertex.setdefault(i, []).append(_G(k))
        self.data = types.SimpleNamespace(
            vertices=[_V(i, co, per_vertex.get(i, [])) for i, co in enumerate(verts)],
            polygons=[_P(tuple(f)) for f in faces])


def build_scene(session, blend, with_zones=True, vessels="export", vessel_shift=0.0):
    """Scene of 12 ventricle frames plus atrium/aorta and (optionally) the A4 zones.

    vessels="export": atrium/aorta from dataset/example/export. That export has the vessel discs
    in the valve plane (like A3), so it does not match an A4 scene. vessels="A4": atrium/aorta
    placed from A4_Atrium/A4_Aorta, which start on top of the porous zones (template z = 0.1).
    vessel_shift moves the placed vessels along the valve normal (mm), to provoke a gap.
    """
    def disc_members(verts, faces, valve):
        s = session[valve]
        frame = vgc.valve_frame(s["translation"], s["angles"])
        loc = [frame.to_local(v) for v in verts]
        tol, r_max = 1e-3 * max(s["r_long"], s["r_small"]), 1.3 * max(s["r_long"], s["r_small"])
        members = set()
        for f in faces:
            if all(abs(loc[i][2]) <= tol and math.hypot(loc[i][0], loc[i][1]) <= r_max for i in f):
                members.update(f)
        return members

    def placed(template, valve, shift=0.0):  # what scale_rotate_translate_object() does with ratio = 1
        verts, faces, _ = blend.mesh(template)
        s = session[valve]
        sx, sy = s["r_long"], s["r_small"]
        sz = (sx + sy) / 2
        R = vgc.euler_xyz_matrix([math.radians(a) for a in s["angles"]])
        t = s["translation"]
        out = []
        for x, y, z in verts:
            p = (x * sx, y * sy, z * sz + shift)
            out.append(tuple(sum(R[r][c] * p[c] for c in range(3)) + t[r] for r in range(3)))
        return out, faces

    objects = _Objects()
    ventricles = []
    for k in range(12):
        verts, faces = diag.read_stl(EXPORT / f"ventricle_{k}.stl")
        o = _Obj(f"ventricle_{k}", verts, faces,
                 {"MV": disc_members(verts, faces, "MV"), "AV": disc_members(verts, faces, "AV")})
        objects[o.name] = o
        ventricles.append(o)
    for n, template, valve in (("atrium", "Atrium", "MV"), ("aorta", "Aorta", "AV")):
        if vessels == "export":
            verts, faces = diag.read_stl(EXPORT / f"{n}.stl")
        else:
            verts, faces = placed(f"{vessels}_{template}", valve, vessel_shift)
        objects[n] = _Obj(n, verts, faces)
    if with_zones:
        for valve in ("MV", "AV"):
            for z in vgc.ZONE_TYPES:
                v, f = placed(f"A4_{valve}_{z}", valve)
                objects[f"{valve.lower()}_{z}"] = _Obj(f"{valve.lower()}_{z}", v, f)
    for approach in (3, 4):
        for valve in ("MV", "AV"):
            for part in ("Surface",) + vgc.ZONE_TYPES:
                name = f"A{approach}_{valve}_{part}"
                v, f, _ = blend.mesh(name)
                objects[name] = _Obj(name, v, f)
    return objects, ventricles


def run_check(objects, ventricles, session, approach=4, imperm_open=True, import_dir=None):
    bpy = types.ModuleType("bpy")
    bpy.data = types.SimpleNamespace(objects=objects)
    bpy.path = types.SimpleNamespace(abspath=lambda p: p)
    sys.modules["bpy"] = bpy
    # calculate_valve_diameters imports numpy/pandas/trimesh; only resolve_case_dir is needed here.
    cvd = types.ModuleType("calculate_valve_diameters")
    cvd.resolve_case_dir = lambda stl_dir: Path(stl_dir).parent
    sys.modules["calculate_valve_diameters"] = cvd
    mv, av = session["MV"], session["AV"]
    scene = types.SimpleNamespace(
        approach=approach, valve_check_imperm_open=imperm_open, valve_check_tolerance=2.0,
        ventricle_import_dir=import_dir if import_dir is not None else str(REPO / "dataset/example/STL"),
        mitral_radius_long=mv["r_long"], mitral_radius_small=mv["r_small"], aortic_radius=av["r_long"],
        translation_mitral=mv["translation"], angle_mitral=mv["angles"],
        translation_aortic=av["translation"], angle_aortic=av["angles"])
    return vgc.run_valve_area_check(types.SimpleNamespace(scene=scene), ventricles)


def main() -> int:
    blend = diag.BlendFile(REPO / "GVR-Pipeline.blend")
    templates = diag.load_templates(blend)

    print("-- template areas (unit scale)")
    a4mv, a4av = templates[(4, "MV")].areas, templates[(4, "AV")].areas
    check(close(a4mv.ring, 2.5643, 1e-4), f"A4 MV ring a = {a4mv.ring:.4f} (expected 2.5643)")
    check(close(a4mv.open, 2.0042, 1e-4), f"A4 MV perm+res = {a4mv.open:.4f} (expected 2.0042)")
    check(close(a4av.ring, 3.1214, 1e-4), f"A4 AV ring a = {a4av.ring:.4f} (expected 3.1214)")
    check(close(a4av.open, 2.5284, 1e-4), f"A4 AV perm+res = {a4av.open:.4f} (expected 2.5284)")
    a3mv = templates[(3, "MV")].areas
    check(close(a3mv.open, a4mv.open, 1e-4), "A3 MV zones (flat) equal the A4 MV zones (prisms)")

    session = diag.load_session(SESSION)
    objects, ventricles = build_scene(session, blend, vessels="A4")

    print("-- Blender front end, A4 with zone objects, no summary.csv (target = GUI radii)")
    ok, lines, short = run_check(objects, ventricles, session, import_dir="/nonexistent")
    text = "\n".join(lines)
    check(not ok, "MV is reported (WARN) with the current pipeline")
    check("MV: target 34.49 mm2 [GUI radii (no summary.csv)]  ->  WARN" in text, "MV target pi*3.6*3.05 and WARN")
    check("AV: target 26.24 mm2 [GUI radii (no summary.csv)]  ->  OK" in text, "AV within tolerance (OK)")
    check("ring      28.20 mm2 = 81.7 % of target" in text, "MV disc 28.20 mm2 = 81.7 %")
    check("of target   (12 frames, identical)" in text, "ring measured in all 12 frames, identical")
    check("zones: zone objects" in text, "zones measured from the mv_*/av_* objects")
    sz_mv = (session["MV"]["r_long"] + session["MV"]["r_small"]) / 2
    h, share = disc_note(text, "atrium")
    check(close(h, 0.1 * sz_mv, 1e-2) and "'atrium' bottom disc" in text and "on top of the porous zones" in text,
          f"A4 atrium found on top of the MV zones at {h} mm (0.1 * s_z = {0.1 * sz_mv:.4f} mm)")
    check(close(share, 100.0, 5e-3), f"A4 atrium bottom disc = ventricle disc ({share} %, export vs. template)")
    h, share = disc_note(text, "aorta")
    check(close(h, 0.1 * session["AV"]["r_long"], 1e-2) and text.count("on top of the porous zones") == 2,
          f"A4 aorta found on top of the AV zones at {h} mm")
    check("no bottom disc found" not in text and "The porous zones end" not in text, "no interface note in A4")

    print("-- Blender front end, A4, a single frame selected")
    ok, lines, short = run_check(objects, ventricles[:1], session, import_dir="/nonexistent")
    text = "\n".join(lines)
    check("of target   (1 frame)" in text, "ring line says 1 frame")

    print("-- Blender front end, A4, atrium/aorta lifted 0.05 mm off the zones (gap)")
    objects_gap, ventricles_gap = build_scene(session, blend, vessels="A4", vessel_shift=0.05)
    ok, lines, short = run_check(objects_gap, ventricles_gap, session, import_dir="/nonexistent")
    text = "\n".join(lines)
    h, _ = disc_note(text, "atrium")
    check(close(h, 0.1 * sz_mv + 0.05, 1e-2) and "on top of the porous zones" not in text,
          f"gap: atrium height {h} mm reported, not 'on top of the zones'")
    check(text.count("The porous zones end") == 2 and "not at this disc." in text,
          "gap between zones and atrium/aorta is reported")

    print("-- Blender front end, A4, imperm as wall, with summary.csv")
    ok, lines, short = run_check(objects, ventricles, session, imperm_open=False)
    text = "\n".join(lines)
    check("Reference: perm+res (imperm is a wall)" in text, "switch selects perm+res")
    check("The radii used for the geometry differ from summary.csv" in text,
          "note about GUI radii vs. summary.csv (the example summary belongs to another run)")
    check("summary.csv (A_fit)" in text, "target taken from summary.csv")

    print("-- Blender front end, A3 without zone objects (estimate from templates)")
    objects3, ventricles3 = build_scene(session, blend, with_zones=False)
    ok, lines, short = run_check(objects3, ventricles3, session, approach=3, import_dir="/nonexistent")
    text = "\n".join(lines)
    check("estimated from A3 templates" in text, "A3 zones estimated from the templates")
    check("perm+res  22.04 mm2 = 63.9 % of target" in text, "A3 MV perm+res estimate 22.04 mm2")
    check("'atrium' disc in the valve plane: 28.20 mm2 (100.0 % of the ventricle disc)" in text,
          "vessel disc in the valve plane (export, like A3) matches the ventricle disc")

    print("-- Blender front end, A5 (MV not checked)")
    ok, lines, short = run_check(objects, ventricles, session, approach=5, import_dir="/nonexistent")
    text = "\n".join(lines)
    check("A5 builds the MV as a 3D valve" in text, "A5 MV skipped with a note")
    check("AV: target 26.24 mm2 [GUI radii (no summary.csv)]  ->  OK" in text, "A5 AV still checked")

    print()
    print(f"{len(FAILURES)} failure(s)" if FAILURES else "all checks passed")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
