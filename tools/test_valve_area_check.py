#!/usr/bin/env python3
"""Tests for addons/valve_geometry.py, runnable without Blender (standard library only).

  python tools/test_valve_area_check.py

T1. Valve sizes: template outlines read from GVR-Pipeline.blend, sizes from A_fit, stretch
    factors, a round trip with the real templates, and the imperm lock.
T2. The Blender front end run_valve_area_check() against a minimal bpy mock. The scenes hold
    the exported example geometry (dataset/example/export) or valve discs built from the
    templates, plus the A4 porous zones, placed exactly like scale_rotate_translate_object().

The tests pass before and after tools/merge_imperm_into_res.py: while the .blend still holds
*_imperm, the test scene adds it to *_res in memory (template_mesh).
"""
from __future__ import annotations

import math
import re
import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "addons"))
sys.path.insert(0, str(REPO / "tools"))
import valve_geometry as vgc  # noqa: E402
import valve_area_diagnosis as diag  # noqa: E402

SESSION = REPO / "dataset/example/Session_Logs/session_2026-10-06_13-45-54.json"
EXPORT = REPO / "dataset/example/export"
# A4 template outlines in GVR-Pipeline.blend (Boundary edge loops at unit scale, measured 2026-11-18).
A4_MV = vgc.TemplateShape(2.56430, 2.00729, 1.68554)
A4_AV = vgc.TemplateShape(3.12145, 2.0, 2.0)
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
    def __init__(self, name, verts, faces, vgroups=None, edges=()):
        self.name, self.type, self.parent, self.matrix_world = name, "MESH", None, _Mat()
        vgroups = vgroups or {}
        self.vertex_groups = _Dict({n: _VG(n, k) for k, n in enumerate(vgroups)})
        per_vertex = {}
        for k, members in enumerate(vgroups.values()):
            for i in members:
                per_vertex.setdefault(i, []).append(_G(k))
        self.data = types.SimpleNamespace(
            vertices=[_V(i, co, per_vertex.get(i, [])) for i, co in enumerate(verts)],
            polygons=[_P(tuple(f)) for f in faces],
            edges=[_P(tuple(e)) for e in edges])


def template_mesh(blend, name):
    """(verts, faces, edges) of a template. While the .blend still holds *_imperm, *_res
    includes it, like tools/merge_imperm_into_res.py does; so the tests pass before and
    after the merge. The double inner walls stay, bottom_faces_area() ignores them."""
    verts, faces, edges = blend.mesh(name)
    imperm = name[:-len("res")] + "imperm"
    if name.endswith("_res") and imperm in blend.objects:
        v2, f2, e2 = blend.mesh(imperm)
        n = len(verts)
        verts = list(verts) + list(v2)
        faces = list(faces) + [tuple(i + n for i in f) for f in f2]
        edges = list(edges) + [(a + n, b + n) for a, b in e2]
    return verts, faces, edges


def template_objects(blend):
    """Mock objects of all valve templates (A3_/A4_/A5_ Boundary, Surface, perm, res), merged."""
    objects = _Objects()
    for name in blend.objects:
        m = diag.TEMPLATE_RE.match(name)
        if m and m.group(3) != "imperm":
            v, f, e = template_mesh(blend, name)
            objects[name] = _Obj(name, v, f, edges=e)
    return objects


def mock_bpy(objects):
    bpy = types.ModuleType("bpy")
    bpy.data = types.SimpleNamespace(objects=objects)
    bpy.path = types.SimpleNamespace(abspath=lambda p: p)
    sys.modules["bpy"] = bpy
    return bpy


def panel_from_session(session, shapes):
    """Panel values that rebuild the example export: the old session log holds stretch factors."""
    (sx, sy), (s_av, _) = session["MV"]["stretch"], session["AV"]["stretch"]
    return {"MV": (sx * shapes["MV"].width, sy * shapes["MV"].depth), "AV": 2 * s_av}


def build_scene(session, blend, panel, ventricles="export", with_zones=True, vessels="export", vessel_shift=0.0):
    """Scene with ventricle frames, atrium/aorta, (optionally) the A4 zones and all templates.

    ventricles="export": the 12 frames of dataset/example/export (built with the old session log).
    ventricles="templates": one frame made of A4_MV_Surface and A4_AV_Surface placed with the panel
    values, i.e. the valve discs the pipeline builds; everything else of a ventricle is not needed.
    vessels="export": atrium/aorta from dataset/example/export. That export has the vessel discs
    in the valve plane (like A3), so it does not match an A4 scene. vessels="A4": atrium/aorta
    placed from A4_Atrium/A4_Aorta, which start on top of the porous zones (template z = 0.1).
    vessel_shift moves the placed vessels along the valve normal (mm), to provoke a gap.
    """
    templates = template_objects(blend)
    shapes = vgc.template_shapes(mock_bpy(templates), 4)
    stretch = {"MV": vgc.mv_stretch(*panel["MV"], shapes["MV"]), "AV": vgc.av_stretch(panel["AV"])}

    def disc_members(verts, faces, valve):
        s = session[valve]
        frame = vgc.valve_frame(s["translation"], s["angles"])
        loc = [frame.to_local(v) for v in verts]
        tol, r_max = 1e-3 * max(s["stretch"]), 1.3 * max(s["stretch"])
        members = set()
        for f in faces:
            if all(abs(loc[i][2]) <= tol and math.hypot(loc[i][0], loc[i][1]) <= r_max for i in f):
                members.update(f)
        return members

    def placed(template, valve, shift=0.0):  # what scale_rotate_translate_object() does with ratio = 1
        verts, faces, _ = template_mesh(blend, template)
        sx, sy, sz = stretch[valve]
        R = vgc.euler_xyz_matrix([math.radians(a) for a in session[valve]["angles"]])
        t = session[valve]["translation"]
        out = []
        for x, y, z in verts:
            p = (x * sx, y * sy, z * sz + shift)
            out.append(tuple(sum(R[r][c] * p[c] for c in range(3)) + t[r] for r in range(3)))
        return out, faces

    objects = _Objects()
    vents = []
    if ventricles == "export":
        for k in range(12):
            verts, faces = diag.read_stl(EXPORT / f"ventricle_{k}.stl")
            vents.append(_Obj(f"ventricle_{k}", verts, faces,
                              {"MV": disc_members(verts, faces, "MV"), "AV": disc_members(verts, faces, "AV")}))
    else:
        verts, faces, groups = [], [], {}
        for valve in ("MV", "AV"):
            v, f = placed(f"A4_{valve}_Surface", valve)
            n = len(verts)
            verts += v
            faces += [tuple(i + n for i in fc) for fc in f]
            groups[valve] = set(range(n, n + len(v)))
        vents.append(_Obj("ventricle_0", verts, faces, groups))
    for o in vents:
        objects[o.name] = o
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
    objects.update(templates)
    return objects, vents


def run_check(objects, ventricles, session, panel, approach=4, import_dir=None):
    mock_bpy(objects)
    # calculate_valve_diameters imports numpy/pandas/trimesh; only resolve_case_dir is needed here.
    cvd = types.ModuleType("calculate_valve_diameters")
    cvd.resolve_case_dir = lambda stl_dir: Path(stl_dir).parent
    sys.modules["calculate_valve_diameters"] = cvd
    mv, av = session["MV"], session["AV"]
    scene = types.SimpleNamespace(
        approach=approach, valve_check_tolerance=2.0, valve_R_target=1.3,
        ventricle_import_dir=import_dir if import_dir is not None else str(REPO / "dataset/example/STL"),
        mitral_width=panel["MV"][0], mitral_depth=panel["MV"][1], aortic_diameter=panel["AV"],
        translation_mitral=mv["translation"], angle_mitral=mv["angles"],
        translation_aortic=av["translation"], angle_aortic=av["angles"])
    return vgc.run_valve_area_check(types.SimpleNamespace(scene=scene), ventricles)


NEW_HEADER = ("path,bpm_mv [1/min],bpm_av [1/min],approach,A_MV [m^2],A_AV [m^2],R_target [-],"
              "width_MV [m],depth_MV [m],d_AV [m]")
OLD_HEADER = "r_MV_minor,r_MV_major,r_AV,path,bpm_mv,bpm_av,A_MV,d_MV_minor,d_MV_major,A_AV,d_AV"


def case_with_summary(tmp, name, header, row):
    """Case folder <tmp>/<name> with an STL folder and calc_valve_diameters_outputs/summary.csv."""
    case = Path(tmp) / name
    (case / "STL").mkdir(parents=True)
    (case / "calc_valve_diameters_outputs").mkdir()
    (case / "calc_valve_diameters_outputs" / "summary.csv").write_text(f"{header}\n{row}\n", encoding="ascii")
    return str(case / "STL")


def placed_area_ratio(verts, faces, stretch):
    """Top-view area and width/depth of a template mesh multiplied by the stretch factors."""
    p = [(x * stretch[0], y * stretch[1], z * stretch[2]) for x, y, z in verts]
    x0, x1, y0, y1 = vgc.xy_extent(p)
    return vgc.projected_faces_area(p, faces, (0.0, 0.0, 1.0)), (x1 - x0) / (y1 - y0)


def test_valve_sizes(blend) -> None:
    """T1: template outlines, sizes from A_fit, stretch factors, round trip, imperm lock."""
    print("-- T1 template outlines (A{n}_{MV,AV}_Boundary at unit scale)")
    objects = template_objects(blend)
    s4, s5 = vgc.template_shapes(mock_bpy(objects), 4), vgc.template_shapes(mock_bpy(objects), 5)
    mv4, av4 = s4["MV"], s4["AV"]
    check(all(close(getattr(mv4, k), getattr(A4_MV, k), 1e-4) for k in ("area", "width", "depth")),
          f"A4 MV a = {mv4.area:.5f}, W_t = {mv4.width:.5f}, D_t = {mv4.depth:.5f} (expected {A4_MV})")
    check(close(mv4.fill, 0.75791, 1e-4), f"A4 MV fill factor c = {mv4.fill:.5f} (expected 0.75791)")
    check(close(av4.area, A4_AV.area, 1e-4), f"A4 AV a = {av4.area:.5f} (expected {A4_AV.area})")
    check(close(s5["MV"].area, 2.51766, 1e-4) and close(s5["AV"].area, 3.13655, 1e-4),
          f"A5 MV a = {s5['MV'].area:.5f}, AV a = {s5['AV'].area:.5f} (expected 2.51766, 3.13655)")

    print("-- T1 sizes from A_fit (test case A4: MV 107.382 mm2, AV 36.202 mm2, R_target 1.3)")
    w, d = vgc.mv_size(107.382, mv4, 1.3)
    check(close(w, 13.572, 1e-4) and close(d, 10.440, 1e-4), f"MV width/depth {w:.3f}/{d:.3f} mm (expected 13.572/10.440)")
    sx, sy, sz = vgc.mv_stretch(w, d, mv4)
    check(close(sx, 6.761, 2e-4) and close(sy, 6.194, 2e-4) and close(sz, 6.4775, 2e-4),
          f"MV stretch factors {sx:.3f}/{sy:.3f}/{sz:.4f} (expected 6.761/6.194/6.4775)")
    dav = vgc.av_diameter(36.202, av4)
    check(close(dav, 6.811, 1e-4), f"AV diameter {dav:.3f} mm (expected 6.811)")
    check(all(close(s, 3.406, 2e-4) for s in vgc.av_stretch(dav)), "AV stretch factor 3.406 in x, y and z")
    w5, d5 = vgc.mv_size(107.382, s5["MV"], 1.3)
    dav5 = vgc.av_diameter(36.202, s5["AV"])
    check(close(w5, 13.508, 1e-4) and close(d5, 10.391, 1e-4) and close(dav5, 6.795, 1e-4),
          f"A5: MV {w5:.3f}/{d5:.3f} mm, AV {dav5:.3f} mm (expected 13.508/10.391, 6.795)")

    print("-- T1 round trip: the real templates placed with these stretch factors")
    area, ratio = placed_area_ratio(*template_mesh(blend, "A4_MV_Surface")[:2], (sx, sy, sz))
    check(close(area, 107.382, 1e-4) and close(ratio, 1.30, 1e-4), f"A4 MV built {area:.3f} mm2, width/depth {ratio:.4f}")
    area, _ = placed_area_ratio(*template_mesh(blend, "A4_AV_Surface")[:2], vgc.av_stretch(dav))
    check(close(area, 36.202, 1e-4), f"A4 AV built {area:.3f} mm2")

    print("-- T1 imperm lock")
    objects["A4_MV_imperm"] = _Obj("A4_MV_imperm", [], [])
    try:
        vgc.template_shapes(mock_bpy(objects), 4)
        check(False, "a remaining imperm template raises TemplateError")
    except vgc.TemplateError as e:
        check("A4_MV_imperm" in str(e) and "merge_imperm_into_res.py" in str(e),
              "a remaining imperm template raises TemplateError naming it and the merge script")


def main() -> int:
    blend = diag.BlendFile(REPO / "GVR-Pipeline.blend")
    test_valve_sizes(blend)

    print("-- template areas (unit scale, diagnosis tool)")
    templates = diag.load_templates(blend)
    a4mv, a4av = templates[(4, "MV")].areas, templates[(4, "AV")].areas
    check(close(a4mv.ring, 2.5643, 1e-4) and close(a4mv.open, 2.5643, 1e-4),
          f"A4 MV ring a = {a4mv.ring:.4f}, perm+res = {a4mv.open:.4f} (expected 2.5643 both)")
    check(close(a4av.ring, 3.1214, 1e-4) and close(a4av.open, 3.1214, 1e-4),
          f"A4 AV ring a = {a4av.ring:.4f}, perm+res = {a4av.open:.4f} (expected 3.1214 both)")
    a3mv = templates[(3, "MV")].areas
    check(close(a3mv.open, a4mv.open, 1e-4), "A3 MV zones (flat) equal the A4 MV zones (prisms)")
    check(close(templates[(4, "MV")].shape.fill, 0.75791, 1e-4), "diagnosis tool measures the same template shape")

    session = diag.load_session(SESSION)
    shapes = vgc.template_shapes(mock_bpy(template_objects(blend)), 4)
    old_panel = panel_from_session(session, shapes)
    objects, ventricles = build_scene(session, blend, old_panel, vessels="A4")

    print("-- T2 example export, A4 with zone objects, no summary.csv (target = panel values)")
    ok, lines, short = run_check(objects, ventricles, session, old_panel, import_dir="/nonexistent")
    text = "\n".join(lines)
    check("MV: target 28.16 mm2 [panel values (no summary.csv)]  ->  OK" in text, "MV target c * width * depth, OK")
    check("AV: target 26.07 mm2 [panel values (no summary.csv)]  ->  OK" in text, "AV target a * (d/2)^2, OK")
    check("ring      28.20 mm2 = 100.1 % of target" in text, "MV disc 28.20 mm2 = 100.1 %")
    check("of target   (12 frames, identical)" in text, "ring measured in all 12 frames, identical")
    check("perm+res  28.16 mm2 = 100.0 % of target   (zones: zone objects)" in text,
          "zones perm+res (imperm counted with res) measured from the mv_*/av_* objects")
    check("width/depth 1.40 outside 1.15 to 1.35" in text and not ok and "width/depth" in short,
          "the old example (stretch factors 3.6/3.05) is outside the width/depth band: warning")
    check("stretch factors x 3.600, y 3.050, z 3.325" in text, "MV stretch factors in the console")
    sz_mv = sum(session["MV"]["stretch"]) / 2
    h, share = disc_note(text, "atrium")
    check(close(h, 0.1 * sz_mv, 1e-2) and "'atrium' bottom disc" in text and "on top of the porous zones" in text,
          f"A4 atrium found on top of the MV zones at {h} mm (0.1 * s_z = {0.1 * sz_mv:.4f} mm)")
    check(close(share, 100.0, 5e-3), f"A4 atrium bottom disc = ventricle disc ({share} %, export vs. template)")
    h, share = disc_note(text, "aorta")
    check(close(h, 0.1 * session["AV"]["stretch"][0], 1e-2) and text.count("on top of the porous zones") == 2,
          f"A4 aorta found on top of the AV zones at {h} mm")
    check("no bottom disc found" not in text and "The porous zones end" not in text, "no interface note in A4")

    print("-- T2 example export, a single frame selected")
    ok, lines, short = run_check(objects, ventricles[:1], session, old_panel, import_dir="/nonexistent")
    check("of target   (1 frame)" in "\n".join(lines), "ring line says 1 frame")

    print("-- T2 example export, atrium/aorta lifted 0.05 mm off the zones (gap)")
    objects_gap, ventricles_gap = build_scene(session, blend, old_panel, vessels="A4", vessel_shift=0.05)
    ok, lines, short = run_check(objects_gap, ventricles_gap, session, old_panel, import_dir="/nonexistent")
    text = "\n".join(lines)
    h, _ = disc_note(text, "atrium")
    check(close(h, 0.1 * sz_mv + 0.05, 1e-2) and "on top of the porous zones" not in text,
          f"gap: atrium height {h} mm reported, not 'on top of the zones'")
    check(text.count("The porous zones end") == 2 and "not at this disc." in text,
          "gap between zones and atrium/aorta is reported")

    print("-- T2 example export, A3 without zone objects (estimate from perm + res)")
    objects3, ventricles3 = build_scene(session, blend, old_panel, with_zones=False)
    ok, lines, short = run_check(objects3, ventricles3, session, old_panel, approach=3, import_dir="/nonexistent")
    text = "\n".join(lines)
    check("estimated from A3 templates" in text, "A3 zones estimated from the templates")
    check("perm+res  28.20 mm2 = 100.1 % of target   (zones: estimated from A3 templates)" in text,
          "A3 MV perm+res estimate = the whole disc")
    check("'atrium' disc in the valve plane: 28.20 mm2 (100.0 % of the ventricle disc)" in text,
          "vessel disc in the valve plane (export, like A3) matches the ventricle disc")

    print("-- T2 example export, A5 (MV not checked)")
    ok, lines, short = run_check(objects, ventricles, session, old_panel, approach=5, import_dir="/nonexistent")
    text = "\n".join(lines)
    check("A5 builds the MV as a 3D valve" in text, "A5 MV skipped with a note")
    check("AV: target 26.20 mm2 [panel values (no summary.csv)]  ->  OK" in text, "A5 AV still checked (A5 template a = 3.13655)")

    # Test case A4 (R_target 1.3): A_fit MV 107.382 mm2, AV 36.202 mm2 -> 13.572 / 10.440 mm, AV 6.811 mm.
    panel = {"MV": (13.572, 10.440), "AV": 6.811}
    new_row = "{case},129,123,{approach},1.07382e-04,3.6202e-05,1.3,0.013572,0.010440,0.006811"
    objects_t, ventricles_t = build_scene(session, blend, panel, ventricles="templates", vessels="A4")
    with tempfile.TemporaryDirectory() as tmp:
        print("-- T2 valves built from the panel values, new summary.csv")
        stl = case_with_summary(tmp, "new", NEW_HEADER, new_row.format(case="new", approach=4))
        ok, lines, short = run_check(objects_t, ventricles_t, session, panel, import_dir=stl)
        text = "\n".join(lines)
        check(ok and "MV: target 107.38 mm2 [summary.csv (A_fit)]  ->  OK" in text, "MV built = A_fit, OK")
        check("AV: target 36.20 mm2 [summary.csv (A_fit)]  ->  OK" in text, "AV built = A_fit, OK")
        check("width 13.57 mm, depth 10.44 mm, width/depth 1.30" in text and "outside" not in text,
              "MV width/depth measured 1.30, inside the band")
        check("old format" not in text and "differ from summary.csv" not in text and "approach" not in text,
              "no note about format, panel values or approach")

        print("-- T2 width and depth swapped in the panel")
        swapped = {"MV": (10.440, 13.572), "AV": 6.811}
        objects_s, ventricles_s = build_scene(session, blend, swapped, ventricles="templates", vessels="A4")
        ok, lines, short = run_check(objects_s, ventricles_s, session, swapped, import_dir=stl)
        text = "\n".join(lines)
        check(not ok and "width/depth 0.77 outside 1.15 to 1.35" in text and "width/depth" in short,
              "swap: width/depth 0.77 is reported")
        check("MV width 10.440 mm vs. 13.572 mm" in text, "swap: panel values differ from summary.csv")

        print("-- T2 old summary.csv (no width/depth/diameter, A_AV negative)")
        stl = case_with_summary(tmp, "old", OLD_HEADER, "0.004,0.005,0.0034,old,129,123,1.07382e-04,0.008,0.010,"
                                                        "-3.6202e-05,0.0068")
        ok, lines, short = run_check(objects_t, ventricles_t, session, panel, import_dir=stl)
        text = "\n".join(lines)
        check(ok and "MV: target 107.38 mm2 [summary.csv (A_fit)]  ->  OK" in text
              and "AV: target 36.20 mm2 [summary.csv (A_fit)]  ->  OK" in text, "old format: A_fit read, |A_AV| used")
        check("summary.csv has the old format" in text and "Run 'Calculate valve sizes' again" in text
              and "differ from summary.csv" not in text, "old format: note, no comparison with the panel")

        print("-- T2 summary.csv computed for another approach")
        stl = case_with_summary(tmp, "a5", NEW_HEADER, new_row.format(case="a5", approach=5))
        ok, lines, short = run_check(objects_t, ventricles_t, session, panel, import_dir=stl)
        check("summary.csv was computed for approach A5, the scene uses A4" in "\n".join(lines), "approach note")

    print("-- T2 imperm lock")
    objects_t["A4_AV_imperm"] = _Obj("A4_AV_imperm", [], [])
    try:
        run_check(objects_t, ventricles_t, session, panel, import_dir="/nonexistent")
        check(False, "the check aborts while an imperm template is left")
    except vgc.TemplateError as e:
        check("merge_imperm_into_res.py" in str(e), "the check aborts while an imperm template is left")

    print()
    print(f"{len(FAILURES)} failure(s)" if FAILURES else "all checks passed")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
