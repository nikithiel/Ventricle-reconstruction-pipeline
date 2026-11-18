#!/usr/bin/env python3
"""Diagnose the valve areas of the Geometrical heart reconstruction pipeline, without Blender.

What it does
  1. Reads the valve templates directly from GVR-Pipeline.blend (A3_/A4_/A5_ *_Boundary,
     *_Surface, *_perm, *_res) and reports their outline at unit scale: area a, width W_t,
     depth D_t and fill factor c = a / (W_t * D_t). Built area = c * width * depth (MV) or
     a * (d/2)^2 (AV). A *_imperm left in the .blend is counted with *_res (not merged yet).
  2. --export DIR --session JSON: measures the valve discs in the exported atrium.stl,
     aorta.stl and ventricle_*.stl, in the valve planes from the session log (old logs with
     radius_* = stretch factors, or new ones with sizes in mm and stretch factors).
  3. --summary CSV: compares with A_fit from calc_valve_diameters_outputs/summary.csv (old or
     new format) and prints the sizes that would make the area match.
  4. --t-cfd SECONDS: adds the cycle-duration factor T_Doppler / T_CFD (T_Doppler = 60/bpm).

Only the Python standard library is needed. The area arithmetic and the comparison are the
same code the Blender add-on uses (addons/valve_geometry.py).

Examples (from the repository root)
  python tools/valve_area_diagnosis.py
  python tools/valve_area_diagnosis.py --map
  python tools/valve_area_diagnosis.py --export dataset/example/export ^
      --session dataset/example/Session_Logs/session_2026-10-06_13-45-54.json ^
      --summary dataset/example/calc_valve_diameters_outputs/summary.csv --t-cfd 0.557
"""
from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "addons"))
import valve_geometry as vgc  # noqa: E402  (bpy-free core)

TEMPLATE_RE = re.compile(r"^A([345])_(MV|AV)_(Boundary|Surface|perm|res|imperm)$")


# -----------------------------------------------------------------------------
# Minimal .blend reader (Blender 3.x, uncompressed, 64 bit little endian)
# -----------------------------------------------------------------------------

class BlendError(RuntimeError):
    pass


class BlendFile:
    """Reads mesh data of named objects from a .blend file using its SDNA.

    Supports files written by Blender versions that still store MVert/MPoly/MLoop arrays
    (up to 3.4). Field offsets come from the file's own DNA, not from hard-coded values.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        data = self.path.read_bytes()
        if data[:2] == b"\x1f\x8b" or data[:4] == b"\x28\xb5\x2f\xfd":
            raise BlendError(f"{self.path} is compressed. Save it uncompressed in Blender "
                             f"(File > Save As, untick 'Compress') and try again.")
        if not data.startswith(b"BLENDER"):
            raise BlendError(f"{self.path} is not a .blend file.")
        if data[7:8] != b"-" or data[8:9] != b"v":
            raise BlendError("Only 64 bit little endian .blend files are supported.")
        self.version = data[9:12].decode()
        self.data = data
        self.blocks: List[Tuple[bytes, int, int, int, int, int]] = []  # code, size, old, sdna, count, offset
        self.by_old: Dict[int, Tuple[bytes, int, int, int, int, int]] = {}
        pos = 12
        while pos + 24 <= len(data):
            code = data[pos:pos + 4]
            size, = struct.unpack_from("<i", data, pos + 4)
            old, = struct.unpack_from("<Q", data, pos + 8)
            sdna, count = struct.unpack_from("<ii", data, pos + 16)
            block = (code, size, old, sdna, count, pos + 24)
            self.blocks.append(block)
            self.by_old[old] = block
            pos += 24 + size
            if code == b"ENDB":
                break
        self._parse_dna()
        self.objects: Dict[str, int] = {}
        for code, size, old, sdna, count, off in self.blocks:
            if code == b"OB\x00\x00":
                name = self._cstr(off + self.off("Object", "id") + self.off("ID", "name"), 66)
                self.objects[name[2:]] = off

    # --- DNA -----------------------------------------------------------------
    def _parse_dna(self) -> None:
        dna = next((b for b in self.blocks if b[0] == b"DNA1"), None)
        if dna is None:
            raise BlendError("No DNA1 block found.")
        d = self.data[dna[5]:dna[5] + dna[1]]
        p = 4  # 'SDNA'

        def align(x: int) -> int:
            return (x + 3) & ~3

        def strings(p: int, tag: bytes) -> Tuple[List[str], int]:
            assert d[p:p + 4] == tag, tag
            n, = struct.unpack_from("<i", d, p + 4)
            p += 8
            out = []
            for _ in range(n):
                e = d.index(b"\0", p)
                out.append(d[p:e].decode())
                p = e + 1
            return out, align(p)

        names, p = strings(p, b"NAME")
        types, p = strings(p, b"TYPE")
        assert d[p:p + 4] == b"TLEN"
        tlen = list(struct.unpack_from(f"<{len(types)}h", d, p + 4))
        p = align(p + 4 + 2 * len(types))
        assert d[p:p + 4] == b"STRC"
        n, = struct.unpack_from("<i", d, p + 4)
        p += 8
        self.struct_fields: Dict[str, Dict[str, Tuple[int, str, int]]] = {}
        self.struct_size: Dict[str, int] = {}
        for _ in range(n):
            t, nf = struct.unpack_from("<hh", d, p)
            p += 4
            off = 0
            fields: Dict[str, Tuple[int, str, int]] = {}
            for _ in range(nf):
                ft, fn = struct.unpack_from("<hh", d, p)
                p += 4
                name = names[fn]
                dims = 1
                for x in re.findall(r"\[(\d+)\]", name):
                    dims *= int(x)
                is_ptr = name.startswith("*") or name.startswith("(*")
                size = (8 if is_ptr else tlen[ft]) * dims
                base = re.sub(r"\[.*", "", name).replace("(", "").replace(")", "").lstrip("*")
                fields[base] = (off, types[ft], size)
                off += size
            self.struct_fields[types[t]] = fields
            self.struct_size[types[t]] = tlen[t]

    def off(self, struct_name: str, field_name: str) -> int:
        try:
            return self.struct_fields[struct_name][field_name][0]
        except KeyError:
            raise BlendError(f"Field {struct_name}.{field_name} not in this file's DNA "
                             f"(Blender {self.version}). Files from Blender >= 3.5 are not supported.")

    def _cstr(self, off: int, n: int) -> str:
        raw = self.data[off:off + n]
        return raw[:raw.index(b"\0")].decode(errors="replace") if b"\0" in raw else raw.decode(errors="replace")

    def _ptr(self, off: int) -> int:
        return struct.unpack_from("<Q", self.data, off)[0]

    def _i32(self, off: int) -> int:
        return struct.unpack_from("<i", self.data, off)[0]

    def _block(self, ptr: int, what: str):
        b = self.by_old.get(ptr)
        if b is None:
            raise BlendError(f"Data block for {what} not found.")
        return b

    # --- meshes --------------------------------------------------------------
    def mesh(self, object_name: str):
        """(verts, faces, edges) of the object's mesh in mesh coordinates (no object transform)."""
        ob = self.objects.get(object_name)
        if ob is None:
            raise KeyError(object_name)
        me_ptr = self._ptr(ob + self.off("Object", "data"))
        me_block = self._block(me_ptr, f"{object_name}.data")
        if me_block[0] != b"ME\x00\x00":
            raise BlendError(f"{object_name} is not a mesh object.")
        mo = me_block[5]
        tv = self._i32(mo + self.off("Mesh", "totvert"))
        te = self._i32(mo + self.off("Mesh", "totedge"))
        tp = self._i32(mo + self.off("Mesh", "totpoly"))
        tl = self._i32(mo + self.off("Mesh", "totloop"))
        verts: List[Tuple[float, float, float]] = []
        if tv:
            vb = self._block(self._ptr(mo + self.off("Mesh", "mvert")), f"{object_name} vertices")
            vs, co = self.struct_size["MVert"], self.off("MVert", "co")
            verts = [struct.unpack_from("<3f", self.data, vb[5] + vs * i + co) for i in range(tv)]
        faces: List[Tuple[int, ...]] = []
        if tp:
            pb = self._block(self._ptr(mo + self.off("Mesh", "mpoly")), f"{object_name} polygons")
            lb = self._block(self._ptr(mo + self.off("Mesh", "mloop")), f"{object_name} loops")
            ps, ls_, lv = self.struct_size["MPoly"], self.struct_size["MLoop"], self.off("MLoop", "v")
            loops = [self._i32(lb[5] + ls_ * i + lv) for i in range(tl)]
            o_start, o_tot = self.off("MPoly", "loopstart"), self.off("MPoly", "totloop")
            for i in range(tp):
                s = self._i32(pb[5] + ps * i + o_start)
                n = self._i32(pb[5] + ps * i + o_tot)
                faces.append(tuple(loops[s:s + n]))
        edges: List[Tuple[int, int]] = []
        if te:
            eb = self._block(self._ptr(mo + self.off("Mesh", "medge")), f"{object_name} edges")
            es, v1, v2 = self.struct_size["MEdge"], self.off("MEdge", "v1"), self.off("MEdge", "v2")
            edges = [(struct.unpack_from("<I", self.data, eb[5] + es * i + v1)[0],
                      struct.unpack_from("<I", self.data, eb[5] + es * i + v2)[0]) for i in range(te)]
        return verts, faces, edges


# -----------------------------------------------------------------------------
# Templates
# -----------------------------------------------------------------------------

class TemplateSet:
    """Unit-scale outline, zone areas and ring points of one approach/valve template set."""

    def __init__(self, approach: int, valve: str, shape: vgc.TemplateShape, areas: vgc.TemplateAreas,
                 ring_source: str, extent: Tuple[float, float, float, float],
                 ring_xy: List[Tuple[float, float]], zones: Dict[str, list], imperm_left: bool):
        self.approach, self.valve, self.shape, self.areas = approach, valve, shape, areas
        self.ring_source, self.extent, self.zones = ring_source, extent, zones
        self.imperm_left = imperm_left  # *_imperm still in the .blend; counted with res here
        self._ring_xy = ring_xy  # ring vertices (x, y) at unit scale; multiplied by the stretch factors later

    def ring_points(self) -> List[Tuple[float, float]]:
        return self._ring_xy


def load_templates(blend: BlendFile) -> Dict[Tuple[int, str], TemplateSet]:
    """Template sets per (approach, valve). The outline follows vgc.template_shapes(): the
    *_Boundary edge loop, else the *_Surface faces. A *_imperm left in the .blend is added to
    *_res, as tools/merge_imperm_into_res.py does."""
    found: Dict[Tuple[int, str], Dict[str, str]] = {}
    for name in blend.objects:
        m = TEMPLATE_RE.match(name)
        if m:
            found.setdefault((int(m.group(1)), m.group(2)), {})[m.group(3)] = name
    out: Dict[Tuple[int, str], TemplateSet] = {}
    for (approach, valve), parts in sorted(found.items()):
        boundary = surface = None
        if "Boundary" in parts:
            verts, _, edges = blend.mesh(parts["Boundary"])
            boundary = (verts, edges)
        if "Surface" in parts:
            surface = blend.mesh(parts["Surface"])[:2]
        outline = vgc.template_outline(boundary, surface)
        if outline is None:
            continue
        shape, pts, used = outline
        ring_source = f"{parts[used]} ({'edge loop' if used == 'Boundary' else 'faces'})"
        ring_xy = [(q[0], q[1]) for q in pts]
        zone_areas: Dict[str, float] = {}
        zone_tris: Dict[str, list] = {}
        for z in vgc.ZONE_TYPES + ("imperm",):
            if z in parts:
                verts, faces, _ = blend.mesh(parts[z])
                key = "res" if z == "imperm" else z
                zone_areas[key] = zone_areas.get(key, 0.0) + vgc.bottom_faces_area(verts, faces, vgc.UNIT_FRAME)
                zone_tris.setdefault(key, []).extend(_bottom_triangles(verts, faces))
        areas = vgc.TemplateAreas(shape.area, zone_areas.get("perm"), zone_areas.get("res"))
        out[(approach, valve)] = TemplateSet(approach, valve, shape, areas, ring_source,
                                             vgc.xy_extent(ring_xy), ring_xy, zone_tris,
                                             "imperm" in parts)
    return out


def _bottom_triangles(verts, faces):
    """Bottom faces of a zone prism as 2D triangles (for the ASCII map)."""
    zs = [v[2] for v in verts]
    zmin, zmax = min(zs), max(zs)
    lateral = max(math.hypot(v[0], v[1]) for v in verts)
    flat = (zmax - zmin) <= 0.01 * lateral  # same rule as vgc.bottom_faces_area
    limit = zmax if flat else zmin + 0.25 * (zmax - zmin)
    tris = []
    for f in faces:
        if all(verts[i][2] <= limit for i in f):
            for k in range(1, len(f) - 1):  # fan
                tris.append([(verts[f[0]][0], verts[f[0]][1]), (verts[f[k]][0], verts[f[k]][1]),
                             (verts[f[k + 1]][0], verts[f[k + 1]][1])])
    return tris


def _inside(p, t) -> bool:
    (x, y), ((x1, y1), (x2, y2), (x3, y3)) = p, t
    d1 = (x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)
    d2 = (x - x3) * (y2 - y3) - (x2 - x3) * (y - y3)
    d3 = (x - x1) * (y3 - y1) - (x3 - x1) * (y - y1)
    return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))


def ascii_map(ts: TemplateSet, width: int = 56) -> List[str]:
    """Top view of the zones: 'P' perm, '.' res (y up)."""
    if not ts.zones or ts.extent is None:
        return ["    (no zones to draw)"]
    x0, x1, y0, y1 = ts.extent
    pad = 0.03 * max(x1 - x0, y1 - y0)
    x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
    height = max(8, int(round(width * (y1 - y0) / (x1 - x0) * 0.5)))  # terminal cells are ~2:1
    lines = []
    for j in range(height):
        y = y1 - (j + 0.5) * (y1 - y0) / height
        row = ""
        for i in range(width):
            x = x0 + (i + 0.5) * (x1 - x0) / width
            ch = " "
            for z, c in (("perm", "P"), ("res", ".")):
                if any(_inside((x, y), t) for t in ts.zones.get(z, [])):
                    ch = c
                    break
            row += ch
        lines.append("    |" + row + "|")
    lines.append("    legend: 'P' perm   '.' res   (template x to the right, y up)")
    return lines


# -----------------------------------------------------------------------------
# STL
# -----------------------------------------------------------------------------

def read_stl(path: Path) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
    """ASCII or binary STL -> (verts, triangles) with shared vertices merged."""
    raw = Path(path).read_bytes()
    tris: List[List[Tuple[float, float, float]]] = []
    is_ascii = raw[:5].lower() == b"solid" and b"facet" in raw[:2048]
    if is_ascii:
        cur: List[Tuple[float, float, float]] = []
        for line in raw.decode(errors="replace").splitlines():
            s = line.split()
            if s and s[0] == "vertex":
                cur.append((float(s[1]), float(s[2]), float(s[3])))
                if len(cur) == 3:
                    tris.append(cur)
                    cur = []
    else:
        n, = struct.unpack_from("<I", raw, 80)
        for i in range(n):
            vals = struct.unpack_from("<12f", raw, 84 + 50 * i)
            tris.append([tuple(vals[3:6]), tuple(vals[6:9]), tuple(vals[9:12])])
    index: Dict[Tuple[float, float, float], int] = {}
    verts: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    for t in tris:
        ids = []
        for p in t:
            key = (round(p[0], 6), round(p[1], 6), round(p[2], 6))
            if key not in index:
                index[key] = len(verts)
                verts.append(p)
            ids.append(index[key])
        faces.append(tuple(ids))
    return verts, faces


def _natural_key(p: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]


# -----------------------------------------------------------------------------
# Report sections
# -----------------------------------------------------------------------------

def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def report_templates(templates: Dict[Tuple[int, str], TemplateSet], show_map: bool) -> None:
    section("1) Valve templates in the .blend at unit scale")
    print("built area = c * width * depth (MV) or a * (d/2)^2 (AV), with c = a / (W_t * D_t)")
    print("Used by the pipeline: A4 zones for the MV in A4 and for the AV in A4 and A5; A3 builds no zone objects.")
    for (approach, valve), ts in templates.items():
        sh, a = ts.shape, ts.areas
        x0, x1, y0, y1 = ts.extent
        print()
        print(f"A{approach} {valve}   a = {sh.area:.5f}   W_t x D_t = {sh.width:.5f} x {sh.depth:.5f}   "
              f"c = {sh.fill:.5f}   [{ts.ring_source}]")
        print(f"    extent x [{x0:.5f}, {x1:.5f}]  y [{y0:.5f}, {y1:.5f}]")
        if a.open is not None:
            print(f"    zones: perm {a.perm:.4f}  res {a.res:.4f}  (sum {a.open:.4f} = {100 * a.open / sh.area:.1f} % of a)")
        if ts.imperm_left:
            print(f"    A{approach}_{valve}_imperm is still in the .blend (counted with res here); "
                  f"run tools/merge_imperm_into_res.py in Blender 3.1.")
        if show_map and ts.zones:
            for line in ascii_map(ts):
                print(line)


def load_session(path: Path) -> dict:
    """Valve placement from a session log: translation, angles and the stretch factors (x, y)."""
    s = json.loads(Path(path).read_text(encoding="utf-8"))
    v = s["valves"]

    def stretch(entry: dict, old_keys: Tuple[str, str]) -> Tuple[float, float]:
        if "stretch_factors" in entry:  # schema 2: sizes in mm plus the stretch factors the build used
            if entry["stretch_factors"] is None:
                raise ValueError(f"{path}: no stretch factors logged (the .blend still held imperm templates).")
            return float(entry["stretch_factors"][0]), float(entry["stretch_factors"][1])
        return float(entry[old_keys[0]]), float(entry[old_keys[1]])  # schema 1: the radius_* values are stretch factors

    return {
        "MV": {"translation": v["mitral"]["translation_mm"], "angles": v["mitral"]["angle_deg"],
               "stretch": stretch(v["mitral"], ("radius_long", "radius_small"))},
        "AV": {"translation": v["aortic"]["translation_mm"], "angles": v["aortic"]["angle_deg"],
               "stretch": stretch(v["aortic"], ("radius", "radius"))},
        "approach": (s.get("pipeline") or {}).get("approach"),
        "time_rr": ((s.get("setup_variables") or {}).get("interpolation") or {}).get("time_rr"),
    }


def report_export(export: Path, session: dict, templates) -> Dict[str, vgc.PlaneDisc]:
    section(f"2) Valve discs in the exported geometry ({export})")
    approach = session.get("approach") or 4
    measured: Dict[str, vgc.PlaneDisc] = {}
    files = {"MV": [export / "atrium.stl"], "AV": [export / "aorta.stl"]}
    ventricles = sorted(export.glob("ventricle_*.stl"), key=_natural_key)
    for valve in vgc.VALVES:
        s = session[valve]
        frame = vgc.valve_frame(s["translation"], s["angles"])
        sx, sy = s["stretch"]
        ts = templates.get((approach, valve)) or templates.get((4, valve))
        if ts:
            r_max = 1.05 * max(math.hypot(x * sx, y * sy) for x, y in ts.ring_points())
        else:
            r_max = 1.3 * max(sx, sy)
        tol = 1e-3 * max(sx, sy) + 1e-6
        predicted = ts.shape.area * sx * sy if ts else None
        print(f"\n{valve} (stretch factors x {sx:g}, y {sy:g})"
              + (f"; template predicts {predicted:.2f} mm2" if predicted else ""))
        ven_discs = []
        for path in files[valve] + ventricles:
            if not path.is_file():
                print(f"    {path.name:18s} missing")
                continue
            verts, faces = read_stl(path)
            vessel = not path.name.startswith("ventricle_")
            if vessel:  # atrium/aorta: bottom disc, in A4 on top of the porous zones (template z = 0.1)
                disc = vgc.lowest_disc(verts, faces, frame, tol, r_max)
            else:
                disc = vgc.plane_faces_area(verts, faces, frame, tol, r_max)
            if disc.n_faces == 0:
                print(f"    {path.name:18s} no faces in the valve plane" + (" or at the vessel bottom" if vessel else ""))
                continue
            if vessel and abs(disc.height) > tol:
                print(f"    {path.name:18s} bottom disc {disc.height:.3f} mm above the valve plane "
                      f"(A4: top of the porous zones at 0.1 * s_z = {0.1 * (sx + sy) / 2:.3f} mm)")
            if not vessel:
                ven_discs.append(disc)
                if len(ven_discs) > 1:
                    continue  # all frames share the disc; summarise below
            x0, x1 = disc.x_extent
            y0, y1 = disc.y_extent
            pct = f" = {100 * disc.area / predicted:5.1f} % of the prediction" if predicted else ""
            print(f"    {path.name:18s} {disc.area:8.2f} mm2{pct}   "
                  f"({disc.n_faces} faces; local x [{x0:.2f}, {x1:.2f}], y [{y0:.2f}, {y1:.2f}] mm)")
        if ven_discs:
            areas = [d.area for d in ven_discs]
            spread = (max(areas) - min(areas)) / max(areas)
            print(f"    ventricle discs in {len(areas)} frames: {min(areas):.2f} ... {max(areas):.2f} mm2 "
                  f"(spread {100 * spread:.3f} %)")
            measured[valve] = ven_discs[0]
    return measured


def report_summary(summary_path: Path, session: Optional[dict], templates, measured: Dict[str, vgc.PlaneDisc],
                   approach: int, t_cfd: Optional[float], t_cfd_source: str) -> None:
    section(f"3) Comparison with A_fit from {summary_path}")
    summary = vgc.read_fit_summary(summary_path)
    sets = {v: templates.get((approach, v)) or templates.get((4, v)) for v in vgc.VALVES}
    shapes = {v: ts.shape for v, ts in sets.items()}
    if session:
        (sx, sy), (s_av, _) = session["MV"]["stretch"], session["AV"]["stretch"]
        panel = vgc.ValveSizes(sx * shapes["MV"].width, sy * shapes["MV"].depth, 2 * s_av)
        sizes_source = "session log"
    elif not summary.old_format:
        panel = summary.sizes_mm
        sizes_source = "summary.csv (no session log given)"
    else:
        print("The summary.csv has the old format (no sizes) and no session log is given: nothing to compare.")
        return
    print(f"A_fit: MV {summary.A_MV_mm2:.2f} mm2, AV {summary.A_AV_mm2:.2f} mm2; bpm_mv {summary.bpm_mv}, "
          f"bpm_av {summary.bpm_av}; format {'old' if summary.old_format else 'new'}")
    print(f"Sizes used for the geometry ({sizes_source}): MV width {panel.width_MV:.3f} mm, "
          f"depth {panel.depth_MV:.3f} mm, AV diameter {panel.d_AV:.3f} mm")
    exp, notes = vgc.expected_areas(summary, panel, shapes, approach)
    meas = {}
    for v in vgc.VALVES:
        m = vgc.Measurement(v)
        disc = measured.get(v)
        if disc is not None:
            m.ring_mm2 = disc.area
            if v == "MV":
                m.width_mm, m.depth_mm = disc.x_extent[1] - disc.x_extent[0], disc.y_extent[1] - disc.y_extent[0]
            src = "ring measured in the export"
        else:
            m.ring_mm2 = exp[v].panel_mm2
            if v == "MV":
                m.width_mm, m.depth_mm = panel.width_MV, panel.depth_MV
            src = "ring predicted from the template"
        a = sets[v].areas if sets[v] else None
        if a is not None and a.open:
            m.perm_mm2 = m.ring_mm2 * a.perm / a.open
            m.res_mm2 = m.ring_mm2 * a.res / a.open
            m.zones_source = f"A{sets[v].approach} template shares"
        meas[v] = m
        notes.append(f"{v}: {src}.")
    rows, ok = vgc.evaluate(meas, exp, 2.0)
    print()
    for line in vgc.format_report(rows, [], 2.0):
        print(line)
    for r in rows:
        if r.size_factor is None:
            continue
        fac = r.size_factor
        if r.valve == "MV":
            print(f"    -> MV sizes that give the target: width {panel.width_MV * fac:.3f} mm, depth {panel.depth_MV * fac:.3f} mm")
        else:
            print(f"    -> AV diameter that gives the target: {panel.d_AV * fac:.3f} mm")
        if t_cfd:
            bpm = summary.bpm_mv if r.valve == "MV" else summary.bpm_av
            if bpm:
                t_d = 60.0 / bpm
                print(f"    -> {r.valve} including the cycle duration (T_Doppler {t_d:.3f} s / T_CFD {t_cfd:.3f} s "
                      f"= {t_d / t_cfd:.3f}{t_cfd_source}): CFD mean velocity / Doppler "
                      f"x{r.velocity_factor * t_d / t_cfd:.3f}")
    print()
    for n in notes:
        print(f"Note: {n}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blend", type=Path, default=REPO / "GVR-Pipeline.blend", help="pipeline .blend with the templates")
    ap.add_argument("--map", action="store_true", help="draw a top view of the zones of each template set")
    ap.add_argument("--export", type=Path, help="export folder with atrium.stl, aorta.stl, ventricle_*.stl")
    ap.add_argument("--session", type=Path, help="session log JSON (valve position, angles, sizes)")
    ap.add_argument("--summary", type=Path, help="calc_valve_diameters_outputs/summary.csv")
    ap.add_argument("--approach", type=int, choices=(3, 4, 5), help="approach for the zone shares (default: session log, else 4)")
    ap.add_argument("--t-cfd", type=float, help="CFD cycle duration in s (default: time_rr from the session log)")
    args = ap.parse_args(argv)

    try:
        blend = BlendFile(args.blend)
        templates = load_templates(blend)
    except (BlendError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    print(f"Templates read from {args.blend} (Blender {blend.version[0]}.{int(blend.version[1:])})")
    report_templates(templates, args.map)

    session = load_session(args.session) if args.session else None
    measured: Dict[str, vgc.PlaneDisc] = {}
    if args.export:
        if not session:
            print("\n--export needs --session (valve planes and sizes).", file=sys.stderr)
            return 2
        measured = report_export(args.export, session, templates)
    if args.summary:
        approach = args.approach or (session or {}).get("approach") or 4
        t_cfd, src = args.t_cfd, ""
        if t_cfd is None and session and session.get("time_rr"):
            t_cfd, src = float(session["time_rr"]), ", T_CFD = time_rr from the session log"
        report_summary(args.summary, session, templates, measured, int(approach), t_cfd, src)
    return 0


if __name__ == "__main__":
    sys.exit(main())
