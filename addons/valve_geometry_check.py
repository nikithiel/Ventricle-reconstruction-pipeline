"""Check that the built valve geometry has the area calculate_valve_diameters.py computed.

calculate_valve_diameters.py fits an effective valve area A_fit from the volume curve and
the Doppler traces and turns it into radii, assuming a circle (AV, A = pi*r^2) or an
ellipse (MV, A = pi*r_long*r_small). scale_rotate_translate_object() in
geometrical_heart_reconstruction.py does not draw a circle or an ellipse with these
radii. It multiplies the coordinates of a template mesh (A*_MV_Boundary, A*_MV_Surface,
the porous zones, the atrium, ...) with them. The built area is therefore
a_template * r_long * r_small, where a_template is the template's area at unit scale.
For the AV template (a unit circle) a_template is close to pi. For the D-shaped MV
template it is about 2.56, so the built ring is about 18 % smaller than computed.
On top of that the *_imperm zone, if it is treated as a wall in the CFD, removes
about 20 % of the remaining area.

This module measures the built geometry and compares it with A_fit.

Layout:
- The first part (geometry helpers, summary.csv reader, evaluation, report) is free of
  bpy and numpy. tools/valve_area_diagnosis.py imports it outside Blender, so both
  front ends compute exactly the same thing.
- run_valve_area_check() at the bottom is the Blender front end. It imports bpy lazily.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Vec = Tuple[float, float, float]

VALVES = ("MV", "AV")
VALVE_MODE = {"MV": "Mitral", "AV": "Aortic"}
ZONE_TYPES = ("perm", "res", "imperm")


# -----------------------------------------------------------------------------
# Vector helpers
# -----------------------------------------------------------------------------

def _sub(a: Sequence[float], b: Sequence[float]) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Sequence[float], b: Sequence[float]) -> Vec:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(_dot(a, a))


def _unit(a: Sequence[float]) -> Vec:
    n = _norm(a)
    if n <= 0.0:
        raise ValueError("zero-length vector")
    return (a[0] / n, a[1] / n, a[2] / n)


# -----------------------------------------------------------------------------
# Valve frame: the transformation scale_rotate_translate_object() applies
# -----------------------------------------------------------------------------

def euler_xyz_matrix(angles_rad: Sequence[float]) -> List[List[float]]:
    """Rotation matrix of a Blender Euler rotation in mode 'XYZ' (R = Rz @ Ry @ Rx)."""
    ax, ay, az = angles_rad
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    return [
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ]


@dataclass
class ValveFrame:
    """Local valve frame: origin = valve translation, columns of `rot` = local axes in world.

    Local z is the valve normal (the template's z axis), local x is the template's x axis,
    which scale_rotate_translate_object() scales with the long mitral radius.
    """
    origin: Vec
    rot: List[List[float]]

    @property
    def normal(self) -> Vec:
        return (self.rot[0][2], self.rot[1][2], self.rot[2][2])

    def to_local(self, p: Sequence[float]) -> Vec:
        d = _sub(p, self.origin)
        r = self.rot
        return (
            r[0][0] * d[0] + r[1][0] * d[1] + r[2][0] * d[2],
            r[0][1] * d[0] + r[1][1] * d[1] + r[2][1] * d[2],
            r[0][2] * d[0] + r[1][2] * d[1] + r[2][2] * d[2],
        )


def valve_frame(translation: Sequence[float], angles_deg: Sequence[float]) -> ValveFrame:
    """Frame of a valve from the scene settings (translation in mm, angles in degrees)."""
    return ValveFrame(tuple(float(v) for v in translation),
                      euler_xyz_matrix([math.radians(a) for a in angles_deg]))


def frame_from_normal(origin: Sequence[float], normal: Sequence[float]) -> ValveFrame:
    """Frame with the given normal as local z. The in-plane axes are arbitrary but orthonormal."""
    n = _unit(normal)
    helper = (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 1.0, 0.0)
    x = _unit(_cross(helper, n))
    y = _cross(n, x)
    rot = [[x[0], y[0], n[0]], [x[1], y[1], n[1]], [x[2], y[2], n[2]]]
    return ValveFrame(tuple(float(v) for v in origin), rot)


UNIT_FRAME = ValveFrame((0.0, 0.0, 0.0), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


# -----------------------------------------------------------------------------
# Areas
# -----------------------------------------------------------------------------

def triangle_area(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    return 0.5 * _norm(_cross(_sub(b, a), _sub(c, a)))


def polygon_area_xy(points: Sequence[Sequence[float]]) -> float:
    """Shoelace area of a closed polygon given in order, using x and y only."""
    s = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i][0], points[i][1]
        x2, y2 = points[(i + 1) % n][0], points[(i + 1) % n][1]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _face_vector_area(verts: Sequence[Sequence[float]], face: Sequence[int]) -> Vec:
    """Vector area (Newell) of a planar polygon: direction = normal, length = area."""
    nx = ny = nz = 0.0
    m = len(face)
    for i in range(m):
        x1, y1, z1 = verts[face[i]]
        x2, y2, z2 = verts[face[(i + 1) % m]]
        nx += (y1 - y2) * (z1 + z2)
        ny += (z1 - z2) * (x1 + x2)
        nz += (x1 - x2) * (y1 + y2)
    return (0.5 * nx, 0.5 * ny, 0.5 * nz)


def faces_area(verts: Sequence[Sequence[float]], faces: Iterable[Sequence[int]]) -> float:
    """Total surface area of the given faces (triangles or planar polygons)."""
    return sum(_norm(_face_vector_area(verts, f)) for f in faces)


def projected_faces_area(verts: Sequence[Sequence[float]], faces: Iterable[Sequence[int]],
                         normal: Sequence[float]) -> float:
    """Sum of the face areas projected onto the plane with the given normal."""
    n = _unit(normal)
    return sum(abs(_dot(_face_vector_area(verts, f), n)) for f in faces)


def plane_from_faces(verts: Sequence[Sequence[float]], faces: Sequence[Sequence[int]]) -> Tuple[Vec, Vec]:
    """(area-weighted centroid, unit normal) of a set of roughly coplanar faces."""
    total = (0.0, 0.0, 0.0)
    area_sum = 0.0
    cx = cy = cz = 0.0
    for f in faces:
        va = _face_vector_area(verts, f)
        # Orient all face normals into one hemisphere before summing, so that a disc with
        # inconsistent winding does not cancel itself out.
        if _dot(va, total) < 0.0:
            va = (-va[0], -va[1], -va[2])
        total = (total[0] + va[0], total[1] + va[1], total[2] + va[2])
        a = _norm(va)
        m = len(f)
        cx += a * sum(verts[i][0] for i in f) / m
        cy += a * sum(verts[i][1] for i in f) / m
        cz += a * sum(verts[i][2] for i in f) / m
        area_sum += a
    if area_sum <= 0.0:
        raise ValueError("faces have no area")
    return (cx / area_sum, cy / area_sum, cz / area_sum), _unit(total)


def bottom_faces_area(verts: Sequence[Sequence[float]], faces: Sequence[Sequence[int]],
                      frame: ValveFrame, rel_tol: float = 0.25) -> float:
    """Top-view area of a porous zone.

    The A4 zones are prisms: the 2D zone extruded along the valve normal. Their bottom faces
    (closest to the valve plane) projected onto the plane give the area the flow passes.
    Flat zones (A3 templates) have no height; then all faces are projected.
    """
    if not faces:
        return 0.0
    n = frame.normal
    d = [_dot(_sub(v, frame.origin), n) for v in verts]
    used = sorted({i for f in faces for i in f})
    dmin = min(d[i] for i in used)
    dmax = max(d[i] for i in used)
    span = dmax - dmin
    # Lateral size of the zone in the plane. A zone whose height is below 1 % of it is flat
    # (the A3 templates carry a few vertices 1e-4 above the plane); project all its faces.
    lateral = max(_norm(_sub(_sub(verts[i], frame.origin), tuple(d[i] * c for c in n))) for i in used)
    if span <= 0.01 * lateral:
        return projected_faces_area(verts, faces, n)
    limit = dmin + rel_tol * span
    bottom = [f for f in faces if all(d[i] <= limit for i in f)]
    return projected_faces_area(verts, bottom, n)


@dataclass
class PlaneDisc:
    area: float            # true area of the faces found in the plane (mm^2 for real geometry)
    n_faces: int
    x_extent: Tuple[float, float]  # extents in the local valve frame
    y_extent: Tuple[float, float]
    height: float = 0.0    # distance of the disc from the valve plane (lowest_disc only)


def plane_faces_area(verts: Sequence[Sequence[float]], faces: Sequence[Sequence[int]],
                     frame: ValveFrame, tol: float, r_max: Optional[float] = None) -> PlaneDisc:
    """Faces lying in the valve plane (all vertices within tol), optionally within r_max of the origin.

    Used for meshes without vertex groups (exported STLs) and for the atrium/aorta discs.
    """
    local = [frame.to_local(v) for v in verts]
    picked = []
    for f in faces:
        if all(abs(local[i][2]) <= tol for i in f):
            if r_max is None or all(math.hypot(local[i][0], local[i][1]) <= r_max for i in f):
                picked.append(f)
    if not picked:
        return PlaneDisc(0.0, 0, (0.0, 0.0), (0.0, 0.0))
    xs = [local[i][0] for f in picked for i in f]
    ys = [local[i][1] for f in picked for i in f]
    return PlaneDisc(faces_area(verts, picked), len(picked), (min(xs), max(xs)), (min(ys), max(ys)))


def lowest_disc(verts: Sequence[Sequence[float]], faces: Sequence[Sequence[int]],
                frame: ValveFrame, tol: float, r_max: Optional[float] = None) -> PlaneDisc:
    """Bottom disc of a vessel (atrium, aorta): its faces closest to the valve plane.

    The disc does not always lie in the valve plane. The A3 atrium and aorta start in it, but
    A4_Atrium and A4_Aorta start at template z = 0.1, i.e. on top of the porous zones, which are
    prisms of that height. Heights are measured along the valve normal oriented towards the
    vessel, so `height` is positive for a disc above the plane. Only vertices within r_max of
    the valve axis count, if given.
    """
    empty = PlaneDisc(0.0, 0, (0.0, 0.0), (0.0, 0.0))
    local = [frame.to_local(v) for v in verts]
    used = sorted({i for f in faces for i in f})
    if not used:
        return empty
    sign = 1.0 if sum(local[i][2] for i in used) >= 0.0 else -1.0  # the vessel side of the plane

    def inside(i: int) -> bool:
        return r_max is None or math.hypot(local[i][0], local[i][1]) <= r_max

    candidates = [i for i in used if inside(i)]
    if not candidates:
        return empty
    h_min = min(sign * local[i][2] for i in candidates)
    picked = [f for f in faces if all(inside(i) and abs(sign * local[i][2] - h_min) <= tol for i in f)]
    if not picked:
        return empty
    xs = [local[i][0] for f in picked for i in f]
    ys = [local[i][1] for f in picked for i in f]
    return PlaneDisc(faces_area(verts, picked), len(picked), (min(xs), max(xs)), (min(ys), max(ys)), h_min)


def zone_height(verts: Sequence[Sequence[float]], faces: Sequence[Sequence[int]], frame: ValveFrame) -> float:
    """Largest distance of a zone's vertices from the valve plane (height of an A4 prism zone)."""
    used = {i for f in faces for i in f}
    return max((abs(frame.to_local(verts[i])[2]) for i in used), default=0.0)


def order_edge_loop(edges: Sequence[Tuple[int, int]]) -> List[int]:
    """Order the vertices of a single closed edge loop."""
    adj: Dict[int, List[int]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    if not adj or any(len(v) != 2 for v in adj.values()):
        raise ValueError("edges do not form a single closed loop")
    start = edges[0][0]
    loop = [start]
    prev, cur = None, start
    while True:
        nxt = adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
        if nxt == start:
            break
        loop.append(nxt)
        prev, cur = cur, nxt
        if len(loop) > len(adj):
            raise ValueError("edges do not form a single closed loop")
    if len(loop) != len(adj):
        raise ValueError("edges form more than one loop")
    return loop


# -----------------------------------------------------------------------------
# Template coefficients
# -----------------------------------------------------------------------------

@dataclass
class TemplateAreas:
    """Areas of one valve template set at unit scale (radii = 1)."""
    ring: float                      # area of the full valve disc (= perm + res + imperm)
    perm: Optional[float] = None
    res: Optional[float] = None
    imperm: Optional[float] = None

    @property
    def open(self) -> Optional[float]:
        if self.perm is None or self.res is None:
            return None
        return self.perm + self.res

    @property
    def open_fraction(self) -> Optional[float]:
        zones = (self.perm, self.res, self.imperm)
        if any(z is None for z in zones):
            return None
        total = sum(zones)
        return (self.perm + self.res) / total if total > 0 else None


# -----------------------------------------------------------------------------
# Expected values: summary.csv of calculate_valve_diameters.py and GUI radii
# -----------------------------------------------------------------------------

@dataclass
class FitSummary:
    path: Path
    case_path: str
    A_MV_mm2: float
    A_AV_mm2: float
    r_MV_minor_mm: float
    r_MV_major_mm: float
    r_AV_mm: float
    bpm_mv: Optional[float]
    bpm_av: Optional[float]


def _float(row: Dict[str, str], key: str) -> Optional[float]:
    raw = (row.get(key) or "").strip()
    if not raw:
        return None
    try:
        val = float(raw)
    except ValueError:
        return None
    return val if math.isfinite(val) else None


def read_fit_summary(path: Path) -> FitSummary:
    """Read calc_valve_diameters_outputs/summary.csv (SI units) and convert to mm / mm^2.

    A_AV is negative in that file (the AV Doppler trace is negative), so |A| is used.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} contains no data row")
    row = rows[-1]
    needed = ("A_MV", "A_AV", "r_MV_minor", "r_MV_major", "r_AV")
    missing = [k for k in needed if _float(row, k) is None]
    if missing:
        raise ValueError(f"{path} lacks valid values for {missing}")
    return FitSummary(
        path=path,
        case_path=(row.get("path") or "").strip(),
        A_MV_mm2=abs(_float(row, "A_MV")) * 1e6,
        A_AV_mm2=abs(_float(row, "A_AV")) * 1e6,
        r_MV_minor_mm=_float(row, "r_MV_minor") * 1e3,
        r_MV_major_mm=_float(row, "r_MV_major") * 1e3,
        r_AV_mm=_float(row, "r_AV") * 1e3,
        bpm_mv=_float(row, "bpm_mv"),
        bpm_av=_float(row, "bpm_av"),
    )


@dataclass
class Expectation:
    valve: str
    target_mm2: float
    source: str                 # where target_mm2 comes from
    gui_ellipse_mm2: float      # pi * r_long * r_small (MV) or pi * r^2 (AV) from the radii used


def _norm_path(p: str) -> str:
    return str(p).replace("\\", "/").rstrip("/").lower()


def expected_areas(summary: Optional[FitSummary], r_mv_long: float, r_mv_small: float, r_av: float,
                   case_dir: Optional[Path] = None, radius_tol: float = 0.005
                   ) -> Tuple[Dict[str, Expectation], List[str]]:
    """Target area per valve plus notes.

    The target is A_fit from summary.csv when available, otherwise the area the GUI radii
    stand for in calculate_valve_diameters.py (pi*r_long*r_small, pi*r^2).
    """
    notes: List[str] = []
    gui = {"MV": math.pi * r_mv_long * r_mv_small, "AV": math.pi * r_av * r_av}
    exp: Dict[str, Expectation] = {}
    if summary is None:
        for v in VALVES:
            exp[v] = Expectation(v, gui[v], "GUI radii (no summary.csv)", gui[v])
        notes.append("No summary.csv found: the target is the area the GUI radii stand for "
                     "(pi*r_long*r_small, pi*r^2), not a fitted area.")
        return exp, notes

    exp["MV"] = Expectation("MV", summary.A_MV_mm2, "summary.csv (A_fit)", gui["MV"])
    exp["AV"] = Expectation("AV", summary.A_AV_mm2, "summary.csv (A_fit)", gui["AV"])

    def differs(a: float, b: float) -> bool:
        return abs(a - b) > radius_tol * max(abs(b), 1e-12)

    pairs = [("MV long radius", r_mv_long, summary.r_MV_major_mm),
             ("MV small radius", r_mv_small, summary.r_MV_minor_mm),
             ("AV radius", r_av, summary.r_AV_mm)]
    diffs = [f"{name} {used:.3f} mm vs. {fit:.3f} mm" for name, used, fit in pairs if differs(used, fit)]
    if diffs:
        notes.append("The radii used for the geometry differ from summary.csv (" + "; ".join(diffs)
                     + "). Either they were changed by hand after 'Calculate Radii' or the summary "
                     "belongs to another case.")
    if case_dir is not None and summary.case_path and _norm_path(summary.case_path) != _norm_path(case_dir):
        notes.append(f"summary.csv was written for '{summary.case_path}', the current case folder is "
                     f"'{case_dir}'.")
    return exp, notes


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

@dataclass
class Measurement:
    valve: str
    ring_mm2: Optional[float] = None          # full valve disc as built
    perm_mm2: Optional[float] = None
    res_mm2: Optional[float] = None
    imperm_mm2: Optional[float] = None
    zones_source: str = "n/a"                  # "zone objects", "template estimate", "n/a"
    frame: Optional[ValveFrame] = None         # valve plane the zones were measured in
    zone_height_mm: Optional[float] = None     # height of the zone objects above the valve plane
    n_frames: int = 0                          # ventricle frames the ring was measured in
    frame_spread: float = 0.0                  # (max - min) / max of the ring area over the frames
    notes: List[str] = field(default_factory=list)

    @property
    def open_mm2(self) -> Optional[float]:
        if self.perm_mm2 is None or self.res_mm2 is None:
            return None
        return self.perm_mm2 + self.res_mm2


@dataclass
class Row:
    valve: str
    target_mm2: float
    target_source: str
    gui_ellipse_mm2: float
    ring_mm2: Optional[float]
    open_mm2: Optional[float]
    imperm_mm2: Optional[float]
    zones_source: str
    checked: str                  # "ring" or "perm+res"
    checked_mm2: Optional[float]
    status: str                   # "OK", "WARN", "n/a"
    n_frames: int = 0
    frame_spread: float = 0.0

    def pct(self, area: Optional[float]) -> Optional[float]:
        return None if area is None or self.target_mm2 <= 0 else 100.0 * area / self.target_mm2

    @property
    def velocity_factor(self) -> Optional[float]:
        """Mean velocity in the CFD relative to the fitted one, at equal flow rate."""
        if not self.checked_mm2:
            return None
        return self.target_mm2 / self.checked_mm2

    @property
    def radius_factor(self) -> Optional[float]:
        """Factor for both GUI radii of this valve that makes the checked area equal the target."""
        vf = self.velocity_factor
        return None if vf is None else math.sqrt(vf)


def evaluate(measurements: Dict[str, Measurement], expectations: Dict[str, Expectation],
             imperm_open: bool, tol_percent: float) -> Tuple[List[Row], bool]:
    rows: List[Row] = []
    ok = True
    for v in VALVES:
        m = measurements.get(v)
        e = expectations.get(v)
        if m is None or e is None:
            continue
        if imperm_open:
            checked, area = "ring", m.ring_mm2
        else:
            checked, area = "perm+res", m.open_mm2
        if area is None or e.target_mm2 <= 0:
            status = "n/a"
        else:
            status = "OK" if abs(area / e.target_mm2 - 1.0) * 100.0 <= tol_percent else "WARN"
        if status == "WARN":
            ok = False
        rows.append(Row(v, e.target_mm2, e.source, e.gui_ellipse_mm2, m.ring_mm2, m.open_mm2,
                        m.imperm_mm2, m.zones_source, checked, area, status, m.n_frames, m.frame_spread))
    return rows, ok


def _fmt(val: Optional[float], spec: str = ".2f", suffix: str = "") -> str:
    return "-" if val is None else f"{val:{spec}}{suffix}"


def format_report(rows: Sequence[Row], notes: Sequence[str], imperm_open: bool,
                  tol_percent: float) -> List[str]:
    """Console lines: one block per valve, then notes."""
    lines = [f"Reference: {'full ring (imperm counts as open)' if imperm_open else 'perm+res (imperm is a wall)'}, "
             f"tolerance +-{tol_percent:g} % area"]
    for r in rows:
        lines.append(f"{r.valve}: target {r.target_mm2:.2f} mm2 [{r.target_source}]  ->  {r.status}")
        frames = ""
        if r.n_frames:
            frames = f"   ({r.n_frames} frame{'s' if r.n_frames != 1 else ''}"
            if r.n_frames > 1:
                frames += ", identical" if r.frame_spread <= 1e-4 else f", spread {100 * r.frame_spread:.3f} %"
            frames += ")"
        lines.append(f"    ring      {_fmt(r.ring_mm2)} mm2 = {_fmt(r.pct(r.ring_mm2), '.1f', ' %')} of target{frames}")
        lines.append(f"    perm+res  {_fmt(r.open_mm2)} mm2 = {_fmt(r.pct(r.open_mm2), '.1f', ' %')} of target"
                     f"   (imperm {_fmt(r.imperm_mm2)} mm2, zones: {r.zones_source})")
        if r.velocity_factor is not None:
            lines.append(f"    checked area '{r.checked}': mean velocity x{r.velocity_factor:.3f} at equal flow; "
                         f"radius factor to match the target x{r.radius_factor:.3f}")
        if abs(r.gui_ellipse_mm2 - r.target_mm2) > 1e-9 * max(r.target_mm2, 1.0):
            lines.append(f"    (the radii used stand for {r.gui_ellipse_mm2:.2f} mm2 in calculate_valve_diameters.py)")
    for n in notes:
        lines.append(f"Note: {n}")
    return lines


# -----------------------------------------------------------------------------
# Blender front end
# -----------------------------------------------------------------------------

def _world_mesh(obj) -> Tuple[List[Vec], List[Tuple[int, ...]]]:
    mw = obj.matrix_world
    verts = [tuple(mw @ v.co) for v in obj.data.vertices]
    faces = [tuple(p.vertices) for p in obj.data.polygons]
    return verts, faces


def _group_faces(obj, group_name: str) -> Optional[List[Tuple[int, ...]]]:
    """Faces whose vertices all belong to the vertex group, or None if the group is missing."""
    vg = obj.vertex_groups.get(group_name)
    if vg is None:
        return None
    idx = vg.index
    members = {v.index for v in obj.data.vertices if any(g.group == idx and g.weight > 0.0 for g in v.groups)}
    return [tuple(p.vertices) for p in obj.data.polygons if all(i in members for i in p.vertices)]


def _find_object(bpy, name: str):
    """Object by exact name, otherwise the first top-level mesh whose name starts with name + '.'."""
    obj = bpy.data.objects.get(name)
    if obj is not None:
        return obj
    for o in bpy.data.objects:
        if o.type == 'MESH' and o.parent is None and o.name.startswith(name + "."):
            return o
    return None


def _template_areas(bpy, approach: int, valve: str) -> Optional[TemplateAreas]:
    """Unit-scale zone areas of the templates shipped in the .blend (used for the A3 estimate)."""
    names = {z: f"A{approach}_{valve}_{z}" for z in ZONE_TYPES}
    areas = {}
    for z, n in names.items():
        obj = bpy.data.objects.get(n)
        if obj is None or obj.type != 'MESH':
            return None
        verts = [tuple(v.co) for v in obj.data.vertices]  # mesh data: unit scale, template frame
        faces = [tuple(p.vertices) for p in obj.data.polygons]
        areas[z] = bottom_faces_area(verts, faces, UNIT_FRAME)
    total = sum(areas.values())
    return TemplateAreas(total, areas["perm"], areas["res"], areas["imperm"])


def _measure_valve(bpy, scene, valve: str, ventricles) -> Measurement:
    m = Measurement(valve)
    approach = int(getattr(scene, "approach", 4))
    if valve == "MV" and approach == 5:
        m.notes.append("A5 builds the MV as a 3D valve inside the ventricle; its area is not checked.")
        return m

    # Ring: the valve disc carries the vertex group 'MV' / 'AV' of the *_Surface template.
    rings = []
    frame = None
    for obj in ventricles:
        faces = _group_faces(obj, valve)
        if not faces:
            continue
        verts, _ = _world_mesh(obj)
        rings.append((obj.name, faces_area(verts, faces), len(faces)))
        if frame is None:
            origin, normal = plane_from_faces(verts, faces)
            frame = frame_from_normal(origin, normal)
    if rings:
        areas = [a for _, a, _ in rings]
        m.ring_mm2 = areas[0]
        spread = (max(areas) - min(areas)) / max(areas) if max(areas) > 0 else 0.0
        m.n_frames, m.frame_spread = len(areas), spread
        if spread > 1e-4:
            m.notes.append(f"{valve} disc area differs between frames by {100 * spread:.3f} % "
                           f"({min(areas):.2f} to {max(areas):.2f} mm2); the first frame is reported.")
        template = bpy.data.objects.get(f"A{approach}_{valve}_Surface")
        if template is not None and template.type == 'MESH' and rings[0][2] != len(template.data.polygons):
            m.notes.append(f"{valve} disc of '{rings[0][0]}' has {rings[0][2]} faces in vertex group "
                           f"'{valve}', the template A{approach}_{valve}_Surface has "
                           f"{len(template.data.polygons)}. The ring area may include extra faces.")
    else:
        m.notes.append(f"No ventricle carries faces in vertex group '{valve}'; the ring area cannot be measured.")

    # Zones: separate objects in A4 (MV) and A4/A5 (AV); A3 has none.
    prefix = valve.lower()
    zone_objs = {z: _find_object(bpy, f"{prefix}_{z}") for z in ZONE_TYPES}
    if frame is None:  # no disc found: fall back to the valve settings of the scene
        frame = valve_frame(getattr(scene, f"translation_{VALVE_MODE[valve].lower()}"),
                            getattr(scene, f"angle_{VALVE_MODE[valve].lower()}"))
    m.frame = frame
    if all(o is not None for o in zone_objs.values()):
        vals = {}
        heights = []
        for z, o in zone_objs.items():
            verts, faces = _world_mesh(o)
            vals[z] = bottom_faces_area(verts, faces, frame)
            heights.append(zone_height(verts, faces, frame))
        m.zone_height_mm = max(heights)
        m.perm_mm2, m.res_mm2, m.imperm_mm2 = vals["perm"], vals["res"], vals["imperm"]
        m.zones_source = "zone objects"
        if m.ring_mm2:
            total = sum(vals.values())
            if abs(total / m.ring_mm2 - 1.0) > 0.01:
                m.notes.append(f"{valve} zones cover {total:.2f} mm2 but the disc has {m.ring_mm2:.2f} mm2.")
    elif m.ring_mm2 is not None:
        # No zone objects (A3, or the check runs before 'Add atrium, aorta and valves'):
        # split the measured disc with the zone shares of the templates.
        source = f"A{approach}"
        tmpl = _template_areas(bpy, approach, valve)
        if tmpl is None and approach != 4:
            tmpl, source = _template_areas(bpy, 4, valve), "A4"
        if tmpl is not None and tmpl.open_fraction is not None:
            total = tmpl.perm + tmpl.res + tmpl.imperm
            m.perm_mm2 = m.ring_mm2 * tmpl.perm / total
            m.res_mm2 = m.ring_mm2 * tmpl.res / total
            m.imperm_mm2 = m.ring_mm2 * tmpl.imperm / total
            m.zones_source = f"estimated from {source} templates"
    return m


def disc_info_text(name: str, disc: PlaneDisc, ring_mm2: float, tol: float,
                   zone_height_mm: Optional[float]) -> str:
    """Console note for the bottom disc of a vessel (see lowest_disc)."""
    if disc.n_faces == 0:
        return f"'{name}': no bottom disc found near the valve."
    share = f"{disc.area:.2f} mm2 ({100 * disc.area / ring_mm2:.1f} % of the ventricle disc)"
    on_zones = zone_height_mm is not None and abs(disc.height - zone_height_mm) <= tol
    if abs(disc.height) <= tol:
        text = f"'{name}' disc in the valve plane: {share}."
    elif on_zones:
        text = (f"'{name}' bottom disc {disc.height:.3f} mm above the valve plane, "
                f"on top of the porous zones: {share}.")
    else:
        side = "above" if disc.height > 0 else "below"
        text = f"'{name}' bottom disc {abs(disc.height):.3f} mm {side} the valve plane: {share}."
    if zone_height_mm is not None and not on_zones:
        text += f" The porous zones end {zone_height_mm:.3f} mm above the valve plane, not at this disc."
    return text


def _disc_info(bpy, name: str, m: Measurement) -> Optional[str]:
    """Interface cross-check: the bottom disc of atrium/aorta against the ventricle disc."""
    obj = _find_object(bpy, name)
    if obj is None or m.ring_mm2 is None or m.frame is None:
        return None
    verts, faces = _world_mesh(obj)
    tol = 1e-3 * math.sqrt(m.ring_mm2) + 1e-6
    r_max = 2.0 * math.sqrt(m.ring_mm2 / math.pi)  # well outside the valve rim, inside the vessel's far end
    disc = lowest_disc(verts, faces, m.frame, tol, r_max)
    return disc_info_text(obj.name, disc, m.ring_mm2, tol, m.zone_height_mm)


def run_valve_area_check(context, ventricles) -> Tuple[bool, List[str], str]:
    """Measure the built valves and compare them with summary.csv / the GUI radii.

    Returns (ok, console_lines, short_message). ok is False if a checked area is outside the
    tolerance. Never raises for missing data; it reports it instead.
    """
    import os
    import warnings
    import bpy  # noqa: F401  (only available inside Blender)

    scene = context.scene
    imperm_open = bool(getattr(scene, "valve_check_imperm_open", True))
    tol = float(getattr(scene, "valve_check_tolerance", 2.0))
    notes: List[str] = []

    measurements = {v: _measure_valve(bpy, scene, v, ventricles) for v in VALVES}
    for m in measurements.values():
        notes.extend(m.notes)

    # Atrium / aorta: their bottom disc must match the ventricle disc (interface check, info only).
    # In A4 that disc sits on top of the porous zones, not in the valve plane.
    for valve, name in (("MV", "atrium"), ("AV", "aorta")):
        info = _disc_info(bpy, name, measurements[valve])
        if info:
            notes.append(info)

    # Target: summary.csv of the case the import folder belongs to (as 'Calculate Radii' resolves it).
    summary = None
    case_dir = None
    raw_dir = (getattr(scene, "ventricle_import_dir", "") or "").strip()
    import_dir = bpy.path.abspath(raw_dir) if raw_dir else ""
    if import_dir and os.path.isdir(import_dir):
        try:
            from calculate_valve_diameters import resolve_case_dir
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                case_dir = resolve_case_dir(Path(import_dir))
        except Exception as e:  # keep the check usable even if the module cannot be imported
            notes.append(f"Could not resolve the case folder: {e}")
        if case_dir is not None:
            path = Path(case_dir) / "calc_valve_diameters_outputs" / "summary.csv"
            if path.is_file():
                try:
                    summary = read_fit_summary(path)
                    notes.append(f"Target areas from {path}")
                except Exception as e:
                    notes.append(f"Cannot read {path}: {e}")
    else:
        notes.append("Import folder not set or missing; summary.csv cannot be located.")

    expectations, exp_notes = expected_areas(summary, scene.mitral_radius_long, scene.mitral_radius_small,
                                             scene.aortic_radius, case_dir)
    notes = exp_notes + notes
    rows, ok = evaluate(measurements, expectations, imperm_open, tol)
    lines = format_report(rows, notes, imperm_open, tol)
    warn = [r.valve for r in rows if r.status == "WARN"]
    if warn:
        short = (f"Valve area check: {', '.join(warn)} outside +-{tol:g} % of the target area "
                 f"(see System Console).")
    else:
        short = "Valve area check: all checked valve areas within tolerance."
    return ok, lines, short
