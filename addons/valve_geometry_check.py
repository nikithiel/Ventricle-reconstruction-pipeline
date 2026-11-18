"""Valve sizes of the pipeline, and the check that the built valves have the fitted areas.

calculate_valve_diameters.py fits an effective valve area A_fit from the volume curve and
the Doppler traces. scale_rotate_translate_object() in geometrical_heart_reconstruction.py
does not draw a circle or an ellipse: it multiplies the coordinates of template meshes
(A*_MV_Boundary, A*_MV_Surface, the porous zones, the atrium, ...) with stretch factors.
The built area is therefore a * s_x * s_y, where a is the template's area at unit scale
(about 2.56 for the D-shaped MV template, 3.12 for the AV template).

This module is the one place that links the two:
- template_shapes() measures a, the template width W_t and depth D_t in the .blend;
- mv_size() / av_diameter() turn A_fit into the panel values (MV width and depth with
  width/depth = R_target, AV diameter), mv_stretch() / av_stretch() turn the panel values
  into the stretch factors the build uses;
- run_valve_area_check() measures the built geometry and compares it with A_fit.

Layout:
- Everything except the Blender front end is free of bpy and numpy.
  tools/valve_area_diagnosis.py and calculate_valve_diameters.py import it outside Blender, so
  all of them compute exactly the same thing.
- The Blender front end (template_shapes(), run_valve_area_check()) at the bottom gets bpy
  passed in or imports it lazily.
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
ZONE_TYPES = ("perm", "res")  # imperm is merged into res (ADR 0001)


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
    which scale_rotate_translate_object() multiplies with the MV width stretch factor.
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
# Valve sizes: the one place that links A_fit, the panel values and the stretch factors
# -----------------------------------------------------------------------------

RATIO_BAND = (1.15, 1.35)  # plausible MV width/depth (healthy adults); outside it the panel and the check warn
MERGE_HINT = ("Run tools/merge_imperm_into_res.py once in Blender 3.1 (see README, 'Merge imperm into res'), "
              "then save the .blend.")


@dataclass(frozen=True)
class TemplateShape:
    """Outline of a valve template at unit scale (A{n}_{MV,AV}_Boundary).

    area is the flow opening area at factor 1 (A3/A4 and all AV: the whole disc).
    A5 MV: also the disc for now; scaling the opening of the open leaflets (backlog P13) goes here.
    """
    area: float   # a
    width: float  # W_t, extent in template x (commissure to commissure for the MV)
    depth: float  # D_t, extent in template y (antero-posterior)

    @property
    def fill(self) -> float:
        """c = a / (W_t * D_t): built MV area = c * width * depth."""
        return self.area / (self.width * self.depth)


def _shape(points: Sequence[Sequence[float]], area: float) -> TemplateShape:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return TemplateShape(area, max(xs) - min(xs), max(ys) - min(ys))


def shape_from_loop(verts: Sequence[Sequence[float]], edges: Sequence[Tuple[int, int]]) -> TemplateShape:
    """Shape of a closed edge loop (the *_Boundary templates). Raises ValueError if it is no single loop."""
    pts = [verts[i] for i in order_edge_loop(edges)]
    return _shape(pts, polygon_area_xy(pts))


def shape_from_faces(verts: Sequence[Sequence[float]], faces: Sequence[Sequence[int]]) -> TemplateShape:
    """Shape of a flat disc (the *_Surface templates), the fallback if the boundary is no single loop."""
    used = sorted({i for f in faces for i in f})
    return _shape([verts[i] for i in used], projected_faces_area(verts, faces, (0.0, 0.0, 1.0)))


def mv_size(a_fit: float, shape: TemplateShape, ratio: float) -> Tuple[float, float]:
    """(width, depth) of the MV whose built area is a_fit and whose width/depth is ratio."""
    depth = math.sqrt(a_fit / (shape.fill * ratio))
    return ratio * depth, depth


def av_diameter(a_fit: float, shape: TemplateShape) -> float:
    """Diameter of the AV whose built area is a_fit (the AV template has circumradius 1)."""
    return 2.0 * math.sqrt(a_fit / shape.area)


def mv_stretch(width: float, depth: float, shape: TemplateShape) -> Vec:
    """Stretch factors (x, y, z) that build the MV with this width and depth; z as the pipeline always did."""
    sx, sy = width / shape.width, depth / shape.depth
    return sx, sy, (sx + sy) / 2.0


def av_stretch(diameter: float) -> Vec:
    s = diameter / 2.0
    return s, s, s


def ratio_in_band(ratio: float) -> bool:
    return RATIO_BAND[0] <= ratio <= RATIO_BAND[1]


# -----------------------------------------------------------------------------
# Template coefficients
# -----------------------------------------------------------------------------

@dataclass
class TemplateAreas:
    """Areas of one valve template set at unit scale."""
    ring: float                      # area of the full valve disc (= perm + res)
    perm: Optional[float] = None
    res: Optional[float] = None

    @property
    def open(self) -> Optional[float]:
        if self.perm is None or self.res is None:
            return None
        return self.perm + self.res


# -----------------------------------------------------------------------------
# Expected values: summary.csv of calculate_valve_diameters.py and the panel values
# -----------------------------------------------------------------------------

@dataclass
class FitSummary:
    path: Path
    case_path: str
    A_MV_mm2: float
    A_AV_mm2: float
    bpm_mv: Optional[float]
    bpm_av: Optional[float]
    approach: Optional[int] = None
    R_target: Optional[float] = None
    width_MV_mm: Optional[float] = None  # None in the old format (radii instead of sizes)
    depth_MV_mm: Optional[float] = None
    d_AV_mm: Optional[float] = None

    @property
    def old_format(self) -> bool:
        return self.width_MV_mm is None


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

    Column names may carry their unit ('A_MV [m^2]') or not (old files). Old files store A_AV
    negative (Doppler sign convention), so |A| is used.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as fh:
        rows = [{k.split(" [")[0].strip(): v for k, v in r.items() if k} for r in csv.DictReader(fh)]
    if not rows:
        raise ValueError(f"{path} contains no data row")
    row = rows[-1]
    missing = [k for k in ("A_MV", "A_AV") if _float(row, k) is None]
    if missing:
        raise ValueError(f"{path} lacks valid values for {missing}")

    def mm(key: str) -> Optional[float]:
        v = _float(row, key)
        return None if v is None else v * 1e3

    approach = _float(row, "approach")
    return FitSummary(
        path=path,
        case_path=(row.get("path") or "").strip(),
        A_MV_mm2=abs(_float(row, "A_MV")) * 1e6,
        A_AV_mm2=abs(_float(row, "A_AV")) * 1e6,
        bpm_mv=_float(row, "bpm_mv"),
        bpm_av=_float(row, "bpm_av"),
        approach=None if approach is None else int(approach),
        R_target=_float(row, "R_target"),
        width_MV_mm=mm("width_MV"),
        depth_MV_mm=mm("depth_MV"),
        d_AV_mm=mm("d_AV"),
    )


@dataclass
class Expectation:
    valve: str
    target_mm2: float
    source: str                 # where target_mm2 comes from
    panel_mm2: float            # area the panel values stand for (c*width*depth, a*(d/2)^2)
    stretch: Vec                # stretch factors the build uses for the panel values


def _norm_path(p: str) -> str:
    return str(p).replace("\\", "/").rstrip("/").lower()


def expected_areas(summary: Optional[FitSummary], width: float, depth: float, diameter: float,
                   shapes: Dict[str, TemplateShape], approach: Optional[int] = None,
                   case_dir: Optional[Path] = None, size_tol: float = 0.005
                   ) -> Tuple[Dict[str, Expectation], List[str]]:
    """Target area per valve plus notes.

    The target is A_fit from summary.csv when available, otherwise the area the panel values
    stand for with the measured templates.
    """
    notes: List[str] = []
    panel = {"MV": shapes["MV"].fill * width * depth, "AV": shapes["AV"].area * (diameter / 2.0) ** 2}
    stretch = {"MV": mv_stretch(width, depth, shapes["MV"]), "AV": av_stretch(diameter)}
    exp: Dict[str, Expectation] = {}
    if summary is None:
        for v in VALVES:
            exp[v] = Expectation(v, panel[v], "panel values (no summary.csv)", panel[v], stretch[v])
        notes.append("No summary.csv found: the target is the area the panel values stand for "
                     "(c * width * depth, a * (d/2)^2), not a fitted area.")
        return exp, notes

    exp["MV"] = Expectation("MV", summary.A_MV_mm2, "summary.csv (A_fit)", panel["MV"], stretch["MV"])
    exp["AV"] = Expectation("AV", summary.A_AV_mm2, "summary.csv (A_fit)", panel["AV"], stretch["AV"])

    if summary.old_format:
        notes.append("summary.csv has the old format (radii, no width/depth/diameter): only A_fit is used. "
                     "Run 'Calculate Radii' again.")
    else:
        pairs = [("MV width", width, summary.width_MV_mm), ("MV depth", depth, summary.depth_MV_mm),
                 ("AV diameter", diameter, summary.d_AV_mm)]
        diffs = [f"{name} {used:.3f} mm vs. {fit:.3f} mm" for name, used, fit in pairs
                 if fit is not None and abs(used - fit) > size_tol * max(abs(fit), 1e-12)]
        if diffs:
            notes.append("The panel values differ from summary.csv (" + "; ".join(diffs)
                         + "). Either they were changed by hand after 'Calculate Radii' or the summary "
                         "belongs to another case.")
        if approach is not None and summary.approach is not None and summary.approach != approach:
            notes.append(f"summary.csv was computed for approach A{summary.approach}, the scene uses "
                         f"A{approach}. Run 'Calculate Radii' again.")
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
    zones_source: str = "n/a"                  # "zone objects", "template estimate", "n/a"
    frame: Optional[ValveFrame] = None         # valve plane the zones were measured in
    zone_height_mm: Optional[float] = None     # height of the zone objects above the valve plane
    n_frames: int = 0                          # ventricle frames the ring was measured in
    frame_spread: float = 0.0                  # (max - min) / max of the ring area over the frames
    width_mm: Optional[float] = None           # MV disc extent in the valve frame of the scene (x)
    depth_mm: Optional[float] = None           # ... and in y
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
    panel_mm2: float
    stretch: Vec
    ring_mm2: Optional[float]
    open_mm2: Optional[float]
    zones_source: str
    checked_mm2: Optional[float]  # perm+res, the disc if no zones are known
    status: str                   # "OK", "WARN", "n/a"
    n_frames: int = 0
    frame_spread: float = 0.0
    width_mm: Optional[float] = None
    depth_mm: Optional[float] = None

    def pct(self, area: Optional[float]) -> Optional[float]:
        return None if area is None or self.target_mm2 <= 0 else 100.0 * area / self.target_mm2

    @property
    def velocity_factor(self) -> Optional[float]:
        """Mean velocity in the CFD relative to the fitted one, at equal flow rate."""
        if not self.checked_mm2:
            return None
        return self.target_mm2 / self.checked_mm2

    @property
    def size_factor(self) -> Optional[float]:
        """Factor for width and depth (MV) or the diameter (AV) that makes the checked area equal the target."""
        vf = self.velocity_factor
        return None if vf is None else math.sqrt(vf)

    @property
    def ratio(self) -> Optional[float]:
        return self.width_mm / self.depth_mm if self.width_mm and self.depth_mm else None


def evaluate(measurements: Dict[str, Measurement], expectations: Dict[str, Expectation],
             tol_percent: float) -> Tuple[List[Row], bool]:
    """Rows per valve; ok is False if an area is outside the tolerance or the MV width/depth outside RATIO_BAND."""
    rows: List[Row] = []
    ok = True
    for v in VALVES:
        m = measurements.get(v)
        e = expectations.get(v)
        if m is None or e is None:
            continue
        area = m.open_mm2 if m.open_mm2 is not None else m.ring_mm2
        if area is None or e.target_mm2 <= 0:
            status = "n/a"
        else:
            status = "OK" if abs(area / e.target_mm2 - 1.0) * 100.0 <= tol_percent else "WARN"
        row = Row(v, e.target_mm2, e.source, e.panel_mm2, e.stretch, m.ring_mm2, m.open_mm2, m.zones_source,
                  area, status, m.n_frames, m.frame_spread, m.width_mm, m.depth_mm)
        if status == "WARN" or (row.ratio is not None and not ratio_in_band(row.ratio)):
            ok = False
        rows.append(row)
    return rows, ok


def _fmt(val: Optional[float], spec: str = ".2f", suffix: str = "") -> str:
    return "-" if val is None else f"{val:{spec}}{suffix}"


def format_report(rows: Sequence[Row], notes: Sequence[str], tol_percent: float) -> List[str]:
    """Console lines: one block per valve, then notes."""
    lines = [f"Reference: perm+res (imperm is part of res), tolerance +-{tol_percent:g} % area, "
             f"MV width/depth {RATIO_BAND[0]:g} to {RATIO_BAND[1]:g}"]
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
                     f"   (zones: {r.zones_source})")
        if r.velocity_factor is not None:
            lines.append(f"    mean velocity x{r.velocity_factor:.3f} at equal flow; "
                         f"size factor to match the target x{r.size_factor:.3f}")
        if r.ratio is not None:
            band = "" if ratio_in_band(r.ratio) else f" outside {RATIO_BAND[0]:g} to {RATIO_BAND[1]:g} -> WARN"
            if r.ratio < 1.0:
                band += " (width and depth swapped?)"
            lines.append(f"    width {r.width_mm:.2f} mm, depth {r.depth_mm:.2f} mm, width/depth {r.ratio:.2f}{band}")
        sx, sy, sz = r.stretch
        lines.append(f"    stretch factors x {sx:.3f}, y {sy:.3f}, z {sz:.3f}; "
                     f"the panel values stand for {r.panel_mm2:.2f} mm2")
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


class TemplateError(RuntimeError):
    """The valve templates in the .blend cannot be used (imperm not merged, outline missing)."""


def template_shapes(bpy, approach: int) -> Dict[str, TemplateShape]:
    """MV and AV template outline of an approach, measured in the .blend.

    The one place Calculate Radii, the build (get_valve_data) and the check take the template
    sizes from. Raises TemplateError while any A{n}_*_imperm template is left: the build copies
    only perm and res, so the zones would miss their outer band.
    """
    left = [n for n in (f"A{a}_{v}_imperm" for a in (3, 4, 5) for v in VALVES) if bpy.data.objects.get(n) is not None]
    if left:
        raise TemplateError(f"The .blend still holds the imperm templates {', '.join(left)}. {MERGE_HINT}")
    shapes = {}
    for v in VALVES:
        boundary = bpy.data.objects.get(f"A{approach}_{v}_Boundary")
        surface = bpy.data.objects.get(f"A{approach}_{v}_Surface")
        try:
            if boundary is None:
                raise ValueError("missing")
            shapes[v] = shape_from_loop([tuple(p.co) for p in boundary.data.vertices],
                                        [tuple(e.vertices) for e in boundary.data.edges])
        except ValueError:
            if surface is None:
                raise TemplateError(f"Template A{approach}_{v}_Boundary is missing or not a single edge loop, "
                                    f"and A{approach}_{v}_Surface is missing.")
            shapes[v] = shape_from_faces([tuple(p.co) for p in surface.data.vertices],
                                         [tuple(p.vertices) for p in surface.data.polygons])
    return shapes


def _template_areas(bpy, approach: int, valve: str) -> Optional[TemplateAreas]:
    """Unit-scale zone areas of the templates shipped in the .blend (used for the A3 estimate)."""
    areas = {}
    for z in ZONE_TYPES:
        obj = bpy.data.objects.get(f"A{approach}_{valve}_{z}")
        if obj is None or obj.type != 'MESH':
            return None
        verts = [tuple(v.co) for v in obj.data.vertices]  # mesh data: unit scale, template frame
        faces = [tuple(p.vertices) for p in obj.data.polygons]
        areas[z] = bottom_faces_area(verts, faces, UNIT_FRAME)
    return TemplateAreas(sum(areas.values()), areas["perm"], areas["res"])


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
            if valve == "MV":  # extent in the frame the build placed the template in (local x/y = template x/y)
                scene_frame = valve_frame(scene.translation_mitral, scene.angle_mitral)
                loc = [scene_frame.to_local(verts[i]) for f in faces for i in f]
                m.width_mm = max(p[0] for p in loc) - min(p[0] for p in loc)
                m.depth_mm = max(p[1] for p in loc) - min(p[1] for p in loc)
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
        m.perm_mm2, m.res_mm2 = vals["perm"], vals["res"]
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
        if tmpl is not None and tmpl.ring > 0:
            m.perm_mm2 = m.ring_mm2 * tmpl.perm / tmpl.ring
            m.res_mm2 = m.ring_mm2 * tmpl.res / tmpl.ring
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
    """Measure the built valves and compare them with summary.csv / the panel values.

    Returns (ok, console_lines, short_message). ok is False if a checked area is outside the
    tolerance or the MV width/depth outside RATIO_BAND. Never raises for missing data; it
    reports it instead. Raises TemplateError while imperm templates are left in the .blend.
    """
    import os
    import warnings
    import bpy  # noqa: F401  (only available inside Blender)

    scene = context.scene
    approach = int(getattr(scene, "approach", 4))
    shapes = template_shapes(bpy, approach)  # raises TemplateError while imperm templates are left
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

    expectations, exp_notes = expected_areas(summary, scene.mitral_width, scene.mitral_depth,
                                             scene.aortic_diameter, shapes, approach, case_dir)
    notes = exp_notes + notes
    rows, ok = evaluate(measurements, expectations, tol)
    lines = format_report(rows, notes, tol)
    problems = [f"{r.valve} outside +-{tol:g} % of the target area" for r in rows if r.status == "WARN"]
    problems += [f"MV width/depth {r.ratio:.2f} outside {RATIO_BAND[0]:g} to {RATIO_BAND[1]:g}"
                 for r in rows if r.ratio is not None and not ratio_in_band(r.ratio)]
    if problems:
        short = f"Valve area check: {'; '.join(problems)} (see System Console)."
    else:
        short = "Valve area check: all checked valve areas within tolerance."
    return ok, lines, short
