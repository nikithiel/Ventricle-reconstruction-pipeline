# -*- coding: utf-8 -*-
"""
Remesh the free region of a surface mesh just so that it has a more naturally distributed mesh complexity
Mainly intended as a subscript for the geometrical heart reconstruction blender addon.
Functions should work by itself.

Faces whose vertices all belong to one of the groups in VERTEX_GROUP
are preserved exactly. The rest is remeshed with gmsh: the element size
starts at the interface with the local boundary-edge length and grows by
GROWTH per element row until it reaches TARGET_DISTANCE. Edge flips and
shape-preserving smoothing follow. The result is a new object "<Name><NAME_SUFFIX>",
while the original is only hidden.

Usage: Select an object, adjust the settings below, and just run the script on that object.
remesh_object() can also be run on a script on objects for the same effect.

"""

# =============================================================================
# SETTINGS
# =============================================================================

FIXED_SURFACE = "gruppe"  # "gruppe":  To use all groups in VERTEX_GROUP
                           # "selection": the faces selected in Edit Mode
VERTEX_GROUP = "AV","MV"   # Group(s) for fixed faces, e.g. ["MV", "AV"]
FIXED_BOUNDARIES = "all"  # Open boundaries (edge loops) that must remain vertex-exact:
                        #   "all"   = all remain unchanged
                        #   ["lower_basal_edge_loop"] = only boundaries in these groups; all others are remeshed
                        #   [] or "none" = remesh all boundaries. Remeshed boundaries retain their shape and vertex groups.
NAME_SUFFIX = "_remesh" # Suffix for the new object, e.g. "_A" or "_B"
TARGET_DISTANCE = None      # Edge length far from the interface;
                        # None = median edge length in the current free mesh
GROWTH = 1.2          # Size increase per element row (1.1 ... 1.3 is typical)
ANGLE = 30.0      # Degrees. Edges with a sharper crease are preserved
CURVATURE = 0  #  0: additionally refine based on curvature(elements per full circle, e.g. 24). 0 = off
SMOOTH = 10           # Passes of edge flips and smoothing after gmsh (Shape is preserved). 5 ... 30, 0 = off
SHAPE_SMOOTH = 0       # also smooth the SHAPE (3 ... 10 steps) WARNING:SLIGHTLY CHANGES THE GEOMETRY. 0 = off
SMOOTH_ONLY = False    # True: skip gmsh and only post-process the existing mesh
ALGORITHM = 6         # gmsh algorithm: 6 = Frontal-Delaunay, 5 = Delaunay, 1 = MeshAdapt (robust for difficult surfaces)
INSTALL_GMSH = True  # Automatically install gmsh via pip if needed
VERBOSE = False    # True: verbose gmsh output in the system console

# =============================================================================

import math
import os
import sys
import time
from collections import deque

import numpy as np

try:
    import bpy
    import bmesh
except ImportError:  # outside Blender, e.g. for testing the core
    bpy = None
    bmesh = None


class RemeshError(Exception):
    """Error message"""


# -----------------------------------------------------------------------------
# Load or install gmsh
# -----------------------------------------------------------------------------

def _gmsh_search_dirs():
    """Directories where pip may have placed gmsh.py (including with --user)."""
    import site
    dirs = list(sys.path)
    try:
        dirs += list(site.getsitepackages())
    except Exception:
        pass
    try:
        dirs.append(site.getusersitepackages())
    except Exception:
        pass
    bases = [sys.prefix, getattr(sys, "base_prefix", sys.prefix)]
    try:
        bases.append(site.getuserbase())
    except Exception:
        pass
    pyver = "Python%d%d" % sys.version_info[:2]
    subs = ("lib", "Lib", "lib64", os.path.join(pyver, "site-packages"),
            os.path.join(pyver, "Lib"))
    dirs += [os.path.join(b, sub) for b in bases for sub in subs]
    out = []
    for d in dirs:
        if d and d not in out and os.path.isdir(d):
            out.append(d)
    return out


def _add_gmsh_to_path():
    for d in _gmsh_search_dirs():
        if os.path.isfile(os.path.join(d, "gmsh.py")):
            if d not in sys.path:
                sys.path.append(d)
            return True
    return False


def _run_cmd(cmd):
    import subprocess
    print("  >", " ".join(cmd))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
    except Exception as e:  # e.g. Python executable not found
        print("   ", e)
        return False
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    if out:
        print(out[-3000:])
    return r.returncode == 0


def _try_import_gmsh():
    """Import gmsh. Returns None if not installed; raises an error if broken."""
    try:
        import gmsh
    except ImportError:
        return None
    except Exception as e:  # e.g. OSError while loading the library
        raise RemeshError(_lib_error_text(e))
    try:
        getattr(gmsh.lib, "gmshInitialize")   # Bibliothek wirklich ladbar?
    except Exception as e:
        raise RemeshError(_lib_error_text(e))
    return gmsh


def _lib_error_text(err):
    return ("The gmsh library could not be loaded:\n%s\n"
            "On Linux, required system libraries are often missing, e.g.:\n"
            "sudo apt install libglu1-mesa libxcursor1 libxft2 libxinerama1" % err)


def import_gmsh(auto_install=True):
    """Import the gmsh module; if necessary, install it into Blender’s Python:
    pip (set it up first via ensurepip if needed), otherwise pip --user."""
    gmsh = _try_import_gmsh()
    if gmsh is None and _add_gmsh_to_path():
        gmsh = _try_import_gmsh()
    if gmsh is not None:
        return gmsh

    if not auto_install:
        raise RemeshError(
            "gmsh is not installed. Set INSTALL_GMSH = True or "
            "install it manually:\n<Blenders Python> -m pip install gmsh")

    py = sys.executable
    print("\nInstalling gmsh in Blender’s Python (%s) ..." % py)
    if not _run_cmd([py, "-m", "pip", "--version"]):
        if not _run_cmd([py, "-m", "ensurepip", "--default-pip"]):
            _run_cmd([py, "-m", "ensurepip", "--user"])
    if not (_run_cmd([py, "-m", "pip", "install", "gmsh"])
            or _run_cmd([py, "-m", "pip", "install", "--user", "gmsh"])):
        raise RemeshError(
            "Installing gmsh failed (details in the "
            "system console). On Windows, it often helps to start Blender once as "
            "administrator and run the script again.")

    import importlib
    importlib.invalidate_caches()
    _add_gmsh_to_path()
    gmsh = _try_import_gmsh()
    if gmsh is None:
        raise RemeshError(
            "gmsh was installed, but still cannot be loaded. "
            "Please restart Blender and run the script again.")
    print("gmsh %s is ready." % getattr(gmsh, "__version__", "?"))
    return gmsh


# -----------------------------------------------------------------------------
# Helper structures (usable without Blender)
# -----------------------------------------------------------------------------

class _KD:
    """Point search: use mathutils.kdtree in Blender, otherwise use numpy."""

    def __init__(self, pts):
        self.pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        try:
            import mathutils.kdtree as mkd
        except ImportError:
            self.tree = None
            return
        self.tree = mkd.KDTree(len(self.pts))
        for i, p in enumerate(self.pts.tolist()):
            self.tree.insert(p, i)
        self.tree.balance()

    def nearest(self, q):
        """(Index, distance) of the nearest point."""
        if self.tree is not None:
            _, i, d = self.tree.find(q)
            return int(i), float(d)
        d = np.sqrt(((self.pts - np.asarray(q)) ** 2).sum(1))
        i = int(d.argmin())
        return i, float(d[i])

    def within(self, q, r):
        """Indices of all points within distance <= r."""
        if self.tree is not None:
            return np.fromiter((i for _, i, _ in self.tree.find_range(q, r)),
                               dtype=np.int64)
        d2 = ((self.pts - np.asarray(q)) ** 2).sum(1)
        return np.nonzero(d2 <= r * r)[0]


class GradedSize:
    """Size field that grows outward from the local boundary-edge length.

    s(x) = min( min_i (h_i + g*d_i)  over boundary vertices with h_i <= hmax,
                max(hmax, max_j (h_j - g*d_j))  over boundary vertices with h_j > hmax )

    where g = GROWTH - 1 and d = distance to the boundary vertex. At the boundary, s equals
    the local edge length and changes by at most g per unit of length
    (i.e. neighboring elements differ by no more than a factor of
    GROWTH), and approaches hmax far from the boundary.
    """

    def __init__(self, pts, h, growth, hmax):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        h = np.asarray(h, dtype=np.float64)
        self.g = max(float(growth) - 1.0, 0.01)
        self.hmax = float(hmax)
        fine = h <= self.hmax
        self.Pf, self.Hf = pts[fine], h[fine]
        self.Pc, self.Hc = pts[~fine], h[~fine]
        self.kf = _KD(self.Pf) if len(self.Pf) else None
        self.kc = _KD(self.Pc) if len(self.Pc) else None
        self.hf_min = float(self.Hf.min()) if len(self.Hf) else self.hmax
        self.rc = (float(self.Hc.max()) - self.hmax) / self.g if len(self.Hc) else 0.0

    def __call__(self, x, y, z):
        q = (x, y, z)
        cap = self.hmax
        if self.kc is not None:
            idx = self.kc.within(q, self.rc)
            if len(idx):
                d = np.sqrt(((self.Pc[idx] - q) ** 2).sum(1))
                cap = max(cap, float((self.Hc[idx] - self.g * d).max()))
        if self.kf is None:
            return cap
        i0, d0 = self.kf.nearest(q)
        best = min(self.Hf[i0] + self.g * d0, cap)
        r = (best - self.hf_min) / self.g
        if r > 0:
            idx = self.kf.within(q, r)
            if len(idx):
                d = np.sqrt(((self.Pf[idx] - q) ** 2).sum(1))
                best = min(best, float((self.Hf[idx] + self.g * d).min()))
        return float(best)


class _PointLookup:
    """Find boundary vertices by their coordinates (exact match, otherwise within tolerance)."""

    def __init__(self, pts, ids, tol):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        self.ids = np.asarray(ids, dtype=np.int64)
        self.exact = {tuple(p): int(i) for p, i in zip(pts.tolist(), self.ids.tolist())}
        self.tol = tol
        self.kd = _KD(pts) if len(pts) else None

    def find(self, p):
        p = tuple(p)
        i = self.exact.get(p)
        if i is not None:
            return i
        if self.kd is None:
            return -1
        j, d = self.kd.nearest(p)
        return int(self.ids[j]) if d <= self.tol else -1

    def find_many(self, xyz):
        return np.array([self.find(p) for p in np.asarray(xyz).tolist()],
                        dtype=np.int64)


class _Edges:
    """Edges of a triangle mesh, sorted once; everything else is derived from this.

    Half-edge 3k+j of triangle k runs from T[k, j] to T[k, (j+1) % 3].
      E       (3t, 2)  Halbkanten (gerichtet)
      code    (3t,)    key of the undirected edge: min * n + max
      sign    (3t,)    +1 if the half-edge runs from min to max
      cnt     (3t,)    number of half-edges with this key (1 = boundary)
      e1, e2           neighbor pairs: half-edges with the same key
      unique  (k, 2)   undirected edges, sorted by key
      ucode   (k,)     their keys
    """

    def __init__(self, T, n):
        T = np.asarray(T, dtype=np.int64)
        self.n = np.int64(max(n, 1))
        self.E = E = T[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
        self.code = (np.minimum(E[:, 0], E[:, 1]) * self.n
                     + np.maximum(E[:, 0], E[:, 1]))
        self.sign = np.where(E[:, 0] < E[:, 1], 1, -1).astype(np.int8)
        order = np.argsort(self.code, kind="stable")
        cs = self.code[order]
        same = np.nonzero(cs[:-1] == cs[1:])[0]
        self.e1, self.e2 = order[same], order[same + 1]
        first = np.ones(len(cs), dtype=bool)
        first[1:] = cs[1:] != cs[:-1]
        starts = np.nonzero(first)[0]
        runs = np.diff(np.append(starts, len(cs)))
        self.cnt = np.empty(len(cs), dtype=np.int64)
        self.cnt[order] = np.repeat(runs, runs)
        self.ucode = cs[starts]
        self.unique = np.stack([self.ucode // self.n, self.ucode % self.n], axis=1)
        self.nonmanifold = int((runs > 2).sum())

    @property
    def boundary(self):
        """Boundary edges (oriented as in the mesh)."""
        return self.E[self.cnt == 1]


def _min_angles_deg(P, T):
    if len(T) == 0:
        return np.zeros(0)
    A, B, C = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    a = np.linalg.norm(B - C, axis=1)
    b = np.linalg.norm(C - A, axis=1)
    c = np.linalg.norm(A - B, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.stack([(b * b + c * c - a * a) / (2 * b * c),
                        (c * c + a * a - b * b) / (2 * c * a),
                        (a * a + b * b - c * c) / (2 * a * b)], axis=1)
    ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return np.nan_to_num(ang, nan=0.0).min(axis=1)


def _areas(P, T):
    return 0.5 * np.linalg.norm(
        np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]), axis=1)


def quality(P, T):
    """Triangle-mesh metrics: count, minimum and mean
    minimum interior angle, fraction < 30°, size jump between neighbors
    (square root of the area ratio, 95th percentile)."""
    P = np.asarray(P, dtype=np.float64)
    T = np.asarray(T, dtype=np.int64)
    ang = _min_angles_deg(P, T)
    res = {"n": int(len(T)),
           "min_angle": float(ang.min()) if len(ang) else 0.0,
           "mean_min_angle": float(ang.mean()) if len(ang) else 0.0,
           "below30": float((ang < 30.0).mean()) if len(ang) else 0.0,
           "ratio95": 1.0}
    if len(T) > 1:
        ed = _Edges(T, len(P))
        A = _areas(P, T)
        a1, a2 = A[ed.e1 // 3], A[ed.e2 // 3]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.sqrt(np.maximum(a1, a2) / np.minimum(a1, a2))
        r = r[np.isfinite(r)]
        if len(r):
            res["ratio95"] = float(np.percentile(r, 95))
    return res


# -----------------------------------------------------------------------------
# Orientation
# -----------------------------------------------------------------------------

def _make_vote(P0, T0):
    """Orientation of a mesh component by comparison with the original normals."""
    C0 = P0[T0].mean(axis=1)
    N0 = np.cross(P0[T0[:, 1]] - P0[T0[:, 0]], P0[T0[:, 2]] - P0[T0[:, 0]])

    def vote(P, T):
        k = min(len(T), 31)
        s = 0.0
        for t in T[np.linspace(0, len(T) - 1, k).astype(np.int64)]:
            c = P[t].mean(axis=0)
            nn = np.cross(P[t[1]] - P[t[0]], P[t[2]] - P[t[0]])
            j = int(((C0 - c) ** 2).sum(axis=1).argmin())
            s += np.sign(float(np.dot(nn, N0[j])))
        return 1 if s >= 0 else -1

    return vote


def _orient(T, n, ref, vote):
    """Orient all triangles consistently.

    ref: Schlüssel einer festen Randkante -> +1/-1 (direction in the original mesh).
    vote: Funktion(Dreiecke) -> +1/-1, decides for components without a fixed
    boundary edge. Returns: (triangles, {"conflicts", "parts_without_boundary"}).
    """
    T = np.asarray(T, dtype=np.int64)
    t = len(T)
    ed = _Edges(T, n)
    code, sign, cnt, e1, e2 = ed.code, ed.sign, ed.cnt, ed.e1, ed.e2
    t1, t2 = e1 // 3, e2 // 3
    rel = (-sign[e1].astype(np.int64) * sign[e2].astype(np.int64))
    src = np.concatenate([t1, t2])
    dst = np.concatenate([t2, t1])
    rr = np.concatenate([rel, rel])
    o = np.argsort(src, kind="stable")
    src, dst, rr = src[o], dst[o], rr[o]
    start = np.searchsorted(src, np.arange(t + 1)).tolist()
    dst, rr = dst.tolist(), rr.tolist()

    flip = [0] * t
    conflicts = 0

    def spread(k0, f0, comp=None):
        nonlocal conflicts
        flip[k0] = f0
        dq = deque([k0])
        while dq:
            k = dq.popleft()
            if comp is not None:
                comp.append(k)
            fk = flip[k]
            for j in range(start[k], start[k + 1]):
                nb, want = dst[j], rr[j] * fk
                if flip[nb] == 0:
                    flip[nb] = want
                    dq.append(nb)
                elif flip[nb] != want:
                    conflicts += 1

    for e in np.nonzero(cnt == 1)[0].tolist():
        r = ref.get(int(code[e]))
        if r is None:
            continue
        k0, f0 = e // 3, int(sign[e]) * r
        if flip[k0] == 0:
            spread(k0, f0)
        elif flip[k0] != f0:
            conflicts += 1

    free_parts = 0
    for k0 in range(t):
        if flip[k0] != 0:
            continue
        comp = []
        spread(k0, 1, comp)
        comp = np.asarray(comp, dtype=np.int64)
        cur = T[comp].copy()
        neg = np.asarray([flip[k] for k in comp.tolist()]) < 0
        cur[neg] = cur[neg][:, [0, 2, 1]]
        if vote(cur) < 0:
            for k in comp.tolist():
                flip[k] = -flip[k]
        free_parts += 1

    flip = np.asarray(flip)
    out = T.copy()
    m = flip < 0
    out[m] = out[m][:, [0, 2, 1]]
    return out, {"conflicts": conflicts // 2, "parts_without_boundary": free_parts}


# -----------------------------------------------------------------------------
# gmsh
# -----------------------------------------------------------------------------

def _gmsh_init(gmsh):
    try:
        if gmsh.isInitialized():
            gmsh.finalize()
    except Exception:
        pass
    # Older gmsh versions do not support all arguments
    for kwargs in ({"readConfigFiles": False, "interruptible": False},
                   {"readConfigFiles": False}):
        try:
            gmsh.initialize(**kwargs)
            return
        except TypeError:
            continue
    gmsh.initialize()   # last attempt without arguments


def _run_gmsh(gmsh, P, T, lookup, size_fn, size_cap, feature_angle,
              curvature_elements, verbose, algorithm=6, split_curves=False):
    _gmsh_init(gmsh)
    logging = False
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("General.Verbosity", 5 if verbose else 2)
        try:
            gmsh.logger.start()
            logging = True
        except Exception:
            pass

        gmsh.model.add("blender_remesh")
        s0 = gmsh.model.addDiscreteEntity(2)
        n = len(P)
        gmsh.model.mesh.addNodes(2, s0, list(range(1, n + 1)), P.ravel().tolist())
        gmsh.model.mesh.addElementsByType(s0, 2, list(range(1, len(T) + 1)),
                                          (T + 1).ravel().tolist())

        # Split into patches that can each be
        # parameterized with a single chart; edges above the crease angle become curves.
        # split_curves: Split boundary curves at creases so corners of a newly
        # remeshed boundary remain as vertices
        curve_angle = math.radians(feature_angle) if split_curves else math.pi
        gmsh.model.mesh.classifySurfaces(math.radians(feature_angle), True, True,
                                         curve_angle)

        surfaces = []
        for _, s in gmsh.model.getEntities(2):
            types, _, _ = gmsh.model.mesh.getElements(2, s)
            if len(types):
                surfaces.append(s)

        # Curves consisting only of interface vertices remain discrete
        # (without parameterization) and are therefore not remeshed.
        keep, remesh = [], []
        for _, cv in gmsh.model.getEntities(1):
            _, xyz, _ = gmsh.model.mesh.getNodes(1, cv, True, False)
            xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
            if len(xyz) == 0:
                continue
            if all(lookup.find(p) >= 0 for p in xyz.tolist()):
                keep.append(cv)
            else:
                remesh.append(cv)

        gmsh.model.mesh.createGeometry([(2, s) for s in surfaces]
                                       + [(1, c) for c in remesh])

        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", int(curvature_elements))
        gmsh.option.setNumber("Mesh.MeshSizeMax", float(size_cap))
        gmsh.option.setNumber("Mesh.Algorithm", int(algorithm))

        size_errors = {"n": 0, "first": None}

        def size_callback(dim, tag, x, y, z, lc):
            try:
                return min(lc, size_fn(x, y, z))
            except Exception:
                size_errors["n"] += 1
                if size_errors["first"] is None:
                    import traceback
                    size_errors["first"] = traceback.format_exc()
                return min(lc, size_cap)

        gmsh.model.mesh.setSizeCallback(size_callback)
        gmsh.model.mesh.generate(2)

        node_tags, xyz, _ = gmsh.model.mesh.getNodes(-1, -1, False, False)
        surf_tris = {}
        for s in surfaces:
            types, _, ntags = gmsh.model.mesh.getElements(2, s)
            for ty, nt in zip(types, ntags):
                if int(ty) != 2:
                    raise RemeshError("gmsh generated unexpected elements (type %d) "
                                      "erzeugt." % int(ty))
                surf_tris[s] = np.asarray(nt, dtype=np.int64).reshape(-1, 3)

        if size_errors["n"]:
            print("ERROR in size field (%d calls), first error:\n%s"
                  % (size_errors["n"], size_errors["first"]))
        info = {"surfaces": len(surfaces), "kept_curves": len(keep),
                "remeshed_curves": len(remesh), "size_errors": size_errors["n"],
                "version": getattr(gmsh, "__version__",
                                   getattr(gmsh, "GMSH_API_VERSION", "?"))}
        return (np.asarray(node_tags, dtype=np.int64),
                np.asarray(xyz, dtype=np.float64).reshape(-1, 3), surf_tris, info)
    except RemeshError:
        raise
    except Exception as e:
        tail = ""
        if logging:
            try:
                lines = [l for l in gmsh.logger.get() if l.strip()]
                tail = "\nLast gmsh messages:\n" + "\n".join(lines[-12:])
            except Exception:
                pass
        raise RemeshError("gmsh aborted during remeshing: %s%s\n\n"
                          "Tips: change ANGLE (e.g. 30 or 60), clean the mesh "
                          "beforehand using 'Merge by Distance'."
                          % (e, tail))
    finally:
        if logging:
            try:
                gmsh.logger.stop()
            except Exception:
                pass
        try:
            gmsh.finalize()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Post-processing: edge flips + smoothing (usable without Blender)
# -----------------------------------------------------------------------------

def _tri_normals(P, T):
    return np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-300)


def _vertex_normals(P, T):
    N = np.zeros_like(P)
    tn = _tri_normals(P, T)
    for k in range(3):
        np.add.at(N, T[:, k], tn)
    return _unit(N)


def _node_min(T, vals, n):
    out = np.full(n, np.inf)
    for k in range(3):
        np.minimum.at(out, T[:, k], vals)
    return out


def _sharp_edges(P, T, angle_deg):
    """Sharp edges (crease greater than angle_deg) as vertex pairs.

    The normal of a very narrow triangle is unreliable. A crease
    therefore counts only if both triangles are usable (angle >= 10°)
    or the vertex normals at the opposite corners confirm it."""
    n = len(P)
    if len(T) < 2:
        return np.zeros((0, 2), dtype=np.int64)
    ed = _Edges(T, n)
    e1, e2 = ed.e1, ed.e2
    t1, t2 = e1 // 3, e2 // 3
    cmax = math.cos(math.radians(angle_deg))
    U = _unit(_tri_normals(P, T))
    sharp = (U[t1] * U[t2]).sum(axis=1) < cmax
    amin = _min_angles_deg(P, T)
    good = (amin[t1] >= 10.0) & (amin[t2] >= 10.0)
    opp = T[:, [2, 0, 1]].ravel()
    VN = _vertex_normals(P, T)
    conf = (VN[opp[e1]] * VN[opp[e2]]).sum(axis=1) < cmax
    sharp &= good | conf
    return ed.E[e1[sharp]]


class _Segments:
    """Collection of line segments (A[i] -> B[i]); checks whether points lie on them.
    Tolerance: 1% of segment length (gmsh vertices lie exactly on them)."""

    def __init__(self, A, B, diag):
        self.A = np.asarray(A, dtype=np.float64).reshape(-1, 3)
        self.B = np.asarray(B, dtype=np.float64).reshape(-1, 3)
        self.count = len(self.A)
        if self.count:
            L = np.linalg.norm(self.B - self.A, axis=1)
            self.tol = np.maximum(0.01 * L, 1e-6 * max(float(diag), 1e-12))
            self.kd = _KD(0.5 * (self.A + self.B))
            self.r = 0.5 * float(L.max()) + float(self.tol.max())

    def locate(self, Q):
        """For each point: (lies on a segment, segment index, parameter t)."""
        Q = np.asarray(Q, dtype=np.float64).reshape(-1, 3)
        ok = np.zeros(len(Q), dtype=bool)
        seg = np.full(len(Q), -1, dtype=np.int64)
        par = np.zeros(len(Q))
        if not self.count:
            return ok, seg, par
        for i, q in enumerate(Q.tolist()):
            idx = np.asarray(self.kd.within(q, self.r), dtype=np.int64)
            if not len(idx):
                continue
            a, b = self.A[idx], self.B[idx]
            ab = b - a
            t = np.clip(((np.asarray(q) - a) * ab).sum(axis=1)
                        / np.maximum((ab * ab).sum(axis=1), 1e-300), 0.0, 1.0)
            d = np.linalg.norm(a + t[:, None] * ab - q, axis=1)
            good = d <= self.tol[idx]
            if good.any():
                j = int(np.argmin(np.where(good, d, np.inf)))
                ok[i], seg[i], par[i] = True, idx[j], t[j]
        return ok, seg, par

    def on_line(self, Q):
        return self.locate(Q)[0]


def _diag(P):
    P = np.asarray(P, dtype=np.float64)
    return float(np.linalg.norm(P.max(axis=0) - P.min(axis=0))) if len(P) else 1.0


def _feature_lines(P_ref, T_ref, angle_deg):
    """Sharp edges of the reference surface as line segments."""
    P_ref = np.asarray(P_ref, dtype=np.float64)
    E = _sharp_edges(P_ref, np.asarray(T_ref, dtype=np.int64), angle_deg)
    return _Segments(P_ref[E[:, 0]], P_ref[E[:, 1]], _diag(P_ref))


def _flip_round(P, T, max_dihedral_deg=20.0, eps_deg=0.1, feat=None,
                features=None):
    """One round of edge flips: swap an edge if the minimum angle of the
    two triangles increases (in the plane = Delaunay). Only nearly planar
    pairs, no boundary edges; vertex valence remains at least 3. Independent
    flips (with no shared triangles) are sorted by improvement
    and executed simultaneously. Returns: (T, number of flips)."""
    n, t = len(P), len(T)
    if t < 2:
        return T, 0
    ed = _Edges(T, n)
    nn = ed.n
    a, b = ed.E[:, 0], ed.E[:, 1]
    c = T[:, [2, 0, 1]].ravel()
    h1, h2 = ed.e1, ed.e2
    keep = (a[h1] == b[h2]) & (b[h1] == a[h2])  # opposite directions = consistent
    h1, h2 = h1[keep], h2[keep]
    if not len(h1):
        return T, 0
    u, v, cc, dd = a[h1], b[h1], c[h1], c[h2]
    t1, t2 = h1 // 3, h2 // 3

    code_cd = np.minimum(cc, dd) * nn + np.maximum(cc, dd)
    pos = np.minimum(np.searchsorted(ed.ucode, code_cd), len(ed.ucode) - 1)
    ok = (cc != dd) & (ed.ucode[pos] != code_cd)   # diagonal c-d does not exist yet

    deg = np.bincount(ed.unique.ravel(), minlength=n)
    ok &= (deg[u] > 3) & (deg[v] > 3)

    amin = _min_angles_deg(P, T)
    old = np.minimum(amin[t1], amin[t2])
    Tn1 = np.stack([u, dd, cc], axis=1)
    Tn2 = np.stack([dd, v, cc], axis=1)
    new = np.minimum(_min_angles_deg(P, Tn1), _min_angles_deg(P, Tn2))
    N = _tri_normals(P, T)
    n3, n4 = _unit(_tri_normals(P, Tn1)), _unit(_tri_normals(P, Tn2))
    cmax = math.cos(math.radians(max_dihedral_deg))
    # Area-weighted direction of the pair (narrow triangles contribute little)
    s = _unit(N[t1] + N[t2])
    # Geometry: distance between the two diagonals = shape change caused by the flip.
    # Swap only nearly planar quadrilaterals (sharp edges are preserved).
    d1 = P[v] - P[u]
    d2 = P[dd] - P[cc]
    cr = np.cross(d1, d2)
    crn = np.linalg.norm(cr, axis=1)
    L = np.maximum(np.linalg.norm(d1, axis=1), np.linalg.norm(d2, axis=1))
    dev = np.abs(((P[cc] - P[u]) * cr).sum(axis=1))
    ok &= (crn > 1e-12 * L * L) & (dev <= 0.1 * L * crn)
    # new triangles: not flipped over, tilted by at most max_dihedral_deg
    ok &= ((n3 * s).sum(axis=1) >= cmax) & ((n4 * s).sum(axis=1) >= cmax)
    ok &= (n3 * n4).sum(axis=1) >= cmax
    gain = new - old
    ok &= gain > eps_deg
    # Never flip away sharp edges or create a shortcut between two
    # edge vertices across the surface
    if feat is not None and features is not None and features.count:
        for e_a, e_b, want_on in ((u, v, False), (cc, dd, True)):
            m = np.nonzero(ok & feat[e_a] & feat[e_b])[0]
            if len(m):
                on = features.on_line(0.5 * (P[e_a[m]] + P[e_b[m]]))
                ok[m[on != want_on]] = False

    cand = np.nonzero(ok)[0]
    if not len(cand):
        return T, 0
    cand = cand[np.argsort(-gain[cand], kind="stable")]
    used = np.zeros(t, dtype=bool)
    degc = deg.copy()
    new_codes = set()
    sel = []
    for i in cand.tolist():
        if used[t1[i]] or used[t2[i]]:
            continue
        k = int(code_cd[i])
        if k in new_codes or degc[u[i]] <= 3 or degc[v[i]] <= 3:
            continue
        used[t1[i]] = used[t2[i]] = True
        new_codes.add(k)
        degc[u[i]] -= 1
        degc[v[i]] -= 1
        degc[cc[i]] += 1
        degc[dd[i]] += 1
        sel.append(i)
    sel = np.asarray(sel, dtype=np.int64)
    T = T.copy()
    T[t1[sel]] = Tn1[sel]
    T[t2[sel]] = Tn2[sel]
    return T, int(len(sel))


def _laplace_step(P, T, movable, h, factor, tangential=True):
    """Weighted Laplace step. Weight per edge = 1 / target size, thereby
    preserving the size gradation instead of making everything the same size."""
    n = len(P)
    E = _Edges(T, n).unique
    w = 2.0 / (h[E[:, 0]] + h[E[:, 1]])
    num = np.zeros_like(P)
    den = np.zeros(n)
    np.add.at(num, E[:, 0], w[:, None] * P[E[:, 1]])
    np.add.at(num, E[:, 1], w[:, None] * P[E[:, 0]])
    np.add.at(den, E[:, 0], w)
    np.add.at(den, E[:, 1], w)
    d = num / np.maximum(den, 1e-300)[:, None] - P
    if tangential:
        N = _vertex_normals(P, T)
        d -= (d * N).sum(axis=1)[:, None] * N
    out = P.copy()
    out[movable] += factor * d[movable]
    return out


def _guard(P_old, P_new, T, movable, check_angles=True, floor=35.0):
    """Revert movements that flip triangles or make a triangle
    worse than the worst triangle previously found at its vertices.
    This ensures that the minimum angle can never decrease."""
    n = len(P_old)
    P_new = P_new.copy()
    n_old = _tri_normals(P_old, T)
    # Triangles that were already reversed may change freely
    ref_n = _vertex_normals(P_old, T)[T].sum(axis=1)
    ok_old = (n_old * ref_n).sum(axis=1) > 0
    if check_angles:
        loc_old = _node_min(T, _min_angles_deg(P_old, T), n)
        # Good triangles may become slightly worse as long as they remain above
        # 'floor' bleiben (floor >= the previous minimum angle of the entire mesh)
        tri_ref = np.minimum(loc_old[T].min(axis=1), floor)
    reverted = np.zeros(n, dtype=bool)
    while True:
        bad_tri = ok_old & ((_tri_normals(P_new, T) * n_old).sum(axis=1) <= 0)
        if check_angles:
            bad_tri |= _min_angles_deg(P_new, T) < tri_ref - 1e-9
        bad = np.zeros(n, dtype=bool)
        bad[T[bad_tri].ravel()] = True
        bad &= movable & (P_new != P_old).any(axis=1)
        if not bad.any():
            break
        P_new[bad] = P_old[bad]
        reverted |= bad
    return P_new, int(reverted.sum())


def _closest_on_tris(p, a, b, c):
    """Closest point on triangles (a, b, c) to p (arrays (..., 3),
    broadcastable). Method from Ericson, 'Real-Time Collision Detection'."""
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    shape = d1.shape + (3,)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = 1.0 / (va + vb + vc)
        res = np.array(np.broadcast_to(
            a + ab * (vb * denom)[..., None] + ac * (vc * denom)[..., None], shape))
        cases = [
            ((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0),
             b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[..., None]),
            ((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + ac * (d2 / (d2 - d6))[..., None]),
            ((d6 >= 0) & (d5 <= d6), c),
            ((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + ab * (d1 / (d1 - d3))[..., None]),
            ((d3 >= 0) & (d4 <= d3), b),
            ((d1 <= 0) & (d2 <= 0), a),
        ]
        # Cases are processed in reverse order compared with Ericson: the last
        # listed matching case (the first tested by Ericson) takes precedence
        for mask, val in cases:
            res[mask] = np.broadcast_to(val, shape)[mask]
    return res


class _Projector:
    """Project points onto the reference surface: the closest point whose
    triangle faces the same direction. Use a BVH tree in Blender, otherwise numpy
    (search all triangles; only for tests outside Blender)."""

    def __init__(self, P0, T0):
        self.P0 = np.asarray(P0, dtype=np.float64)
        self.T0 = np.asarray(T0, dtype=np.int64)
        self.N0 = _unit(_tri_normals(self.P0, self.T0))
        try:
            from mathutils import Vector
            from mathutils.bvhtree import BVHTree
            self.tree = BVHTree.FromPolygons(self.P0.tolist(), self.T0.tolist(),
                                             all_triangles=True)
            self.Vector = Vector
        except Exception:
            self.tree = None
            self.A, self.B, self.C = (self.P0[self.T0[:, k]] for k in range(3))

    def project(self, Q, N, radius):
        """Returns: (projected points, ok flag per point). Points that cannot be projected
        remain where they are."""
        Q = np.asarray(Q, dtype=np.float64).reshape(-1, 3)
        N = np.asarray(N, dtype=np.float64).reshape(-1, 3)
        if self.tree is not None:
            return self._project_bvh(Q, N, np.asarray(radius, dtype=np.float64))
        return self._project_numpy(Q, N)

    def _project_bvh(self, Q, N, radius):
        out = Q.copy()
        ok = np.ones(len(Q), dtype=bool)
        V = self.Vector
        for i, (q, nv, r) in enumerate(zip(Q.tolist(), N.tolist(), radius.tolist())):
            nv = V(nv)
            loc, nor, _, _ = self.tree.find_nearest(q)
            if loc is not None and nor.dot(nv) > 0:
                out[i] = tuple(loc)
                continue
            # the nearest triangle faces the wrong way (e.g. a thin wall):
            # search for the nearest suitable one within radius r
            best = None
            for loc2, nor2, _, d2 in self.tree.find_nearest_range(q, r):
                if nor2.dot(nv) > 0 and (best is None or d2 < best[1]):
                    best = (loc2, d2)
            if best is None:
                ok[i] = False
            else:
                out[i] = tuple(best[0])
        return out, ok

    def _project_numpy(self, Q, N):
        out = Q.copy()
        ok = np.zeros(len(Q), dtype=bool)
        step = max(1, int(3e5 // max(len(self.T0), 1)))
        for s in range(0, len(Q), step):
            q = Q[s:s + step]
            cp = _closest_on_tris(q[:, None, :], self.A[None], self.B[None],
                                  self.C[None])
            d2 = ((cp - q[:, None, :]) ** 2).sum(-1)
            cons = (self.N0[None] * N[s:s + step, None, :]).sum(-1) > 0
            d2 = np.where(cons & np.isfinite(d2), d2, np.inf)
            j = d2.argmin(axis=1)
            r = np.arange(len(q))
            good = np.isfinite(d2[r, j])
            out[s:s + step][good] = cp[r, j][good]
            ok[s:s + step] = good
        return out, ok


def improve_mesh(P, T, fixed, P_ref, T_ref, h, iterations=10, feature_angle=40.0,
                 shape_steps=0, omega=0.5):
    """Improve the mesh: alternate edge flips and size-weighted,
    tangential smoothing; after each step, project back onto the reference surface.

    fixed: bool per vertex (interface); these remain unchanged. Vertices on
    sharp edges of the reference surface (> feature_angle) also remain
    fixed, and these edges are never flipped away.
    shape_steps > 0: first stabilize the shape with Taubin smoothing
    (slightly changes the geometry).
    """
    P = np.array(P, dtype=np.float64)
    T = np.array(T, dtype=np.int64)
    n = len(P)
    h = np.maximum(np.asarray(h, dtype=np.float64), 1e-300)
    fixed = np.asarray(fixed, dtype=bool)
    flip_angle = min(20.0, 0.5 * feature_angle)
    info = {"smooth_iterations": 0, "flips": 0, "reverted": 0,
            "feature_nodes": 0, "sharp_edges": 0, "shape_steps": int(shape_steps)}

    # Sharp edges from the reference surface (the original), not from the
    # current mesh: this prevents them from disappearing due to flips or smoothing
    features = _feature_lines(P_ref, T_ref, feature_angle)
    on_feat = features.on_line(P)
    movable = ~fixed & ~on_feat
    info["feature_nodes"] = int((on_feat & ~fixed).sum())
    info["sharp_edges"] = int(features.count)

    def flips():
        nonlocal T
        total = 0
        for _ in range(50):
            T, k = _flip_round(P, T, flip_angle, feat=on_feat, features=features)
            total += k
            if k == 0:
                break
        info["flips"] += total
        return total

    if iterations > 0 or shape_steps > 0:
        flips()

    if shape_steps > 0:
        for _ in range(int(shape_steps)):
            for lam in (0.5, -0.53):
                Pn = _laplace_step(P, T, movable, h, lam, tangential=False)
                P, _ = _guard(P, Pn, T, movable, check_angles=False)
        P_ref, T_ref = P.copy(), T.copy()

    proj = _Projector(P_ref, T_ref)
    hmin = float(h.min()) if n else 0.0
    idx = np.nonzero(movable)[0]
    # The minimum angle of the entire mesh must never decrease due to smoothing
    floor = max(35.0, float(_min_angles_deg(P, T).min())) if len(T) else 35.0
    for _ in range(int(iterations)):
        nf = flips()
        Pn = _laplace_step(P, T, movable, h, omega, tangential=True)
        if len(idx):
            N = _vertex_normals(P, T)
            q, ok = proj.project(Pn[idx], N[idx], 2.0 * h[idx])
            Pn[idx[ok]] = q[ok]
            Pn[idx[~ok]] = P[idx[~ok]]
        Pn, rv = _guard(P, Pn, T, movable, check_angles=True, floor=floor)
        move = float(np.linalg.norm(Pn - P, axis=1).max()) if n else 0.0
        P = Pn
        info["smooth_iterations"] += 1
        info["reverted"] += rv
        if nf == 0 and move < 1e-4 * hmin:
            break
    if iterations > 0 or shape_steps > 0:
        flips()
    return P, T, info


# -----------------------------------------------------------------------------
# Core: remesh the free region (usable without Blender)
# -----------------------------------------------------------------------------

def _check_result(T_out, n_out, ref, stage, expected=None):
    """Check manifoldness and boundaries.

    ref: {key of a fixed boundary edge: direction}; these edges must remain boundary edges
    edges. expected: if provided, the set of boundary edges must match exactly.
    Returns: (misoriented fixed edges, boundary-edge keys,
    boundary edges as vertex pairs)."""
    ed = _Edges(T_out, n_out)
    if ed.nonmanifold:
        raise RemeshError("The mesh has %d non-manifold edges after %s."
                          % (stage, ed.nonmanifold))
    bnd = ed.cnt == 1
    new_b = set(ed.code[bnd].tolist())
    if expected is not None:
        if new_b != expected:
            raise RemeshError("Internal error: the boundary changed during %s."
                              % stage)
    else:
        missing = set(ref.keys()) - new_b
        if missing:
            raise RemeshError(
                "The mesh boundary does not match the interface after %s "
                "(%d edges are missing)." % (stage, len(missing)))
    e_dir = dict(zip(ed.code[bnd].tolist(), ed.sign[bnd].tolist()))
    wrong = sum(1 for k, r in ref.items() if e_dir.get(k) != r)
    return wrong, new_b, ed.E[bnd]


def _boundary_components(bE, n):
    """Connected boundary segments: component number for each boundary edge."""
    parent = {}

    def find(a):
        root = a
        while parent.setdefault(root, root) != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    for a, b in bE.tolist():
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = [find(a) for a in bE[:, 0].tolist()]
    _, comp = np.unique(np.asarray(roots, dtype=np.int64), return_inverse=True)
    return comp.reshape(-1)


def _split_boundary(bE_all, n, fixed_nodes):
    """Split boundary segments into fixed and to-be-remeshed segments.

    fixed_nodes None: all boundary segments remain fixed. Otherwise, a segment remains fixed
    if at least one of its vertices is fixed (partially fixed ->
    entire segment fixed). Returns: (fixed per boundary edge, fixed per segment, number of
    partially fixed boundary segments)."""
    if not len(bE_all):
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=bool), 0
    comp = _boundary_components(bE_all, n)
    ncomp = int(comp.max()) + 1
    if fixed_nodes is None:
        comp_fixed = np.ones(ncomp, dtype=bool)
        return comp_fixed[comp], comp_fixed, 0
    fn = np.asarray(fixed_nodes, dtype=bool)
    all_c = np.ones(ncomp, dtype=bool)
    any_c = np.zeros(ncomp, dtype=bool)
    np.logical_and.at(all_c, comp, fn[bE_all[:, 0]] & fn[bE_all[:, 1]])
    np.logical_or.at(any_c, comp, fn[bE_all[:, 0]] | fn[bE_all[:, 1]])
    comp_fixed = any_c          # (= all_c | any_c): partially fixed -> entirely fixed
    return comp_fixed[comp], comp_fixed, int((any_c & ~all_c).sum())


def _size_field(P, E, bE, bverts, target_size, growth):
    """Build the size field. E: all half-edges, bE: fixed boundary edges.

    h_i for each fixed boundary vertex = mean length of its adjacent fixed boundary edges.
    Remeshed boundaries are not used as a source.
    Returns: (size_fn, upper limit for gmsh, hmax, H or None)."""
    # Typical edge length (the median is robust against individual long fan edges)
    h_typ = float(np.median(np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1)))
    hmax = float(target_size) if target_size else h_typ
    if hmax <= 0:
        raise RemeshError("TARGET_DISTANCE must be greater than 0.")
    if not len(bverts):
        return (lambda x, y, z: hmax), hmax, hmax, None
    n = len(P)
    bl = np.linalg.norm(P[bE[:, 0]] - P[bE[:, 1]], axis=1)
    hsum = np.zeros(n)
    hcnt = np.zeros(n)
    np.add.at(hsum, bE[:, 0], bl)
    np.add.at(hsum, bE[:, 1], bl)
    np.add.at(hcnt, bE[:, 0], 1)
    np.add.at(hcnt, bE[:, 1], 1)
    H = hsum[bverts] / hcnt[bverts]
    return GradedSize(P[bverts], H, growth, hmax), max(hmax, float(H.max())), hmax, H


def _gmsh_remesh(gmsh, P, T, bverts, size_fn, size_cap, feature_angle,
                 curvature_elements, verbose, algorithm, split_curves):
    """Run gmsh on normalized geometry and return the result
    convert back. Returns: (vertices, triangles, orig, info); for each
    new vertex, orig is the index of the corresponding original vertex or -1."""
    diag = _diag(P)
    lookup = _PointLookup(P[bverts], bverts, max(1e-9 * diag, 1e-12))

    # gmsh limits every element size to its "characteristic length".
    # Through the API, it is never calculated from the model size and remains 1.0.
    # Therefore, the geometry passed to gmsh is normalized to a diagonal of 1 (then
    # this limit and gmsh’s internal tolerances are consistent); the result is
    # converted back afterward. DO NOT REMOVE THIS, otherwise every size > 1
    # is clipped and the mesh becomes uniform.
    c0 = 0.5 * (P.min(axis=0) + P.max(axis=0))
    sc = 1.0 / max(diag, 1e-300)
    Pn = (P - c0) * sc
    lookup_n = _PointLookup(Pn[bverts], bverts, 1e-9)

    def size_n(x, y, z):
        return sc * size_fn(c0[0] + x / sc, c0[1] + y / sc, c0[2] + z / sc)

    t0 = time.time()
    node_tags, node_xyz, surf_tris, info = _run_gmsh(
        gmsh, Pn, T, lookup_n, size_n, size_cap * sc, feature_angle,
        curvature_elements, verbose, algorithm, split_curves)
    info["seconds"] = time.time() - t0
    node_xyz = np.asarray(node_xyz, dtype=np.float64) / sc + c0

    if not surf_tris or sum(len(v) for v in surf_tris.values()) == 0:
        raise RemeshError("gmsh hat keine Dreiecke erzeugt.")
    all_tris = np.concatenate(list(surf_tris.values()), axis=0)

    # Keep only used vertices; match interface vertices by their coordinates
    used = np.unique(all_tris)
    order = np.argsort(node_tags)
    st = node_tags[order]
    pos = np.searchsorted(st, used)
    if np.any(pos >= len(st)) or np.any(st[np.minimum(pos, len(st) - 1)] != used):
        raise RemeshError("gmsh-Ausgabe ist unvollständig (Knoten fehlen).")
    out_P = node_xyz[order[pos]].copy()
    orig = lookup.find_many(out_P)
    is_old = orig >= 0
    if len(np.unique(orig[is_old])) != int(is_old.sum()):
        raise RemeshError("Multiple new vertices lie on the same boundary vertex.")
    missing = len(bverts) - int(is_old.sum())
    if missing:
        raise RemeshError(
            "%d interface vertices are missing from the new mesh. The connection to "
            "the fixed faces would not be conforming, so the operation is aborted."
            % missing)
    out_P[is_old] = P[orig[is_old]]          # Originalkoordinaten, bitgenau
    return out_P, np.searchsorted(used, all_tris), orig, info


def _orient_like(P, T, out_P, out_T, orig, bE):
    """Orient the new mesh like the original: using the fixed boundary edges,
    and for components without a fixed boundary, using the original normals.
    Returns: (triangles, direction of fixed boundary edges, info)."""
    n_out = len(out_P)
    is_old = orig >= 0
    local_to_out = np.full(len(P), -1, dtype=np.int64)
    local_to_out[orig[is_old]] = np.nonzero(is_old)[0]
    ref = {}
    for a, b in local_to_out[bE].tolist():
        ref[min(a, b) * n_out + max(a, b)] = 1 if a < b else -1
    vote_base = _make_vote(P, T)
    out_T, info = _orient(out_T, n_out, ref, lambda tt: vote_base(out_P, tt))
    return out_T, ref, info


def _locate_new_boundary_nodes(P, bE_free, out_P, new_b):
    """New boundary vertices must lie on the old (remeshed) boundary edges
    Returns: (vertices, original vertices a and b, parameter t) for
    interpolating vertex groups."""
    if not len(new_b):
        return (np.zeros(0, dtype=np.int64),) * 3 + (np.zeros(0),)
    segs = _Segments(P[bE_free[:, 0]], P[bE_free[:, 1]], _diag(P))
    ok, sg, tt = segs.locate(out_P[new_b])
    if not ok.all():
        raise RemeshError(
            "%d new boundary vertices do not lie on the original open boundary."
            % int((~ok).sum()))
    return new_b, bE_free[sg, 0], bE_free[sg, 1], tt


def _size_diagnostics(P, bverts, H, hmax, growth, out_P, out_T):
    """Metrics for the size progression: expected transition length, maximum distance
    to the fixed boundary, and actual edge lengths near the boundary / far away."""
    if H is None or not len(bverts):
        return {}
    g = max(float(growth) - 1.0, 1e-9)
    h_med = float(np.median(H))
    kd = _KD(P[bverts])
    d_free = np.array([kd.nearest(p)[1] for p in P.tolist()])
    d_max = float(d_free.max()) if len(d_free) else 0.0
    E = _Edges(out_T, len(out_P)).unique
    L = np.linalg.norm(out_P[E[:, 0]] - out_P[E[:, 1]], axis=1)
    mid = 0.5 * (out_P[E[:, 0]] + out_P[E[:, 1]])
    d_mid = np.array([kd.nearest(p)[1] for p in mid.tolist()])
    near = d_mid <= 1.5 * h_med
    far = d_mid >= 0.7 * d_max
    return {"h_bnd_median": h_med,
            "transition_length": max(hmax - h_med, 0.0) / g,
            "dist_max": d_max,
            "edge_near": float(L[near].mean()) if near.any() else float("nan"),
            "edge_far": float(L[far].mean()) if far.any() else float("nan")}


def remesh_free_region(points, tris, target_size=None, growth=1.2,
                       feature_angle=40.0, curvature_elements=0, verbose=False,
                       gmsh=None, auto_install=True, smooth_iterations=10,
                       shape_smoothing=0, algorithm=6, use_gmsh=True,
                       fixed_nodes=None):
    """Remesh the triangle mesh (points, tris) of the free region.

    fixed_nodes: bool per vertex. None = preserve all boundary vertices.
        Otherwise, only boundary segments containing fixed vertices are preserved;
        the remaining open boundaries are remeshed (same shape,
        new vertices).
    use_gmsh=False: only perform post-processing (all boundary vertices remain).

    Returns a dict containing
      points  (k,3)  new vertices
      tris    (t,3)  new triangles, oriented like the original
      orig    (k,)   original vertex index or -1 for new vertices
      src     (k,)   source for vertex groups (original index or -1)
      bnd_interp     new boundary vertices on old boundary edges: (vertex, a, b, t)
      before / gmsh / after  quality metrics (gmsh: None if skipped)
      info    additional information
    """
    P = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
    T = np.ascontiguousarray(tris, dtype=np.int64).reshape(-1, 3)
    if len(T) == 0:
        raise RemeshError("The free region contains no faces.")
    n = len(P)

    # 1. Boundary: fixed boundary edges (interface, fixed boundaries) and free edges
    ed = _Edges(T, n)
    if ed.nonmanifold:
        raise RemeshError(
            "The free region has %d non-manifold edges (more than two "
            "faces on an edge). Please clean this up first, e.g. using "
            "Select -> Select All by Trait -> Non Manifold." % ed.nonmanifold)
    bE_all = ed.boundary
    e_fixed, comp_fixed, n_mixed = _split_boundary(
        bE_all, n, fixed_nodes if use_gmsh else None)
    bE, bE_free = bE_all[e_fixed], bE_all[~e_fixed]
    bverts = np.unique(bE)
    free_bverts = np.setdiff1d(np.unique(bE_free), bverts)

    # 2. Size field
    size_fn, size_cap, hmax, H = _size_field(P, ed.E, bE, bverts, target_size,
                                             growth)

    # 3. gmsh (or keep the mesh unchanged)
    if use_gmsh:
        gm = gmsh if gmsh is not None else import_gmsh(auto_install)
        out_P, out_T, orig, info = _gmsh_remesh(
            gm, P, T, bverts, size_fn, size_cap, feature_angle, curvature_elements,
            verbose, algorithm, split_curves=len(bE_free) > 0)
        src = orig.copy()
    else:
        out_P, out_T = P.copy(), T.copy()
        orig = np.full(n, -1, dtype=np.int64)
        orig[bverts] = bverts
        src = np.arange(n, dtype=np.int64)  # verschobene Knoten bleiben "dieselben"
        info = {"surfaces": 0, "kept_curves": 0, "remeshed_curves": 0,
                "version": "-", "seconds": 0.0, "gmsh_skipped": True}
    is_old = orig >= 0
    n_out = len(out_P)

    # 4. Orientation and checks
    out_T, ref, oinfo = _orient_like(P, T, out_P, out_T, orig, bE)
    info.update(oinfo)
    wrong, bset, bedges = _check_result(out_T, n_out, ref, "gmsh")
    info["wrong_boundary_orientation"] = wrong
    on_bnd = np.zeros(n_out, dtype=bool)
    on_bnd[bedges.ravel()] = True
    new_b = np.nonzero(on_bnd & ~is_old)[0]
    bnd_interp = _locate_new_boundary_nodes(P, bE_free, out_P, new_b)
    q_gmsh = quality(out_P, out_T) if use_gmsh else None

    # 5. Post-processing (all boundary vertices remain fixed)
    if smooth_iterations > 0 or shape_smoothing > 0:
        t0 = time.time()
        h_nodes = np.array([size_fn(*p) for p in out_P.tolist()], dtype=np.float64)
        out_P, out_T, sinfo = improve_mesh(
            out_P, out_T, is_old | on_bnd, P, T, h_nodes, smooth_iterations,
            feature_angle, shape_smoothing)
        sinfo["smooth_seconds"] = time.time() - t0
        info.update(sinfo)
        info["wrong_boundary_orientation"] = _check_result(
            out_T, n_out, ref, "smoothing", expected=bset)[0]
        if not np.array_equal(out_P[is_old], P[orig[is_old]]):
            raise RemeshError("Internal error: interface vertices were moved.")

    # 6. Diagnostics
    info.update(_size_diagnostics(P, bverts, H, hmax, growth, out_P, out_T))
    info.update({"hmax": hmax,
                 "h_boundary_min": float(H.min()) if H is not None else float("nan"),
                 "h_boundary_max": float(H.max()) if H is not None else float("nan"),
                 "boundary_nodes": int(len(bverts)),
                 "growth": float(growth),
                 "bnd_comps_fixed": int(comp_fixed.sum()),
                 "bnd_comps_free": int((~comp_fixed).sum()),
                 "bnd_comps_mixed": n_mixed,
                 "bnd_free_nodes_before": int(len(free_bverts)),
                 "bnd_free_nodes_after": int(len(new_b))})
    return {"points": out_P, "tris": out_T, "orig": orig, "src": src,
            "bnd_interp": bnd_interp,
            "before": quality(P, T), "gmsh": q_gmsh,
            "after": quality(out_P, out_T), "info": info}


def assemble_output(P, polys, fixed_face, mat, free_used, res):
    """Combine fixed faces and new triangles into a mesh.

    P: (n,3) original vertices, polys: vertex lists of all faces, fixed_face:
    bool per face, mat: material index per face, free_used: global
    Knotenindizes des freien Bereichs (res['orig'] zeigt darauf).
    Returns: verts, faces, mats, indices of fixed vertices, source for each
    vertex (global original index or -1) and boundary interpolation
    (new vertex, original a, original b, t).
    """
    P = np.asarray(P, dtype=np.float64)
    nv = len(P)
    fixed_polys = [p for p, f in zip(polys, fixed_face) if f]
    fixed_mats = [int(m) for m, f in zip(mat, fixed_face) if f]
    free_mats = [int(m) for m, f in zip(mat, fixed_face) if not f]
    free_mat = int(np.bincount(free_mats).argmax()) if free_mats else 0
    free_used = np.asarray(free_used, dtype=np.int64)

    keep = np.zeros(nv, dtype=bool)
    for p in fixed_polys:
        keep[p] = True
    orig = np.asarray(res["orig"], dtype=np.int64)
    is_old = orig >= 0
    glob = np.full(len(orig), -1, dtype=np.int64)
    glob[is_old] = free_used[orig[is_old]]
    keep[glob[is_old]] = True

    old_ids = np.nonzero(keep)[0]
    old_to_out = np.full(nv, -1, dtype=np.int64)
    old_to_out[old_ids] = np.arange(len(old_ids))
    res_to_out = np.empty(len(orig), dtype=np.int64)
    res_to_out[is_old] = old_to_out[glob[is_old]]
    res_to_out[~is_old] = len(old_ids) + np.arange(int((~is_old).sum()))

    verts = np.vstack([P[old_ids], np.asarray(res["points"])[~is_old]])
    faces = [old_to_out[p].tolist() for p in fixed_polys]
    faces += res_to_out[np.asarray(res["tris"])].tolist()
    mats = fixed_mats + [free_mat] * len(res["tris"])
    fixed_verts = np.unique(np.concatenate([old_to_out[p] for p in fixed_polys])) \
        if fixed_polys else np.zeros(0, dtype=np.int64)
    # Source of new vertices (for vertex groups); with SMOOTH_ONLY,
    # even moved vertices retain their source
    src = np.asarray(res.get("src", orig), dtype=np.int64)[~is_old]
    new_src = np.full(len(src), -1, dtype=np.int64)
    new_src[src >= 0] = free_used[src[src >= 0]]
    vert_src = np.concatenate([old_ids, new_src])
    bi = res.get("bnd_interp")
    if bi is not None and len(bi[0]):
        interp = (res_to_out[bi[0]], free_used[bi[1]], free_used[bi[2]],
                  np.asarray(bi[3], dtype=np.float64))
    else:
        interp = (np.zeros(0, dtype=np.int64),) * 3 + (np.zeros(0),)
    return verts, faces, mats, fixed_verts, vert_src, interp


def report_lines(res, new_name=None, fixed_faces=None, group_stats=None,
                 only_smooth_note=False, scale_note=False):
    """All lines for the results window.

    fixed_faces: (count, {group: count}) of preserved fixed faces,
    group_stats: {group: (before, after, newly assigned)},
    only_smooth_note / scale_note: notes from the Blender section."""
    i = res["info"]
    smoothed = i.get("smooth_iterations", 0) > 0 or i.get("shape_steps", 0) > 0
    stages = [("Original", res["before"])]
    if res.get("gmsh") is not None:
        stages.append(("gmsh", res["gmsh"]))
    if smoothed:
        stages.append(("smoothed", res["after"]))
    elif res.get("gmsh") is None:
        stages.append(("neu", res["after"]))

    def row(label, key, fmt, scale=1.0, unit=""):
        return label + ": " + " -> ".join(fmt % (scale * q[key]) for _, q in stages) + unit

    lines = []
    if fixed_faces is not None:
        total, counts = fixed_faces
        lines.append("Fixed faces preserved: %d (%s)" % (
            total, ", ".join("%s %d" % kv for kv in counts.items())))
        lines += ["Note: group '%s' does not produce any fixed faces (does it contain only "
                  "vertices or edges?)" % nm
                  for nm in reversed([nm for nm, c in counts.items() if c == 0])]
    if i.get("size_errors"):
        lines.append("WARNING: size field failed %d times; the mesh is therefore "
                     "uniform. See the system console for details." % i["size_errors"])
    lines += [
        "Values: " + " -> ".join(name for name, _ in stages),
        row("Triangles in free region", "n", "%d"),
        row("Minimum interior angle", "min_angle", "%.1f°"),
        row("Mean minimum angle", "mean_min_angle", "%.1f°"),
        row("Triangles with angle < 30°", "below30", "%.1f", 100.0, " %"),
        row("Size jump between neighboring triangles (95%)", "ratio95", "%.2f"),
        "Fixed boundary vertices (interface, fixed boundaries): %d unchanged"
        % i["boundary_nodes"],
    ]
    if i.get("bnd_comps_free"):
        lines.append("Open boundaries remeshed: %d (boundary vertices %d -> %d)"
                     % (i["bnd_comps_free"], i["bnd_free_nodes_before"],
                        i["bnd_free_nodes_after"]))
    if i.get("bnd_comps_mixed"):
        lines.append("Note: %d boundary segment(s) only partially fixed; the entire segment remains "
                     "unchanged" % i["bnd_comps_mixed"])
    if i["boundary_nodes"]:
        lines.append("Boundary edges %.4g ... %.4g, target size %.4g, growth %.2f"
                     % (i["h_boundary_min"], i["h_boundary_max"], i["hmax"],
                        i["growth"]))
        if "h_bnd_median" in i:
            m, Z = i["h_bnd_median"], i["hmax"]
            lines.append("Size progression: boundary (median) %.4g -> target %.4g, transition "
                         "approx. %.4g long; maximum distance to fixed boundary %.4g"
                         % (m, Z, i["transition_length"], i["dist_max"]))
            lines.append("Resulting edge length: near fixed boundary %.4g, far away %.4g"
                         % (i["edge_near"], i["edge_far"]))
            if Z < 1.5 * m:
                lines.append("NOTE: target size is barely larger than the boundary edges. "
                             "For visible growth, set TARGET_DISTANCE = %.3g ... %.3g"
                             % (3 * m, 4 * m))
            elif i["transition_length"] > i["dist_max"]:
                lines.append("NOTE: the transition is longer than the distance across the region; "
                             "the target size will not be reached (increase GROWTH "
                             "or reduce TARGET_DISTANCE)")
    else:
        lines.append("No fixed boundary: uniform size %.4g" % i["hmax"])
    if i.get("gmsh_skipped"):
        lines.append("gmsh skipped (SMOOTH_ONLY)")
    else:
        lines.append("gmsh %s: %d surfaces, %.1f s"
                     % (i.get("version", "?"), i["surfaces"], i["seconds"]))
    if smoothed:
        txt = "Smoothing: passes %d, edge flips %d, %.1f s" % (
            i["smooth_iterations"], i["flips"], i.get("smooth_seconds", 0.0))
        if i.get("shape_steps"):
            txt += ", Shape smoothing %d" % i["shape_steps"]
        lines.append(txt)
        if i.get("feature_nodes"):
            lines.append("Vertices held fixed on sharp edges: %d"
                         % i["feature_nodes"])
    if i.get("conflicts"):
        lines.append("Warning: orientation is ambiguous (%d edges)"
                     % i["conflicts"])
    if new_name:
        lines.append("New object: %s" % new_name)
    if only_smooth_note:
        lines.append("Note: with SMOOTH_ONLY, all open boundaries remain "
                     "unchanged")
    if group_stats:
        lines.append("Vertex groups preserved: " + ", ".join(group_stats))
        for name, (before, after, gained) in group_stats.items():
            if gained:
                lines.append("Group '%s': boundary remeshed; now %d vertices "
                             "(previously %d)" % (name, after, before))
            elif after < before:
                lines.append("Note: group '%s' loses %d vertices (they were in the "
                             "remeshed region)" % (name, before - after))
    if scale_note:
        lines.append("Note: object scale was not applied; lengths are local.")
    return lines


# -----------------------------------------------------------------------------
# Blender integration
# -----------------------------------------------------------------------------

def _as_list(x):
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return [str(v) for v in x]


def _group_masks(obj, names, what):
    """Boolean mask per group and vertex: vertices with weight > 0.5."""
    missing = [nm for nm in names if obj.vertex_groups.get(nm) is None]
    if missing:
        have = [g.name for g in obj.vertex_groups]
        raise RemeshError(
            "Vertex group(s) for %s not found: %s\nAvailable in object '%s': %s"
            % (what, ", ".join(missing), obj.name,
               ", ".join(have) if have else "(none)"))
    nv = len(obj.data.vertices)
    index = {obj.vertex_groups.get(nm).index: k for k, nm in enumerate(names)}
    masks = np.zeros((len(names), nv), dtype=bool)
    for v in obj.data.vertices:
        for g in v.groups:
            k = index.get(g.group)
            if k is not None and g.weight > 0.5:
                masks[k, v.index] = True
    return masks


def _read_mesh(obj, groups, mode="gruppe"):
    """Read vertices, faces, materials, and fixed faces.
    A face is fixed if all its vertices belong to ONE of the groups."""
    me = obj.data
    nv = len(me.vertices)
    co = np.empty(nv * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    P = co.reshape(-1, 3).astype(np.float64)
    npoly = len(me.polygons)
    ls = np.empty(npoly, dtype=np.int32)
    lt = np.empty(npoly, dtype=np.int32)
    mat = np.empty(npoly, dtype=np.int32)
    me.polygons.foreach_get("loop_start", ls)
    me.polygons.foreach_get("loop_total", lt)
    me.polygons.foreach_get("material_index", mat)
    lv = np.empty(len(me.loops), dtype=np.int32)
    me.loops.foreach_get("vertex_index", lv)
    polys = [lv[s:s + t].tolist() for s, t in zip(ls.tolist(), lt.tolist())]

    if str(mode).lower().startswith("aus"):
        sel = np.zeros(npoly, dtype=bool)
        me.polygons.foreach_get("select", sel)
        if not sel.any():
            raise RemeshError(
                "No faces selected. Please select the fixed faces in Edit Mode "
                "(FIXED_SURFACE = 'auswahl').")
        return P, polys, mat, sel, {"Selection": int(sel.sum())}

    names = _as_list(groups)
    if not names:
        raise RemeshError("VERTEX_GROUP is empty. Please specify at least one group "
                          "angeben, z. B. VERTEX_GROUP = [\"MV\", \"AV\"].")
    masks = _group_masks(obj, names, "the fixed faces (VERTEX_GROUP)")
    fixed_f = np.zeros(npoly, dtype=bool)
    counts = {}
    if npoly:
        order = np.argsort(ls, kind="stable")
        for k, nm in enumerate(names):
            m = np.zeros(npoly, dtype=bool)
            m[order] = np.logical_and.reduceat(masks[k][lv], ls[order])
            counts[nm] = int(m.sum())
            fixed_f |= m
    return P, polys, mat, fixed_f, counts


def _free_triangles(me, fixed_f):
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.index_update()
    bm.faces.index_update()
    lay = bm.faces.layers.int.new("_remesh_frei")
    for f in bm.faces:
        f[lay] = 0 if fixed_f[f.index] else 1
    big = [f for f in bm.faces if f[lay] == 1 and len(f.verts) > 3]
    if big:
        bmesh.ops.triangulate(bm, faces=big, quad_method='BEAUTY',
                              ngon_method='BEAUTY')
    tris = [[v.index for v in f.verts] for f in bm.faces if f[lay] == 1]
    bm.free()
    return np.array(tris, dtype=np.int64).reshape(-1, 3)


def _copy_vertex_groups(src, obj, vert_src, interp, extra_group=None,
                        extra_verts=()):
    """Copy all vertex groups and their weights to the preserved vertices.
    New vertices on remeshed boundaries receive a group if both
    endpoints of the original boundary edge belong to it (weight is linearly interpolated).
    Returns: {group: (before, after, newly assigned)}."""
    names = [g.name for g in src.vertex_groups]
    weights = [dict() for _ in names]
    for v in src.data.vertices:
        for g in v.groups:
            if 0 <= g.group < len(names) and g.weight > 0.0:
                weights[g.group][v.index] = g.weight
    new_of = {int(s): i for i, s in enumerate(np.asarray(vert_src).tolist()) if s >= 0}
    i_new, i_a, i_b, i_t = (np.asarray(x).tolist() for x in interp)
    stats = {}
    for gi, name in enumerate(names):
        W = weights[gi]
        vg = obj.vertex_groups.new(name=name)
        by_weight = {}
        kept = 0
        for vi, w in W.items():
            ni = new_of.get(vi)
            if ni is not None:
                by_weight.setdefault(w, []).append(ni)
                kept += 1
        gained = 0
        for ni, a, b, t in zip(i_new, i_a, i_b, i_t):
            wa, wb = W.get(a), W.get(b)
            if wa is not None and wb is not None:
                w = round((1.0 - t) * wa + t * wb, 6)
                by_weight.setdefault(w, []).append(int(ni))
                gained += 1
        for w, ids in by_weight.items():
            vg.add(ids, float(w), 'REPLACE')
        stats[name] = (len(W), kept + gained, gained)
    if extra_group and extra_group not in names:
        vg = obj.vertex_groups.new(name=extra_group)
        if len(extra_verts):
            vg.add([int(i) for i in extra_verts], 1.0, 'REPLACE')
    return stats


def _build_object(src, name, verts, faces, mats, n_fixed_faces, vert_src, interp,
                  extra_group=None, extra_verts=()):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts.tolist(), [], faces)
    for m in src.data.materials:
        me.materials.append(m)
    if len(me.materials):
        me.polygons.foreach_set("material_index", mats)
    # Keep fixed faces selected so a second run can be performed immediately
    sel = [True] * n_fixed_faces + [False] * (len(faces) - n_fixed_faces)
    me.polygons.foreach_set("select", sel)
    me.update()
    obj = bpy.data.objects.new(name, me)
    colls = list(src.users_collection) or [bpy.context.scene.collection]
    for coll in colls:
        coll.objects.link(obj)
    obj.matrix_world = src.matrix_world.copy()
    stats = _copy_vertex_groups(src, obj, vert_src, interp, extra_group, extra_verts)
    return obj, stats


def _boundary_setting(fixed_boundaries):
    """Evaluate FIXED_BOUNDARIES: (keep all open boundaries fixed?, group names)."""
    fb = fixed_boundaries
    if isinstance(fb, str) and fb.strip().lower() == "all":
        return True, []
    if isinstance(fb, str) and fb.strip().lower() == "none":
        return False, []
    return False, _as_list(fb)


def remesh_object(obj, group_name=VERTEX_GROUP, target_size=TARGET_DISTANCE,
                  growth=GROWTH, feature_angle=ANGLE,
                  curvature_elements=CURVATURE, verbose=VERBOSE,
                  mode=FIXED_SURFACE, smooth_iterations=SMOOTH,
                  shape_smoothing=FORM_SMOOTH, only_smooth=SMOOTH_ONLY,
                  algorithm=ALGORITHM, fixed_boundaries=FIXED_BOUNDARIES,
                  name_suffix=NAME_SUFFIX):
    """Remesh the active object. Returns: (new object, report lines)."""
    print("This is the translated version")
    if obj is None or obj.type != 'MESH':
        raise RemeshError("Please select a mesh object (active object).")
    if obj.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    if only_smooth and smooth_iterations <= 0 and shape_smoothing <= 0:
        raise RemeshError("SMOOTH_ONLY is enabled, but SMOOTH and FORM_SMOOTH "
                          "are 0. Nothing to do.")

    # Check fixed-boundary groups before loading gmsh
    keep_all_bnd, bnd_names = _boundary_setting(fixed_boundaries)
    bnd_masks = _group_masks(obj, bnd_names, "the fixed boundaries (FIXED_BOUNDARIES)") \
        if bnd_names else None

    P, polys, mat, fixed_f, counts = _read_mesh(obj, group_name, mode)
    if fixed_f.all():
        raise RemeshError("All faces are fixed; there is nothing to remesh.")

    # Load gmsh before starting the actual work so errors are caught early
    gmsh = None if only_smooth else import_gmsh(INSTALL_GMSH)

    if not fixed_f.any():
        print("Note: no fixed faces found; the entire mesh will be "
              "remeshed.")
    free_T_global = _free_triangles(obj.data, fixed_f)
    free_used = np.unique(free_T_global)
    free_T = np.searchsorted(free_used, free_T_global)

    # Fixed vertices for open boundaries: vertices of fixed faces + boundary groups
    if keep_all_bnd or only_smooth:
        fixed_nodes = None
    else:
        keep = np.zeros(len(P), dtype=bool)
        for p, f in zip(polys, fixed_f):
            if f:
                keep[p] = True
        if bnd_masks is not None:
            keep |= bnd_masks.any(axis=0)
        fixed_nodes = keep[free_used]

    print("\nRemesh '%s': %d feste Flächen, %d freie Dreiecke"
          % (obj.name, int(fixed_f.sum()), len(free_T)))
    res = remesh_free_region(P[free_used], free_T, target_size, growth,
                             feature_angle, curvature_elements, verbose, gmsh=gmsh,
                             smooth_iterations=smooth_iterations,
                             shape_smoothing=shape_smoothing, algorithm=algorithm,
                             use_gmsh=not only_smooth, fixed_nodes=fixed_nodes)

    verts, faces, mats, fixed_verts, vert_src, interp = assemble_output(
        P, polys, fixed_f, mat, free_used, res)
    suffix = name_suffix or "_remesh"
    if only_smooth and suffix == "_remesh":
        suffix = "_glatt"
    selection_mode = str(mode).lower().startswith("aus")
    new, stats = _build_object(obj, obj.name + suffix, verts, faces, mats,
                               int(fixed_f.sum()), vert_src, interp,
                               extra_group="fest" if selection_mode else None,
                               extra_verts=fixed_verts)

    obj.hide_set(True)
    for o in bpy.context.selected_objects:
        o.select_set(False)
    new.select_set(True)
    bpy.context.view_layer.objects.active = new

    lines = report_lines(res, new.name, fixed_faces=(int(fixed_f.sum()), counts),
                         group_stats=stats,
                         only_smooth_note=only_smooth and not keep_all_bnd,
                         scale_note=any(abs(s - 1.0) > 1e-6 for s in obj.scale))
    return new, lines

def _popup(lines, title, icon='INFO'):
    def draw(self, context):
        for line in lines:
            self.layout.label(text=line)
    try:
        bpy.context.window_manager.popup_menu(draw, title=title, icon=icon)
    except Exception:
        pass


def main():
    try:
        new, lines = remesh_object(bpy.context.active_object)
        print("\n".join(lines))
        _popup(lines, "Remesh fertig")
    except RemeshError as e:
        msg = str(e)
        print("\nREMESH ABGEBROCHEN:\n" + msg)
        _popup(msg.splitlines()[:15], "Remesh aborted", 'ERROR')
    except Exception as e:  # unexpected error: see the console for details
        import traceback
        traceback.print_exc()
        _popup(["Unexpected error: %s" % e,
                "See the system console for details."],
               "Remesh aborted", 'ERROR')


if __name__ == "__main__" and bpy is not None:
    main()
