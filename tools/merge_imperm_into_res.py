"""Merge the imperm zone templates of GVR-Pipeline.blend into the res templates (one-off).

Decision: .private/docs/adr/0001-imperm-geht-in-res-auf.md. imperm is the outer band of a valve disc
and is now part of res, so a built valve has the zones perm and res only.

Run once in Blender 3.1: Scripting workspace > Text > Open this file > Run Script. The report goes to
the System Console (Window > Toggle System Console). Nothing is changed unless every template pair
passes its checks. Save the file afterwards (Ctrl+S). Running it a second time does nothing.
Guide: reports/anleitung_imperm_in_res_mergen.md
"""
import bpy
import bmesh

APPROACHES = (3, 4, 5)
VALVES = ("MV", "AV")
WELD_DIST = 1e-6  # imperm and res share their interface vertices exactly (checked offline)


def key(co):
    return (round(co.x, 5), round(co.y, 5), round(co.z, 5))


def is_wall(face):
    """Side face of a prism zone (normal in the valve plane), not a top or bottom face."""
    return abs(face.normal.z) < 0.5


def bottom_area(bm):
    """Top-view area of a zone, same rule as valve_geometry_check.bottom_faces_area."""
    z = [v.co.z for v in bm.verts]
    lo, hi = min(z), max(z)
    if hi - lo < 1e-3:  # flat zone (A3)
        return sum(f.calc_area() for f in bm.faces)
    limit = lo + 0.25 * (hi - lo)
    return sum(f.calc_area() for f in bm.faces if all(v.co.z <= limit for v in f.verts))


def unit_transform(obj):
    m = obj.matrix_world
    return all(abs(m[i][j] - (1.0 if i == j else 0.0)) < 1e-9 for i in range(4) for j in range(4))


def merged_mesh(res, imperm):
    """(bmesh of res with imperm joined in, info dict, None) or (None, None, reason)."""
    bm, bm_imp = bmesh.new(), bmesh.new()
    bm.from_mesh(res.data)
    bm_imp.from_mesh(imperm.data)
    bm.normal_update()
    bm_imp.normal_update()
    info = {"res": bottom_area(bm), "imperm": bottom_area(bm_imp)}
    res_keys = {key(v.co) for v in bm.verts}
    imp_keys = {key(v.co) for v in bm_imp.verts}

    # The A4/A5 zones are closed prisms: where imperm and res meet, both carry a side wall. Inside the
    # merged zone those walls would be interior faces, so they are left out on both sides.
    walls_res = [f for f in bm.faces if is_wall(f) and all(key(v.co) in imp_keys for v in f.verts)]
    walls_imp = {f for f in bm_imp.faces if is_wall(f) and all(key(v.co) in res_keys for v in f.verts)}
    if len(walls_res) != len(walls_imp):
        bm.free(); bm_imp.free()
        return None, None, f"{len(walls_res)} interface walls in res but {len(walls_imp)} in imperm"
    bmesh.ops.delete(bm, geom=walls_res, context='FACES')

    copied = {}
    for f in bm_imp.faces:
        if f in walls_imp:
            continue
        for v in f.verts:
            if v not in copied:
                copied[v] = bm.verts.new(v.co)
        bm.faces.new([copied[v] for v in f.verts])
    expected_welds = sum(1 for v in copied if key(v.co) in res_keys)
    bm_imp.free()

    n = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=WELD_DIST)
    bm.normal_update()
    info.update(walls=len(walls_res), welded=n - len(bm.verts), merged=bottom_area(bm))

    problems = []
    if expected_welds == 0:
        problems.append("imperm and res share no vertices")
    if info["welded"] != expected_welds:
        problems.append(f"{info['welded']} vertices joined, expected {expected_welds}")
    if any(len(e.link_faces) > 2 for e in bm.edges):
        problems.append("edges with more than two faces (interior faces left)")
    if any(not e.link_faces for e in bm.edges) or any(not v.link_faces for v in bm.verts):
        problems.append("loose edges or vertices")
    if any(len(e.link_faces) == 2 and not e.is_contiguous for e in bm.edges):
        problems.append("imperm and res faces point in opposite directions")
    if abs(info["merged"] - info["res"] - info["imperm"]) > 1e-6 * info["merged"]:
        problems.append(f"area {info['merged']:.6f} != res + imperm {info['res'] + info['imperm']:.6f}")
    if problems:
        bm.free()
        return None, None, "; ".join(problems)
    return bm, info, None


def main():
    print("\n=== Merge imperm into res ===")
    todo, problems = [], []
    for a in APPROACHES:
        for v in VALVES:
            name = f"A{a}_{v}"
            imp = bpy.data.objects.get(f"{name}_imperm")
            res = bpy.data.objects.get(f"{name}_res")
            if imp is None:
                print(f"{name}: no imperm template, skipped")
                continue
            if res is None or res.type != 'MESH' or imp.type != 'MESH':
                problems.append(f"{name}: res or imperm template missing or not a mesh")
                continue
            if not (unit_transform(res) and unit_transform(imp)):
                problems.append(f"{name}: res or imperm template is moved, rotated or scaled")
                continue
            bm, info, issue = merged_mesh(res, imp)
            if issue:
                problems.append(f"{name}: {issue}")
            else:
                todo.append((name, res, imp, bm, info))

    if problems:
        for _, _, _, bm, _ in todo:
            bm.free()
        for p in problems:
            print("FAIL  " + p)
        raise RuntimeError("Nothing was changed. See the FAIL lines in the System Console.")

    for name, res, imp, bm, info in todo:
        bm.to_mesh(res.data)
        res.data.update()
        bm.free()
        mesh = imp.data
        bpy.data.objects.remove(imp, do_unlink=True)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
        print(f"OK    {name}: res {info['res']:.4f} + imperm {info['imperm']:.4f} = {info['merged']:.4f}, "
              f"{info['welded']} vertices joined, {info['walls']} wall faces removed per side")
    print(f"Done: {len(todo)} template(s) merged. Check them, then save the file (Ctrl+S).")


main()
