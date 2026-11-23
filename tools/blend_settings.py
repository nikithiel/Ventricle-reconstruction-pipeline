#!/usr/bin/env python3
"""Show the scene settings and objects stored in GVR-Pipeline.blend, without Blender (read-only).

    python tools/blend_settings.py                 # settings and objects of the working copy
    python tools/blend_settings.py --against HEAD  # only what differs from the committed version
    python tools/blend_settings.py --self-test     # run by tools/check.py

Use it after the .blend was saved in Blender and before it is committed. It warns about folder
settings that are not relative to the .blend ('//...') and about values with spaces at either end.
Standard library only.
"""
import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "addons"))
from valve_area_diagnosis import BlendFile  # noqa: E402
from load_settings import FOLDERS  # noqa: E402

BLEND = "GVR-Pipeline.blend"
ARRAY_FORMATS = {1: "i", 2: "f", 8: "d"}  # IDProperty array subtypes


def scene_settings(blend):
    """{name: value} of the custom properties of every scene (the add-on's settings)."""
    o = lambda f: blend.off("IDProperty", f)  # noqa: E731
    od = lambda f: blend.off("IDPropertyData", f)  # noqa: E731
    out = {}
    for code, _size, _old, _sdna, _count, sc in blend.blocks:
        if code != b"SC\x00\x00":
            continue
        group = blend.by_old.get(blend._ptr(sc + blend.off("Scene", "id") + blend.off("ID", "properties")))
        ptr = blend._ptr(group[5] + o("data") + od("group")) if group else 0  # first child
        while ptr:
            p = blend.by_old[ptr][5]
            name, kind, data = blend._cstr(p + o("name"), 64), blend.data[p + o("type")], p + o("data")
            target = blend.by_old.get(blend._ptr(data + od("pointer")))
            if kind == 0:  # string
                value = blend._cstr(target[5], target[1]) if target else ""
            elif kind == 1:  # int, bool, enum
                value = blend._i32(data + od("val"))
            elif kind in (2, 8):  # float, double
                value = round(struct.unpack_from("<f" if kind == 2 else "<d", blend.data, data + od("val"))[0], 5)
            elif kind == 5 and target and blend.data[p + o("subtype")] in ARRAY_FORMATS:
                fmt = ARRAY_FORMATS[blend.data[p + o("subtype")]]
                n = blend._i32(p + o("len"))
                value = tuple(round(x, 4) for x in struct.unpack_from(f"<{n}{fmt}", blend.data, target[5]))
            else:
                value = f"<IDProperty type {kind}>"
            out.setdefault(name, value)
            ptr = blend._ptr(p + o("next"))
    return out


def warnings(settings):
    out = []
    for name, value in settings.items():
        if isinstance(value, str) and value != value.strip():
            out.append(f"{name}: space at the start or end: {value!r}")
    for name, _key in FOLDERS:
        value = settings.get(name)
        if isinstance(value, str) and value.strip() and not value.startswith("//"):
            out.append(f"{name}: not relative to the .blend (should start with '//'): {value!r}")
    return out


def committed(rev):
    """The .blend at a git revision, read from a temporary copy."""
    data = subprocess.run(["git", "show", f"{rev}:{BLEND}"], cwd=REPO, capture_output=True, check=True).stdout
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / BLEND
        path.write_bytes(data)
        return BlendFile(path)


def differences(old, new):
    a, b = scene_settings(old), scene_settings(new)
    lines = [f"{k}: {a.get(k)!r} -> {b.get(k)!r}" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    for label, names in (("objects only in the new file", set(new.objects) - set(old.objects)),
                         ("objects only in the old file", set(old.objects) - set(new.objects))):
        if names:
            lines.append(f"{label} ({len(names)}): {', '.join(sorted(names))}")
    return lines


def self_test():
    blend = BlendFile(REPO / BLEND)
    s = scene_settings(blend)
    problems = [f"{name} missing or not a folder string" for name, _ in FOLDERS if not isinstance(s.get(name), str)]
    if not isinstance(s.get("approach"), int):
        problems.append("approach missing or not an int")
    if not (isinstance(s.get("translation_mitral"), tuple) and len(s["translation_mitral"]) == 3):
        problems.append("translation_mitral missing or not a 3-vector")
    if not any(n.startswith("A4_MV_") for n in blend.objects):
        problems.append("no A4_MV_* valve template objects")
    if differences(blend, blend):
        problems.append("a file differs from itself")
    print("\n".join(problems) or f"ok: {len(s)} settings, {len(blend.objects)} objects")
    return 1 if problems else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--against", metavar="REV", help="show only the differences to the .blend at this git revision")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    blend = BlendFile(REPO / BLEND)
    if args.against:
        print("\n".join(differences(committed(args.against), blend)) or f"no differences to {args.against}")
    else:
        for name, value in sorted(scene_settings(blend).items()):
            print(f"{name} = {value!r}")
        print(f"objects ({len(blend.objects)}): {', '.join(sorted(blend.objects))}")
    for w in warnings(scene_settings(blend)):
        print(f"WARNING {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
