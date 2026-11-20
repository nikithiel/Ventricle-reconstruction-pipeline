"""Repo checks, run before every commit by .githooks/pre-commit.

    python tools/check.py

Enable the hook once per clone:  git config core.hooksPath .githooks
(skip it for one commit with  git commit --no-verify).

1. every addons/*.py and tools/*.py parses with the Python 3.10 grammar (Blender 3.1);
2. T1/T2: tools/test_valve_area_check.py (standard library);
3. T3: tools/test_calculate_valve_diameters.py, run through uv (skipped without uv);
4. T4: tools/test_load_settings.py (standard library);
5. no undefined names (pyflakes through uv, skipped without uv);
6. the staged changes add no private paths (network shares, Windows user folders, sandbox
   paths) and no references to the untracked folders .private/ and reports/.
   Text files: added lines only. .blend files: strings that the HEAD version does not have.
"""
import ast
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
T3_PACKAGES = ["numpy", "pandas", "scipy", "matplotlib", "trimesh"]

HOST = r"[A-Za-z0-9._$-]{2,}"
# \\server\share as written (2 + 1 backslashes) or escaped in JSON/Python (4 + 2); "dir\\sub\\" is neither
SHARE = rf"(?<!\\)(?:\\{{2}}{HOST}\\(?!\\)|\\{{4}}{HOST}\\{{2}}(?!\\))"
PRIVATE = [
    ("network share", SHARE),
    ("Windows user folder", r"[A-Za-z]:[\\/]+Users[\\/]"),
    ("sandbox path", r"/(?:c/Users|home/agent|tmp/claude-)"),
    ("untracked folder", r"(?<![\w/])\.private/|(?<![\w.-])reports/"),
]
PRIVATE_TEXT = [(name, re.compile(rx)) for name, rx in PRIVATE]
PRIVATE_BYTES = [(name, re.compile(rx.encode())) for name, rx in PRIVATE]
SCAN_SKIP = {"tools/check.py", ".gitignore"}  # they name the patterns on purpose


def git(*args, text=True):
    kw = {"encoding": "utf-8", "errors": "replace"} if text else {}
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, **kw).stdout


def uv_python(packages, args):
    return ["uv", "run", "--no-project", "--quiet", *[x for p in packages for x in ("--with", p)], "python", *args]


def py_files():
    return sorted(p.relative_to(ROOT).as_posix() for p in [*ROOT.glob("addons/*.py"), *ROOT.glob("tools/*.py")])


def check_syntax():
    bad = []
    for f in py_files():
        try:
            ast.parse((ROOT / f).read_bytes(), f, feature_version=(3, 10))
        except SyntaxError as e:
            bad.append(f"{f}:{e.lineno}: {e.msg}")
    return bad


def run_test(cmd):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace")
    return [] if r.returncode == 0 else (r.stdout + r.stderr).strip().splitlines()[-15:]


def check_undefined_names():
    r = subprocess.run(uv_python(["pyflakes"], ["-m", "pyflakes", *py_files()]), cwd=ROOT,
                       capture_output=True, encoding="utf-8", errors="replace")
    msgs = [m.groups() for m in map(re.compile(r"^(.+?):(\d+):\d+:? (.*)$").match, r.stdout.splitlines()) if m]
    if not msgs and r.returncode != 0:
        return (r.stdout + r.stderr).strip().splitlines()[-5:]
    # Blender property annotations (name: StringProperty(name="Some text")) look like broken
    # forward annotations to pyflakes; it then reports words of that text as undefined names.
    annotation_lines = {(f, n) for f, n, t in msgs if "forward annotation" in t}
    return [f"{f}:{n}: {t}" for f, n, t in msgs if t.startswith("undefined name") and (f, n) not in annotation_lines]


def check_staged():
    bad, path, line = [], None, 0
    for d in git("diff", "--cached", "-U0", "--no-color", "--no-ext-diff", "--diff-filter=d").splitlines():
        if d.startswith("+++ "):
            path = d[6:] if d.startswith("+++ b/") else None
        elif d.startswith("@@"):
            line = int(re.search(r"\+(\d+)", d).group(1))
        elif d.startswith("+") and path:
            if path not in SCAN_SKIP:
                bad += [f"{path}:{line}: {name}: {d[1:].strip()[:120]}" for name, rx in PRIVATE_TEXT if rx.search(d)]
            line += 1
    for path in git("diff", "--cached", "--name-only", "--diff-filter=d", "--", "*.blend").splitlines():
        found = lambda blob: {(name, m.group()) for name, rx in PRIVATE_BYTES
                              for m in rx.finditer(git("show", blob, text=False))}
        known = found(f"HEAD:{path}") if git("cat-file", "-t", f"HEAD:{path}").strip() else set()
        bad += [f"{path}: {name}: {s.decode(errors='replace')}…" for name, s in sorted(found(f":{path}") - known)]
    return bad


def main():
    has_uv = shutil.which("uv") is not None
    checks = [
        ("syntax (Python 3.10 grammar)", check_syntax),
        ("T1/T2 tools/test_valve_area_check.py", lambda: run_test([sys.executable, "tools/test_valve_area_check.py"])),
        ("T3 tools/test_calculate_valve_diameters.py",
         lambda: run_test(uv_python(T3_PACKAGES, ["tools/test_calculate_valve_diameters.py"])) if has_uv else None),
        ("T4 tools/test_load_settings.py", lambda: run_test([sys.executable, "tools/test_load_settings.py"])),
        ("undefined names (pyflakes)", lambda: check_undefined_names() if has_uv else None),
        ("staged changes: no private paths", check_staged),
    ]
    failed = 0
    for name, check in checks:
        problems = check()
        if problems is None:
            print(f"SKIP {name} (uv not found)")
        elif problems:
            failed += 1
            print(f"FAIL {name}\n" + "\n".join(f"     {p}" for p in problems))
        else:
            print(f"OK   {name}")
        sys.stdout.flush()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
