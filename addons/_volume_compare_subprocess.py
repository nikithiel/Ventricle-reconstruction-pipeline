"""Detached renderer for the volume-comparison plot so Blender's main thread never
runs the Qt event loop (which froze the UI until the window was closed).

Imports ONLY stl_plot (no bpy), so it can run as a standalone process launched with
Blender's bundled python.exe. MUST be deployed next to stl_plot.py.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # ensure `import stl_plot`


def _find(module):
    import importlib.util
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False


def _safe_rmtree(path, temp_root):
    """Delete `path` only if it is a real directory strictly under `temp_root`."""
    import shutil
    if not path or not temp_root:
        return
    try:
        p = os.path.normcase(os.path.abspath(path))
        root = os.path.normcase(os.path.abspath(temp_root))
        if os.path.isdir(p) and p != root and os.path.commonpath([p, root]) == root:
            shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass  # different drive / bad path -> never delete


def _png_fallback(fig):
    import tempfile
    try:
        png = os.path.join(tempfile.gettempdir(), "volume_curve_comparison.png")
        fig.savefig(png)
        os.startfile(png)  # Windows default viewer, non-blocking
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-path", required=True)
    ap.add_argument("--plot-input-dir", required=True)
    ap.add_argument("--base-dir", default="")          # rotated_dir; "" -> no raw curve
    ap.add_argument("--interp-method", type=int, default=None)
    ap.add_argument("--save-csv", action="store_true")
    ap.add_argument("--save-png", default="")
    ap.add_argument("--delete-dir", default="")
    ap.add_argument("--temp-root", default="")
    a = ap.parse_args()

    import matplotlib
    import stl_plot  # NOTE: pins matplotlib.use('Qt5Agg') at import (lazy: no crash w/o Qt)

    have_qt = any(_find(m) for m in ("PyQt5", "PySide2", "PySide6", "PyQt6"))
    if have_qt:
        try:
            matplotlib.use("QtAgg", force=True)  # override the Qt5Agg pin; picks Qt5 or Qt6
        except Exception:
            have_qt = False
    if not have_qt:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    _ed, _es, _diag, fig = stl_plot.derive_ed_es_from_volume_curve(
        a.input_path, a.plot_input_dir, a.base_dir, a.interp_method, "volume", a.save_csv)

    if a.save_png:
        try:
            fig.savefig(a.save_png)
        except Exception:
            pass

    try:
        if have_qt:
            try:
                fig.canvas.manager.set_window_title("Volume curve comparison")
            except Exception:
                pass
            plt.show(block=True)  # blocks until the user closes the window
        else:
            _png_fallback(fig)
    except Exception:
        _png_fallback(fig)
    finally:
        plt.close(fig)
        # Delete the temp folder only NOW. For the interactive window this is AFTER the user
        # closed it, so they had the chance to copy files from it; for the PNG fallback it is
        # right after the external viewer was launched (the shown PNG lives elsewhere).
        _safe_rmtree(a.delete_dir, a.temp_root)


if __name__ == "__main__":
    main()
