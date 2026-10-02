"""JSON settings log for the GVR-Pipeline addon.

Replaces the manual Excel bookkeeping of the addon settings. Three triggers write JSON:

  A) 'Export ventricle'     -> <export_dir>/gvr_settings.json          (full settings)
  B) 'Translate and rotate' -> <import_dir>/rotated/gvr_rotation.json  (landmarks only)
  C) Blender shutdown       -> <case_dir>/Session_Logs/session_<stamp>.json (full settings)

<case_dir> is one level above the import folder, the same place where
resolve_raw_and_settings_paths() in the main addon looks for inputPython.txt.

Blender has no quit handler, so trigger C is built from two parts: a bpy.app timer keeps a
bpy-free snapshot of the settings up to date (and already writes it to disk, which doubles as
crash protection), and an atexit handler flushes that cached snapshot one last time. The atexit
path must never touch bpy: during Py_Finalize() bpy.data and bpy.context are already partly
torn down.
"""

import atexit
import json
import math
import os
import time

import bpy

SCHEMA_VERSION = 1
LOG_DIRNAME = "Session_Logs"
FULL_LOG_NAME = "gvr_settings.json"
ROTATION_LOG_NAME = "gvr_rotation.json"
TICK_INTERVAL = 5.0  # Seconds between snapshots of the scene settings.


class _Vec(list):
    """A coordinate triple. Dumped on a single line so the file stays readable."""


def _f(value, digits=6):
    """Round a scalar property to a JSON-friendly float."""
    return round(float(value), digits)


def _v(vector, digits=6):
    """Convert a FloatVectorProperty (bpy_prop_array, not JSON serializable) to a list."""
    return _Vec(round(float(x), digits) for x in vector)


def _deg(angles_rad, digits=4):
    return _Vec(round(math.degrees(float(a)), digits) for a in angles_rad)


def _dumps(payload):
    """json.dumps with indent, but coordinate triples kept on one line.

    json has no per-value indent control, so every _Vec is swapped for a token that cannot
    occur in the data, then substituted back once the surrounding text is formatted.
    """
    tokens = {}

    def tokenize(value):
        if isinstance(value, _Vec):
            key = f"@@vec{len(tokens)}@@"
            tokens[key] = "[" + ", ".join(json.dumps(x) for x in value) + "]"
            return key
        if isinstance(value, dict):
            return {k: tokenize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [tokenize(v) for v in value]
        return value

    text = json.dumps(tokenize(payload), indent=2, ensure_ascii=False)
    for key, inline in tokens.items():
        text = text.replace(f'"{key}"', inline)
    return text


def write_json(path, data):
    """Write data to path atomically. Never raises, never touches bpy (atexit-safe)."""
    tmp = str(path) + ".tmp"
    try:
        payload = dict(data)
        meta = dict(payload.get("meta") or {})
        meta["written_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        payload["meta"] = meta

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(_dumps(payload))
            handle.write("\n")
        os.replace(tmp, path)
        return True
    except Exception as exc:
        print(f"[session_log] cannot write '{path}': {exc}")
        try:
            os.remove(tmp)
        except Exception:
            pass
        return False


def resolve_case_dir(scene):
    """Case folder = one level above the import folder. None if it cannot be resolved."""
    raw = (getattr(scene, "ventricle_import_dir", "") or "").strip()
    if not raw or raw == "//":
        return None
    import_dir = os.path.normpath(bpy.path.abspath(raw))
    if not os.path.isdir(import_dir):
        return None
    case_dir = os.path.dirname(import_dir)
    if not case_dir or not os.path.isdir(case_dir):
        return None
    return case_dir


def resolve_session_log_path(scene, stamp):
    """Path of this session's log file, or None if there is no valid case folder."""
    case_dir = resolve_case_dir(scene)
    if not case_dir:
        return None
    return os.path.join(case_dir, LOG_DIRNAME, f"session_{stamp}.json")


def _build_rotation_section(scene):
    """The landmarks the user picked plus the transformation derived from them.

    None until 'Translate and rotate' has run once in this .blend. rotate_ventricle()
    overwrites pos_top/pos_bot/pos_septum, so the picked points live in their own
    pos_*_selected properties.
    """
    performed_at = getattr(scene, "last_rotation_time", "") or ""
    if not performed_at:
        return None
    angles = list(scene.last_rotation_angles)
    return {
        "performed_at": str(performed_at),
        "selected_points": {
            "top": _v(scene.pos_top_selected),
            "bot": _v(scene.pos_bot_selected),
            "septum": _v(scene.pos_septum_selected),
        },
        "rotation_rad": _v(angles),
        "rotation_deg": _deg(angles),
        "translation": _v(scene.last_rotation_translation),
    }


def build_full_settings(scene, *, trigger, addon_version=""):
    """All scene settings, grouped like the addon panels. 'written_at' is stamped by write_json."""
    return {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "addon_version": str(addon_version),
            "trigger": str(trigger),
            "written_at": None,
            "blend_file": bpy.data.filepath or None,
        },
        "files": {
            "import_dir": str(scene.ventricle_import_dir),
            "export_dir": str(scene.ventricle_export_dir),
        },
        "position": {
            "pos_top": _v(scene.pos_top),
            "pos_bot": _v(scene.pos_bot),
            "pos_septum": _v(scene.pos_septum),
            "rotation": _build_rotation_section(scene),
        },
        "valves": {
            "mitral": {
                "translation_mm": _v(scene.translation_mitral),
                "angle_deg": _v(scene.angle_mitral),
                "radius_long": _f(scene.mitral_radius_long),
                "radius_small": _f(scene.mitral_radius_small),
            },
            "aortic": {
                "translation_mm": _v(scene.translation_aortic),
                "angle_deg": _v(scene.angle_aortic),
                "radius": _f(scene.aortic_radius),
            },
        },
        "setup_variables": {
            "remove_basal_threshold": _f(scene.remove_basal_threshold),
            "mean_reference": bool(scene.mean_reference),
            "interpolation": {
                "time_rr": _f(scene.time_rr),
                "time_diastole": _f(scene.time_diastole),
                "frames_ventricle": int(scene.frames_ventricle),
            },
            "poisson_depth": int(scene.poisson_depth),
            "connection": {
                "connection_twist": int(scene.connection_twist),
                "inset_faces_refinement_steps": int(scene.inset_faces_refinement_steps),
            },
            "connection_smoothing": {
                "max_con_sm_iter": int(scene.max_con_sm_iter),
                "min_con_sm_iter": int(scene.min_con_sm_iter),
                "sm_reps": int(scene.sm_reps),
            },
        },
        "pipeline": {
            "approach": int(scene.approach),
        },
        "comparison": {
            "plot_input_path": str(scene.plot_input_path),
            "live_compare": bool(scene.live_compare),
            "live_delete_temp": bool(scene.live_delete_temp),
        },
        "internal": {
            "height_plane": _f(scene.height_plane),
            "min_valves": _f(scene.min_valves),
            "ref_minima": _v(scene.ref_minima),
            "ref_maxima": _v(scene.ref_maxima),
            "reference_object_name": str(scene.reference_object_name),
        },
    }


def build_rotation_log(scene, rotation_result):
    """Slim log of the landmarks and the transformation applied by 'Translate and rotate'."""
    angles = list(rotation_result.get("rotation_rad", (0, 0, 0)))
    return {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "trigger": "rotate",
            "written_at": None,
        },
        "selected_points": {
            "top": _v(scene.pos_top_selected),
            "bot": _v(scene.pos_bot_selected),
            "septum": _v(scene.pos_septum_selected),
        },
        "position_after": {
            "top": _v(scene.pos_top),
            "bot": _v(scene.pos_bot),
            "septum": _v(scene.pos_septum),
        },
        "transformation": {
            "translation": _v(rotation_result.get("translation", (0, 0, 0))),
            "rotation_rad": _v(angles),
            "rotation_deg": _deg(angles),
        },
    }


def write_full_settings(scene, out_dir, *, trigger, addon_version=""):
    """Trigger A. Returns the written path, or None on failure."""
    try:
        data = build_full_settings(scene, trigger=trigger, addon_version=addon_version)
        path = os.path.join(out_dir, FULL_LOG_NAME)
        return path if write_json(path, data) else None
    except Exception as exc:
        print(f"[session_log] cannot build settings log: {exc}")
        return None


def write_rotation_log(scene, rotated_dir, rotation_result):
    """Trigger B. Returns the written path, or None on failure."""
    try:
        data = build_rotation_log(scene, rotation_result)
        path = os.path.join(rotated_dir, ROTATION_LOG_NAME)
        return path if write_json(path, data) else None
    except Exception as exc:
        print(f"[session_log] cannot build rotation log: {exc}")
        return None


# ---------------------------------------------------------------------------
# Trigger C: session logging (timer keeps the snapshot fresh, atexit flushes it)
# ---------------------------------------------------------------------------

_state = {
    "active": False,
    "stamp": None,
    "addon_version": "",
    "snapshot": None,
    "path": None,
    "last_written": None,
    "warned": False,
}


def _resolve_scene():
    scene = getattr(bpy.context, "scene", None)
    if scene is not None:
        return scene
    return bpy.data.scenes[0] if bpy.data.scenes else None


def _tick():
    try:
        if not _state["active"]:
            return None  # Backstop: an orphaned timer unregisters itself.
        scene = _resolve_scene()
        if scene is not None:
            snapshot = build_full_settings(scene, trigger="session_close",
                                           addon_version=_state["addon_version"])
            path = resolve_session_log_path(scene, _state["stamp"])
            _state["snapshot"] = snapshot
            if path:
                _state["path"] = path  # Keep the last good path for the atexit flush.
                # 'written_at' is stamped by write_json, so unchanged settings compare equal
                # and the timer stays silent instead of rewriting the file every tick.
                if snapshot != _state["last_written"] and write_json(path, snapshot):
                    _state["last_written"] = snapshot
            elif not _state["warned"]:
                _state["warned"] = True
                print("[session_log] no valid case folder; session log paused until "
                      "'Import folder' points into one.")
    except Exception as exc:
        print(f"[session_log] tick failed: {exc}")
    return TICK_INTERVAL  # Must always be a float, otherwise Blender drops the timer.


def _flush_atexit():
    """Final write on interpreter shutdown. Must not touch bpy: it is already torn down."""
    try:
        snapshot = _state.get("snapshot")
        path = _state.get("path")
        if not snapshot or not path:
            return
        data = dict(snapshot)
        meta = dict(data.get("meta") or {})
        meta["closed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        data["meta"] = meta
        write_json(path, data)
    except Exception:
        pass


def start_session_logging(addon_version=""):
    """Idempotent: a reload calls unregister() first, and start also cleans up defensively."""
    stop_session_logging()
    _state.update(active=True, stamp=time.strftime("%Y-%m-%d_%H-%M-%S"),
                  addon_version=str(addon_version),
                  snapshot=None, path=None, last_written=None, warned=False)
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=TICK_INTERVAL, persistent=True)
    atexit.unregister(_flush_atexit)
    atexit.register(_flush_atexit)


def stop_session_logging():
    _state["active"] = False
    try:
        if bpy.app.timers.is_registered(_tick):
            bpy.app.timers.unregister(_tick)
    except Exception:
        pass
    try:
        atexit.unregister(_flush_atexit)
    except Exception:
        pass
