"""Plan 'Load settings': which scene properties an export, session or rotation log sets.

bpy-free, so tools/test_load_settings.py runs it without Blender. The main addon collects the
current scene values, calls plan_load() and sets plan.values; nothing here touches the scene.
Only case settings are loaded: no model parameters (R_target, the valve area check tolerance),
no values the pipeline steps compute ('internal', the positions after the rotation) and no
interface preferences (the live_* check boxes).
"""
import json
import math
import os
from dataclasses import dataclass, field

SCHEMAS = (1, 2, 3)
LOG_KINDS = {"export": "Export log", "session_close": "Session log", "rotate": "Rotation log"}

TODO_VALVE_SIZES = "Calculate valve sizes"
TODO_PICK_LANDMARKS = "Pick the landmarks by hand"
TODO_ROTATE = "Press 'Translate and rotate' only after a fresh import"
ROTATION_LOG = os.path.join("rotated", "gvr_rotation.json")  # in the Import folder, see session_log.write_rotation_log

# Folders: property -> path in the log. Set only if the folder exists on this computer.
FOLDERS = [
    ("ventricle_import_dir", "files.import_dir"),
    ("ventricle_export_dir", "files.export_dir"),
    ("plot_input_path", "comparison.plot_input_path"),
]
# Case settings: (property, path in the log, type), in panel order. Taken 1:1, nothing is converted.
SETTINGS = [
    ("translation_mitral", "valves.mitral.translation_mm", "vector"),
    ("angle_mitral", "valves.mitral.angle_deg", "vector"),
    ("translation_aortic", "valves.aortic.translation_mm", "vector"),
    ("angle_aortic", "valves.aortic.angle_deg", "vector"),
    ("remove_basal_threshold", "setup_variables.remove_basal_threshold", "number"),
    ("mean_reference", "setup_variables.mean_reference", "bool"),
    ("time_rr", "setup_variables.interpolation.time_rr", "number"),
    ("time_diastole", "setup_variables.interpolation.time_diastole", "number"),
    ("frames_ventricle", "setup_variables.interpolation.frames_ventricle", "int"),
    ("poisson_depth", "setup_variables.poisson_depth", "int"),
    ("connection_twist", "setup_variables.connection.connection_twist", "int"),
    ("inset_faces_refinement_steps", "setup_variables.connection.inset_faces_refinement_steps", "int"),
    ("max_con_sm_iter", "setup_variables.connection_smoothing.max_con_sm_iter", "int"),
    ("min_con_sm_iter", "setup_variables.connection_smoothing.min_con_sm_iter", "int"),
    ("sm_reps", "setup_variables.connection_smoothing.sm_reps", "int"),
    ("con_fade_percentage", "setup_variables.connection_smoothing.con_fade_percentage", "number"),
    ("final_sm_iter", "setup_variables.connection_smoothing.final_sm_iter", "int"),
    ("approach", "pipeline.approach", "int"),
]
# Valve sizes in mm, from schema 2 on. Schema 1 logs stretch factors (radius_*) instead.
SIZES = [
    ("mitral_width", "valves.mitral.width_mm"),
    ("mitral_depth", "valves.mitral.depth_mm"),
    ("aortic_diameter", "valves.aortic.diameter_mm"),
]
# Landmarks from the rotation log: the nodes picked before 'Translate and rotate'.
LANDMARKS = [
    ("pos_top", "selected_points.top"),
    ("pos_bot", "selected_points.bot"),
    ("pos_septum", "selected_points.septum"),
]
TYPE_NAMES = {"number": "a number", "int": "a whole number", "bool": "true or false", "vector": "three numbers",
              "text": "a text"}

_MISSING = object()


class SettingsLogError(Exception):
    """The file is no readable log of this addon; nothing may be loaded from it."""


@dataclass
class LoadPlan:
    values: dict  # property -> new value, in panel order
    header: list  # case, file, kind of log, date, number of values
    notes: list = field(default_factory=list)  # the schema 1 note
    kept: list = field(default_factory=list)  # what keeps its current value, with the reason
    todo: list = field(default_factory=list)  # the next steps
    changes: list = field(default_factory=list)  # 'property: old -> new' for the console
    import_dir: str = None  # the resolved Import folder of the loaded case, None if unknown
    valve_sizes_out_of_date: bool = False  # schema 1 or a size missing: To do 'Calculate valve sizes'
    path: str = ""  # the loaded log

    def status(self):
        """The status line text. Always an info: a warning would open a second popup over the window."""
        left_out = []
        if self.valve_sizes_out_of_date:
            left_out.append("valve sizes out of date")
        if "pos_top" not in self.values:  # the landmarks come all three or none
            left_out.append("landmarks not loaded")
        if self.kept:
            left_out.append(f"{len(self.kept)} kept")
        text = f"Load settings: {len(self.values)} settings loaded from '{os.path.basename(self.path)}'"
        return text + (f"; {', '.join(left_out)} (see the console)." if left_out else ".")

    def console_lines(self):
        """What the console shows: every change as 'old -> new', then what was kept and what is left to do."""
        lines = [f"Loaded from: {self.path}"] + self.header + self.notes
        for title, items in (("Changed", self.changes), ("Kept", self.kept), ("To do", self.todo)):
            if items:
                lines += [f"{title}:"] + [f"  {item}" for item in items]
        return lines + [f"{len(self.values) - len(self.changes)} loaded values were already set."]


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _typed(value, kind, path):
    """The value converted to its property type, or SettingsLogError for a wrong type."""
    if kind == "number" and _is_number(value):
        return float(value)
    if kind == "int" and isinstance(value, int) and not isinstance(value, bool):
        return value
    if kind == "bool" and isinstance(value, bool):
        return value
    if kind == "text" and isinstance(value, str):
        return value
    if kind == "vector" and isinstance(value, list) and len(value) == 3 and all(map(_is_number, value)):
        return tuple(float(x) for x in value)
    raise SettingsLogError(f"'{path}' should be {TYPE_NAMES[kind]}, the file has {json.dumps(value)}")


def _get(data, path, kind):
    """The typed value at a dotted path, or _MISSING if a key on the way is missing."""
    node = data
    for key in path.split("."):
        if not isinstance(node, dict):
            raise SettingsLogError(f"'{path}': '{key}' is not inside a section")
        if key not in node:
            return _MISSING
        node = node[key]
    return _typed(node, kind, path)


def _read_log(path):
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise SettingsLogError(f"cannot read the file: {exc}") from None
    except ValueError:  # also UnicodeDecodeError
        raise SettingsLogError("the file is not JSON") from None
    meta = data.get("meta") if isinstance(data, dict) else None
    if not isinstance(meta, dict):
        raise SettingsLogError("no 'meta' section: not a settings log of this addon")
    schema = meta.get("schema_version")
    if not (isinstance(schema, int) and not isinstance(schema, bool) and schema in SCHEMAS):
        raise SettingsLogError(f"unknown schema version {json.dumps(schema)} (this addon reads {SCHEMAS[0]} to "
                               f"{SCHEMAS[-1]})")
    if meta.get("trigger") not in LOG_KINDS:
        raise SettingsLogError(f"unknown kind of log {json.dumps(meta.get('trigger'))}")
    return data


def _existing_dir(raw, abspath):
    """The resolved folder if raw is absolute or relative to the .blend ('//') and exists, else None.

    Other relative paths would depend on the working folder of Blender (or of the test).
    """
    if not raw or not (raw.startswith("//") or os.path.isabs(raw)):
        return None
    resolved = os.path.normpath(abspath(raw))
    return resolved if os.path.isdir(resolved) else None


def _case_import_dir(raw, abspath):
    """The resolved Import folder of a case, or None. The default '//' (next to the .blend) means no
    Import folder, as in session_log.resolve_case_dir."""
    if (raw or "").strip() in ("", "//"):
        return None
    return _existing_dir(raw, abspath)


def _text(value):
    if isinstance(value, tuple):
        return "(" + ", ".join(map(_text, value)) + ")"
    return f"{value:g}" if isinstance(value, float) else str(value)


def _same(a, b):
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(map(_same, a, b))
    if _is_number(a) and _is_number(b) and not (isinstance(a, int) and isinstance(b, int)):
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)  # the log rounds to 6 digits
    return a == b


def _landmarks(log):
    """pos_top, pos_bot and pos_septum from a rotation log. All three or SettingsLogError:
    landmarks of two logs must never mix."""
    found = {prop: _get(log, key, "vector") for prop, key in LANDMARKS}
    missing = [key for (prop, key) in LANDMARKS if found[prop] is _MISSING]
    if missing:
        raise SettingsLogError(f"'{missing[0]}' is missing")
    return found


def _plan_case_settings(data, current, abspath, plan):
    """Export or session log: folders, valves, setup variables and approach. Returns this log's Import
    folder (resolved), or None if it is not found or only '//'."""
    for prop, key in FOLDERS:
        raw = _get(data, key, "text")
        if raw is _MISSING:
            plan.kept.append(f"{key}: missing in the log")
        elif _existing_dir(raw, abspath):
            plan.values[prop] = raw
        else:
            plan.kept.append(f"{key}: folder '{raw}' not found")

    for prop, key, kind in SETTINGS:
        value = _get(data, key, kind)
        if value is _MISSING:
            plan.kept.append(f"{key}: missing in the log")
        else:
            plan.values[prop] = value

    if data["meta"]["schema_version"] == 1:
        plan.notes.append("This log is from before the valve size fix (schema 1): its valve sizes are out of date "
                          "and were not loaded.")
        plan.valve_sizes_out_of_date = True
    else:
        for prop, key in SIZES:
            value = _get(data, key, "number")
            if value is _MISSING:
                plan.kept.append(f"{key}: missing in the log")
                plan.valve_sizes_out_of_date = True
            else:
                plan.values[prop] = value
        r_target = _get(data, "valves.R_target", "number")
        if r_target is not _MISSING and not _same(r_target, current.get("valve_R_target")):
            plan.kept.append(f"valves.R_target: {r_target:g} in the log, {current.get('valve_R_target'):g} in the "
                             "scene (model parameter, not loaded)")
    if plan.valve_sizes_out_of_date:
        plan.todo.append(TODO_VALVE_SIZES)

    # The landmarks come from the rotation log of this log's own Import folder, never from the rotation
    # part of this log or from the scene's Import folder: both may belong to another case.
    import_dir = _case_import_dir(plan.values.get("ventricle_import_dir"), abspath)
    rotation_log = os.path.join(import_dir, ROTATION_LOG) if import_dir else None
    if rotation_log and os.path.isfile(rotation_log):
        try:
            plan.values.update(_landmarks(_read_log(rotation_log)))
            plan.todo.append(TODO_ROTATE)
        except SettingsLogError as exc:  # a broken side file only leaves out the landmarks
            plan.kept.append(f"Landmarks: '{rotation_log}' is unreadable: {exc}")
            plan.todo.append(TODO_PICK_LANDMARKS)
    else:
        plan.todo.append(TODO_PICK_LANDMARKS)
    return import_dir


def plan_load(path, current, abspath):
    """Plan loading the log at path. Changes nothing.

    current: property name -> value in the scene now (vectors as tuples).
    abspath: turns a '//' path into a real one (bpy.path.abspath in Blender).
    Raises SettingsLogError if the file is broken or foreign.
    """
    data = _read_log(path)
    meta = data["meta"]
    plan = LoadPlan(values={}, header=[], path=path)
    if meta["trigger"] == "rotate":  # only the landmarks
        plan.values.update(_landmarks(data))
        plan.todo.append(TODO_ROTATE)
        # The case is where the file lies (<case>/<Import>/rotated/), not the scene's Import folder.
        folder = os.path.dirname(os.path.abspath(path))
        import_dir = os.path.dirname(folder) if os.path.basename(folder) == "rotated" else None
        scene_dir = _case_import_dir(current.get("ventricle_import_dir"), abspath)
        if not (import_dir and scene_dir and os.path.normcase(import_dir) == os.path.normcase(scene_dir)):
            plan.notes.append("This rotation log is not in the current Import folder: its landmarks may belong to "
                              "another case.")
    else:
        import_dir = _plan_case_settings(data, current, abspath, plan)

    plan.import_dir = import_dir
    case = os.path.basename(os.path.dirname(import_dir)) if import_dir else "unknown (no valid Import folder)"
    plan.header = [
        f"Case: {case}",
        f"File: {os.path.basename(path)}",
        f"{LOG_KINDS[meta['trigger']]}, written {meta.get('written_at') or 'unknown'}",
        f"{len(plan.values)} settings loaded",
    ]
    plan.changes = [f"{p}: {_text(current.get(p))} -> {_text(v)}" for p, v in plan.values.items()
                    if not _same(current.get(p), v)]
    return plan
