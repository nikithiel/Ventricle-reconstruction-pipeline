"""Plan 'Load settings': which scene properties an export, session or rotation log sets.

bpy-free, so tools/test_load_settings.py runs it without Blender. The main addon collects the
current scene values, calls plan_load() and sets plan.values; nothing here touches the scene.
An export or session log of another case first returns a Question ('Switch' to the log's case or
'Stay' in the scene's case); the window asks it, then plan_load() runs again with the choice.
On another computer the case folder is where the log lies, if it has the same name.
Only case settings are loaded: no model parameters (R_target, the valve area check tolerance),
no values the pipeline steps compute ('internal', the positions after the rotation) and no
interface preferences (the live_* check boxes).
"""
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

SCHEMAS = (1, 2, 3)
LOG_KINDS = {"export": "Export log", "session_close": "Session log", "rotate": "Rotation log"}

TODO_VALVE_SIZES = "Calculate valve sizes"
TODO_PICK_LANDMARKS = "Pick the landmarks by hand"
TODO_ROTATE = "Press 'Translate and rotate' only after a fresh import"
TODO_SET_FOLDER = {"ventricle_export_dir": "Set the Export folder",
                   "plot_input_path": "Set the Processed geometries folder (empty: the Export folder is used)"}
SWITCH, STAY = "SWITCH", "STAY"  # the choices of the window 'Load settings: which case?'
ROTATION_LOG = os.path.join("rotated", "gvr_rotation.json")  # in the Import folder, see session_log.write_rotation_log

# Folders: property -> path in the log.
FOLDERS = [
    ("ventricle_import_dir", "files.import_dir"),
    ("ventricle_export_dir", "files.export_dir"),
    ("plot_input_path", "comparison.plot_input_path"),
]
FOLDER_LABELS = {"ventricle_import_dir": "Import folder", "ventricle_export_dir": "Export folder",
                 "plot_input_path": "Processed geometries"}
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
        if TODO_PICK_LANDMARKS in self.todo:  # not after 'Stay': there the landmarks stay on purpose
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


@dataclass
class Question:
    """An export or session log of another case: the window 'Load settings: which case?' asks first."""
    path: str
    kind: str  # 'Export log' or 'Session log'
    log_case: str  # the case name of the log, None if unknown
    log_case_dir: str  # its case folder on this computer, None if neither found nor derived
    derived: bool  # the folders of the log are not on this computer: the case is where the log lies
    scene_case: str  # the case name of the scene, None if the scene has no case
    scene_case_dir: str
    log_folders: list  # (label, folder as in the log, found on this computer)
    switch_folders: list  # (label, folder after 'Switch'), empty without 'Switch'
    reason: str  # why there is no 'Switch', else None
    choices: tuple  # SWITCH and STAY, or only STAY
    default: str

    def lines(self):
        """The text above the choices."""
        file = os.path.basename(self.path)
        kind = {"Export log": "an export log", "Session log": "a session log"}[self.kind]
        lines = [f"'{file}' is {kind} of case '{self.log_case}'." if self.log_case else
                 f"'{file}' is {kind} of an unknown case (no case folder in the log)."]
        if self.derived:
            lines.append("Its folders are not on this computer:")
            lines += [f"  {label}  {raw}" for label, raw, found in self.log_folders
                      if not found and raw.strip()]  # empty is not missing: it means the default
            lines.append(f"The log lies in the case folder {self.log_case_dir} (same case name).")
        return lines

    def _names(self):
        log = f"'{self.log_case}'" if self.log_case else "the log"
        return log, (f"'{self.scene_case}'" if self.scene_case else "this scene")

    def label(self, choice):
        """The text of a choice; the case folders follow if both cases have the same name."""
        if choice == SWITCH:
            text, folder = f"Switch to case '{self.log_case}'", self.log_case_dir
        elif self.scene_case:
            text, folder = f"Stay in case '{self.scene_case}'", self.scene_case_dir
        else:
            return "Keep the folders of this scene (no case set)"
        same_name = (self.log_case or "").lower() == (self.scene_case or "").lower()
        return f"{text} ({folder})" if same_name else text

    def explanation(self, choice):
        """The lines under a choice in the window."""
        log, scene = self._names()
        if choice == SWITCH:
            return [f"Work on {log} from now on: its folders and all its settings."]
        return [f"Keep working on {scene} with its folders, landmarks and valve sizes.",
                f"The other settings come from {log} as a starting point."]

    def tooltip(self, choice):
        """The help text when the mouse is over a choice."""
        log, scene = self._names()
        if choice == SWITCH:
            rest = f"; nothing of {scene} is used any more." if self.scene_case else "."
            return f"The scene becomes case {log}. Import, export and session logs use the folders of {log}{rest}"
        stays = f"The scene stays case {scene}" if self.scene_case else "The scene keeps its folders (no case set)"
        return (f"{stays}: import, export and session logs keep using its folders. From {log}: valve position and "
                f"angle, the 13 setup variables, the approach. Not from {log}: folders, landmarks, valve sizes "
                f"(calculate them for {scene}).")


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


def _no_import_dir(raw):
    """An empty Import folder or the default '//' (next to the .blend) means no Import folder, as in
    session_log.resolve_case_dir."""
    return (raw or "").strip() in ("", "//")


def _case_import_dir(raw, abspath):
    """The resolved Import folder of a case, or None."""
    return None if _no_import_dir(raw) else _existing_dir(raw, abspath)


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


def _split(raw):
    """(root, parts) of a folder as written in a log (Windows), the same on every system. The root is '//'
    (relative to the .blend), a drive 'C:', a UNC share (two backslashes, server and share) or ''. A leading
    '//' is always Blender's, never UNC: Python's ntpath would read '//..\\CaseA' as a share and lose the case
    name."""
    text = raw.strip().rstrip("\\/")
    if text.startswith("//"):
        root, text = "//", text[2:]
    elif text.startswith("\\\\"):
        server_share = re.split(r"[\\/]+", text[2:], maxsplit=2)
        root, text = "\\\\" + "\\".join(server_share[:2]), "".join(server_share[2:])
    elif text[1:2] == ":":
        root, text = text[:2], text[2:]
    else:
        root = ""
    return root, [p for p in re.split(r"[\\/]+", text) if p]


def _case_name(parts):
    """The case name, the folder above the Import folder (parts of the Import folder); None if there is none."""
    if len(parts) < 2 or {parts[-1], parts[-2]} & {".", ".."}:
        return None
    return parts[-2]


def _rest_in_case(raw, root, case_parts):
    """The parts of raw after the case folder (root, case_parts): [] for the case folder itself, None outside,
    also for a '..' after the case folder."""
    r, parts = _split(raw)
    n = len(case_parts)
    if r.lower() != root.lower() or [p.lower() for p in parts[:n]] != [p.lower() for p in case_parts]:
        return None
    return None if {".", ".."} & set(parts[n:]) else parts[n:]


def _case_key(folder):
    """Case folders compare as on Windows: without case."""
    return os.path.normcase(os.path.normpath(folder)).lower()


def _as_blender_folder(folder, blend_dir):
    """A derived folder as Blender's folder browser writes it: '//...' under the .blend, else absolute, with a
    trailing separator. blend_dir is None for an unsaved .blend."""
    if blend_dir:
        try:
            rel = os.path.relpath(folder, blend_dir)
        except ValueError:  # another drive on Windows
            rel = os.pardir
        if rel == os.curdir:
            return "//"
        if rel != os.pardir and not rel.startswith(os.pardir + os.sep):
            return "//" + os.path.join(rel, "")
    return os.path.join(folder, "")


@dataclass
class _Case:
    """Where the case of an export or session log is on this computer."""
    raws: dict  # property -> folder as in the log, None if missing
    found: dict  # property -> the folder resolved on this computer, None if not found
    root: str = ""  # the case folder as in the log: root and parts, from the Import folder
    parts: list = field(default_factory=list)
    name: str = None  # the case name, None if unknown
    case_dir: str = None  # the case folder on this computer, None if neither found nor derived
    import_dir: str = None
    derived: bool = False  # the folders of the log are not found: the case is where the log lies
    reason: str = None  # why there is no 'Switch'
    blend_dir: str = None  # the folder of the .blend, None if it is not saved


def _find_case(path, data, abspath):
    raws = {}
    for prop, key in FOLDERS:
        raw = _get(data, key, "text")
        raws[prop] = None if raw is _MISSING else raw
    found = {prop: _existing_dir(raw, abspath) for prop, raw in raws.items()}  # each a network check for UNC
    if _no_import_dir(raws["ventricle_import_dir"]):
        found["ventricle_import_dir"] = None
    blend_dir = abspath("//")  # '' for an unsaved .blend in Blender; '//' left as is resolves nothing
    saved = not blend_dir.startswith("//") and os.path.isabs(blend_dir) and os.path.isdir(blend_dir)
    case = _Case(raws, found, blend_dir=blend_dir if saved else None)
    import_raw = raws["ventricle_import_dir"]
    root, parts = ("", []) if _no_import_dir(import_raw) else _split(import_raw)
    case.root, case.parts = root, parts[:-1]
    if found["ventricle_import_dir"]:
        case.import_dir = found["ventricle_import_dir"]
        case.case_dir = os.path.dirname(case.import_dir)
        case.name = os.path.basename(case.case_dir)
        return case
    case.name = _case_name(parts)
    if case.name is None:
        case.reason = "To switch to the case of this log, set its folders by hand."
        return case
    new_case = _case_dir_where_log_lies(path, data["meta"]["trigger"], raws["ventricle_export_dir"], case.root,
                                        case.parts)
    import_dir = os.path.join(new_case, parts[-1]) if new_case else None
    if new_case is None:
        case.reason = (f"The folders of case '{case.name}' are not on this computer, and the log does not lie in its "
                       "case folder: set the folders by hand.")
    elif os.path.basename(new_case).lower() != case.name.lower():
        case.reason = (f"The log is from case '{case.name}' but lies in '{os.path.basename(new_case)}': rename the "
                       "folder or set the folders by hand.")
    elif not os.path.isdir(import_dir):
        case.reason = f"The case folder {new_case} has no Import folder '{parts[-1]}': set the folders by hand."
    else:
        case.derived, case.case_dir, case.import_dir = True, new_case, import_dir
    return case


def _case_dir_where_log_lies(path, trigger, export_raw, root, case_parts):
    """The case folder where the log lies: above Session_Logs for a session log. An export log lies in the Export
    folder: the case folder is that folder without the Export folder's rest in the case of the log."""
    log_dir = os.path.dirname(os.path.abspath(path))
    if trigger == "session_close":
        return os.path.dirname(log_dir) if os.path.basename(log_dir).lower() == "session_logs" else None
    rest = _rest_in_case(export_raw, root, case_parts) if export_raw else None
    if rest is None:
        return None
    parts = list(Path(log_dir).parts)
    n = len(parts) - len(rest)
    if n < 1 or [p.lower() for p in parts[n:]] != [p.lower() for p in rest]:
        return None
    return str(Path(*parts[:n]))


def _plan_folders(case, same, plan):
    """'Switch': the folders of the log's case on this computer, never a folder of the scene's other case."""
    for prop, key in FOLDERS:
        raw = case.raws[prop]
        rest = _rest_in_case(raw, case.root, case.parts) if raw else None
        if case.found[prop]:
            plan.values[prop] = raw
        elif prop == "ventricle_import_dir":
            plan.values[prop] = _as_blender_folder(case.import_dir, case.blend_dir)
        elif raw is not None and not raw.strip():  # empty: next to the .blend, or the Export folder
            plan.values[prop] = raw
        elif rest is not None:  # in the case folder: set even if it does not exist yet, the export creates it
            plan.values[prop] = _as_blender_folder(os.path.join(case.case_dir, *rest), case.blend_dir)
        elif same:  # the scene's folder belongs to this case, so it stays
            plan.kept.append(f"{key}: missing in the log" if raw is None else f"{key}: folder '{raw}' not found")
        else:
            plan.values[prop] = "//"
            plan.todo.append(TODO_SET_FOLDER[prop])


def _check_types(data):
    """A wrong type anywhere loading reads raises SettingsLogError, so a broken file fails before any question."""
    for _prop, key, kind in SETTINGS:
        _get(data, key, kind)
    for _prop, key in SIZES:
        _get(data, key, "number")
    _get(data, "valves.R_target", "number")


def _plan_settings(data, current, plan):
    """Valve translations and angles, setup variables and approach, and a note on a deviating R_target."""
    for prop, key, kind in SETTINGS:
        value = _get(data, key, kind)
        if value is _MISSING:
            plan.kept.append(f"{key}: missing in the log")
        else:
            plan.values[prop] = value
    r_target = _get(data, "valves.R_target", "number")
    if r_target is not _MISSING and not _same(r_target, current.get("valve_R_target")):
        plan.kept.append(f"valves.R_target: {r_target:g} in the log, {current.get('valve_R_target'):g} in the "
                         "scene (model parameter, not loaded)")


def _plan_sizes(data, plan):
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
    if plan.valve_sizes_out_of_date:
        plan.todo.append(TODO_VALVE_SIZES)


def _plan_landmarks(import_dir, plan):
    """The landmarks come from the rotation log of the case's own Import folder, never from the rotation part of
    the log or from the scene's Import folder: both may belong to another case."""
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


def _question(path, kind, case, scene_dir, choices):
    folders = LoadPlan(values={}, header=[])
    if SWITCH in choices:
        _plan_folders(case, False, folders)
    return Question(
        path=path, kind=kind, log_case=case.name, log_case_dir=case.case_dir, derived=case.derived,
        scene_case=os.path.basename(scene_dir) if scene_dir else None, scene_case_dir=scene_dir,
        log_folders=[(FOLDER_LABELS[prop], case.raws[prop], case.found[prop] is not None) for prop, _key in FOLDERS
                     if case.raws[prop] is not None],
        switch_folders=[(FOLDER_LABELS[prop], value + "  (not found: set it after loading)"
                         if TODO_SET_FOLDER.get(prop) in folders.todo else value)
                        for prop, value in folders.values.items()],
        reason=case.reason, choices=choices, default=choices[0])


def plan_load(path, current, abspath, choice=None):
    """Plan loading the log at path. Changes nothing.

    current: property name -> value in the scene now (vectors as tuples).
    abspath: turns a '//' path into a real one (bpy.path.abspath in Blender).
    choice: None, SWITCH or STAY. Without a choice, an export or session log of another case returns the
    Question of the window instead of a LoadPlan; then plan again with the choice taken there.
    Raises SettingsLogError if the file is broken or foreign, before any question.
    """
    data = _read_log(path)
    meta = data["meta"]
    plan = LoadPlan(values={}, header=[], path=path)
    if meta["trigger"] == "rotate":  # only the landmarks, never a question
        if choice is not None:
            raise ValueError(f"load settings: the choice {choice!r} was not offered (a rotation log asks nothing)")
        plan.values.update(_landmarks(data))
        plan.todo.append(TODO_ROTATE)
        # The case is where the file lies (<case>/<Import>/rotated/), not the scene's Import folder.
        folder = os.path.dirname(os.path.abspath(path))
        import_dir = os.path.dirname(folder) if os.path.basename(folder) == "rotated" else None
        scene_dir = _case_import_dir(current.get("ventricle_import_dir"), abspath)
        if not (import_dir and scene_dir and os.path.normcase(import_dir) == os.path.normcase(scene_dir)):
            plan.notes.append("This rotation log is not in the current Import folder: its landmarks may belong to "
                              "another case.")
        case_line = os.path.basename(os.path.dirname(import_dir)) if import_dir else "unknown (no valid Import folder)"
    else:
        _check_types(data)
        case = _find_case(path, data, abspath)
        scene_import = _case_import_dir(current.get("ventricle_import_dir"), abspath)
        scene_dir = os.path.dirname(scene_import) if scene_import else None
        same = bool(case.case_dir and scene_dir and _case_key(case.case_dir) == _case_key(scene_dir))
        # 'Switch' needs the Import folder of the log's case on this computer; the same case asks nothing.
        choices = (SWITCH,) if same else (SWITCH, STAY) if case.import_dir else (STAY,)
        if choice is None:
            if not same:
                return _question(path, LOG_KINDS[meta["trigger"]], case, scene_dir, choices)
            choice = SWITCH
        if choice not in choices:
            raise ValueError(f"load settings: the choice {choice!r} was not offered ({', '.join(choices)})")
        if choice == STAY:  # the scene's folders, landmarks and valve sizes stay
            _plan_settings(data, current, plan)
            plan.todo.append(TODO_VALVE_SIZES)
            plan.valve_sizes_out_of_date = True
            import_dir = scene_import
            source = f"settings from case '{case.name}'" if case.name else "settings from an unknown case"
            case_line = f"{os.path.basename(scene_dir)} ({source})" if scene_dir else \
                f"unknown (no valid Import folder), {source}"
        else:
            _plan_folders(case, same, plan)
            _plan_settings(data, current, plan)
            _plan_sizes(data, plan)
            _plan_landmarks(case.import_dir, plan)
            import_dir = case.import_dir
            case_line = case.name if same else f"{case.name} (folders in {case.case_dir})"

    plan.import_dir = import_dir
    plan.header = [
        f"Case: {case_line}",
        f"File: {os.path.basename(path)}",
        f"{LOG_KINDS[meta['trigger']]}, written {meta.get('written_at') or 'unknown'}",
        f"{len(plan.values)} settings loaded",
    ]
    plan.changes = [f"{p}: {_text(current.get(p))} -> {_text(v)}" for p, v in plan.values.items()
                    if not _same(current.get(p), v)]
    return plan
