#!/usr/bin/env python3
"""T4: tests for addons/load_settings.py ('Load settings'), runnable without Blender.

  python tools/test_load_settings.py

The planning function plan_load() reads an export, session or rotation log and returns the
scene properties it would set plus the notes for the 'Settings loaded' window. For a log of
another case it first returns the Question of the window 'Load settings: which case?' and plans
only with the choice. Expected values are the numbers written literally in the logs, the
values of the mock scene or the temporary case folders of the test. The logs of moved cases hold the
Windows folders of another computer, so the tests run the same on Linux and Windows.
"""
from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "addons"))
import load_settings as ls  # noqa: E402

EXAMPLE_LOG = REPO / "dataset/example/Session_Logs/session_2026-10-06_13-45-54.json"
FAILURES: list = []

# Scene of another case, with the addon defaults: what the scene holds before loading.
CURRENT = {
    "ventricle_import_dir": "/nonexistent/other_case/STL", "ventricle_export_dir": "//", "plot_input_path": "//",
    "translation_mitral": (0.0, 0.0, 1.0), "angle_mitral": (0.0, 0.0, 0.0),
    "translation_aortic": (0.0, 0.0, 1.0), "angle_aortic": (0.0, 0.0, 0.0),
    "mitral_width": 13.0, "mitral_depth": 10.0, "aortic_diameter": 7.0, "valve_R_target": 1.3,
    "remove_basal_threshold": 28.5, "mean_reference": True, "time_rr": 0.6, "time_diastole": 0.35,
    "frames_ventricle": 10, "poisson_depth": 10, "connection_twist": 0, "inset_faces_refinement_steps": 1,
    "max_con_sm_iter": 25, "min_con_sm_iter": 2, "sm_reps": 3, "con_fade_percentage": 20.0, "final_sm_iter": 10,
    "approach": 3, "pos_top": (0.0, 0.0, 1.0), "pos_bot": (0.0, 0.0, 0.0), "pos_septum": (0.0, 1.0, 0.0),
}
NEVER = {"height_plane", "min_valves", "ref_minima", "ref_maxima", "reference_object_name", "valve_R_target",
         "valve_check_tolerance", "live_compare", "live_delete_temp", "last_rotation_angles",
         "last_rotation_translation", "last_rotation_time", "pos_top_selected", "pos_bot_selected",
         "pos_septum_selected"}


def check(cond: bool, what: str) -> None:
    print(("PASS  " if cond else "FAIL  ") + what)
    if not cond:
        FAILURES.append(what)


def same(a, b) -> bool:
    return abs(a - b) <= 1e-6 if isinstance(a, float) else a == b


def identity(path: str) -> str:
    return path


def test_example_log() -> None:
    print("-- T4.1 example session log (schema 1): its case folder is where it lies")
    example = REPO / "dataset" / "example"
    blend = blend_in(REPO)  # GVR-Pipeline.blend lies in the repository folder
    q = ls.plan_load(str(EXAMPLE_LOG), CURRENT, blend)
    check(getattr(q, "log_case", None) == "example" and q.derived and q.log_case_dir == str(example)
          and q.choices == (ls.SWITCH, ls.STAY),
          f"the question offers 'Switch to case example': the shortened folders are derived "
          f"({getattr(q, 'log_case_dir', q)})")
    plan = ls.plan_load(str(EXAMPLE_LOG), CURRENT, blend, ls.SWITCH)
    v = plan.values
    check(v.get("approach") == 4 and same(v.get("remove_basal_threshold"), 42.0) and v.get("frames_ventricle") == 70,
          "approach 4, threshold 42.0, 70 frames")
    check(v.get("translation_mitral") == (0.0, -4.0, 50.5) and v.get("angle_aortic") == (-15.0, 0.0, 0.0),
          "MV translation (0, -4, 50.5), AV angle (-15, 0, 0)")
    check(same(v.get("time_rr"), 0.557) and same(v.get("time_diastole"), 0.31892) and v.get("mean_reference") is False
          and v.get("min_con_sm_iter") == 4 and v.get("sm_reps") == 2 and same(v.get("con_fade_percentage"), 20.0),
          "setup variables from the file (nested under interpolation and connection_smoothing)")
    check(not {"mitral_width", "mitral_depth", "aortic_diameter"} & set(v), "schema 1: no valve sizes")
    check(any("schema 1" in n for n in plan.notes) and ls.TODO_VALVE_SIZES in plan.todo,
          "schema 1 note and 'To do: Calculate valve sizes'")
    check(not NEVER & set(v), "nothing from internal, position, rotation and live_*")
    check(v.get("ventricle_import_dir") == "//" + os.path.join("dataset", "example", "STL", "")
          and v.get("ventricle_export_dir") == "//" + os.path.join("dataset", "example", "export", "")
          and v.get("plot_input_path") == "//" + os.path.join("dataset", "example", "export", ""),
          f"the folders of the example, relative to the .blend ({v.get('ventricle_import_dir')})")
    # The landmarks depend on the working folder: STL/rotated/gvr_rotation.json is neither tracked nor ignored.
    rest = set(v) - {"pos_top", "pos_bot", "pos_septum"}
    check(len(rest) == 21, f"21 values besides the landmarks: 3 folders, 4 valve vectors, 13 setup variables, "
          f"approach (got {len(rest)})")
    check(plan.valve_sizes_out_of_date is True and plan.import_dir == str(example / "STL"),
          f"window: valve sizes out of date, the Import folder exists ('Calculate valve sizes now') ({plan.import_dir})")
    status = plan.status()
    check(f"{len(v)} settings loaded from 'session_2026-10-06_13-45-54.json'" in status
          and "valve sizes out of date" in status and "kept" not in status,
          f"status line names the count, the file and the old valve sizes, nothing kept ({status})")
    lines = plan.console_lines()
    titles = [line for line in lines if line in ("Changed:", "Kept:", "To do:")]
    check(lines[0] == f"Loaded from: {EXAMPLE_LOG}" and f"Case: example (folders in {example})" in lines
          and any("schema 1" in line for line in lines) and titles[0] == "Changed:" and titles[-1] == "To do:"
          and "  approach: 3 -> 4" in lines and "  Calculate valve sizes" in lines,
          "console: file, header, schema 1 note, then Changed and To do with indented lines")
    check("approach: 3 -> 4" in plan.changes and "translation_mitral: (0, 0, 1) -> (0, -4, 50.5)" in plan.changes
          and "time_rr: 0.6 -> 0.557" in plan.changes and not any("poisson_depth" in c for c in plan.changes),
          f"console: old -> new for changed values only ({plan.changes[:3]} ...)")


def real_session_log():
    """addons/session_log.py with the bpy mock of T1/T2, so the tests use the real writers."""
    sys.path.insert(0, str(REPO / "tools"))
    import test_valve_area_check as t12
    t12.mock_bpy(t12._Objects()).data.filepath = ""  # no valve templates: the log keeps the sizes only
    import session_log
    return session_log


def make_case(tmp, name, landmarks=None):
    """Case folder <tmp>/<name> with STL/ and export/, and a rotation log written by the real writer."""
    case = Path(tmp) / name
    (case / "STL").mkdir(parents=True)
    (case / "export").mkdir()
    if landmarks:
        top, bot, septum = landmarks
        scene = types.SimpleNamespace(pos_top_selected=top, pos_bot_selected=bot, pos_septum_selected=septum,
                                      pos_top=(0.1, 0.2, 50.0), pos_bot=(0.0, 0.0, 0.0), pos_septum=(-9.0, 1.0, 40.0))
        rotation = {"translation": (-3.0, 1.0, -2.0), "rotation_rad": (0.1, 0.0, 0.6)}
        check(real_session_log().write_rotation_log(scene, str(case / "STL" / "rotated"), rotation) is not None,
              "the real writer wrote the rotation log")
    return case


def blend_in(case):
    """bpy.path.abspath for a .blend saved in the case folder."""
    return lambda p: str(case / p[2:]) if p.startswith("//") else p


def write_log(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(path)


def example_with(**sections):
    """The example log with some sections replaced."""
    data = json.loads(EXAMPLE_LOG.read_text(encoding="utf-8"))
    data.update(sections)
    return data


LANDMARKS = ((1.5, 2.5, 60.0), (3.0, -1.0, 2.0), (-20.0, 4.0, 45.0))


def after_question(log, current, abspath, choice, choices=("SWITCH", "STAY")):
    """Without a choice the question comes and offers choices; then the plan for the choice."""
    q = ls.plan_load(log, current, abspath)
    offered = getattr(q, "choices", None)
    check(offered == choices, f"first the question, it offers {', '.join(choices)} ({offered})")
    return ls.plan_load(log, current, abspath, choice)


def test_example_in_case(tmp) -> None:
    print("-- T4.2 example log in a case folder with real folders and a rotation log")
    case = make_case(tmp, "Case_example", LANDMARKS)
    log = write_log(case / "Session_Logs" / "session_x.json", example_with(
        files={"import_dir": "//STL/", "export_dir": "//export/"},
        comparison={"plot_input_path": "//export/", "live_compare": True, "live_delete_temp": True}))
    plan = after_question(log, CURRENT, blend_in(case), ls.SWITCH)
    v = plan.values
    check(v.get("ventricle_import_dir") == "//STL/" and v.get("ventricle_export_dir") == "//export/"
          and v.get("plot_input_path") == "//export/", "the folders exist and are set as written in the log")
    check(v.get("pos_top") == LANDMARKS[0] and v.get("pos_bot") == LANDMARKS[1] and v.get("pos_septum") == LANDMARKS[2],
          "landmarks from STL/rotated/gvr_rotation.json, not from the rotation part of the log")
    check(ls.TODO_ROTATE in plan.todo and ls.TODO_PICK_LANDMARKS not in plan.todo,
          "To do: 'Translate and rotate' only after a fresh import")
    check(plan.header[0] == f"Case: Case_example (folders in {case})" and len(v) == 24 and not plan.kept and not NEVER & set(v),
          f"case named after the case folder, 24 values, nothing kept (got {len(v)}, kept {plan.kept})")
    check(plan.valve_sizes_out_of_date is True and plan.import_dir == str(case / "STL"),
          f"window: valve sizes out of date, the Import folder exists ('Calculate valve sizes now') ({plan.import_dir})")


CASE_SETTINGS = {
    "ventricle_import_dir", "ventricle_export_dir", "plot_input_path",
    "translation_mitral", "angle_mitral", "mitral_width", "mitral_depth",
    "translation_aortic", "angle_aortic", "aortic_diameter",
    "remove_basal_threshold", "mean_reference", "time_rr", "time_diastole", "frames_ventricle", "poisson_depth",
    "connection_twist", "inset_faces_refinement_steps", "max_con_sm_iter", "min_con_sm_iter", "sm_reps",
    "con_fade_percentage", "final_sm_iter", "approach", "pos_top", "pos_bot", "pos_septum",
}


def mock_scene(case):
    """Every property build_full_settings() reads, with values unlike CURRENT and the example."""
    return types.SimpleNamespace(
        ventricle_import_dir=str(case / "STL"), ventricle_export_dir=str(case / "export"),
        plot_input_path=str(case / "export"),
        pos_top=(0.1, 0.2, 50.0), pos_bot=(0.0, 0.0, 0.0), pos_septum=(-9.0, 1.0, 40.0),
        last_rotation_time="2026-07-09 17:28:32", last_rotation_angles=(0.1, 0.0, 0.6),
        last_rotation_translation=(-22.0, -15.0, -13.0), pos_top_selected=(22.4, 20.5, 65.9),
        pos_bot_selected=(22.2, 15.0, 13.1), pos_septum_selected=(31.8, 33.4, 60.3),
        valve_R_target=1.25, valve_check_tolerance=3.5,
        translation_mitral=(1.0, -3.0, 48.25), angle_mitral=(5.0, -2.0, 10.0), mitral_width=13.572, mitral_depth=10.44,
        translation_aortic=(-1.0, 5.0, 49.0), angle_aortic=(-12.0, 3.0, 0.0), aortic_diameter=6.811,
        remove_basal_threshold=40.5, mean_reference=False, time_rr=0.8, time_diastole=0.45, frames_ventricle=60,
        poisson_depth=9, connection_twist=2, inset_faces_refinement_steps=3, max_con_sm_iter=30, min_con_sm_iter=5,
        sm_reps=4, con_fade_percentage=25.0, final_sm_iter=12, approach=5,
        live_compare=True, live_delete_temp=True, height_plane=45.0, min_valves=49.0,
        ref_minima=(-12.0, -14.0, 0.0), ref_maxima=(14.0, 8.0, 40.0), reference_object_name="ventricle_3")


def test_round_trip(tmp) -> None:
    print("-- T4.3 round trip: the real writer (schema 3) -> plan_load")
    case = make_case(tmp, "Case_round_trip", LANDMARKS)
    scene = mock_scene(case)
    log = real_session_log().write_full_settings(scene, str(case / "export"), trigger="export", addon_version="2.2")
    data = json.loads(Path(log).read_text(encoding="utf-8"))
    check(data["meta"]["schema_version"] == 3 and data["valves"].get("area_check_tolerance_percent") == 3.5,
          "the export log has schema 3 and the valve area check tolerance")
    expected = {p: getattr(scene, p) for p in CASE_SETTINGS} | dict(zip(("pos_top", "pos_bot", "pos_septum"), LANDMARKS))
    current = expected | {"valve_R_target": 1.3}
    plan = ls.plan_load(log, current, identity)
    check(set(plan.values) == CASE_SETTINGS, f"exactly the 27 case settings are planned (extra "
          f"{sorted(set(plan.values) - CASE_SETTINGS)}, missing {sorted(CASE_SETTINGS - set(plan.values))})")
    wrong = [p for p in CASE_SETTINGS if not same_value(plan.values.get(p), expected[p])]
    check(not wrong, f"every case setting comes back with the value of the mock scene (wrong: {wrong})")
    check(not plan.changes, f"the scene already holds these values: no change for the console ({plan.changes})")
    lines = plan.console_lines()
    check("Changed:" not in lines and lines[-1] == "27 loaded values were already set.",
          f"console: no 'Changed' section, all 27 values already set ({lines[-1]})")
    check(len(plan.kept) == 1 and "R_target" in plan.kept[0] and "1.25" in plan.kept[0] and "1.3" in plan.kept[0],
          f"R_target 1.25 in the log vs. 1.3 in the scene: only a note in 'Kept' ({plan.kept})")
    check("valve_check_tolerance" not in plan.values and "valve_R_target" not in plan.values,
          "the model parameters R_target and the tolerance are never planned")
    check(not plan.notes and ls.TODO_VALVE_SIZES not in plan.todo and "Export log" in " ".join(plan.header)
          and plan.valve_sizes_out_of_date is False,
          "schema 3: no schema 1 note, no 'Calculate valve sizes', valve sizes up to date, kind 'Export log'")


def test_rotation_log(tmp) -> None:
    print("-- T4.4 a rotation log chosen directly")
    case = make_case(tmp, "Case_rotation", LANDMARKS)
    rotation_log = str(case / "STL" / "rotated" / "gvr_rotation.json")
    plan = ls.plan_load(rotation_log, CURRENT | {"ventricle_import_dir": str(case / "STL")}, identity)
    check(plan.values == dict(zip(("pos_top", "pos_bot", "pos_septum"), LANDMARKS)),
          f"only pos_top, pos_bot and pos_septum are planned ({sorted(plan.values)})")
    check("Rotation log" in " ".join(plan.header) and ls.TODO_ROTATE in plan.todo and not plan.kept and not plan.notes,
          "kind 'Rotation log', To do 'Translate and rotate' only after a fresh import, nothing kept, no note")
    check("Case: Case_rotation" in plan.header, f"case named after the folder above <Import>/rotated ({plan.header[0]})")
    check(plan.status() == "Load settings: 3 settings loaded from 'gvr_rotation.json'.",
          f"status line with nothing left out: only the count and the file ({plan.status()})")

    other = make_case(tmp, "Case_rotation_other", LANDMARKS_X)
    plan = ls.plan_load(rotation_log, scene_on(other), identity)  # the scene is on another case
    check(isinstance(plan, ls.LoadPlan), "no question for a rotation log, even in a scene of another case")
    check("Case: Case_rotation" in plan.header and any("current Import folder" in n for n in plan.notes),
          f"the case still comes from the file location, a note says it is not the current Import folder ({plan.notes})")
    try:
        ls.plan_load(rotation_log, scene_on(other), identity, ls.STAY)
        check(False, "a choice for a rotation log is rejected")
    except ValueError as e:
        check(True, f"a choice for a rotation log is a programming error ({e})")


def test_missing_value(tmp) -> None:
    print("-- T4.5 a single value missing (logs of 2026-07-09 have no fade length)")
    data = example_with()
    del data["setup_variables"]["connection_smoothing"]["con_fade_percentage"]
    plan = after_question(write_log(Path(tmp) / "no_fade.json", data), CURRENT, identity, ls.STAY, (ls.STAY,))
    check("con_fade_percentage" not in plan.values and any("con_fade_percentage" in k for k in plan.kept),
          "the fade length keeps its current value and is listed in 'Kept'")
    check(len(plan.values) == 17 and plan.values.get("final_sm_iter") == 10, "the other 17 values are planned")

    print("-- T4.5 a schema 3 log without the MV width")
    case = make_case(tmp, "Case_no_width")
    log = real_session_log().write_full_settings(mock_scene(case), str(case / "export"), trigger="export")
    data = json.loads(Path(log).read_text(encoding="utf-8"))
    del data["valves"]["mitral"]["width_mm"]
    plan = after_question(write_log(Path(log), data), CURRENT, identity, ls.SWITCH)
    check("mitral_width" not in plan.values and plan.values.get("mitral_depth") == 10.44
          and any("width_mm" in k for k in plan.kept) and ls.TODO_VALVE_SIZES in plan.todo,
          "the width is kept, the depth is loaded, To do 'Calculate valve sizes'")


def test_broken_files(tmp) -> None:
    print("-- T4.6 broken or foreign files: an error with a reason, nothing planned")
    poisson_text = example_with()
    poisson_text["setup_variables"]["poisson_depth"] = "10"
    rotation_without_septum = {"meta": {"schema_version": 3, "trigger": "rotate"},
                               "selected_points": {"top": [1.0, 2.0, 3.0], "bot": [0.0, 0.0, 0.0]}}
    cases = [
        ("no JSON", "not json {", "not JSON"),
        ("JSON without meta", json.dumps({"files": {}}), "meta"),
        ("JSON list", "[1, 2]", "meta"),
        ("schema 4", json.dumps(example_with(meta={"schema_version": 4, "trigger": "export"})), "schema version 4"),
        ("unknown kind of log", json.dumps(example_with(meta={"schema_version": 2, "trigger": "manual"})), "manual"),
        ("text instead of a number", json.dumps(poisson_text), "setup_variables.poisson_depth"),
        ("a rotation log without the septum", json.dumps(rotation_without_septum), "selected_points.septum"),
    ]
    for name, text, reason in cases:
        path = Path(tmp) / "broken.json"
        path.write_text(text, encoding="utf-8")
        try:
            ls.plan_load(str(path), CURRENT, identity)
            check(False, f"{name}: rejected")
        except ls.SettingsLogError as e:
            check(reason in str(e), f"{name}: rejected with the reason '{e}'")


def test_unreadable_rotation_log(tmp) -> None:
    print("-- T4.7 an unreadable rotation log next to a valid export log")
    case = make_case(tmp, "Case_bad_rotation")
    write_log(case / "STL" / "rotated" / "gvr_rotation.json", {"meta": "broken"})
    log = real_session_log().write_full_settings(mock_scene(case), str(case / "export"), trigger="export")
    plan = after_question(log, CURRENT, identity, ls.SWITCH)
    check(not {"pos_top", "pos_bot", "pos_septum"} & set(plan.values), "the landmarks are not planned")
    check(any("gvr_rotation.json" in k and "meta" in k for k in plan.kept) and ls.TODO_PICK_LANDMARKS in plan.todo,
          f"'Kept' names the rotation log and why, To do 'Pick the landmarks by hand' ({plan.kept})")
    check(len(plan.values) == 24 and plan.values.get("approach") == 5, "the other 24 case settings are planned")


def test_case_only_from_the_log(tmp) -> None:
    print("-- T4.9 the example log in a scene of another case")
    other = make_case(tmp, "Case_in_scene", LANDMARKS)
    scene_on_other = CURRENT | {"ventricle_import_dir": str(other / "STL")}
    q = ls.plan_load(str(EXAMPLE_LOG), scene_on_other, identity)
    check(getattr(q, "log_case", None) == "example" and getattr(q, "scene_case", None) == "Case_in_scene"
          and getattr(q, "choices", None) == (ls.SWITCH, ls.STAY),
          "the question offers 'Switch to case example' and 'Stay in case Case_in_scene'")
    plan = ls.plan_load(str(EXAMPLE_LOG), scene_on_other, identity, ls.STAY)
    check(not {"pos_top", "pos_bot", "pos_septum"} & set(plan.values) and plan.todo == [ls.TODO_VALVE_SIZES],
          f"'Stay': no landmarks, of neither case, To do only 'Calculate valve sizes' ({plan.todo})")
    check(plan.header[0] == "Case: Case_in_scene (settings from case 'example')", f"the Case line ({plan.header[0]})")

    print("-- T4.9 the Import folder '//' (the default) counts as no Import folder")
    case = make_case(tmp, "Case_blend_folder", LANDMARKS)
    log = write_log(case / "Session_Logs" / "session_default.json", example_with(files={"import_dir": "//"}))
    # '//' resolves to a folder with a rotation log: still no Import folder, so the case of the log is unknown.
    plan = after_question(log, CURRENT, blend_in(case / "STL"), ls.STAY, (ls.STAY,))
    check(not {"pos_top", "pos_bot", "pos_septum"} & set(plan.values) and plan.import_dir is None
          and plan.header[0] == "Case: unknown (no valid Import folder), settings from an unknown case",
          f"only 'Stay': no landmarks, case unknown, no button ({plan.header[0]})")


A_CASE = "C:\\Daten\\A\\Faelle\\CaseA"  # where person A keeps the case: not on this computer
LANDMARKS_X = ((9.0, 9.0, 9.0), (8.0, 8.0, 8.0), (7.0, 7.0, 7.0))  # landmarks of the scene's case CaseX


def foreign_log(path, trigger="session_close", **folders):
    """A schema 3 log written by the real writer on person A's computer: A's Windows folders, mock_scene values."""
    scene = mock_scene(Path("unused"))
    scene.ventricle_import_dir, scene.ventricle_export_dir, scene.plot_input_path = (
        A_CASE + "\\STL\\", A_CASE + "\\export\\", A_CASE + "\\export\\")
    for name, value in folders.items():
        setattr(scene, name, value)
    return write_log(Path(path), real_session_log().build_full_settings(scene, trigger=trigger))


def scene_on(case):
    """CURRENT with the Import folder of a case (a real folder of the test)."""
    return CURRENT | {"ventricle_import_dir": str(case / "STL"), "ventricle_export_dir": str(case / "export"),
                      "plot_input_path": str(case / "export")}


def folder(path) -> str:
    """A derived folder as Blender's folder browser writes it: native separators and a trailing one."""
    return str(path) + os.sep


def test_moved_case_session_log(tmp) -> None:
    print("-- T4.10 moved case, session log: the folders are derived from where the log lies")
    case = make_case(Path(tmp) / "moved", "CaseA", LANDMARKS)
    other = make_case(Path(tmp) / "moved", "CaseX", LANDMARKS_X)
    log = foreign_log(case / "Session_Logs" / "session_2026-11-20_14-00-00.json")
    q = ls.plan_load(log, scene_on(other), identity)
    check(isinstance(q, ls.Question), f"without a choice: the question, nothing planned ({type(q).__name__})")
    if not isinstance(q, ls.Question):
        return
    check(q.log_case == "CaseA" and q.scene_case == "CaseX" and q.kind == "Session log",
          f"the question names the case of the log and of the scene ({q.log_case}, {q.scene_case})")
    check(q.log_folders == [("Import folder", A_CASE + "\\STL\\", False), ("Export folder", A_CASE + "\\export\\", False),
                            ("Processed geometries", A_CASE + "\\export\\", False)],
          f"the folders as in the log, none found on this computer ({q.log_folders})")
    check(q.derived and q.log_case_dir == str(case) and q.reason is None,
          f"derived: the case folder is where the log lies ({q.log_case_dir})")
    check(q.switch_folders == [("Import folder", folder(case / "STL")), ("Export folder", folder(case / "export")),
                               ("Processed geometries", folder(case / "export"))],
          f"the folders after 'Switch' are in the case folder on this computer ({q.switch_folders})")
    check(q.choices == (ls.SWITCH, ls.STAY) and q.default == ls.SWITCH, "both choices, 'Switch' preselected")
    lines = q.lines()
    check(lines[0] == "'session_2026-11-20_14-00-00.json' is a session log of case 'CaseA'."
          and "Its folders are not on this computer:" in lines and f"  Import folder  {A_CASE}\\STL\\" in lines
          and f"The log lies in the case folder {case} (same case name)." in lines,
          f"the window says why it asks ({lines})")

    plan = ls.plan_load(log, scene_on(other), identity, ls.SWITCH)
    v = plan.values
    check(v.get("ventricle_import_dir") == folder(case / "STL") and v.get("ventricle_export_dir") == folder(case / "export")
          and v.get("plot_input_path") == folder(case / "export"), "'Switch': the derived folders are planned")
    check((v.get("pos_top"), v.get("pos_bot"), v.get("pos_septum")) == LANDMARKS and ls.TODO_ROTATE in plan.todo,
          "the landmarks come from the rotation log in the derived Import folder")
    check(plan.header[0] == f"Case: CaseA (folders in {case})" and plan.import_dir == str(case / "STL"),
          f"the Case line names the case folder on this computer ({plan.header[0]})")
    check(not any("files." in k or "plot_input_path" in k for k in plan.kept) and len(v) == 27,
          f"no folder kept, all 27 case settings planned ({plan.kept}, {len(v)})")


def test_moved_case_export_log(tmp) -> None:
    print("-- T4.12 moved case, export log: its folder is the Export folder")
    case = make_case(Path(tmp) / "moved_export", "CaseA", LANDMARKS)
    other = make_case(Path(tmp) / "moved_export", "CaseX", LANDMARKS_X)
    log = foreign_log(case / "export" / "gvr_settings.json", trigger="export")
    q = ls.plan_load(log, scene_on(other), identity)
    check(getattr(q, "derived", None) and q.log_case_dir == str(case) and q.choices == (ls.SWITCH, ls.STAY)
          and q.kind == "Export log", f"derived from the export log: case folder {getattr(q, 'log_case_dir', None)}")
    check(q.lines()[0] == "'gvr_settings.json' is an export log of case 'CaseA'.", f"the first line ({q.lines()[0]})")
    plan = ls.plan_load(log, scene_on(other), identity, ls.SWITCH)
    v = plan.values
    check(v.get("ventricle_export_dir") == folder(case / "export") and v.get("ventricle_import_dir") == folder(case / "STL")
          and v.get("plot_input_path") == folder(case / "export") and v.get("pos_top") == LANDMARKS[0],
          "'Switch': Export folder where the log lies, Import folder and Processed geometries derived, landmarks")

    print("-- T4.12 the Export folder is the case folder itself")
    case = make_case(Path(tmp) / "export_is_case", "CaseA", LANDMARKS)
    for log in (foreign_log(case / "Session_Logs" / "session_2026-11-20_14-00-00.json", ventricle_export_dir=A_CASE + "\\",
                            plot_input_path=A_CASE),
                foreign_log(case / "gvr_settings.json", trigger="export", ventricle_export_dir=A_CASE + "\\",
                            plot_input_path=A_CASE)):
        plan = after_question(log, scene_on(other), identity, ls.SWITCH)
        check(plan.values.get("ventricle_export_dir") == folder(case) and plan.values.get("plot_input_path") == folder(case)
              and plan.values.get("ventricle_import_dir") == folder(case / "STL"),
              f"{Path(log).name}: Export folder and Processed geometries are the case folder ({plan.values.get('ventricle_export_dir')})")


def test_no_switch(tmp) -> None:
    print("-- T4.13 another case name where the log lies: no 'Switch'")
    case = make_case(Path(tmp) / "no_switch", "CaseA_B", LANDMARKS)
    log = foreign_log(case / "Session_Logs" / "session_2026-11-20_14-00-00.json")
    q = ls.plan_load(log, CURRENT, identity)
    check(getattr(q, "choices", None) == (ls.STAY,) and q.default == ls.STAY and q.switch_folders == []
          and not q.derived and q.log_case == "CaseA",
          f"only 'Stay', preselected, no folders for 'Switch' ({getattr(q, 'choices', None)})")
    check(getattr(q, "reason", None) == "The log is from case 'CaseA' but lies in 'CaseA_B': rename the folder or set "
          "the folders by hand.", f"the question says why ({getattr(q, 'reason', None)})")

    print("-- T4.13 the case folder where the log lies has no Import folder")
    case = Path(tmp) / "no_switch" / "CaseA"
    log = foreign_log(case / "Session_Logs" / "session_2026-11-20_14-00-00.json")
    q = ls.plan_load(log, CURRENT, identity)
    check(getattr(q, "choices", None) == (ls.STAY,) and "STL" in (q.reason or ""),
          f"only 'Stay', the reason names the missing Import folder ({getattr(q, 'reason', None)})")

    print("-- T4.13 the Import folder '//' in the log: unknown case")
    case = make_case(Path(tmp) / "no_switch", "CaseC", LANDMARKS)
    log = foreign_log(case / "Session_Logs" / "session_2026-11-20_14-00-00.json", ventricle_import_dir="//")
    q = ls.plan_load(log, scene_on(case), identity)
    check(getattr(q, "choices", None) == (ls.STAY,) and q.log_case is None
          and q.lines()[0] == "'session_2026-11-20_14-00-00.json' is a session log of an unknown case (no case folder in "
          "the log).", f"only 'Stay', the case of the log is unknown ({q.lines()[0] if q.choices else q})")
    plan = ls.plan_load(log, scene_on(case), identity, ls.STAY)
    check(plan.header[0] == "Case: CaseC (settings from an unknown case)", f"the Case line ({plan.header[0]})")

    print("-- T4.13 a choice that was not offered")
    try:
        ls.plan_load(log, scene_on(case), identity, ls.SWITCH)
        check(False, "'Switch' without the offer is rejected")
    except ls.SettingsLogError:
        check(False, "'Switch' without the offer is a programming error, not a broken file")
    except ValueError as e:
        check("SWITCH" in str(e), f"'Switch' without the offer is a programming error ({e})")


def test_switch_folders(tmp) -> None:
    print("-- T4.14 'Switch' with an Export folder outside the case folder that is not on this computer")
    case = make_case(Path(tmp) / "outside", "CaseA", LANDMARKS)
    other = make_case(Path(tmp) / "outside", "CaseX", LANDMARKS_X)
    outside = "E:\\Exporte\\CaseA_A4\\"
    log = foreign_log(case / "Session_Logs" / "session_2026-11-20_14-00-00.json", ventricle_export_dir=outside,
                      plot_input_path=outside)
    plan = after_question(log, scene_on(other), identity, ls.SWITCH)
    check(plan.values.get("ventricle_export_dir") == "//" and plan.values.get("plot_input_path") == "//",
          f"Export folder and Processed geometries become '//', the scene's folders never stay "
          f"({plan.values.get('ventricle_export_dir')}, {plan.values.get('plot_input_path')})")
    check("Set the Export folder" in plan.todo
          and "Set the Processed geometries folder (empty: the Export folder is used)" in plan.todo
          and not any("files." in k or "plot_input_path" in k for k in plan.kept),
          f"To do: set both folders, nothing kept ({plan.todo})")

    print("-- T4.14 an empty Processed geometries folder is not missing: it means the Export folder")
    log = foreign_log(case / "Session_Logs" / "session_2026-11-20_14-00-00.json", plot_input_path="")
    q = ls.plan_load(log, scene_on(other), identity)
    check(q.derived and not any(line.startswith("  Processed geometries") for line in q.lines())
          and ("Processed geometries", "") in q.switch_folders,
          f"not listed under 'not on this computer', kept empty after 'Switch' ({q.lines()})")

    print("-- T4.14 an Export folder in the case folder that does not exist yet")
    log = foreign_log(case / "Session_Logs" / "session_2026-11-20_14-00-00.json",
                      ventricle_export_dir=A_CASE + "\\results\\")
    plan = after_question(log, scene_on(other), identity, ls.SWITCH)
    check(plan.values.get("ventricle_export_dir") == folder(case / "results") and not (case / "results").exists(),
          f"derived: it is set although it does not exist ({plan.values.get('ventricle_export_dir')})")
    log = foreign_log(case / "Session_Logs" / "session_2026-11-20_14-00-00.json", ventricle_import_dir=str(case / "STL"),
                      ventricle_export_dir=str(case / "results"), plot_input_path=str(case / "results"))
    plan = after_question(log, scene_on(other), identity, ls.SWITCH)
    check(plan.values.get("ventricle_export_dir") == folder(case / "results")
          and plan.values.get("plot_input_path") == folder(case / "results")
          and not any(t.startswith("Set the") for t in plan.todo),
          f"Import folder found: it is set as well ({plan.values.get('ventricle_export_dir')})")

    print("-- T4.14 a '//..' folder leaves the case folder, also when the .blend lies in it")
    log = foreign_log(case / "Session_Logs" / "session_2026-11-20_14-00-00.json", ventricle_import_dir="//STL/",
                      ventricle_export_dir="//../exports/CaseA/", plot_input_path="")
    plan = after_question(log, scene_on(other), blend_in(case), ls.SWITCH)
    check(plan.values.get("ventricle_export_dir") == "//" and plan.todo.count("Set the Export folder") == 1,
          f"'Switch': outside, so '//' and To do ({plan.values.get('ventricle_export_dir')}, {plan.todo})")
    check(plan.values.get("plot_input_path") == "" and not any("Processed geometries" in t for t in plan.todo),
          f"an empty Processed geometries folder means the Export folder: set as in the log, no To do ({plan.todo})")
    plan = ls.plan_load(log, scene_on(case), blend_in(case))
    check("ventricle_export_dir" not in plan.values and "files.export_dir: folder '//../exports/CaseA/' not found" in plan.kept,
          f"the same case: the scene's Export folder stays ({plan.kept})")

    print("-- T4.14 the scene has the case of the log already: no question")
    log = foreign_log(case / "Session_Logs" / "session_2026-11-20_14-00-00.json", ventricle_import_dir=str(case / "STL"),
                      ventricle_export_dir=str(case / "export"), plot_input_path=str(case / "export"))
    plan = ls.plan_load(log, scene_on(case), identity)
    check(isinstance(plan, ls.LoadPlan) and plan.values.get("ventricle_import_dir") == str(case / "STL")
          and plan.values.get("ventricle_export_dir") == str(case / "export") and plan.header[0] == "Case: CaseA"
          and plan.values.get("pos_top") == LANDMARKS[0],
          f"found folders: as written in the log, 'Case: CaseA' ({getattr(plan, 'header', [plan])[0]})")
    log = foreign_log(case / "Session_Logs" / "session_2026-11-20_14-00-00.json")
    plan = ls.plan_load(log, scene_on(case), identity)
    check(isinstance(plan, ls.LoadPlan) and plan.values.get("ventricle_import_dir") == folder(case / "STL")
          and plan.values.get("ventricle_export_dir") == folder(case / "export") and plan.header[0] == "Case: CaseA"
          and plan.values.get("pos_top") == LANDMARKS[0],
          f"derived folders: the derived folders, the landmarks, 'Case: CaseA' ({getattr(plan, 'header', [plan])[0]})")

    print("-- T4.14 the same case and an Export folder outside it that is not found")
    log = foreign_log(case / "Session_Logs" / "session_2026-11-20_14-00-00.json", ventricle_export_dir=outside)
    plan = ls.plan_load(log, scene_on(case), identity)
    check(isinstance(plan, ls.LoadPlan) and "ventricle_export_dir" not in plan.values
          and f"files.export_dir: folder '{outside}' not found" in plan.kept and "Set the Export folder" not in plan.todo,
          f"the scene's Export folder stays and is listed under 'Kept' ({getattr(plan, 'kept', plan)})")


def test_blend_folder(tmp) -> None:
    print("-- T4.15 a derived folder under the .blend is planned as '//...'")
    blend_dir = Path(tmp) / "blend"
    case = make_case(blend_dir / "Faelle", "CaseA", LANDMARKS)
    log = foreign_log(case / "Session_Logs" / "session_2026-11-20_14-00-00.json")
    plan = after_question(log, CURRENT, blend_in(blend_dir), ls.SWITCH)
    check(plan.values.get("ventricle_import_dir") == "//" + os.path.join("Faelle", "CaseA", "STL", "")
          and plan.values.get("ventricle_export_dir") == "//" + os.path.join("Faelle", "CaseA", "export", "")
          and plan.values.get("pos_top") == LANDMARKS[0],
          f"'//Faelle/CaseA/STL/' and the landmarks of its rotation log ({plan.values.get('ventricle_import_dir')})")

    print("-- T4.15 an unsaved .blend: '//' resolves to '' and the derived folders are absolute")
    plan = after_question(log, CURRENT, lambda p: p[2:] if p.startswith("//") else p, ls.SWITCH)
    check(plan.values.get("ventricle_import_dir") == folder(case / "STL"),
          f"absolute ({plan.values.get('ventricle_import_dir')})")


def test_choice_labels(tmp) -> None:
    print("-- T4.16 a scene without a case (Import folder '//') and found folders: the question comes")
    case = make_case(Path(tmp) / "labels", "CaseA", LANDMARKS)
    log = foreign_log(case / "Session_Logs" / "session_2026-11-20_14-00-00.json", ventricle_import_dir=str(case / "STL"),
                      ventricle_export_dir=str(case / "export"), plot_input_path=str(case / "export"))
    no_case = CURRENT | {"ventricle_import_dir": "//"}
    q = ls.plan_load(log, no_case, blend_in(case / "STL"))
    check(isinstance(q, ls.Question) and q.scene_case is None and q.choices == (ls.SWITCH, ls.STAY)
          and not q.derived and q.label(ls.SWITCH) == "Switch to case 'CaseA'"
          and q.label(ls.STAY) == "Keep the folders of this scene (no case set)",
          f"'Switch to case CaseA' and 'Keep the folders of this scene' ({q.label(ls.STAY) if q.choices else q})")
    plan = ls.plan_load(log, no_case, blend_in(case / "STL"), ls.STAY)
    check(plan.header[0] == "Case: unknown (no valid Import folder), settings from case 'CaseA'"
          and plan.import_dir is None, f"'Stay' without a case: {plan.header[0]}")

    print("-- T4.16 the same case name on both sides: the labels name the case folders")
    old = make_case(Path(tmp) / "labels_old", "CaseA", LANDMARKS_X)
    q = ls.plan_load(log, scene_on(old), identity)
    check(q.label(ls.SWITCH) == f"Switch to case 'CaseA' ({case})" and q.label(ls.STAY) == f"Stay in case 'CaseA' ({old})",
          f"({q.label(ls.SWITCH)}; {q.label(ls.STAY)})")
    other = make_case(Path(tmp) / "labels", "CaseX", LANDMARKS_X)
    q = ls.plan_load(log, scene_on(other), identity)
    check(q.label(ls.SWITCH) == "Switch to case 'CaseA'" and q.label(ls.STAY) == "Stay in case 'CaseX'",
          f"other names: no folders ({q.label(ls.SWITCH)}; {q.label(ls.STAY)})")


def derived_as_case_a(tmp, name, import_dir, export_dir, abspath=identity) -> None:
    """A session log with these folders of person A in a moved CaseA: the case is derived."""
    case = make_case(Path(tmp) / name, "CaseA", LANDMARKS)
    log = foreign_log(case / "Session_Logs" / "session_2026-11-20_14-00-00.json", ventricle_import_dir=import_dir,
                      ventricle_export_dir=export_dir, plot_input_path=export_dir)
    q = ls.plan_load(log, CURRENT, abspath)
    check(getattr(q, "log_case", None) == "CaseA" and q.derived
          and q.switch_folders == [("Import folder", folder(case / "STL")), ("Export folder", folder(case / "export")),
                                   ("Processed geometries", folder(case / "export"))],
          f"{import_dir}: case 'CaseA', derived ({getattr(q, 'switch_folders', q)})")


def test_path_forms(tmp) -> None:
    print("-- T4.17 UNC folders in the log (a network drive of person A)")
    share = "\\\\127.0.0.1\\gvr_no_share"  # fails fast on Windows: no name lookup, no share of that name
    start = time.perf_counter()
    os.path.isdir(share + "\\Faelle\\CaseA\\STL")
    took = time.perf_counter() - start
    check(took < 5, f"the check 'folder exists?' for this UNC folder takes {took:.2f} s")
    derived_as_case_a(tmp, "unc", share + "\\Faelle\\CaseA\\STL\\", share + "\\Faelle\\CaseA\\export\\")

    print("-- T4.17 folders relative to the .blend of person A, not resolved on this computer")
    elsewhere = blend_in(Path(tmp) / "elsewhere" / "blend")
    for i, base in enumerate(("//..\\CaseA\\", "//CaseA\\", "//..\\..\\Faelle\\CaseA\\")):
        derived_as_case_a(tmp, f"relative{i}", base + "STL\\", base + "export\\", elsewhere)


STAY_VALUES = {prop for prop, _key, _kind in ls.SETTINGS}  # valve positions and angles, 13 setup variables, approach


def test_stay(tmp) -> None:
    print("-- T4.11 'Stay': the settings of another case as a starting point")
    case = make_case(Path(tmp) / "stay", "CaseA", LANDMARKS)
    other = make_case(Path(tmp) / "stay", "CaseX", LANDMARKS_X)
    log = foreign_log(case / "Session_Logs" / "session_2026-11-20_14-00-00.json")
    data = json.loads(Path(log).read_text(encoding="utf-8"))
    del data["valves"]["mitral"]["width_mm"]
    write_log(Path(log), data)
    plan = after_question(log, scene_on(other), identity, ls.STAY)
    v = plan.values
    check(set(v) == STAY_VALUES, f"only valve positions and angles, 13 setup variables and approach "
          f"(extra {sorted(set(v) - STAY_VALUES)}, missing {sorted(STAY_VALUES - set(v))})")
    check(v.get("translation_mitral") == (1.0, -3.0, 48.25) and v.get("angle_aortic") == (-12.0, 3.0, 0.0)
          and same(v.get("time_rr"), 0.8) and v.get("frames_ventricle") == 60 and v.get("approach") == 5,
          "the values come from the log")
    check(plan.todo == [ls.TODO_VALVE_SIZES] and plan.valve_sizes_out_of_date and plan.import_dir == str(other / "STL"),
          f"To do exactly 'Calculate valve sizes', the button uses the scene's Import folder ({plan.todo}, {plan.import_dir})")
    check(not any("files." in k or "plot_input_path" in k or "width_mm" in k or "Landmarks" in k for k in plan.kept)
          and len(plan.kept) == 1 and "R_target" in plan.kept[0],
          f"'Kept': no folder, no missing valve size, only the deviating R_target ({plan.kept})")
    check(plan.header[0] == "Case: CaseX (settings from case 'CaseA')", f"the Case line names both cases ({plan.header[0]})")
    status = plan.status()
    check("valve sizes out of date" in status and "landmarks" not in status,
          f"the status line names the valve sizes, not the landmarks ({status})")


def registered_properties() -> set:
    """Keys of scene_properties in the main addon, read from its source without Blender."""
    tree = ast.parse((REPO / "addons/geometrical_heart_reconstruction.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "scene_properties" for t in node.targets):
            return {k.value for k in node.value.keys}
    return set()


def test_property_names() -> None:
    print("-- T4.8 property names")
    registered = registered_properties()
    # T4.3 shows that plan_load plans exactly CASE_SETTINGS; it also reads valve_R_target from the scene.
    unknown = sorted((CASE_SETTINGS | {"valve_R_target"}) - registered)
    check(len(registered) > 30 and not unknown, f"every planned property is registered in the addon (unknown: {unknown})")


def same_value(a, b) -> bool:
    if isinstance(b, tuple):
        return isinstance(a, tuple) and len(a) == len(b) and all(map(same, a, b))
    return type(a) is type(b) and same(a, b)


def main() -> int:
    test_example_log()
    with tempfile.TemporaryDirectory() as tmp:
        test_example_in_case(tmp)
        test_round_trip(tmp)
        test_rotation_log(tmp)
        test_missing_value(tmp)
        test_broken_files(tmp)
        test_unreadable_rotation_log(tmp)
        test_case_only_from_the_log(tmp)
        test_moved_case_session_log(tmp)
        test_stay(tmp)
        test_moved_case_export_log(tmp)
        test_no_switch(tmp)
        test_switch_folders(tmp)
        test_blend_folder(tmp)
        test_choice_labels(tmp)
        test_path_forms(tmp)
    test_property_names()
    print()
    print(f"{len(FAILURES)} failure(s)" if FAILURES else "all checks passed")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
