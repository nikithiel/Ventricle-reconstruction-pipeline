#!/usr/bin/env python3
"""T4: tests for addons/load_settings.py ('Load settings'), runnable without Blender.

  python tools/test_load_settings.py

The planning function plan_load() reads an export, session or rotation log and returns the
scene properties it would set plus the notes for the 'Settings loaded' window. Expected values
are the numbers written literally in the logs or the values of the mock scene.
"""
from __future__ import annotations

import ast
import json
import sys
import tempfile
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
    print("-- T4.1 example session log (schema 1), unchanged")
    plan = ls.plan_load(str(EXAMPLE_LOG), CURRENT, identity)
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
    check(not (NEVER | {"pos_top", "pos_bot", "pos_septum"}) & set(v),
          "nothing from internal, position, rotation and live_*")
    folders = ("ventricle_import_dir", "ventricle_export_dir", "plot_input_path")
    kept = "\n".join(plan.kept)
    check(not set(folders) & set(v) and "files.import_dir" in kept and "files.export_dir" in kept
          and "comparison.plot_input_path" in kept and "dataset\\example\\STL\\" in kept,
          "the shortened example folders cannot be resolved and are kept")
    check(len(v) == 18, f"18 values: 4 valve vectors, 13 setup variables, approach (got {len(v)})")
    check(ls.TODO_PICK_LANDMARKS in plan.todo, "no rotation log: 'Pick the landmarks by hand'")
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


def test_example_in_case(tmp) -> None:
    print("-- T4.2 example log in a case folder with real folders and a rotation log")
    case = make_case(tmp, "Case_P15", LANDMARKS)
    log = write_log(case / "Session_Logs" / "session_x.json", example_with(
        files={"import_dir": "//STL/", "export_dir": "//export/"},
        comparison={"plot_input_path": "//export/", "live_compare": True, "live_delete_temp": True}))
    plan = ls.plan_load(log, CURRENT, blend_in(case))
    v = plan.values
    check(v.get("ventricle_import_dir") == "//STL/" and v.get("ventricle_export_dir") == "//export/"
          and v.get("plot_input_path") == "//export/", "the folders exist and are set as written in the log")
    check(v.get("pos_top") == LANDMARKS[0] and v.get("pos_bot") == LANDMARKS[1] and v.get("pos_septum") == LANDMARKS[2],
          "landmarks from STL/rotated/gvr_rotation.json, not from the rotation part of the log")
    check(ls.TODO_ROTATE in plan.todo and ls.TODO_PICK_LANDMARKS not in plan.todo,
          "To do: 'Translate and rotate' only after a fresh import")
    check("Case: Case_P15" in plan.header and len(v) == 24 and not plan.kept and not NEVER & set(v),
          f"case named after the case folder, 24 values, nothing kept (got {len(v)}, kept {plan.kept})")


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
    check(len(plan.kept) == 1 and "R_target" in plan.kept[0] and "1.25" in plan.kept[0] and "1.3" in plan.kept[0],
          f"R_target 1.25 in the log vs. 1.3 in the scene: only a note in 'Kept' ({plan.kept})")
    check("valve_check_tolerance" not in plan.values and "valve_R_target" not in plan.values,
          "the model parameters R_target and the tolerance are never planned")
    check(not plan.notes and ls.TODO_VALVE_SIZES not in plan.todo and "Export log" in " ".join(plan.header),
          "schema 3: no schema 1 note, no 'Calculate valve sizes', kind 'Export log'")


def test_rotation_log(tmp) -> None:
    print("-- T4.4 a rotation log chosen directly")
    case = make_case(tmp, "Case_rotation", LANDMARKS)
    plan = ls.plan_load(str(case / "STL" / "rotated" / "gvr_rotation.json"), CURRENT, identity)
    check(plan.values == dict(zip(("pos_top", "pos_bot", "pos_septum"), LANDMARKS)),
          f"only pos_top, pos_bot and pos_septum are planned ({sorted(plan.values)})")
    check("Rotation log" in " ".join(plan.header) and ls.TODO_ROTATE in plan.todo and not plan.kept and not plan.notes,
          "kind 'Rotation log', To do 'Translate and rotate' only after a fresh import, nothing kept")


def test_missing_value(tmp) -> None:
    print("-- T4.5 a single value missing (logs of 2026-07-09 have no fade length)")
    data = example_with()
    del data["setup_variables"]["connection_smoothing"]["con_fade_percentage"]
    plan = ls.plan_load(write_log(Path(tmp) / "no_fade.json", data), CURRENT, identity)
    check("con_fade_percentage" not in plan.values and any("con_fade_percentage" in k for k in plan.kept),
          "the fade length keeps its current value and is listed in 'Kept'")
    check(len(plan.values) == 17 and plan.values.get("final_sm_iter") == 10, "the other 17 values are planned")

    print("-- T4.5 a schema 3 log without the MV width")
    case = make_case(tmp, "Case_no_width")
    log = real_session_log().write_full_settings(mock_scene(case), str(case / "export"), trigger="export")
    data = json.loads(Path(log).read_text(encoding="utf-8"))
    del data["valves"]["mitral"]["width_mm"]
    plan = ls.plan_load(write_log(Path(log), data), CURRENT, identity)
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
    plan = ls.plan_load(log, CURRENT, identity)
    check(not {"pos_top", "pos_bot", "pos_septum"} & set(plan.values), "the landmarks are not planned")
    check(any("gvr_rotation.json" in k and "meta" in k for k in plan.kept) and ls.TODO_PICK_LANDMARKS in plan.todo,
          f"'Kept' names the rotation log and why, To do 'Pick the landmarks by hand' ({plan.kept})")
    check(len(plan.values) == 24 and plan.values.get("approach") == 5, "the other 24 case settings are planned")


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
    test_property_names()
    print()
    print(f"{len(FAILURES)} failure(s)" if FAILURES else "all checks passed")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
