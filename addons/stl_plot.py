# interpolation of 'surface_'-files for improved temporal discretization
# requires folder 'PTS' containing the 'surface_'-files created by stl2pts

import os
import csv
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from scipy.interpolate import pchip
from scipy.interpolate import interp1d

# Borrowed from the other script
def parseRunTimeVariables_unique(inputpath):
    runtimeTree = {}        
    # If no matching files are found, then return exception
    dictData = dict(np.genfromtxt(inputpath, dtype=str))

    print("\n---------------------------------")
    print("---------------------------------")

    if "correlationFile" in dictData.keys():
        runtimeTree["correlationFile"] = int(dictData["correlationFile"])

        if runtimeTree["correlationFile"] == 1:
            print("Correlation file will be created!")
        else:
            print("No correlation file will be created!")
    else:
        runtimeTree["correlationFile"] = 1
        print("Correlation file will be created!")
    if "refFrameID" in dictData.keys():
        runtimeTree["refFrameID"] = int(dictData["refFrameID"])
        print("Reference frame:", runtimeTree["refFrameID"])
    else:
        runtimeTree["refFrameID"] = 0
        print("Reference frame:", runtimeTree["refFrameID"])
    if "startFrameID" in dictData.keys():
        runtimeTree["startFrameID"] = int(dictData["startFrameID"])
        print("Starting frame:", runtimeTree["startFrameID"])
    else:
        runtimeTree["startFrameID"] = 0
        print("Starting frame:", runtimeTree["startFrameID"])
    if "endFrameID" in dictData.keys():
        runtimeTree["endFrameID"] = int(dictData["endFrameID"])
        print("End frame:", runtimeTree["endFrameID"])
    else:
        runtimeTree["endFrameID"] = 11
        print("End frame:", runtimeTree["endFrameID"])
    if "numberFrames" in dictData.keys():
        runtimeTree["numberFrames"] = int(dictData["numberFrames"])
        print("Number of frames:", runtimeTree["numberFrames"])
    else:
        runtimeTree["numberFrames"] = 12
        print("Number of frames:", runtimeTree["numberFrames"])
    if "numberInterm" in dictData.keys():
        runtimeTree["numberInterm"] = int(dictData["numberInterm"])
        print("Number of intermediate frames:", runtimeTree["numberInterm"])
    else:
        runtimeTree["numberInterm"] = 20
        print("Number of intermediate frames:", runtimeTree["numberInterm"])
    if "interMethod" in dictData.keys():
        runtimeTree["interMethod"] = int(dictData["interMethod"])
        print("Interpolation method chosen:", runtimeTree["interMethod"])
    else:
        runtimeTree["interMethod"] = 4
        print("Interpolation method chosen:", runtimeTree["interMethod"])

    print("---------------------------------")
    print("---------------------------------\n")

    return (
        runtimeTree["correlationFile"],
        runtimeTree["refFrameID"],
        runtimeTree["startFrameID"],
        runtimeTree["endFrameID"],
        runtimeTree["numberFrames"],
        runtimeTree["numberInterm"],
        runtimeTree["interMethod"],
    )

def _connectivity_errors(conn_dir, start_frame_id, end_frame_id):
    """Return list of error strings for missing connectivity files (empty if all present)."""
    errors = []
    if not os.path.isdir(conn_dir):
        errors.append("Connectivity directory '{}' does not exist.".format(conn_dir))
        return errors
    faces_path = os.path.join(conn_dir, 'ventricle_faces.txt')
    if not os.path.isfile(faces_path):
        errors.append("Missing connectivity file: {}".format(faces_path))
    missing_verts = []
    for fid in range(start_frame_id, end_frame_id + 1):
        vpath = os.path.join(conn_dir, 'ventricle_verts_{}.txt'.format(fid))
        if not os.path.isfile(vpath):
            missing_verts.append(vpath)
    if missing_verts:
        errors.append("Missing ventricle_verts_*.txt connectivity files:\n  " + "\n  ".join(missing_verts))
    return errors

def check_stl_and_connectivity(inputPath, num_frames, start_frame_id, end_frame_id):
    """
    Pre-check that all necessary STL and connectivity files exist.

    Conditions:
      - STL/ventricle_<i>.stl exist for all i = start_frame_id .. end_frame_id
      - numberFrames from inputPython.txt matches (end_frame_id - start_frame_id + 1)
      - aorta, atrium, av_perm, av_res, mv_perm, mv_res STL files exist
      - STL/Connectivity/ventricle_faces.txt exists
      - STL/Connectivity/ventricle_verts_<i>.txt exist for all frames
    """
    print("\n--- Pre-check: STL & Connectivity files ---")
    errors = []

    stl_dir = inputPath
    if not os.path.isdir(stl_dir):
        errors.append("STL directory '{}' does not exist.".format(stl_dir))
    else:
        # 1) Single geometry STLs
        required_single = ['aorta', 'atrium', 'av_perm', 'av_res', 'mv_perm', 'mv_res']
        for name in required_single:
            path = os.path.join(stl_dir, name + '.stl')
            if not os.path.isfile(path):
                errors.append("Missing STL: {}".format(path))

        # 2) Ventricle STL sequence
        expected_count = end_frame_id - start_frame_id + 1
        if expected_count != num_frames:
            errors.append(
                "Frame count mismatch: numberFrames={} but startFrameID={} and endFrameID={} "
                "imply {} frames.".format(num_frames, start_frame_id, end_frame_id, expected_count)
            )

        missing_vent = []
        for fid in range(start_frame_id, end_frame_id + 1):
            p = os.path.join(stl_dir, 'ventricle_{}.stl'.format(fid))
            if not os.path.isfile(p):
                missing_vent.append(p)
        if missing_vent:
            errors.append(
                "Missing ventricle STL files:\n  " + "\n  ".join(missing_vent)
            )

        # 3) Connectivity
        conn_dir = os.path.join(stl_dir, 'Connectivity')
        errors.extend(_connectivity_errors(conn_dir, start_frame_id, end_frame_id))

    if errors:
        msg = "\n".join("- " + e for e in errors)
        raise RuntimeError("STL/connectivity pre-check failed:\n" + msg)
    else:
        print("STL/connectivity pre-check passed.")
        
    return 

# From original script
def remove_if_exists(p):
    p = Path(p)
    try:
        if p.exists():
            p.unlink()
    except Exception as e:
        print(f"WARNING: Could not delete existing file {p}: {e}")

# linear, quadratic, cubic or monotonic cubic spline interpolation between points 
def itpl(PTS, numPTS, degree, sID, eID):
    t = range(len(PTS))
    ipl_t = np.linspace(sID, eID, numPTS * (eID - sID) // (len(PTS) - 1) + 1)

    if degree == 1:
        newX = interp1d (t, [row[0] for row in PTS], kind='slinear') (ipl_t)
        newY = interp1d (t, [row[1] for row in PTS], kind='slinear') (ipl_t)
        newZ = interp1d (t, [row[2] for row in PTS], kind='slinear') (ipl_t)
    elif degree == 2:
        newX = interp1d (t, [row[0] for row in PTS], kind='quadratic') (ipl_t)
        newY = interp1d (t, [row[1] for row in PTS], kind='quadratic') (ipl_t)
        newZ = interp1d (t, [row[2] for row in PTS], kind='quadratic') (ipl_t)
    elif degree == 3:
        newX = interp1d (t, [row[0] for row in PTS], kind='cubic') (ipl_t)
        newY = interp1d (t, [row[1] for row in PTS], kind='cubic') (ipl_t)
        newZ = interp1d (t, [row[2] for row in PTS], kind='cubic') (ipl_t)
    elif degree == 4:
        newX = pchip (t, [row[0] for row in PTS]) (ipl_t)
        newY = pchip (t, [row[1] for row in PTS]) (ipl_t)
        newZ = pchip (t, [row[2] for row in PTS]) (ipl_t)
    else:
        newX = interp1d (t, [row[0] for row in PTS], kind='quadratic') (ipl_t)
        newY = interp1d (t, [row[1] for row in PTS], kind='quadratic') (ipl_t)
        newZ = interp1d (t, [row[2] for row in PTS], kind='quadratic') (ipl_t)

    outPTS = []
    for i in range(0, len(newX)):
        outPTS.append([newX[i], newY[i], newZ[i]])

    return outPTS

def update_exeDynMesh(file_path, new_values):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    for i, line in enumerate(lines):
        if line.startswith("#define ALL_PTS_NUM"):
            # Update the line with the new value
            lines[i] = f"#define ALL_PTS_NUM {new_values[0]}     /* total number of nodes on moving boundaries: output of ps_detNPts */\n"
        if line.startswith("#define N_TIME_STEPS"):
            # Update the line with the new value
            lines[i] = f"#define N_TIME_STEPS {new_values[1]}     /* total number of frames after interpolation: output of ps_intNPts */\n"
    
    with open(file_path, 'w') as file:
        file.writelines(lines)
        
    print("udf_exeDynMesh.c updated!")
    
def interpolationMain(numFrame, numInter, k):
    
    # Create folder
    folderName = 'UDFPTS'
    if not os.path.exists(folderName):
        os.makedirs(folderName)
    
    print ('Generate UDF Points for FLUENT')
    
    # load points of all frames into one global array
    allFrame = []
    for frameID in range(0, numFrame):    
        if os.name == 'posix':
            rawPtName = 'PTS/surface_{}.asc'.format(frameID)
        else:
            rawPtName = 'PTS\\surface_{}.asc'.format(frameID)
    
        eachFrame = np.loadtxt(rawPtName)
            
        allFrame.append(eachFrame)    
    
    print ('Total number of nodes on surface: {}'.format(len(allFrame[0])))

    # temporal interpolation between frames and writing of new files
    # after every 1000 points: write data to 'udfsurface_'-file in order to minimize memory usage
    entirePts = []
    i = 0
    for ptID in range(0, len(allFrame[0])):
        #extract data of indiviual point ID for all frames, tracking the movement of the individual point over time
        oo = []
        for frameID in range(0, numFrame):
            oo.append(allFrame[frameID][ptID]) 
        
        # movement of point three times in a row. 
        # Reason to this remains a secret to god and the original developer
        pts = oo+oo+oo 
        pts.append(oo[0]) # append array by first entry  
        
    	# interpolation for each point ID extracting this point ID from every frame
        denPts = itpl(pts, 3*numInter*numFrame, k, numFrame, 2*numFrame)
        entirePts.append(denPts) # append array with new interpolated points 
        
    	# write first point to file
        if ptID == 0:
            print ('0')
            for frame in range(0,numInter*numFrame+1):
                framePts=[]       
                framePts.append(entirePts[ptID][frame])
                if os.name=='posix':
                    frameFileName='UDFPTS/udfsurface_{}.asc'.format(frame+1)
                else:
                    frameFileName='UDFPTS\\udfsurface_{}.asc'.format(frame+1)
                f = open(frameFileName, "a")
                np.savetxt(f, framePts)
                f.close()
            entirePts = []
    
        # write data every 1000 points to file
        if ptID%1000 == 0 and ptID != 0:
            print (int(ptID/1000))
            for frame in range(0, numInter*numFrame+1):
                framePts = []      
                for pt in range(0, 1000):
                    framePts.append(entirePts[pt][frame])
                if os.name == 'posix':
                    frameFileName = 'UDFPTS/udfsurface_{}.asc'.format(frame+1)
                else:
                    frameFileName = 'UDFPTS\\udfsurface_{}.asc'.format(frame+1)
                f = open(frameFileName, "a")
                np.savetxt(f, framePts)
                f.close()
            entirePts = []
            i += 1
    
    # write remaining points to file
    print ('Total Number of Frames after interpolation: {}'.format(len(denPts)))
    
    # Update udf_exeDynMesh.c
    update_exeDynMesh('udf_exeDynMesh.c', [len(allFrame[0]), len(denPts)])
    
    for frameID in range(0, numInter*numFrame+1):
        framePts = []
        c = len(allFrame[0]) - i * 1000 -1
        for ptID in range(0, c):
            framePts.append(entirePts[ptID][frameID])
        if os.name == 'posix':
            frameFileName='UDFPTS/udfsurface_{}.asc'.format(frameID+1)
        else:
            frameFileName='UDFPTS\\udfsurface_{}.asc'.format(frameID+1)
    
        f = open(frameFileName, "a")
        np.savetxt(f, framePts)
        f.close()

    # ------------------------------------------------------------------
    # NEW: derive ED/ES from volume curve and optionally update inputPython.txt
    # ------------------------------------------------------------------
    try:
        # The flag is read from inputPython.txt: autoEdEsFromVolume 0/1
        ed_ms, es_ms, diag = derive_ed_es_from_volume_curve(
            input_path="inputPython.txt",
            inter_method=k,
            out_prefix="volume"
        )

        if int(diag.get("auto_flag", 0)) == 1:
            _update_input_kv({
                "EndDiastoleInMS": f"{ed_ms:.6f}",
                "EndSystoleInMS": f"{es_ms:.6f}",
            }, input_path="inputPython.txt", make_backup=True)
            print(f"Updated inputPython.txt: EndDiastoleInMS={ed_ms:.3f} ms, EndSystoleInMS={es_ms:.3f} ms")
        else:
            print("autoEdEsFromVolume=0 -> using EndDiastoleInMS/EndSystoleInMS from inputPython.txt (not modifying input). Wrote volume CSVs + plot.")
    
    except Exception as e:
        print("WARNING: ED/ES auto-derivation failed:", e)
        return
    
# -------------------------------------------------------------------------
# NEW: derive ED/ES from ventricle volume curve + optionally update inputPython.txt
# -------------------------------------------------------------------------

def _read_input_kv(input_path="inputPython.txt"):
    """
    Read key/value pairs from inputPython.txt (ignores comments and blank lines).
    Returns dict[str, str].
    """
    params = {}
    if not os.path.exists(input_path):
        raise RuntimeError(f"Cannot find '{input_path}'. Run from the folder containing inputPython.txt.")
    with open(input_path, "r") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) >= 2:
                params[parts[0]] = parts[1]
    return params

def _update_input_kv(updates, input_path="inputPython.txt", make_backup=True):
    """
    Update (or append) key/value lines in inputPython.txt.
    Keeps the rest of each line (e.g. trailing tokens) if present.
    """
    if not os.path.exists(input_path):
        raise RuntimeError(f"Cannot find '{input_path}' to update.")
    if make_backup:
        bak = input_path + ".bak"
        if not os.path.exists(bak):
            with open(input_path, "r") as f_in, open(bak, "w") as f_out:
                f_out.write(f_in.read())

    with open(input_path, "r") as f:
        lines = f.readlines()

    found = set()
    for i, line in enumerate(lines):
        s = line.strip()
        if (not s) or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) >= 2:
            key = parts[0]
            if key in updates:
                rest = parts[2:]  # preserve trailing tokens if any
                new_line = f"{key} {updates[key]}"
                if rest:
                    new_line += " " + " ".join(rest)
                new_line += "\n"
                lines[i] = new_line
                found.add(key)

    for key, val in updates.items():
        if key not in found:
            lines.append(f"{key} {val}\n")

    with open(input_path, "w") as f:
        f.writelines(lines)

def _first_existing_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def _load_faces_connectivity(path):
    faces_path = path
    faces = np.loadtxt(faces_path, dtype=int)
    if faces.ndim == 1:
        faces = faces.reshape(1, -1)
    if faces.shape[1] != 3:
        raise RuntimeError(f"Expected triangle faces with 3 indices per row in {faces_path}, got shape {faces.shape}")
    return faces

def _load_verts_connectivity(path,frame_id):
    verts_path = os.path.join(path,"ventricle_verts_" + str(frame_id) + ".txt")
    verts = np.loadtxt(verts_path)
    if verts.ndim == 1:
        verts = verts.reshape(1, -1)
    if verts.shape[1] != 3:
        raise RuntimeError(f"Expected 3 columns (x y z) in {verts_path}, got shape {verts.shape}")
    return verts

def _mesh_volume_mm3(verts, faces):
    """
    Compute enclosed volume of a closed triangle mesh via divergence theorem.
    verts: (V,3), faces: (F,3)
    Returns abs(volume) in units^3 (mm^3 if verts are mm).
    """
    tri = verts[faces]              # (F,3,3)
    v0 = tri[:, 0, :]
    v1 = tri[:, 1, :]
    v2 = tri[:, 2, :]
    vol = np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0
    return float(abs(vol))

def summarize_volume_diff(vol_processed_ml, vol_raw_ml, t_frames, frame_ids):
    """Single source of truth for the processed-vs-raw volume difference metric.

    Per-frame percentage difference and its min/max/absmax/absmean summary. Pure (no I/O),
    so both the file-based comparison (derive_ed_es_from_volume_curve) and the in-Blender
    console report (compute_frame_volume_diff) compute the numbers the exact same way.
    Denominator is the raw volume, so diff_pct > 0 means the processed mesh OVER-estimates.
    """
    vol_processed_ml = np.asarray(vol_processed_ml, dtype=float)
    vol_raw_ml = np.asarray(vol_raw_ml, dtype=float)
    t_frames = np.asarray(t_frames, dtype=float)
    frame_ids = list(frame_ids)

    diff_pct = (vol_processed_ml - vol_raw_ml) / vol_raw_ml * 100.0
    i_min = int(np.argmin(diff_pct))
    i_max = int(np.argmax(diff_pct))
    i_absmax = int(np.argmax(np.abs(diff_pct)))
    return {
        "frame_ids": frame_ids,
        "time_ms": t_frames,
        "vol_processed_ml": vol_processed_ml,
        "vol_raw_ml": vol_raw_ml,
        "diff_pct": diff_pct,
        "min_pct": float(diff_pct[i_min]), "min_frame": frame_ids[i_min], "min_time_ms": float(t_frames[i_min]),
        "max_pct": float(diff_pct[i_max]), "max_frame": frame_ids[i_max], "max_time_ms": float(t_frames[i_max]),
        "absmax_pct": float(abs(diff_pct[i_absmax])), "absmax_frame": frame_ids[i_absmax],
        "absmean_pct": float(np.mean(np.abs(diff_pct))),
    }

def format_volume_diff_lines(d):
    """Console lines for a summarize_volume_diff() result. Single source for the text so the
    subprocess log and the Blender console never drift apart."""
    return [
        "Volume difference (processed vs. raw = (processed - raw) / raw * 100):",
        f"  min diff = {d['min_pct']:+.2f} %  at frame {d['min_frame']} (t = {d['min_time_ms']:.1f} ms)",
        f"  max diff = {d['max_pct']:+.2f} %  at frame {d['max_frame']} (t = {d['max_time_ms']:.1f} ms)",
        f"  max |diff| = {d['absmax_pct']:.2f} %  at frame {d['absmax_frame']};  "
        f"mean |diff| = {d['absmean_pct']:.2f} %",
    ]

def compute_frame_volume_diff(input_path, plot_input_dir, plot_input_dirbase):
    """Per-frame LV volume of processed vs. raw geometries and their percentage difference,
    computed from the Connectivity meshes WITHOUT interpolation or plotting.

    Cheap enough to run in Blender's main process (only the N input frames are loaded, no
    dense temporal interpolation), so callers can surface the min/max difference in the
    Blender console -- the interactive plot itself runs in a detached subprocess whose
    stdout only reaches a log file.

    Denominator is the raw volume, so diff_pct > 0 means the processed mesh OVER-estimates.
    The min/max are computed over the same N input frames written to
    _<prefix>_frames_volumes.csv, so the numbers match that file exactly.

    Returns a dict (per-frame arrays + min/max/absmax/absmean summary), or None if no raw
    folder was given. Raises RuntimeError on missing/invalid connectivity or input mismatch.
    """
    if not plot_input_dirbase:
        return None

    params = _read_input_kv(input_path)
    N = int(float(params.get("numberFrames")))
    start_id = int(float(params.get("startFrameID", 0)))
    end_id = int(float(params.get("endFrameID", start_id + N - 1)))
    rr_ms = float(params.get("RRDurationInMS"))
    frame_ids = list(range(start_id, end_id + 1))
    if len(frame_ids) != N:
        raise RuntimeError(
            f"Input mismatch: numberFrames={N} but startFrameID={start_id}, "
            f"endFrameID={end_id} implies {len(frame_ids)} frames."
        )

    def _frame_volumes_ml(folder):
        conn = os.path.join(folder, "Connectivity")
        errs = _connectivity_errors(conn, start_id, end_id)
        if errs:
            raise RuntimeError("Connectivity pre-check failed:\n" + "\n".join("- " + e for e in errs))
        faces = _load_faces_connectivity(os.path.join(conn, "ventricle_faces.txt"))
        vols = [_mesh_volume_mm3(_load_verts_connectivity(conn, fid), faces) for fid in frame_ids]
        return np.asarray(vols, dtype=float) / 1000.0  # mm^3 -> mL

    vol_proc = _frame_volumes_ml(plot_input_dir)
    vol_raw = _frame_volumes_ml(plot_input_dirbase)
    t_frames = np.arange(N, dtype=float) * (rr_ms / float(N))
    return summarize_volume_diff(vol_proc, vol_raw, t_frames, frame_ids)

def derive_ed_es_from_volume_curve(input_path="",plot_input_dir="", plot_input_dirbase="", inter_method=None, out_prefix="volume", save_csv=False):
    """
    Derive ED/ES timings from the ventricle volume curve (from Connectivity mesh).
    - Computes volumes for the N input frames (startFrameID..endFrameID)
    - Builds an equidistant time axis using RRDurationInMS and numberFrames
      (dt_frame = RR / numberFrames) consistent with periodic interpolation in this pipeline.
    - Interpolates the scalar volume curve to the same temporal resolution as UDFPTS:
      dt_interp = RR / (numberFrames * numberInterm)
    - Finds ED (max) and ES (min) on that interpolated curve.

    Outputs:
      - <out_prefix>_frames_volumes.csv  (per input frame; when raw data is present:
        frame_i, frameID, time_ms, volume_raw_mL, volume_processed_mL, diff_percent)
      - <out_prefix>_curve.png          (volume curve + ED/ES lines)
    Also prints the min/max per-frame percentage difference (processed vs. raw) to the console.
    Returns:
      ed_ms, es_ms (float, in ms), plus diagnostic dict.
    """
    params = _read_input_kv(input_path)
    print(params)

    # Required params
    N = int(float(params.get("numberFrames")))
    start_id = int(float(params.get("startFrameID", 0)))
    end_id = int(float(params.get("endFrameID", start_id + N - 1)))
    num_interm = int(float(params.get("numberInterm", 1)))
    rr_ms = float(params.get("RRDurationInMS"))

    # Auto flag (0/1)
    auto_flag = int(float(params.get("autoEdEsFromVolume", 0)))

    # Optional: existing header values (for plotting reference)
    ed_in = float(params.get("EndDiastoleInMS", "nan"))
    es_in = float(params.get("EndSystoleInMS", "nan"))

    # Basic consistency
    expected = end_id - start_id + 1
    if expected != N:
        raise RuntimeError(
            f"Input mismatch: numberFrames={N} but startFrameID={start_id}, endFrameID={end_id} implies {expected} frames."
        )

    facespath = os.path.join(plot_input_dir,"Connectivity","ventricle_faces.txt")
    faces = _load_faces_connectivity(facespath)

    vertspath = os.path.join(plot_input_dir,"Connectivity")
    
    # Volumes for original frames
    frame_ids = list(range(start_id, end_id + 1))
    vol_mm3 = []
    verts_frames = []
    for fid in frame_ids:
        verts = _load_verts_connectivity(vertspath,fid)
        verts_frames.append(verts)
        vol_mm3.append(_mesh_volume_mm3(verts, faces))
    vol_mm3 = np.asarray(vol_mm3, dtype=float)
    vol_ml = vol_mm3 / 1000.0  # mm^3 -> mL

    # Time axis consistent with periodic cycle of length RR and N frames around the loop
    dt_frame = rr_ms / float(N)
    t_frames = np.arange(N, dtype=float) * dt_frame  # [0, RR) in N bins
    
    if plot_input_dirbase != "":
        base_conn_dir = os.path.join(plot_input_dirbase, "Connectivity")
        base_errors = _connectivity_errors(base_conn_dir, start_id, end_id)
        if base_errors:
            raise RuntimeError(
                "Raw/base connectivity pre-check failed:\n" + "\n".join("- " + e for e in base_errors)
            )
        facespathbase = os.path.join(plot_input_dirbase, "Connectivity", "ventricle_faces.txt")
        facesbase = _load_faces_connectivity(facespathbase)
        
        vertspathbase = os.path.join(plot_input_dirbase, "Connectivity")
        
        # Volumes for original frames
        frame_ids_base = list(range(start_id, end_id + 1))
        vol_mm3_base = []
        for fid in frame_ids_base:
            vertsbase = _load_verts_connectivity(vertspathbase,fid)
            vol_mm3_base.append(_mesh_volume_mm3(vertsbase, facesbase))
        vol_mm3_base = np.asarray(vol_mm3_base, dtype=float)
        vol_ml_base = vol_mm3_base / 1000.0  # mm^3 -> mL

        # Per-frame % difference (processed vs. raw). summarize_volume_diff is the single
        # source of truth for the metric; format_volume_diff_lines for its console text.
        vol_diff = summarize_volume_diff(vol_ml, vol_ml_base, t_frames, frame_ids)
        diff_pct = vol_diff["diff_pct"]
        for _line in format_volume_diff_lines(vol_diff):
            print(_line)


    post_dir = plot_input_dir
    if save_csv: 
        # Export per-frame volumes
        # Persistent postprocessing output folder in the same input folder
        csv_path = os.path.join(post_dir,"volume_comparison",f"_{out_prefix}_frames_volumes.csv")

        remove_if_exists(csv_path) # Remove existing file if any

        with open(csv_path, "w+", newline="") as f:
            w = csv.writer(f)
            if plot_input_dirbase != "":
                # Raw present: store raw volume, processed volume and their % difference.
                w.writerow(["frame_i", "frameID", "time_ms",
                            "volume_raw_mL", "volume_processed_mL", "diff_percent"])
                for i, fid in enumerate(frame_ids):
                    w.writerow([i, fid, f"{t_frames[i]:.6f}",
                                f"{vol_ml_base[i]:.6f}", f"{vol_ml[i]:.6f}",
                                f"{diff_pct[i]:.6f}"])
            else:
                w.writerow(["frame_i", "frameID", "time_ms", "volume_mL"])
                for i, fid in enumerate(frame_ids):
                    w.writerow([i, fid, f"{t_frames[i]:.6f}", f"{vol_ml[i]:.6f}"])

    # Interpolate FULL MESH to match UDFPTS temporal resolution (compute volume at every interpolated time step)
    # UDFPTS has (numInterm * numberFrames + 1) frames over one cycle, including a periodic duplicate of the start state.
    M = int(num_interm * N)
    t_interp = np.linspace(0.0, rr_ms, M + 1)  # [0, RR] inclusive

    # Stack vertices for all frames -> (N, V, 3)
    verts_frames = np.stack(verts_frames, axis=0)
    V = verts_frames.shape[1]

    # Build periodic extension (3 cycles + repeat first) in index space, mirroring the point interpolation strategy
    verts_ext = np.concatenate([verts_frames, verts_frames, verts_frames, verts_frames[0:1]], axis=0)  # (3N+1, V, 3)
    t_idx = np.arange(verts_ext.shape[0], dtype=float)

    # Interpolate over the central cycle [N, 2N] to reduce boundary artifacts; output has M+1 samples
    ipl_t = np.linspace(float(N), float(2 * N), M + 1)

    # Choose interpolation method consistent with motion interpolation
    if inter_method == 1:
        f = interp1d(t_idx, verts_ext, kind="slinear", axis=0)
        verts_interp = f(ipl_t)
    elif inter_method == 2:
        f = interp1d(t_idx, verts_ext, kind="quadratic", axis=0)
        verts_interp = f(ipl_t)
    elif inter_method == 3:
        f = interp1d(t_idx, verts_ext, kind="cubic", axis=0)
        verts_interp = f(ipl_t)
    else:
        # Default / recommended: monotonic cubic spline (PCHIP)
        verts_interp = pchip(t_idx, verts_ext)(ipl_t)

    # Compute volume at each interpolated time step (vectorized in face-chunks to keep memory low)
    faces_arr = np.asarray(faces, dtype=np.int64)
    nF = faces_arr.shape[0]
    vol_interp_mm3 = np.zeros(verts_interp.shape[0], dtype=float)

    chunk = 600  # triangles per chunk (tune if needed)
    for j0 in range(0, nF, chunk):
        fj = faces_arr[j0:j0 + chunk]
        v0 = verts_interp[:, fj[:, 0], :]  # (T, chunk, 3)
        v1 = verts_interp[:, fj[:, 1], :]
        v2 = verts_interp[:, fj[:, 2], :]
        cr = np.cross(v1, v2)
        dot = np.einsum("tij,tij->ti", v0, cr)
        vol_interp_mm3 += np.sum(dot, axis=1)

    vol_interp_mm3 = np.abs(vol_interp_mm3) / 6.0
    vol_interp_ml = vol_interp_mm3 / 1000.0  # mm^3 -> mL

    # ED/ES from interpolated mesh volume curve (volume-derived candidates)
    vmax = float(np.max(vol_interp_ml))
    vmin = float(np.min(vol_interp_ml))
    ed_candidates = np.where(np.isclose(vol_interp_ml, vmax, rtol=0.0, atol=1e-9))[0]
    es_candidates = np.where(np.isclose(vol_interp_ml, vmin, rtol=0.0, atol=1e-9))[0]
    ed_idx_vol = int(ed_candidates[0]) if len(ed_candidates) else int(np.argmax(vol_interp_ml))
    es_idx_vol = int(es_candidates[0]) if len(es_candidates) else int(np.argmin(vol_interp_ml))

    # If ED/ES hit the duplicated last sample (t=RR), map it to t=0 for cleaner downstream logic
    if ed_idx_vol == M:
        ed_idx_vol = 0
    if es_idx_vol == M:
        es_idx_vol = 0

    ed_ms_vol = float(t_interp[ed_idx_vol])
    es_ms_vol = float(t_interp[es_idx_vol])

    # Decide which ED/ES to USE (controlled by autoEdEsFromVolume)
    # - autoEdEsFromVolume=1: use volume-derived ED/ES and (optionally) overwrite inputPython.txt
    # - autoEdEsFromVolume=0: use the values provided in inputPython.txt (do NOT derive)
    if auto_flag == 1:
        ed_ms = ed_ms_vol
        es_ms = es_ms_vol
        ed_idx_used = ed_idx_vol
        es_idx_used = es_idx_vol
        ed_source = "from volume"
        es_source = "from volume"
    else:
        if (not np.isfinite(ed_in)) or (not np.isfinite(es_in)):
            raise RuntimeError("autoEdEsFromVolume=0 but EndDiastoleInMS/EndSystoleInMS are missing or invalid in inputPython.txt")
        # Keep timings within [0, RR) to support wrap-around conventions
        ed_ms = float(ed_in) % rr_ms
        es_ms = float(es_in) % rr_ms
        ed_idx_used = int(np.argmin(np.abs(t_interp - ed_ms)))
        es_idx_used = int(np.argmin(np.abs(t_interp - es_ms)))
        ed_source = "from input"
        es_source = "from input"

    if save_csv:
        # Export interpolated-step volume curve
        csv_interp_path = os.path.join(post_dir,"volume_comparison",f"_{out_prefix}_interp_volumes.csv")
        remove_if_exists(csv_interp_path) # Remove existing file if any

        with open(csv_interp_path, "w+", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step_i", "time_ms", "volume_mL"])
            for i in range(M + 1):
                w.writerow([i, f"{t_interp[i]:.6f}", f"{vol_interp_ml[i]:.6f}"])

    # Plot interpolated curve with ED/ES (only the ED/ES that are actually used)
    fig_path = os.path.join(post_dir,f"_{out_prefix}_curve.png")

    #-----
    
    # START PLOTTING
    
    #-----
    fig = plt.figure(figsize=(5, 5))

    # Interpolated mesh volume (every interpolated time step)
    plt.plot(t_interp, vol_interp_ml, linewidth=1.5, label="Interpolated frames")

    # STL-frame mesh volumes (original frames). Plot them on the same time axis.
    # Frames are treated as N phase bins over one RR: t_i = i * RR / N (same convention used above).
    # For visualization, add a wrap-around point at t=RR to show periodic closure.
    t_frames_plot = np.concatenate([t_frames, [rr_ms]])
    vol_frames_plot = np.concatenate([vol_ml, [vol_ml[0]]])
    plt.plot(t_frames_plot, vol_frames_plot, marker="o", linewidth=1.0, label="Original STL frames")
    if plot_input_dirbase != "":
        vol_frames_plotbase = np.concatenate([vol_ml_base, [vol_ml_base[0]]])
        plt.plot(t_frames_plot, vol_frames_plotbase, marker="o", linewidth=1.0, label="Raw frames")

    plt.xlabel("Time in ms")
    plt.ylabel("LV volume in mL")

    ed_color = "red"
    es_color = "purple"
    plt.axvline(ed_ms, color=ed_color, linestyle="dashed", label=f"ED at {ed_ms:.3f} ms")
    plt.axvline(es_ms, color=es_color, linestyle="dashed", label=f"ES at {es_ms:.3f} ms")

    plt.legend()
    plt.tight_layout()

    diag = {
        "auto_flag": auto_flag,
        "N": N,
        "numInterm": num_interm,
        "RR_ms": rr_ms,
        "dt_frame_ms": dt_frame,
        "M_interp": M,
        "dt_interp_ms": rr_ms / float(M),
        "ed_idx_used": ed_idx_used,
        "es_idx_used": es_idx_used,
        "ed_idx_vol": ed_idx_vol,
        "es_idx_vol": es_idx_vol,
        "ed_ms_vol": ed_ms_vol,
        "es_ms_vol": es_ms_vol,
        "ed_ms_used": ed_ms,
        "es_ms_used": es_ms,
        "ed_source": ed_source,
        "es_source": es_source,
        #"frames_csv_path": str(csv_path),
        #"interp_csv_path": str(csv_interp_path),
        "fig_path": str(fig_path),
    }

    if plot_input_dirbase != "":
        diag.update({
            "diff_min_pct": vol_diff["min_pct"],
            "diff_max_pct": vol_diff["max_pct"],
            "diff_absmax_pct": vol_diff["absmax_pct"],
            "diff_absmean_pct": vol_diff["absmean_pct"],
        })

    return ed_ms, es_ms, diag, fig
