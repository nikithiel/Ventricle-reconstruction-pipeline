#!/usr/bin/env python3
from __future__ import annotations

import re
import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import os

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import trimesh


# -----------------------------
# IO / parsing
# -----------------------------

def parse_inputpython_txt(path: Path) -> Dict[str, object]:
    """
    Parses lines: key value
    '#' starts a comment (also inline).
    Keys are stored lowercase.
    """
    params: Dict[str, object] = {}
    if not path.exists():
        return params

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        parts = line.split()
        if len(parts) < 2:
            continue

        key = parts[0].strip().lower()
        val_str = " ".join(parts[1:]).strip()

        try:
            if re.fullmatch(r"[+-]?\d+", val_str):
                val: object = int(val_str)
            else:
                val = float(val_str)
        except ValueError:
            val = val_str

        params[key] = val

    return params


def get_float(params: Dict[str, object], key: str) -> Optional[float]:
    v = params.get(key.lower(), None)
    return None if v is None else float(v)


def read_velocity_csv(path: Path) -> pd.DataFrame:
    """
    Your format:
      - no header
      - semicolon separated
      - decimal comma
      - columns: time ; velocity
    """
    df = pd.read_csv(
        path,
        sep=";",
        header=None,
        names=["t", "v"],
        decimal=",",
        skipinitialspace=True,
        engine="python",
    )
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df["v"] = pd.to_numeric(df["v"], errors="coerce")/100                                                  # Conversion from cm/s to m/s
    df = df.dropna().sort_values("t").reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"Velocity file parsed empty: {path}")
    return df


# Captures the LAST run of digits immediately before the '.stl' extension.
# Anchoring to '\.stl$' means any prefix (case id, date such as 3.29.22,
# anatomy label, ...) is ignored -- only the trailing frame number is used.
_STL_INDEX_RE = re.compile(r"(\d+)\.stl$", re.IGNORECASE)


def stl_index_from_name(name: str) -> Optional[int]:
    """
    Return the phase index encoded at the END of an STL filename, ignoring the
    prefix. The index is the last digit run before the '.stl' extension:

        'ventricle_7.stl'                             -> 7
        'CP RA 3.29.22_BeutelRevision+LVOT_000.stl'   -> 0
        'CP SJ 12.27.22_BeutelRevision+LVOT_042.stl'  -> 42

    Leading zeros are fine (int() strips them). Returns None when there is no
    trailing number (e.g. 'aorta_static.stl'), so such files are skipped.
    """
    m = _STL_INDEX_RE.search(name)
    return int(m.group(1)) if m else None


def stl_series_prefix(name: str) -> Optional[str]:
    """
    Everything before the trailing numeric index. Used to group frames that
    belong to the same series so a folder may also hold unrelated meshes.
    Returns None when there is no trailing number.
    """
    m = _STL_INDEX_RE.search(name)
    return name[: m.start(1)] if m else None


def discover_phase_stls(stl_dir: Path, pattern: Optional[str] = None) -> List[Tuple[int, Path]]:
    """
    Find the ventricle phase STL series in `stl_dir`, robust to naming.

    Any file ending in '<digits>.stl' (any case) is a candidate frame.
    Candidates are grouped by the text preceding those digits (the 'series
    prefix'), so the folder is allowed to contain unrelated meshes too. If more
    than one series is present, the longest one is used (preferring a series
    that starts at index 0) and a warning names what was skipped.

    Pass `pattern` (a glob such as '*LVOT_*.stl') to restrict the search
    explicitly if auto-detection ever picks the wrong series.

    Returns [(index, path), ...] sorted by index, with duplicate indices dropped.
    """
    files_iter = stl_dir.glob(pattern) if pattern else stl_dir.iterdir()

    series: Dict[str, List[Tuple[int, Path]]] = {}
    for f in files_iter:
        if not f.is_file():
            continue
        idx = stl_index_from_name(f.name)
        if idx is None:
            continue
        series.setdefault(stl_series_prefix(f.name), []).append((idx, f))

    if not series:
        raise FileNotFoundError(
            f"No numbered STL frames (e.g. '..._000.stl') found in: {stl_dir}"
        )

    # Prefer the longest series; break ties toward one that starts at index 0,
    # then by prefix name so the choice is deterministic.
    def series_key(prefix: str) -> Tuple[int, bool, str]:
        idxs = [i for i, _ in series[prefix]]
        return (len(series[prefix]), min(idxs) == 0, prefix)

    chosen = max(series, key=series_key)
    if len(series) > 1:
        skipped = sorted(p for p in series if p != chosen)
        warnings.warn(
            f"Multiple STL series in {stl_dir}: {sorted(series)!r}. "
            f"Using prefix {chosen!r} ({len(series[chosen])} frames); "
            f"ignoring {skipped!r}. Pass `pattern=...` to override."
        )

    # De-duplicate indices (keep first seen), then sort by index.
    by_idx: Dict[int, Path] = {}
    for idx, f in series[chosen]:
        if idx in by_idx:
            warnings.warn(
                f"Duplicate phase index {idx} in {stl_dir}: keeping "
                f"'{by_idx[idx].name}', ignoring '{f.name}'."
            )
            continue
        by_idx[idx] = f

    return sorted(by_idx.items())


def compute_volume_vs_phase_from_stl(stl_dir: Path, pattern: Optional[str] = None) -> pd.DataFrame:
    """
    Reads the ventricle phase STL series, computes volume and phase phi in [0,1].

    Filenames are auto-detected via discover_phase_stls, so both the legacy
    'ventricle_<idx>.stl' and arbitrary '<prefix>_<idx>.stl' (e.g.
    'CP RA 3.29.22_BeutelRevision+LVOT_000.stl') work with no code changes.
    Pass `pattern` to force a specific glob if needed.

    Time is NOT assigned here (because you have bpm_mv and bpm_av separately).

    Returns columns: idx, phi, V
    """
    items = discover_phase_stls(stl_dir, pattern=pattern)   # [(idx, path), ...] sorted
    idxs = [idx for idx, _ in items]

    N = max(idxs)
    if N <= 0:
        raise ValueError(
            f"Need at least two phase frames with indices like _000, _001, ... "
            f"in {stl_dir} (got max index N={N})."
        )

    records: List[Tuple[int, float, float]] = []
    for idx, f in items:
        phi = float(idx) / float(N)

        mesh = trimesh.load_mesh(f, force="mesh")
        if hasattr(mesh, "is_watertight") and not mesh.is_watertight:
            warnings.warn(f"Mesh not watertight: {f}. Volume may be unreliable.")
        V = float(mesh.volume) / 1000**3                                                                        # Conversion from mm^3 to m^3
        records.append((idx, phi, V))

    return pd.DataFrame(records, columns=["idx", "phi", "V"]).reset_index(drop=True)


# -----------------------------
# Signal processing
# -----------------------------

def smooth_and_dVdphi(vol_df: pd.DataFrame, window: int = 4, polyorder: int = 3) -> pd.DataFrame:
    """
    Adds V_smooth and dVdphi (derivative wrt phase).
    """
    df = vol_df.copy()
    phi = df["phi"].to_numpy()
    V = df["V"].to_numpy()
    n = len(df)

    if n < 5:
        df["V_smooth"] = V
        df["dVdphi"] = np.gradient(V, phi) if n >= 2 else np.zeros_like(V)
        return df

    w = min(window, n if n % 2 == 1 else n - 1)
    w = max(w, 5)
    if w % 2 == 0:
        w -= 1
    w = min(w, n if n % 2 == 1 else n - 1)

    p = min(polyorder, w - 2)

    V_s = savgol_filter(V, window_length=w, polyorder=p)
    dVdphi = np.gradient(V_s, phi)

    df["V_smooth"] = V_s
    df["dVdphi"] = dVdphi
    return df


def resample_to_phi_grid(phi_in: np.ndarray, y_in: np.ndarray, phi_grid: np.ndarray) -> np.ndarray:
    idx = np.argsort(phi_in)
    return np.interp(phi_grid, phi_in[idx], y_in[idx], left=y_in[idx][0], right=y_in[idx][-1])


def average_cycle_on_phi_grid(t: np.ndarray, y: np.ndarray, bpm: float, phi_grid: np.ndarray) -> np.ndarray:
    """
    Fold multi-beat Doppler signal into one cycle using bpm, then interpolate to phi_grid in [0,1).
    """
    if bpm is None or bpm <= 0:
        raise ValueError("bpm must be > 0")
    T = 60.0 / float(bpm)

    t0 = float(t[0])
    phi = np.mod(t - t0, T) / T  # in [0,1)
    idx = np.argsort(phi)
    phi_s = phi[idx]
    y_s = y[idx]

    # average duplicates
    tmp = pd.DataFrame({"phi": phi_s, "y": y_s}).groupby("phi", as_index=False).mean()
    return np.interp(phi_grid, tmp["phi"].to_numpy(), tmp["y"].to_numpy())


def fit_scaling_signed(v: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """
    Least squares scale s minimizing ||s*v - target|| on mask.
    s can be negative (use abs(s) for diameter if sign convention differs).
    """
    v_m = v[mask]
    t_m = target[mask]
    denom = float(np.dot(v_m, v_m))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(v_m, t_m) / denom)


def area_to_diameter(A: float) -> float:
    """
    Interpret |A| as effective area and return diameter.
    d = 2*sqrt(A/pi)
    """
    Aeff = abs(A)
    if not np.isfinite(Aeff) or Aeff <= 0:
        return float("nan")
    return float(2.0 * np.sqrt(Aeff / np.pi))

def area_to_ellipse_diameters(A: float, ratio: float = 1.3) -> tuple[float, float]:
    """
    Convert effective area |A| to ellipse diameters (d_minor, d_major)
    assuming d_major / d_minor = ratio.

    A = (pi/4) * d_major * d_minor
    with d_major = ratio * d_minor
    """
    Aeff = abs(A)
    if not np.isfinite(Aeff) or Aeff <= 0 or not np.isfinite(ratio) or ratio <= 0:
        return float("nan"), float("nan")

    d_minor = float(np.sqrt(4.0 * Aeff / (np.pi * ratio)))
    d_major = float(ratio * d_minor)
    return d_minor, d_major

def shift_on_phi_grid(phi_grid: np.ndarray, y: np.ndarray, shift: float) -> np.ndarray:
    """
    Periodic phase shift: returns y(phi + shift mod 1).
    Works for arbitrary (non-integer) shift via interpolation.
    """
    shift = float(shift) % 1.0
    phi_ext = np.concatenate([phi_grid, phi_grid + 1.0])
    y_ext = np.concatenate([y, y])
    phi_q = (phi_grid + shift) % 1.0
    return np.interp(phi_q, phi_ext, y_ext)

def best_shift_and_scale(phi_grid: np.ndarray,
                         v_cycle: np.ndarray,
                         target: np.ndarray,
                         mask: np.ndarray,
                         n_shift: int = 240) -> tuple[float, float, float, np.ndarray]:
    shifts = np.linspace(0.0, 1.0, n_shift, endpoint=False)

    best_rmse = np.inf
    best_shift = 0.0
    best_A = np.nan
    best_v = v_cycle

    for s in shifts:
        v_s = shift_on_phi_grid(phi_grid, v_cycle, s)
        A = fit_scaling_signed(v_s, target, mask)
        if not np.isfinite(A):
            continue
        resid = (A * v_s - target)[mask]
        rmse = float(np.sqrt(np.mean(resid**2))) if resid.size else np.inf
        if rmse < best_rmse:
            best_rmse = rmse
            best_shift = float(s)
            best_A = float(A)
            best_v = v_s

    return best_shift, best_A, best_rmse, best_v

def resample_1cycle_to_phi(v_cycle: np.ndarray, phi_grid: np.ndarray) -> np.ndarray:
    """
    Periodic resampling of one cycle to phi_grid in [0,1).
    Prevents the 'flat tail' from np.interp when L < len(phi_grid).
    """
    v_cycle = np.asarray(v_cycle, float)
    L = len(v_cycle)
    if L < 2:
        return np.full_like(phi_grid, v_cycle[0] if L == 1 else np.nan, dtype=float)

    phi_in = np.linspace(0.0, 1.0, L, endpoint=False)

    # append wrap point at phi=1 with value equal to phi=0 (periodic)
    phi_ext = np.concatenate([phi_in, [1.0]])
    v_ext = np.concatenate([v_cycle, [v_cycle[0]]])

    return np.interp(phi_grid, phi_ext, v_ext)

from scipy.signal import find_peaks  # add this import near the top

def resample_window_to_phi_time(df_raw: pd.DataFrame,
                                t_start: float,
                                T: float,
                                phi_grid: np.ndarray,
                                min_points: int = 10) -> Optional[np.ndarray]:
    """
    Resample non-uniformly sampled v(t) in [t_start, t_start+T] to uniform phase grid phi_grid in [0,1).

    Returns v_cycle(phi_grid) or None if too few points.
    Uses periodic closure (phi=1 mapped to value at phi=0) to avoid a flat tail.
    """
    t0 = float(t_start)
    t1 = t0 + float(T)

    w = df_raw[(df_raw["t"] >= t0) & (df_raw["t"] <= t1)]
    if len(w) < min_points:
        return None

    tt = w["t"].to_numpy()
    vv = w["v"].to_numpy()

    # phase in [0,1]
    phi = (tt - t0) / float(T)
    idx = np.argsort(phi)
    phi = phi[idx]
    vv = vv[idx]

    # merge duplicate phases (rare, but safe)
    tmp = pd.DataFrame({"phi": phi, "v": vv}).groupby("phi", as_index=False).mean()
    phi_u = tmp["phi"].to_numpy()
    v_u = tmp["v"].to_numpy()

    # periodic closure to avoid "last value repeated" due to extrapolation
    # add phi=1 with value at phi=0
    phi_ext = np.concatenate([phi_u, [1.0]])
    v_ext = np.concatenate([v_u, [v_u[0]]])

    return np.interp(phi_grid, phi_ext, v_ext)

from scipy.signal import find_peaks

def extract_best_cycle_by_time_peak_aligned(df_raw: pd.DataFrame,
                                            bpm: float,
                                            phi_grid: np.ndarray,
                                            target: np.ndarray,
                                            mask: np.ndarray,
                                            phi_event_target: float,
                                            event: str,
                                            n_events: int = 50,
                                            min_points: int = 10,
                                            # peak detection parameters (tune if needed)
                                            prominence_global: float = 0.0,
                                            distance_frac: float = 0.4,
                                            width_frac: float = 0.0,
                                            # refinement around the chosen global peak
                                            refine_iter: int = 2,
                                            anchor_tol_frac: float = 0.12) -> Dict[str, object]:
    """
    Picks the HIGHEST (global) peak that allows building a full cycle window [t_start, t_start+T].
    Then fits A ONLY on that selected window.

    event='max': choose largest positive peak
    event='min': choose most negative peak (i.e. largest peak of -v)

    Returns dict with:
      t_start, T, t_evt_global, t_evt (used peak time), v_cycle, A, rmse, n_points_window,
      phi_evt_found, phi_evt_error
    """
    if bpm is None or bpm <= 0:
        raise ValueError("bpm must be > 0")
    if event not in ("max", "min"):
        raise ValueError("event must be 'max' or 'min'")

    df = df_raw.sort_values("t").reset_index(drop=True)
    t = df["t"].to_numpy(dtype=float)
    v = df["v"].to_numpy(dtype=float)

    T = 60.0 / float(bpm)
    t_min, t_max = float(t.min()), float(t.max())
    if t_max - t_min < 0.9 * T:
        raise RuntimeError("Doppler series shorter than one beat (based on bpm).")

    phi_target = float(phi_event_target % 1.0)

    # Work in "sig" where we always want MAX peaks
    sig = v if event == "max" else -v

    # Estimate sampling step for distance/width in samples
    dt = np.diff(t)
    dt_med = float(np.median(dt[dt > 0])) if np.any(dt > 0) else (T / 200.0)

    distance = max(1, int(distance_frac * T / dt_med))  # suppress multiple nearby peaks
    kwargs = dict(prominence=prominence_global, distance=distance)

    if width_frac and width_frac > 0:
        width = max(1, int(width_frac * T / dt_med))
        kwargs["width"] = width

    peaks, props = find_peaks(sig, **kwargs)

    # fallback if find_peaks finds nothing: use absolute max sample
    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(sig))], dtype=int)

    # Sort peaks by amplitude (highest first)
    order = np.argsort(sig[peaks])[::-1]
    peaks = peaks[order][:int(n_events)]

    anchor_tol = float(anchor_tol_frac) * T

    def refine_t_start_keep_same_peak(t_start_init: float, t_anchor: float) -> tuple[float, float]:
        """
        Refine t_start by re-locating the peak time near t_anchor (within +/- anchor_tol)
        using the local extremum (max/min) in that neighborhood.
        This avoids jumping to another peak (e.g. E vs A wave).
        """
        t_start = float(t_start_init)
        t_peak_used = float("nan")

        for _ in range(int(refine_iter)):
            t0, t1 = t_start, t_start + T
            w = df[(df["t"] >= t0) & (df["t"] <= t1)]
            if len(w) < 5:
                break

            tt = w["t"].to_numpy(dtype=float)
            vv = w["v"].to_numpy(dtype=float)
            ss = vv if event == "max" else -vv

            # restrict search close to the anchored global peak time
            close = np.abs(tt - t_anchor) <= anchor_tol
            if np.any(close):
                tt2 = tt[close]
                ss2 = ss[close]
                j = int(np.argmax(ss2))
                t_peak = float(tt2[j])
            else:
                # fallback: closest sample to anchor (still anchored)
                j = int(np.argmin(np.abs(tt - t_anchor)))
                t_peak = float(tt[j])

            t_peak_used = t_peak
            t_start = t_peak - phi_target * T

        return float(t_start), float(t_peak_used)

    # --- pick the first (highest) peak that yields a valid full window
    for p in peaks:
        t_evt_global = float(t[p])

        # Build window so that this peak lies at phase phi_target
        t_start = t_evt_global - phi_target * T

        # Must be able to build full cycle window
        if t_start < t_min or (t_start + T) > t_max:
            continue

        # Optional refinement, but anchored to SAME peak (won't jump to secondary)
        t_start, t_peak_used = refine_t_start_keep_same_peak(t_start, t_anchor=t_evt_global)

        if t_start < t_min or (t_start + T) > t_max:
            continue

        v_cycle = resample_window_to_phi_time(df, t_start, T, phi_grid, min_points=min_points)
        if v_cycle is None:
            continue

        A = fit_scaling_signed(v_cycle, target, mask)
        if not np.isfinite(A):
            continue

        resid = (A * v_cycle - target)[mask]
        rmse = float(np.sqrt(np.mean(resid**2))) if resid.size else float("inf")

        npts = int(((df["t"] >= t_start) & (df["t"] <= t_start + T)).sum())

        phi_evt_found = float((t_peak_used - t_start) / T) if np.isfinite(t_peak_used) else float("nan")
        phi_evt_error = float(((phi_evt_found - phi_target + 0.5) % 1.0) - 0.5) if np.isfinite(phi_evt_found) else float("nan")

        return {
            "t_start": float(t_start),
            "T": float(T),
            "t_evt_global": float(t_evt_global),
            "t_evt": float(t_peak_used),
            "A": float(A),
            "rmse": rmse,
            "v_cycle": np.asarray(v_cycle, float),
            "n_points_window": npts,
            "phi_evt_found": phi_evt_found,
            "phi_evt_error": phi_evt_error,
        }

    raise RuntimeError("No usable peak found that allows a full cycle window with enough points.")

def _shade_phase_mask(ax, phi_grid: np.ndarray, mask: np.ndarray,
                      color: str = "C2", alpha: float = 0.15, label: str = "fit mask") -> None:
    """
    Shade contiguous True segments of a boolean mask over phi_grid.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != phi_grid.shape or mask.sum() == 0:
        return

    dphi = float(phi_grid[1] - phi_grid[0]) if len(phi_grid) > 1 else 1.0

    # Find runs of True
    idx = np.where(mask)[0]
    # split where discontinuous
    splits = np.where(np.diff(idx) > 1)[0] + 1
    runs = np.split(idx, splits)

    first = True
    for run in runs:
        left = float(phi_grid[run[0]])
        right = float(phi_grid[run[-1]] + dphi)
        right = min(right, 1.0)  # cap to domain end
        ax.axvspan(left, right, color=color, alpha=alpha, label=(label if first else None))
        first = False

def plot_curve_with_mask_alpha(ax,
                               x: np.ndarray,
                               y: np.ndarray,
                               mask: np.ndarray,
                               color: str,
                               label: Optional[str] = None,
                               lw: float = 1.5,
                               alpha_out: float = 0.25,
                               alpha_in: float = 1.0) -> None:
    """
    Plot y(x) transparent everywhere, and overlay the masked segments opaque.
    Handles multiple disjoint True-runs (wrap-around masks).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.asarray(mask, dtype=bool)

    # base: whole curve transparent (no legend entry)
    ax.plot(x, y, color=color, lw=lw, alpha=alpha_out)

    idx = np.where(mask)[0]
    if idx.size == 0:
        # if you still want a legend entry even when mask empty:
        if label is not None:
            ax.plot([], [], color=color, lw=lw, alpha=alpha_in, label=label)
        return

    # split into contiguous runs
    splits = np.where(np.diff(idx) > 1)[0] + 1
    runs = np.split(idx, splits)

    first = True
    for run in runs:
        ax.plot(x[run], y[run], color=color, lw=lw, alpha=alpha_in,
                label=(label if first else None))
        first = False

def save_validation_plot_time_axis(df_raw: pd.DataFrame,
                                   best: Dict[str, object],
                                   out_path: Path,
                                   title: str,
                                   phi_grid: np.ndarray,
                                   target: np.ndarray,
                                   fit_curve: np.ndarray,
                                   mask_phase: Optional[np.ndarray] = None,
                                   phi_ref: Optional[float] = None) -> None:
    """
    Validation plot for non-uniform Doppler sampling:
      (1) raw v(t) from CSV with the extracted window [t_start, t_start+T] highlighted
      (2) extracted cycle v(phi) vs target and fit + shaded phase mask used for fitting
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = df_raw.sort_values("t").reset_index(drop=True)
    t = df["t"].to_numpy()
    v = df["v"].to_numpy()

    t_start = float(best["t_start"])
    T = float(best["T"])
    t_end = t_start + T
    mask_win = (t >= t_start) & (t <= t_end)

    v_cycle = np.asarray(best["v_cycle"], float)
    fit_curve = np.asarray(fit_curve, float)

    fig, axs = plt.subplots(2, 1, figsize=(12, 6.5), gridspec_kw={"hspace": 0.25})

    # (1) time plot (true CSV time axis)
    axs[0].plot(t, v, color="0.7", lw=1.0, label="raw v(t) from CSV")
    axs[0].plot(t[mask_win], v[mask_win], color="C0", lw=1.6, label="selected 1-cycle window")
    axs[0].axvspan(t_start, t_end, color="C0", alpha=0.10)

    if "t_evt" in best and np.isfinite(best["t_evt"]):
        axs[0].axvline(float(best["t_evt"]), color="k", ls=":", lw=1.0, label="aligned peak time")

    npts = best.get("n_points_window", None)
    npts_str = f", points={npts}" if npts is not None else ""
    axs[0].set_title(f"{title}\nwindow: [{t_start:.6g}, {t_end:.6g}] (T={T:.6g}s){npts_str}")
    axs[0].set_xlabel("t (from CSV)")
    axs[0].set_ylabel("velocity")
    axs[0].grid(True, alpha=0.25)
    axs[0].legend(loc="best")

    # (2) phase plot + mask shading
    if mask_phase is not None:
        _shade_phase_mask(axs[1], phi_grid, mask_phase, color="C2", alpha=0.15, label="fitting time window")

    axs[1].plot(phi_grid, target, color="k", ls="--", lw=1.2, label="STL volume change")
    axs[1].plot(phi_grid, fit_curve, color="C3", lw=1.4, label="extracted, fittet cycle")

    if phi_ref is not None:
        axs[1].axvline(float(phi_ref), color="k", ls=":", lw=1.1, label="Max Volume Change")

    axs[1].set_xlabel("cycle phase $\\phi$ in [0,1)")
    axs[1].set_ylabel("Volume Change")
    axs[1].grid(True, alpha=0.25)
    axs[1].legend(loc="best")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# -----------------------------
# Case processing
# -----------------------------

@dataclass
class CaseData:
    name: str
    path: str
    params: Dict[str, object]
    bpm_mv: Optional[float]
    bpm_av: Optional[float]
    av: Optional[pd.DataFrame]
    mv: Optional[pd.DataFrame]
    volume: Optional[pd.DataFrame]
    fit: Dict[str, object]


def find_valve_files(case_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    # support .csv and .scv
    av = next(iter(sorted(list(case_dir.glob("*_AV.csv")) + list(case_dir.glob("*_AV.scv")))), None)
    mv = next(iter(sorted(list(case_dir.glob("*_MV.csv")) + list(case_dir.glob("*_MV.scv")))), None)
    return av, mv


def resolve_case_dir(stl_dir: Path) -> Path:
    """
    Locate the 'case' folder that holds inputPython.txt and the AV/MV CSVs.

    You point the script at the folder containing the mesh frames (your Blender
    '/STL/' export). The case-level inputs may sit right there (flat layout) or
    one level up in the parent case folder. This returns whichever of
    {stl_dir, stl_dir.parent} actually contains those inputs, preferring
    stl_dir. Falls back to the parent (with a warning) if neither does.
    """
    def has_inputs(d: Path) -> bool:
        av, mv = find_valve_files(d)
        return (d / "inputPython.txt").exists() or av is not None or mv is not None

    if has_inputs(stl_dir):
        return stl_dir
    if has_inputs(stl_dir.parent):
        return stl_dir.parent
    warnings.warn(
        f"Could not find inputPython.txt or *_AV/_MV.csv in {stl_dir} "
        f"or its parent {stl_dir.parent}. Using the parent as the case folder."
    )
    return stl_dir.parent


def process_case(case_dir: Path, n_phase: int = 400, stl_dir: Optional[Path] = None) -> CaseData:
    # case_dir -> holds inputPython.txt, the AV/MV CSVs, and receives the outputs
    # stl_dir  -> holds the mesh frames; defaults to case_dir (flat layout)
    if stl_dir is None:
        stl_dir = case_dir

    fit: Dict[str, object] = {}
    params = parse_inputpython_txt(case_dir / "inputPython.txt")
    bpm_mv = get_float(params, "bpm_mv")
    bpm_av = get_float(params, "bpm_av")

    av_file, mv_file = find_valve_files(case_dir)
    av_df = read_velocity_csv(av_file) if av_file else None
    mv_df = read_velocity_csv(mv_file) if mv_file else None

    vol_df = compute_volume_vs_phase_from_stl(stl_dir)
    vol_df = smooth_and_dVdphi(vol_df, polyorder = 3)

    # Common phase grid
    phi_grid = np.linspace(0.0, 1.0, n_phase, endpoint=False)

    # Resample volume signals to phase grid
    Vc = resample_to_phi_grid(vol_df["phi"].to_numpy(), vol_df["V_smooth"].to_numpy(), phi_grid)
    dVdphi_c = resample_to_phi_grid(vol_df["phi"].to_numpy(), vol_df["dVdphi"].to_numpy(), phi_grid)

    phi_mv_ref = float(phi_grid[np.argmax(dVdphi_c)])
    phi_av_ref = float(phi_grid[np.argmin(dVdphi_c)])

    i_peak = int(np.argmax(Vc))
    phi_peak = float(phi_grid[i_peak])

    dphi = (phi_grid - phi_peak) % 1.0   # in [0,1)

    av_mask = dphi < 0.5                 # half-cycle after peak (wrap-safe)
    mv_mask = ~av_mask                   # half-cycle before peak

    fit.update({
        "phi_mv_ref_from_stl": phi_mv_ref,
        "phi_av_ref_from_stl": phi_av_ref,
        "phi_peak": phi_peak,
        "phi_grid": phi_grid.tolist(),
        "V_cycle": Vc.tolist(),
        "dVdphi_cycle": dVdphi_c.tolist(),
    })

    # Compute dV/dt separately for MV and AV using their own bpm
    if bpm_mv is not None and bpm_mv > 0:
        T_mv = 60.0 / bpm_mv
        dVdt_mv = (1.0 / T_mv) * dVdphi_c
        fit["T_mv"] = T_mv
        fit["dVdt_mv_cycle"] = dVdt_mv.tolist()
    else:
        dVdt_mv = None

    if bpm_av is not None and bpm_av > 0:
        T_av = 60.0 / bpm_av
        dVdt_av = (1.0 / T_av) * dVdphi_c
        fit["T_av"] = T_av
        fit["dVdt_av_cycle"] = dVdt_av.tolist()
    else:
        dVdt_av = None

    # Doppler cycles (folded) and fits
    val_dir = case_dir / Path("calc_valve_diameters_outputs/validation")
    val_dir.mkdir(parents=True, exist_ok=True)

    # MV: fit A*v ~ +dVdt_mv on mv_mask
    if mv_df is not None and dVdt_mv is not None and bpm_mv is not None:
        v_full_mv = mv_df["v"].to_numpy()
        t_full_mv = mv_df["t"].to_numpy()

        best_mv = extract_best_cycle_by_time_peak_aligned(
            df_raw=mv_df,
            bpm=bpm_mv,
            phi_grid=phi_grid,
            target=dVdt_mv,
            mask=mv_mask,
            phi_event_target=phi_mv_ref,
            event="max",
            min_points=10,
        )

        A_mv = best_mv["A"]
        fit["A_MV"] = A_mv
        fit["mv_fit_curve"] = (A_mv * best_mv["v_cycle"]).tolist()
        fit["t_start_MV"] = best_mv["t_start"]
        d_mv_minor, d_mv_major = area_to_ellipse_diameters(A_mv, ratio=1.3)
        fit["d_MV_minor"] = d_mv_minor
        fit["d_MV_major"] = d_mv_major
        fit["rmse_MV"] = best_mv["rmse"]

        mv_fit_curve = A_mv * best_mv["v_cycle"]

        save_validation_plot_time_axis(
            df_raw=mv_df,
            best=best_mv,
            out_path=val_dir / Path("cycle_extraction_MV.png"),
            title=f"{case_dir.name} - MV (aligned to STL +dV peak phase)",
            phi_grid=phi_grid,
            target=dVdt_mv,
            fit_curve=mv_fit_curve,
            mask_phase=mv_mask,
            phi_ref=phi_mv_ref,
        )

    # AV: fit A*v ~ -dVdt_av on av_mask  (so target is -dVdt_av)
    if av_df is not None and dVdt_av is not None and bpm_av is not None:
        v_full_av = av_df["v"].to_numpy()
        t_full_av = av_df["t"].to_numpy()

        best_av = extract_best_cycle_by_time_peak_aligned(
            df_raw=av_df,
            bpm=bpm_av,
            phi_grid=phi_grid,
            target=-dVdt_av,
            mask=av_mask,
            phi_event_target=phi_av_ref,
            event="min",
            min_points=10,
        )

        A_av = best_av["A"]
        fit["A_AV"] = A_av
        fit["t_start_AV"] = best_av["t_start"]
        fit["d_AV"] = area_to_diameter(A_av)
        fit["rmse_AV"] = best_av["rmse"]

        fit["av_fit_curve_target_minus_dVdt"] = (A_av * best_av["v_cycle"]).tolist()
        fit["av_fit_curve_plot"] = (-A_av * best_av["v_cycle"]).tolist()  # compare to +dVdt_av

        av_fit_curve_plot = -A_av * best_av["v_cycle"]

        save_validation_plot_time_axis(
            df_raw=av_df,
            best=best_av,
            out_path=val_dir / Path("cycle_extraction_AV.png"),
            title=f"{case_dir.name} - AV (aligned to STL -dV peak phase)",
            phi_grid=phi_grid,
            target=dVdt_av,              # not sign-flipped in the plot
            fit_curve=av_fit_curve_plot, # -A*v so it matches dVdt_av
            mask_phase=av_mask,
            phi_ref=phi_av_ref,
        )

    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"hspace": 0.1})

    # top subplot: left axis = Volume, right axis = dV/dphi
    axV = axs[0]          # left y-axis
    axD = axV.twinx()     # right y-axis

    # Volume (left)
    axV.plot(phi_grid, Vc, color="C0", label="Volume (smoothed)")
    axV.axvline(phi_peak, color="k", linestyle="--", linewidth=1, label="V peak")
    axV.set_ylabel("V (m$^3$)")
    axV.grid(True, alpha=0.3)

    # Volume change (right)
    axD.plot(phi_grid, dVdphi_c, color="C1", label="dV/d$\\phi$")
    axD.axhline(0, color="k", linewidth=0.8)
    axD.set_ylabel("dV/d$\\phi$")

    # one combined legend
    h1, l1 = axV.get_legend_handles_labels()
    h2, l2 = axD.get_legend_handles_labels()
    axV.legend(h1 + h2, l1 + l2, loc="best")

    axs[1].axvline(phi_peak, color="k", linestyle="--", linewidth=1)

    if dVdt_mv is not None:
        axs[1].plot(phi_grid, dVdt_mv, label="dV/dt (using bpm_mv)", alpha=0.7, linestyle="--")
    if "mv_fit_curve" in fit:
        y = np.asarray(fit["mv_fit_curve"], float)
        plot_curve_with_mask_alpha(
            ax=axs[1],
            x=phi_grid,
            y=y,
            mask=mv_mask,              # <-- EXACT fitting mask
            color="C2",
            label=f"MV, r_minor={fit['d_MV_minor']/2:.4g}, r_major={fit['d_MV_major']/2:.4g}",
            alpha_out=0.25,
            alpha_in=1.0,
            lw=1.5
        )
    if dVdt_av is not None:
        axs[1].plot(phi_grid, dVdt_av, label="dV/dt (using bpm_av)", alpha=0.7, linestyle="--")
    if "av_fit_curve_plot" in fit:
        y = np.asarray(fit["av_fit_curve_plot"], float)
        plot_curve_with_mask_alpha(
            ax=axs[1],
            x=phi_grid,
            y=y,
            mask=av_mask,              # <-- EXACT fitting mask
            color="C3",
            label=f"AV, r={fit['d_AV']/2:.4g}",
            alpha_out=0.25,
            alpha_in=1.0,
            lw=1.5
        )

    axs[1].axhline(0, color="k", linewidth=0.8)
    axs[1].set_xlabel("Cycle phase $\\phi$ in [0,1)")
    axs[1].set_ylabel("Volume Change")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="best")

    fig.suptitle(f"Case: {case_dir.name} (bpm_mv={bpm_mv}, bpm_av={bpm_av})")
    fig.savefig(os.path.join(case_dir, "calc_valve_diameters_outputs", f"{case_dir.name}_fitted.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return CaseData(
        name=case_dir.name,
        path=str(case_dir),
        params=params,
        bpm_mv=bpm_mv,
        bpm_av=bpm_av,
        av=av_df,
        mv=mv_df,
        volume=vol_df,
        fit=fit
    ), fig


def cases_to_jsonable(cases: Dict[str, CaseData]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for k, c in cases.items():
        d = asdict(c)
        for field in ("av", "mv", "volume"):
            df = getattr(c, field)
            d[field] = None if df is None else df.to_dict(orient="list")
        out[k] = d
    return out

def main_calc_diameter(root, n_phase):
    # `root` is the folder you point Blender at -- the one holding the mesh
    # frames (your '/STL/' export). inputPython.txt and the AV/MV CSVs may sit
    # there or one level up; resolve_case_dir figures out which. All outputs go
    # under the resolved case folder.
    stl_dir = Path(root)
    case_dir = resolve_case_dir(stl_dir)

    out_dir = case_dir / "calc_valve_diameters_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    figs = []
    summary_rows = []

    case, _ = process_case(case_dir, n_phase=n_phase, stl_dir=stl_dir)
    #figs.append(fig)
    summary_rows.append({
        "r_MV_minor": case.fit.get("d_MV_minor", np.nan)/2,
        "r_MV_major": case.fit.get("d_MV_major", np.nan)/2,
        "r_AV": case.fit.get("d_AV", np.nan)/2,
        "path": case.path,
        "bpm_mv": case.bpm_mv,
        "bpm_av": case.bpm_av,
        "A_MV": case.fit.get("A_MV", np.nan),
        "d_MV_minor": case.fit.get("d_MV_minor", np.nan),
        "d_MV_major": case.fit.get("d_MV_major", np.nan),
        "A_AV": case.fit.get("A_AV", np.nan),
        "d_AV": case.fit.get("d_AV", np.nan),
    })

    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)
    
    return figs, (case.fit.get("d_MV_minor", np.nan)/2, case.fit.get("d_MV_major", np.nan)/2, case.fit.get("d_AV", np.nan)/2)