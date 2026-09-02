"""
summary.py — End-of-run summary for DISSOLVE.

Prints a formatted terminal report and saves output/summary.png.
Called once from dissolve.main() after the time loop finishes.
"""

import csv
import os
import sys
import numpy as np


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_result(path: str):
    """Read result.txt → (times_h, mass_loss_pct) as numpy arrays."""
    try:
        data = np.loadtxt(path, skiprows=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data[:, 0], data[:, 1]
    except Exception:
        return np.array([0.0]), np.array([0.0])


def _load_diagnostics(path: str) -> dict:
    """Read diagnostics.csv → dict of series + scalar averages."""
    result: dict = {}
    try:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return result

        def col(key):
            return [float(r[key]) for r in rows if r.get(key, "")]

        result["t_series"]     = col("t_hours")
        result["dt_series"]    = col("dt_hours")
        result["iters_series"] = col("total_iters")

        def avg(key):
            v = col(key)
            return sum(v) / len(v) if v else 0.0

        result["avg_dt"]         = avg("dt_hours")
        result["avg_iters"]      = avg("total_iters")
        result["avg_zn"]         = avg("Zn_iters")
        result["avg_cl"]         = avg("Cl_iters")
        result["avg_film"]       = avg("Film_iters")
        result["avg_o2"]         = avg("O2_iters")
        result["avg_oh"]         = avg("OH_iters")
        result["avg_step_s"]     = avg("step_time_s")

        all_conv = [
            all(int(r[k]) for k in
                ["Zn_converged", "Cl_converged", "Film_converged",
                 "O2_converged", "OH_converged"])
            for r in rows
            if all(k in r for k in ["Zn_converged", "Cl_converged",
                                     "Film_converged", "O2_converged",
                                     "OH_converged"])
        ]
        result["conv_pct"] = 100.0 * sum(all_conv) / len(all_conv) if all_conv else 100.0

    except Exception:
        pass
    return result


def _count_vtk_frames(vtk_base: str) -> int:
    """Count .vtu files written to the output directory."""
    try:
        d = os.path.dirname(vtk_base) or "."
        return len([f for f in os.listdir(d) if f.endswith(".vtu")])
    except Exception:
        return 0


# ── Figure ────────────────────────────────────────────────────────────────────

def _save_figure(params, times, ml, diag, fig_path: str) -> bool:
    """
    Save a 3-panel summary figure (PNG).

    Layout
    ------
    Left (wide)   : Mass loss % vs time — the main result
    Top-right     : Degradation rate (%/h) vs time
    Bottom-right  : Adaptive time step (h) vs time
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.ticker import AutoMinorLocator
    except ImportError:
        return False

    if len(times) < 2:
        return False

    # ── Derived series ──────────────────────────────────────────────────
    rate      = np.gradient(ml, times)               # %/h
    dt_series = diag.get("dt_series", [])
    t_series  = diag.get("t_series",  [])

    # ── Colour palette ──────────────────────────────────────────────────
    C_ML   = "#1565C0"   # deep blue   — mass loss curve
    C_FILL = "#90CAF9"   # light blue  — area fill
    C_RATE = "#E65100"   # deep orange — rate curve
    C_DT   = "#2E7D32"   # dark green  — dt curve
    C_GRID = "#E0E0E0"   # light grey  — grid lines
    C_VTK  = "#B0BEC5"   # blue-grey   — VTK markers

    # ── Layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 7))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            width_ratios=[2.2, 1],
                            hspace=0.45, wspace=0.32)

    ax_ml   = fig.add_subplot(gs[:, 0])   # left — full height
    ax_rate = fig.add_subplot(gs[0, 1])   # top-right
    ax_dt   = fig.add_subplot(gs[1, 1])   # bottom-right

    from version import VERSION, APP_NAME
    fig.suptitle(
        f"{APP_NAME} v{VERSION}  —  Dissolve 2.0 Simulation Summary",
        fontsize=14, fontweight="bold", y=1.01,
    )

    # ── Helper ──────────────────────────────────────────────────────────
    def _style(ax, title, xlabel, ylabel):
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, color=C_GRID, linewidth=0.7, zorder=0)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(labelsize=9)
        ax.set_facecolor("#FAFAFA")
        for spine in ax.spines.values():
            spine.set_edgecolor("#BDBDBD")

    # ── Panel 1: Mass loss ──────────────────────────────────────────────
    ax_ml.fill_between(times, ml, alpha=0.18, color=C_FILL, zorder=1)
    ax_ml.plot(times, ml, color=C_ML, linewidth=2.5, zorder=3)

    # Mark VTK snapshot times (24 h boundaries)
    _vtk_int = getattr(params, "save_vtk_each_time", 24.0)
    if _vtk_int > 0:
        _vtk_ts = np.arange(0, times[-1] + 1e-6, _vtk_int)
        for vt in _vtk_ts:
            ax_ml.axvline(vt, color=C_VTK, linewidth=0.6,
                          linestyle="--", alpha=0.7, zorder=2)
        ax_ml.plot([], [], color=C_VTK, linewidth=0.8,
                   linestyle="--", label=f"VTK snapshot (every {_vtk_int:.0f} h)")
        ax_ml.legend(fontsize=9, loc="upper left",
                     framealpha=0.85, edgecolor="#BDBDBD")

    # Annotate final value
    ax_ml.annotate(
        f"{ml[-1]:.4f} %",
        xy=(times[-1], ml[-1]),
        xytext=(-70, 14), textcoords="offset points",
        fontsize=11, fontweight="bold", color=C_ML,
        arrowprops=dict(arrowstyle="->", color=C_ML, lw=1.5),
    )

    # Stats box (top-right of the panel)
    _peak_idx   = int(np.argmax(rate))
    _peak_rate  = rate[_peak_idx]
    _peak_t     = times[_peak_idx]
    _avg_rate   = ml[-1] / times[-1] if times[-1] > 0 else 0.0
    _avg_dt     = diag.get("avg_dt", 1.0)
    _vtk_count  = _count_vtk_frames(params.vtk_prefix)
    stats_txt = (
        f"Final loss  : {ml[-1]:.4f} %\n"
        f"Avg rate    : {_avg_rate:.3e} %/h\n"
        f"Peak rate   : {_peak_rate:.3e} %/h\n"
        f"  at t = {_peak_t:.0f} h\n"
        f"Avg dt      : {_avg_dt:.2f} h\n"
        f"VTK frames  : {_vtk_count}"
    )
    ax_ml.text(0.985, 0.03, stats_txt,
               transform=ax_ml.transAxes,
               ha="right", va="bottom",
               fontsize=9.5, fontfamily="monospace",
               bbox=dict(boxstyle="round,pad=0.55",
                         facecolor="white", edgecolor=C_ML,
                         linewidth=1.4, alpha=0.92))

    _style(ax_ml, "Scaffold Mass Loss", "Time (h)", "Mass Loss (%)")
    ax_ml.set_xlim(0, times[-1])
    ax_ml.set_ylim(bottom=0)

    # ── Panel 2: Degradation rate ───────────────────────────────────────
    ax_rate.plot(times, rate, color=C_RATE, linewidth=1.6)
    ax_rate.fill_between(times, rate, alpha=0.12, color=C_RATE)
    _style(ax_rate, "Degradation Rate", "Time (h)", "%/h")
    ax_rate.set_xlim(0, times[-1])

    # ── Panel 3: Time step (fixed) ──────────────────────────────────────
    if dt_series and t_series:
        ax_dt.step(t_series, dt_series, color=C_DT,
                   linewidth=1.6, where="post", zorder=3)
        ax_dt.axhline(params.dt_hours, color="#2E7D32", linewidth=0.9,
                      linestyle="--", alpha=0.7,
                      label=f"dt = {params.dt_hours} h (fixed)")
        ax_dt.legend(fontsize=8, framealpha=0.85, edgecolor="#BDBDBD")
        ax_dt.set_xlim(0, t_series[-1])
        ax_dt.set_ylim(0, params.dt_hours * 2)
    _style(ax_dt, "Time Step (fixed)", "Time (h)", "dt (h)")

    # ── Save ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(fig_path) or ".", exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    return True


# ── Terminal summary box ──────────────────────────────────────────────────────

def print_run_summary(params, Vinit: float, V_final: float,
                      total_steps: int, wall_time_s: float, comm) -> None:
    """Print end-of-run summary to terminal and save summary.png (rank 0 only)."""
    if comm.rank != 0:
        return

    _CY = "\033[96m"; _GR = "\033[92m"; _BO = "\033[1m"
    _DI = "\033[2m";  _RS = "\033[0m"
    W   = 65

    def row(text: str = "", cc: str = "") -> None:
        pad = W - len(text)
        if pad < 0:
            text, pad = text[:W], 0
        if cc:
            print(f"║{cc}{text}{_RS}{' ' * pad}║")
        else:
            print(f"║{text:<{W}}║")

    def div():
        print("╠" + "═" * W + "╣")

    # ── Load data ───────────────────────────────────────────────────────
    times, ml = _load_result(params.results_file)
    diag      = _load_diagnostics(params.diagnostics_file)
    vtk_n     = _count_vtk_frames(params.vtk_prefix)

    # ── Compute scalars ──────────────────────────────────────────────────
    ml_final   = float(ml[-1])  if len(ml)    else 0.0
    t_final    = float(times[-1]) if len(times) else params.sim_duration
    vol_lost   = max(0.0, Vinit - V_final)
    avg_rate   = ml_final / t_final if t_final > 0 else 0.0

    if len(ml) > 2 and len(times) > 2:
        rate      = np.gradient(ml, times)
        peak_idx  = int(np.argmax(rate))
        peak_rate = float(rate[peak_idx])
        peak_t    = float(times[peak_idx])
    else:
        peak_rate = avg_rate
        peak_t    = 0.0

    avg_dt   = diag.get("avg_dt",    params.dt_hours)
    avg_iter = diag.get("avg_iters", 0.0)
    conv_pct = diag.get("conv_pct",  100.0)

    h  = int(wall_time_s // 3600)
    m  = int((wall_time_s % 3600) // 60)
    s  = int(wall_time_s % 60)
    wt = f"{h}h {m}m {s}s" if h else f"{m}m {s}s"

    # ── Figure ───────────────────────────────────────────────────────────
    out_dir  = os.path.dirname(params.results_file) or "output"
    fig_path = os.path.join(out_dir, "summary.png")
    fig_ok   = _save_figure(params, times, ml, diag, fig_path)

    # ── Print box ────────────────────────────────────────────────────────
    print()
    print("╔" + "═" * W + "╗")
    row("  DISSOLVE — Simulation Complete", _CY + _BO)
    div()

    row("  Simulation")
    row(f"    Duration       :  {t_final:.1f} h  ({t_final/24:.1f} days)")
    row(f"    Steps          :  {total_steps}   (avg dt: {avg_dt:.2f} h)")
    row(f"    Wall time      :  {wt}")
    row()

    row("  Biodegradation", _BO)
    row(f"    Initial vol    :  {Vinit:.6f} mm³")
    row(f"    Final vol      :  {V_final:.6f} mm³")
    row(f"    Mass loss      :  {ml_final:.4f} %   ({vol_lost:.4e} mm³)", _GR)
    row(f"    Avg rate       :  {avg_rate:.4e} %/h")
    row(f"    Peak rate      :  {peak_rate:.4e} %/h   at t = {peak_t:.1f} h")
    row()

    row("  Solver")
    row(f"    Avg iters/step :  {avg_iter:.0f}  "
        f"(Zn:{diag.get('avg_zn',0):.0f}  "
        f"Cl:{diag.get('avg_cl',0):.0f}  "
        f"Film:{diag.get('avg_film',0):.0f}  "
        f"O2:{diag.get('avg_o2',0):.0f}  "
        f"OH:{diag.get('avg_oh',0):.0f})")
    row(f"    Converged      :  {conv_pct:.1f}%  of steps")
    row()

    row("  Output", _DI)
    row(f"    Mass loss data :  {params.results_file}", _DI)
    row(f"    3D snapshots   :  {params.vtk_prefix}.pvd  ({vtk_n} frames)", _DI)
    if params.write_diagnostics:
        row(f"    Diagnostics    :  {params.diagnostics_file}", _DI)
    if fig_ok:
        row(f"    Summary plot   :  {fig_path}", _DI)

    print("╚" + "═" * W + "╝")
    print()
    sys.stdout.flush()
