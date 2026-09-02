#!/usr/bin/env python3
"""
Dissolve 2.0-gpu — Degradation In Silico SOlver for biodegradable zinc alloys.

DOLFINx 0.10.0 + PETSc CUDA GPU offload.  Port of the FreeFem++ Dissolve
solver (dissolve.edp and its config/domain/physics/numerics/io modules --
https://github.com/sanjukadupul-wq/Dissolve-1.0/tree/main/Src%20Codes).

GPU strategy: when --use_gpu=1 and PETSc is built with CUDA, Cl/O2/OH transport
solves use aijcusparse matrices and cuda vectors so GMRES runs on the GPU.
Zn and Film stay on CPU because hypre BoomerAMG (essential at TGV≈1e8
conditioning) can't ingest aijcusparse.  Falls back to CPU sparse silently
if CUDA is unavailable.

Usage:
    python3 dissolve.py --input_mesh mesh_adaptive_800k.xdmf --sim_duration 24.0
    python3 dissolve.py --use_gpu 0  # force CPU
    mpirun -np 4 python3 dissolve.py --input_mesh mesh_adaptive_800k.xdmf
"""

import sys
import os
import csv
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import fem, io
import dolfinx.fem.petsc
import ufl

# ── Path bootstrap for the folder reorganization ────────────────────────────
# Every local module below still uses a flat, same-directory import (e.g.
# `from parameters import parse_args`) -- unchanged from before this file was
# split into config/domain/physics/numerics/gpu/io subfolders. This block is
# the only thing that makes those unmodified imports still resolve: it adds
# each subfolder to sys.path so Python finds them exactly as if everything
# were still flat. No other file in this project was edited for the move.
_ROOT = os.path.dirname(os.path.abspath(__file__))
for _subdir in ("config", "domain", "physics", "numerics", "gpu", "io"):
    sys.path.insert(0, os.path.join(_ROOT, _subdir))

from parameters import parse_args
from version import VERSION, FULL_NAME
from summary import print_run_summary
from mesh_utils import (convert_mesh_to_xdmf, load_mesh, initialize_phi,
                        initialize_phi_analytic,
                        redistance, compute_scaffold_volume,
                        compute_scaffold_volume_exact, compute_surface_area,
                        _compute_h_min, _compute_h_avg)
from weak_forms import (zinc_forms, chloride_forms, oxygen_forms,
                        hydroxide_forms, levelset_forms)
from solvers import solve_transport
from interface_velocity import (compute_effective_diffusion, compute_interface_velocity_paper,
                                compute_norm_grad_phi)
from output import save_step, print_step_info


# ── GPU detection ─────────────────────────────────────────────────────────

def _detect_petsc_cuda():
    """
    Test if PETSc was compiled with CUDA by trying to create a cuda Vec.
    This is the most reliable detection method — it exercises the actual code path.
    """
    try:
        v = PETSc.Vec().create(MPI.COMM_SELF)
        v.setType("cuda")
        v.setSizes(1)
        v.setUp()
        v.destroy()
        return True
    except Exception:
        return False


# ── Header ────────────────────────────────────────────────────────────────

def print_header(comm, gpu_mode: str):
    if comm.rank != 0:
        return

    # ── ANSI colour codes (supported by WSL2 / Linux terminals) ───────────
    _CY = "\033[96m"    # bright cyan   → ASCII art
    _BO = "\033[1m"     # bold           → tagline
    _DI = "\033[2m"     # dim            → metadata
    _RS = "\033[0m"     # reset

    # ── ASCII art — pyfiglet if installed, built-in fallback otherwise ─────
    # Override without editing code:  DISSOLVE_FONT=doom python3 dissolve.py ...
    _font = os.environ.get("DISSOLVE_FONT", "larry3d")
    try:
        import pyfiglet
        _raw = pyfiglet.figlet_format("DISSOLVE", font=_font)
        art = [ln.rstrip() for ln in _raw.splitlines() if ln.strip()]
    except Exception:
        # Built-in fallback — zero extra dependencies
        art = [
            "   ____   ___  ____  ____    ___   _    __     __ _____ ",
            "  |  _ \\ |_ _|/ ___|/ ___|  / _ \\ | |   \\ \\   / /| ____|",
            "  | | | | | | \\___ \\\\___ \\ | | | || |    \\ \\ / / |  _|  ",
            "  | |_| | | |  ___) ___) || |_| || |___   \\ V /  | |___ ",
            "  |____/ |___||____/|____/  \\___/ |_____|   \\_/   |_____|",
        ]

    # ── Auto-size box to fit the widest art line ───────────────────────────
    _indent = 2
    W = max(63, max(len(ln) for ln in art) + _indent + 2)

    def row(text: str = "", cc: str = "") -> None:
        """One ║…║ row. Padding uses visible length so ANSI codes don't shift the border."""
        pad = W - len(text)
        if pad < 0:
            text, pad = text[:W], 0
        if cc:
            print(f"║{cc}{text}{_RS}{' ' * pad}║")
        else:
            print(f"║{text:<{W}}║")

    # ── Draw ──────────────────────────────────────────────────────────────
    print("╔" + "═" * W + "╗")
    row()
    for ln in art:
        row(" " * _indent + ln, _CY)
    row()
    print("╠" + "═" * W + "╣")
    row(f"  {FULL_NAME}  ·  FEniCSx + PETSc CUDA", _BO)
    print("╠" + "═" * W + "╣")
    row(f"  GPU  :  {gpu_mode}", _DI)
    row(f"  Build:  v{VERSION}  ·  Dissolve 2.0 Simulator  ·  2025", _DI)
    row()
    print("╚" + "═" * W + "╝")
    sys.stdout.flush()


# ── Validation reporter ───────────────────────────────────────────────────

def _print_validation(errors: list, warnings: list, comm) -> bool:
    """Print a formatted validation report on rank 0.

    Returns True if the simulation may proceed (no errors).
    All ranks return the same value so callers can safely ``sys.exit``.
    """
    _RD = "\033[91m"   # bright red    — errors
    _YL = "\033[93m"   # yellow        — warnings
    _BO = "\033[1m"    # bold
    _DI = "\033[2m"    # dim
    _RS = "\033[0m"    # reset
    W   = 72           # matches logo box width; fits 80-col terminals

    def row(text: str = "", cc: str = "") -> None:
        pad = W - len(text)
        if pad < 0:
            text, pad = text[:W], 0
        if cc:
            print(f"║{cc}{text}{_RS}{' ' * pad}║")
        else:
            print(f"║{text:<{W}}║")

    if comm.rank == 0:
        # ── Warnings (non-fatal) ─────────────────────────────────────────
        if warnings:
            print()
            nw = len(warnings)
            print("╔" + "═" * W + "╗")
            row(f"  ⚠  DISSOLVE — {nw} warning{'s' if nw > 1 else ''}", _YL + _BO)
            print("╠" + "═" * W + "╣")
            for i, w in enumerate(warnings, 1):
                row(f"  [{i}] {w['msg']}", _YL)
            row()
            print("╚" + "═" * W + "╝")
            print()
            sys.stdout.flush()

        # ── Errors (fatal) ───────────────────────────────────────────────
        if errors:
            ne = len(errors)
            print("╔" + "═" * W + "╗")
            row(f"  ✗  DISSOLVE — {ne} error{'s' if ne > 1 else ''}"
                f" — simulation cannot start", _RD + _BO)
            print("╠" + "═" * W + "╣")
            for i, e in enumerate(errors, 1):
                row()
                row(f"  [{i}]  {e['msg']}", _RD)
                if e.get("fix"):
                    for fix_line in e["fix"].splitlines():
                        row(f"        → {fix_line}", _DI)
            row()
            print("╚" + "═" * W + "╝")
            print()
            sys.stdout.flush()

    # Broadcast error flag so every MPI rank exits together
    has_errors = bool(errors)
    has_errors = comm.bcast(has_errors, root=0)
    return not has_errors   # True = OK to proceed


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    comm     = MPI.COMM_WORLD
    params   = parse_args()
    dt_fixed = params.dt              # fixed timestep (adaptive dt removed)
    dt       = dt_fixed
    Tfinal   = params.sim_duration + dt

    # ── GPU detection ──────────────────────────────────────────────────────
    gpu_available = params.use_gpu and _detect_petsc_cuda()
    gpu_mode      = "PETSc CUDA" if gpu_available else "CPU (no CUDA)"

    print_header(comm, gpu_mode)

    # ── Validate all parameters before touching the mesh ──────────────────
    _errs, _warns = params.validate()
    if not _print_validation(_errs, _warns, comm):
        sys.exit(1)

    if comm.rank == 0:
        print(f"  mesh       : {params.input_mesh}")
        print(f"  sim_duration : {params.sim_duration} h   |   dt_hours : {params.dt} h")
        print(f"  GPU        : {gpu_mode}")
        print(f"  output     : {params.vtk_prefix}.pvd")
        sys.stdout.flush()

    # ── Load mesh ──────────────────────────────────────────────────────────
    if comm.rank == 0:
        print("Loading mesh...")

    if params.input_mesh.endswith(".mesh"):
        xdmf_path = convert_mesh_to_xdmf(params.input_mesh)
    else:
        xdmf_path = params.input_mesh

    domain, cell_tags = load_mesh(xdmf_path, comm)

    if comm.rank == 0:
        tdim      = domain.topology.dim
        num_cells = domain.topology.index_map(tdim).size_global
        num_verts = domain.topology.index_map(0).size_global
        print(f"Mesh loaded: {num_verts} vertices, {num_cells} cells")

    # ── Function space ─────────────────────────────────────────────────────
    V  = fem.functionspace(domain, ("Lagrange", 1))
    dx = ufl.Measure("dx", domain=domain)

    if comm.rank == 0:
        print(f"Function space DOF: {V.dofmap.index_map.size_global}")

    # ── GPU gradient assembler (dedicated, mesh-geometry only) ─────────────
    # Created once here so compute_interface_velocity_stefan can use GPU
    # gradient projection every step without depending on the solver cache
    # being warm.
    # Pre-warming (one nodal_gradient call) pays the CUDA buffer allocation
    # cost upfront rather than during the first timestep.
    _grad_asm = None
    if gpu_available:
        try:
            from gpu_assembler import GPUSparseAssembler
            if comm.rank == 0:
                print("Building GPU gradient assembler...")
            _grad_asm = GPUSparseAssembler(domain, V, comm=comm)
            _dummy = _grad_asm.nodal_gradient(np.zeros(V.dofmap.index_map.size_local))
            del _dummy
            if comm.rank == 0:
                print("GPU gradient assembler ready.")
        except Exception as e:
            if comm.rank == 0:
                print(f"GPU gradient assembler unavailable ({e}); using CPU projection.")
            _grad_asm = None

    # Register _grad_asm as the shared geometry source for all field assemblers.
    # Each subsequent GPUSparseAssembler (Zn, Cl, Film, O2, OH) will borrow
    # d_elem_nodes / d_detJ / d_Gflat / d_JinvT / d_indptr / d_indices
    # instead of re-uploading ~150 MB of identical data → saves ~750 MB total.
    if _grad_asm is not None:
        from solvers import set_shared_geometry
        set_shared_geometry(_grad_asm)
        if comm.rank == 0:
            print("GPU shared geometry registered — field assemblers will reuse geometry arrays.")

    # ── Level-set initialisation ───────────────────────────────────────────
    if getattr(params, "analytic_sdf", ""):
        if params.analytic_sdf == "disc":
            from mesh_generator_disc_adaptive import sdf_disc as _sdf_fn
        elif params.analytic_sdf == "stent":
            from mesh_generator_stent_adaptive import sdf_stent as _sdf_fn
        else:
            raise ValueError(f"Unknown analytic_sdf '{params.analytic_sdf}' "
                              f"(expected 'disc' or 'stent')")
        phi = initialize_phi_analytic(V, _sdf_fn)
    else:
        phi = initialize_phi(domain, cell_tags, params, V)

    if params.enable_redistance:
        if comm.rank == 0:
            print("Initial redistancing (15 iters)...")
        redistance(phi, domain, num_iters=15, gpu_asm=_grad_asm)

    # ── Solution functions ─────────────────────────────────────────────────
    Zn     = fem.Function(V, name="Zn")
    F_film = fem.Function(V, name="F")
    Cl     = fem.Function(V, name="Cl")
    OH     = fem.Function(V, name="OH")
    O2     = fem.Function(V, name="O2")

    phi_old = fem.Function(V, name="phi_old")
    Zn_old  = fem.Function(V, name="Zn_old")
    F_old   = fem.Function(V, name="F_old")
    Cl_old  = fem.Function(V, name="Cl_old")
    OH_old  = fem.Function(V, name="OH_old")
    O2_old  = fem.Function(V, name="O2_old")

    DeZn = fem.Function(V, name="DeZn")
    DeCl = fem.Function(V, name="DeCl")
    DeOH = fem.Function(V, name="DeOH")
    DeO2 = fem.Function(V, name="DeO2")

    v_interface = fem.Function(V, name="v")
    normgradphi = fem.Function(V, name="normgradphi")

    # ── Element sizes ──────────────────────────────────────────────────────
    h = _compute_h_min(domain)
    if h < 1e-6:
        h = params.h_min_fallback
    h_avg = _compute_h_avg(domain)
    if comm.rank == 0:
        print(f"Element size: h_min={h:.6f}  h_avg={h_avg:.6f}")

    # Interface velocity (compute_interface_velocity_paper, Eqs 13-16) uses the
    # FE nodal concentration gradient dotted with the interface normal -- no
    # neighbor-map / off-grid probe needed (the old two-point probe is gone).

    # ── Initial conditions ─────────────────────────────────────────────────
    phi_arr = phi.x.array
    # Smoothed Heaviside function to avoid jagged t=0 visualization on P1 mesh
    eps_h = 1.5 * h_avg
    H_phi = np.where(phi_arr > eps_h, 1.0, 
                     np.where(phi_arr < -eps_h, 0.0, 
                              0.5 * (1.0 + phi_arr / eps_h + np.sin(np.pi * phi_arr / eps_h) / np.pi)))
                              
    Zn.x.array[:]     = H_phi * params.Znbc
    F_film.x.array[:] = 0.0
    Cl.x.array[:]     = (1.0 - H_phi) * params.cl0
    OH.x.array[:]     = (1.0 - H_phi) * params.oh0
    O2.x.array[:]     = (1.0 - H_phi) * params.o2_initial
    O2_old.x.array[:] = O2.x.array[:]

    for fn in [Zn, F_film, Cl, OH, O2, O2_old]:
        fn.x.scatter_forward()

    DeZn.x.array[:] = params.diff_zn
    DeCl.x.array[:] = params.diff_cl
    DeOH.x.array[:] = params.diff_oh
    DeO2.x.array[:] = params.diff_o2
    for fn in [DeZn, DeCl, DeOH, DeO2]:
        fn.x.scatter_forward()

    def _scaffold_vol(phi_fn):
        if params.use_exact_volume:
            return compute_scaffold_volume_exact(phi_fn, domain)
        return compute_scaffold_volume(phi_fn, domain, gpu_asm=_grad_asm)

    Vinit   = _scaffold_vol(phi)
    _cons_cum = [0.0]    # cumulative O2 consumed by the ORR sink [g]
    _inv0     = [None]   # initial total O2 inventory in the domain [g]
    # ORR delta band half-width. FreeFem's int2d(levelset=phi) is zero-thickness;
    # this is the regularised stand-in, so smaller = closer to Stage-1 parity.
    epsilon = params.orr_eps_factor * h_avg

    if comm.rank == 0:
        print(f"Initial scaffold volume: {Vinit:.6f}")
        print(f"ORR sink band: epsilon = {params.orr_eps_factor} * h_avg = "
              f"{epsilon:.6f} mm  (full band {2*epsilon:.6f} mm)")

    # ── Boundary conditions (O2 Dirichlet on exterior) ─────────────────────
    # FreeFem reference feeds O2 only from the Inlet face: on(Inlet, O2=O2initial).
    # The 800k adaptive mesh lost its facet tags, so this port pins O2=bulk on ALL
    # exterior facets — over-supplying O2 and preventing interface depletion.
    # o2_bc_mode controls this:
    #   "all"   -> O2=bulk on every exterior facet (legacy port behaviour)
    #   "none"  -> no O2 Dirichlet; O2 only from initial condition + sink (max depletion)
    #   "xmin"  -> O2=bulk only on the min-x face
    #   "ztop"  -> O2=bulk only on the max-z (TOP) face, no-flux elsewhere.
    #             This matches the paper's open-air replenishment from the top
    #             surface, so that as the film restricts transport the O2 path to
    #             the buried interface gets starved -> reaction->diffusion
    #             transition -> the decelerating (biphasic) curve.
    o2_bcs = []
    _o2_mode = getattr(params, "o2_bc_mode", "all")
    try:
        fdim = domain.topology.dim - 1
        domain.topology.create_connectivity(fdim, domain.topology.dim)
        try:
            boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
        except AttributeError:
            boundary_facets = dolfinx.mesh.exterior_facets(domain)

        def _keep_face(axis, side):
            """Keep exterior facets whose vertices all lie on the min/max `axis` face."""
            coords = domain.geometry.x
            fmap = domain.topology.connectivity(fdim, 0)
            lo = comm.allreduce(coords[:, axis].min(), op=MPI.MIN)
            hi = comm.allreduce(coords[:, axis].max(), op=MPI.MAX)
            tol = 1e-3 * (hi - lo + 1e-30)
            target = lo if side == "min" else hi
            keep = []
            for f in boundary_facets:
                verts = fmap.links(f)
                if side == "min" and np.all(coords[verts, axis] <= target + tol):
                    keep.append(f)
                elif side == "max" and np.all(coords[verts, axis] >= target - tol):
                    keep.append(f)
            return np.array(keep, dtype=np.int32)

        if _o2_mode == "none":
            boundary_facets = np.array([], dtype=np.int32)
        elif _o2_mode == "xmin":
            boundary_facets = _keep_face(0, "min")
        elif _o2_mode == "ztop":
            boundary_facets = _keep_face(2, "max")   # max-z = top surface

        if len(boundary_facets) > 0:
            boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
            o2_bc  = fem.dirichletbc(PETSc.ScalarType(params.o2_initial), boundary_dofs, V)
            o2_bcs = [o2_bc]
        if comm.rank == 0:
            print(f"Applied O2 Dirichlet BC (mode={_o2_mode}) on {len(boundary_facets)} boundary facets")
    except Exception as e:
        if comm.rank == 0:
            print(f"Warning: Could not set up BCs: {e}")

    # Extract DOF indices once — reused every step by the GPU RHS kernel.
    # This lets O2 always take the GPU RHS path (_solve_gpu_asm with u_old)
    # regardless of BC mode, eliminating the DOLFINx CPU assembly fallback
    # that was triggered whenever o2_bcs was non-empty (e.g. ztop mode).
    _o2_bc_dofs  = (o2_bcs[0].dof_indices()[0].astype(np.int32)
                    if o2_bcs else np.array([], dtype=np.int32))
    _o2_bc_value = float(params.o2_initial)

    # ── XDMF output ───────────────────────────────────────────────────────
    xdmf_file = None
    vtk_file  = None
    # Either flag turns on visualization writing — they're aliases.
    _want_vtu = getattr(params, "write_vtu", True) or getattr(params, "emit_vtk", True)
    if _want_vtu:
        vtk_dir = os.path.dirname(params.vtk_prefix)
        if vtk_dir and comm.rank == 0:
            os.makedirs(vtk_dir, exist_ok=True)
        comm.Barrier()

        # PRIMARY: .pvd + per-step .vtu (universal ParaView/VisIt format).
        # DOLFINx 0.10's XDMFFile writes a non-standard layout that ParaView
        # can't parse, so we use VTK as the only reliable output.  Each .vtu
        # contains all 6/7 P1 fields as point data for that timestep.
        try:
            vtk_file = io.VTKFile(comm,
                                  params.vtk_prefix + ".pvd", "w")
            if comm.rank == 0:
                print(f"[output] VTU time series: {params.vtk_prefix}.pvd")
                print(f"[output]   → open the .pvd in ParaView; scrub the time slider.")
        except Exception as e:
            if comm.rank == 0:
                print(f"[output] VTU writer unavailable ({e})")
            vtk_file = None

        # OPTIONAL (off by default): also write XDMF for compactness.
        # Currently broken in DOLFINx 0.10 — keep flag for future when
        # XDMFFile.write_function semantics get fixed upstream.
        if getattr(params, "write_xdmf", False):
            try:
                xdmf_file = io.XDMFFile(
                    comm, params.vtk_prefix + ".xdmf", "w",
                    encoding=io.XDMFFile.Encoding.HDF5,
                )
                xdmf_file.write_mesh(domain)
                if comm.rank == 0:
                    print(f"[output] XDMF time series: {params.vtk_prefix}.xdmf")
                    print(f"[output]   WARNING: DOLFINx 0.10 XDMF layout is not")
                    print(f"[output]   ParaView-compatible; use the .pvd instead.")
            except Exception:
                xdmf_file = None

    if comm.rank == 0:
        if gpu_available:
            print("Solver mode: PETSc CUDA sparse (aijcusparse + cuda vectors)")
        else:
            print("Solver mode: PETSc CPU sparse")
        sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════════════
    # MAIN TIME LOOP
    # ══════════════════════════════════════════════════════════════════════
    if comm.rank == 0:
        print("\n" + "=" * 61)
        print("Starting main time loop...")
        print("=" * 61 + "\n")
        sys.stdout.flush()

    _t_run_start      = time.time()                         # wall-clock for summary
    count             = 0
    t                 = 0.0
    t_last_redistance = -params.redistance_time_interval   # force redistance on first step
    _vtk_interval     = params.save_vtk_each_time           # 0 = step-count mode
    t_last_vtk        = -_vtk_interval if _vtk_interval > 0 else 0.0  # force VTK at t=0

    import os as _os
    _LOOP_PROF = _os.environ.get("LOOP_PROF", "0") == "1"
    def _tprt(label, t_start):
        if _LOOP_PROF and comm.rank == 0:
            print(f"    [LOOP] {label}: {time.time()-t_start:.3f}s")

    # ── Diagnostics CSV setup ──────────────────────────────────────────────
    _diag_writer = None
    _diag_file   = None
    if params.write_diagnostics and comm.rank == 0:
        _diag_dir = os.path.dirname(params.diagnostics_file)
        if _diag_dir:
            os.makedirs(_diag_dir, exist_ok=True)
        _diag_file = open(params.diagnostics_file, "w", newline="")
        _diag_writer = csv.writer(_diag_file)
        _diag_writer.writerow([
            "step", "t_hours", "dt_hours",
            "scaffold_vol", "vol_loss_pct",
            "Zn_iters", "Zn_time_s", "Zn_converged",
            "Cl_iters", "Cl_time_s", "Cl_converged",
            "Film_iters", "Film_time_s", "Film_converged",
            "O2_iters", "O2_time_s", "O2_converged",
            "OH_iters", "OH_time_s", "OH_converged",
            "total_iters", "step_time_s",
            "Zn_l2norm", "Cl_l2norm", "O2_l2norm",
            "delta_Zn_rel", "delta_Cl_rel", "delta_O2_rel",
            "De_min", "De_max",
            # --- interface diagnostics ---
            "O2_iface_mean", "O2_iface_min", "v_iface_mean_abs",
            "pH_mean", "pH_min",
            "Da_mean",  # Damköhler = k_orr * h / D_eff_O2; >1 = diffusion-limited (decel), <1 = reaction-limited (linear)
            # --- topology-comparison degradation metrics ---
            "surface_area_mm2",     # S(t): interface area ∫δ(φ)|∇φ|dΩ
            "S_over_V_permm",       # specific surface S/V (drives surface-controlled corrosion)
            "penetration_mm_yr",    # volume-based penetration rate = (dV/dt)/S, in mm/year
            "v_iface_cov",          # corrosion UNIFORMITY: std/mean of |v_iface| (higher = more localized)
            "v_iface_max_abs",      # worst-case local recession rate (localized attack)
        ])

    # ── GPU memory baseline ────────────────────────────────────────────────
    try:
        import cupy as _cp_mem
        _gpu_mem_available = True
    except ImportError:
        _gpu_mem_available = False

    _vol_prev = None   # previous-step scaffold volume, for penetration-rate metric
    _mech_writer = None; _mech_file = None; _E_star_0 = None   # Phase-2 mechanics

    while t < Tfinal:
        t0_step = time.time()

        # Fixed timestep: reset every iteration so the clamps below only affect
        # the current step (they never permanently shrink dt).
        dt = dt_fixed

        # Clamp dt so the last step lands exactly on sim_duration, not past it.
        remaining = params.sim_duration - t
        if 0.0 < remaining < dt:
            dt = remaining

        # Snap dt to land exactly on the next VTK time boundary
        # (guarantees a step always falls on t=24, 48, 72, …).
        if _vtk_interval > 0:
            _next_vtk_t = (int(t / _vtk_interval + 1e-9) + 1) * _vtk_interval
            _to_vtk     = _next_vtk_t - t
            if 0.0 < _to_vtk < dt:
                dt = _to_vtk

        # Store old values
        _t = time.time()
        phi_old.x.array[:] = phi.x.array[:]
        Zn_old.x.array[:]  = Zn.x.array[:]
        F_old.x.array[:]   = F_film.x.array[:]
        Cl_old.x.array[:]  = Cl.x.array[:]
        OH_old.x.array[:]  = OH.x.array[:]
        O2_old.x.array[:]  = O2.x.array[:]
        _tprt("snapshot_old_values", _t)

        _t = time.time()
        scaffold_vol = _scaffold_vol(phi)
        _tprt("compute_scaffold_volume", _t)
        if count == 0:
            Vinit = scaffold_vol

        print_step_info(t, count, Vinit, scaffold_vol, params, comm)

        if Vinit > 1e-12 and (Vinit - scaffold_vol) / Vinit >= 0.98:
            if comm.rank == 0:
                print("The part is fully degraded!")
            break

        # Decide whether to write a VTK snapshot this step.
        # Time-based (save_vtk_each_time > 0): save when t has reached or
        # passed the next 24 h boundary — always true because dt was snapped.
        # Step-count fallback (save_vtk_each_time = 0): let output.py decide.
        if _vtk_interval > 0:
            _do_vtk = (t - t_last_vtk) >= _vtk_interval - 1e-9
            if _do_vtk:
                t_last_vtk = t
                if comm.rank == 0:
                    print(f"  [VTK] Saving snapshot at t={t:.1f} h")
        else:
            _do_vtk = None   # output.py uses vis_each_steps

        _t = time.time()
        save_step(domain, phi, Zn, F_film, Cl, OH, O2, params,
                  t, count, Vinit, xdmf_file,
                  vtk_file=vtk_file, v_interface=v_interface,
                  do_vtk=_do_vtk, gpu_asm=_grad_asm, scaffold_vol=scaffold_vol)

        # ── Phase-2 mechanics: 3D FE apparent modulus E*(t) on the {phi>0}
        # scaffold, at the VTK-snapshot cadence.  All ranks (create_submesh is
        # collective); guarded so a mechanics failure never stops degradation.
        if params.mechanics and (_do_vtk or count == 0):
            try:
                from mechanics_fe import scaffold_stiffness
                _mech = scaffold_stiffness(
                    domain, phi,
                    E_solid=params.mech_E_solid, nu=params.mech_nu,
                    axis=params.mech_axis, strain=params.mech_strain,
                    sigma_yield=params.mech_yield)
                if comm.rank == 0:
                    if _mech_writer is None:
                        _md = os.path.dirname(params.mechanics_file)
                        if _md:
                            os.makedirs(_md, exist_ok=True)
                        _mech_file = open(params.mechanics_file, "w", newline="")
                        _mech_writer = csv.writer(_mech_file)
                        _mech_writer.writerow([
                            "step", "t_hours", "scaffold_vol_mm3",
                            "E_star_MPa", "E_star_GPa", "E_retention",
                            "sigma_vm_max_MPa", "connected", "yielded"])
                    _Es = _mech["E_star_MPa"]
                    if _E_star_0 is None and _Es > 0:
                        _E_star_0 = _Es
                    _ret = (_Es / _E_star_0) if _E_star_0 else 0.0
                    _mech_writer.writerow([
                        count, f"{t:.4f}", f"{_mech['solid_vol_mm3']:.4f}",
                        f"{_Es:.2f}", f"{_Es/1000:.4f}", f"{_ret:.4f}",
                        f"{_mech['sigma_vm_max_MPa']:.3f}",
                        int(_mech["connected"]), int(_mech.get("yielded", False))])
                    _mech_file.flush()
                    print(f"  [MECH] t={t:.1f}h  E*={_Es/1000:.3f} GPa  "
                          f"retention={_ret*100:.1f}%  connected={_mech['connected']}")
            except Exception as _me:
                if comm.rank == 0:
                    print(f"  [MECH] skipped at t={t:.1f}h: {_me}")
        _tprt("save_step", _t)

        # ── Effective diffusion + interface velocity ───────────────────
        _t = time.time()
        compute_effective_diffusion(phi, F_film, params, DeZn, DeCl, DeOH, DeO2)
        _tprt("compute_effective_diffusion", _t)

        # Exact (marching-tetrahedra) phi=0 ORR quantities, computed ONCE per
        # step from the current O2 field so the SAME per-cell value drives
        # both the velocity update (below) and the O2/OH sinks (later this
        # step, before O2 is re-solved) -- see interface_velocity.py's
        # compute_exact_orr_quantities docstring for why this closes the
        # conservation gap the smeared delta_eps(phi) coupling had.
        _exact_orr = None
        if params.use_exact_orr:
            from interface_velocity import compute_exact_orr_quantities
            _exact_orr = compute_exact_orr_quantities(phi, O2, params.k_orr)

        _t = time.time()
        compute_interface_velocity_paper(
            phi, Zn, O2, DeZn, DeO2, v_interface, params, h_avg, t,
            gpu_asm=_grad_asm,
            exact_O2_override=_exact_orr)
        _tprt("compute_interface_velocity_paper", _t)

        # ── Level-set (algebraic Euler update) ────────────────────────
        # phi > 0 = scaffold; v_interface < 0 for shrinkage
        if params.enable_levelset:
            if comm.rank == 0:
                print("Updating level set field (Algebraic)...")
            _t = time.time()
            phi.x.array[:] = phi_old.x.array[:] + v_interface.x.array[:] * dt
            phi.x.scatter_forward()
            _tprt("level_set_update", _t)

        # Periodic redistancing restores |∇φ| ≈ 1 after algebraic drift.
        # Fixed dt: redistance every redistance_interval steps.
        _do_redistance = params.enable_redistance and (
            count % params.redistance_interval == 0
        )
        if _do_redistance:
            if comm.rank == 0:
                since = t - t_last_redistance
                print(f"Redistancing level-set (5 iters, {since:.2f}h since last)...")
            if getattr(params, "debug_vn_integral", False):
                from mechanics_fe import _solid_fraction
                _cn_dbg = V.dofmap.list
                _xg_dbg = domain.geometry.x
                _vtet_dbg = _xg_dbg[_cn_dbg]
                _tv_dbg = np.abs(np.einsum('ij,ij->i', _vtet_dbg[:,1]-_vtet_dbg[:,0],
                          np.cross(_vtet_dbg[:,2]-_vtet_dbg[:,0], _vtet_dbg[:,3]-_vtet_dbg[:,0]))) / 6.0
                _vol_before = comm.allreduce(float((_solid_fraction(phi.x.array[_cn_dbg])*_tv_dbg).sum()), op=MPI.SUM)
            _t = time.time()
            redistance(phi, domain, num_iters=params.redistance_iters, gpu_asm=_grad_asm)
            t_last_redistance = t
            _tprt("redistance", _t)
            if getattr(params, "debug_vn_integral", False):
                _vol_after = comm.allreduce(float((_solid_fraction(phi.x.array[_cn_dbg])*_tv_dbg).sum()), op=MPI.SUM)
                if comm.rank == 0:
                    print(f"  [REDIST-CHECK] t={t:.1f}h  EXACT_vol before={_vol_before:.6f}  "
                          f"after={_vol_after:.6f}  delta={_vol_after-_vol_before:+.6f}")

            # Warm-start for all solvers becomes stale after φ changes (because
            # medium/scaffold masks shift).  Zero out cached solution vectors so
            # the next KSP solve starts fresh instead of thrashing.
            from solvers import _GPU_ASM_CACHE, _GPU_CACHE
            for _entry in _GPU_ASM_CACHE.values():
                _entry["x"].set(0.0)
            for _entry in _GPU_CACHE.values():
                _entry["x"].set(0.0)

        # Coarea factor |∇φ| for the ORR surface integral (uses the current,
        # redistanced φ that the transport solves below also use).
        compute_norm_grad_phi(phi, normgradphi, gpu_asm=_grad_asm)

        # ── Per-step result accumulators ──────────────────────────────
        _res_zn   = {"converged": True, "iterations": 0, "time_s": 0.0}
        _res_cl   = {"converged": True, "iterations": 0, "time_s": 0.0}
        _res_film = {"converged": True, "iterations": 0, "time_s": 0.0}
        _res_o2   = {"converged": True, "iterations": 0, "time_s": 0.0}
        _res_oh   = {"converged": True, "iterations": 0, "time_s": 0.0}

        # ── Zinc ──────────────────────────────────────────────────────
        if params.enable_zn:
            if comm.rank == 0:
                print("Solving Zn concentration equation...")
            a_zn, L_zn = zinc_forms(V, Zn_old, phi, DeZn, F_film, Cl,
                                     dt, params.kf, params.kd, params.Fmax,
                                     params.Znbc, params.TGV, dx)
            if gpu_available:
                import cupy as _cp
                _phi_d  = _cp.asarray(phi.x.array)
                _F_d    = _cp.asarray(F_film.x.array)
                _Cl_d   = _cp.asarray(Cl.x.array)
                scaffold_dofs = _cp.where(_phi_d > 0.0)[0].astype(_cp.int32).get()
                _medium_zn_d  = _cp.where(_phi_d <= 0.0, 1.0, 0.0)
                _alpha_zn = _medium_zn_d * params.kf * (1.0 - _F_d / params.Fmax)
                _src_zn   = _medium_zn_d * params.kd * _F_d * _Cl_d ** 2
            else:
                scaffold_dofs = np.where(phi.x.array > 0.0)[0].astype(np.int32)
                _medium_zn = np.where(phi.x.array <= 0.0, 1.0, 0.0)
                _alpha_zn  = _medium_zn * params.kf * (
                    1.0 - F_film.x.array / params.Fmax)
                _src_zn    = (_medium_zn * params.kd
                              * F_film.x.array * Cl.x.array ** 2)
            zn_bc = fem.dirichletbc(
                PETSc.ScalarType(params.Znbc), scaffold_dofs, V
            )
            DeZn._solver_inv_dt  = 1.0 / dt
            DeZn._solver_bc_dofs = scaffold_dofs
            _res_zn = solve_transport("Zn", a_zn, L_zn, Zn, bcs=[zn_bc],
                                      ksp_type="gmres", pc_type="bjacobi",
                                      rtol=1e-8, max_it=2000, comm=comm,
                                      use_gpu=gpu_available,
                                      de_field=DeZn,
                                      alpha_field=_alpha_zn,
                                      u_old=Zn_old,
                                      source_field=_src_zn,
                                      bc_value=float(params.Znbc))
            if comm.rank == 0:
                print(f"  Converged: {_res_zn['converged']}, iters: {_res_zn['iterations']}, "
                      f"time: {_res_zn['time_s']:.2f}s")

        # ── Chloride ──────────────────────────────────────────────────
        if params.enable_cl:
            if comm.rank == 0:
                print("Solving Cl ion concentration equation...")
            a_cl, L_cl = chloride_forms(V, Cl_old, phi, DeCl, dt, dx)
            DeCl._solver_inv_dt = 1.0 / dt        # tag for the GPU assembler
            # Cl L = (Cl_old/dt · v) dx     (no source term)
            _res_cl = solve_transport("Cl", a_cl, L_cl, Cl,
                                      rtol=1e-6, comm=comm,
                                      use_gpu=gpu_available,
                                      de_field=DeCl,
                                      u_old=Cl_old)   # → GPU RHS
            if comm.rank == 0:
                print(f"  Converged: {_res_cl['converged']}, iters: {_res_cl['iterations']}, "
                      f"time: {_res_cl['time_s']:.2f}s")

        # ── Film ──────────────────────────────────────────────────────
        if params.enable_film:
            if comm.rank == 0:
                print("Solving protective film formation equation...")
            _t0_film = time.time()
            # Film has no diffusion term — the weak form reduces to
            # (1/dt + α(x))·M·F = RHS, which decouples at every node.
            # Lumped-mass gives the exact pointwise solution in one pass;
            # no KSP is needed and this avoids ~6 GMRES iterations of overhead.
            if gpu_available:
                import cupy as _cp
                _xp      = _cp
                _phi_d   = _cp.asarray(phi.x.array)
                _Zn_d    = _cp.asarray(Zn.x.array)
                _Cl_d    = _cp.asarray(Cl.x.array)
                _Fold_d  = _cp.asarray(F_old.x.array)
            else:
                _xp      = np
                _phi_d   = phi.x.array
                _Zn_d    = Zn.x.array
                _Cl_d    = Cl.x.array
                _Fold_d  = F_old.x.array
            _inv_dt     = 1.0 / dt
            _medium_d   = _xp.where(_phi_d <= 0.0, 1.0, 0.0)
            _alpha_film = _medium_d * (params.kf * _Zn_d / params.Fmax
                                       + params.kd * _Cl_d ** 2)
            _src_film   = _medium_d * params.kf * _Zn_d
            _F_new      = _xp.clip(
                (_Fold_d * _inv_dt + _src_film) / (_inv_dt + _alpha_film),
                0.0, params.Fmax
            )
            F_film.x.array[:] = _cp.asnumpy(_F_new) if gpu_available else _F_new
            F_film.x.scatter_forward()
            _res_film = {"converged": True, "iterations": 0,
                         "time_s": time.time() - _t0_film}
            if comm.rank == 0:
                print(f"  Converged: {_res_film['converged']}, iters: {_res_film['iterations']} (direct), "
                      f"time: {_res_film['time_s']:.4f}s")

        # ── Oxygen (ORR, under-relaxation ω=0.3 for stability) ────────
        if getattr(params, "force_o2_bulk", False):
            # Diagnostic: skip the O2 transport PDE entirely, freezing O2 at
            # the bulk initial value everywhere (no depletion, ever). Isolates
            # whether the mass-loss shortfall is O2 transport/supply-limited
            # vs a reaction-kinetics/film-floor issue.
            O2.x.array[:] = params.o2_initial
            O2.x.scatter_forward()
            O2_old.x.array[:] = O2.x.array[:]
            _res_o2 = {"converged": True, "iterations": 0, "time_s": 0.0}
            if comm.rank == 0:
                print(f"  [force_o2_bulk] O2 frozen at {params.o2_initial:.4e} g/mm3 everywhere")
        elif params.enable_o2:
            if comm.rank == 0:
                print("Solving O2 concentration equation (ORR)...")
            O2_old.x.array[:] = O2.x.array[:]
            a_o2, L_o2 = oxygen_forms(V, O2_old, phi, DeO2, dt,
                                       params.k_orr, epsilon, dx,
                                       exact_mode=params.use_exact_orr)
            # O2 bilinear: a(O2,v) = (1/dt + δε·kORR)·M·v + DeO2·K·v.
            #   δε(φ) regularised delta (interface band)
            # Level-set surface delta with coarea factor |∇φ| (matches FreeFem++
            # int2d(levelset=phi)): ∫ f·δ_ε(φ)·|∇φ| dx = ∫_{φ=0} f dS.
            #
            # use_exact_orr: the reaction term moves OUT of the implicit matrix
            # (alpha_o2 stays 0 -- no delta_eps band) and becomes an explicit
            # NEGATIVE source (consumption) built from the exact phi=0 patch,
            # computed once per step above as _exact_orr.
            if params.use_exact_orr:
                _alpha_o2 = None
                _o2_src = -_exact_orr["dof_sink_density"]
            elif gpu_available:
                import cupy as _cp
                _phi_d_o2  = _cp.asarray(phi.x.array)
                _ngphi_d   = _cp.asarray(normgradphi.x.array)
                _delta_d   = _cp.where(_cp.abs(_phi_d_o2) < epsilon,
                                       (1.0 / (2.0 * epsilon)) *
                                       (1.0 + _cp.cos(_cp.pi * _phi_d_o2 / epsilon)),
                                       0.0) * _ngphi_d
                _alpha_o2  = _delta_d * params.k_orr
                _o2_src = None
            else:
                _phi_a    = phi.x.array
                _delta    = np.where(np.abs(_phi_a) < epsilon,
                                     (1.0 / (2.0 * epsilon)) *
                                     (1.0 + np.cos(np.pi * _phi_a / epsilon)),
                                     0.0) * normgradphi.x.array
                _alpha_o2 = _delta * params.k_orr
                _o2_src = None
            # O2 RHS: O2_old/dt · v (no source). Dirichlet on boundary facets
            # is still applied through the bcs= path inside _solve_gpu_asm
            # via DOLFINx assemble_vector + set_bc when source_field is given
            # AND bc_dofs from de_field — but we don't set bc_dofs for O2,
            # so the GPU-RHS path is used and DOLFINx isn't called.  The
            # boundary Dirichlet for O2 is therefore NOT enforced on GPU; we
            # fall back to DOLFINx RHS when bcs is non-empty.
            # BC is enforced via GPU RHS kernel (bc_dofs/bc_value path in
            # assemble_rhs), so bcs=[] here — no DOLFINx CPU assembly fallback.
            DeO2._solver_inv_dt  = 1.0 / dt
            DeO2._solver_bc_dofs = _o2_bc_dofs
            _res_o2 = solve_transport("O2", a_o2, L_o2, O2, bcs=[],
                                      ksp_type="gmres", pc_type="bjacobi",
                                      rtol=1e-6, under_relax=0.3, comm=comm,
                                      use_gpu=gpu_available,
                                      de_field=DeO2,
                                      alpha_field=_alpha_o2,
                                      u_old=O2_old,
                                      source_field=_o2_src,
                                      bc_value=_o2_bc_value)
            if comm.rank == 0:
                print(f"  Converged: {_res_o2['converged']}, iters: {_res_o2['iterations']}, "
                      f"time: {_res_o2['time_s']:.2f}s")

        # ── Hydroxide ─────────────────────────────────────────────────
        if params.enable_oh:
            if comm.rank == 0:
                print("Solving pH equation...")
            a_oh, L_oh = hydroxide_forms(V, OH_old, phi, DeOH, F_film, Cl, O2,
                                          dt, params.kd, params.k_orr, epsilon, dx,
                                          exact_mode=params.use_exact_orr)
            DeOH._solver_inv_dt = 1.0 / dt
            # OH RHS: OH_old/dt · v  +  medium·kd·F·Cl² · v  +  4·δε·kORR·O2 · v
            # (use_exact_orr: the last term is replaced by 4·dof_sink_density,
            #  the exact phi=0-patch equivalent -- same quantity consumed by O2.)
            if params.use_exact_orr:
                _medium_oh = np.where(phi.x.array <= 0.0, 1.0, 0.0)
                _src_oh = (_medium_oh * params.kd * F_film.x.array * Cl.x.array ** 2
                           + 4.0 * _exact_orr["dof_sink_density"])
            elif gpu_available:
                import cupy as _cp
                _phi_d_oh    = _cp.asarray(phi.x.array)
                _ngphi_d_oh  = _cp.asarray(normgradphi.x.array)
                _F_d_oh      = _cp.asarray(F_film.x.array)
                _Cl_d_oh     = _cp.asarray(Cl.x.array)
                _O2_d_oh     = _cp.asarray(O2.x.array)
                _medium_oh_d = _cp.where(_phi_d_oh <= 0.0, 1.0, 0.0)
                _delta_oh_d  = _cp.where(_cp.abs(_phi_d_oh) < epsilon,
                                         (1.0 / (2.0 * epsilon)) *
                                         (1.0 + _cp.cos(_cp.pi * _phi_d_oh / epsilon)),
                                         0.0) * _ngphi_d_oh
                _src_oh = (_medium_oh_d * params.kd * _F_d_oh * _Cl_d_oh ** 2
                           + 4.0 * _delta_oh_d * params.k_orr * _O2_d_oh)
            else:
                _medium_oh = np.where(phi.x.array <= 0.0, 1.0, 0.0)
                _phi_oh    = phi.x.array
                _delta_oh  = np.where(np.abs(_phi_oh) < epsilon,
                                      (1.0 / (2.0 * epsilon)) *
                                      (1.0 + np.cos(np.pi * _phi_oh / epsilon)),
                                      0.0) * normgradphi.x.array
                _src_oh    = (_medium_oh * params.kd * F_film.x.array * Cl.x.array ** 2
                              + 4.0 * _delta_oh * params.k_orr * O2.x.array)
            _res_oh = solve_transport("OH", a_oh, L_oh, OH,
                                      rtol=1e-6, comm=comm,
                                      use_gpu=gpu_available,
                                      de_field=DeOH,
                                      u_old=OH_old,
                                      source_field=_src_oh)
            if comm.rank == 0:
                print(f"  Converged: {_res_oh['converged']}, iters: {_res_oh['iterations']}, "
                      f"time: {_res_oh['time_s']:.2f}s")

        # Field-bounds diagnostic: which state variable goes illegal first,
        # ahead of a late-stage blowup. No positivity/Fmax clamping exists
        # anywhere in the O2/OH/film weak forms (linear variational solves),
        # so this checks whether that's actually the trigger.
        if getattr(params, "check_bounds", False) and comm.rank == 0:
            _o2_a = O2.x.array; _oh_a = OH.x.array; _f_a = F_film.x.array
            _zn_a = Zn.x.array
            print(f"  [BOUNDS] t={t:.1f}h  O2=[{_o2_a.min():.4e},{_o2_a.max():.4e}]  "
                  f"OH=[{_oh_a.min():.4e},{_oh_a.max():.4e}]  "
                  f"F=[{_f_a.min():.4e},{_f_a.max():.4e}]  "
                  f"F/Fmax_max={_f_a.max()/params.Fmax:.4f}  "
                  f"Zn=[{_zn_a.min():.4e},{_zn_a.max():.4e}]")

        count += 1
        t     += dt

        step_time = time.time() - t0_step

        # Aggregate KSP iterations for diagnostics/console output.
        _total_iters = (
            _res_zn["iterations"]   + _res_cl["iterations"] +
            _res_film["iterations"] + _res_o2["iterations"] +
            _res_oh["iterations"]
        )

        # Relative solution change per hour — logged to diagnostics CSV.
        # *_old still holds the previous step values here.
        def _rel_change(new, old, dt_h):
            n_new = float(np.linalg.norm(new))
            n_old = float(np.linalg.norm(old))
            return abs(n_new - n_old) / max(n_old, 1e-30) / dt_h
        _dZn_rel = _rel_change(Zn.x.array,     Zn_old.x.array,  dt)
        _dCl_rel = _rel_change(Cl.x.array,     Cl_old.x.array,  dt)
        _dO2_rel = _rel_change(O2.x.array,     O2_old.x.array,  dt)

        # Early calculation of pH and Da for console output
        _pKw = 13.6  # water ion product at 37°C (neutral pH = 6.8); oh0
                     # in parameters.py is set consistently for baseline pH 7.4.
        _oh_conc_mol_per_l = np.maximum(OH.x.array * 1e6 / 17.0, 1e-14)
        _pOH = -np.log10(_oh_conc_mol_per_l)
        _pH_arr = _pKw - _pOH
        # restrict pH stats to medium (φ≤0): scaffold nodes have OH≈0 → pH=−1 artifact
        _med = phi.x.array <= 0.0
        _pH_valid = _pH_arr[_med & np.isfinite(_pH_arr)] if _med.any() else _pH_arr[np.isfinite(_pH_arr)]
        _pH_mean = float(np.mean(_pH_valid)) if len(_pH_valid) > 0 else float(_pKw)
        _pH_min  = float(np.min(_pH_valid))  if len(_pH_valid) > 0 else float(_pKw)
        _band = np.abs(phi.x.array) < 3.0 * h_avg
        _DeO2_iface = DeO2.x.array[_band] if _band.any() else np.array([params.diff_o2])
        _DeO2_mean = float(np.mean(_DeO2_iface))
        # Dimensionless sub-grid film Damköhler, identical to interface_velocity.py:
        #   Da = k_orr[mm/h] * R_diff[h/mm],  R_diff = film_length / DeO2.
        # >1 => diffusion(film)-limited (deceleration); <1 => reaction-limited.
        _Da_mean = (float(params.k_orr * params.film_length / (_DeO2_mean + 1e-30))
                    if _DeO2_mean > 0 else 0.0)

        # ── topology-comparison metrics (ALL ranks: assemble_scalar allreduces) ──
        # Surface area S(t) and volume-based penetration rate (mm/year).  These are
        # the primary surface-controlled-corrosion descriptors for comparing
        # gyroid vs BCC at matched porosity (npj Mater. Degrad. framework).
        if params.write_diagnostics:
            _S_iface = compute_surface_area(phi, domain)
            _SV = _S_iface / scaffold_vol if scaffold_vol > 1e-12 else 0.0
            if _vol_prev is not None and _S_iface > 1e-12 and dt > 0:
                _pen_mm_yr = max(0.0, (_vol_prev - scaffold_vol) / dt / _S_iface) * 8760.0
            else:
                _pen_mm_yr = 0.0
            _vol_prev = scaffold_vol
        else:
            _S_iface = _SV = _pen_mm_yr = 0.0

        # ── Conservation audit: does consumed O2 actually become dissolved Zn? ──
        # Chain that MUST close (to numerical tolerance) if the ORR sink and the
        # recession law are consistent:
        #   O2 influx  ->  O2 consumed  ->  stoichiometric Zn  ->  Zn from interface motion
        # Zn + 1/2 O2 + H2O -> Zn(OH)2, i.e. 2 mol Zn per mol O2.
        # A large "consumed >> stoichiometric-from-motion" gap localises the defect
        # to the sink/velocity coupling rather than to transport or kinetics.
        if getattr(params, "check_conservation", False):
            _dxc  = ufl.Measure("dx", domain=domain)
            _gp   = ufl.grad(phi)
            _gm   = ufl.sqrt(ufl.inner(_gp, _gp) + 1e-12)
            _dl   = ufl.conditional(ufl.lt(ufl.algebra.Abs(phi), epsilon),
                                    (1.0/(2.0*epsilon))*(1.0 + ufl.cos(ufl.pi*phi/epsilon)), 0.0)
            # instantaneous sink rate [g/h] = ∫ δ|∇φ| k_orr O2 dΩ  (same form as the PDE sink)
            _sink_rate = comm.allreduce(
                fem.assemble_scalar(fem.form(_dl*_gm*params.k_orr*O2*_dxc)), op=MPI.SUM)
            _inv = comm.allreduce(fem.assemble_scalar(fem.form(O2*_dxc)), op=MPI.SUM)
            _cons_cum[0] += _sink_rate * dt                      # g O2 consumed
            if _inv0[0] is None:
                _inv0[0] = _inv
            # inventory balance: influx = Δinventory + consumed
            _influx_cum = (_inv - _inv0[0]) + _cons_cum[0]
            # stoichiometric Zn mass from consumed O2, and Zn mass actually lost
            _zn_stoich = (_cons_cum[0] / 31.998) * 2.0 * 65.38   # g Zn
            _zn_actual = (Vinit - scaffold_vol) * params.rho_zn
            if comm.rank == 0:
                _ratio = (_zn_actual / _zn_stoich) if _zn_stoich > 1e-30 else float('nan')
                print(f"  [cons] O2_in={_influx_cum:.4e}  O2_consumed={_cons_cum[0]:.4e} g | "
                      f"Zn_stoich={_zn_stoich:.4e}  Zn_actual={_zn_actual:.4e} g | "
                      f"actual/stoich={_ratio:.4f}")

        # ── DEBUG-ONLY: properly area-weighted v_n, independent of vol_loss ──
        # ∫ δ(φ)|∇φ| v_n dΩ / ∫ δ(φ)|∇φ| dΩ  -- same measure as compute_surface_area,
        # so this is NOT derived from scaffold_vol (unlike penetration_mm_yr) and is
        # a genuine independent check of the true area-weighted recession rate.
        if getattr(params, "debug_vn_integral", False):
            _eps_s = 2.0 * h_avg
            _dx = ufl.Measure("dx", domain=domain)
            _gphi = ufl.grad(phi)
            _gmag = ufl.sqrt(ufl.inner(_gphi, _gphi) + 1e-12)
            _delta = ufl.conditional(ufl.lt(ufl.algebra.Abs(phi), _eps_s),
                                     (1.0/(2.0*_eps_s))*(1.0 + ufl.cos(ufl.pi*phi/_eps_s)), 0.0)
            _S_dbg = comm.allreduce(fem.assemble_scalar(fem.form(_delta*_gmag*_dx)), op=MPI.SUM)
            _vn_int = comm.allreduce(fem.assemble_scalar(fem.form(_delta*_gmag*v_interface*_dx)), op=MPI.SUM)
            _vn_area_avg = _vn_int / _S_dbg if _S_dbg > 1e-12 else 0.0
            _band_dbg = np.abs(phi.x.array) < 3.0 * h_avg
            _naive_mean = float(np.abs(v_interface.x.array[_band_dbg]).mean()) if _band_dbg.any() else 0.0
            # EXACT (non-smoothed) volume: marching-tetrahedra solid fraction from
            # the raw sign of phi at each vertex -- no epsilon, no |grad phi|
            # sensitivity.  Cross-checks whether compute_scaffold_volume's SMOOTHED
            # Heaviside integral is faithfully tracking the true interface position.
            from mechanics_fe import _solid_fraction
            _cellnodes_dbg = V.dofmap.list
            _phiv_dbg = phi.x.array[_cellnodes_dbg]
            _rho_dbg = _solid_fraction(_phiv_dbg)
            _xg_dbg = domain.geometry.x
            _vtet = _xg_dbg[_cellnodes_dbg]
            _tetvol_dbg = np.abs(np.einsum('ij,ij->i', _vtet[:,1]-_vtet[:,0],
                                 np.cross(_vtet[:,2]-_vtet[:,0], _vtet[:,3]-_vtet[:,0]))) / 6.0
            _exact_vol = comm.allreduce(float((_rho_dbg * _tetvol_dbg).sum()), op=MPI.SUM)
            if comm.rank == 0:
                print(f"  [VN-CHECK] t={t:.1f}h  S={_S_dbg:.4f}  "
                      f"area-weighted_v_n={_vn_area_avg:.6e}  naive_mean|v|={_naive_mean:.6e}  "
                      f"smoothed_vol={scaffold_vol:.6f}  EXACT_vol={_exact_vol:.6f}")

        if comm.rank == 0:
            _vol_pct = 0.0 if Vinit < 1e-12 else (Vinit - scaffold_vol) / Vinit * 100.0
            print(f"  Step completed in {step_time:.2f}s  (KSP iters: {_total_iters})  "
                  f"pH: {_pH_mean:.2f}  Da: {_Da_mean:.3e}  Vol_loss: {_vol_pct:.3f}%\n")
            sys.stdout.flush()

        # ── Diagnostics CSV ───────────────────────────────────────────
        if _diag_writer is not None and comm.rank == 0:
            # _vol_pct, _pH_mean, _pH_min, _Da_mean already calculated above
            _de_all = np.concatenate([DeZn.x.array, DeCl.x.array, DeOH.x.array, DeO2.x.array])
            _de_all_pos = _de_all[_de_all > 0]
            _de_min = float(_de_all_pos.min()) if len(_de_all_pos) > 0 else 0.0
            _de_max = float(_de_all.max())

            # Interface diagnostics: is O2 depleting at the dissolving surface?
            # This is THE quantity that decides decelerating vs linear mass loss.
            if _band.any():
                _o2b = O2.x.array[_band]
                _o2_iface_mean = float(_o2b.mean())
                _o2_iface_min  = float(_o2b.min())
                _vb = np.abs(v_interface.x.array[_band])
                _v_iface_mean  = float(_vb.mean())
                # corrosion uniformity: CoV (=std/mean) and worst-case local rate
                _v_iface_cov   = float(_vb.std() / (_v_iface_mean + 1e-30))
                _v_iface_max   = float(_vb.max())
            else:
                _o2_iface_mean = _o2_iface_min = _v_iface_mean = 0.0
                _v_iface_cov = _v_iface_max = 0.0

            _diag_writer.writerow([
                count, f"{t:.4f}", f"{dt:.4f}",
                f"{scaffold_vol:.6e}", f"{_vol_pct:.3f}",
                _res_zn["iterations"],   f"{_res_zn['time_s']:.3f}",   int(_res_zn["converged"]),
                _res_cl["iterations"],   f"{_res_cl['time_s']:.3f}",   int(_res_cl["converged"]),
                _res_film["iterations"], f"{_res_film['time_s']:.3f}", int(_res_film["converged"]),
                _res_o2["iterations"],   f"{_res_o2['time_s']:.3f}",   int(_res_o2["converged"]),
                _res_oh["iterations"],   f"{_res_oh['time_s']:.3f}",   int(_res_oh["converged"]),
                _total_iters, f"{step_time:.3f}",
                f"{float(np.linalg.norm(Zn.x.array)):.6e}",
                f"{float(np.linalg.norm(Cl.x.array)):.6e}",
                f"{float(np.linalg.norm(O2.x.array)):.6e}",
                f"{_dZn_rel:.4e}", f"{_dCl_rel:.4e}", f"{_dO2_rel:.4e}",
                f"{_de_min:.6e}", f"{_de_max:.6e}",
                f"{_o2_iface_mean:.6e}", f"{_o2_iface_min:.6e}", f"{_v_iface_mean:.6e}",
                f"{_pH_mean:.2f}", f"{_pH_min:.2f}",
                f"{_Da_mean:.3e}",
                f"{_S_iface:.6e}", f"{_SV:.6e}", f"{_pen_mm_yr:.6e}",
                f"{_v_iface_cov:.6e}", f"{_v_iface_max:.6e}",
            ])
            _diag_file.flush()

        # ── GPU memory report (every 10 steps) ────────────────────────
        if _gpu_mem_available and count % 10 == 0 and comm.rank == 0:
            try:
                mem = _cp_mem.get_default_memory_pool()
                print(f"  [GPU mem] used={mem.used_bytes()/1024**3:.2f} GB  "
                      f"total={mem.total_bytes()/1024**3:.2f} GB")
            except Exception:
                pass

        comm.Barrier()

    # ── Cleanup ───────────────────────────────────────────────────────────
    if xdmf_file is not None:
        xdmf_file.close()
    if vtk_file is not None:
        vtk_file.close()
    if _diag_file is not None:
        _diag_file.close()

    print_run_summary(
        params,
        Vinit        = Vinit,
        V_final      = scaffold_vol,
        total_steps  = count,
        wall_time_s  = time.time() - _t_run_start,
        comm         = comm,
    )


if __name__ == "__main__":
    main()
