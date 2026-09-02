"""
solvers.py — Unified CPU + GPU solver for Dissolve 2.0 using DOLFINx 0.10.0.

Priority order per solve call:
  1. libCEED matrix-free GPU  — when ceed_op provided and libceed installed
  2. PETSc CUDA sparse        — when use_gpu=True and PETSc built with CUDA
  3. PETSc CPU sparse         — automatic fallback

Zn uses bjacobi on aijcusparse (dirichletbc replaced the old TGV penalty,
so conditioning is ~1e3 and AMG is no longer needed).
Film uses a direct pointwise solve (no KSP) — the equation has no diffusion
so the system decouples nodally and lumped-mass gives the exact solution.
(CEED path: MatShell cannot expose entries to AMG — Jacobi is used there.)
"""

import time
import numpy as np
from mpi4py import MPI
from dolfinx import fem
import dolfinx.fem.petsc
from petsc4py import PETSc

# When PETSc is built with CUDA but the MPI implementation isn't GPU-aware
# (conda mpich is not), this option silences the abort.  Safe for single-rank
# runs where no GPU-to-GPU MPI transfer happens.
PETSc.Options().setValue("use_gpu_aware_mpi", "0")

try:
    from ceed_operators import (libceed_available, make_ceed,
                                extract_ceed_mesh_data, CeedDiffusionMassOp)
    _HAS_CEED = libceed_available()
except ImportError:
    _HAS_CEED = False

_ceed_ctx   = None
_ceed_bname = None
_mesh_data  = None

# Shared GPU geometry — set once from dissolve.py after _grad_asm is built.
# All subsequent GPUSparseAssembler instances borrow these arrays instead of
# re-uploading ~150 MB of identical geometric data per field.
_SHARED_GEOM = None

def set_shared_geometry(asm) -> None:
    """Register the primary GPUSparseAssembler as the shared geometry source."""
    global _SHARED_GEOM
    _SHARED_GEOM = asm

# Persistent GPU solver cache, keyed by `name` from solve_transport().
# Each entry keeps the aijcusparse Mat + cuda Vecs + configured KSP alive across
# timesteps so we pay setup cost once instead of every solve.
#   {name: {"A": Mat, "b": Vec, "x": Vec, "ksp": KSP}}
_GPU_CACHE = {}

# Persistent cache for the cupy GPU assembler path (M/dt + De·K).
# Each entry: {"asm": GPUSparseAssembler, "b": Vec, "x": Vec, "ksp": KSP}
_GPU_ASM_CACHE = {}

# Persistent CPU solver cache for the hypre-AMG path.
# Re-using LinearProblem (and its KSP) saves dolfinx setup overhead per call;
# enabling KSPSetReusePreconditioner skips the BoomerAMG hierarchy rebuild
# (which dominates the Zn / Film solve time at ~2 s per call).
#   {name: {"problem": LinearProblem, "calls": int}}
_CPU_CACHE = {}
_CPU_PC_REBUILD_INTERVAL = 8   # rebuild AMG every Nth solve so it adapts


def _destroy_gpu_cache():
    """Destroy all cached PETSc/CUDA objects before PETSc finalization."""
    for entry in _GPU_CACHE.values():
        for key in ("ksp", "A", "b", "x"):
            obj = entry.get(key)
            if obj is not None:
                try:
                    obj.destroy()
                except Exception:
                    pass
    for entry in _GPU_ASM_CACHE.values():
        for key in ("ksp", "b", "x"):
            obj = entry.get(key)
            if obj is not None:
                try:
                    obj.destroy()
                except Exception:
                    pass
        # The assembler's mat is owned by asm; its destroy is implicit on GC.
    _GPU_CACHE.clear()
    _GPU_ASM_CACHE.clear()
    _CPU_CACHE.clear()      # LinearProblem manages its own PETSc objects


import atexit as _atexit
_atexit.register(_destroy_gpu_cache)


def _vec_copy_d2d(petsc_vec, cupy_arr):
    """Write a cupy device array directly into a PETSc CUDA Vec device buffer.

    Uses getCUDAHandle to alias the Vec's device pointer — eliminates the
    explicit D2H (cp.asnumpy) + H2D (getArray write) round-trip that the
    old code performed every step.  Falls back transparently if the PETSc
    build doesn't expose getCUDAHandle.
    """
    import cupy as _cp
    try:
        n   = int(cupy_arr.shape[0])
        ptr = petsc_vec.getCUDAHandle("w")
        mem = _cp.cuda.UnownedMemory(ptr, n * 8, petsc_vec)
        d_view = _cp.ndarray((n,), dtype=_cp.float64,
                             memptr=_cp.cuda.MemoryPointer(mem, 0))
        d_view[:] = cupy_arr          # D2D: stays on device
        petsc_vec.restoreCUDAHandle()
    except Exception:
        petsc_vec.getArray()[:] = _cp.asnumpy(cupy_arr)   # fallback


def _vec_read_d2h(petsc_vec, out_arr):
    """Read a PETSc CUDA Vec device buffer directly into a host numpy array.

    Mirror of _vec_copy_d2d: uses getCUDAHandle("r") to alias the device
    pointer, reads via cupy, and writes to the host array — bypassing
    PETSc's host-mirror update cycle.  Falls back to getArray() if the
    build doesn't expose getCUDAHandle.
    """
    import cupy as _cp
    try:
        n   = int(out_arr.shape[0])
        ptr = petsc_vec.getCUDAHandle("r")
        mem = _cp.cuda.UnownedMemory(ptr, n * 8, petsc_vec)
        d_view = _cp.ndarray((n,), dtype=_cp.float64,
                             memptr=_cp.cuda.MemoryPointer(mem, 0))
        out_arr[:] = _cp.asnumpy(d_view)
        petsc_vec.restoreCUDAHandle()
    except Exception:
        out_arr[:] = petsc_vec.getArray()


def _gpu_safe_pc(pc_type: str) -> str:
    """
    GPU-compatible PC dispatch.

    PETSc IS built with hypre-CUDA (--download-hypre --with-cuda), and the
    GPU BoomerAMG path runs.  Empirically it inherits the same parallel
    coarsening non-determinism as CPU hypre (random 5000-iter blow-ups), so
    we route hypre to CPU where it is at least more stable.  Cl / O2 / OH
    keep using bjacobi on aijcusparse and benefit from GPU.
    """
    if pc_type in ("hypre", "gamg", "lu", "ilu", "cholesky", "icc"):
        return None
    return pc_type or "bjacobi"


def init_ceed(domain, V, use_gpu: bool = True):
    """
    Initialise the libCEED context and extract mesh data.
    Call once after mesh loading, before the time loop.
    Returns (ceed, backend_name, mesh_data) or (None, "disabled", None).
    """
    global _ceed_ctx, _ceed_bname, _mesh_data
    if not _HAS_CEED:
        return None, "disabled", None
    try:
        _ceed_ctx, _ceed_bname = make_ceed(use_gpu)
        _mesh_data = extract_ceed_mesh_data(domain, V)
        return _ceed_ctx, _ceed_bname, _mesh_data
    except Exception as e:
        if domain.comm.rank == 0:
            print(f"[CEED] Warning: init failed ({e}). Falling back to CPU sparse.")
        return None, "disabled", None


class MatrixFreeOperator:
    """
    Wraps a libCEED operator in a PETSc MatShell so PETSc's Krylov solver
    can call op.apply() for every matrix-vector product — no sparse matrix
    is ever assembled.
    """

    def __init__(self, ceed_op, ndofs: int, comm):
        self.ceed_op = ceed_op
        self.ndofs   = ndofs
        self.comm    = comm
        A = PETSc.Mat().createPython([ndofs, ndofs], comm=comm)
        A.setPythonContext(self)
        A.setUp()
        self.mat = A

    def mult(self, A, x, y):
        import libceed
        # PETSc locks x read-only before calling mult(); must use readonly=True.
        x_arr = x.getArray(readonly=True)
        y_arr = y.getArray()
        u_ceed = self.ceed_op.ceed.Vector(len(x_arr))
        v_ceed = self.ceed_op.ceed.Vector(len(y_arr))
        u_ceed.set_array(x_arr.copy(), memtype=libceed.MEM_HOST, cmode=libceed.COPY_VALUES)
        v_ceed.set_value(0.0)
        self.ceed_op.apply(u_ceed, v_ceed)
        v_host = v_ceed.get_array_read(memtype=libceed.MEM_HOST)
        y_arr[:] = v_host[:]
        v_ceed.restore_array_read()

    def destroy(self, A):
        pass


def solve_transport(name: str, a_form, L_form, u: fem.Function,
                    bcs: list          = None,
                    ksp_type: str      = "gmres",
                    pc_type: str       = "bjacobi",
                    rtol: float        = 1e-8,
                    max_it: int        = 1000,
                    under_relax: float = None,
                    comm               = None,
                    extra_petsc_opts: dict = None,
                    ceed_op            = None,
                    use_gpu: bool      = False,
                    de_field: "fem.Function" = None,
                    alpha_field=None,
                    u_old=None,
                    source_field=None,
                    bc_value: float = 0.0) -> dict:
    """
    Solve a(u,v) = L(v).

    Parameters
    ----------
    extra_petsc_opts : dict, optional
        Additional PETSc options merged into petsc_options.  Use to pass
        e.g. {"pc_hypre_type": "boomeramg"} for the standard/GPU path.
        Ignored by the libCEED path (MatShell cannot use AMG).
    ceed_op : MatrixFreeOperator, optional
        libCEED shell matrix.  When provided the solver assembles only the
        RHS vector; all matrix-vector products call ceed_op.apply().
    use_gpu : bool
        When True and ceed_op is None, request PETSc CUDA sparse backend
        (aijcusparse + cuda vectors).  Silently falls back to CPU if CUDA
        is unavailable.
    """
    if bcs is None:
        bcs = []
    if comm is None:
        comm = MPI.COMM_WORLD

    t0 = time.time()

    if under_relax is not None and under_relax < 1.0:
        u_old = u.x.array.copy()

    if ceed_op is not None:
        converged, iterations = _solve_ceed(
            name, a_form, L_form, u, bcs, ksp_type, pc_type, rtol, max_it,
            ceed_op, comm
        )
    elif use_gpu and de_field is not None and pc_type in (None, "bjacobi", "jacobi"):
        # Fast GPU path: cupy CUDA kernel assembles M/dt + De·K (+ α·M) into an
        # aijcusparse Mat — ~14× faster than DOLFINx CPU assembly at 800k.
        try:
            converged, iterations = _solve_gpu_asm(
                name, de_field, L_form, u, bcs,
                ksp_type, pc_type or "bjacobi", rtol, max_it, comm,
                alpha_field=alpha_field,
                u_old=u_old, source_field=source_field, bc_value=bc_value,
            )
        except Exception as e:
            if comm.rank == 0:
                print(f"  [GPU-ASM] {name}: failed ({e}); falling back to sparse path.")
            _GPU_ASM_CACHE.pop(name, None)
            converged, iterations = _solve_standard(
                name, a_form, L_form, u, bcs, ksp_type, pc_type, rtol, max_it,
                extra_petsc_opts, use_gpu, comm
            )
    else:
        converged, iterations = _solve_standard(
            name, a_form, L_form, u, bcs, ksp_type, pc_type, rtol, max_it,
            extra_petsc_opts, use_gpu, comm
        )

    if under_relax is not None and under_relax < 1.0:
        u.x.array[:] = under_relax * u.x.array[:] + (1.0 - under_relax) * u_old

    u.x.scatter_forward()
    elapsed = time.time() - t0

    return {"converged": converged, "iterations": iterations, "time_s": elapsed}


def _solve_gpu_asm(name, de_field, L_form, u, bcs,
                   ksp_type, pc_type, rtol, max_it, comm,
                   alpha_field=None,
                   u_old=None, source_field=None, bc_value: float = 0.0):
    """
    Fast path: cupy CUDA kernel assembles M/dt + De·K straight into an
    aijcusparse Mat.  Cache the assembler + RHS Vec + solution Vec + KSP
    across timesteps (only the matrix entries and RHS are refreshed).

    Eliminates the ~1.3 s DOLFINx CPU assembly bottleneck.
    """
    domain = de_field.function_space.mesh
    V      = de_field.function_space
    inv_dt = float(getattr(de_field, "_solver_inv_dt", 1.0))

    # Unwrap alpha_field if it's a fem.Function; pass through numpy as-is.
    if alpha_field is not None and hasattr(alpha_field, "x"):
        alpha_arr = alpha_field.x.array
    else:
        alpha_arr = alpha_field

    # Extract BC DOFs (set on de_field._solver_bc_dofs by the caller).
    bc_dofs = getattr(de_field, "_solver_bc_dofs", None)

    cache = _GPU_ASM_CACHE.get(name)
    if cache is None:
        from gpu_assembler import GPUSparseAssembler
        # Use shared geometry if available — saves ~150 MB per field by
        # borrowing d_elem_nodes, d_detJ, d_Gflat, d_JinvT, d_indptr,
        # d_indices, _grad_buf, _wsum_buf from the primary assembler.
        asm = GPUSparseAssembler(domain, V, comm=comm,
                                 ref_asm=_SHARED_GEOM)
        # First assembly populates sparsity entries on the device.
        asm.assemble(de_field.x.array, inv_dt, alpha=alpha_arr, bc_dofs=bc_dofs)
        # Persistent cuda Vecs and KSP — created once, reused every step.
        b_gpu = asm.mat.createVecLeft()
        x_gpu = asm.mat.createVecRight()
        ksp   = PETSc.KSP().create(comm)
        ksp.setOperators(asm.mat)
        ksp.setType(ksp_type)
        ksp.getPC().setType(pc_type)
        ksp.setTolerances(rtol=rtol, max_it=max_it)
        ksp.setInitialGuessNonzero(True)  # warm-start: reuse x from previous step
        # GMRES restart=50: reduces restart overhead for fields taking 30-70 iters.
        # Default restart=30 can cause multiple restarts per solve; 50 avoids this
        # at the cost of 50 Krylov vectors (~54 MB for 135k DOFs) vs 30 (~32 MB).
        if ksp_type in ("gmres", "fgmres"):
            ksp.setGMRESRestart(50)
        ksp.setUp()
        _GPU_ASM_CACHE[name] = {"asm": asm, "b": b_gpu, "x": x_gpu, "ksp": ksp}
        cache = _GPU_ASM_CACHE[name]
    else:
        cache["asm"].assemble(de_field.x.array, inv_dt,
                              alpha=alpha_arr, bc_dofs=bc_dofs)

    asm   = cache["asm"]
    b_gpu = cache["b"]
    x_gpu = cache["x"]
    ksp   = cache["ksp"]

    # RHS: GPU path if u_old/source supplied; else fall back to DOLFINx.
    if u_old is not None:
        u_old_arr = u_old.x.array if hasattr(u_old, "x") else u_old
        src_arr   = (source_field.x.array
                     if hasattr(source_field, "x") else source_field)
        d_b = asm.assemble_rhs(u_old_arr, inv_dt, f=src_arr,
                               bc_dofs=bc_dofs, bc_value=bc_value)
        # Write cupy result directly into the PETSc CUDA Vec device buffer —
        # eliminates the D2H (asnumpy) + H2D (getArray write) round-trip.
        _vec_copy_d2d(b_gpu, d_b)
    else:
        L_compiled = fem.form(L_form)
        b_cpu = fem.petsc.assemble_vector(L_compiled)
        if bcs:
            fem.petsc.set_bc(b_cpu, bcs)
        b_gpu.getArray()[:] = b_cpu.getArray()
        b_cpu.destroy()
    # warm-start: H2D direct into device buffer — avoids spurious D2H that
    # x_gpu.getArray() would trigger before the write.
    import cupy as _cp
    _vec_copy_d2d(x_gpu, _cp.asarray(u.x.array))

    ksp.solve(b_gpu, x_gpu)
    _vec_read_d2h(x_gpu, u.x.array)

    return ksp.getConvergedReason() > 0, ksp.getIterationNumber()


def _solve_ceed(name, a_form, L_form, u, bcs, ksp_type, pc_type,
                rtol, max_it, ceed_op, comm):
    """Solve using a libCEED MatShell + PETSc Krylov solver."""
    # Assemble only the RHS (cheap: vector, not matrix)
    b = fem.petsc.assemble_vector(fem.form(L_form))
    fem.petsc.apply_lifting(b, [fem.form(a_form)], bcs=[[bc for bc in bcs]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, bcs)

    # MatShell has no matrix entries, so no preconditioner that needs a diagonal
    # or matrix factorization can be used.  "none" is always safe and correct.
    safe_pc = "none"

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(ceed_op.mat)
    ksp.setType(ksp_type)
    ksp.getPC().setType(safe_pc)
    ksp.setTolerances(rtol=rtol, max_it=max_it)
    ksp.setFromOptions()

    # DOLFINx 0.10: PETSc Vec lives at u.x.petsc_vec (not u.vector)
    ksp.solve(b, u.x.petsc_vec)
    u.x.scatter_forward()

    converged  = ksp.getConvergedReason() > 0
    iterations = ksp.getIterationNumber()

    ksp.destroy()
    b.destroy()
    return converged, iterations


def _solve_standard(name, a_form, L_form, u, bcs, ksp_type, pc_type,
                    rtol, max_it, extra_petsc_opts, use_gpu, comm):
    """
    Standard sparse path.
    use_gpu=True: assemble into aijcusparse Mat + cuda Vec so the entire
                  GMRES iteration (SpMV + dot + AXPY) runs on the GPU.
    use_gpu=False: classic DOLFINx LinearProblem on CPU.
    """
    base_opts = {
        "ksp_type":   ksp_type,
        "pc_type":    pc_type,
        "ksp_rtol":   str(rtol),
        "ksp_max_it": str(max_it),
    }
    if extra_petsc_opts:
        base_opts.update(extra_petsc_opts)

    # PCs that need CPU (hypre BoomerAMG, gamg in this PETSc build) skip GPU.
    if use_gpu and _gpu_safe_pc(base_opts.get("pc_type")) is not None:
        try:
            return _run_gpu(name, a_form, L_form, u, bcs, base_opts, comm)
        except Exception as e:
            if comm.rank == 0:
                print(f"  [GPU] {name}: CUDA path failed ({e}); falling back to CPU.")
            _GPU_CACHE.pop(name, None)

    return _run_cpu(name, a_form, L_form, u, bcs, base_opts)


def _run_cpu(name, a_form, L_form, u, bcs, opts):
    """Standard DOLFINx CPU sparse path."""
    problem = dolfinx.fem.petsc.LinearProblem(
        a_form, L_form, bcs=bcs,
        petsc_options=opts,
        petsc_options_prefix=f"{name}_"
    )
    uh = problem.solve()
    u.x.array[:] = uh.x.array[:]
    s = problem.solver
    return s.getConvergedReason() > 0, s.getIterationNumber()


def _run_gpu(name, a_form, L_form, u, bcs, opts, comm):
    """
    Manual assembly into aijcusparse Mat + cuda Vec, then GMRES on GPU.
    Uses _GPU_CACHE so Mat / Vec / KSP / AMG hierarchy persist across
    timesteps — first call pays the setup cost, subsequent calls only
    refresh matrix entries + RHS + run ksp.solve.

    When SOLVERS_PROFILE=1, prints a per-call timing breakdown.
    """
    import os
    profile = os.environ.get("SOLVERS_PROFILE", "0") == "1"

    def _tic():
        if profile:
            comm.Barrier()
        return time.perf_counter()

    t0 = _tic()
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)
    t_form = _tic()

    cache   = _GPU_CACHE.get(name)
    is_init = cache is None

    if is_init:
        # ── First call for this name: build everything ──────────────────
        A = dolfinx.fem.petsc.create_matrix(a_compiled)
        A.setType("aijcusparse")
        # Trigger the initial assembly so sizes / memory get allocated.
        A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(A, a_compiled, bcs=bcs)
        A.assemble()

        n_local, _ = A.getLocalSize()
        n_global, _ = A.getSize()
        b = PETSc.Vec().create(comm); b.setType("cuda")
        b.setSizes((n_local, n_global)); b.setUp()
        x = PETSc.Vec().create(comm); x.setType("cuda")
        x.setSizes((n_local, n_global)); x.setUp()

        ksp = PETSc.KSP().create(comm)
        ksp.setOperators(A)
        ksp.setType(opts.get("ksp_type", "gmres"))
        pc_type = _gpu_safe_pc(opts.get("pc_type", "bjacobi"))
        ksp.getPC().setType(pc_type)

        prefix  = f"{name}_"
        opts_db = PETSc.Options()
        if pc_type == "hypre":
            # GPU-tuned BoomerAMG (per hypre's own --with-cuda guidance):
            #   PMIS coarsening  — graph parallel, GPU-friendly (vs CLJP/Falgout)
            #   Chebyshev relax  — pure SpMV+axpy, no host triangular solves
            #   keep_transpose   — avoids rebuilding R^T per V-cycle
            #   max_levels 25    — generous; AMG truncates earlier if matrix small
            opts_db.setValue(f"{prefix}pc_hypre_type",
                             opts.get("pc_hypre_type", "boomeramg"))
            opts_db.setValue(f"{prefix}pc_hypre_boomeramg_coarsen_type",        "PMIS")
            opts_db.setValue(f"{prefix}pc_hypre_boomeramg_interp_type",         "ext+i")
            opts_db.setValue(f"{prefix}pc_hypre_boomeramg_relax_type_all",      "Chebyshev")
            opts_db.setValue(f"{prefix}pc_hypre_boomeramg_keep_transpose",      "true")
            opts_db.setValue(f"{prefix}pc_hypre_boomeramg_no_CF",               "true")
            opts_db.setValue(f"{prefix}pc_hypre_boomeramg_strong_threshold",    "0.5")
            opts_db.setValue(f"{prefix}pc_hypre_boomeramg_max_levels",          "25")
        if extra := opts.get("pc_hypre_type"):
            opts_db.setValue(f"{prefix}pc_hypre_type", extra)
        ksp.setOptionsPrefix(prefix)
        ksp.setFromOptions()

        ksp.setTolerances(rtol=float(opts.get("ksp_rtol", 1e-8)),
                          max_it=int(opts.get("ksp_max_it", 1000)))
        ksp.setInitialGuessNonzero(True)  # warm-start across timesteps
        ksp.setUp()

        _GPU_CACHE[name] = {"A": A, "b": b, "x": x, "ksp": ksp}
        cache = _GPU_CACHE[name]
    else:
        # ── Reuse cached Mat: just refresh the entries ──────────────────
        A = cache["A"]
        A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(A, a_compiled, bcs=bcs)
        A.assemble()
    t_mat = _tic()

    b_cache = cache["b"]; x_cache = cache["x"]; ksp = cache["ksp"]

    # ── Assemble RHS on CPU, copy onto the cached CUDA Vec ──────────────
    b_cpu = dolfinx.fem.petsc.assemble_vector(L_compiled)
    dolfinx.fem.petsc.apply_lifting(b_cpu, [a_compiled], bcs=[bcs])
    b_cpu.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b_cpu, bcs)
    t_rhs = _tic()

    b_cache.getArray()[:] = b_cpu.getArray()[:]
    # warm-start: seed with current solution (set(0) only on first call)
    if is_init:
        x_cache.set(0.0)
    else:
        x_cache.getArray()[:] = u.x.petsc_vec.getArray()[:]
    b_cpu.destroy()
    t_h2d = _tic()

    # ── Solve (re-setup happens lazily inside KSP only if operator changed) ──
    ksp.solve(b_cache, x_cache)
    t_solve = _tic()

    u.x.petsc_vec.getArray()[:] = x_cache.getArray()[:]
    t_d2h = _tic()

    converged  = ksp.getConvergedReason() > 0
    iterations = ksp.getIterationNumber()

    if profile and comm.rank == 0:
        tag = "INIT" if is_init else "warm"
        print(
            f"  [PROF {name:>4} {tag}] form={t_form-t0:.3f}s  "
            f"matAsm={t_mat-t_form:.3f}s  rhsAsm={t_rhs-t_mat:.3f}s  "
            f"h2d={t_h2d-t_rhs:.3f}s  "
            f"solve={t_solve-t_h2d:.3f}s ({iterations} it)  "
            f"d2h={t_d2h-t_solve:.3f}s  TOTAL={t_d2h-t0:.3f}s"
        )

    return converged, iterations
