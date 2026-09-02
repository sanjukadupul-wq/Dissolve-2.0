"""
output.py — Output routines for Dissolve 2.0 FEniCSx.
Saves mass loss data, mechanism data, and VTK/XDMF files.
Port of FreeFem++ io/write_results.idp and io/export_geometry.idp.
"""

import os
import numpy as np
from mpi4py import MPI
from dolfinx import fem, io
import ufl

from mesh_utils import compute_scaffold_volume, compute_surface_area


def save_step(domain, phi, Zn, F_film, Cl, OH, O2, params,
              t: float, count: int, Vinit: float,
              xdmf_file=None, vtk_file=None, v_interface=None,
              do_vtk: bool = None, gpu_asm=None, scaffold_vol: float = None):
    """
    Save output at the current time step (if scheduled).
    Matches FreeFem++ io/write_results.idp logic.

    Visualization output goes to two formats simultaneously:
      - XDMF (.xdmf + .h5)  : compact time-indexed, ideal for big runs
      - VTK  (.pvd + .vtu)  : universal, opens in ParaView/VisIt/MayaVi/etc.
    """
    comm = domain.comm
    save_each = params.save_each_steps

    if count % save_each != 0:
        return

    # scaffold_vol should be passed in (the caller's main loop already computed
    # it, honoring params.use_exact_volume). Recomputing it here independently
    # via the smoothed method was a real bug: result.txt (and hence the final
    # "Mass loss" summary, which reads ml_final from result.txt) silently used
    # a DIFFERENT volume measure than every console print / conservation check
    # in the same run, giving wrong-signed, wrong-magnitude final numbers.
    if scaffold_vol is None:
        scaffold_vol = compute_scaffold_volume(phi, domain, gpu_asm=gpu_asm)

    # --- 1. Save VTK/XDMF for visualization ---
    # do_vtk=True/False  → time-based decision made by the caller (dissolve.py)
    # do_vtk=None        → fall back to step-count cadence (vis_each_steps)
    if do_vtk is not None:
        do_visualize = params.emit_vtk and do_vtk
    else:
        vis_each = getattr(params, "vis_each_steps", 24)
        do_visualize = params.emit_vtk and (count % vis_each == 0)

    if do_visualize:
        # Tag each Function with a clean readable name (ParaView shows these).
        phi.name     = "phi_levelset"
        Zn.name      = "Zn_concentration"
        F_film.name  = "F_film"
        Cl.name      = "Cl_concentration"
        OH.name      = "OH_concentration"
        O2.name      = "O2_concentration"
        if v_interface is not None:
            v_interface.name = "v_interface"

        # XDMF time-series (one file, multiple time steps inside).
        if xdmf_file is not None:
            if comm.rank == 0:
                print(f"Saving XDMF file at t={t:.1f}...")
            xdmf_file.write_function(phi, t)
            xdmf_file.write_function(Zn, t)
            xdmf_file.write_function(F_film, t)
            xdmf_file.write_function(Cl, t)
            xdmf_file.write_function(OH, t)
            xdmf_file.write_function(O2, t)
            if v_interface is not None:
                xdmf_file.write_function(v_interface, t)

        # VTK collection (.pvd + .vtu per step) — PRIMARY output now.
        # DOLFINx writes all 6/7 fields as point data in ONE .vtu per
        # timestep, exactly the layout ParaView expects.  Each .vtu is
        # ~60 MB at 800k DOFs; use --vis_each_steps N to cut frequency.
        if vtk_file is not None:
            if comm.rank == 0:
                print(f"Saving VTK files at t={t:.1f}...")
            fields = [phi, Zn, F_film, Cl, OH, O2]
            if v_interface is not None:
                fields.append(v_interface)
            vtk_file.write_function(fields, t)

    # --- 2. Save mass loss data ---
    if comm.rank == 0:
        mass_loss_pct = 0.0
        if Vinit > 1e-12:
            mass_loss_pct = (Vinit - scaffold_vol) / Vinit * 100.0

        output_dir = os.path.dirname(params.results_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        mode = "w" if count == 0 else "a"
        with open(params.results_file, mode) as f:
            if count == 0:
                f.write("TimeHours\tMassLossPercent\n")
            f.write(f"{t:.6f}\t{mass_loss_pct:.6f}\n")

    # --- 3. Save mechanism data ---
    _save_mechanism_data(domain, phi, Zn, F_film, Cl, OH, O2,
                         params, t, count, Vinit, scaffold_vol)


def _save_phi0_surface(domain, phi, Zn, F_film, Cl, OH, O2,
                       v_interface, params, t, count):
    """Extract the phi=0 isosurface + sampled fields → write small .vtu."""
    try:
        from gpu_isosurface import (
            extract_phi0_surface, write_isosurface_vtu, append_to_pvd
        )
    except Exception:
        return
    comm = domain.comm
    if comm.size > 1 and comm.rank != 0:
        return

    V = phi.function_space

    field_arrays = {
        "Zn":     Zn.x.array,
        "F_film": F_film.x.array,
        "Cl":     Cl.x.array,
        "OH":     OH.x.array,
        "O2":     O2.x.array,
    }
    if v_interface is not None:
        field_arrays["v_interface"] = v_interface.x.array

    points, tris, sampled = extract_phi0_surface(
        domain, V, phi.x.array, field_arrays=field_arrays,
    )

    base = params.vtk_prefix + "_surface"
    vtu_dir = base
    os.makedirs(vtu_dir, exist_ok=True)
    vtu_file_rel = f"{os.path.basename(vtu_dir)}/step{count:06d}.vtu"
    vtu_file_abs = os.path.join(os.path.dirname(base) or ".", vtu_file_rel)

    write_isosurface_vtu(vtu_file_abs, points, tris, sampled)
    append_to_pvd(base + ".pvd", vtu_file_rel, t, count)

    if comm.rank == 0:
        size_kb = os.path.getsize(vtu_file_abs) / 1024
        print(f"  scaffold surface: {len(points):,} verts, "
              f"{len(tris):,} tris, {size_kb:.1f} KB")


def _save_mechanism_data(domain, phi, Zn, F_film, Cl, OH, O2,
                         params, t, count, Vinit, scaffold_vol):
    """Compute and save mechanism data (concentrations, extremes, surface area)."""
    comm = domain.comm
    dx = ufl.Measure("dx", domain=domain)
    medium = ufl.conditional(ufl.le(phi, 0), 1.0, 0.0)

    # Volume integrals
    def integrate(field):
        form = fem.form(field * dx)
        local = fem.assemble_scalar(form)
        return comm.allreduce(local, op=MPI.SUM)

    total_Zn = integrate(Zn)
    total_Cl = integrate(Cl)
    total_OH = integrate(OH)
    total_O2 = integrate(O2)
    total_Film = integrate(F_film)
    liquid_vol = integrate(medium)

    # Surface area
    surface_area = compute_surface_area(phi, domain)

    # Extremes
    max_Zn_local = Zn.x.array.max() if len(Zn.x.array) > 0 else -1e20
    max_OH_local = OH.x.array.max() if len(OH.x.array) > 0 else -1e20
    min_O2_local = O2.x.array.min() if len(O2.x.array) > 0 else 1e20

    max_Zn = comm.allreduce(max_Zn_local, op=MPI.MAX)
    max_OH = comm.allreduce(max_OH_local, op=MPI.MAX)
    min_O2 = comm.allreduce(min_O2_local, op=MPI.MIN)

    if comm.rank == 0:
        vol = max(liquid_vol, 1e-9)
        avg_Zn = total_Zn / vol
        avg_Cl = total_Cl / vol
        avg_OH = total_OH / vol
        avg_O2 = total_O2 / vol

        output_dir = os.path.dirname(params.results_file)
        if not output_dir:
            output_dir = "output"
        mech_file = os.path.join(output_dir, "mechanism_data.txt")

        mode = "w" if count == 0 else "a"
        with open(mech_file, mode) as f:
            if count == 0:
                f.write("TimeHours\tLiquidVol\tSurfaceArea\tAvgZn\tAvgCl\t"
                        "AvgOH\tAvgO2\tTotalFilm\tMaxZn\tMaxOH\tMinO2\n")
            f.write(f"{t:.4f}\t{liquid_vol:.6e}\t{surface_area:.6e}\t"
                    f"{avg_Zn:.6e}\t{avg_Cl:.6e}\t{avg_OH:.6e}\t{avg_O2:.6e}\t"
                    f"{total_Film:.6e}\t{max_Zn:.6e}\t{max_OH:.6e}\t{min_O2:.6e}\n")


def print_step_info(t, count, Vinit, scaffold_vol, params, comm):
    """Print progress information to the console (rank 0 only)."""
    if comm.rank != 0:
        return

    rhoZn = params.rhoZn
    mass_lost = max(0.0, (Vinit - scaffold_vol) * rhoZn)
    mass_O2 = mass_lost * 0.2447  # stoichiometry ratio

    pct = 0.0
    if Vinit > 1e-12:
        pct = (Vinit - scaffold_vol) / Vinit * 100.0

    divider = "=" * 61
    print(divider)
    print(f"Time: {t:.2f}    Step: {count}    Consumed O2 (g): {mass_O2:.6e}")
    print(f"Initial size: {Vinit:.6f}    Current size: {scaffold_vol:.6f}"
          f"    % Change: {pct:.4f}")
    print(divider)
