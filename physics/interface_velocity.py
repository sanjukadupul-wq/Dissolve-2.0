"""
interface_velocity.py -- Interface recession velocity for Dissolve 2.0.

Velocity law: resistance-in-series driven by the LOCAL dissolved O2 (paper Eq. 15)
  J     = C_O2|probe / (1/k_ORR + R_diff)   [g/mm2/h]
  v_n   = -_V_ORR * J                       [mm/h, negative = recession]
  R_diff = film_length / DeO2_probed          [h/mm]

O2 coupling: C_O2|probe is the dissolved-O2 concentration sampled at the
medium-side neighbor_map probe (x_i + h*n_i, one element into the medium).
This is the LOCAL O2 supply feeding the interface, taken from the solved O2
PDE -- NOT a global constant.  Interfaces that are O2-starved (buried deep in
the scaffold, or shadowed from the open-air replenishment face) therefore
corrode slower, giving genuine spatial ORR-transport feedback.

Two nested transport resistances, deliberately kept distinct to avoid
double-counting:
  * The O2 PDE resolves the MACRO-scale O2 field (bulk depletion, starvation
    of buried pores).  Its result enters here as C_O2|probe.
  * R_diff = film_length/DeO2_probed is the SUB-GRID film resistance the coarse
    mesh (h > film thickness) cannot resolve.  DeO2_probed is film-reduced by
    compute_effective_diffusion (from local F), so R_diff still rises as the
    film saturates even when the O2 PDE alone would miss the thin-film gradient.

film_length (mm) is the effective O2 diffusion path through the corrosion
film at the scaffold surface.  Physical bounds:
  sqrt(D_Zn_eff / k_f) = 9.6 um  (reaction-penetration in saturated film)
  sqrt(D_Zn    / k_f)  = 147 um  (reaction-penetration in open medium)

As film saturates DeO2 -> D_O2 * eps*delta/tau and R_diff rises; combined with
the falling C_O2|probe from the O2 PDE this reproduces the experimental
biphasic deceleration.  Da = k_ORR * R_diff (dimensionless) reports the
sub-grid film contribution to that deceleration.

GPU acceleration: when a `gpu_asm` argument is provided, O2 and DeO2 are
read in one cupy kernel call at medium-side probe points.
"""

import numpy as np
from mpi4py import MPI
from dolfinx import fem

try:
    import cupy as cp
except ImportError:
    cp = None

# -- ORR velocity constants (mm / g / h unit system) --------------------------
_M_ZN        = 65.38            # g/mol  Zn molar mass
_M_O2        = 32.0             # g/mol  O2 molar mass
_RHO_ZN      = 7.14e-3          # g/mm3  Zn solid density
_ZN_SOLID    = _RHO_ZN / _M_ZN  # mol/mm3  molar density of solid Zn = 1.092e-4
# v_n [mm/h] = -_V_ORR * k_orr * C_bulk [g/mm3]
# FreeFem's actual vO2React (physics/interface_velocity.idp) is the pure, undivided
# -2.0*kORR*O2/Znsolid -- O2 (mass conc., g/mm3) divided DIRECTLY by Znsolid
# (mol/mm3), with NO O2-molar-mass conversion. The previous _V_ORR =
# 2/(M_O2*Zn_solid) added an extra /32 (M_O2, g/mol) that has no counterpart
# in the reference model -- it made v_orr 32x weaker than FreeFem's for the
# same k_orr/O2, and is the reason calibration had to inflate k_orr ~1000x
# (to ~11-20) above FreeFem's actual validated value of 0.0144 mm/h to get
# any appreciable recession. Matches FreeFem exactly now.
_V_ORR       = 2.0 / _ZN_SOLID   # = 18320.0


def compute_norm_grad_phi(phi: fem.Function,
                          normgradphi: fem.Function,
                          gpu_asm=None):
    """Nodal |grad phi|, the coarea factor for the level-set surface integral.

    CPU and GPU both use the same volume-weighted-average nodal gradient
    (mesh_utils.nodal_gradient_cpu mirrors gpu_asm.nodal_gradient exactly) so
    the two paths produce numerically consistent |grad phi| -- previously the
    CPU fallback used a naive fem.Expression+interpolate that silently
    overwrites shared-DOF values instead of averaging them, which measurably
    degraded accuracy relative to the GPU path (see mesh_utils.py).
    """
    if gpu_asm is not None and cp is not None:
        g = gpu_asm.nodal_gradient(phi.x.array)
        nm = cp.sqrt(g[:, 0]**2 + g[:, 1]**2 + g[:, 2]**2 + 1e-12)
        normgradphi.x.array[:] = cp.asnumpy(nm)
    else:
        from mesh_utils import nodal_gradient_cpu
        V = phi.function_space
        domain = V.mesh
        g = nodal_gradient_cpu(phi.x.array, domain, V)
        normgradphi.x.array[:] = np.sqrt(g[:, 0]**2 + g[:, 1]**2 + g[:, 2]**2 + 1e-12)
    normgradphi.x.scatter_forward()


def compute_effective_diffusion(phi: fem.Function,
                                F_film: fem.Function,
                                params,
                                DeZn: fem.Function,
                                DeCl: fem.Function,
                                DeOH: fem.Function,
                                DeO2: fem.Function):
    """
    Effective diffusivity through the evolving porous film -- paper Eq. (12):
        D_i_eff = D_i0 * [ (1 - F/Fmax) + (F/Fmax) * (eps*delta/tau^2) ]
    Floor at full film: eps*delta/tau^2 = 0.25*1.0/2.0^2 = 0.0625 (Table 1:
    eps=0.25, delta=1.0, tau=2.0).  The previous code used eps*delta/tau (no
    tau^2) with a calibrated delta=0.0339, giving floor=4.24e-3 -- ~15x too
    small, which over-collapsed D_eff once the film saturated and froze the
    gradient-driven interface velocity terms.
    Cl, OH, O2: only active in medium (phi <= 0).
    """
    Fmax  = params.Fmax
    tau   = params.film_tortuosity
    eps   = params.film_porosity
    delta = params.film_constrictivity
    Fmax_val = max(Fmax, 1e-20)
    # FreeFem's own physics/interface_velocity.idp uses DeZn = DZn*((1-F/Fmax) + (F/Fmax)*del*eps/tau)
    # -- linear in tau, giving floor=0.125 at del=1.0. But per the published paper's
    # Eq. (12), the literature formula is tau^2 (floor=0.0625, half the FreeFem-code
    # value) -- using that here per explicit instruction.
    film_floor = eps * delta / (tau * tau)

    if cp is not None:
        xp = cp
        F_d   = cp.asarray(F_film.x.array)
        phi_d = cp.asarray(phi.x.array)
    else:
        xp = np
        F_d   = F_film.x.array
        phi_d = phi.x.array

    saturation  = xp.clip(F_d / Fmax_val, 0.0, 1.0)
    film_factor = (1.0 - saturation) + saturation * film_floor
    medium_mask = (phi_d <= 0.0).astype(xp.float64)

    results = {
        DeZn: params.diff_zn * film_factor,
        DeCl: params.diff_cl * film_factor * medium_mask,
        DeOH: params.diff_oh * film_factor * medium_mask,
        DeO2: params.diff_o2 * film_factor * medium_mask,
    }

    for fn, arr_d in results.items():
        if not bool(xp.all(xp.isfinite(arr_d))):
            arr_cpu = cp.asnumpy(arr_d) if cp is not None else arr_d
            raise RuntimeError(
                f"[bounds] {fn.name} contains NaN/Inf after effective diffusion update. "
                f"Check F_film saturation (max={float(F_d.max()):.3e}, Fmax={Fmax_val:.3e})."
            )
        if bool(arr_d.min() < 0.0):
            n_neg = int((arr_d < 0.0).sum())
            raise RuntimeError(
                f"[bounds] {fn.name} has {n_neg} negative values "
                f"(min={float(arr_d.min()):.3e}). "
                "Film saturation or mesh interpolation produced unphysical diffusion."
            )

    for fn, arr_d in results.items():
        fn.x.array[:] = cp.asnumpy(arr_d) if cp is not None else arr_d
        fn.x.scatter_forward()


def build_medium_neighbor_map(phi: fem.Function, h: float, gpu_asm=None,
                              probe_dist: float = None) -> np.ndarray:
    """
    For each interface DOF (|phi| < 3h) find the nearest medium DOF (phi <= 0)
    at probe point x_i + probe_dist*n_i (n_i = inward normal, -grad(phi)/|grad(phi)|).
    Used to sample Zn/O2/DeZn/DeO2 on the medium side of the interface,
    bypassing the Dirichlet-contaminated interface-node value.

    probe_dist defaults to h (the original single-probe distance used by the
    resistance-in-series O2 model).  Pass 2*h for the second, farther probe
    point needed by the Stage-1 Stefan-condition two-point finite-difference
    gradient (compute_interface_velocity_stefan) -- same direction field,
    just projected twice as far.

    Works on both backends: gpu_asm.nodal_gradient (cupy) when gpu_asm is
    provided, else mesh_utils.nodal_gradient_cpu (numpy) -- the same
    volume-weighted-average algorithm either way, so CPU and GPU runs build
    the same neighbor map and therefore probe the same physical location.

    Returns int32 array length n_dofs.  Non-interface entries hold -1.
    Rebuilt after every redistancing step.
    """
    from scipy.spatial import cKDTree as _cKDTree

    if probe_dist is None:
        probe_dist = h

    V = phi.function_space
    coords  = V.tabulate_dof_coordinates()
    phi_arr = phi.x.array
    n_dofs  = len(phi_arr)

    neighbor_map = np.full(n_dofs, -1, dtype=np.int32)

    iband  = np.abs(phi_arr) < 3.0 * h
    medium = phi_arr <= 0.0
    ib_idx  = np.where(iband)[0]
    med_idx = np.where(medium)[0]

    if len(ib_idx) == 0 or len(med_idx) == 0:
        return neighbor_map

    if gpu_asm is not None and cp is not None:
        g_phi = gpu_asm.nodal_gradient(phi_arr)
        gx = cp.asnumpy(g_phi[ib_idx, 0])
        gy = cp.asnumpy(g_phi[ib_idx, 1])
        gz = cp.asnumpy(g_phi[ib_idx, 2])
    else:
        from mesh_utils import nodal_gradient_cpu
        g_phi = nodal_gradient_cpu(phi_arr, V.mesh, V)
        gx, gy, gz = g_phi[ib_idx, 0], g_phi[ib_idx, 1], g_phi[ib_idx, 2]

    mag = np.sqrt(gx*gx + gy*gy + gz*gz + 1e-12)
    nx = -gx / mag
    ny = -gy / mag
    nz = -gz / mag

    probe_pts = coords[ib_idx] + probe_dist * np.column_stack((nx, ny, nz))

    tree = _cKDTree(coords[med_idx])
    _, nn_local = tree.query(probe_pts, workers=-1)
    neighbor_map[ib_idx] = med_idx[nn_local].astype(np.int32)

    return neighbor_map


def compute_interface_velocity(phi: fem.Function,
                               Zn: fem.Function,
                               O2: fem.Function,
                               DeZn: fem.Function,
                               DeO2: fem.Function,
                               v_field: fem.Function,
                               params,
                               h: float,
                               t: float,
                               gpu_asm=None,
                               F_film: fem.Function = None,
                               dt: float = 1.0,
                               neighbor_map=None):
    """
    Interface recession velocity -- resistance-in-series on LOCAL O2 (paper Eq. 15):

        R_diff = film_length / DeO2_probed              [h/mm]
        v_n    = -_V_ORR * C_O2|probe / (1/k_ORR + R_diff)   [mm/h]

    C_O2|probe is the solved dissolved-O2 field sampled one element into the
    medium (neighbor_map) -- the local O2 supply, so O2-starved interfaces
    corrode slower.  film_length is the sub-grid O2-blocking film thickness.
    DeO2_probed is film-reduced by compute_effective_diffusion; as the film
    saturates DeO2 falls and R_diff rises, which -- together with the falling
    C_O2|probe -- reproduces the biphasic deceleration at any mesh resolution.

    c_o2_bulk is retained only as the fallback O2 value where the medium-side
    probe is unavailable (no neighbor_map / CPU path at a starved node).
    """
    k_orr         = getattr(params, 'k_orr',         0.015)   # mm/h (single ORR rate constant)
    c_o2_bulk     = getattr(params, 'o2_initial',    5.44e-9)  # g/mm3 (fallback only)
    film_length    = getattr(params, 'film_length',    0.033)    # mm
    velocity_mode = getattr(params, 'velocity_mode', 'surrogate')
    k_orr_react   = getattr(params, 'k_orr_react',   0.90)     # mm/h (physical mode)

    interface_band = np.abs(phi.x.array) < 3.0 * h
    v_field.x.array[:] = 0.0

    if np.sum(interface_band) == 0:
        v_field.x.scatter_forward()
        return

    if gpu_asm is not None and cp is not None:
        _compute_orr_velocity_gpu(
            gpu_asm, phi, O2, DeO2, v_field,
            h, k_orr, c_o2_bulk, film_length,
            interface_band, phi.function_space.mesh.comm,
            dt=dt, neighbor_map=neighbor_map,
            velocity_mode=velocity_mode, k_orr_react=k_orr_react,
        )
    else:
        _compute_orr_velocity_cpu(
            phi, O2, DeO2, v_field,
            h, k_orr, c_o2_bulk, film_length,
            interface_band, dt=dt,
            velocity_mode=velocity_mode, k_orr_react=k_orr_react,
            neighbor_map=neighbor_map,
        )

    v_field.x.scatter_forward()


def _compute_orr_velocity_gpu(asm, phi, O2, DeO2, v_field,
                               h, k_orr, c_o2_bulk, film_length,
                               interface_band, comm,
                               dt: float = 1.0, neighbor_map=None,
                               velocity_mode="surrogate", k_orr_react=0.90):
    """
    GPU interface velocity on LOCAL O2 (probed at the medium-side neighbor point).

    velocity_mode "surrogate" — unified resistance-in-series:
        v_n = -_V_ORR * C_O2|probe / (1/k_orr + film_length/DeO2)
    velocity_mode "physical"  — paper Eq.14+16 mixed control:
        v_ORR = -_V_ORR * k_orr_react * C_O2|probe          (reaction-limited)
        v_O2  = -_V_ORR * C_O2|probe * DeO2 / film_length   (diffusion-limited)
        v_n   = max(v_ORR, v_O2)   (least-negative = rate-limiting mechanism)
    """
    O2_d   = cp.asarray(O2.x.array)
    De_d   = cp.asarray(DeO2.x.array)
    band_d = cp.asarray(interface_band)

    if neighbor_map is not None:
        nn_d  = cp.asarray(neighbor_map)
        safe  = cp.maximum(nn_d, 0)
        valid = band_d & (nn_d >= 0)
        O2_probed = cp.where(valid, O2_d[safe], O2_d)
        De_probed = cp.where(valid, De_d[safe], De_d)
    else:
        O2_probed = O2_d
        De_probed = De_d

    O2_safe = cp.maximum(O2_probed, 0.0)                        # g/mm3, local supply
    De_safe = cp.maximum(De_probed, 1e-30)
    R_diff  = film_length / De_safe                             # h/mm
    if velocity_mode == "physical":
        v_react = -_V_ORR * k_orr_react * O2_safe               # reaction limit
        v_diff  = -_V_ORR * O2_safe * De_safe / film_length     # diffusion limit
        v_n     = cp.maximum(v_react, v_diff)                   # rate-limiting (least neg)
    else:
        v_n     = -_V_ORR * O2_safe / (1.0 / k_orr + R_diff)
    v_out   = cp.where(band_d, v_n, 0.0)
    v_field.x.array[:] = cp.asnumpy(v_out)

    if comm.rank == 0 and int(band_d.sum()) > 0:
        v_b  = v_n[band_d]
        De_b = De_probed[band_d]
        Da_b = k_orr * R_diff[band_d]
        print(f"  v_ORR [{float(v_b.min()):.3e}, {float(v_b.max()):.3e}] mm/h  "
              f"DeO2_iface={float(De_b.mean()):.3e} mm2/h  "
              f"Da_mean={float(Da_b.mean()):.2f}")


def _compute_orr_velocity_cpu(phi, O2, DeO2, v_field,
                               h, k_orr, c_o2_bulk, film_length,
                               interface_band, dt: float = 1.0,
                               velocity_mode="surrogate", k_orr_react=0.90,
                               neighbor_map=None):
    """
    CPU interface velocity on LOCAL O2, probed at the medium-side neighbor
    point when neighbor_map is provided (mirrors _compute_orr_velocity_gpu
    exactly, in numpy instead of cupy, so CPU and GPU sample the same
    physical location instead of CPU always using the raw, Dirichlet-
    contaminated on-node value).
    See _compute_orr_velocity_gpu for the "surrogate" / "physical" mode formulas.
    """
    comm  = phi.function_space.mesh.comm
    O2_d  = O2.x.array
    De_d  = DeO2.x.array

    if neighbor_map is not None:
        safe  = np.maximum(neighbor_map, 0)
        valid = interface_band & (neighbor_map >= 0)
        O2_probed = np.where(valid, O2_d[safe], O2_d)
        De_probed = np.where(valid, De_d[safe], De_d)
    else:
        O2_probed = O2_d
        De_probed = De_d

    O2_arr  = np.maximum(O2_probed, 0.0)                        # g/mm3, local supply
    De_arr  = np.maximum(De_probed, 1e-30)
    R_diff  = film_length / De_arr
    if velocity_mode == "physical":
        v_react = -_V_ORR * k_orr_react * O2_arr
        v_diff  = -_V_ORR * O2_arr * De_arr / film_length
        v_n     = np.maximum(v_react, v_diff)
    else:
        v_n     = -_V_ORR * O2_arr / (1.0 / k_orr + R_diff)
    v_field.x.array[:] = np.where(interface_band, v_n, 0.0)

    if comm.rank == 0:
        v_b  = v_n[interface_band]
        De_b = De_arr[interface_band]
        if len(v_b) > 0:
            Da_b = k_orr * film_length / De_b
            print(f"  v_ORR [{v_b.min():.3e}, {v_b.max():.3e}] mm/h  "
                  f"DeO2_iface={De_b.mean():.3e} mm2/h  "
                  f"Da_mean={Da_b.mean():.2f}")


def compute_interface_velocity_stefan(phi: fem.Function,
                                      Zn: fem.Function,
                                      O2: fem.Function,
                                      DeZn: fem.Function,
                                      DeO2: fem.Function,
                                      v_field: fem.Function,
                                      params,
                                      h_band: float,
                                      t: float,
                                      neighbor_map_h: np.ndarray,
                                      neighbor_map_2h: np.ndarray,
                                      h_probe: float = None) -> None:
    """
    Dual Zn/O2 Stefan-condition interface velocity -- exact port of the
    Stage-1 FreeFem reference (physics/interface_velocity.idp), replacing the single-
    species resistance-in-series formula above.

    For each interface DOF, two probe points are sampled along the inward
    normal (n = -grad(phi)/|grad(phi)|): probe1 at x+h_probe*n, probe2 at
    x+2*h_probe*n (neighbor_map_h / neighbor_map_2h -- nearest-medium-DOF
    approximations of FreeFem's exact off-grid point evaluation Zn(x+dirx,...)
    etc, built with build_medium_neighbor_map(..., probe_dist=h_probe/2*h_probe)).

    h_band sets the interface-band mask (|phi| < 3*h_band) -- an addition not
    present in Stage-1 (whose implicit variational level-set update naturally
    down-weights v away from the interface via the normgradphi factor; this
    port's explicit algebraic update needs an explicit mask instead), so it
    uses the actual mesh spacing (h_avg), not the Stage-1 probe constant.

    h_probe is the actual Stefan finite-difference length scale and MUST
    match Stage-1's literal constant (params.stefan_probe_h = 0.002 mm --
    state/initial_state.idp computes h from the mesh then unconditionally overwrites
    it with 0.002) for the two formulas below to reproduce Stage-1 exactly.
    Defaults to h_band only for backward compatibility; callers should always
    pass params.stefan_probe_h explicitly.

        vZn      = -(Zn|p1 - Zn|p2) * DeZn|p1 * vDenom / h_probe   (<=0)
                   vDenom = 1/(Znsolid - Znbc)
        vO2React = -2*kORR * O2|p1 / Znsolid                       (reaction-limited)
        vO2Diff  = -2*(O2|p1 - O2|p2) * DeO2|p1 / (Znsolid*h_probe) (diffusion-limited, <=0)
        vO2      = max(vO2React, vO2Diff)     -- slower of the two O2 mechanisms wins
        v_n      = max(vZn, vO2)              -- slower of Zn-limited / O2-limited wins

    At t=0 (no established gradient yet) FreeFem uses the pure reaction-rate
    estimate alone: v_n = -2*kORR*O2|p1/Znsolid (matches physics/interface_velocity.idp's
    "Reaction Limited Initialization" branch).
    """
    if h_probe is None:
        h_probe = h_band

    k_orr   = getattr(params, 'k_orr', 0.015)   # mm/h, same rate constant as the O2 PDE sink
    Znsolid = params.Znsolid
    Znbc    = params.Znbc
    vDenom  = 1.0 / (Znsolid - Znbc)

    interface_band = np.abs(phi.x.array) < 3.0 * h_band
    v_field.x.array[:] = 0.0

    if np.sum(interface_band) == 0:
        v_field.x.scatter_forward()
        return

    Zn_arr, O2_arr = Zn.x.array, O2.x.array
    DeZn_arr, DeO2_arr = DeZn.x.array, DeO2.x.array

    def _probe(arr, nmap):
        safe  = np.maximum(nmap, 0)
        valid = interface_band & (nmap >= 0)
        return np.where(valid, arr[safe], arr)

    # Clamp probed concentrations to >= 0.  The FE/PETSc O2 (and Zn) solves can
    # undershoot slightly negative near the interface (a normal discretization
    # artifact).  Stage-1 FreeFem's solver keeps these non-negative so it never
    # hit this, but here an unclamped negative O2_p1 flips vO2React POSITIVE
    # (-2*k*O2/Znsolid with O2<0), injecting spurious interface GROWTH that
    # cancels the legitimate recession elsewhere and stalls the whole surface
    # (observed: mass loss plateaus at ~0.049% from t~116h instead of tracking
    # Stage-1's continued decay to 0.30%).  Matches the max(.,0) clamp the
    # previous resistance-in-series model already applied to O2.
    O2_p1 = np.maximum(_probe(O2_arr, neighbor_map_h), 0.0)

    if abs(t) < 1e-9:
        # Reaction-limited init: no established concentration gradient yet.
        v_n = -2.0 * k_orr * O2_p1 / Znsolid
    else:
        Zn_p1   = np.maximum(_probe(Zn_arr,   neighbor_map_h), 0.0)
        Zn_p2   = np.maximum(_probe(Zn_arr,   neighbor_map_2h), 0.0)
        DeZn_p1 = _probe(DeZn_arr, neighbor_map_h)
        O2_p2   = np.maximum(_probe(O2_arr,   neighbor_map_2h), 0.0)
        DeO2_p1 = _probe(DeO2_arr, neighbor_map_h)

        vZn = -(Zn_p1 - Zn_p2) * DeZn_p1 * vDenom / h_probe
        vZn = np.minimum(vZn, 0.0)

        vO2React = -2.0 * k_orr * O2_p1 / Znsolid
        vO2Diff  = -2.0 * (O2_p1 - O2_p2) * DeO2_p1 / (Znsolid * h_probe)
        vO2Diff  = np.minimum(vO2Diff, 0.0)
        vO2      = np.maximum(vO2React, vO2Diff)

        v_n = np.maximum(vZn, vO2)

    # Final safety: this is a pure-dissolution model -- the interface only
    # recedes (v_n <= 0), never grows.  Guards against any residual positive
    # excursion from the finite-difference probes.
    v_n = np.minimum(v_n, 0.0)

    v_field.x.array[:] = np.where(interface_band, v_n, 0.0)
    v_field.x.scatter_forward()

    comm = phi.function_space.mesh.comm
    if comm.rank == 0:
        v_b = v_n[interface_band]
        if len(v_b) > 0:
            print(f"  v_Stefan [{v_b.min():.3e}, {v_b.max():.3e}] mm/h  "
                  f"(dual Zn/O2 Stefan, Stage-1 mechanism)")


def _nodal_grad(field_arr, phi, gpu_asm):
    """Volume-weighted nodal gradient (n_dofs,3) of a P1 field, GPU or CPU."""
    if gpu_asm is not None and cp is not None:
        g = gpu_asm.nodal_gradient(field_arr)
        return cp.asnumpy(g) if hasattr(g, "get") or cp is not None else g
    from mesh_utils import nodal_gradient_cpu
    V = phi.function_space
    return nodal_gradient_cpu(field_arr, V.mesh, V)


def _project_cell_to_dofs(cell_values, cell_weights, cellnodes, n_dofs):
    """Volume-weighted nodal average of a per-cell scalar (mass-consistent,
    not the lossy max-by-cell scatter gpu_assembler.assemble_rhs falls back
    to for per-cell input -- this builds a genuine per-DOF array instead so
    that path is never hit)."""
    num = np.zeros(n_dofs)
    den = np.zeros(n_dofs)
    contrib = cell_values * cell_weights
    np.add.at(num, cellnodes.ravel(), np.repeat(contrib, 4))
    np.add.at(den, cellnodes.ravel(), np.repeat(cell_weights, 4))
    out = np.zeros(n_dofs)
    mask = den > 1e-30
    out[mask] = num[mask] / den[mask]
    return out


def compute_exact_orr_quantities(phi: fem.Function, O2: fem.Function, k_orr: float):
    """EXACT (marching-tetrahedra) ORR quantities, replacing the regularised
    delta_eps(phi)*|grad phi| band with the true phi=0 surface.

    Returns dict with:
      dof_sink_density : (n_dofs,) g/(mm3.h) -- feed as source_field to the O2
                          (and 4x for OH) solve_transport call in exact_mode.
      dof_O2_interface  : (n_dofs,) g/mm3 -- exact patch-averaged O2 AT phi=0,
                          projected to nodes; feed to compute_interface_velocity_paper
                          as exact_O2_override so the SAME quantity that was
                          consumed also drives recession (closes the
                          conservation gap by construction, not by tuning).
      total_area, total_sink_rate : scalars, for diagnostics.

    Both quantities come from ONE cut_patch_geometry call so the sink and the
    velocity are guaranteed to agree on which cells are "the interface" and
    what O2 they saw there.
    """
    from mechanics_fe import cut_patch_geometry

    V = phi.function_space
    domain = V.mesh
    cellnodes = V.dofmap.list
    coords = domain.geometry.x
    n_dofs = phi.x.array.shape[0]

    phi_v = phi.x.array[cellnodes]
    o2_v  = O2.x.array[cellnodes]
    cell_coords = coords[cellnodes]

    area, avg_o2 = cut_patch_geometry(phi_v, cell_coords, field_v=o2_v)
    cut = area > 1e-30

    total_area = float(area.sum())
    total_sink_rate = float((k_orr * avg_o2[cut] * area[cut]).sum())

    # tet volumes, for the sink's cell-to-DOF spread (surface source treated
    # as uniform over the CUTaCELL's own volume -- localises the equivalent
    # volumetric density to exactly the cells the interface actually crosses,
    # unlike the epsilon-band version which spreads over neighbouring cells too)
    v0, v1, v2, v3 = cell_coords[:, 0], cell_coords[:, 1], cell_coords[:, 2], cell_coords[:, 3]
    cell_vol = np.abs(np.einsum('ij,ij->i', v1 - v0, np.cross(v2 - v0, v3 - v0))) / 6.0

    sink_rate_cell = np.zeros_like(area)
    sink_rate_cell[cut] = k_orr * avg_o2[cut] * area[cut]           # g/h, per cell
    density_cell = np.zeros_like(area)
    safe_vol = cell_vol > 1e-30
    density_cell[cut & safe_vol] = sink_rate_cell[cut & safe_vol] / cell_vol[cut & safe_vol]

    dof_sink_density = _project_cell_to_dofs(density_cell, cell_vol, cellnodes, n_dofs)
    # O2-at-interface: area-weighted nodal average (weight by patch area, not
    # cell volume -- this is a surface quantity, not a volumetric one).
    dof_O2_interface = _project_cell_to_dofs(avg_o2, area, cellnodes, n_dofs)

    cut_mask_dofs = np.zeros(n_dofs, dtype=bool)
    cut_mask_dofs[cellnodes[cut].ravel()] = True

    return {
        "dof_sink_density": dof_sink_density,
        "dof_O2_interface": dof_O2_interface,
        "cut_mask_dofs": cut_mask_dofs,
        "total_area": total_area,
        "total_sink_rate": total_sink_rate,
    }


def _eval_at_points(fn: fem.Function, pts: np.ndarray) -> np.ndarray:
    """Evaluate a P1 Function at arbitrary points (n,3); NaN where not found.

    Mirrors FreeFem's `O2(x+dirx, y+diry, z+dirz)` P1 interpolation at an
    off-node location.  Points outside this rank's cells (or outside the mesh)
    come back NaN so the caller can fall back to the nodal value.
    """
    from dolfinx import geometry as _geom
    mesh = fn.function_space.mesh
    pts = np.ascontiguousarray(pts, dtype=np.float64)
    out = np.full(len(pts), np.nan, dtype=np.float64)
    if len(pts) == 0:
        return out
    tree      = _geom.bb_tree(mesh, mesh.topology.dim)
    cand      = _geom.compute_collisions_points(tree, pts)
    colliding = _geom.compute_colliding_cells(mesh, cand, pts)
    keep_pts, keep_cells, keep_idx = [], [], []
    for i in range(len(pts)):
        links = colliding.links(i)
        if len(links) > 0:
            keep_idx.append(i)
            keep_cells.append(links[0])
            keep_pts.append(pts[i])
    if keep_idx:
        vals = fn.eval(np.array(keep_pts, dtype=np.float64),
                       np.array(keep_cells, dtype=np.int32))
        out[np.array(keep_idx)] = np.asarray(vals).reshape(-1)
    return out


def compute_interface_velocity_paper(phi: fem.Function,
                                     Zn: fem.Function,
                                     O2: fem.Function,
                                     DeZn: fem.Function,
                                     DeO2: fem.Function,
                                     v_field: fem.Function,
                                     params,
                                     h: float,
                                     t: float,
                                     gpu_asm=None,
                                     exact_O2_override=None) -> None:
    """
    Paper-faithful interface recession velocity -- Ariyarathna et al.,
    Corrosion Science, Eqs. (13)-(16).  Mixed reaction/transport control:

      n      = grad(phi)/|grad(phi)|                       (points into solid)
      v_Zn   = -D_Zn^eff (grad C_Zn . n) / ([Zn]_sol - [Zn]_sat)   Eq.(13)
      v_ORR  = -2 k_ORR C_O2|Gamma / [Zn]_sol                       Eq.(14)
      v_O2   = -2 D_O2^eff (grad C_O2 . n) / [Zn]_sol               Eq.(15)
      v_n    = max(v_Zn, v_ORR, v_O2)                               Eq.(16)

    All three are recession rates (<= 0); the max picks the locally
    rate-limiting (least-negative) mechanism, giving a continuous reaction ->
    O2-transport -> Zn-transport control transition without switching logic.

    Uses the exact FE nodal concentration gradient dotted with the interface
    normal (the paper's grad C . n) -- NOT a two-point nearest-DOF probe.  The
    probe approach failed because Stage-1's 0.002 mm probe length is far below
    the FEniCSx mesh spacing, collapsing both probe points to one vertex.  The
    nodal gradient is resolution-consistent and reuses the same volume-weighted
    gradient kernel already used for redistancing / the coarea factor.

    [Zn]_sol = rho_Zn/M_Zn (mol/mm3); [Zn]_sat from params.Znbc.  k_ORR from
    params.k_orr.  D_*^eff are the film-reduced diffusivities (Eq.12).
    """
    k_orr    = getattr(params, 'k_orr', 0.015)     # mm/h
    Zn_sol   = _ZN_SOLID                             # mol/mm3 (rho_Zn/M_Zn)
    Zn_sat   = getattr(params, 'Znbc', 0.0)          # saturation conc (denominator)
    denom_zn = Zn_sol - Zn_sat
    if abs(denom_zn) < 1e-30:
        denom_zn = Zn_sol

    interface_band = np.abs(phi.x.array) < 3.0 * h
    v_field.x.array[:] = 0.0
    if np.sum(interface_band) == 0:
        v_field.x.scatter_forward()
        return

    # Interface normal n = grad(phi)/|grad(phi)|  (points toward solid, +phi)
    gphi = _nodal_grad(phi.x.array, phi, gpu_asm)
    gmag = np.sqrt(gphi[:, 0]**2 + gphi[:, 1]**2 + gphi[:, 2]**2 + 1e-12)
    nx, ny, nz = gphi[:, 0]/gmag, gphi[:, 1]/gmag, gphi[:, 2]/gmag

    # grad C . n for Zn and O2
    gZn = _nodal_grad(Zn.x.array, phi, gpu_asm)
    gO2 = _nodal_grad(O2.x.array, phi, gpu_asm)
    dZn_n = gZn[:, 0]*nx + gZn[:, 1]*ny + gZn[:, 2]*nz
    dO2_n = gO2[:, 0]*nx + gO2[:, 1]*ny + gO2[:, 2]*nz

    DeZn_a = np.maximum(DeZn.x.array, 0.0)
    DeO2_a = np.maximum(DeO2.x.array, 0.0)

    # O2 that drives the reaction term.  Priority:
    #   1. exact_O2_override (from compute_exact_orr_quantities): the SAME
    #      per-cell exact phi=0 patch value that was actually consumed by the
    #      O2 PDE's exact sink -- closes the conservation gap by construction.
    #   2. orr_probe_dist > 0: FIXED physical distance into the electrolyte
    #      along -n, reproducing Stage-1 FreeFem's hardcoded 0.002 mm probe.
    #   3. local nodal value (legacy default).
    # The nodal value collapses toward 0 as the mesh is refined (the regularised
    # ORR sink sharpens), which makes recession *decrease* under refinement.
    if exact_O2_override is not None:
        # exact_O2_override is the dict from compute_exact_orr_quantities.
        # Only DOFs actually adjacent to a cut cell (cut_mask_dofs) have a
        # meaningful exact value -- elsewhere the projection defaults to 0,
        # which must NOT be used verbatim (it would zero out v_orr for most
        # of the wider |phi|<3h_avg band, since that band is much wider than
        # the true cut-cell zone). Fall back to the nodal value there instead.
        _ov = exact_O2_override
        O2_a = np.maximum(O2.x.array, 0.0).copy()
        O2_a[_ov["cut_mask_dofs"]] = np.maximum(_ov["dof_O2_interface"][_ov["cut_mask_dofs"]], 0.0)
    else:
        probe_d = float(getattr(params, 'orr_probe_dist', 0.0) or 0.0)
        if probe_d > 0.0:
            band_idx = np.flatnonzero(interface_band)
            coords   = phi.function_space.tabulate_dof_coordinates()
            pts = coords[band_idx].copy()
            pts[:, 0] -= probe_d * nx[band_idx]
            pts[:, 1] -= probe_d * ny[band_idx]
            pts[:, 2] -= probe_d * nz[band_idx]
            O2_a = np.maximum(O2.x.array, 0.0).copy()
            probed = _eval_at_points(O2, pts)
            ok = np.isfinite(probed)
            O2_a[band_idx[ok]] = np.maximum(probed[ok], 0.0)
        else:
            O2_a = np.maximum(O2.x.array, 0.0)

    v_zn  = -DeZn_a * np.abs(dZn_n) / denom_zn
    v_o2d = -2.0 * DeO2_a * np.abs(dO2_n) / Zn_sol

    # Eq.(14): reaction-limited term. _V_ORR = 2/Zn_sol = 18320.0, matching
    # FreeFem's undivided -2*kORR*O2/Znsolid exactly (no O2 molar-mass term).
    #
    # FreeFem's actual vO2React (physics/interface_velocity.idp) is the PURE, undivided
    # -2*kORR*O2/Znsolid -- no film-transport resistance term at all. Film
    # slowdown enters ONLY through DeO2 in the separate v_o2d branch below (and
    # through however that feeds back into the O2 field over time), never
    # baked into the reaction term itself.
    #
    # The R_diff=film_length/DeO2 term previously added here has no FreeFem
    # counterpart: it's an extra resistance-in-series this code invented, which
    # analytically suppresses v_orr to ~46% of FreeFem's value at late-time
    # film state (R_diff=0.085 vs 1/k_orr=0.071 at k_orr=14) -- a real,
    # self-inflicted damping with no basis in the reference model.
    if getattr(params, 'legacy_vorr_r_diff', False):
        film_len = getattr(params, 'film_length', 0.050)
        R_diff   = film_len / np.maximum(DeO2_a, 1e-30)
        v_orr    = -_V_ORR * O2_a / (1.0 / k_orr + R_diff)
    else:
        v_orr    = -_V_ORR * k_orr * O2_a

    # O2-transport limited recession flux (paper Eq. 15): v_n = v_o2d = -2 D_O2^eff (grad C_O2 . n) / [Zn]_sol
    # As protective film F forms (F -> Fmax), DeO2 drops from 9.36 to 0.585 mm2/h, causing
    # v_o2d to decelerate from 1.63e-3 down to 8.73e-5 mm/h (matching Stage-1 decelerating mass loss).
    v_n = np.maximum(v_zn, np.maximum(v_orr, v_o2d))  # Eq.(16): least-negative = rate-limiting
    v_n = np.minimum(v_n, 0.0)   # pure dissolution: recession only

    v_field.x.array[:] = np.where(interface_band, v_n, 0.0)
    v_field.x.scatter_forward()

    comm = phi.function_space.mesh.comm
    if comm.rank == 0:
        b = interface_band
        if b.any():
            print(f"  v_paper [{v_n[b].min():.3e}, {v_n[b].max():.3e}] mm/h  "
                  f"vZn[{v_zn[b].min():.2e},{v_zn[b].max():.2e}] "
                  f"vORR[{v_orr[b].min():.2e},{v_orr[b].max():.2e}] "
                  f"vO2[{v_o2d[b].min():.2e},{v_o2d[b].max():.2e}]")
            _winner = np.stack([v_zn[b], v_orr[b], v_o2d[b]], axis=1)
            _which = np.argmax(_winner, axis=1)   # 0=vZn 1=vORR 2=vO2d
            _frozen_mask = np.abs(v_n[b]) < 1e-12
            _frozen = np.mean(_frozen_mask) * 100.0
            print(f"  [WINNER] vZn={np.mean(_which==0)*100:.1f}%  vORR={np.mean(_which==1)*100:.1f}%  "
                  f"vO2={np.mean(_which==2)*100:.1f}%  frozen(v_n~0)={_frozen:.1f}%  "
                  f"total_mean|v_n|={np.abs(v_n[b]).mean():.4e}  "
                  f"mean|v_n|_over_ORRwinners={np.abs(v_n[b][_which==1]).mean() if (_which==1).any() else 0.0:.4e}")

            # Bucket frozen fraction by distance from the true phi=0 front, to
            # separate real stalling AT the interface from harmless "band
            # padding" DOFs beyond where O2/Zn have any nonzero gradient
            # (DeO2/DeCl/DeOH are masked to 0 for phi>0, and O2's IC is 0
            # inside the scaffold with nothing to carry it further in, so
            # DOFs beyond ~2h into the solid are guaranteed v_orr=v_o2d=0
            # regardless of whether the front itself is moving normally).
            _abs_phi_b = np.abs(phi.x.array)[b]
            _edges  = [0.0, h, 2.0*h, 3.0*h]
            _labels = ["[0,h)", "[h,2h)", "[2h,3h)"]
            _parts = []
            for _lo, _hi, _lbl in zip(_edges[:-1], _edges[1:], _labels):
                _band_i = (_abs_phi_b >= _lo) & (_abs_phi_b < _hi)
                _n_i = int(_band_i.sum())
                _pct_frozen_i = (np.mean(_frozen_mask[_band_i]) * 100.0) if _n_i > 0 else float('nan')
                _parts.append(f"{_lbl} n={_n_i} frozen={_pct_frozen_i:.1f}%")
            print(f"  [FROZEN by dist]  " + "  ".join(_parts))

