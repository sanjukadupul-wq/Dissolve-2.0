#!/usr/bin/env python3
"""
mechanics_fe.py — 3D linear-elastic FE with phi-WEIGHTED (SIMP) stiffness (DOLFINx 0.10).

Phase 2 of the topology comparison.  The scaffold degrades via a level set; over a
week the interface recedes only microns — far below one element — so a BINARY
in/out mesh extraction cannot register thinning (E* stays frozen).  Instead we use
a density-based (SIMP-style) modulus:

    E(x) = E_solid * rho(x),   rho = clamp( H_eps(phi), rho_min, 1 )

so each element's stiffness scales with its SOLID FRACTION.  Sub-element recession
lowers rho -> lowers E -> E*(t) responds continuously, and strut thinning shows up
long before the interface crosses a whole element.

We solve on the {phi > -eps} sub-domain (scaffold + a thin partially-degraded
shell), uniaxial compression via platen-contact band BCs, and report the apparent
modulus  E* = |F_axial| / (A_envelope * strain).  Called per timepoint from
dissolve.py (phi in memory).
"""
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx import fem
import dolfinx.fem.petsc


def _corner(v, a_idx):
    """Product of edge cut-params from vertex a_idx to the other 3 (corner tet vol)."""
    rows = np.arange(v.shape[0])
    a = v[rows, a_idx][:, None]
    denom = a - v
    denom[rows, a_idx] = 1.0            # avoid /0 at the self column
    r = a / denom
    r[rows, a_idx] = 1.0               # exclude self from the product
    return np.prod(r, axis=1)


def _tetvol(a, b, c, d):
    return np.abs(np.einsum('ij,ij->i', b - a, np.cross(c - a, d - a))) / 6.0


def _solid_fraction(phiv):
    """EXACT per-cell fraction where the linear field phiv (n,4) is > 0
    (affine-invariant marching-tetrahedra).  Continuous in phi: for mixed-sign
    cells the cut plane moves smoothly with phi, so sub-element recession registers
    even before any vertex flips sign."""
    n = phiv.shape[0]
    frac = np.zeros(n)
    pos = phiv > 0.0
    npos = pos.sum(axis=1)
    frac[npos == 4] = 1.0

    m1 = npos == 1                      # single positive → corner tet at it
    if m1.any():
        v = phiv[m1]
        frac[m1] = _corner(v, np.argmax(v > 0.0, axis=1))

    m3 = npos == 3                      # single negative → 1 - corner tet at it
    if m3.any():
        v = phiv[m3]
        frac[m3] = 1.0 - _corner(v, np.argmin(v > 0.0, axis=1))

    m2 = npos == 2                      # 2-2 wedge: reference-tet prism decomposition
    if m2.any():
        v = phiv[m2]
        m = v.shape[0]; rows = np.arange(m)
        order = np.argsort(~(v > 0.0), axis=1)         # positives first
        pa = v[rows, order[:, 0]]; pb = v[rows, order[:, 1]]
        na = v[rows, order[:, 2]]; nb = v[rows, order[:, 3]]
        z = np.zeros(m)
        A = np.stack([z, z, z], 1); B = np.stack([z + 1, z, z], 1)
        q1 = np.stack([z, pa / (pa - na), z], 1)                 # pa–na
        q2 = np.stack([z, z, pa / (pa - nb)], 1)                 # pa–nb
        q3 = np.stack([1 - pb / (pb - na), pb / (pb - na), z], 1)  # pb–na
        q4 = np.stack([1 - pb / (pb - nb), z, pb / (pb - nb)], 1)  # pb–nb
        frac[m2] = 6.0 * (_tetvol(A, q1, q2, B)
                          + _tetvol(q1, q2, B, q3)
                          + _tetvol(q2, B, q3, q4))
    return np.clip(frac, 0.0, 1.0)


def _tri_area(p0, p1, p2):
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=-1)


def cut_patch_geometry(phiv, coords, field_v=None):
    """EXACT per-cell phi=0 cut patch (affine-invariant marching-tetrahedra).

    Companion to _solid_fraction: that function returns the volume on one side
    of the cut; this returns the geometry OF the cut itself -- the patch area,
    and (if field_v given) the patch-averaged value of another P1 field there.
    Reuses the same edge-intersection parameters, just returns the interpolated
    points instead of only the enclosed volume.

    phiv     : (n,4) phi at each cell's 4 vertices
    coords   : (n,4,3) vertex coordinates for the same cells
    field_v  : optional (n,4) another P1 field's vertex values (e.g. O2), to
               average over the cut patch. A P1 field restricted to any planar
               cut of a tet is exactly affine in the cut plane, so the mean of
               a triangle's 3 corner values is the exact patch average (no
               quadrature error) -- same for a planar quad split into 2 tris.

    Returns (area (n,), avg_field (n,) or None).  area=0 for uncut cells
    (all 4 vertices the same sign); avg_field=0 there too.
    """
    n = phiv.shape[0]
    area = np.zeros(n)
    avgf = np.zeros(n) if field_v is not None else None
    pos = phiv > 0.0
    npos = pos.sum(axis=1)

    # 1-3 split (corner tet cut off): find the single "minority" vertex (whichever
    # sign has only one vertex), cut points lie on its 3 edges to the other 3.
    m13 = (npos == 1) | (npos == 3)
    if m13.any():
        v  = phiv[m13]
        xc = coords[m13]
        minority_is_pos = (npos[m13] == 1)
        # index of the lone minority vertex in each row
        a_idx = np.where(minority_is_pos, np.argmax(pos[m13], axis=1),
                                           np.argmin(pos[m13], axis=1))
        rows = np.arange(len(v))
        a_val = v[rows, a_idx]
        a_pt  = xc[rows, a_idx]
        a_f   = field_v[m13][rows, a_idx] if field_v is not None else None
        # Build the 3 "other" vertex indices per row explicitly (order doesn't
        # matter for area/avg, only that we hit all 3 non-a vertices once each).
        all_idx = np.tile(np.arange(4), (len(v), 1))
        other_mask = all_idx != a_idx[:, None]
        other_idx = all_idx[other_mask].reshape(len(v), 3)
        cut_pts = np.empty((len(v), 3, 3))
        cut_f   = np.empty((len(v), 3)) if field_v is not None else None
        for k in range(3):
            b_idx = other_idx[:, k]
            b_val = v[rows, b_idx]
            t = a_val / (a_val - b_val)               # phi=0 crossing fraction from a
            cut_pts[:, k, :] = a_pt + t[:, None] * (xc[rows, b_idx] - a_pt)
            if field_v is not None:
                b_f = field_v[m13][rows, b_idx]
                cut_f[:, k] = a_f + t * (b_f - a_f)
        tri_area = _tri_area(cut_pts[:, 0], cut_pts[:, 1], cut_pts[:, 2])
        area[m13] = tri_area
        if field_v is not None:
            avgf[m13] = cut_f.mean(axis=1)

    # 2-2 split (wedge cut): quad patch q1-q3-q4-q2 (cyclic), split into 2 tris.
    m2 = npos == 2
    if m2.any():
        v  = phiv[m2]
        xc = coords[m2]
        m = v.shape[0]; rows = np.arange(m)
        order = np.argsort(~(v > 0.0), axis=1)          # positives first: [pa, pb, na, nb]
        pa_i, pb_i, na_i, nb_i = order[:, 0], order[:, 1], order[:, 2], order[:, 3]
        pa, pb = v[rows, pa_i], v[rows, pb_i]
        na, nb = v[rows, na_i], v[rows, nb_i]
        Pa, Pb = xc[rows, pa_i], xc[rows, pb_i]
        Na, Nb = xc[rows, na_i], xc[rows, nb_i]

        def _cut(P, Q, pv, qv):
            t = pv / (pv - qv)
            return P + t[:, None] * (Q - P), t

        q1, t1 = _cut(Pa, Na, pa, na)   # edge pa-na
        q2, t2 = _cut(Pa, Nb, pa, nb)   # edge pa-nb
        q3, t3 = _cut(Pb, Na, pb, na)   # edge pb-na
        q4, t4 = _cut(Pb, Nb, pb, nb)   # edge pb-nb

        area_1 = _tri_area(q1, q3, q4)
        area_2 = _tri_area(q1, q4, q2)
        area[m2] = area_1 + area_2

        if field_v is not None:
            fv = field_v[m2]
            fpa, fpb = fv[rows, pa_i], fv[rows, pb_i]
            fna, fnb = fv[rows, na_i], fv[rows, nb_i]
            f1 = fpa + t1 * (fna - fpa)
            f2 = fpa + t2 * (fnb - fpa)
            f3 = fpb + t3 * (fna - fpb)
            f4 = fpb + t4 * (fnb - fpb)
            avg_1 = (f1 + f3 + f4) / 3.0
            avg_2 = (f1 + f4 + f2) / 3.0
            total = area_1 + area_2
            safe = total > 1e-30
            combined = np.zeros(m)
            combined[safe] = (area_1[safe]*avg_1[safe] + area_2[safe]*avg_2[safe]) / total[safe]
            avgf[m2] = combined

    return area, avgf


def _largest_connected_component(included, cellnodes):
    """Restrict `included` cells to the LARGEST connected component (cells sharing
    a vertex).  Level-set redistancing can leave spurious isolated "phantom" solid
    islands far from the true structure (verified: PDE reinitialization degrades
    with distance from the interface -- |grad phi| deviated 7x from the ideal 1.0
    far away, occasionally flipping sign). A single disconnected outlier corrupts
    amin/amax (hence gauge length and contact-band placement) and, physically,
    isolated debris cannot carry compressive load anyway -- so keeping only the
    largest component is both the numerics fix and the physically correct model."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    verts = cellnodes[included]                 # (n_incl, 4) vertex ids
    n = len(included)
    local_cell = np.repeat(np.arange(n), 4)
    vflat = verts.reshape(-1)
    order = np.argsort(vflat)
    vs, cs = vflat[order], local_cell[order]
    # cells adjacent in the vertex-sorted order that share the same vertex -> edge
    same_vertex = vs[1:] == vs[:-1]
    src, dst = cs[:-1][same_vertex], cs[1:][same_vertex]
    graph = coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    ncomp, labels = connected_components(graph, directed=False)
    if ncomp <= 1:
        return included
    sizes = np.bincount(labels, minlength=ncomp)
    keep = labels == np.argmax(sizes)
    return included[keep]


def _sigma(w, lam, mu, tdim):
    e = ufl.sym(ufl.grad(w))
    return lam * ufl.tr(e) * ufl.Identity(tdim) + 2.0 * mu * e


def scaffold_stiffness(domain, phi, E_solid=90000.0, nu=0.30, axis=2,
                       strain=0.005, sigma_yield=140.0, bc_band=0.5,
                       eps=0.5, rho_min=1e-4):
    """
    phi-weighted apparent modulus of the degrading scaffold [MPa].
    eps      : Heaviside/shell half-width (mm); ~2x element size.
    rho_min  : void modulus floor (keeps the domain non-singular).
    Returns dict(E_star_MPa, sigma_vm_max_MPa, connected, solid_vol_mm3, ...).
    """
    tdim = domain.topology.dim
    V = phi.function_space
    cellnodes = V.dofmap.list                      # (ncells, nverts) = vertices (P1)
    phi_v = phi.x.array[cellnodes]                 # nodal phi per cell (ncells,4)
    # SIMP density = EXACT analytic per-cell solid fraction (marching-tetrahedra).
    # Integrates to the true {phi>0} volume (no phantom shell, no thin-feature bias)
    # AND is continuous in phi (mixed-sign cells' cut plane moves smoothly), so
    # sub-element recession lowers E* immediately.
    rho_all = _solid_fraction(phi_v)
    included = np.where(rho_all > 0.0)[0].astype(np.int32)   # cells touching solid
    fail = dict(E_star_MPa=0.0, sigma_vm_max_MPa=0.0, connected=False,
                solid_vol_mm3=0.0, n_cells=int(len(included)))
    if len(included) == 0:
        return fail
    included = _largest_connected_component(included, cellnodes)

    sub, em, *_ = dolfinx.mesh.create_submesh(domain, tdim, included)
    nc_sub = sub.topology.index_map(tdim).size_local
    parent = np.asarray(em.sub_topology_to_topology(
        np.arange(nc_sub, dtype=np.int32), False))   # submesh cell -> parent cell
    xg = sub.geometry.x
    amin, amax = xg[:, axis].min(), xg[:, axis].max()
    H = amax - amin
    lat = [d for d in range(tdim) if d != axis]
    A_env = float(np.prod([xg[:, d].max() - xg[:, d].min() for d in lat]))
    if H <= 1e-9 or A_env <= 1e-12:
        return fail

    # per-cell modulus on the submesh (DG0), scaled by solid fraction
    rho_sub = np.clip(rho_all[parent], rho_min, 1.0)
    Vdg = fem.functionspace(sub, ("DG", 0))
    Efield = fem.Function(Vdg)
    cell_dof = Vdg.dofmap.list.reshape(-1)         # cell -> its DG0 dof
    Efield.x.array[cell_dof] = E_solid * rho_sub
    solid_vol = float(fem.assemble_scalar(fem.form((Efield / E_solid) * ufl.dx)))

    mu = Efield / (2.0 * (1.0 + nu))
    lam = Efield * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    V = fem.functionspace(sub, ("Lagrange", 1, (tdim,)))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    k_reg = E_solid * 1e-8   # tiny elastic foundation: regularises floating/near-singular parts
    a = (ufl.inner(_sigma(u, lam, mu, tdim), ufl.sym(ufl.grad(v)))
         + k_reg * ufl.inner(u, v)) * ufl.dx
    L = ufl.inner(fem.Constant(sub, np.zeros(tdim)), v) * ufl.dx

    def bot(x): return x[axis] <= amin + bc_band + 1e-9
    def top(x): return x[axis] >= amax - bc_band - 1e-9
    Vax, _ = V.sub(axis).collapse()
    bot_dofs = fem.locate_dofs_geometrical((V.sub(axis), Vax), bot)
    top_dofs = fem.locate_dofs_geometrical((V.sub(axis), Vax), top)
    if len(bot_dofs[0]) == 0 or len(top_dofs[0]) == 0:
        return dict(fail, solid_vol_mm3=solid_vol)

    # Frictionless-platen BCs: constrain ONLY the axial component at both bands,
    # leave lateral (Poisson) displacement free. Clamping all 3 components at the
    # bottom (as before) over-constrains lateral expansion there, producing a
    # confined/oedometric-like state -- E* is inflated by ~E(1-nu)/[(1+nu)(1-2nu)]
    # (~1.35x at nu=0.30), NOT the true Young's modulus. Verified via a uniform
    # solid-cube sanity check (E* should equal E_solid exactly at rho=1 everywhere).
    delta = strain * H
    u_bot = fem.Function(Vax); u_bot.x.array[:] = 0.0
    bc_bot = fem.dirichletbc(u_bot, bot_dofs, V.sub(axis))
    u_top = fem.Function(Vax); u_top.x.array[:] = -delta
    bc_top = fem.dirichletbc(u_top, top_dofs, V.sub(axis))
    bcs = [bc_bot, bc_top]

    a_c, L_c = fem.form(a), fem.form(L)
    A = dolfinx.fem.petsc.create_matrix(a_c); A.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(A, a_c, bcs=bcs); A.assemble()
    b = dolfinx.fem.petsc.assemble_vector(L_c)
    dolfinx.fem.petsc.apply_lifting(b, [a_c], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, bcs)

    uh = fem.Function(V)
    def _solve(mode):
        ksp = PETSc.KSP().create(sub.comm)
        ksp.setOperators(A)
        if mode == "lu":                       # robust direct fallback
            ksp.setType("preonly"); pc = ksp.getPC(); pc.setType("lu")
            try: pc.setFactorSolverType("mumps")
            except Exception: pass
        else:
            ksp.setType("cg"); ksp.getPC().setType("gamg")
            ksp.setTolerances(rtol=1e-8, max_it=1000)
        try:
            ksp.solve(b, uh.x.petsc_vec); uh.x.scatter_forward()
            return ksp.getConvergedReason() > 0
        except Exception:
            return False
    ok = _solve("iter")
    if not ok:                                 # near-singular network (e.g. thin necks)
        uh.x.array[:] = 0.0
        ok = _solve("lu")
    if not ok:
        return dict(fail, solid_vol_mm3=solid_vol)

    # axial reaction = sum of internal nodal forces (K.u) at the pushed top dofs
    R = dolfinx.fem.petsc.assemble_vector(fem.form(ufl.action(a, uh)))
    R.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    F_axial = float(np.sum(R.getArray()[top_dofs[0]]))
    # The two contact bands are each forced to move as a rigid slab (uniform
    # prescribed axial displacement throughout their thickness), so they carry
    # zero internal strain -- the true gauge length is H reduced by both bands,
    # not H itself. Using nominal strain=delta/H here would inflate E* by
    # H/(H-2*bc_band) (verified via a uniform-solid-cube sanity check).
    H_eff = max(H - 2.0 * bc_band, 1e-9)
    strain_eff = delta / H_eff
    E_star = abs(F_axial) / (A_env * strain_eff) if strain_eff > 0 else 0.0

    try:
        s = _sigma(uh, lam, mu, tdim)
        dev = s - (1.0 / 3.0) * ufl.tr(s) * ufl.Identity(tdim)
        vm = ufl.sqrt(1.5 * ufl.inner(dev, dev) + 1e-30)
        vmf = fem.Function(Vdg)
        vmf.interpolate(fem.Expression(vm, Vdg.element.interpolation_points()))
        vm_max = float(vmf.x.array.max())
    except Exception:
        vm_max = 0.0

    return dict(E_star_MPa=E_star, sigma_vm_max_MPa=vm_max, connected=True,
                solid_vol_mm3=solid_vol, n_cells=int(len(included)),
                yielded=bool(vm_max > sigma_yield))


# ── standalone validation on an initial (undegraded) mesh ─────────────────────
if __name__ == "__main__":
    import argparse
    from mesh_utils import convert_mesh_to_xdmf, load_mesh, redistance
    p = argparse.ArgumentParser()
    p.add_argument("--input_mesh", required=True)
    p.add_argument("--E_solid", type=float, default=90000.0)
    p.add_argument("--strain", type=float, default=0.005)
    p.add_argument("--eps", type=float, default=0.5)
    args = p.parse_args()

    comm = MPI.COMM_WORLD
    xdmf = convert_mesh_to_xdmf(args.input_mesh) if args.input_mesh.endswith(".mesh") else args.input_mesh
    domain, cell_tags = load_mesh(xdmf, comm)
    # build phi from region tags (scaffold=1) then redistance, like the solver
    V = fem.functionspace(domain, ("Lagrange", 1))
    phi = fem.Function(V)
    phi.x.array[:] = -1.0
    scaf = cell_tags.indices[cell_tags.values == 1]
    phi.x.array[V.dofmap.list[scaf].reshape(-1)] = 1.0
    redistance(phi, domain, num_iters=15)
    r = scaffold_stiffness(domain, phi, E_solid=args.E_solid, strain=args.strain, eps=args.eps)
    print(f"mesh {args.input_mesh}")
    print(f"  effective solid volume : {r['solid_vol_mm3']:.2f} mm^3")
    print(f"  apparent modulus E*    : {r['E_star_MPa']:.1f} MPa  ({r['E_star_MPa']/1000:.3f} GPa)")
    print(f"  E*/E_solid             : {r['E_star_MPa']/args.E_solid:.4f}")
    print(f"  peak von Mises         : {r['sigma_vm_max_MPa']:.2f} MPa   connected={r['connected']}")
