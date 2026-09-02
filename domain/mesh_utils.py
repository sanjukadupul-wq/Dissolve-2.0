"""
mesh_utils.py — Mesh loading, φ initialization, and redistancing for Dissolve 2.0.
Converts Medit .mesh → XDMF for DOLFINx, initializes signed distance function.
"""

import os
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import mesh as dmesh, fem, io
import dolfinx.fem.petsc
import ufl

try:
    import meshio
except ImportError:
    meshio = None


def nodal_gradient_cpu(phi_array: np.ndarray, domain, V) -> np.ndarray:
    """
    Volume-weighted-average nodal gradient of a P1 scalar field, pure numpy.

    This mirrors gpu_assembler.py's `nodal_gradient_p1` CUDA kernel exactly
    (same per-cell constant gradient, same cell-volume weighting, same
    scatter-accumulate-then-divide reduction) so the CPU and GPU code paths
    compute numerically consistent geometry-derived quantities.

    Previously the CPU fallback used `fem.Expression(ufl.grad(phi)[i],
    interpolation_points).interpolate(...)`, which for a P1 field's
    (piecewise-constant, cell-discontinuous) gradient just overwrites each
    shared DOF with whichever cell DOLFINx processes last -- no averaging at
    all.  That produced a systematically noisier, non-representative |grad
    phi| field, which measurably degraded level-set redistancing convergence
    (observed: CPU |grad phi| mean=1.20 vs GPU mean=1.01 after the same 15
    iterations) and cascaded into a >60x divergence in predicted mass loss
    between the two paths on the same problem.

    Returns
    -------
    (n_dofs, 3) ndarray of [gx, gy, gz] at each P1 DOF (mesh vertex).
    """
    tdim = domain.topology.dim
    n_cells = domain.topology.index_map(tdim).size_local
    n_dofs = len(phi_array)

    elem_nodes = np.array(
        [V.dofmap.cell_dofs(i) for i in range(n_cells)], dtype=np.int64
    )
    coords = domain.geometry.x[:, :3]
    edges = coords[elem_nodes][:, 1:, :] - coords[elem_nodes][:, 0:1, :]
    J = edges.transpose(0, 2, 1)
    detJ = np.linalg.det(J)
    Jinv = np.linalg.inv(J)
    JinvT = Jinv.transpose(0, 2, 1)

    grad_ref = np.array([[-1.0, -1.0, -1.0],
                         [ 1.0,  0.0,  0.0],
                         [ 0.0,  1.0,  0.0],
                         [ 0.0,  0.0,  1.0]])

    u = phi_array[elem_nodes]                          # (n_cells, 4)
    gu_ref = np.einsum('ca,ad->cd', u, grad_ref)        # (n_cells, 3)
    gu_phys = np.einsum('cij,cj->ci', JinvT, gu_ref)    # (n_cells, 3)

    w = np.abs(detJ) / 6.0                              # cell volume

    grad_accum = np.zeros((n_dofs, 3))
    wsum_accum = np.zeros(n_dofs)
    for local in range(4):
        node_ids = elem_nodes[:, local]
        np.add.at(grad_accum, node_ids, w[:, None] * gu_phys)
        np.add.at(wsum_accum, node_ids, w)

    wsum_accum = np.maximum(wsum_accum, 1e-300)
    return grad_accum / wsum_accum[:, None]


def convert_mesh_to_xdmf(mesh_path: str, output_dir: str = None) -> str:
    """
    Convert a Medit .mesh file to XDMF format for DOLFINx.
    Uses a built-in parser (no meshio dependency for reading).
    Returns the path to the output XDMF file.
    Only rank 0 does the conversion.
    """
    import h5py

    comm = MPI.COMM_WORLD
    if output_dir is None:
        output_dir = os.path.dirname(mesh_path) or "."

    base = os.path.splitext(os.path.basename(mesh_path))[0]
    xdmf_path = os.path.join(output_dir, f"{base}.xdmf")
    h5_path = os.path.join(output_dir, f"{base}.h5")

    if comm.rank == 0:
        if not os.path.exists(xdmf_path):
            print(f"Converting {mesh_path} → {xdmf_path} ...")
            points, cells, cell_data = _read_medit_mesh(mesh_path)
            _write_xdmf(xdmf_path, h5_path, points, cells, cell_data, base)
            print(f"  → Written {len(cells)} tetrahedra, {len(points)} vertices")
        else:
            print(f"XDMF already exists: {xdmf_path}")

    comm.Barrier()
    return xdmf_path


def _read_medit_mesh(path: str):
    """
    Parse a Medit .mesh file (MeshVersionFormatted 1).
    Returns (points, tetra_cells, cell_regions).
    """
    points = []
    cells = []
    cell_data = []

    with open(path, "r") as f:
        lines = f.readlines()

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()

        if line == "Vertices":
            i += 1
            nv = int(lines[i].strip())
            i += 1
            for _ in range(nv):
                parts = lines[i].strip().split()
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                i += 1

        elif line == "Tetrahedra":
            i += 1
            nt = int(lines[i].strip())
            i += 1
            for _ in range(nt):
                parts = lines[i].strip().split()
                # Medit uses 1-based indexing
                cells.append([int(parts[0]) - 1, int(parts[1]) - 1,
                              int(parts[2]) - 1, int(parts[3]) - 1])
                cell_data.append(int(parts[4]) if len(parts) > 4 else 0)
                i += 1
        else:
            i += 1

    return (np.array(points, dtype=np.float64),
            np.array(cells, dtype=np.int64),
            np.array(cell_data, dtype=np.int32))


def _write_xdmf(xdmf_path, h5_path, points, cells, cell_data, name):
    """Write mesh data to XDMF + HDF5 for DOLFINx."""
    import h5py

    h5_basename = os.path.basename(h5_path)

    # Write HDF5
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("mesh/points", data=points)
        h5.create_dataset("mesh/cells", data=cells)
        if len(cell_data) > 0:
            h5.create_dataset("mesh/cell_data", data=cell_data)

    # Write XDMF (XML)
    npts = len(points)
    ncells = len(cells)

    xdmf_content = f"""<?xml version="1.0"?>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="Grid" GridType="Uniform">
      <Topology TopologyType="Tetrahedron" NumberOfElements="{ncells}">
        <DataItem DataType="Int" Dimensions="{ncells} 4" Format="HDF">
          {h5_basename}:/mesh/cells
        </DataItem>
      </Topology>
      <Geometry GeometryType="XYZ">
        <DataItem DataType="Float" Dimensions="{npts} 3" Format="HDF" Precision="8">
          {h5_basename}:/mesh/points
        </DataItem>
      </Geometry>
      <Attribute Name="region" AttributeType="Scalar" Center="Cell">
        <DataItem DataType="Int" Dimensions="{ncells}" Format="HDF">
          {h5_basename}:/mesh/cell_data
        </DataItem>
      </Attribute>
    </Grid>
  </Domain>
</Xdmf>
"""
    with open(xdmf_path, "w") as f:
        f.write(xdmf_content)


def load_mesh(xdmf_path: str, comm=None):
    """
    Load a mesh from XDMF into DOLFINx.
    Returns (mesh, cell_tags) where cell_tags may be None.

    Two-stage cell-tag loading:
      1. Try DOLFINx's read_meshtags (works for files written by DOLFINx itself).
      2. Fallback: read cell_data directly from the companion HDF5 file and
         build MeshTags manually.  This handles meshes converted via _write_xdmf
         (Gmsh .mesh → XDMF) whose Attribute format is not the DOLFINx meshtags
         format and therefore makes read_meshtags return empty/None data.
    """
    if comm is None:
        comm = MPI.COMM_WORLD

    with io.XDMFFile(comm, xdmf_path, "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")
        try:
            cell_tags = xdmf.read_meshtags(domain, name="Grid")
            # Validate: must contain the scaffold label (1) AND at least one
            # other label.  If read_meshtags reads our custom Attribute-format
            # XDMF (from _write_xdmf) it can return junk values that pass a
            # naive non-None / non-empty check — the scaffold label test catches
            # that case and forces the HDF5 fallback.
            if cell_tags is not None and len(cell_tags.values) > 0:
                unique = set(cell_tags.values.tolist())
                if 1 not in unique or len(unique) < 2:
                    cell_tags = None   # junk data — trigger HDF5 fallback
            else:
                cell_tags = None
        except Exception:
            cell_tags = None

    # ── Fallback: read cell_data directly from HDF5 ───────────────────────────
    # _write_xdmf writes the region labels into mesh/cell_data in the .h5 file.
    # In a serial run DOLFINx preserves the cell ordering, so index i in the
    # HDF5 array corresponds to DOLFINx cell i.
    if cell_tags is None:
        h5_path = xdmf_path.replace(".xdmf", ".h5")
        if os.path.exists(h5_path):
            try:
                import h5py
                with h5py.File(h5_path, "r") as h5f:
                    if "mesh/cell_data" in h5f:
                        raw = np.array(h5f["mesh/cell_data"], dtype=np.int32)
                        tdim    = domain.topology.dim
                        n_local = domain.topology.index_map(tdim).size_local
                        n_use   = min(n_local, len(raw))
                        indices = np.arange(n_use, dtype=np.int32)
                        values  = raw[:n_use]
                        cell_tags = dolfinx.mesh.meshtags(
                            domain, tdim, indices, values
                        )
                        unique = sorted(set(values.tolist()))
                        if comm.rank == 0:
                            print(f"  [mesh] Cell labels from HDF5: {unique}  "
                                  f"(1=scaffold, 2=fluid)")
                    else:
                        if comm.rank == 0:
                            print("  [mesh] HDF5 has no mesh/cell_data key")
            except Exception as _e:
                if comm.rank == 0:
                    print(f"  [mesh] HDF5 fallback failed: {_e}")
        else:
            if comm.rank == 0:
                print(f"  [mesh] No HDF5 companion at {h5_path}")

    return domain, cell_tags


def initialize_phi_analytic(V: fem.FunctionSpace, sdf_func) -> fem.Function:
    """
    Initialize phi = -sdf_func(x,y,z) directly from this rank's own DOF
    coordinates (V.tabulate_dof_coordinates()) -- exact analytic signed
    distance, no approximation.

    This exists because _init_phi_from_regions() only has cell region
    labels to work with, so it approximates the signed distance as the
    Euclidean distance to the nearest INTERFACE DOF (a KD-tree over
    boundary vertices), not the true nearest-point-on-surface distance.
    That approximation has ridge/facet artifacts wherever interface-DOF
    spacing is uneven (visible as pockmarks on the phi=0 contour on a
    graded mesh). For our own analytically-defined geometries (disc,
    stent) we already know the exact SDF, so we can evaluate it directly
    -- and since nothing is written to/read from a file, there's no risk
    of the vertex-reordering-scrambles-a-.sol-file problem a real .sol
    would hit.
    """
    phi = fem.Function(V, name="phi")
    coords = V.tabulate_dof_coordinates()
    phi.x.array[:] = -sdf_func(coords[:, 0], coords[:, 1], coords[:, 2])
    phi.x.scatter_forward()
    if MPI.COMM_WORLD.rank == 0:
        print(f"  Initialized phi analytically (exact SDF, no approximation)")
        print(f"  phi range: [{phi.x.array.min():.4f}, {phi.x.array.max():.4f}]")
    return phi


def initialize_phi(domain, cell_tags, params, V: fem.FunctionSpace) -> fem.Function:
    """
    Initialize the level-set function φ.
    φ > 0 inside scaffold, φ < 0 in medium.

    If a .sol file exists, load from it.
    Otherwise, initialize from region labels:
      scaffold (label 1) → +0.5
      medium   (label 2) → -0.5
    """
    phi = fem.Function(V, name="phi")
    sol_path = params.input_mesh + ".sol"

    if os.path.exists(sol_path):
        # Load from .sol file (binary/ASCII Medit solution)
        if MPI.COMM_WORLD.rank == 0:
            print(f"Loading phi from {sol_path}")
        _load_sol_to_function(sol_path, phi, domain)
    elif cell_tags is not None:
        if MPI.COMM_WORLD.rank == 0:
            print("Initializing phi from region labels...")
        _init_phi_from_regions(phi, cell_tags, params.tag_scaffold)
    else:
        if MPI.COMM_WORLD.rank == 0:
            print("WARNING: No .sol file or region tags. Initializing phi = -0.5 (no scaffold)")
        phi.x.array[:] = -0.5

    return phi


def _load_sol_to_function(sol_path: str, phi: fem.Function, domain):
    """Load a Medit .sol file into a FE function (rank 0 reads, then scatter)."""
    comm = domain.comm
    if comm.rank == 0:
        try:
            values = _read_medit_sol(sol_path)
            # Distribute to the function
            phi.x.array[:len(values)] = values[:len(phi.x.array)]
        except Exception as e:
            print(f"Warning: Could not read .sol file: {e}")
            phi.x.array[:] = -0.5
    phi.x.scatter_forward()


def _read_medit_sol(path: str) -> np.ndarray:
    """Read a Medit .sol file and return the scalar values."""
    values = []
    with open(path, "r") as f:
        lines = f.readlines()

    reading = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("SolAtVertices"):
            reading = True
            continue
        if reading and stripped == "End":
            break
        if reading:
            parts = stripped.split()
            for p in parts:
                try:
                    values.append(float(p))
                except ValueError:
                    pass

    return np.array(values, dtype=np.float64)


def _init_phi_from_regions(phi: fem.Function, cell_tags, scaffold_label: int):
    """
    Initialize phi as signed distance from scaffold/medium boundary.
    Positive inside scaffold, negative in medium.
    """
    domain = phi.function_space.mesh
    V = phi.function_space
    comm = domain.comm

    # Get scaffold cells
    scaffold_cells = cell_tags.find(scaffold_label)
    scaffold_set = set(scaffold_cells)

    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, 0)

    dofmap = V.dofmap
    num_cells = domain.topology.index_map(tdim).size_local

    # Step 1: Mark DOFs as scaffold (+1) or medium (-1)
    dof_sign = np.full(len(phi.x.array), -1.0)  # default: medium
    for cell_idx in scaffold_cells:
        dofs = dofmap.cell_dofs(cell_idx)
        dof_sign[dofs] = 1.0

    # Step 2: Find TRUE interface DOFs — only vertices shared between a scaffold
    # cell and a fluid cell.  The previous approach added ALL dofs of straddling
    # fluid cells (including interior fluid vertices), which pulled the phi=0
    # reference set into the fluid and inflated the scaffold volume by ~7%.
    #
    # Correct approach: a DOF is on the interface if and only if it belongs to
    # at least one scaffold cell AND at least one non-scaffold cell.
    scaffold_dofs = set()
    for cell_idx in scaffold_cells:
        scaffold_dofs.update(dofmap.cell_dofs(cell_idx).tolist())

    fluid_dofs = set()
    for c in range(num_cells):
        if c not in scaffold_set:
            fluid_dofs.update(dofmap.cell_dofs(c).tolist())

    interface_dofs = scaffold_dofs & fluid_dofs   # exact intersection = surface DOFs

    interface_dofs = np.array(sorted(interface_dofs), dtype=np.int32)  # surface DOFs only

    if comm.rank == 0:
        print(f"  Interface DOFs found: {len(interface_dofs)}")

    if len(interface_dofs) == 0:
        # Fallback: no interface found, use simple ±0.5
        phi.x.array[:] = dof_sign * 0.5
        phi.x.scatter_forward()
        return

    # Step 3: Compute signed distance using nearest interface DOF
    coords = V.tabulate_dof_coordinates()
    interface_coords = coords[interface_dofs]

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(interface_coords)
        distances, _ = tree.query(coords)
    except ImportError:
        # Fallback: brute force (slower)
        if comm.rank == 0:
            print("  Warning: scipy not available, using brute force distance")
        distances = np.zeros(len(coords))
        for i in range(len(coords)):
            diff = interface_coords - coords[i]
            d = np.sqrt(np.sum(diff**2, axis=1))
            distances[i] = d.min()

    # Signed distance: positive inside scaffold, negative in medium
    phi.x.array[:] = dof_sign * distances

    if comm.rank == 0:
        print(f"  phi range: [{phi.x.array.min():.4f}, {phi.x.array.max():.4f}]")
        near_zero = np.sum(np.abs(phi.x.array) < 0.3)
        print(f"  DOFs with |φ| < 0.3: {near_zero}")

    phi.x.scatter_forward()


def redistance(phi: fem.Function, domain, num_iters: int = 8,
               dt_redist: float = None, h: float = None,
               gpu_asm=None):
    """
    Redistance the level-set function using Sussman pseudo-time iteration.
    Restores |∇φ| ≈ 1 without shifting the zero level-set (interface location).

    Algorithm: ∂ψ/∂τ = sign(φ₀)(1 - |∇ψ|), explicit Euler in pseudo-time.

    Key design decisions (all driven by adaptive mesh behaviour):
      - Uses h_avg (NOT h_min).  On an adaptively-refined mesh h_min can be
        100× smaller than the typical interface element; using h_min makes dtau
        microscopic and the interface band essentially empty, causing catastrophic
        sign-flipping updates near φ=0.
      - Interface band width = 3·h_avg.  Keeps the φ=0 contour frozen over a
        realistically-wide neighbourhood rather than a single node layer.
      - Per-node CFL: dtau_i = 0.5·h_avg / max(|∇φ|_i, 0.5).  Prevents any
        single node from receiving an update larger than its local grid spacing,
        which is what caused the |∇φ| max=1594 instability.
    """
    V = phi.function_space
    comm = domain.comm

    # Always base redistancing parameters on h_avg, not h_min
    h_avg = _compute_h_avg(domain)
    if h is None:
        h = h_avg

    # Frozen interface band: 3·h_avg keeps the zero-contour stable
    interface_width = 3.0 * h_avg
    interface_mask = np.abs(phi.x.array) < interface_width

    phi0_arr = phi.x.array.copy()      # frozen for sign; never modified
    eps_sign = 1.5 * h_avg
    sign_arr = phi0_arr / np.sqrt(phi0_arr**2 + eps_sign**2)

    # Choose gradient backend once, before the iteration loop.
    # GPU path (gpu_asm provided): one nodal_gradient kernel call per iter.
    # CPU path (fallback when gpu_asm is None): nodal_gradient_cpu -- the same
    # volume-weighted-average algorithm as the GPU kernel, just in numpy, so
    # CPU and GPU redistancing converge to numerically consistent geometry
    # instead of diverging (see nodal_gradient_cpu's docstring for why the
    # previous fem.Expression+interpolate CPU fallback was wrong).
    _use_gpu_grad = gpu_asm is not None

    def _get_grad_mag(phi_fn):
        if _use_gpu_grad:
            try:
                import cupy as _cp
                g = gpu_asm.nodal_gradient(phi_fn.x.array)  # (n_dofs, 3) cupy
                gx_a = _cp.asnumpy(g[:, 0])
                gy_a = _cp.asnumpy(g[:, 1])
                gz_a = _cp.asnumpy(g[:, 2])
                return np.sqrt(gx_a**2 + gy_a**2 + gz_a**2 + 1e-12)
            except Exception:
                pass   # fall through to CPU on any GPU error
        # CPU path
        g = nodal_gradient_cpu(phi_fn.x.array, domain, V)
        return np.sqrt(g[:, 0]**2 + g[:, 1]**2 + g[:, 2]**2 + 1e-12)

    if _use_gpu_grad:
        import cupy as _cp
        interface_mask_d = _cp.asarray(interface_mask)
        sign_d           = _cp.asarray(sign_arr)
        phi_d            = _cp.asarray(phi.x.array)
        for _ in range(num_iters):
            g      = gpu_asm.nodal_gradient(phi_d)   # accepts cupy — no H2D
            gm_d   = _cp.sqrt(g[:,0]**2 + g[:,1]**2 + g[:,2]**2 + 1e-12)
            dtau_d = 0.5 * h_avg / _cp.maximum(gm_d, 0.5)
            phi_d += _cp.where(interface_mask_d, 0.0, dtau_d * sign_d * (1.0 - gm_d))
            phi.x.array[:] = _cp.asnumpy(phi_d)
            phi.x.scatter_forward()
            if comm.size > 1:
                phi_d = _cp.asarray(phi.x.array)  # re-sync ghost nodes after MPI exchange
    else:
        for _ in range(num_iters):
            grad_mag  = _get_grad_mag(phi)
            dtau_node = (dt_redist if dt_redist is not None
                         else 0.5 * h_avg / np.maximum(grad_mag, 0.5))
            phi.x.array[:] += np.where(interface_mask, 0.0,
                                       dtau_node * sign_arr * (1.0 - grad_mag))
            phi.x.scatter_forward()

    if comm.rank == 0:
        grad_final = _get_grad_mag(phi)
        far_mask = np.abs(phi.x.array) > interface_width
        if far_mask.sum() > 0:
            gf = grad_final[far_mask]
            print(f"  Redistance: |∇φ| mean={gf.mean():.3f} max={gf.max():.3f} "
                  f"(ideal 1.0, {num_iters} iters, h_avg={h_avg:.4f})")
        arr = phi.x.array
        n_frozen = int(interface_mask.sum())
        print(f"  φ ∈ [{arr.min():.4f}, {arr.max():.4f}]  frozen DOFs: {n_frozen}")


def _compute_h_min(domain) -> float:
    """Compute the minimum element diameter across all processes."""
    tdim = domain.topology.dim
    num_cells = domain.topology.index_map(tdim).size_local
    cell_indices = np.arange(num_cells, dtype=np.int32)
    try:
        h = dolfinx.mesh.h(domain, tdim, cell_indices)
    except (AttributeError, TypeError):
        h = dolfinx.cpp.mesh.h(domain._cpp_object, tdim, cell_indices)
    h_local = h.min() if len(h) > 0 else 1e10
    h_global = domain.comm.allreduce(h_local, op=MPI.MIN)
    return h_global


def _compute_h_avg(domain) -> float:
    """Compute the mean element diameter across all processes.

    Used in redistancing — h_avg is far more representative than h_min for
    adaptively-refined meshes where h_min can be orders of magnitude smaller
    than the typical element near the interface.
    """
    tdim = domain.topology.dim
    num_cells = domain.topology.index_map(tdim).size_local
    cell_indices = np.arange(num_cells, dtype=np.int32)
    try:
        h = dolfinx.mesh.h(domain, tdim, cell_indices)
    except (AttributeError, TypeError):
        h = dolfinx.cpp.mesh.h(domain._cpp_object, tdim, cell_indices)
    h_sum = domain.comm.allreduce(float(h.sum()) if len(h) > 0 else 0.0, op=MPI.SUM)
    n_cells = domain.comm.allreduce(int(len(h)), op=MPI.SUM)
    return h_sum / n_cells if n_cells > 0 else 1.0


def compute_scaffold_volume(phi: fem.Function, domain,
                            epsilon: float = None, gpu_asm=None) -> float:
    """
    Compute scaffold volume using smoothed Heaviside:
    H_ε(φ) = 0 if φ < -ε, 1 if φ > ε,
    0.5*(1 + φ/ε + sin(πφ/ε)/π) otherwise.

    When gpu_asm is provided the integral is evaluated entirely on device
    via gpu_assembler.scaffold_volume (vertex-averaged P1 quadrature).
    """
    if epsilon is None:
        epsilon = 0.5 * _compute_h_min(domain)

    if gpu_asm is not None:
        vol_local = gpu_asm.scaffold_volume(phi.x.array, epsilon)
        return domain.comm.allreduce(vol_local, op=MPI.SUM)

    dx = ufl.Measure("dx", domain=domain)
    H = ufl.conditional(
        ufl.lt(phi, -epsilon), 0.0,
        ufl.conditional(
            ufl.gt(phi, epsilon), 1.0,
            0.5 * (1.0 + phi / epsilon + ufl.sin(ufl.pi * phi / epsilon) / ufl.pi)
        )
    )
    vol_form = fem.form(H * dx)
    vol_local = fem.assemble_scalar(vol_form)
    return domain.comm.allreduce(vol_local, op=MPI.SUM)


def compute_scaffold_volume_exact(phi: fem.Function, domain) -> float:
    """
    EXACT scaffold volume via affine-invariant marching-tetrahedra
    (mechanics_fe._solid_fraction) -- no epsilon, no smoothing.

    compute_scaffold_volume()'s regularized Heaviside integral has a
    mesh-curvature-dependent bias: verified (via the debug_vn_integral /
    check_conservation diagnostics) to OVER-estimate volume LOSS by ~2.5x on
    a rough/high-A-over-V mesh and UNDER-estimate it by ~2.35x on a clean,
    geometry-validated one -- opposite-signed errors that made mass-loss
    numbers from the two mesh families non-comparable. This is exact (to
    floating-point precision) regardless of mesh quality, so it should be
    the primary reported volume/mass-loss metric; the smoothed version above
    is kept for the surface-area/other integrals that still need epsilon.
    """
    from mechanics_fe import _solid_fraction

    V = phi.function_space
    cellnodes = V.dofmap.list
    phi_v = phi.x.array[cellnodes]
    rho = _solid_fraction(phi_v)
    coords = domain.geometry.x[cellnodes]
    tet_vol = np.abs(np.einsum('ij,ij->i', coords[:, 1] - coords[:, 0],
                                np.cross(coords[:, 2] - coords[:, 0],
                                         coords[:, 3] - coords[:, 0]))) / 6.0
    vol_local = float((rho * tet_vol).sum())
    return domain.comm.allreduce(vol_local, op=MPI.SUM)


def compute_surface_area(phi: fem.Function, domain, epsilon: float = None) -> float:
    """
    Compute the interface area using a regularized delta function:
    A ≈ ∫ δ_ε(φ) |∇φ| dΩ
    """
    if epsilon is None:
        epsilon = 2.0 * _compute_h_min(domain)

    dx = ufl.Measure("dx", domain=domain)
    grad_phi = ufl.grad(phi)
    grad_mag = ufl.sqrt(ufl.inner(grad_phi, grad_phi) + 1e-12)

    # Regularized delta: δ_ε(φ) = (1/(2ε)) * (1 + cos(πφ/ε)) for |φ| < ε
    delta = ufl.conditional(
        ufl.lt(ufl.algebra.Abs(phi), epsilon),
        (1.0 / (2.0 * epsilon)) * (1.0 + ufl.cos(ufl.pi * phi / epsilon)),
        0.0
    )

    area_form = fem.form(delta * grad_mag * dx)
    area_local = fem.assemble_scalar(area_form)
    return domain.comm.allreduce(area_local, op=MPI.SUM)
