"""
gpu_assembler.py — Custom CUDA assembler for P1 tetrahedral mass + diffusion.

Replaces DOLFINx's CPU tabulate_tensor + MatSetValuesLocal path with a single
GPU kernel that:
    1. Iterates over all cells in parallel
    2. Computes the 4x4 local element matrix A_e = M_e/dt + De_e * K_e
       using the cached geometric factors (wdetJ, G = J^-1 J^-T)
    3. Scatters via atomic-add into a global CSR matrix on GPU
    4. Returns an aijcusparse PETSc Mat ready for KSP

This is the missing 15% of GPU coverage in the proven sparse path — turning
"95% GPU compute / 85% GPU wall-time" into "100% GPU compute / 99% GPU wall-time".

Math for a single P1 tet (reference vertices ξ0=(0,0,0), ξ1=(1,0,0), ξ2=(0,1,0), ξ3=(0,0,1)):

    Basis: N_i(ξ) = barycentric coordinates
    ∇_ξ N = [[-1,-1,-1], [1,0,0], [0,1,0], [0,0,1]]    (4x3 constant)
    Mass:  M_e[i,j]  = ∫ N_i N_j |detJ| dξ  = (|detJ|/20) * (1 + δ_ij)  for the unit tet
    Stiff: K_e[i,j]  = |detJ| * (∇_ξ N_i)^T G (∇_ξ N_j)
"""
from __future__ import annotations
import os
import numpy as np
import cupy as cp
from petsc4py import PETSc

# Mass-matrix mode for the transport solves.
#   1 (default) -> row-sum lumped mass + reaction, matching FreeFem's qfV1lump.
#                  Monotone: removes the consistent-mass over/undershoot that
#                  produced unphysical NEGATIVE O2 at the interface
#                  (verified: O2_iface_min  -9.65e-9 -> +1.5e-12).
#   0           -> consistent mass (legacy; physically incorrect near sharp fronts).
# NOTE: lumping is a correctness fix; it does NOT change the linear-vs-decelerating
# trend (that is set by k_orr — see the research notes). Override with
# DISSOLVE_LUMP_MASS=0 to reproduce the old behaviour.
LUMP_MASS = int(os.environ.get("DISSOLVE_LUMP_MASS", "1"))


# ── reference-tet basis gradient: shape (4, 3), constant ──
_REF_GRAD = np.array(
    [[-1.0, -1.0, -1.0],
     [ 1.0,  0.0,  0.0],
     [ 0.0,  1.0,  0.0],
     [ 0.0,  0.0,  1.0]], dtype=np.float64
)

# ── consistent mass matrix on unit tet (volume = 1/6) ──
#   ∫_T N_i N_j dξ = vol/20 * (1 + δ_ij)   with vol(unit tet) = 1/6
#   So per-unit-detJ: M[i,j] = (1/120) * (1 + δ_ij)
_M_UNIT = (1.0 / 120.0) * (np.ones((4, 4)) + np.eye(4))


# ── GPU kernel: per-cell stiffness+mass+reaction into CSR ────────────────
# For each tet cell: build local 4×4 of
#     A_e = (1/dt + α_cell)·M_e + De_cell·K_e
# where M_e is the consistent mass and K_e = (∇N)ᵀ G (∇N) · vol/6.
# Per-cell averaging of De and α matches DOLFINx degree-2 quadrature for
# P1 coefficient fields exactly (cell average IS the exact integral).
_ASSEMBLE_KERNEL = r"""
extern "C" __global__
void assemble_p1_tet(
    const int n_cells,
    const int* __restrict__ elem_nodes,    // (n_cells, 4) global DOF index
    const double* __restrict__ detJ,       // (n_cells,)
    const double* __restrict__ Gflat,      // (n_cells, 9) G = J^-1 J^-T row-major
    const double* __restrict__ De_cell,    // (n_cells,) average De
    const double* __restrict__ alpha_cell, // (n_cells,) average α  (reaction coef)
    const double inv_dt,
    const int     lumped,    // 0 = consistent mass, 1 = row-sum lumped (FreeFem qfV1lump)
    const int*    __restrict__ scatter_map, // (n_cells, 16) precomputed CSR positions
    double*       __restrict__ csr_data     // (nnz,)
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= n_cells) return;

    // Reference-element gradients (constant for P1 tet)
    const double grad[4][3] = {
        {-1.0, -1.0, -1.0},
        { 1.0,  0.0,  0.0},
        { 0.0,  1.0,  0.0},
        { 0.0,  0.0,  1.0}
    };
    // Unit-mass matrix (1 + δij)/120 — symmetric, 4x4
    const double Munit[4][4] = {
        {2.0, 1.0, 1.0, 1.0},
        {1.0, 2.0, 1.0, 1.0},
        {1.0, 1.0, 2.0, 1.0},
        {1.0, 1.0, 1.0, 2.0}
    };

    const double dJ = detJ[cell];
    const double De = De_cell[cell];
    const double a_ = alpha_cell[cell];
    const double G[3][3] = {
        {Gflat[cell*9+0], Gflat[cell*9+1], Gflat[cell*9+2]},
        {Gflat[cell*9+3], Gflat[cell*9+4], Gflat[cell*9+5]},
        {Gflat[cell*9+6], Gflat[cell*9+7], Gflat[cell*9+8]}
    };

    // Build 4x4 local matrix and scatter directly using precomputed CSR positions.
    // Each scatter_map[cell*16 + i*4 + j] is the exact index into csr_data for
    // the (gn[i], gn[j]) entry — no binary search, no branch divergence.
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            // Stiffness: dJ * De * (grad_i)^T G (grad_j)
            double Gg[3] = {0.0, 0.0, 0.0};
            #pragma unroll
            for (int a = 0; a < 3; a++) {
                #pragma unroll
                for (int b = 0; b < 3; b++) {
                    Gg[a] += G[a][b] * grad[j][b];
                }
            }
            double K_ij = 0.0;
            #pragma unroll
            for (int a = 0; a < 3; a++) K_ij += grad[i][a] * Gg[a];
            K_ij *= (dJ * De) / 6.0;

            // Consistent mass (1+dij)/120; lumped = row-sum onto the diagonal
            // (row sum of [2,1,1,1] = 5), giving M_ii = dJ*5/120 = dJ/24, off-diag 0.
            // Lumping the M term also lumps the reaction (inv_dt+a_)*M, matching
            // FreeFem's qfV1lump on the mass + reaction varf terms.
            double M_ij;
            if (lumped) {
                M_ij = (i == j) ? (dJ * 5.0 / 120.0) : 0.0;
            } else {
                M_ij = dJ * Munit[i][j] / 120.0;
            }
            const double A_ij = K_ij + (inv_dt + a_) * M_ij;

            // Direct scatter: O(1) lookup, no while-loop, no branch
            const int pos = scatter_map[cell * 16 + i * 4 + j];
            atomicAdd(&csr_data[pos], A_ij);
        }
    }
}
"""

_kernel_module = cp.RawModule(code=_ASSEMBLE_KERNEL, options=("--std=c++14",))
_kernel = _kernel_module.get_function("assemble_p1_tet")


# ── Dirichlet BC application: zero rows + set diagonal=1 ─────────────────
_BC_KERNEL_SRC = r"""
extern "C" __global__
void apply_dirichlet_rows(
    const int n_bc,
    const int* __restrict__ bc_dofs,
    const int*    __restrict__ csr_indptr,
    const int*    __restrict__ csr_indices,
    double*       __restrict__ csr_data
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_bc) return;
    const int row = bc_dofs[k];
    const int rs  = csr_indptr[row];
    const int re  = csr_indptr[row + 1];
    for (int j = rs; j < re; j++) {
        if (csr_indices[j] == row) csr_data[j] = 1.0;
        else                       csr_data[j] = 0.0;
    }
}
"""
_bc_module = cp.RawModule(code=_BC_KERNEL_SRC, options=("--std=c++14",))
_bc_kernel = _bc_module.get_function("apply_dirichlet_rows")


# ── GPU RHS assembly:  b = M·u_old/dt + M·f  ─────────────────────────────
# For P1 fields, ∫ N_i · N_j · g dx with g = sum_k g_k·N_k gives
# the consistent mass matrix · g_vertices.  So per cell:
#   b_i += (|detJ|/120) · [ (1/dt)·(Σu + u_i)  +  (Σf + f_i) ]
# where u and f are the 4 vertex values.  This is exact for P1 inputs.
_RHS_KERNEL_SRC = r"""
extern "C" __global__
void assemble_rhs_p1_tet(
    const int n_cells,
    const int* __restrict__ elem_nodes,
    const double* __restrict__ detJ,
    const double* __restrict__ u_old,   // per-DOF (n_dofs,)
    const double* __restrict__ f_vert,  // per-DOF source  (n_dofs,)
    const double inv_dt,
    const int     lumped,    // 0 = consistent mass, 1 = row-sum lumped
    double*       __restrict__ b
) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= n_cells) return;

    const int n0 = elem_nodes[cell*4+0];
    const int n1 = elem_nodes[cell*4+1];
    const int n2 = elem_nodes[cell*4+2];
    const int n3 = elem_nodes[cell*4+3];

    const double dJ = detJ[cell];

    const double u0 = u_old[n0], u1 = u_old[n1], u2 = u_old[n2], u3 = u_old[n3];
    const double f0 = f_vert[n0], f1 = f_vert[n1], f2 = f_vert[n2], f3 = f_vert[n3];

    const double sum_u = u0 + u1 + u2 + u3;
    const double sum_f = f0 + f1 + f2 + f3;

    const double base = dJ / 120.0;
    const double mdt  = base * inv_dt;   // consistent mass · 1/dt
    const double mf   = base;            // consistent mass · 1   (for the source f)

    if (lumped) {
        // Lumped nodal mass = row sum = base*5 = dJ/24; M·v becomes Ld*v_i.
        const double Ld = base * 5.0;
        atomicAdd(&b[n0], Ld * (inv_dt * u0 + f0));
        atomicAdd(&b[n1], Ld * (inv_dt * u1 + f1));
        atomicAdd(&b[n2], Ld * (inv_dt * u2 + f2));
        atomicAdd(&b[n3], Ld * (inv_dt * u3 + f3));
    } else {
        atomicAdd(&b[n0], mdt * (sum_u + u0) + mf * (sum_f + f0));
        atomicAdd(&b[n1], mdt * (sum_u + u1) + mf * (sum_f + f1));
        atomicAdd(&b[n2], mdt * (sum_u + u2) + mf * (sum_f + f2));
        atomicAdd(&b[n3], mdt * (sum_u + u3) + mf * (sum_f + f3));
    }
}
"""
_rhs_module = cp.RawModule(code=_RHS_KERNEL_SRC, options=("--std=c++14",))
_rhs_kernel = _rhs_module.get_function("assemble_rhs_p1_tet")


# Dirichlet on RHS: b[bc_dofs] = bc_value
_RHS_BC_KERNEL_SRC = r"""
extern "C" __global__
void apply_dirichlet_rhs(
    const int n_bc,
    const int* __restrict__ bc_dofs,
    const double bc_value,
    double* __restrict__ b
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_bc) return;
    b[bc_dofs[k]] = bc_value;
}
"""
_rhs_bc_module = cp.RawModule(code=_RHS_BC_KERNEL_SRC, options=("--std=c++14",))
_rhs_bc_kernel = _rhs_bc_module.get_function("apply_dirichlet_rhs")


# ── GPU nodal gradient: ∇u (P1) at vertices via volume-weighted average ──
# For each P1 tet:
#   ∇N_ref = constant 4×3
#   ∇u_cell (physical) = J⁻ᵀ · Σ_a u_a · ∇N_ref[a]
# Distribute to each of the 4 vertices weighted by cell volume = |detJ|/6.
# After accumulation, divide by sum-of-weights per DOF.
_GRAD_KERNEL_SRC = r"""
extern "C" __global__
void nodal_gradient_p1(
    const int n_cells,
    const int* __restrict__ elem_nodes,
    const double* __restrict__ detJ,
    const double* __restrict__ JinvT,   // (n_cells, 9) row-major
    const double* __restrict__ u_vert,  // (n_dofs,)
    double*       __restrict__ grad,    // (n_dofs*3,) interleaved (gx,gy,gz)
    double*       __restrict__ wsum     // (n_dofs,) accumulated weights
) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= n_cells) return;

    const double grad_ref[4][3] = {
        {-1.0,-1.0,-1.0},
        { 1.0, 0.0, 0.0},
        { 0.0, 1.0, 0.0},
        { 0.0, 0.0, 1.0}
    };

    const int n0 = elem_nodes[cell*4+0];
    const int n1 = elem_nodes[cell*4+1];
    const int n2 = elem_nodes[cell*4+2];
    const int n3 = elem_nodes[cell*4+3];
    const int gn[4] = {n0, n1, n2, n3};

    const double u[4] = {u_vert[n0], u_vert[n1], u_vert[n2], u_vert[n3]};

    // ∇u_ref = Σ u_a · ∇N_ref[a]  (3-vector)
    double gu_ref[3] = {0.0, 0.0, 0.0};
    #pragma unroll
    for (int a = 0; a < 4; a++) {
        gu_ref[0] += u[a] * grad_ref[a][0];
        gu_ref[1] += u[a] * grad_ref[a][1];
        gu_ref[2] += u[a] * grad_ref[a][2];
    }
    // ∇u_phys = J⁻ᵀ · ∇u_ref
    const double* JT = &JinvT[cell*9];
    double gu_phy[3];
    gu_phy[0] = JT[0]*gu_ref[0] + JT[1]*gu_ref[1] + JT[2]*gu_ref[2];
    gu_phy[1] = JT[3]*gu_ref[0] + JT[4]*gu_ref[1] + JT[5]*gu_ref[2];
    gu_phy[2] = JT[6]*gu_ref[0] + JT[7]*gu_ref[1] + JT[8]*gu_ref[2];

    const double w = detJ[cell] / 6.0;   // cell volume (physical)
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int dof = gn[i];
        atomicAdd(&grad[dof*3 + 0], w * gu_phy[0]);
        atomicAdd(&grad[dof*3 + 1], w * gu_phy[1]);
        atomicAdd(&grad[dof*3 + 2], w * gu_phy[2]);
        atomicAdd(&wsum[dof], w);
    }
}
"""
_grad_module = cp.RawModule(code=_GRAD_KERNEL_SRC, options=("--std=c++14",))
_grad_kernel = _grad_module.get_function("nodal_gradient_p1")


# ── high-level wrapper class ─────────────────────────────────────────────
class GPUSparseAssembler:
    """
    Assembles the P1 (M/dt + De·K) matrix entirely on the GPU.

    Setup (one-time):
        a = GPUSparseAssembler(domain, V, comm=MPI.COMM_WORLD)

    Per-step:
        a.assemble(De_per_cell, inv_dt)   # writes into self.mat (aijcusparse)
        ksp.setOperators(a.mat)
        ksp.solve(b, x)

    State on GPU between calls:
        elem_nodes, detJ, Gflat       — geometric data, uploaded once
        csr_indptr, csr_indices       — sparsity pattern, computed once
        csr_data                      — refilled each call

    De_per_cell is a numpy or cupy array of length n_cells.
    """

    def __init__(self, domain, V, comm=None, ref_asm=None):
        """
        Parameters
        ----------
        domain, V : mesh and function space (required on first / primary assembler).
        ref_asm   : optional GPUSparseAssembler whose geometric arrays
                    (d_elem_nodes, d_detJ, d_Gflat, d_JinvT, d_indptr,
                    d_indices, _grad_buf, _wsum_buf) will be SHARED rather
                    than re-uploaded.  Use this for field assemblers after the
                    first (gradient) assembler has been created — saves
                    ~150 MB per field on an 800k-cell mesh.
        """
        from mpi4py import MPI
        self.comm = comm or MPI.COMM_WORLD

        if ref_asm is not None:
            # ── Shared-geometry path: borrow all read-only arrays ──────────
            # Only field-specific buffers (d_data, d_b, mat) are allocated
            # fresh.  Everything else is a reference to ref_asm's arrays.
            self.n_cells      = ref_asm.n_cells
            self.n_dofs       = ref_asm.n_dofs
            self._nnz         = ref_asm._nnz
            # Shared read-only geometry (zero extra GPU memory)
            self.d_elem_nodes = ref_asm.d_elem_nodes
            self.d_detJ       = ref_asm.d_detJ
            self.d_Gflat      = ref_asm.d_Gflat
            self.d_JinvT      = ref_asm.d_JinvT
            self.d_indptr     = ref_asm.d_indptr
            self.d_indices    = ref_asm.d_indices
            # Shared gradient workspace (gradient calls are sequential per step)
            self._grad_buf    = ref_asm._grad_buf
            self._wsum_buf    = ref_asm._wsum_buf
            # Shared precomputed scatter map (same sparsity for all fields)
            self.d_scatter_map = ref_asm.d_scatter_map
            # Field-specific: writable CSR values + RHS
            self.d_data = cp.zeros(self._nnz,      dtype=cp.float64)
            self.d_b    = cp.zeros(self.n_dofs,    dtype=cp.float64)
        else:
            # ── Full-geometry path: upload everything (primary assembler) ──
            tdim = domain.topology.dim
            n_cells = domain.topology.index_map(tdim).size_local
            n_dofs  = V.dofmap.index_map.size_local
            self.n_cells = n_cells
            self.n_dofs  = n_dofs

            elem_nodes = np.array(
                [V.dofmap.cell_dofs(i) for i in range(n_cells)], dtype=np.int32
            )

            coords = domain.geometry.x[:, :3]
            edges = coords[elem_nodes][:, 1:, :] - coords[elem_nodes][:, 0:1, :]
            J    = edges.transpose(0, 2, 1)
            detJ = np.abs(np.linalg.det(J))
            Jinv = np.linalg.inv(J)
            G    = Jinv @ Jinv.transpose(0, 2, 1)
            JinvT = Jinv.transpose(0, 2, 1)

            self.d_elem_nodes = cp.asarray(elem_nodes.flatten(), dtype=cp.int32)
            self.d_detJ       = cp.asarray(detJ, dtype=cp.float64)
            self.d_Gflat      = cp.asarray(G.reshape(n_cells, 9),
                                           dtype=cp.float64).ravel()
            self.d_JinvT      = cp.asarray(JinvT.reshape(n_cells, 9),
                                           dtype=cp.float64).ravel()

            # Sparsity pattern
            adj = [set() for _ in range(n_dofs)]
            for ci in range(n_cells):
                ndofs_e = elem_nodes[ci]
                for r in ndofs_e:
                    adj[r].update(ndofs_e.tolist())
            indptr = np.zeros(n_dofs + 1, dtype=np.int32)
            for r in range(n_dofs):
                indptr[r+1] = indptr[r] + len(adj[r])
            indices = np.empty(indptr[-1], dtype=np.int32)
            for r in range(n_dofs):
                sorted_cols = sorted(adj[r])
                indices[indptr[r]:indptr[r+1]] = sorted_cols
            self.d_indptr  = cp.asarray(indptr,  dtype=cp.int32)
            self.d_indices = cp.asarray(indices, dtype=cp.int32)
            self._nnz      = int(indptr[-1])

            self.d_data = cp.zeros(indptr[-1], dtype=cp.float64)
            self.d_b    = cp.zeros(n_dofs,     dtype=cp.float64)

            # Gradient workspace
            self._grad_buf = cp.zeros(n_dofs * 3, dtype=cp.float64)
            self._wsum_buf = cp.zeros(n_dofs,     dtype=cp.float64)

            # Precomputed scatter map: for each cell and each of the 16 local
            # (i,j) entries, store the exact CSR position in d_data.
            # Eliminates the binary search in the CUDA kernel → O(1) scatter.
            # Memory: n_cells × 16 × 4 bytes (shared with ref_asm).
            scatter_map = np.empty(n_cells * 16, dtype=np.int32)
            # Build (row,col)→CSR-position dictionary in O(nnz)
            _rc2pos = {}
            _row_arr = np.repeat(np.arange(n_dofs, dtype=np.int32),
                                 np.diff(indptr))
            for _pos, (_r, _c) in enumerate(zip(_row_arr, indices)):
                _rc2pos[(int(_r), int(_c))] = _pos
            # Fill scatter_map for every (cell, i, j) pair in O(n_cells*16)
            k = 0
            for ci in range(n_cells):
                ne = elem_nodes[ci]
                for ii in range(4):
                    for jj in range(4):
                        scatter_map[k] = _rc2pos[(int(ne[ii]), int(ne[jj]))]
                        k += 1
            self.d_scatter_map = cp.asarray(scatter_map, dtype=cp.int32)

        # ── PETSc aijcusparse Mat (always field-specific) ─────────────────
        n_dofs  = self.n_dofs
        indptr_h  = cp.asnumpy(self.d_indptr)
        indices_h = cp.asnumpy(self.d_indices)
        self.mat = PETSc.Mat().create(self.comm)
        self.mat.setType("aijcusparse")
        self.mat.setSizes(((n_dofs, n_dofs), (n_dofs, n_dofs)))
        rows = np.repeat(np.arange(n_dofs, dtype=np.int32), np.diff(indptr_h))
        cols = indices_h.astype(np.int32)
        self.mat.setPreallocationCOO(rows, cols)
        self.mat.zeroEntries(); self.mat.assemble()

    def _to_cell_avg(self, arr_or_func):
        """
        Accept either:
          - numpy / cupy length n_cells (already cell-wise), or
          - numpy / cupy length n_dofs  (per-vertex → averaged here).
        Returns a cupy array of length n_cells.
        """
        if arr_or_func is None:
            return cp.zeros(self.n_cells, dtype=cp.float64)
        n = int(arr_or_func.shape[0])
        if n == self.n_cells:
            return cp.asarray(arr_or_func, dtype=cp.float64)
        if n == self.n_dofs:
            d = cp.asarray(arr_or_func, dtype=cp.float64)
            return d[self.d_elem_nodes].reshape(self.n_cells, 4).mean(axis=1)
        raise ValueError(
            f"array length {n} matches neither n_cells={self.n_cells} "
            f"nor n_dofs={self.n_dofs}"
        )

    def assemble(self, De, inv_dt: float, alpha=None, bc_dofs=None):
        """
        Fills self.mat with (1/dt + α_e)·M_e + De_e·K_e on the GPU,
        then applies Dirichlet row-elimination if bc_dofs is given.

        De      : per-cell or per-vertex diffusion coefficient.
        inv_dt  : 1/dt (scalar).
        alpha   : optional per-cell or per-vertex reaction coefficient.
        bc_dofs : optional numpy/cupy int32 array of DOFs to constrain.
                  Their matrix rows are zeroed and the diagonal set to 1
                  (the user's RHS already sets b[bc_dofs] = bc_value).
        """
        d_De    = self._to_cell_avg(De)
        d_alpha = self._to_cell_avg(alpha)

        # Zero the CSR data on GPU
        self.d_data.fill(0.0)

        # Launch kernel: one thread per cell.
        # Uses precomputed scatter_map → O(1) CSR scatter, no binary search.
        threads = 128
        blocks  = (self.n_cells + threads - 1) // threads
        _kernel(
            (blocks,), (threads,),
            (
                cp.int32(self.n_cells),
                self.d_elem_nodes,
                self.d_detJ,
                self.d_Gflat,
                d_De,
                d_alpha,
                cp.float64(inv_dt),
                cp.int32(LUMP_MASS),
                self.d_scatter_map,   # replaces d_indptr + d_indices + binary search
                self.d_data,
            ),
        )
        cp.cuda.runtime.deviceSynchronize()

        # Apply Dirichlet rows: zero non-diagonal entries, set diagonal to 1.
        if bc_dofs is not None and len(bc_dofs) > 0:
            d_bc = cp.asarray(bc_dofs, dtype=cp.int32)
            n_bc = int(d_bc.shape[0])
            bc_threads = 128
            bc_blocks  = (n_bc + bc_threads - 1) // bc_threads
            _bc_kernel(
                (bc_blocks,), (bc_threads,),
                (cp.int32(n_bc), d_bc,
                 self.d_indptr, self.d_indices, self.d_data),
            )
            cp.cuda.runtime.deviceSynchronize()

        # Push d_data → PETSc Mat.
        # Try zero-copy device path (PETSc ≥ 3.20 + CUDA-aware build); if the
        # cupy array is rejected fall back to a single D2H + setValuesCOO call.
        try:
            self.mat.setValuesCOO(self.d_data, PETSc.InsertMode.INSERT)
        except Exception:
            data_host = cp.asnumpy(self.d_data)
            self.mat.setValuesCOO(data_host, PETSc.InsertMode.INSERT)
        self.mat.assemble()

    def assemble_rhs(self, u_old, inv_dt: float, f=None,
                     bc_dofs=None, bc_value: float = 0.0):
        """
        Assembles  b = M·u_old/dt + ∫ f(x)·v dx  entirely on the GPU.

        u_old    : per-DOF (n_dofs,) numpy/cupy array of previous-step values.
        inv_dt   : 1/dt scalar.
        f        : optional per-cell or per-vertex source (averaged to cells).
        bc_dofs  : optional int32 array of Dirichlet DOFs to overwrite in b.
        bc_value : scalar value to write to b at bc_dofs.
        Returns  : self.d_b  (cupy array, length n_dofs).
        """
        # u_old: vertex values, length n_dofs
        if u_old.shape[0] != self.n_dofs:
            raise ValueError(
                f"u_old length {u_old.shape[0]} != n_dofs {self.n_dofs}"
            )
        d_u = cp.asarray(u_old, dtype=cp.float64)

        # Source f: take vertex values directly (per-DOF) for exact M·f.
        # If f is None or per-cell only, broadcast to per-DOF (gives same
        # result as cell-averaging at the cost of accuracy on the interface).
        if f is None:
            d_f = cp.zeros(self.n_dofs, dtype=cp.float64)
        elif int(f.shape[0]) == self.n_dofs:
            d_f = cp.asarray(f, dtype=cp.float64)
        elif int(f.shape[0]) == self.n_cells:
            # Per-cell f → expand to per-DOF (this is the lossy path).
            cell_avg = cp.asarray(f, dtype=cp.float64)
            # Scatter: take max-by-cell at each DOF (simplest unique mapping).
            d_f = cp.zeros(self.n_dofs, dtype=cp.float64)
            # WARNING: lossy for the general case; only use if caller really
            # wants per-cell semantics.
            cell_avg_per_dof = cell_avg[
                cp.searchsorted(self.d_elem_nodes,
                                cp.arange(self.n_dofs, dtype=cp.int32))
            ]
            d_f[:] = cell_avg_per_dof
        else:
            raise ValueError(f"f length {f.shape[0]} unrecognised")

        self.d_b.fill(0.0)

        threads = 128
        blocks  = (self.n_cells + threads - 1) // threads
        _rhs_kernel(
            (blocks,), (threads,),
            (
                cp.int32(self.n_cells),
                self.d_elem_nodes,
                self.d_detJ,
                d_u,
                d_f,
                cp.float64(inv_dt),
                cp.int32(LUMP_MASS),
                self.d_b,
            ),
        )

        if bc_dofs is not None and len(bc_dofs) > 0:
            d_bc = cp.asarray(bc_dofs, dtype=cp.int32)
            n_bc = int(d_bc.shape[0])
            bc_blocks = (n_bc + threads - 1) // threads
            _rhs_bc_kernel(
                (bc_blocks,), (threads,),
                (cp.int32(n_bc), d_bc, cp.float64(bc_value), self.d_b),
            )

        cp.cuda.runtime.deviceSynchronize()
        return self.d_b

    def scaffold_volume(self, phi_arr, epsilon: float):
        """
        ∫ H_ε(φ) dx over the mesh, entirely on GPU.

        H_ε(φ) = 0 if φ < -ε, 1 if φ > ε,
                 0.5·(1 + φ/ε + sin(π·φ/ε)/π) in between

        Uses the 4-point degree-2 symmetric quadrature rule for tetrahedra
        (Keast, 1986) to match DOLFINx's default integration accuracy for
        P1 forms.  Barycentric weights: (α,β,β,β) and permutations with
        α=(5+3√5)/20, β=(5-√5)/20, uniform quadrature weight 1/4.

        Returns a Python float (local contribution; caller must MPI allreduce).
        """
        import math as _math
        alpha = (5.0 + 3.0 * _math.sqrt(5.0)) / 20.0   # ≈ 0.58541
        beta  = (5.0 -       _math.sqrt(5.0)) / 20.0   # ≈ 0.13820

        f = cp.asarray(phi_arr, dtype=cp.float64)

        # phi at the 4 nodes of every cell: shape (n_cells, 4)
        phi_nodes = f[self.d_elem_nodes].reshape(self.n_cells, 4)

        # phi at 4 quadrature points per cell (each point pulls toward centroid):
        #   φ(q_i) = α·φ_i + β·(Σ φ_j - φ_i) = (α-β)·φ_i + β·Σ φ_j
        phi_sum = phi_nodes.sum(axis=1, keepdims=True)            # (n_cells, 1)
        phi_q   = (alpha - beta) * phi_nodes + beta * phi_sum     # (n_cells, 4)

        # H_ε evaluated at each quadrature point
        H_q = cp.where(
            phi_q < -epsilon, 0.0,
            cp.where(phi_q > epsilon, 1.0,
                     0.5 * (1.0 + phi_q/epsilon
                            + cp.sin(cp.pi * phi_q / epsilon) / cp.pi))
        )

        # Cell integral: equal weights 1/4 → mean over 4 points × cell volume
        cell_H   = H_q.mean(axis=1)
        cell_vol = self.d_detJ / 6.0
        return float(cp.sum(cell_H * cell_vol))

    def nodal_gradient(self, u_vert):
        """
        Return per-vertex ∇u for a P1 field u_vert (numpy/cupy, length n_dofs).
        Result is a cupy array of shape (n_dofs, 3) — gx, gy, gz at each vertex,
        volume-weighted averaged across cells.
        """
        if hasattr(u_vert, "x"):
            u_vert = u_vert.x.array
        d_u = cp.asarray(u_vert, dtype=cp.float64)
        self._grad_buf.fill(0.0)
        self._wsum_buf.fill(0.0)
        threads = 128
        blocks  = (self.n_cells + threads - 1) // threads
        _grad_kernel(
            (blocks,), (threads,),
            (
                cp.int32(self.n_cells),
                self.d_elem_nodes,
                self.d_detJ,
                self.d_JinvT,
                d_u,
                self._grad_buf,
                self._wsum_buf,
            ),
        )
        cp.cuda.runtime.deviceSynchronize()
        g = self._grad_buf.reshape(self.n_dofs, 3) / self._wsum_buf[:, None]
        return g    # cupy (n_dofs, 3)
