#!/usr/bin/env python3
"""mesh_generator_stent_adaptive.py — graded unstructured Delaunay stent mesh (gmsh).

mesh_generator_stent.py builds a uniform structured background grid, splitting
each cube into 6 tets that all share one long diagonal edge — fine for a quick
laptop test, but those slivers stay badly-shaped at any resolution (refining h
just makes smaller slivers). The real FreeFem stent mesh was mmg3d-generated
(proper unstructured, graded tets), which is what actually needs testing here.

This script keeps the same analytic SDF/geometry (paper Section 2.5 six-crown
sinusoidal stent) but tetrahedralizes the background box with gmsh's HXT
Delaunay algorithm under a graded sizing field: small element size near the
metal/medium interface (band), coarsening smoothly with distance. That gives
well-shaped, locally-refined tets — the actual variable under test — while
staying small enough (target ~1-3M tets) to run on a 16 GB laptop, unlike the
true mmg3d mesh (21M tets, needs ~120 GB).

Usage:
    python3 mesh_generator_stent_adaptive.py                       # defaults
    python3 mesh_generator_stent_adaptive.py --h_min 0.015 --h_max 0.20
    python3 mesh_generator_stent_adaptive.py --output mesh_stent_adaptive
Then:
    python3 dissolve.py --input_mesh mesh_stent_adaptive.xdmf --enable_redistance 0 --o2_initial 4.5e-9
"""
import argparse
import os
import numpy as np

# ── Stent geometry (paper Section 2.5 / FreeFem .edp) — identical to
#    mesh_generator_stent.py ────────────────────────────────────────────────
DOUT      = 3.20
ROUT      = DOUT / 2.0
THICKNESS = 0.15
RIN       = ROUT - THICKNESS
WAXIAL    = 0.15
AMP       = 1.00
NCROWN    = 6
FILLET    = 0.02
CENTERR   = (ROUT + RIN) / 2.0
HALFT     = THICKNESS / 2.0 - FILLET
HALFW     = WAXIAL / 2.0 - FILLET

LX, LY, LZ = 3.6, 3.6, 2.4


def sdf_stent(x, y, z):
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    d_tube = np.abs(r - CENTERR) - HALFT
    cz = AMP * np.sin(NCROWN * theta)
    d_band = np.abs(z - cz) - HALFW
    return np.maximum(d_tube, d_band) - FILLET


def build_sizing_view(gmsh, h_min, h_max, band, bg_n):
    """Sample |sdf| on a coarse background grid, graded h_min->h_max over
    `band` mm of distance from the interface; register as a gmsh PostView
    scalar-point field (no temp files needed)."""
    nx = ny = bg_n
    nz = max(4, int(round(bg_n * LZ / LX)))
    xs = np.linspace(-LX / 2, LX / 2, nx)
    ys = np.linspace(-LY / 2, LY / 2, ny)
    zs = np.linspace(-LZ / 2, LZ / 2, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    d = np.abs(sdf_stent(X, Y, Z))
    size = h_min + (h_max - h_min) * np.clip(d / band, 0.0, 1.0)

    n = X.size
    data = np.empty(4 * n, dtype=np.float64)
    data[0::4] = X.ravel()
    data[1::4] = Y.ravel()
    data[2::4] = Z.ravel()
    data[3::4] = size.ravel()

    view = gmsh.view.add("sizing")
    gmsh.view.addListData(view, "SP", n, data.tolist())
    print(f"  background sizing grid: {nx} x {ny} x {nz} = {n:,} points")
    return view


def build_mesh(h_min, h_max, band, bg_n, mesh_path=None):
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("stent_adaptive")
    gmsh.model.occ.addBox(-LX / 2, -LY / 2, -LZ / 2, LX, LY, LZ, 1)
    gmsh.model.occ.synchronize()

    view = build_sizing_view(gmsh, h_min, h_max, band, bg_n)

    field = gmsh.model.mesh.field.add("PostView")
    gmsh.model.mesh.field.setNumber(field, "ViewTag", view)
    gmsh.model.mesh.field.setAsBackgroundMesh(field)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)   # HXT: fast, robust Delaunay
    gmsh.option.setNumber("General.NumThreads", 8)

    print("  meshing (gmsh HXT Delaunay, this can take a few minutes)...")
    gmsh.model.mesh.generate(3)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    points = node_coords.reshape(-1, 3)
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=3)
    tet_idx = int(np.where(np.asarray(elem_types) == 4)[0][0])   # gmsh type 4 = 4-node tetrahedron
    raw = elem_node_tags[tet_idx].reshape(-1, 4)
    cells = np.vectorize(tag_to_idx.get)(raw).astype(np.int64)

    if mesh_path:
        gmsh.write(mesh_path)

    gmsh.finalize()
    return points, cells


def main():
    p = argparse.ArgumentParser(description="Graded unstructured Delaunay stent mesh (gmsh)")
    p.add_argument("--h_min", type=float, default=0.02,
                   help="finest element size near the interface (mm); default 0.02 (~7.5 cells/strut)")
    p.add_argument("--h_max", type=float, default=0.20,
                   help="coarsest element size far from the interface (mm); default 0.20")
    p.add_argument("--band", type=float, default=0.30,
                   help="distance (mm) over which size grades from h_min to h_max; default 0.30")
    p.add_argument("--bg_n", type=int, default=90,
                   help="background sizing-grid resolution along X/Y; default 90")
    p.add_argument("--output", type=str, default="mesh_stent_adaptive",
                   help="output base name (writes .xdmf + .h5)")
    p.add_argument("--write_mesh", action="store_true",
                   help="also write Medit .mesh (for a downstream mmg3d -ls pass)")
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(here, args.output)
    xdmf_path = base + ".xdmf"
    h5_path = base + ".h5"
    medit_path = base + ".mesh" if args.write_mesh else None

    print("=" * 62)
    print("  Graded unstructured stent mesh (gmsh HXT Delaunay)")
    print(f"  h_min={args.h_min}  h_max={args.h_max}  band={args.band}  bg_n={args.bg_n}")
    print("=" * 62)

    points, cells = build_mesh(args.h_min, args.h_max, args.band, args.bg_n, mesh_path=medit_path)

    cent = points[cells].mean(axis=1)
    d = sdf_stent(cent[:, 0], cent[:, 1], cent[:, 2])
    cell_data = np.where(d <= 0.0, 1, 2).astype(np.int32)
    n_metal = int((cell_data == 1).sum())
    n_med = int((cell_data == 2).sum())

    from mesh_utils import _write_xdmf
    _write_xdmf(xdmf_path, h5_path, points, cells, cell_data, args.output)

    metal_vol = np.pi * (ROUT**2 - RIN**2) * WAXIAL
    print(f"  vertices        : {len(points):,}")
    print(f"  tetrahedra      : {len(cells):,}")
    print(f"  metal tets (r=1): {n_metal:,}   medium tets (r=2): {n_med:,}")
    print(f"  metal fraction  : {100.0 * n_metal / len(cells):.2f} %")
    print(f"  analytical metal vol ref: {metal_vol:.4f} mm^3")
    print(f"  wrote {xdmf_path}\n        {h5_path}")
    if n_metal == 0:
        print("  !! WARNING: no metal cells captured — decrease --h_min/--band")
    print("\n  Run (redistancing OFF is required for thin struts):")
    print(f"    python3 dissolve.py --config simulation.toml "
          f"--input_mesh {args.output}.xdmf --enable_redistance 0 --o2_initial 4.5e-9")


if __name__ == "__main__":
    main()
