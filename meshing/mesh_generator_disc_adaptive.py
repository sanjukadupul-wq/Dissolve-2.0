#!/usr/bin/env python3
"""mesh_generator_disc_adaptive.py — graded unstructured Delaunay disc mesh (gmsh).

Same pipeline as mesh_generator_stent_adaptive.py (graded gmsh HXT background
mesh -> feeds run_mmg_adapt.py's mmg3d -ls pass) but for the disc/cylinder
validation geometry, matching mesh_adaptive_800k.xdmf's domain exactly so the
recalibration is apples-to-apples with the paper's Table 3 targets:

  metal (Zn disc) : cylinder, radius R=5.1 mm, half-height H=1.11 mm
                    (measured directly from mesh_adaptive_800k.h5's metal
                    sub-region bounding box)
  domain box      : 30 x 30 x 20 mm  (x,y in [-15,15], z in [-10,10])

Usage:
    python3 mesh_generator_disc_adaptive.py --write_mesh --output mesh_disc_init --h_min 0.08 --h_max 1.5
Then:
    python3 run_mmg_adapt.py --input mesh_disc_init.mesh --output mesh_disc_mmg \
        --sdf disc --hmin 0.04 --hmax 1.5 --hausd 0.02
"""
import argparse
import os
import numpy as np

# ── Disc geometry (true design dims: 10mm diameter x 2mm thick coupon) ──────
# mesh_adaptive_800k's own metal VOLUME (157.07mm^3) matches this exactly
# (analytical pi*5^2*2 = 157.08mm^3); its surface area (351.14mm^2) does not
# (true value 219.91mm^2) -- confirms 5/2 as the true geometry and pins the
# excess entirely on that mesh's staircase interface, not a volume error.
R_DISC = 5.0      # mm, cylinder radius
H_DISC = 1.0      # mm, cylinder half-height (full thickness = 2*H_DISC = 2.0mm)
FILLET = 0.15     # mm, edge fillet for gradient-recovery robustness. Was 0.05mm,
                  # but that's too tight relative to hmin=0.05mm (~1 element
                  # across the curve) for mmg3d's -hausd criterion to force
                  # real refinement there, leaving a visibly faceted rim even
                  # though the flat top/bottom/side surfaces come out smooth.
                  # The fillet is a numerical convenience, not a modeled
                  # physical feature, so widening it (rather than tightening
                  # hausd/hmin mesh-wide) is the cheap fix.

LX, LY, LZ = 30.0, 30.0, 20.0   # matches mesh_adaptive_800k.h5 bbox exactly


def sdf_disc(x, y, z):
    """Signed distance to a filleted cylinder (< 0 inside metal).

    Radius/half-height are pre-shrunk by FILLET before the rounded-corner
    SDF, then FILLET is added back at the end (round box recipe) — this
    keeps the overall R_DISC/H_DISC extent exact and only rounds the sharp
    edge, instead of inflating the whole disc by FILLET everywhere (the
    earlier bug: applying "- FILLET" without first shrinking left the flat
    faces at R_DISC+FILLET / H_DISC+FILLET, not R_DISC / H_DISC).
    """
    r = np.sqrt(x * x + y * y)
    d_rad = r - (R_DISC - FILLET)
    d_ax = np.abs(z) - (H_DISC - FILLET)
    dr = np.maximum(d_rad, 0.0)
    da = np.maximum(d_ax, 0.0)
    outside = np.sqrt(dr * dr + da * da)
    inside = np.minimum(np.maximum(d_rad, d_ax), 0.0)
    return outside + inside - FILLET


def build_sizing_view(gmsh, h_min, h_max, band, bg_n):
    nx = ny = bg_n
    nz = max(6, int(round(bg_n * LZ / LX)))
    xs = np.linspace(-LX / 2, LX / 2, nx)
    ys = np.linspace(-LY / 2, LY / 2, ny)
    zs = np.linspace(-LZ / 2, LZ / 2, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    d = np.abs(sdf_disc(X, Y, Z))
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
    gmsh.model.add("disc_adaptive")
    gmsh.model.occ.addBox(-LX / 2, -LY / 2, -LZ / 2, LX, LY, LZ, 1)
    gmsh.model.occ.synchronize()

    view = build_sizing_view(gmsh, h_min, h_max, band, bg_n)

    field = gmsh.model.mesh.field.add("PostView")
    gmsh.model.mesh.field.setNumber(field, "ViewTag", view)
    gmsh.model.mesh.field.setAsBackgroundMesh(field)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    gmsh.option.setNumber("General.NumThreads", 8)

    print("  meshing (gmsh HXT Delaunay)...")
    gmsh.model.mesh.generate(3)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    points = node_coords.reshape(-1, 3)
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=3)
    tet_idx = int(np.where(np.asarray(elem_types) == 4)[0][0])
    raw = elem_node_tags[tet_idx].reshape(-1, 4)
    cells = np.vectorize(tag_to_idx.get)(raw).astype(np.int64)

    if mesh_path:
        gmsh.write(mesh_path)

    gmsh.finalize()
    return points, cells


def main():
    p = argparse.ArgumentParser(description="Graded unstructured Delaunay disc mesh (gmsh)")
    p.add_argument("--h_min", type=float, default=0.08,
                   help="finest element size near the interface (mm); default 0.08")
    p.add_argument("--h_max", type=float, default=2.0,
                   help="coarsest element size far from the interface (mm); default 2.0")
    p.add_argument("--band", type=float, default=3.0,
                   help="distance (mm) over which size grades from h_min to h_max; default 3.0")
    p.add_argument("--bg_n", type=int, default=90,
                   help="background sizing-grid resolution along X/Y; default 90")
    p.add_argument("--output", type=str, default="mesh_disc_adaptive",
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
    print("  Graded unstructured disc mesh (gmsh HXT Delaunay)")
    print(f"  R={R_DISC}  H={H_DISC}  h_min={args.h_min}  h_max={args.h_max}  "
          f"band={args.band}  bg_n={args.bg_n}")
    print("=" * 62)

    points, cells = build_mesh(args.h_min, args.h_max, args.band, args.bg_n, mesh_path=medit_path)

    cent = points[cells].mean(axis=1)
    d = sdf_disc(cent[:, 0], cent[:, 1], cent[:, 2])
    cell_data = np.where(d <= 0.0, 1, 2).astype(np.int32)
    n_metal = int((cell_data == 1).sum())
    n_med = int((cell_data == 2).sum())

    from mesh_utils import _write_xdmf
    _write_xdmf(xdmf_path, h5_path, points, cells, cell_data, args.output)

    metal_vol = np.pi * R_DISC**2 * (2 * H_DISC)
    print(f"  vertices        : {len(points):,}")
    print(f"  tetrahedra      : {len(cells):,}")
    print(f"  metal tets (r=1): {n_metal:,}   medium tets (r=2): {n_med:,}")
    print(f"  metal fraction  : {100.0 * n_metal / len(cells):.2f} %")
    print(f"  analytical metal vol ref: {metal_vol:.4f} mm^3")
    print(f"  wrote {xdmf_path}\n        {h5_path}")
    if n_metal == 0:
        print("  !! WARNING: no metal cells captured — decrease --h_min/--band")


if __name__ == "__main__":
    main()
