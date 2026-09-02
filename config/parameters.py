"""
parameters.py — All simulation parameters for the Dissolve 2.0 FEniCSx model.
Mirror of FreeFem++ config/settings.idp with identical defaults.
"""

import argparse
import dataclasses
import os
import sys
from typing import Optional


def _load_toml(path: str) -> dict:
    """Return a flat {field_name: value} dict from a TOML config file."""
    try:
        import tomllib                        # Python 3.11+ stdlib
    except ImportError:
        try:
            import tomli as tomllib           # pip install tomli  (Python ≤ 3.10)
        except ImportError:
            sys.exit("ERROR: TOML support requires Python 3.11+ or 'pip install tomli'")

    with open(path, "rb") as fh:
        data = tomllib.load(fh)

    flat: dict = {}
    for val in data.values():
        if isinstance(val, dict):             # each [section] becomes a sub-dict
            flat.update(val)
        # top-level bare keys (uncommon) are silently ignored
    return flat


@dataclasses.dataclass
class SimulationParameters:
    """All simulation parameters, grouped by category."""

    # --- Solver control ---
    enable_zn: bool = True
    enable_film: bool = True
    enable_cl: bool = True
    enable_oh: bool = True
    enable_o2: bool = True
    enable_levelset: bool = True
    enable_flow: bool = False
    enable_full_ns: bool = True
    flow_solve_interval: int = 10

    # --- Output ---
    results_file: str = "output/result.txt"
    vtk_prefix: str = "output/dissolve"  # base name → output/dissolve.pvd + .vtu
    write_vtu: bool = True            # master switch — write .pvd + per-step .vtu
    emit_vtk: bool = True            # alias for write_vtu (legacy name)
    write_xdmf: bool = False          # also write .xdmf (broken in DOLFINx 0.10)
    vis_each_steps: int = 1           # write visualization every N steps (1 = every step)
    dump_initial_mesh: bool = False
    save_scale_factor: float = 1.0
    normalize_by_area: bool = False
    dump_final_state: bool = True
    export_flow_field: bool = True

    # --- Mesh ---
    load_mesh: bool = True
    input_mesh: str = "mesh/sphere.mesh"
    tag_scaffold: int = 1
    tag_medium: int = 2
    tag_wall: int = 3
    tag_inlet: int = 4
    tag_outlet: int = 5
    analytic_sdf: str = ""    # "disc"/"stent" -> init phi from the exact analytic
                              # SDF at DOF coordinates instead of the region-label
                              # nearest-interface-DOF approximation (which produces
                              # faceted/pockmarked phi=0 contours on graded meshes).
                              # "" (default) keeps the old region-label behaviour.

    # --- Mesh generation (internal tetgen) ---
    domain_length: float = 20.0
    scaffold_size_x: float = 13.0
    scaffold_size_y: float = 13.0
    scaffold_size_z: float = 4.0
    target_mesh_size: int = 32

    # --- Mesh refinement ---
    adapt_initial_mesh: bool = False
    adapt_mesh_runtime: bool = False
    refine_threshold_pct: float = 5.0
    mshmet_tolerance: float = 1e-2
    mesh_h_min: float = 4e-2
    mesh_h_max: float = 0.8

    # --- Redistancing ---
    enable_redistance: bool = True
    # Stage-1 FreeFem default: dtRedistance defaults to dt (redistance_time,
    # via state/initial_state.idp's `dtRedistance = dtRedistance>=dt?dtRedistance:dt`) --
    # i.e. every single timestep by default.  Matched here (was 5).
    redistance_interval: int = 1          # redistance every N steps (used only when adaptive_dt=False)
    redistance_time_interval: float = 5.0 # redistance every N simulated hours (used when adaptive_dt=True)
    # Per-step redistancing iterations (main loop; separate from the one-time
    # initial redistancing, which always uses 15).  A single algebraic
    # level-set step (phi += v*dt) distorts |grad(phi)| more at larger dt --
    # 5 Sussman pseudo-time iterations may under-resolve that distortion at
    # dt=4h even though it's enough at dt=1h, which can systematically bias
    # the recession rate at coarse dt.  Tunable here to test/fix that.
    redistance_iters: int = 5

    # --- Time stepping (hours) ---
    dt_hours: float = 1.0
    sim_duration: float = 672.0
    redistance_time: float = 0.25
    save_each: float = 1.0

    # --- Scaffold export ---
    export_geometry: bool = False
    export_geometry_each: float = 1.0
    export_geometry_volume: bool = True
    export_geometry_surface: bool = True

    # --- Film properties (dimensionless) — paper Table 1 ---
    film_tortuosity: float = 2.00      # tortuosity (fixed, literature)
    film_porosity: float = 0.25      # porosity (fixed, literature)
    film_constrictivity: float = 1.0       # constrictivity δ (paper Table 1; D_eff floor = eps·δ/τ²)
    # --- ORR interface discretisation (Stage-1 FreeFem parity) ---
    # FreeFem applies the ORR sink as an EXACT zero-thickness surface integral,
    # int2d(Mesh, levelset=phi)(kORR*O2*v).  We approximate it with the coarea
    # form delta_eps(phi)*|grad phi|, whose band half-width is
    #     epsilon = orr_eps_factor * h_avg.
    # The historical default 2.0 gives eps=0.77mm on the disc mesh -> a 1.54mm
    # thick sink band on a 2mm thick specimen, which drains O2 from a volume
    # instead of a surface, collapsing the interfacial O2 that v_ORR reads.
    # Lower values approach FreeFem's true surface sink; too low and the P1
    # quadrature can no longer resolve the delta (sink under-integrated).
    orr_eps_factor: float = 2.0
    # If > 0, evaluate the O2 that drives v_ORR at this FIXED physical distance
    # (mm) into the electrolyte along -n, instead of using the local nodal value.
    # Stage-1 FreeFem hardcodes h = 0.002 mm (state/initial_state.idp) and probes
    # O2(x + dir) with dir = -h*grad(phi)/|grad phi|.  A fixed offset is
    # mesh-INDEPENDENT, which is why FreeFem does not lose an order of magnitude
    # of corrosion under mesh refinement while the nodal-value version does.
    orr_probe_dist: float = 0.0
    # Report the O2 <-> Zn conservation chain each diagnostics step.
    check_conservation: bool = False
    check_bounds: bool = False  # print min/max(O2,OH,F,Zn) + F/Fmax every step
    # Set True to restore the old v_orr = -_V_ORR*O2/(1/k_orr+R_diff) formula
    # (an extra film-transport resistance baked into the reaction branch that
    # has no FreeFem counterpart -- analytically confirmed to suppress v_orr
    # to ~46% of FreeFem's value at late-time film saturation). Default False
    # now matches FreeFem's actual undivided vO2React = -2*kORR*O2/Znsolid.
    legacy_vorr_r_diff: bool = False
    force_o2_bulk: bool = False  # diagnostic: freeze O2 == o2_initial everywhere,
                                 # skipping the transport PDE entirely (no depletion
                                 # anywhere, ever). If mass loss jumps toward the
                                 # Table 3 target with this on, the discrepancy is in
                                 # O2 transport/supply, not reaction kinetics.
    # Replace the regularised delta_eps(phi)*|grad phi| sink/velocity coupling
    # entirely with the EXACT marching-tetrahedra phi=0 surface (mechanics_fe.
    # cut_patch_geometry via interface_velocity.compute_exact_orr_quantities).
    # The O2 sink becomes an explicit per-DOF source (orr_eps_factor/epsilon
    # no longer apply to O2/OH) and v_ORR reads the SAME per-cell exact O2 the
    # sink consumed -- the sink and the recession can no longer disagree about
    # what O2 was at the interface, which is the root cause identified in the
    # conservation audit (actual/stoich ~ 0.14 under the old smeared coupling).
    use_exact_orr: bool = False
    # Primary mass-loss volume metric: exact marching-tetrahedra (default) vs
    # the old regularised Heaviside integral. The smoothed one has a mesh-
    # curvature-dependent bias (over-estimates volume LOSS ~2.5x on a rough
    # mesh, under-estimates it ~2.35x on a clean one -- verified via the
    # conservation audit) that made mass-loss % non-comparable across mesh
    # families. Set False only to reproduce old (pre-fix) numbers for A/B
    # comparison against historical runs.
    use_exact_volume: bool = True

    bruggeman_tau: bool = False     # derive τ from ε via Bruggeman τ = ε^(-1/2) instead of
                                    # using film_tortuosity as given. Makes ε the single knob for
                                    # film transport: Eq.12's floor ε·δ/τ² collapses to ε²·δ.
                                    # Consistency check — the paper's own Table 1 pair already
                                    # satisfies this exactly: 0.25^(-1/2) = 2.00 = film_tortuosity,
                                    # giving floor = 0.25²·1.0 = 0.0625. So enabling this with
                                    # ε=0.25 reproduces the literature values bit-for-bit, and
                                    # raising ε stays on the same physical relationship.

    # --- Reaction kinetics ---
    kf: float = 125                 # film formation rate (1/hour)
    kd: float = 40                  # film dissolution rate (mm^6/(g^2·hour))

    # --- Material densities (g/mm³) ---
    rho_zn: float = 0.00714     # Zn solid density
    rho_film: float = 0.005606  # ρ_film ZnO (paper Table 1: 5.606 g/cm³)
    film_capacity_scale: float = 1.0

    # --- Zn concentration ---
    zn_molar_conc: float = 1.09e-4  # mol/mm³  (Znsolid)
    zn_boundary_conc: float = 1.00e-5     # g/mm³    (Znbc)

    # --- Diffusion coefficients (mm²/hour) — paper Table 1 (37°C, HBSS) ---
    diff_zn: float = 2.72      # 7.56e-4 mm²/s × 3600
    diff_cl: float = 7.78      # 2.16e-3 mm²/s × 3600
    diff_oh: float = 20.16     # 5.60e-3 mm²/s × 3600
    diff_o2: float = 9.36      # 2.60e-3 mm²/s × 3600

    # --- O2 boundary condition mode ---
    # "all" = O2=bulk on all exterior facets (legacy); "none" = no O2 Dirichlet
    # (max depletion); "xmin" = only the min-x face (FreeFem Inlet stand-in).
    o2_bc_mode: str = "all"

    # --- Initial concentrations (g/mm³) — paper Table 1 ---
    cl0: float = 5.044e-6    # 142.3 mmol/L Cl- (paper Table 1)
    oh0: float = 1.07e-11    # baseline pH 7.40 at 37°C (pKw=13.6):
                                    # pOH=6.20 → [OH-]=10^-6.20=6.31e-7 mol/L → 1.07e-11 g/mm³
    o2_initial: float = 3.5e-9      # 3.5 mg/L dissolved O2 (paper Table 1, disc validation)

    # --- ORR parameters ---
    # k_orr is the SINGLE ORR surface reaction-rate constant [mm/h], used
    # consistently everywhere the model needs an ORR rate:
    #   (1) O2 PDE sink & OH source:  sink = δ_ε(φ)·|∇φ|·k_orr·O2
    #       (δ·|∇φ| carries units 1/mm (dS/dV), so k_orr acts as a
    #        mass-transfer velocity, not a volumetric 1/h rate)
    #   (2) interface (recession) velocity, resistance-in-series:
    #       v_n = -_V_ORR * C_O2 / (1/k_orr + film_length/DeO2_probed)
    # There is deliberately only ONE ORR rate constant: physically, O2
    # consumption and metal recession are stoichiometrically tied to the same
    # surface reaction, so they must share the same rate constant rather than
    # being independently tunable (the old split into k_orr + k_orr_surface
    # let the two drift apart and made calibration ambiguous).
    # Calibrated via Bayesian RMSE minimization against experimental mass-loss
    # data (rounded calibrated default; raw optimizer output was 0.0144).
    k_orr: float = 0.015            # mm/h  ORR surface rate (O2/OH sink AND interface velocity)
    film_length:   float = 0.050    # mm    effective O2 diffusion path through corrosion film
                                    #       bounds: sqrt(D_Zn_eff/k_f)=9.6um .. sqrt(D_Zn/k_f)=147um

    # Interface-velocity formulation:
    #   "surrogate" (default) — unified resistance-in-series (paper Eq.15 reduced):
    #       v_n = -V_ORR * C_O2 / (1/k_orr + film_length/DeO2)
    #   "physical" — paper Eq.14+16 mixed control, max(v_ORR, v_O2):
    #       v_ORR = -V_ORR * k_orr_react * C_O2           (reaction-limited)
    #       v_O2  = -V_ORR * C_O2 * DeO2 / film_length    (diffusion-limited)
    #       v_n   = max(v_ORR, v_O2)   (least-negative = rate-limiting mechanism)
    #     Use for reaction-limited geometries (e.g. thin stent struts) where the
    #     physical ORR rate governs, so it transfers across geometries.
    velocity_mode: str = "surrogate"
    k_orr_react:   float = 0.90     # mm/h  physical ORR reaction rate for "physical" mode
                                    #       (paper k_ORR=0.0144 in its unit system; ~0.9 here
                                    #       after the V_ORR=2/(M_O2·[Zn]_sol)=572 prefactor)

    # Film-coverage passivation of the dissolution velocity:
    #   v_react *= block,  block = film_block_min + (1-film_block_min)*(1 - F/Fmax)
    # film_block_min is the residual active fraction once the film is fully formed
    # (a porous/imperfect film never blocks 100%). 0 = total passivation (rate->0),
    # ~0.1 = 10% residual rate so dissolution continues slowly (matches data that
    # keeps creeping up rather than stopping).
    film_block_min: float = 0.0
    use_orr: bool = True

    # --- Stefan initial velocity ---
    stefan_alpha: float = 0.0              # alpha multiplier (0 = no initial velocity)

    # --- Fluid flow (mm/hour) ---
    inflow_x: float = 36.0
    inflow_y: float = 0.0
    inflow_z: float = 0.0
    kinematic_viscosity: float = 3.607         # kinematic viscosity

    # --- GPU acceleration ---
    use_gpu: bool = True            # enable GPU when PETSc CUDA or libCEED available
    # Interface-velocity gradient operator:
    #   False -> DOLFINx FE-projection (identical accuracy to the original CPU model)
    #   True  -> GPU volume-averaged nodal_gradient (faster, slightly different)
    # Heavy transport solves stay on GPU either way.
    velocity_use_gpu_grad: bool = False

    # --- Adaptive time stepping ---
    adaptive_dt: bool = False       # enable adaptive dt based on KSP iteration counts
    dt_min: float = 0.1             # minimum allowed dt (hours)
    time_step_max: float = 4.0             # maximum allowed dt (hours)
    dt_target_iters: int = 30       # target KSP iterations per step (all fields combined)
    dt_growth_factor: float = 1.2     # multiply dt by this when iters < target
    dt_shrink_factor: float = 0.7   # multiply dt by this when iters > 2*target

    # --- Time-based VTK output ---
    save_vtk_each_time: float = 24.0  # write a VTK snapshot every N hours
                                     # (0 = fall back to vis_each_steps count)

    # --- Diagnostics ---
    write_diagnostics: bool = True  # write per-step CSV with timing/iters/norms
    diagnostics_file: str = "output/diagnostics.csv"
    debug_vn_integral: bool = False  # print the AREA-WEIGHTED (not naive-mean) v_n
                                     # each step -- independent check, not derived
                                     # from vol_loss (unlike penetration_mm_yr)

    # --- Mechanics (Phase 2): 3D FE apparent modulus E*(t) at each VTK snapshot ---
    mechanics: bool = False              # run linear-elastic FE on {phi>0} at snapshots
    mechanics_file: str = "output/mechanics.csv"
    mech_E_solid: float = 90000.0        # solid Zn Young's modulus [MPa]
    mech_nu: float = 0.30                # Poisson ratio
    mech_axis: int = 2                   # compression axis (0=x,1=y,2=z)
    mech_strain: float = 0.005           # applied nominal compressive strain
    mech_yield: float = 140.0            # solid yield stress [MPa] (first-yield flag)

    # --- Numerics ---
    TGV: float = 1e8                # penalty for Dirichlet inside scaffold
    h_min_fallback: float = 0.002   # fallback smallest element size

    # --- Stefan-condition interface velocity (Stage-1 FreeFem exact port) ---
    # Stage-1 state/initial_state.idp computes h from the mesh (hTriangle min) then
    # unconditionally overwrites it with a fixed h=0.002 mm -- every probe
    # distance in physics/interface_velocity.idp (the two-point Zn/O2 finite-difference
    # gradient) uses this literal constant, not the actual mesh spacing.
    # Kept as its own named parameter (not reusing h_min_fallback, which is a
    # different fallback-only value) so it's clear this is a physics constant
    # from the reference model, not a numerical safety net.
    stefan_probe_h: float = 0.002   # mm, fixed probe distance (matches Stage-1 exactly)

    def __post_init__(self):
        # Bruggeman: tortuosity is not independent of porosity -- tau = eps^(-1/2).
        # Applied here (not in validate()) so it holds for every construction path,
        # including parse_args, and so validate()'s tau >= 1 check sees the derived
        # value.  Note eps^(-1/2) >= 1 for all eps in (0,1], so it always passes.
        if self.bruggeman_tau:
            self.film_tortuosity = self.film_porosity ** -0.5

    # --- Derived ---
    @property
    def Fmax(self) -> float:
        """Maximum protective film concentration (g/mm³)."""
        return self.rho_film * (1.0 - self.film_porosity) * self.film_capacity_scale

    @property
    def dt(self) -> float:
        return self.dt_hours

    @property
    def Znsolid(self) -> float:
        return self.zn_molar_conc

    @property
    def Znbc(self) -> float:
        return self.zn_boundary_conc

    @property
    def rhoZn(self) -> float:
        return self.rho_zn

    @property
    def save_each_steps(self) -> int:
        ratio = self.save_each / self.dt_hours
        return max(1, int(ratio))

    # ── Validation ────────────────────────────────────────────────────────
    def validate(self) -> tuple:
        """Check all parameters before the simulation starts.

        Returns
        -------
        errors   : list of {"msg": str, "fix": str}  — must be fixed; sim cannot run
        warnings : list of {"msg": str}              — informational; sim will still run
        """
        errors: list   = []
        warnings: list = []

        def err(msg: str, fix: str = "") -> None:
            errors.append({"msg": msg, "fix": fix})

        def warn(msg: str) -> None:
            warnings.append({"msg": msg})

        # ── Mesh file ────────────────────────────────────────────────────
        if not self.input_mesh:
            err("input_mesh is empty",
                "Set input_mesh under [mesh] in simulation.toml")
        elif not os.path.exists(self.input_mesh):
            err(f"Mesh file not found: '{self.input_mesh}'",
                "Set [mesh] input_mesh in simulation.toml\n"
                "       or pass --input_mesh /path/to/mesh.xdmf")
        else:
            ext = os.path.splitext(self.input_mesh)[1].lower()
            if ext not in {".xdmf", ".mesh", ".msh"}:
                err(f"Unsupported mesh format '{ext}'",
                    "Use .xdmf (preferred), .mesh, or .msh")

        # Mesh label uniqueness
        _labels = {"tag_scaffold": self.tag_scaffold,
                   "tag_medium":   self.tag_medium,
                   "tag_wall":     self.tag_wall}
        _seen: dict = {}
        for _lname, _lval in _labels.items():
            if _lval in _seen:
                err(f"Duplicate mesh label {_lval}: '{_lname}' clashes with '{_seen[_lval]}'",
                    "Each mesh region must have a unique integer tag")
            _seen[_lval] = _lname

        # ── Time stepping ────────────────────────────────────────────────
        if self.sim_duration <= 0:
            err(f"sim_duration must be > 0  (got {self.sim_duration} h)",
                "Set sim_duration under [time] in simulation.toml")

        if self.dt_hours <= 0:
            err(f"dt_hours must be > 0  (got {self.dt_hours} h)")
        elif self.sim_duration > 0 and self.dt_hours > self.sim_duration:
            err(f"dt_hours ({self.dt_hours} h) exceeds sim_duration ({self.sim_duration} h)",
                "Reduce dt_hours or increase sim_duration")
        elif self.dt_hours > 4.0:
            warn(f"dt_hours = {self.dt_hours} h is coarse — solver accuracy may degrade")

        # Adaptive dt
        if self.adaptive_dt:
            if self.dt_min <= 0:
                err(f"dt_min must be > 0  (got {self.dt_min} h)")
            if self.time_step_max <= 0:
                err(f"time_step_max must be > 0  (got {self.time_step_max} h)")
            if self.dt_min > 0 and self.time_step_max > 0 and self.time_step_max <= self.dt_min:
                err(f"time_step_max ({self.time_step_max} h) must be greater than dt_min ({self.dt_min} h)",
                    "Set time_step_max > dt_min under [time]")
            if self.dt_growth_factor <= 1.0:
                err(f"dt_growth_factor must be > 1.0  (got {self.dt_growth_factor})",
                    "Typical value: 1.2")
            if not (0.0 < self.dt_shrink_factor < 1.0):
                err(f"dt_shrink_factor must be in (0, 1)  (got {self.dt_shrink_factor})",
                    "Typical value: 0.7")

        # ── Diffusion coefficients ───────────────────────────────────────
        for _n, _v in [("diff_zn", self.diff_zn), ("diff_cl", self.diff_cl),
                       ("diff_oh", self.diff_oh), ("diff_o2", self.diff_o2)]:
            if _v <= 0:
                err(f"{_n} must be > 0  (got {_v} mm²/h)",
                    f"Set {_n} under [diffusion] in simulation.toml")

        # ── Reaction kinetics ────────────────────────────────────────────
        if self.kf <= 0:
            err(f"kf (film formation rate) must be > 0  (got {self.kf} /h)")
        if self.kd <= 0:
            err(f"kd (film dissolution rate) must be > 0  (got {self.kd})")
        if self.use_orr and self.k_orr <= 0:
            err(f"k_orr must be > 0 when use_orr = true  (got {self.k_orr} mm/h)",
                "Calibrated baseline: 0.015 mm/h")

        # ── Material properties ──────────────────────────────────────────
        if not (0.0 < self.film_porosity < 1.0):
            err(f"film_porosity (porosity) must be between 0 and 1  (got {self.film_porosity})",
                "Typical range: 0.3 – 0.6")
        if self.film_tortuosity < 1.0:
            err(f"film_tortuosity (tortuosity) must be ≥ 1.0  (got {self.film_tortuosity})",
                "Tortuosity is always ≥ 1 by definition")
        if self.rho_zn <= 0:
            err(f"rho_zn must be > 0  (got {self.rho_zn} g/mm³)")
        if self.rho_film <= 0:
            err(f"rho_film must be > 0  (got {self.rho_film} g/mm³)")

        # ── Concentrations ───────────────────────────────────────────────
        for _n, _v in [("cl0", self.cl0),
                       ("oh0", self.oh0),
                       ("o2_initial", self.o2_initial)]:
            if _v < 0:
                err(f"{_n} must be ≥ 0  (got {_v} g/mm³)",
                    f"Set {_n} under [concentrations] in simulation.toml")
        if self.zn_boundary_conc <= 0:
            err(f"zn_boundary_conc must be > 0  (got {self.zn_boundary_conc} g/mm³)")
        if self.zn_molar_conc <= 0:
            err(f"zn_molar_conc must be > 0  (got {self.zn_molar_conc} mol/mm³)")

        # ── Redistancing ─────────────────────────────────────────────────
        if self.enable_redistance and self.redistance_interval < 1:
            err(f"redistance_interval must be ≥ 1  (got {self.redistance_interval})")

        # ── Output ───────────────────────────────────────────────────────
        if self.vis_each_steps < 1:
            err(f"vis_each_steps must be ≥ 1  (got {self.vis_each_steps})")

        if self.dt_hours > 0 and self.sim_duration > 0:
            _n_steps = int(self.sim_duration / self.dt_hours)
            # Only warn about vis_each_steps when time-based interval is not in use
            if (self.save_vtk_each_time <= 0 and self.write_vtu
                    and _n_steps > 0 and self.vis_each_steps > _n_steps):
                warn(f"vis_each_steps ({self.vis_each_steps}) > total steps ({_n_steps})"
                     f" — VTK output disabled")

        return errors, warnings


def _print_version_and_exit() -> None:
    """Print a styled version card and exit 0.  No heavy imports — runs instantly."""
    from version import (VERSION, APP_NAME, FULL_NAME,
                         PRODUCT_NAME, BUILD_DATE, AUTHOR)

    _CY = "\033[96m"   # bright cyan
    _BO = "\033[1m"    # bold
    _DI = "\033[2m"    # dim
    _RS = "\033[0m"    # reset

    W = 54             # interior width

    def row(text: str = "", cc: str = "") -> None:
        pad = W - len(text)
        if pad < 0:
            text, pad = text[:W], 0
        if cc:
            print(f"║{cc}{text}{_RS}{' ' * pad}║")
        else:
            print(f"║{text:<{W}}║")

    pyver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    print("╔" + "═" * W + "╗")
    row(f"  {APP_NAME}  v{VERSION}", _CY + _BO)
    row(f"  {FULL_NAME}")
    row(f"  {PRODUCT_NAME}", _DI)
    print("╠" + "═" * W + "╣")
    row(f"  Python {pyver}  ·  Build: {BUILD_DATE}", _DI)
    row(f"  {AUTHOR}", _DI)
    print("╚" + "═" * W + "╝")
    sys.exit(0)


def parse_args(argv: Optional[list] = None) -> SimulationParameters:
    """Parse command-line arguments (and optional TOML config) into SimulationParameters.

    Precedence (highest wins):
        CLI flags  >  TOML config file  >  SimulationParameters defaults
    """
    # ── First pass: --version and --config before anything else ──────────
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config",  type=str,  default=None)
    pre.add_argument("--version", "-V", action="store_true", default=False)
    pre_ns, _ = pre.parse_known_args(argv)

    if pre_ns.version:
        _print_version_and_exit()

    toml_vals: dict = {}
    if pre_ns.config:
        toml_vals = _load_toml(pre_ns.config)

    # ── Main parser: each field default comes from TOML (or dataclass) ────
    p = argparse.ArgumentParser(
        description="DISSOLVE — Dissolve 2.0 FEniCSx Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None,
                   help="TOML config file; CLI flags override it")

    base = SimulationParameters()

    for field in dataclasses.fields(base):
        name = f"--{field.name}"
        # TOML value (if present) replaces the dataclass default
        raw = toml_vals.get(field.name, getattr(base, field.name))
        if field.type is bool:
            # TOML booleans are real Python bools; argparse expects 0/1 int
            p.add_argument(name, type=int, default=int(bool(raw)),
                           help=f"(0/1, default={int(bool(raw))})")
        elif field.type is int:
            p.add_argument(name, type=int, default=int(raw))
        elif field.type is float:
            p.add_argument(name, type=float, default=float(raw))
        elif field.type is str:
            p.add_argument(name, type=str, default=str(raw))

    args = p.parse_args(argv)
    kwargs: dict = {}
    for field in dataclasses.fields(base):
        val = getattr(args, field.name)
        if field.type is bool:
            val = bool(val)
        kwargs[field.name] = val

    return SimulationParameters(**kwargs)
