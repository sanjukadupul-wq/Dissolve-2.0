"""
weak_forms.py — UFL variational forms for all Dissolve 2.0 PDEs.
Returns (a, L) bilinear/linear form pairs for DOLFINx LinearProblem.
Exact port of FreeFem++ physics/governing_equations.idp.
"""

import ufl


def scaffold_indicator(phi):
    return ufl.conditional(ufl.gt(phi, 0), 1.0, 0.0)


def medium_indicator(phi):
    return ufl.conditional(ufl.le(phi, 0), 1.0, 0.0)


def delta_regularized(phi, epsilon):
    """Regularized 1-D Dirac in φ (cosine bump, integrates to 1 across the band)."""
    return ufl.conditional(
        ufl.lt(ufl.algebra.Abs(phi), epsilon),
        (1.0 / (2.0 * epsilon)) * (1.0 + ufl.cos(ufl.pi * phi / epsilon)),
        0.0
    )


def levelset_surface_delta(phi, epsilon):
    """Regularized surface Dirac implementing FreeFem++ int2d(Mesh, levelset=phi).

    Coarea formula:  ∫_{φ=0} f dS = ∫_Ω f · δ_ε(φ) · |∇φ| dx.
    The |∇φ| factor is essential — it makes the volume integral equal the exact
    interface surface integral even when |∇φ| ≠ 1 (φ drifts between redistancing
    steps). Omitting it (as the original port did) scales the ORR sink by an
    uncontrolled, time-varying factor and weakens interfacial O2 depletion.
    """
    grad_phi = ufl.grad(phi)
    norm_grad = ufl.sqrt(ufl.dot(grad_phi, grad_phi) + 1e-12)
    return delta_regularized(phi, epsilon) * norm_grad


def zinc_forms(V, Znold, phi, DeZn, F_film, Cl, dt, kf, kd, Fmax, Znbc, TGV, dx):
    """
    Returns (a, L) for Zn²⁺ transport.

    BC enforcement: the FreeFem++ original used a TGV=1e8 penalty term to
    enforce Zn = Znbc on the scaffold region.  That penalty destroys the
    matrix conditioning (cond ~ 1e8) and forces the use of AMG.  Here we
    drop the penalty — dissolve.py instead passes a proper
    fem.dirichletbc on the scaffold DOFs, which DOLFINx eliminates from the
    matrix.  Conditioning stays ~1e3 and bjacobi-on-GPU converges cleanly.

    `Znbc` and `TGV` are accepted for backward-compatible signature but unused.
    """
    Zn = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    medium = medium_indicator(phi)

    a = (Zn / dt * v * dx
         + DeZn * ufl.inner(ufl.grad(Zn), ufl.grad(v)) * dx
         - medium * kf * Zn * F_film / Fmax * v * dx
         + medium * kf * Zn * v * dx)

    L = (Znold / dt * v * dx
         + medium * kd * F_film * Cl * Cl * v * dx)

    return a, L


def film_forms(V, Fold, phi, Zn, Cl, dt, kf, kd, Fmax, dx):
    """Returns (a, L) for protective film. Matches FreeFem++ varf film."""
    F_trial = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    medium = medium_indicator(phi)

    a = (F_trial / dt * v * dx
         + medium * kf * Zn * F_trial / Fmax * v * dx
         + medium * kd * F_trial * Cl * Cl * v * dx)

    L = (Fold / dt * v * dx
         + medium * kf * Zn * v * dx)

    return a, L


def chloride_forms(V, Clold, phi, DeCl, dt, dx):
    """Returns (a, L) for Cl⁻ transport. Matches FreeFem++ varf chloride."""
    Cl = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (Cl / dt * v * dx
         + DeCl * ufl.inner(ufl.grad(Cl), ufl.grad(v)) * dx)

    L = (Clold / dt * v * dx)

    return a, L


def oxygen_forms(V, O2old, phi, DeO2, dt, kORR, epsilon, dx, exact_mode=False):
    """Returns (a, L) for O₂ with ORR surface reaction. Matches FreeFem++ varf oxygen.

    exact_mode: drop the regularised delta_eps(phi)*|grad phi| volumetric sink.
    In this mode the ORR consumption is instead injected as an explicit,
    per-DOF source (source_field in solve_transport), built from the EXACT
    marching-tetrahedra phi=0 patch (mechanics_fe.cut_patch_geometry) rather
    than smeared over an epsilon-wide band. See interface_velocity.py's
    compute_exact_orr_quantities. Prevents double-consuming O2 (once via this
    term, once via the explicit source) when both are active simultaneously.
    """
    O2 = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (O2 / dt * v * dx
         + DeO2 * ufl.inner(ufl.grad(O2), ufl.grad(v)) * dx)
    if not exact_mode:
        delta = levelset_surface_delta(phi, epsilon)
        a += delta * kORR * O2 * v * dx

    L = (O2old / dt * v * dx)

    return a, L


def hydroxide_forms(V, OHold, phi, DeOH, F_film, Cl, O2, dt, kd, kORR, epsilon, dx,
                     exact_mode=False):
    """Returns (a, L) for OH⁻. Matches FreeFem++ varf hydroxide.

    exact_mode: see oxygen_forms -- drops the smeared ORR source here too;
    the equivalent exact contribution is added via source_field instead.
    """
    OH = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    medium = medium_indicator(phi)

    a = (OH / dt * v * dx
         + DeOH * ufl.inner(ufl.grad(OH), ufl.grad(v)) * dx)

    L = (OHold / dt * v * dx
         + medium * kd * F_film * Cl * Cl * v * dx)
    if not exact_mode:
        delta = levelset_surface_delta(phi, epsilon)
        L += delta * 4.0 * kORR * O2 * v * dx

    return a, L


def levelset_forms(V, phiold, v_interface, dt, dx):
    """
    Returns (a, L) for level-set advection: ∂φ/∂t + v = 0
    Convention: phi > 0 = scaffold. v_interface < 0 for shrinkage.
    phi = phi_old + v*dt  (v < 0 → phi decreases → scaffold shrinks)
    """
    phi = ufl.TrialFunction(V)
    vp = ufl.TestFunction(V)

    a = (phi / dt * vp * dx)

    L = (phiold / dt * vp * dx
         + v_interface * vp * dx)

    return a, L
