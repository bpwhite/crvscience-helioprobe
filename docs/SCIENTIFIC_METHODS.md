# HelioProbe v2.3.0 — Scientific Methods & Mathematical Reference

**License:** GNU General Public License v3.0 or later (`GPL-3.0-or-later`, see [LICENSE](../LICENSE))

This document specifies the physical constants, astrodynamic solvers, relativistic cruise equations, Sun–Earth L2 three-body dynamics, cometary plasma/dust tail physics, 3D rendering transforms, and numerical boundary guards implemented in `helioprobe`.

---

## 1. Physical Constants & Reference Frames

Defined in `src/physics.rs` and `src/catalog.rs`:

| Constant | Symbol | Value | Units |
| :--- | :--- | :--- | :--- |
| Astronomical Unit | `AU_KM` | `149_597_870.7` | `km` |
| Solar Gravitational Parameter | `GM_SUN_KM3_S2` | `132_712_440_018.0` | `km^3 / s^2` |
| Earth Gravitational Parameter | `GM_EARTH_KM3_S2` | `398_600.4418` | `km^3 / s^2` |
| Mean Earth Radius | `EARTH_RADIUS_KM` | `6_371.0` | `km` |
| Low Earth Orbit Reference Alt | `LEO_ALTITUDE_KM` | `400.0` | `km` |
| Speed of Light in Vacuum | `SPEED_OF_LIGHT_KM_S` | `299_792.458` | `km / s` |
| CODATA Gravitational Constant | `G` | `6.67430e-20` | `km^3 / (kg * s^2)` |

### Coordinate Mapping
Orbital states `(r_ecl, v_ecl)` in `kepler_state_ecliptic_km` are computed in the right-handed J2000 heliocentric ecliptic frame `(x_ecl, y_ecl, z_ecl)` (`+z_ecl` toward ecliptic north, `(r x v).z > 0` for prograde orbits with `i < 90 deg`). Scene coordinates map ecliptic north onto `+Y` so the 3D camera up-axis aligns with ecliptic north:

```text
r_scene_orientation = (x_ecl, z_ecl, y_ecl)
```

For large dynamic ranges (`0.38 AU` to `7,500 ly`), visual display positions apply a strictly radial monotonic compression (`compress_au_to_scene`) that preserves angular phase and orbital inclination:

```text
r_scene(r_AU) = 16.0 * ln(1.0 + 1.15 * r_AU) + 4.2 * sqrt(r_AU)
```

---

## 2. Keplerian Solvers & Universal-Anomaly Propagation

### 2.1 Safeguarded Elliptical Kepler Solver (`solve_kepler_eccentric_anomaly`)
Given mean anomaly `M` reduced to `[0, 2*pi)` and clamped eccentricity `e in [0, 0.999995]`, Kepler's equation is solved via bracketed Newton–Raphson with bisection fallback:

```text
f(E)  = E - e * sin(E) - M = 0
f'(E) = 1 - e * cos(E)
```

For high-eccentricity comets (`e > 0.8`), iteration seeds at `E_0 = pi` and maintains strict bracket `[lo, hi]`, guaranteeing `|E - e * sin(E) - M| < 1e-12` across `e <= 0.999995`.

### 2.2 Perifocal to J2000 Ecliptic State (`kepler_state_ecliptic_km`)
From eccentric anomaly `E`, true anomaly `nu` and radius `r` are:

```text
tan(nu / 2) = sqrt((1 + e) / (1 - e)) * tan(E / 2)
r = a * (1 - e * cos(E))
p = a * (1 - e^2)

r_pf = [ r * cos(nu),              r * sin(nu),              0 ]
v_pf = [ -sqrt(mu / p) * sin(nu),  sqrt(mu / p) * (e + cos(nu)), 0 ]
```

Rotating via perifocal unit vectors `(P_hat, Q_hat)` from inclination `i`, longitude of ascending node `Omega`, and argument of periapsis `omega` yields exact 3D `(r_km, v_km_s)` satisfying vis-viva energy conservation:

```text
epsilon = 0.5 * |v|^2 - mu_sun / |r| = -mu_sun / (2 * a)
```

### 2.3 Conway's Laguerre Universal-Anomaly Propagator (`propagate_kepler_ecliptic_km`)
Using universal anomaly `chi` with reciprocal semi-major axis `alpha = 2 / |r_0| - |v_0|^2 / mu` and Stumpff functions `C(z)`, `S(z)` (`z = alpha * chi^2`):

```text
F(chi) = (r_0 * v_r0 / sqrt(mu)) * chi^2 * C(z)
       + (1 - alpha * r_0) * chi^3 * S(z)
       + r_0 * chi - sqrt(mu) * dt = 0
```

**Boundary Guards:**
- Near `z = 0` (`|z| <= 1e-6`), 4-term Taylor expansions of `C(z)` and `S(z)` prevent `0/0` cancellation.
- For strongly hyperbolic trajectories (`z < -1e-6`), `sqrt(-z)` is capped at `700.0` before calling `cosh`/`sinh` to avoid IEEE-754 overflow, and `chi` is clamped within `+-600 / sqrt(-alpha)`.
- Steps use `n = 5` Laguerre iteration bounded inside a monotonic root bracket `[lo, hi]`, verified to `< 1 km` agreement against 12,000-step RK4 two-body integration.

---

## 3. 3D Universal-Variable Lambert Porkchop & Giant-Planet Capture

### 3.1 3D Universal-Variable Lambert Solver (`solve_lambert_branch`)
For departure `r_1`, arrival `r_2`, transfer duration `dt > 1 s`, and branch sign `dm in {+1, -1}` (short-way vs long-way):

```text
A = dm * sqrt(|r_1| * |r_2| * (1 + cos(Delta nu)))
y(z) = |r_1| + |r_2| + A * (z * S(z) - 1) / sqrt(C(z))
chi(z) = sqrt(y(z) / C(z))
t(z) = (chi^3 * S(z) + A * sqrt(y(z))) / sqrt(mu_sun)
```

Analytical derivative `dt / dz` (Bate–Mueller–White Eq. 5.3-13) with safeguarded bisection solves `t(z) = dt` to `< 1e-4 s`, recovering `(v_1, v_2)` via Lagrange `f`, `g`, `g_dot` coefficients:

```text
f     = 1 - y / |r_1|
g     = A * sqrt(y / mu_sun)
g_dot = 1 - y / |r_2|

v_1 = (r_2 - f * r_1) / g
v_2 = (g_dot * r_2 - r_1) / g
```

### 3.2 Reconciled Orbital Capture (`compute_capture_orbit`)
Both the Earth LEO Probe Launcher (`src/physics.rs`) and the Lambert Porkchop / Gravity-Assist Planner (`src/astrodynamics_lab.rs`) invoke the single shared `compute_capture_orbit(target, v_inf_arr)` model:

```text
h_cap = 6,100,000 km  (Sun)
      = 1.5 km        (R < 10 km asteroids/comets)
      = 4,500 km      (R > 20,000 km gas/ice giants)
      = 300 km        (terrestrial planets, moons, dwarf planets)

r_cap = R_body + h_cap
e_cap = 0.60 if R_body > 20,000 km (Juno/Cassini elliptical giant capture) else 0.00

v_peri_arr = sqrt(v_inf_arr^2 + 2 * mu_target / r_cap)
v_cap_orb  = sqrt((1 + e_cap) * mu_target / r_cap)
Delta_v_cap = |v_peri_arr - v_cap_orb|
```

Using `e_cap = 0.60` for Jupiter, Saturn, Uranus, and Neptune eliminates unphysical circular-insertion clamping and reconciles Launcher and Porkchop capture `Delta_v` identically (`< 1e-12 km/s` difference).

### 3.3 Patched-Conic Gravity-Assist Slingshots (`compute_gravity_assist`)
Evaluates 2-leg `Earth -> Assist -> Target` transfers across `[venus, mars, jupiter, saturn]`, enforcing radial reachability, hyperbolic bending angle limit at safe periapsis `r_p`:

```text
sin(delta_max / 2) = 1 / (1 + r_p * v_inf_in^2 / mu_assist)
```

and `<= 0.35 km/s` powered Oberth periapsis trim, plus 3D inclination cranking and residual plane-change delta-v on equal footing with direct 3D Hohmann transfers.

---

## 4. Relativistic Interstellar Transfers

For interstellar exoplanets and deep-sky nebulae (`distance_ly > 0.0`, `src/physics.rs`):

```text
beta  = clamp(cruise_beta_c, 0.001, 0.999999)     (strictly subluminal: 0 < beta < 1)
gamma = 1 / sqrt(1 - beta^2)                      (Lorentz factor)

t_coord_days = (distance_ly / beta) * 365.25      (Earth-frame coordinate duration)
tau_ship_days = t_coord_days / gamma              (Relativistic ship proper time)

Delta_v_boost_km_s = c * atanh(beta)              (Relativistic rocket rapidity)
Delta_v_total_km_s = 2 * c * atanh(beta) + v_cap  (Boost + deceleration + orbit capture)
```

Because `atanh(beta) > beta` for all `beta in (0, 1)`, `Delta_v_total_km_s >= c * atanh(beta)` holds strictly across all speeds while guarding `beta >= 1.0`, `beta <= 0.0`, and `NaN`.

---

## 5. Sun–Earth L2 CR3BP Dynamics & Hill Sphere

Implemented in `src/l2_comets.rs` (`compute_sun_earth_l2_dynamics`):

```text
r_L2_Hill = a_E * (mu_E / (3 * mu_sun))^(1/3)  ≈ 1,496,558 km
mu        = mu_E / (mu_sun + mu_E)             ≈ 3.00348e-6
```

Solving the exact Circular Restricted Three-Body Problem (CR3BP) collinear L2 Euler quintic via Newton–Raphson:

```text
(1 - mu + gamma) - (1 - mu) / (1 + gamma)^2 - mu / gamma^2 = 0   => gamma ≈ 0.0100346 (1,501,154 km)
c_2 = mu / gamma^3 + (1 - mu) / (1 + gamma)^3                    ≈ 3.9405
```

With Earth's mean motion `n = sqrt((mu_sun + mu_E) / a_E^3) * 86,400 rad/day`, linearized CR3BP frequencies around L2 are:

```text
omega_xy = n * sqrt((2 - c_2 + sqrt(9 * c_2^2 - 8 * c_2)) / 2)   => T_xy ≈ 177.6 days
omega_z  = n * sqrt(c_2)                                         => T_z  ≈ 184.0 days
A_x / A_y = (2 * omega_xy / n) / ((omega_xy / n)^2 + 1 + 2 * c_2) ≈ 0.314
```

Observatories (`JWST`, `Nancy Grace Roman`, `Comet Interceptor`) evolve along this quasi-halo Lissajous manifold at `(omega_xy, omega_z)`.

---

## 6. Cometary Sublimation & Solar-Wind Ion-Tail Aberration

In `compute_comet_tail_state` (`src/l2_comets.rs`), comet position and velocity `(r_ecl, v_ecl)` come directly from `kepler_state_ecliptic_km` (avoiding compressed-scene radial distortion):

```text
r_hat   = r_phys / |r_phys|
v_r     = dot(v_phys, r_hat)
v_perp  = |v_phys - v_r * r_hat|

v_sw    = 420.0 km/s (radial solar wind)
v_rel   = v_sw * r_hat - v_phys = (v_sw - v_r) * r_hat - v_perp_vec

alpha_ion = atan2(v_perp, v_sw - v_r)
```

Type I plasma tails align along `v_rel / |v_rel|`, matching `atan2(v_perp, v_sw - v_r)` to floating-point precision (`< 1e-6 deg`) across all eccentricities including retrograde `1P/Halley` (`e = 0.9672`). Type II silicate dust tails trail along `-v_hat` with radiation-pressure synchrone curvature `t^1.55`.

---

## 7. 3D View-Space Illumination, Ring Obliquity & Uniform-`E` Orbits

- **View-Space Solar Lighting (`shaders/cosmos.wgsl`, `src/renderer_3d.rs`)**: Transforms world-space `-r_body / |r_body|` into camera basis `(right, up, -forward)` and evaluates Lambert `max(dot(normal, light_dir), 0.0)` so planetary phases track the Sun rather than the camera.
- **3D Obliquity-Tilted Planetary Rings**: Constructs 56-segment 3D annular ring geometry oriented by planetary axial obliquity (`Saturn 26.73 deg`, `Uranus 97.77 deg`, `Neptune 28.32 deg`, `Jupiter 3.13 deg`, `Haumea 28.2 deg`) and pole azimuth, naturally foreshortening with camera pitch and attenuating segments inside the planet's anti-sunward shadow cylinder.
- **Uniform Eccentric-Anomaly Orbit Sampling**: High-eccentricity orbits (`Comet` or `e > 0.35`) sample `192` uniform steps in eccentric anomaly `E in [0, 2*pi]`, preserving perihelion curvature down to `q = a * (1 - e)` (`0.587 AU` for Halley, `0.914 AU` for Hale–Bopp).
- **Porkchop Contours & Colormap**: Uses an 11-stop perceptually uniform Viridis LUT and 2D marching-squares iso-`Delta_v` contour extraction over the memoized Lambert grid.

---

## 8. Primary References

1. Bate, R. R., Mueller, D. D., & White, J. E. (1971). *Fundamentals of Astrodynamics*. Dover Publications (Universal-variable Kepler & Lambert formulations).
2. Vallado, D. A. (2013). *Fundamentals of Astrodynamics and Applications* (4th ed.). Microcosm Press.
3. Richardson, D. L. (1980). "Analytic construction of periodic orbits about the collinear points." *Celestial Mechanics*, 22(3), 241–253.
4. Murray, C. D., & Dermott, S. F. (1999). *Solar System Dynamics*. Cambridge University Press.
5. Brandt, J. C. (1968). "The Physics of Comet Tails." *Annual Review of Astronomy and Astrophysics*, 6, 267–286.
6. NASA NSSDCA Planetary Fact Sheets & JPL Horizons / SBDB Ephemeris Services ([nssdc.gsfc.nasa.gov/planetary/factsheet/](https://nssdc.gsfc.nasa.gov/planetary/factsheet/)).
7. ESA / JAXA *Comet Interceptor* & NASA *JWST / Roman* Sun–Earth L2 Mission Reference Reports.
