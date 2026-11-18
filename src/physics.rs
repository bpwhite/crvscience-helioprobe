// Copyright (C) 2026 CRV Science Network (https://crvscience.com)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://gnu.org>.

//! Keplerian 3D orbital mechanics, Hohmann & relativistic transfer calculations,
//! logarithmic 3D visual coordinate mapping, and perspective ray-picking math.

use crate::catalog::{BodyCategory, CelestialBody};
use std::f64::consts::{PI, TAU};

pub const AU_KM: f64 = 149_597_870.7;
pub const GM_SUN_KM3_S2: f64 = 132_712_440_018.0;
pub const GM_EARTH_KM3_S2: f64 = 398_600.4418;
pub const EARTH_RADIUS_KM: f64 = 6_371.0;
pub const LEO_ALTITUDE_KM: f64 = 400.0;
pub const SPEED_OF_LIGHT_KM_S: f64 = 299_792.458;

#[derive(Debug, Clone, Copy, Default)]
pub struct Vec3d {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3d {
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }

    pub fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }

    pub fn scale(self, s: f64) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    pub fn dot(self, rhs: Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    pub fn length(self) -> f64 {
        self.dot(self).sqrt()
    }

    pub fn normalized(self) -> Self {
        let len = self.length();
        if len > 1e-12 {
            self.scale(1.0 / len)
        } else {
            Self::new(0.0, 1.0, 0.0)
        }
    }

    pub fn lerp(self, target: Self, t: f64) -> Self {
        self.scale(1.0 - t).add(target.scale(t))
    }
}

/// Solves Kepler's Equation `M = E - e * sin(E)` via Newton-Raphson iteration.
pub fn solve_kepler_eccentric_anomaly(mean_anomaly_rad: f64, eccentricity: f64) -> f64 {
    let e = eccentricity.clamp(0.0, 0.999_995);
    let m = mean_anomaly_rad.rem_euclid(TAU);
    if m.abs() < 1e-14 {
        return 0.0;
    }
    let mut lo = 0.0_f64;
    let mut hi = TAU;
    let mut ecc = if e > 0.8 { PI } else { m };
    for _ in 0..36 {
        let f = ecc - e * ecc.sin() - m;
        if f.abs() < 1e-12 {
            break;
        }
        if f < 0.0 {
            lo = ecc;
        } else {
            hi = ecc;
        }
        let fp = (1.0 - e * ecc.cos()).max(1e-12);
        let cand = ecc - f / fp;
        ecc = if cand > lo && cand < hi {
            cand
        } else {
            0.5 * (lo + hi)
        };
    }
    ecc
}

/// Computes physical J2000 3D ecliptic position `(x, z_ecliptic_y, y_up)` in AU at `days_since_j2000`.
/// Note: `y` is vertical above the ecliptic plane so 3D camera up-axis is `+Y`.
pub fn kepler_position_au(
    semi_major_axis_au: f64,
    eccentricity: f64,
    inclination_deg: f64,
    lon_asc_node_deg: f64,
    arg_periapsis_deg: f64,
    mean_anomaly_j2000_deg: f64,
    orbital_period_days: f64,
    days_since_j2000: f64,
) -> Vec3d {
    if semi_major_axis_au <= 0.0 {
        return Vec3d::default();
    }
    let mean_motion = if orbital_period_days > 1e-6 {
        TAU / orbital_period_days
    } else {
        0.0
    };
    let m = mean_anomaly_j2000_deg.to_radians() + mean_motion * days_since_j2000;
    let ecc = solve_kepler_eccentric_anomaly(m, eccentricity);
    let true_anom = 2.0
        * ((1.0 + eccentricity).sqrt() * (ecc * 0.5).sin())
            .atan2((1.0 - eccentricity).sqrt() * (ecc * 0.5).cos());
    let r = semi_major_axis_au * (1.0 - eccentricity * ecc.cos());

    let inc = inclination_deg.to_radians();
    let om = lon_asc_node_deg.to_radians();
    let w = arg_periapsis_deg.to_radians();
    let u = w + true_anom;

    let x_ecl = r * (om.cos() * u.cos() - om.sin() * u.sin() * inc.cos());
    let y_ecl = r * (om.sin() * u.cos() + om.cos() * u.sin() * inc.cos());
    let z_ecl = r * (u.sin() * inc.sin());

    // Map ecliptic (x_ecl, y_ecl, z_ecl) -> 3D scene (x, y_up = z_ecl, z = y_ecl)
    Vec3d::new(x_ecl, z_ecl, y_ecl)
}

/// Evaluates Stumpff functions `C(z)` and `S(z)` with Taylor series continuity near `z = 0`
/// and hyperbolic overflow protection for large negative `z`.
pub fn stumpff_c_s(z: f64) -> (f64, f64) {
    if !z.is_finite() {
        return (0.5, 1.0 / 6.0);
    }
    if z > 1e-6 {
        let s = z.sqrt();
        let c_val = (1.0 - s.cos()) / z;
        let s_val = (s - s.sin()) / (z * s);
        (c_val, s_val)
    } else if z < -1e-6 {
        let s = (-z).sqrt().min(700.0);
        let denom = s * s;
        let c_val = (s.cosh() - 1.0) / denom;
        let s_val = (s.sinh() - s) / (denom * s);
        (c_val, s_val)
    } else {
        let c_val = 0.5 - z / 24.0 + (z * z) / 720.0 - (z * z * z) / 40_320.0;
        let s_val = 1.0 / 6.0 - z / 120.0 + (z * z) / 5_040.0 - (z * z * z) / 362_880.0;
        (c_val, s_val)
    }
}

/// Computes right-handed J2000 ecliptic heliocentric state `(r_km, v_km_s)` at `days_since_j2000`.
/// Preserves right-handedness `(r x v).z > 0` for prograde Solar System orbits (`i < 90 deg`).
pub fn kepler_state_ecliptic_km(
    body: &CelestialBody,
    days_since_j2000: f64,
    mean_anom_offset_deg: f64,
) -> (Vec3d, Vec3d) {
    if body.semi_major_axis_au <= 0.0 {
        return (Vec3d::default(), Vec3d::default());
    }
    let a = body.semi_major_axis_au.max(0.01) * AU_KM;
    let e = body.eccentricity.clamp(0.0, 0.999_995);
    let p = (a * (1.0 - e * e)).max(1.0);
    let mean_motion_rad_s = if body.orbital_period_days > 1e-6 {
        TAU / (body.orbital_period_days * 86_400.0)
    } else {
        (GM_SUN_KM3_S2 / (a * a * a)).sqrt()
    };
    let m = (body.mean_anomaly_j2000_deg + mean_anom_offset_deg).to_radians()
        + mean_motion_rad_s * days_since_j2000 * 86_400.0;
    let ecc = solve_kepler_eccentric_anomaly(m, e);
    let nu = 2.0
        * ((1.0 + e).sqrt() * (ecc * 0.5).sin())
            .atan2((1.0 - e).sqrt() * (ecc * 0.5).cos());
    let r = a * (1.0 - e * ecc.cos());

    let r_pf_x = r * nu.cos();
    let r_pf_y = r * nu.sin();
    let v_fac = (GM_SUN_KM3_S2 / p).sqrt();
    let v_pf_x = -v_fac * nu.sin();
    let v_pf_y = v_fac * (e + nu.cos());

    let inc = body.inclination_deg.to_radians();
    let om = body.lon_ascending_node_deg.to_radians();
    let w = body.arg_periapsis_deg.to_radians();

    let cos_om = om.cos();
    let sin_om = om.sin();
    let cos_w = w.cos();
    let sin_w = w.sin();
    let cos_i = inc.cos();
    let sin_i = inc.sin();

    // Perifocal basis vectors P_hat, Q_hat in right-handed J2000 ecliptic coordinates
    let px = cos_om * cos_w - sin_om * sin_w * cos_i;
    let py = sin_om * cos_w + cos_om * sin_w * cos_i;
    let pz = sin_w * sin_i;

    let qx = -cos_om * sin_w - sin_om * cos_w * cos_i;
    let qy = -sin_om * sin_w + cos_om * cos_w * cos_i;
    let qz = cos_w * sin_i;

    let r_ecl = Vec3d::new(
        px * r_pf_x + qx * r_pf_y,
        py * r_pf_x + qy * r_pf_y,
        pz * r_pf_x + qz * r_pf_y,
    );
    let v_ecl = Vec3d::new(
        px * v_pf_x + qx * v_pf_y,
        py * v_pf_x + qy * v_pf_y,
        pz * v_pf_x + qz * v_pf_y,
    );
    (r_ecl, v_ecl)
}

/// Propagates a 3D Keplerian state `(r0_km, v0_km_s)` forward by `dt_sec` under `mu` via universal anomaly.
pub fn propagate_kepler_ecliptic_km(
    r0: Vec3d,
    v0: Vec3d,
    dt_sec: f64,
    mu: f64,
) -> (Vec3d, Vec3d) {
    if dt_sec.abs() < 1e-9 || mu <= 0.0 {
        return (r0, v0);
    }
    let r0_mag = r0.length().max(1.0);
    let v0_mag2 = v0.dot(v0);
    let r0_dot_v0 = r0.dot(v0);
    let vr0 = r0_dot_v0 / r0_mag;
    let alpha = 2.0 / r0_mag - v0_mag2 / mu;
    let sqrt_mu = mu.sqrt();

    let max_chi = if alpha < -1e-12 {
        600.0 / (-alpha).sqrt()
    } else {
        f64::INFINITY
    };

    let mut chi = if alpha > 1e-12 {
        sqrt_mu * dt_sec * alpha
    } else if alpha < -1e-12 {
        // Bate-Mueller-White Eq. (4.5-10) logarithmic hyperbolic initial guess
        let a = 1.0 / alpha;
        let sign = dt_sec.signum();
        let denom = r0_dot_v0 + sign * (-mu * a).sqrt() * (1.0 - r0_mag * alpha);
        let ratio = if denom.abs() > 1e-12 {
            (-2.0 * mu * alpha * dt_sec / denom).abs().max(1e-12)
        } else {
            1.0
        };
        let log_guess = sign * (-a).sqrt() * ratio.ln();
        if ratio > 1.5 && log_guess.is_finite() {
            log_guess
        } else {
            sqrt_mu * dt_sec / r0_mag
        }
    } else {
        sqrt_mu * dt_sec / r0_mag
    };

    if max_chi.is_finite() {
        chi = chi.clamp(-max_chi, max_chi);
    }

    // f(chi) is strictly monotonically increasing with f(0) = -sqrt_mu * dt_sec.
    // Keep the outer bound open (+/-INFINITY) until an actual evaluation establishes f > 0 / f < 0
    // so bisection never jumps to max_chi/2 (cosh(300) overflow).
    let (mut lo, mut hi) = if dt_sec > 0.0 {
        (0.0_f64, f64::INFINITY)
    } else {
        (f64::NEG_INFINITY, 0.0_f64)
    };
    let max_hyp_step = if alpha < -1e-12 {
        4.0 / (-alpha).sqrt()
    } else {
        f64::INFINITY
    };
    let f_tol = 1e-12 * (sqrt_mu * dt_sec.abs()).max(r0_mag);

    for _ in 0..80 {
        let z = alpha * chi * chi;
        let (c, s) = stumpff_c_s(z);
        let f_chi = (r0_mag * vr0 / sqrt_mu) * chi * chi * c
            + (1.0 - alpha * r0_mag) * chi * chi * chi * s
            + r0_mag * chi
            - sqrt_mu * dt_sec;
        let df_chi = (r0_mag * vr0 / sqrt_mu) * chi * (1.0 - z * s)
            + (1.0 - alpha * r0_mag) * chi * chi * c
            + r0_mag;
        let d2f_chi = (r0_mag * vr0 / sqrt_mu) * (1.0 - z * c)
            + (1.0 - alpha * r0_mag) * chi * (1.0 - z * s);

        if !f_chi.is_finite() || !df_chi.is_finite() || !d2f_chi.is_finite() {
            chi = if lo.is_finite() && hi.is_finite() {
                0.5 * (lo + hi)
            } else {
                chi * 0.5
            };
            continue;
        }

        if f_chi < 0.0 {
            if chi > lo {
                lo = chi;
            }
        } else if chi < hi {
            hi = chi;
        }

        if f_chi.abs() <= f_tol
            || (lo.is_finite() && hi.is_finite() && (hi - lo).abs() <= 1e-13 * chi.abs().max(1.0))
        {
            break;
        }

        // Conway's Laguerre universal-Kepler step (n = 5)
        let n = 5.0_f64;
        let disc = ((n - 1.0) * (n - 1.0) * df_chi * df_chi - n * (n - 1.0) * f_chi * d2f_chi).abs();
        let denom = df_chi + df_chi.signum() * disc.sqrt();
        let raw_step = if denom.abs() > 1e-18 {
            n * f_chi / denom
        } else if df_chi.abs() > 1e-12 {
            f_chi / df_chi
        } else {
            f_chi.signum() * chi.abs().max(1.0) * 0.25
        };
        let step = if (!lo.is_finite() || !hi.is_finite()) && max_hyp_step.is_finite() {
            raw_step.clamp(-max_hyp_step, max_hyp_step)
        } else {
            raw_step
        };
        let cand = chi - step;

        let in_bounds = cand.is_finite() && cand > lo && cand < hi && cand.abs() <= max_chi;
        chi = if in_bounds {
            cand
        } else if lo.is_finite() && hi.is_finite() {
            0.5 * (lo + hi)
        } else if hi.is_finite() {
            (chi - step.abs().min(max_hyp_step).max(1.0)).max(-max_chi)
        } else {
            (chi + step.abs().min(max_hyp_step).max(1.0)).min(max_chi)
        };
    }

    let z = alpha * chi * chi;
    let (c, s) = stumpff_c_s(z);
    let f = 1.0 - (chi * chi / r0_mag) * c;
    let g = dt_sec - (chi * chi * chi / sqrt_mu) * s;
    let r = r0.scale(f).add(v0.scale(g));
    let r_mag = r.length().max(1.0);
    let f_dot = (sqrt_mu / (r_mag * r0_mag)) * (z * s - 1.0) * chi;
    let g_dot = 1.0 - (chi * chi / r_mag) * c;
    let v = r0.scale(f_dot).add(v0.scale(g_dot));
    (r, v)
}

/// Compresses astronomical distances spanning 0.38 AU to 7,500 ly smoothly into interactive 3D world units
/// while preserving orbital geometry, inclination, and true angular phase.
pub fn compress_au_to_scene(pos_au: Vec3d) -> Vec3d {
    let r = pos_au.length();
    if r < 1e-9 {
        return Vec3d::default();
    }
    // Hybrid logarithmic + root scale: 1 AU -> ~18.5 units, 5.2 AU -> ~42 units, 39.5 AU -> ~96 units
    let scene_r = 16.0 * (1.0 + r * 1.15).ln() + 4.2 * r.sqrt();
    pos_au.scale(scene_r / r)
}

/// Computes the 3D scene position of any catalog body (resolving parent planet offsets for moons).
pub fn body_scene_position(
    body: &CelestialBody,
    catalog: &[CelestialBody],
    days_since_j2000: f64,
) -> Vec3d {
    match body.category {
        BodyCategory::Star => Vec3d::default(),
        BodyCategory::Moon => {
            let parent_pos = body
                .parent_id
                .as_deref()
                .and_then(|pid| catalog.iter().find(|b| b.id == pid))
                .map(|p| body_scene_position(p, catalog, days_since_j2000))
                .unwrap_or_default();
            // Exaggerate local moon orbital clearance slightly so moons orbit visibly outside parent sphere
            let local_au = kepler_position_au(
                1.0,
                body.eccentricity,
                body.inclination_deg,
                body.lon_ascending_node_deg,
                body.arg_periapsis_deg,
                body.mean_anomaly_j2000_deg,
                body.orbital_period_days,
                days_since_j2000,
            );
            let orbit_visual_radius = match body.id.as_str() {
                "phobos" => 1.15,
                "charon" => 1.25,
                "moon" => 1.48,
                "triton" => 4.15,
                "io" => 5.10,
                "europa" => 6.30,
                "enceladus" => 7.10,
                "ganymede" => 7.70,
                "titan" => 9.20,
                "callisto" => 9.35,
                _ => 3.80,
            };
            parent_pos.add(local_au.scale(orbit_visual_radius))
        }
        BodyCategory::Exoplanet | BodyCategory::Nebula => {
            // Place interstellar targets on an outer celestial sphere shell preserving inclination & RA/node
            let dir = kepler_position_au(
                1.0,
                0.0,
                body.inclination_deg,
                body.lon_ascending_node_deg,
                body.arg_periapsis_deg,
                body.mean_anomaly_j2000_deg,
                0.0,
                0.0,
            )
            .normalized();
            let shell_r = if body.category == BodyCategory::Exoplanet {
                122.0 + (1.0 + body.distance_ly).ln() * 6.5
            } else {
                148.0 + (1.0 + body.distance_ly / 500.0).ln() * 14.0
            };
            dir.scale(shell_r)
        }
        BodyCategory::LagrangePoint => {
            let earth_scene = catalog
                .iter()
                .find(|b| b.id == "earth")
                .map(|e| body_scene_position(e, catalog, days_since_j2000))
                .unwrap_or(Vec3d::new(16.45, 0.0, 0.0));
            crate::l2_comets::sun_earth_l2_center_scene(earth_scene)
        }
        _ => {
            let pos_au = kepler_position_au(
                body.semi_major_axis_au,
                body.eccentricity,
                body.inclination_deg,
                body.lon_ascending_node_deg,
                body.arg_periapsis_deg,
                body.mean_anomaly_j2000_deg,
                body.orbital_period_days,
                days_since_j2000,
            );
            compress_au_to_scene(pos_au)
        }
    }
}

/// Evaluates `body_scene_position` while honoring any user-triggered comet perihelion mean-anomaly offsets.
pub fn body_scene_position_with_offsets(
    body: &CelestialBody,
    catalog: &[CelestialBody],
    days_since_j2000: f64,
    comet_offsets: &std::collections::BTreeMap<String, f64>,
) -> Vec3d {
    if body.category == BodyCategory::Comet {
        if let Some(&off) = comet_offsets.get(&body.id) {
            let mut shifted = body.clone();
            shifted.mean_anomaly_j2000_deg += off;
            return body_scene_position(&shifted, catalog, days_since_j2000);
        }
    }
    body_scene_position(body, catalog, days_since_j2000)
}

#[derive(Debug, Clone, Copy)]
pub struct TransferSolution {
    /// Total mission delta-v in `km/s` from 400 km Low Earth Orbit.
    pub total_delta_v_km_s: f64,
    /// Characteristic departure energy `C3 = v_inf^2` in `km^2 / s^2`.
    pub c3_km2_s2: f64,
    /// Physical Hohmann or relativistic transit time in Earth days.
    pub physical_flight_days: f64,
    /// Target circular capture orbit velocity in `km/s`.
    pub capture_orbit_vel_km_s: f64,
    /// Target circular capture altitude above surface in `km`.
    pub capture_altitude_km: f64,
    /// Lorentz factor `gamma` (`1.0` for non-relativistic Solar System flights).
    pub lorentz_gamma: f64,
}

/// Computes scientifically grounded Hohmann + patched-conic departure/capture metrics from 400 km LEO,
/// or relativistic laser-sail/fusion cruise metrics for interstellar exoplanets and nebulae.
pub fn compute_transfer_from_earth(
    target: &CelestialBody,
    catalog: &[CelestialBody],
    cruise_beta_c: f64,
) -> TransferSolution {
    let r_leo_km = EARTH_RADIUS_KM + LEO_ALTITUDE_KM;
    let v_leo = (GM_EARTH_KM3_S2 / r_leo_km).sqrt(); // ~7.67 km/s

    if target.distance_ly > 0.0 {
        let beta = cruise_beta_c.clamp(0.05, 0.35);
        let gamma = 1.0 / (1.0 - beta * beta).sqrt();
        let flight_years = target.distance_ly / beta;
        let dv_equiv = beta * SPEED_OF_LIGHT_KM_S;
        let (cap_alt, cap_v) = if target.category == BodyCategory::Exoplanet {
            let r_orb = target.radius_km + 500.0;
            (500.0, (target.gm_km3_s2() / r_orb).sqrt())
        } else {
            // Deep-sky nebula Lagrange/halo observation station
            (0.0, 0.12)
        };
        return TransferSolution {
            total_delta_v_km_s: dv_equiv,
            c3_km2_s2: 185.0,
            physical_flight_days: flight_years * 365.25,
            capture_orbit_vel_km_s: cap_v,
            capture_altitude_km: cap_alt,
            lorentz_gamma: gamma,
        };
    }

    if target.id == "earth" {
        return TransferSolution {
            total_delta_v_km_s: 0.15,
            c3_km2_s2: 0.0,
            physical_flight_days: 0.06,
            capture_orbit_vel_km_s: v_leo,
            capture_altitude_km: LEO_ALTITUDE_KM,
            lorentz_gamma: 1.0,
        };
    }

    if target.id == "moon" {
        // Classic trans-lunar injection (TLI ~3.13 km/s) + Lunar Orbit Insertion (~0.85 km/s)
        let r_lunar_orb = target.radius_km + 100.0;
        let v_lunar_orb = (target.gm_km3_s2() / r_lunar_orb).sqrt();
        return TransferSolution {
            total_delta_v_km_s: 3.98,
            c3_km2_s2: -1.9,
            physical_flight_days: 3.6,
            capture_orbit_vel_km_s: v_lunar_orb,
            capture_altitude_km: 100.0,
            lorentz_gamma: 1.0,
        };
    }

    if target.id == "sun_earth_l2" || target.category == BodyCategory::LagrangePoint {
        // Direct Ariane/Vulcan weak-stability manifold transfer to Sun-Earth L2 (~1.5M km, ~29.5 days)
        return TransferSolution {
            total_delta_v_km_s: 3.20,
            c3_km2_s2: -0.62,
            physical_flight_days: 29.5,
            capture_orbit_vel_km_s: 0.28,
            capture_altitude_km: 500_000.0,
            lorentz_gamma: 1.0,
        };
    }

    // Heliocentric Hohmann transfer between r1 = 1.0 AU and r2 = target (or parent planet) semi-major axis
    let r1_km = AU_KM;
    let effective_a_au = if target.category == BodyCategory::Moon {
        target
            .parent_id
            .as_deref()
            .and_then(|pid| catalog.iter().find(|b| b.id == pid))
            .map(|p| p.semi_major_axis_au)
            .unwrap_or(5.204)
    } else if target.id == "sun" {
        0.08 // Parker-class perihelion solar probe
    } else {
        target.semi_major_axis_au.max(0.08)
    };
    let r2_km = effective_a_au * AU_KM;
    let a_trans = 0.5 * (r1_km + r2_km);

    let v_earth_helio = (GM_SUN_KM3_S2 / r1_km).sqrt();
    let v_trans_dep = (GM_SUN_KM3_S2 * (2.0 / r1_km - 1.0 / a_trans)).sqrt();
    let v_inf_dep = (v_trans_dep - v_earth_helio).abs();
    let c3 = v_inf_dep * v_inf_dep;
    let v_perigee = (v_inf_dep * v_inf_dep + 2.0 * GM_EARTH_KM3_S2 / r_leo_km).sqrt();
    let dv_dep = (v_perigee - v_leo).abs();

    let v_target_helio = (GM_SUN_KM3_S2 / r2_km).sqrt();
    let v_trans_arr = (GM_SUN_KM3_S2 * (2.0 / r2_km - 1.0 / a_trans)).sqrt();
    let v_inf_arr = (v_target_helio - v_trans_arr).abs();

    let cap_alt_km = if target.id == "sun" {
        6_100_000.0
    } else if target.radius_km < 10.0 {
        1.5 // close asteroid station-keeping
    } else if target.radius_km > 20_000.0 {
        4_500.0 // gas/ice giant cloud-top clearance
    } else {
        300.0
    };
    let r_cap_km = target.radius_km + cap_alt_km;
    let gm_target = target.gm_km3_s2();
    let v_cap_circ = (gm_target / r_cap_km.max(1.0)).sqrt();
    let v_periapsis_arr = (v_inf_arr * v_inf_arr + 2.0 * gm_target / r_cap_km.max(1.0)).sqrt();
    // Apply realistic Oberth/gravity-assist braking cap for outer giants
    let dv_arr = (v_periapsis_arr - v_cap_circ).abs().min(8.5);

    let flight_seconds = PI * (a_trans.powi(3) / GM_SUN_KM3_S2).sqrt();
    TransferSolution {
        total_delta_v_km_s: dv_dep + dv_arr,
        c3_km2_s2: c3,
        physical_flight_days: flight_seconds / 86_400.0,
        capture_orbit_vel_km_s: v_cap_circ,
        capture_altitude_km: cap_alt_km,
        lorentz_gamma: 1.0,
    }
}


/// Monotonic physical visual radius in 3D scene units preserving exact astronomical size ordering
/// (Ganymede > Titan > Mercury > Callisto > Io > Moon > Europa > Triton > Pluto > Charon > Enceladus > Phobos).
pub fn visual_body_radius_scene(body: &CelestialBody) -> f64 {
    match body.category {
        BodyCategory::Star => 4.2,
        BodyCategory::Nebula => 8.5,
        BodyCategory::LagrangePoint => 0.14,
        _ => (0.0012 * body.radius_km.max(0.2).powf(0.75)).clamp(0.035, 3.25),
    }
}

pub fn probe_orbit_radius_scene(target: &CelestialBody) -> f64 {
    match target.category {
        BodyCategory::Nebula => 7.2,
        BodyCategory::Star => 6.2,
        BodyCategory::LagrangePoint => 0.62,
        _ => {
            let rb = visual_body_radius_scene(target);
            if target.id == "saturn" {
                rb * 2.45
            } else if target.has_rings {
                rb * 1.88
            } else {
                rb * 1.65 + 0.28
            }
        }
    }
}
