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

//! Sun-Earth L2 Halo Space Telescopes & Dynamic Cometary Solar-Wind Ion / Dust Tail Physics (`src/l2_comets.rs`):
//! - Models the collinear Sun-Earth L2 Lagrange point (~1.5 million km / 0.0100 AU anti-sunward of Earth)
//!   with 3D Lissajous quasi-halo station-keeping for JWST, Nancy Grace Roman, and ESA/JAXA Comet Interceptor.
//! - Computes heliocentric sublimation activity, solar-wind aberrated Type I CO+/H2O+ ion tail vectors,
//!   and radiation-pressure synchrone/syndyne curved Type II silicate dust tail trajectories for eccentric comets.

use crate::catalog::CelestialBody;
use crate::physics::{
    kepler_state_ecliptic_km, Vec3d, AU_KM, GM_EARTH_KM3_S2, GM_SUN_KM3_S2,
};
use serde::{Deserialize, Serialize};
use std::f64::consts::TAU;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct L2ObservatoryDef {
    pub id: &'static str,
    pub name: &'static str,
    pub agency: &'static str,
    pub halo_phase_deg: f64,
    pub deploy_reward_sp: u64,
    pub role_headline: &'static str,
    pub role_detail: &'static str,
    pub source_url: &'static str,
}

pub const L2_OBSERVATORIES: [L2ObservatoryDef; 3] = [
    L2ObservatoryDef {
        id: "jwst_l2",
        name: "James Webb Space Telescope (JWST)",
        agency: "NASA / ESA / CSA",
        halo_phase_deg: 0.0,
        deploy_reward_sp: 160,
        role_headline: "6.5 m Gold-Beryllium IR Primary & 5-Layer Kapton Sunshield",
        role_detail: "Orbits Sun-Earth L2 (~1.5M km anti-sunward) at ~40 K; unlocks remote Optical/NIR and Thermal IR/Mass-Spec pre-scans of Comets, Exoplanets, and Nebulae.",
        source_url: "https://science.nasa.gov/mission/webb/orbit/",
    },
    L2ObservatoryDef {
        id: "roman_l2",
        name: "Nancy Grace Roman Space Telescope",
        agency: "NASA / GSFC / JPL",
        halo_phase_deg: 120.0,
        deploy_reward_sp: 175,
        role_headline: "Wide Field Instrument (100x Hubble IR FOV) & Coronagraph",
        role_detail: "Co-rotates in a Sun-Earth L2 quasi-halo orbit; unlocks the Wide-Field Sky Survey (+180 SP) and remote Optical/NIR astrometry across outer solar and exoplanet targets.",
        source_url: "https://science.nasa.gov/mission/roman-space-telescope/observatory-technical/",
    },
    L2ObservatoryDef {
        id: "interceptor_l2",
        name: "Comet Interceptor Standby Triad",
        agency: "ESA / JAXA",
        halo_phase_deg: 240.0,
        deploy_reward_sp: 150,
        role_headline: "L2 Halo Rapid-Response Mothership + Sub-Probes B1/B2",
        role_detail: "Loiters in low-fuel Sun-Earth L2 halo storage until a comet approaches perihelion, enabling 2x-speed direct L2 intercept launches (+90 SP coma tomography bonus).",
        source_url: "https://sci.esa.int/documents/34923/36148/CometInterceptor_CDF_Study_Report.pdf",
    },
];

#[derive(Debug, Clone, Copy)]
pub struct SunEarthL2Dynamics {
    /// First-order Sun-Earth Hill sphere radius `r_L2 = a_E * (mu_E / (3 * mu_sun))^(1/3)` in km.
    pub hill_radius_km: f64,
    /// Exact dimensionless collinear L2 distance `gamma = d_L2 / a_E` from the CR3BP Euler quintic.
    pub collinear_gamma: f64,
    /// Exact collinear Sun-Earth L2 distance from Earth in km.
    pub l2_distance_km: f64,
    /// CR3BP quadratic potential coefficient `c2` at Sun-Earth L2.
    pub c2: f64,
    /// Linearized CR3BP in-plane halo/Lyapunov angular frequency `omega_xy` in `rad/day`.
    pub omega_xy_rad_day: f64,
    /// Linearized CR3BP out-of-plane vertical angular frequency `omega_z` in `rad/day`.
    pub omega_z_rad_day: f64,
    /// Linearized in-plane halo period `2*pi / omega_xy` in days.
    pub period_xy_days: f64,
    /// Linearized out-of-plane vertical period `2*pi / omega_z` in days.
    pub period_z_days: f64,
    /// Linearized CR3BP radial-to-tangential excursion amplitude ratio `A_x / A_y`.
    pub radial_to_tangential: f64,
}

/// Computes the physical Sun-Earth Hill radius `r_L2 = a_E * (mu_E / (3 * mu_sun))^(1/3)`,
/// exact CR3BP collinear quintic distance, and linearized CR3BP halo/vertical frequencies `(omega_xy, omega_z)`.
pub fn compute_sun_earth_l2_dynamics() -> SunEarthL2Dynamics {
    let a_e = AU_KM;
    let mu_e = GM_EARTH_KM3_S2;
    let mu_s = GM_SUN_KM3_S2;
    let hill_radius_km = a_e * (mu_e / (3.0 * mu_s)).cbrt();

    let mu = mu_e / (mu_s + mu_e);
    let mut gamma = (mu / 3.0).cbrt();
    for _ in 0..16 {
        let one_plus = 1.0 + gamma;
        let f = (1.0 - mu + gamma) - (1.0 - mu) / (one_plus * one_plus) - mu / (gamma * gamma);
        let df = 1.0
            + 2.0 * (1.0 - mu) / (one_plus * one_plus * one_plus)
            + 2.0 * mu / (gamma * gamma * gamma);
        let step = f / df;
        gamma -= step;
        if step.abs() < 1e-14 {
            break;
        }
    }

    let one_plus = 1.0 + gamma;
    let c2 = mu / (gamma * gamma * gamma) + (1.0 - mu) / (one_plus * one_plus * one_plus);
    let n_rad_day = ((mu_s + mu_e) / (a_e * a_e * a_e)).sqrt() * 86_400.0;

    let disc = (9.0 * c2 * c2 - 8.0 * c2).max(0.0).sqrt();
    let omega_xy_norm = ((2.0 - c2 + disc) * 0.5).max(0.0).sqrt();
    let omega_z_norm = c2.max(0.0).sqrt();

    let omega_xy_rad_day = n_rad_day * omega_xy_norm;
    let omega_z_rad_day = n_rad_day * omega_z_norm;
    let radial_to_tangential =
        (2.0 * omega_xy_norm) / (omega_xy_norm * omega_xy_norm + 1.0 + 2.0 * c2);

    SunEarthL2Dynamics {
        hill_radius_km,
        collinear_gamma: gamma,
        l2_distance_km: gamma * a_e,
        c2,
        omega_xy_rad_day,
        omega_z_rad_day,
        period_xy_days: TAU / omega_xy_rad_day,
        period_z_days: TAU / omega_z_rad_day,
        radial_to_tangential,
    }
}

/// Computes the 3D scene position of the Sun-Earth L2 Lagrange center on the anti-sunward
/// Sun->Earth axis (placed cleanly outside the Moon's 1.48-unit scene orbit and inside Mars).
pub fn sun_earth_l2_center_scene(earth_scene: Vec3d) -> Vec3d {
    let r = earth_scene.length();
    if r < 1e-6 {
        return Vec3d::new(20.7, 0.0, 0.0);
    }
    let anti_sun = earth_scene.normalized();
    earth_scene.add(anti_sun.scale(4.25))
}

/// Evaluates a 3D Lissajous quasi-halo station-keeping position around Sun-Earth L2
/// driven by linearized CR3BP frequencies `(omega_xy, omega_z)` (~177.6 d / ~184.0 d).
pub fn sun_earth_l2_halo_position(
    earth_scene: Vec3d,
    days_since_j2000: f64,
    phase_deg: f64,
    halo_radius_scene: f64,
) -> Vec3d {
    let center = sun_earth_l2_center_scene(earth_scene);
    let radial = if earth_scene.length() > 1e-6 {
        earth_scene.normalized()
    } else {
        Vec3d::new(1.0, 0.0, 0.0)
    };
    let up = Vec3d::new(0.0, 1.0, 0.0);
    let tang = radial.cross(up).normalized();

    let dyn_l2 = compute_sun_earth_l2_dynamics();
    let phi = phase_deg.to_radians() + dyn_l2.omega_xy_rad_day * days_since_j2000;
    let phi_z = phase_deg.to_radians() + dyn_l2.omega_z_rad_day * days_since_j2000;

    center
        .add(tang.scale(halo_radius_scene * phi.cos()))
        .add(up.scale(halo_radius_scene * 0.72 * phi_z.sin()))
        .add(radial.scale(halo_radius_scene * dyn_l2.radial_to_tangential * phi.sin()))
}

#[derive(Debug, Clone, Copy)]
pub struct CometTailState {
    pub r_helio_au: f64,
    pub orbital_speed_km_s: f64,
    pub v_radial_km_s: f64,
    pub v_perp_km_s: f64,
    pub solar_wind_km_s: f64,
    pub ion_aberration_deg: f64,
    /// Normalized coma/tail sublimation intensity (`0.0..=2.8`).
    pub activity: f64,
    /// Estimated H2O + CO2 coma outgassing production rate `Q` in molecules/s.
    pub gas_production_q_s: f64,
    /// Unit vector in 3D scene space pointing strictly along the solar-wind aberrated Type I CO+ ion tail.
    pub ion_tail_dir_scene: Vec3d,
    /// Unit vector in 3D scene space pointing backward along `-v_orb` for Type II dust-tail synchrone curvature.
    pub dust_lag_dir_scene: Vec3d,
    /// Strict anti-sunward unit vector (`+r_hat`) in 3D scene space.
    pub anti_sun_dir_scene: Vec3d,
    pub ion_tail_len_scene: f64,
    pub dust_tail_len_scene: f64,
    pub coma_radius_scene: f64,
}

/// Computes physically grounded cometary sublimation, exact ecliptic orbital velocity `v_hat`,
/// radial/perpendicular decomposition, solar-wind aberrated ion tail direction (`v_sw * r_hat - v_orb`),
/// and radiation-pressure curved dust tail geometry.
pub fn compute_comet_tail_state(
    comet: &CelestialBody,
    days_since_j2000: f64,
    perihelion_offset_deg: f64,
) -> CometTailState {
    let (r_ecl_km, v_ecl_km_s) =
        kepler_state_ecliptic_km(comet, days_since_j2000, perihelion_offset_deg);
    // Map J2000 ecliptic (x_ecl, y_ecl, z_ecl) -> 3D scene basis (x, y_up = z_ecl, z = y_ecl)
    let r_phys = Vec3d::new(r_ecl_km.x, r_ecl_km.z, r_ecl_km.y);
    let v_phys = Vec3d::new(v_ecl_km_s.x, v_ecl_km_s.z, v_ecl_km_s.y);

    let r_km = r_phys.length().max(0.08 * AU_KM);
    let r_au = r_km / AU_KM;
    let v_orb_km_s = v_phys.length();

    let anti_sun = if r_phys.length() > 1e-6 {
        r_phys.normalized()
    } else {
        Vec3d::new(1.0, 0.0, 0.0)
    };
    let v_hat = if v_orb_km_s > 1e-9 {
        v_phys.scale(1.0 / v_orb_km_s)
    } else {
        anti_sun.cross(Vec3d::new(0.0, 1.0, 0.0)).normalized()
    };

    // Radial solar wind ~420 km/s deflects the Type I plasma tail by alpha = atan2(v_perp, v_sw - v_r)
    let solar_wind_km_s = 420.0;
    let v_radial_km_s = v_phys.dot(anti_sun);
    let v_perp_vec = v_phys.sub(anti_sun.scale(v_radial_km_s));
    let v_perp_km_s = v_perp_vec.length();
    let ion_aberration_deg = v_perp_km_s
        .atan2((solar_wind_km_s - v_radial_km_s).max(1e-3))
        .to_degrees();
    let ion_dir = anti_sun
        .scale(solar_wind_km_s)
        .sub(v_phys)
        .normalized();
    let dust_lag = v_hat.scale(-1.0);

    // Sublimation scale: strong inside 2.5 AU, tapering smoothly toward dormancy in the outer solar system
    let size_boost = (comet.radius_km.clamp(1.0, 35.0) / 3.5).powf(0.28);
    let raw_insolation = (1.85 / r_au.max(0.24)).powf(1.68) * size_boost;
    let cold_cutoff = if r_au > 5.5 {
        (5.5 / r_au).powi(3)
    } else {
        1.0
    };
    let activity = (raw_insolation * cold_cutoff).clamp(0.015, 2.8);
    let gas_q = 1.2e28 * activity * (comet.radius_km.max(0.8) / 2.5).powf(1.3);

    CometTailState {
        r_helio_au: r_au,
        orbital_speed_km_s: v_orb_km_s,
        v_radial_km_s,
        v_perp_km_s,
        solar_wind_km_s,
        ion_aberration_deg,
        activity,
        gas_production_q_s: gas_q,
        ion_tail_dir_scene: ion_dir,
        dust_lag_dir_scene: dust_lag,
        anti_sun_dir_scene: anti_sun,
        ion_tail_len_scene: (activity * 5.8).clamp(0.0, 14.5),
        dust_tail_len_scene: (activity * 4.4).clamp(0.0, 11.2),
        coma_radius_scene: (0.14 + activity * 0.42).clamp(0.08, 1.35),
    }
}

/// Evaluates a point along a curved Type II cometary silicate dust tail stream (`s` in `0.0..=1.0`, `fan_offset` in `-1.0..=1.0`).
pub fn sample_dust_tail_point(
    comet_scene: Vec3d,
    tail: &CometTailState,
    s: f64,
    fan_offset: f64,
) -> Vec3d {
    let t = s.clamp(0.0, 1.0);
    let normal = tail
        .anti_sun_dir_scene
        .cross(tail.dust_lag_dir_scene)
        .normalized();
    let radial_push = tail.anti_sun_dir_scene.scale(tail.dust_tail_len_scene * t);
    let orbital_lag = tail
        .dust_lag_dir_scene
        .scale(tail.dust_tail_len_scene * 0.64 * t.powf(1.55));
    let fan_spread = tail
        .dust_lag_dir_scene
        .scale(fan_offset * tail.dust_tail_len_scene * 0.16 * t)
        .add(normal.scale(fan_offset * tail.dust_tail_len_scene * 0.06 * t));
    comet_scene.add(radial_push).add(orbital_lag).add(fan_spread)
}
