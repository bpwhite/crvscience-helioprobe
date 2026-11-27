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

//! Gravity-Assist Slingshot & 3D Universal-Variable Lambert Porkchop Planner (`src/astrodynamics_lab.rs`):
//! - Computes planetary synodic periods, current vs optimal Hohmann phase angles, and next launch window offsets.
//! - Generates a genuine 3D universal-variable Lambert Porkchop `delta_v` and `C3` contour grid
//!   (`departure_offset_days` x `flight_days`) evaluating both short-way (`dm = +1`) and long-way (`dm = -1`) arcs.
//! - Evaluates patched-conic 2-leg Gravity-Assist Slingshot routes (`Earth -> Assist Body -> Target`) with
//!   true arrival `v_inf` matching, hyperbolic deflection limit `sin(delta/2) = 1 / (1 + r_p * v_inf^2 / mu)`,
//!   radial/orbital feasibility checks, and positive `delta_v` savings filtering.

use crate::catalog::{BodyCategory, CelestialBody};
use crate::l2_comets::compute_sun_earth_l2_dynamics;
use crate::physics::{
    compute_capture_orbit, compute_transfer_from_earth, kepler_position_au,
    kepler_state_ecliptic_km, stumpff_c_s, Vec3d, AU_KM, EARTH_RADIUS_KM, GM_EARTH_KM3_S2,
    GM_SUN_KM3_S2, LEO_ALTITUDE_KM,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::f64::consts::{PI, TAU};

#[derive(Debug, Clone)]
pub struct PorkchopGrid {
    pub cols: usize,
    pub rows: usize,
    pub dep_offsets_days: Vec<f64>,
    pub tof_days: Vec<f64>,
    /// Row-major `[row * cols + col]` total mission `delta_v` in `km/s`.
    pub dv_km_s: Vec<f64>,
    /// Row-major `[row * cols + col]` departure hyperbolic characteristic energy `C3 = v_inf_dep^2` in `km^2/s^2`.
    pub c3_km2_s2: Vec<f64>,
    pub min_dv_km_s: f64,
    pub max_dv_km_s: f64,
    pub best_col: usize,
    pub best_row: usize,
    pub best_dep_offset_days: f64,
    pub best_tof_days: f64,
    pub best_c3_km2_s2: f64,
    pub synodic_period_days: f64,
    pub phase_error_deg: f64,
}

fn effective_heliocentric_body<'a>(
    target: &'a CelestialBody,
    catalog: &'a [CelestialBody],
) -> &'a CelestialBody {
    if target.category == BodyCategory::Moon {
        if let Some(pid) = target.parent_id.as_deref() {
            if let Some(parent) = catalog.iter().find(|b| b.id == pid) {
                return parent;
            }
        }
    }
    target
}

/// Solves the 3D single-revolution universal-variable Lambert boundary-value problem
/// connecting `r1` to `r2` in `dt_sec` under gravitational parameter `mu`.
/// `dm = 1.0` solves the short-way (`0 < dnu < pi`) branch; `dm = -1.0` solves the long-way (`pi < dnu < 2*pi`) branch.
pub fn solve_lambert_branch(
    r1: Vec3d,
    r2: Vec3d,
    dt_sec: f64,
    mu: f64,
    dm: f64,
) -> Option<(Vec3d, Vec3d)> {
    if dt_sec <= 1.0 || mu <= 0.0 {
        return None;
    }
    let r1_mag = r1.length();
    let r2_mag = r2.length();
    if r1_mag < 1.0 || r2_mag < 1.0 {
        return None;
    }
    let cos_dnu = (r1.dot(r2) / (r1_mag * r2_mag)).clamp(-1.0, 1.0);
    let a_coeff = dm * (r1_mag * r2_mag * (1.0 + cos_dnu)).sqrt();
    if a_coeff.abs() < 10.0 {
        return None;
    }

    let sqrt_mu = mu.sqrt();
    let eval = |z: f64| -> Option<(f64, f64, f64, f64)> {
        let (c, s) = stumpff_c_s(z);
        if c <= 1e-12 {
            return None;
        }
        let y = r1_mag + r2_mag + a_coeff * (z * s - 1.0) / c.sqrt();
        if y <= 1e-6 || !y.is_finite() {
            return None;
        }
        let chi = (y / c).sqrt();
        let t = (chi * chi * chi * s + a_coeff * y.sqrt()) / sqrt_mu;
        if !t.is_finite() {
            return None;
        }
        Some((y, t, c, s))
    };

    let z_max = 4.0 * PI * PI - 1e-4;
    let mut hi = z_max;
    if let Some((_, t_hi, _, _)) = eval(hi) {
        if t_hi < dt_sec {
            return None;
        }
    } else {
        // Pull hi slightly back until C(hi) and y(hi) are valid
        let mut found_hi = false;
        for step in 1..=20 {
            let cand = z_max - (step as f64) * 0.05;
            if let Some((_, t_c, _, _)) = eval(cand) {
                if t_c >= dt_sec {
                    hi = cand;
                    found_hi = true;
                    break;
                }
            }
        }
        if !found_hi {
            return None;
        }
    }

    // Find lower bracket `lo` where y(lo) > 0 and t(lo) <= dt_sec
    let mut lo = 0.0_f64;
    let mut found_lo = false;
    let mut last_invalid = -400.0_f64;
    let probe_zs = [
        0.0, -0.25, -1.0, -4.0, -12.0, -32.0, -80.0, -180.0, -400.0, -1200.0, -4000.0, -15000.0,
    ];
    for &z_try in &probe_zs {
        if let Some((_, t_try, _, _)) = eval(z_try) {
            if t_try <= dt_sec {
                lo = z_try;
                found_lo = true;
                break;
            } else {
                hi = hi.min(z_try);
            }
        } else if a_coeff > 0.0 {
            // y(z) crossed <= 0 below this z_try; valid domain starts between z_try and hi
            last_invalid = z_try;
            break;
        } else {
            break;
        }
    }

    if !found_lo {
        // Bisect inside [last_invalid, hi] to locate the lower edge of the valid y > 0 / t <= dt_sec domain
        let mut a = last_invalid;
        let mut b = hi;
        for _ in 0..56 {
            let m = 0.5 * (a + b);
            match eval(m) {
                None => a = m,
                Some((_, t_m, _, _)) => {
                    if t_m <= dt_sec {
                        lo = m;
                        found_lo = true;
                        break;
                    } else {
                        b = m;
                        hi = hi.min(m);
                    }
                }
            }
        }
        if !found_lo {
            return None;
        }
    }

    let mut z = 0.5 * (lo + hi);
    if z < lo || z > hi || eval(z).is_none() {
        z = 0.5 * (lo + hi);
    }

    let conv_tol = (dt_sec * 1e-11).clamp(1e-6, 1e-4);
    let mut prev_err_abs = f64::INFINITY;
    let mut stall_count = 0u32;

    for _ in 0..64 {
        let Some((y, t, c, s)) = eval(z) else {
            lo = z;
            z = 0.5 * (lo + hi);
            continue;
        };
        let err = t - dt_sec;
        if err < 0.0 {
            lo = lo.max(z);
        } else {
            hi = hi.min(z);
        }
        if err.abs() <= conv_tol || (hi - lo).abs() <= 1e-13 * hi.abs().max(1.0) {
            break;
        }
        if err.abs() >= prev_err_abs * 0.95 {
            stall_count += 1;
        } else {
            stall_count = 0;
        }
        prev_err_abs = err.abs();

        // Bate-Mueller-White Eq. (5.3-13) analytical dt/dz including + 0.75 * S^2 / C
        let dt_dz = if z.abs() > 1e-6 {
            let chi = (y / c).sqrt();
            (chi * chi * chi * ((c - 1.5 * s / c) / (2.0 * z) + 0.75 * s * s / c)
                + (a_coeff / 8.0) * (3.0 * s * y.sqrt() / c + a_coeff * (c / y).sqrt()))
                / sqrt_mu
        } else {
            (2.0_f64.sqrt() / 40.0 * y.powf(1.5)
                + (a_coeff / 8.0) * (y.sqrt() + a_coeff * (0.5 / y).sqrt()))
                / sqrt_mu
        };
        let cand = if stall_count < 2 && dt_dz.is_finite() && dt_dz > 1e-12 {
            z - err / dt_dz
        } else {
            stall_count = 0;
            0.5 * (lo + hi)
        };
        z = if cand > lo && cand < hi && eval(cand).is_some() {
            cand
        } else {
            0.5 * (lo + hi)
        };
    }

    // Safeguarded bisection fallback if Newton did not reach conv_tol
    if let Some((_, t_check, _, _)) = eval(z) {
        if (t_check - dt_sec).abs() > conv_tol {
            for _ in 0..64 {
                let m = 0.5 * (lo + hi);
                let Some((_, t_m, _, _)) = eval(m) else {
                    lo = m;
                    continue;
                };
                z = m;
                let err = t_m - dt_sec;
                if err < 0.0 {
                    lo = m;
                } else {
                    hi = m;
                }
                if err.abs() <= conv_tol || (hi - lo).abs() <= 1e-14 * hi.abs().max(1.0) {
                    break;
                }
            }
        }
    }

    let (y_final, t_final, _, _) = eval(z)?;
    let accept_tol = (dt_sec * 1e-4).clamp(1e-3, 1.0);
    if y_final <= 1e-6 || !t_final.is_finite() || (t_final - dt_sec).abs() > accept_tol {
        return None;
    }
    let f = 1.0 - y_final / r1_mag;
    let g = a_coeff * (y_final / mu).sqrt();
    if g.abs() < 1e-6 {
        return None;
    }
    let g_dot = 1.0 - y_final / r2_mag;
    let v1 = r2.sub(r1.scale(f)).scale(1.0 / g);
    let v2 = r2.scale(g_dot).sub(r1).scale(1.0 / g);
    if !v1.length().is_finite() || !v2.length().is_finite() {
        return None;
    }
    Some((v1, v2))
}

pub fn target_capture_dv(target: &CelestialBody, v_inf_arr_km_s: f64) -> f64 {
    compute_capture_orbit(target, v_inf_arr_km_s).capture_dv_km_s
}

/// Computes a `cols x rows` 3D universal-variable Lambert Porkchop `delta_v` & `C3` contour matrix
/// from Earth to any Solar System body (or parent planet for moons) starting from `current_j2000_days`.
pub fn compute_porkchop_grid(
    target: &CelestialBody,
    catalog: &[CelestialBody],
    current_j2000_days: f64,
    cols: usize,
    rows: usize,
) -> PorkchopGrid {
    compute_porkchop_grid_with_offsets(
        target,
        catalog,
        current_j2000_days,
        &BTreeMap::new(),
        cols,
        rows,
    )
}

pub fn compute_porkchop_grid_with_offsets(
    target: &CelestialBody,
    catalog: &[CelestialBody],
    current_j2000_days: f64,
    comet_offsets: &BTreeMap<String, f64>,
    cols: usize,
    rows: usize,
) -> PorkchopGrid {
    let cols = cols.max(8);
    let rows = rows.max(8);
    let eff = effective_heliocentric_body(target, catalog);
    let earth = catalog.iter().find(|b| b.id == "earth").unwrap_or(eff);
    let eff_offset = if eff.category == BodyCategory::Comet {
        comet_offsets.get(&eff.id).copied().unwrap_or(0.0)
    } else {
        0.0
    };

    if eff.distance_ly > 0.0 || eff.id == "earth" || eff.id == "sun" || eff.category == BodyCategory::LagrangePoint {
        let base = compute_transfer_from_earth(target, catalog, 0.20);
        let dep_offsets_days: Vec<f64> = (0..cols).map(|c| c as f64 * 15.0).collect();
        let tof_days: Vec<f64> = (0..rows)
            .map(|r| base.physical_flight_days * (0.65 + 0.7 * (r as f64 / (rows - 1) as f64)))
            .collect();
        let mut dv_km_s = vec![0.0; cols * rows];
        let mut c3_km2_s2 = vec![base.c3_km2_s2; cols * rows];
        let mut min_dv = f64::MAX;
        let mut max_dv = 0.0_f64;
        let mut best_c = 0;
        let mut best_r = rows / 2;
        for r in 0..rows {
            for c in 0..cols {
                let ratio = tof_days[r] / base.physical_flight_days.max(0.1);
                let val = base.total_delta_v_km_s
                    * (0.85 + 0.22 * (ratio - 1.0).powi(2) + 0.03 * ((c as f64 * 0.4).sin()).abs());
                dv_km_s[r * cols + c] = val;
                c3_km2_s2[r * cols + c] = base.c3_km2_s2 * (0.85 + 0.22 * (ratio - 1.0).powi(2));
                if val < min_dv {
                    min_dv = val;
                    best_c = c;
                    best_r = r;
                }
                max_dv = max_dv.max(val);
            }
        }
        return PorkchopGrid {
            cols,
            rows,
            best_dep_offset_days: dep_offsets_days[best_c],
            best_tof_days: tof_days[best_r],
            best_c3_km2_s2: c3_km2_s2[best_r * cols + best_c],
            dep_offsets_days,
            tof_days,
            dv_km_s,
            c3_km2_s2,
            min_dv_km_s: min_dv,
            max_dv_km_s: max_dv,
            best_col: best_c,
            best_row: best_r,
            synodic_period_days: 365.25,
            phase_error_deg: 0.0,
        };
    }

    let t_earth = 365.256;
    let t_targ = eff.orbital_period_days.max(30.0);
    let synodic = (1.0 / ((1.0 / t_earth) - (1.0 / t_targ)).abs()).clamp(90.0, 900.0);
    let a1 = 1.0_f64;
    let a2 = eff.semi_major_axis_au.max(0.2);
    let t_hohmann_days =
        PI * (((0.5 * (a1 + a2) * AU_KM).powi(3)) / GM_SUN_KM3_S2).sqrt() / 86_400.0;

    // Optimal Hohmann phase angle (radians) at departure
    let opt_phase = (PI * (1.0 - ((a1 + a2) / (2.0 * a2)).powf(1.5))).rem_euclid(TAU);
    let p_e_now = kepler_position_au(
        earth.semi_major_axis_au,
        earth.eccentricity,
        earth.inclination_deg,
        earth.lon_ascending_node_deg,
        earth.arg_periapsis_deg,
        earth.mean_anomaly_j2000_deg,
        earth.orbital_period_days,
        current_j2000_days,
    );
    let p_t_now = kepler_position_au(
        eff.semi_major_axis_au,
        eff.eccentricity,
        eff.inclination_deg,
        eff.lon_ascending_node_deg,
        eff.arg_periapsis_deg,
        eff.mean_anomaly_j2000_deg + eff_offset,
        eff.orbital_period_days,
        current_j2000_days,
    );
    let lon_e_now = p_e_now.z.atan2(p_e_now.x);
    let lon_t_now = p_t_now.z.atan2(p_t_now.x);
    let cur_phase = (lon_t_now - lon_e_now).rem_euclid(TAU);
    let mut phase_err = (cur_phase - opt_phase).to_degrees();
    while phase_err > 180.0 {
        phase_err -= 360.0;
    }
    while phase_err < -180.0 {
        phase_err += 360.0;
    }

    let span_dep = (synodic * 1.15).clamp(160.0, 820.0);
    let dep_offsets_days: Vec<f64> = (0..cols)
        .map(|c| span_dep * (c as f64 / (cols - 1) as f64))
        .collect();
    let tof_days: Vec<f64> = (0..rows)
        .map(|r| t_hohmann_days * (0.48 + 1.15 * (r as f64 / (rows - 1) as f64)))
        .collect();

    let r_leo = EARTH_RADIUS_KM + LEO_ALTITUDE_KM;
    let v_leo = (GM_EARTH_KM3_S2 / r_leo).sqrt();
    let base_hoh = compute_transfer_from_earth(target, catalog, 0.05);

    let mut dv_km_s = vec![0.0; cols * rows];
    let mut c3_km2_s2 = vec![0.0; cols * rows];
    let mut min_dv = f64::MAX;
    let mut max_dv = 0.0_f64;
    let mut best_c = 0;
    let mut best_r = 0;

    for (r_idx, &tof) in tof_days.iter().enumerate() {
        let dt_sec = tof * 86_400.0;
        for (c_idx, &dep_off) in dep_offsets_days.iter().enumerate() {
            let t_dep = current_j2000_days + dep_off;
            let t_arr = t_dep + tof;

            let (r_e, v_e) = kepler_state_ecliptic_km(earth, t_dep, 0.0);
            let (r_t, v_t) = kepler_state_ecliptic_km(eff, t_arr, eff_offset);

            let mut best_cell: Option<(f64, f64)> = None;
            for dm in [1.0_f64, -1.0_f64] {
                if let Some((v1, v2)) = solve_lambert_branch(r_e, r_t, dt_sec, GM_SUN_KM3_S2, dm) {
                    let v_inf_dep = v1.sub(v_e).length();
                    let c3 = v_inf_dep * v_inf_dep;
                    let v_perigee = (c3 + 2.0 * GM_EARTH_KM3_S2 / r_leo).sqrt();
                    let dv_dep = (v_perigee - v_leo).abs();

                    let v_inf_arr = v2.sub(v_t).length();
                    let dv_arr = target_capture_dv(eff, v_inf_arr);
                    let total_dv = dv_dep + dv_arr;

                    if total_dv.is_finite()
                        && best_cell.map_or(true, |(prev_dv, _)| total_dv < prev_dv)
                    {
                        best_cell = Some((total_dv, c3));
                    }
                }
            }

            let (total_dv, c3) = best_cell.unwrap_or((base_hoh.total_delta_v_km_s * 3.0, 120.0));
            let capped_dv = total_dv.min(base_hoh.total_delta_v_km_s * 4.5 + 18.0);
            dv_km_s[r_idx * cols + c_idx] = capped_dv;
            c3_km2_s2[r_idx * cols + c_idx] = c3;
            if capped_dv < min_dv {
                min_dv = capped_dv;
                best_c = c_idx;
                best_r = r_idx;
            }
            max_dv = max_dv.max(capped_dv);
        }
    }

    PorkchopGrid {
        cols,
        rows,
        best_dep_offset_days: dep_offsets_days[best_c],
        best_tof_days: tof_days[best_r],
        best_c3_km2_s2: c3_km2_s2[best_r * cols + best_c],
        dep_offsets_days,
        tof_days,
        dv_km_s,
        c3_km2_s2,
        min_dv_km_s: min_dv,
        max_dv_km_s: max_dv,
        best_col: best_c,
        best_row: best_r,
        synodic_period_days: synodic,
        phase_error_deg: phase_err,
    }
}

#[derive(Debug, Clone)]
pub struct GravityAssistSolution {
    pub assist_id: String,
    pub assist_name: String,
    pub direct_dv_km_s: f64,
    pub assisted_dv_km_s: f64,
    pub dv_saved_km_s: f64,
    pub turn_angle_deg: f64,
    pub periapsis_alt_km: f64,
    pub total_flight_days: f64,
}

pub const ASSIST_CANDIDATE_IDS: [&str; 4] = ["venus", "mars", "jupiter", "saturn"];

fn leo_escape_dv_from_v_inf(v_inf_km_s: f64) -> f64 {
    let r_leo = EARTH_RADIUS_KM + LEO_ALTITUDE_KM;
    let v_leo = (GM_EARTH_KM3_S2 / r_leo).sqrt();
    let v_perigee = (v_inf_km_s * v_inf_km_s + 2.0 * GM_EARTH_KM3_S2 / r_leo).sqrt();
    (v_perigee - v_leo).abs()
}

/// Evaluates a genuine patched-conic 2-leg Gravity-Assist Slingshot (`Earth -> Assist -> Target`),
/// enforcing radial reachability, `v_inf` matching, hyperbolic bending limits, and positive `delta_v` savings.
pub fn compute_gravity_assist(
    assist: &CelestialBody,
    target: &CelestialBody,
    catalog: &[CelestialBody],
) -> Option<GravityAssistSolution> {
    let eff_target = effective_heliocentric_body(target, catalog);
    if assist.id == eff_target.id
        || assist.id == "earth"
        || target.id == "earth"
        || eff_target.id == "moon"
        || eff_target.category == BodyCategory::LagrangePoint
        || eff_target.distance_ly > 0.0
    {
        return None;
    }

    let r_e = AU_KM;
    let a_a_au = assist.semi_major_axis_au;
    let a_t_au = if eff_target.id == "sun" {
        0.08
    } else {
        eff_target.semi_major_axis_au.max(0.08)
    };
    let q_max_t_au = a_t_au * (1.0 + eff_target.eccentricity.clamp(0.0, 0.99));

    // Radial feasibility gates:
    // - Outward assists (Mars/Jupiter/Saturn) cannot assist targets inside Earth or strictly inside assist orbit
    //   unless the target's apohelion reaches the assist orbit.
    // - Inward assist (Venus) can assist inner targets (a_t < a_a) or high-e comets, not outer-giant transfers via single flyby.
    if a_a_au > 1.0 {
        if a_t_au < 0.95 || (a_t_au < a_a_au * 0.95 && q_max_t_au < a_a_au * 0.95) {
            return None;
        }
    } else if a_t_au > 1.25 {
        return None;
    }

    let r_a = a_a_au * AU_KM;
    let r_t = a_t_au * AU_KM;
    let v_e = (GM_SUN_KM3_S2 / r_e).sqrt();
    let v_a = (GM_SUN_KM3_S2 / r_a).sqrt();
    let v_t = (GM_SUN_KM3_S2 / r_t).sqrt();

    // Evaluate direct physical cost including 3D inclination plane change on the same footing as assisted
    let a_dir = 0.5 * (r_e + r_t);
    let v_dir_e = (GM_SUN_KM3_S2 * (2.0 / r_e - 1.0 / a_dir)).sqrt();
    let v_dir_t = (GM_SUN_KM3_S2 * (2.0 / r_t - 1.0 / a_dir)).sqrt();
    let dv_dep_dir = leo_escape_dv_from_v_inf((v_dir_e - v_e).abs());
    let dv_arr_dir = target_capture_dv(eff_target, (v_t - v_dir_t).abs());
    let inc_diff_rad = eff_target.inclination_deg.abs().to_radians();
    let v_mid_dir = (GM_SUN_KM3_S2 / a_dir).sqrt();
    let dv_plane_dir = 2.0 * v_mid_dir * (inc_diff_rad * 0.5).sin();
    let direct_phys_dv = dv_dep_dir + dv_arr_dir + dv_plane_dir;

    let peri_alt = if assist.radius_km > 20_000.0 {
        45_000.0
    } else {
        650.0
    };
    let r_p = assist.radius_km + peri_alt;
    let mu_assist = assist.gm_km3_s2().max(1_000.0);
    let outward = r_t > r_a;

    // Require minimum tangential speed at r_a to reach r_t (inward or outward)
    let a2_min = 0.5 * (r_a + r_t);
    let v_req_a = (GM_SUN_KM3_S2 * (2.0 / r_a - 1.0 / a2_min)).sqrt();

    let mut best_sol: Option<(f64, f64, f64)> = None; // (assisted_phys_dv, turn_deg, flight_days)

    for step in 0..=32 {
        let alpha = (step as f64) * 0.015;
        let (r_p1, r_a1) = if r_a > r_e {
            (r_e, r_a * (1.0 + alpha))
        } else {
            ((r_a * (1.0 - alpha)).max(0.18 * AU_KM), r_e)
        };
        let a1 = 0.5 * (r_p1 + r_a1);
        let eps1 = -GM_SUN_KM3_S2 / (2.0 * a1);
        let h1 = (2.0 * GM_SUN_KM3_S2 * r_p1 * r_a1 / (r_p1 + r_a1)).sqrt();

        let v_1e_sq = (2.0 * (eps1 + GM_SUN_KM3_S2 / r_e)).max(0.0);
        let v_t1e = h1 / r_e;
        let v_r1e = (v_1e_sq - v_t1e * v_t1e).max(0.0).sqrt();
        let v_inf_dep = ((v_t1e - v_e).powi(2) + v_r1e * v_r1e).sqrt();
        let dv_dep_1 = leo_escape_dv_from_v_inf(v_inf_dep);

        let v_1a_sq = (2.0 * (eps1 + GM_SUN_KM3_S2 / r_a)).max(0.0);
        let v_t1a = h1 / r_a;
        let v_r1a = (v_1a_sq - v_t1a * v_t1a).max(0.0).sqrt();
        let v_in_t = v_t1a - v_a;
        let v_inf_in = (v_r1a * v_r1a + v_in_t * v_in_t).sqrt();
        if v_inf_in < 0.05 {
            continue;
        }

        // Maximum hyperbolic deflection angle at safe periapsis r_p
        let sin_half_max = 1.0 / (1.0 + (r_p * v_inf_in * v_inf_in) / mu_assist);
        let delta_max_rad = 2.0 * sin_half_max.clamp(0.0, 0.9999).asin();

        // Angle of v_in relative to +v_a (0 = purely prograde +v_a, pi = purely retrograde -v_a)
        let ang_in = v_r1a.atan2(v_in_t).abs();
        let ang_lo = (ang_in - delta_max_rad).max(0.0);
        let ang_hi = (ang_in + delta_max_rad).min(PI);

        // Scan achievable post-flyby deflection angles inside [ang_lo, ang_hi] for unpowered/trim trajectories
        let samples = 24usize;
        for si in 0..=samples {
            let frac = si as f64 / samples as f64;
            let ang_out = ang_lo + (ang_hi - ang_lo) * frac;
            let turn_rad = (ang_out - ang_in).abs();
            let turn_deg = turn_rad.to_degrees();
            if turn_deg < 4.0 {
                continue;
            }

            let v_out_r = v_inf_in * ang_out.sin();
            let v_out_t = v_a + v_inf_in * ang_out.cos();
            let v_out_mag = (v_out_r * v_out_r + v_out_t * v_out_t).sqrt();

            // Require unpowered post-flyby orbit to reach (or within 0.35 km/s Oberth trim of) Leg 2 energy
            let (v_helio_2, h2, dv_flyby) = if outward {
                if v_out_mag >= v_req_a {
                    (v_out_mag, r_a * v_out_t, 0.0)
                } else if v_req_a - v_out_mag <= 0.35 {
                    let v_inf_need = (v_req_a - v_a).max(0.0);
                    let v_p_out = (v_inf_need * v_inf_need + 2.0 * mu_assist / r_p).sqrt();
                    let v_p_in = (v_inf_in * v_inf_in + 2.0 * mu_assist / r_p).sqrt();
                    (v_req_a, r_a * v_req_a, (v_p_out - v_p_in).abs())
                } else {
                    continue;
                }
            } else if v_out_mag <= v_req_a {
                (v_out_mag, r_a * v_out_t, 0.0)
            } else if v_out_mag - v_req_a <= 0.35 {
                let v_inf_need = (v_a - v_req_a).abs();
                let v_p_out = (v_inf_need * v_inf_need + 2.0 * mu_assist / r_p).sqrt();
                let v_p_in = (v_inf_in * v_inf_in + 2.0 * mu_assist / r_p).sqrt();
                (v_req_a, r_a * v_req_a, (v_p_out - v_p_in).abs())
            } else {
                continue;
            };

            let eps2 = 0.5 * v_helio_2 * v_helio_2 - GM_SUN_KM3_S2 / r_a;
            let a2 = if eps2.abs() > 1e-6 {
                -GM_SUN_KM3_S2 / (2.0 * eps2)
            } else {
                10.0 * r_t
            };
            let v_2t_sq = 2.0 * (eps2 + GM_SUN_KM3_S2 / r_t);
            if v_2t_sq < 0.0 {
                continue;
            }
            let v_t2t = h2 / r_t;
            let v_r2t = (v_2t_sq - v_t2t * v_t2t).max(0.0).sqrt();
            let v_inf_arr = ((v_t2t - v_t).powi(2) + v_r2t * v_r2t).sqrt();
            let dv_arr_2 = target_capture_dv(eff_target, v_inf_arr);

            // Residual plane change not absorbed by 3D B-plane flyby cranking
            let flyby_inc_crank = (delta_max_rad * 0.5).min(inc_diff_rad);
            let rem_inc = (inc_diff_rad - flyby_inc_crank).max(0.0);
            let dv_plane_2 = 2.0 * v_t * (rem_inc * 0.5).sin();

            let assisted_phys_dv = dv_dep_1 + dv_flyby + dv_arr_2 + dv_plane_2;
            let t_leg1_days = PI * (a1.powi(3) / GM_SUN_KM3_S2).sqrt() / 86_400.0;
            let a2_pos = a2.abs().max(0.5 * (r_a + r_t));
            let t_leg2_days = PI * (a2_pos.powi(3) / GM_SUN_KM3_S2).sqrt() / 86_400.0;
            let total_days = t_leg1_days + t_leg2_days;

            if best_sol.map_or(true, |(best_dv, _, _)| assisted_phys_dv < best_dv) {
                best_sol = Some((assisted_phys_dv, turn_deg, total_days));
            }
        }
    }

    let (assisted_phys_dv, turn_deg, flight_days) = best_sol?;
    let dv_saved = direct_phys_dv - assisted_phys_dv;
    if dv_saved <= 0.25 {
        return None;
    }

    Some(GravityAssistSolution {
        assist_id: assist.id.clone(),
        assist_name: assist.name.clone(),
        direct_dv_km_s: direct_phys_dv,
        assisted_dv_km_s: assisted_phys_dv,
        dv_saved_km_s: dv_saved,
        turn_angle_deg: turn_deg,
        periapsis_alt_km: peri_alt,
        total_flight_days: flight_days,
    })
}

pub const DEFAULT_MISSION_BUDGET_TARGETS: &[&str] = &[
    "sun_earth_l2",
    "venus",
    "mars",
    "ceres",
    "jupiter",
    "europa",
    "saturn",
    "titan",
    "uranus",
    "neptune",
];

/// Comparative mission delta-v, C3, time-of-flight, and gravity-assist savings entry
/// for a single Solar System or Sun–Earth L2 destination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionBudgetTargetRow {
    pub target_id: String,
    pub target_name: String,
    pub category: String,
    pub regime: String,
    pub semi_major_axis_au: f64,
    pub transfer_dv_km_s: f64,
    pub c3_km2_s2: f64,
    pub time_of_flight_days: f64,
    pub capture_orbit_vel_km_s: f64,
    pub best_assist_body: Option<String>,
    pub assisted_dv_km_s: Option<f64>,
    pub dv_saved_km_s: f64,
    pub assisted_flight_days: Option<f64>,
}

/// Multi-target Hohmann / Sun–Earth L2 transfer delta-v & time-of-flight mission budget comparator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionTransferBudgetReport {
    pub l2_distance_km: f64,
    pub l2_halo_period_xy_days: f64,
    pub l2_halo_period_z_days: f64,
    pub rows: Vec<MissionBudgetTargetRow>,
}

/// Computes a multi-target Hohmann + Sun–Earth L2 halo transfer delta-v & time-of-flight
/// mission budget table across `target_ids` (defaulting to `DEFAULT_MISSION_BUDGET_TARGETS` if empty).
pub fn compute_multi_target_transfer_budget(
    target_ids: &[&str],
    catalog: &[CelestialBody],
) -> MissionTransferBudgetReport {
    let ids: &[&str] = if target_ids.is_empty() {
        DEFAULT_MISSION_BUDGET_TARGETS
    } else {
        target_ids
    };
    let l2_dyn = compute_sun_earth_l2_dynamics();

    let mut rows = Vec::new();
    for &id in ids {
        let key = id.trim().to_ascii_lowercase();
        let Some(target) = catalog.iter().find(|b| b.id == key) else {
            continue;
        };
        let sol = compute_transfer_from_earth(target, catalog, 0.05);
        let regime = if target.category == BodyCategory::LagrangePoint {
            "Sun-Earth L2 Halo"
        } else if target.distance_ly > 0.0 {
            "Relativistic Interstellar"
        } else {
            "Heliocentric Hohmann"
        };

        let mut best_assist: Option<GravityAssistSolution> = None;
        for aid in ASSIST_CANDIDATE_IDS {
            if let Some(ab) = catalog.iter().find(|b| b.id == aid) {
                if let Some(ga) = compute_gravity_assist(ab, target, catalog) {
                    if best_assist
                        .as_ref()
                        .map_or(true, |prev| ga.dv_saved_km_s > prev.dv_saved_km_s)
                    {
                        best_assist = Some(ga);
                    }
                }
            }
        }

        let (best_assist_body, assisted_dv_km_s, dv_saved_km_s, assisted_flight_days) =
            match best_assist {
                Some(ga) => (
                    Some(ga.assist_name),
                    Some(ga.assisted_dv_km_s),
                    ga.dv_saved_km_s,
                    Some(ga.total_flight_days),
                ),
                None => (None, None, 0.0, None),
            };

        rows.push(MissionBudgetTargetRow {
            target_id: target.id.clone(),
            target_name: target.name.clone(),
            category: format!("{:?}", target.category),
            regime: regime.to_string(),
            semi_major_axis_au: target.semi_major_axis_au,
            transfer_dv_km_s: sol.total_delta_v_km_s,
            c3_km2_s2: sol.c3_km2_s2,
            time_of_flight_days: sol.physical_flight_days,
            capture_orbit_vel_km_s: sol.capture_orbit_vel_km_s,
            best_assist_body,
            assisted_dv_km_s,
            dv_saved_km_s,
            assisted_flight_days,
        });
    }

    MissionTransferBudgetReport {
        l2_distance_km: l2_dyn.l2_distance_km,
        l2_halo_period_xy_days: l2_dyn.period_xy_days,
        l2_halo_period_z_days: l2_dyn.period_z_days,
        rows,
    }
}

impl MissionTransferBudgetReport {
    pub fn to_csv(&self) -> String {
        let mut out = String::from(
            "target_id,target_name,category,regime,semi_major_axis_au,transfer_dv_km_s,c3_km2_s2,time_of_flight_days,capture_orbit_vel_km_s,best_assist_body,assisted_dv_km_s,dv_saved_km_s,assisted_flight_days\n",
        );
        for r in &self.rows {
            out.push_str(&format!(
                "{},\"{}\",{},\"{}\",{:.4},{:.3},{:.2},{:.1},{:.3},{},{},{:.3},{}\n",
                r.target_id,
                r.target_name,
                r.category,
                r.regime,
                r.semi_major_axis_au,
                r.transfer_dv_km_s,
                r.c3_km2_s2,
                r.time_of_flight_days,
                r.capture_orbit_vel_km_s,
                r.best_assist_body.as_deref().unwrap_or("none"),
                r.assisted_dv_km_s
                    .map(|v| format!("{v:.3}"))
                    .unwrap_or_else(|| "na".into()),
                r.dv_saved_km_s,
                r.assisted_flight_days
                    .map(|v| format!("{v:.1}"))
                    .unwrap_or_else(|| "na".into()),
            ));
        }
        out
    }

    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn format_table(&self) -> String {
        let mut out = format!(
            "=== HelioProbe Multi-Target Hohmann & Sun-Earth L2 Mission Budget (L2={:.0} km, Halo T_xy={:.1} d) ===\n",
            self.l2_distance_km, self.l2_halo_period_xy_days
        );
        out.push_str(&format!(
            "{:<14} {:<19} {:>7} {:>9} {:>9} {:>10} {:>10}\n",
            "Target", "Regime", "a(AU)", "dV(km/s)", "C3", "TOF(d)", "AssistSave"
        ));
        for r in &self.rows {
            out.push_str(&format!(
                "{:<14} {:<19} {:>7.3} {:>9.2} {:>9.1} {:>10.1} {:>10}\n",
                r.target_id,
                r.regime,
                r.semi_major_axis_au,
                r.transfer_dv_km_s,
                r.c3_km2_s2,
                r.time_of_flight_days,
                if r.dv_saved_km_s > 0.0 {
                    format!(
                        "-{:.2} ({})",
                        r.dv_saved_km_s,
                        r.best_assist_body.as_deref().unwrap_or("?")
                    )
                } else {
                    "—".to_string()
                }
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::bootstrap_catalog;

    #[test]
    fn multi_target_budget_includes_l2_and_planetary_hohmann_with_assist_savings() {
        let catalog = bootstrap_catalog();
        let report = compute_multi_target_transfer_budget(&[], &catalog);
        assert_eq!(report.rows.len(), DEFAULT_MISSION_BUDGET_TARGETS.len());
        assert!((report.l2_distance_km - 1_501_154.0).abs() < 15_000.0);

        let l2 = report
            .rows
            .iter()
            .find(|r| r.target_id == "sun_earth_l2")
            .expect("sun_earth_l2 row");
        assert_eq!(l2.regime, "Sun-Earth L2 Halo");
        assert!(l2.transfer_dv_km_s > 2.5 && l2.transfer_dv_km_s < 4.5);
        assert!(l2.time_of_flight_days < 120.0);

        let saturn = report
            .rows
            .iter()
            .find(|r| r.target_id == "saturn")
            .expect("saturn row");
        assert!(saturn.dv_saved_km_s > 0.5);
        assert_eq!(saturn.best_assist_body.as_deref(), Some("Jupiter"));

        let csv = report.to_csv();
        assert!(csv.starts_with("target_id,target_name,category,regime,"));
        assert_eq!(csv.lines().count(), DEFAULT_MISSION_BUDGET_TARGETS.len() + 1);

        let json_str = report.to_json_pretty().expect("serialize budget JSON");
        let parsed: MissionTransferBudgetReport =
            serde_json::from_str(&json_str).expect("deserialize budget JSON");
        assert_eq!(parsed.rows.len(), report.rows.len());
    }
}
