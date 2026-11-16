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

//! Gravity-Assist Slingshot & Porkchop Launch-Window Planner (`src/astrodynamics_lab.rs`):
//! - Computes planetary synodic periods, current vs optimal Hohmann phase angles, and next launch window offsets.
//! - Generates a 2D Lambert-approximated Porkchop `delta_v` contour grid (`departure_offset_days` x `flight_days`).
//! - Evaluates patched-conic 2-leg Gravity-Assist Slingshot routes (`Earth -> Assist Body -> Target`) with
//!   hyperbolic deflection angle `delta = 2 * asin(1 / (1 + r_p * v_inf^2 / mu))` and Oberth/flyby `delta_v` savings.

use crate::catalog::{BodyCategory, CelestialBody};
use crate::physics::{
    compute_transfer_from_earth, kepler_position_au, AU_KM, EARTH_RADIUS_KM, GM_EARTH_KM3_S2,
    GM_SUN_KM3_S2, LEO_ALTITUDE_KM,
};
use std::f64::consts::{PI, TAU};

#[derive(Debug, Clone)]
pub struct PorkchopGrid {
    pub cols: usize,
    pub rows: usize,
    pub dep_offsets_days: Vec<f64>,
    pub tof_days: Vec<f64>,
    /// Row-major `[row * cols + col]` total mission `delta_v` in `km/s`.
    pub dv_km_s: Vec<f64>,
    pub min_dv_km_s: f64,
    pub max_dv_km_s: f64,
    pub best_col: usize,
    pub best_row: usize,
    pub best_dep_offset_days: f64,
    pub best_tof_days: f64,
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

/// Computes a `cols x rows` Porkchop launch-window `delta_v` contour matrix from Earth (`1.0 AU`)
/// to any Solar System body (or parent planet for moons) starting from `current_j2000_days`.
pub fn compute_porkchop_grid(
    target: &CelestialBody,
    catalog: &[CelestialBody],
    current_j2000_days: f64,
    cols: usize,
    rows: usize,
) -> PorkchopGrid {
    let cols = cols.max(8);
    let rows = rows.max(8);
    let eff = effective_heliocentric_body(target, catalog);
    let earth = catalog
        .iter()
        .find(|b| b.id == "earth")
        .unwrap_or(eff);

    if eff.distance_ly > 0.0 || eff.id == "earth" || eff.id == "sun" {
        let base = compute_transfer_from_earth(target, catalog, 0.20);
        let dep_offsets_days: Vec<f64> = (0..cols).map(|c| c as f64 * 15.0).collect();
        let tof_days: Vec<f64> = (0..rows)
            .map(|r| base.physical_flight_days * (0.65 + 0.7 * (r as f64 / (rows - 1) as f64)))
            .collect();
        let mut dv_km_s = vec![0.0; cols * rows];
        let mut min_dv = f64::MAX;
        let mut max_dv = 0.0_f64;
        let mut best_c = 0;
        let mut best_r = rows / 2;
        for r in 0..rows {
            for c in 0..cols {
                let ratio = tof_days[r] / base.physical_flight_days.max(0.1);
                let val = base.total_delta_v_km_s * (0.85 + 0.22 * (ratio - 1.0).powi(2) + 0.03 * ((c as f64 * 0.4).sin()).abs());
                dv_km_s[r * cols + c] = val;
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
            dep_offsets_days,
            tof_days,
            dv_km_s,
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
    let t_hohmann_days = PI * (((0.5 * (a1 + a2) * AU_KM).powi(3)) / GM_SUN_KM3_S2).sqrt() / 86_400.0;

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
        eff.mean_anomaly_j2000_deg,
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
    let mut min_dv = f64::MAX;
    let mut max_dv = 0.0_f64;
    let mut best_c = 0;
    let mut best_r = 0;

    for (r_idx, &tof) in tof_days.iter().enumerate() {
        for (c_idx, &dep_off) in dep_offsets_days.iter().enumerate() {
            let t_dep = current_j2000_days + dep_off;
            let t_arr = t_dep + tof;

            let pos_dep = kepler_position_au(
                earth.semi_major_axis_au,
                earth.eccentricity,
                earth.inclination_deg,
                earth.lon_ascending_node_deg,
                earth.arg_periapsis_deg,
                earth.mean_anomaly_j2000_deg,
                earth.orbital_period_days,
                t_dep,
            );
            let pos_arr = kepler_position_au(
                eff.semi_major_axis_au,
                eff.eccentricity,
                eff.inclination_deg,
                eff.lon_ascending_node_deg,
                eff.arg_periapsis_deg,
                eff.mean_anomaly_j2000_deg,
                eff.orbital_period_days,
                t_arr,
            );

            let ang_dep = pos_dep.z.atan2(pos_dep.x);
            let ang_arr = pos_arr.z.atan2(pos_arr.x);
            let transfer_angle = (ang_arr - ang_dep).rem_euclid(TAU);

            // Departure-arrival phase penalty around PI radians (180° Hohmann geometry)
            let d_theta = (transfer_angle - PI).abs();
            let tof_ratio = tof / t_hohmann_days.max(1.0);
            let inc_rad = (eff.inclination_deg - earth.inclination_deg).to_radians().abs();

            let v_earth = (GM_SUN_KM3_S2 / AU_KM).sqrt();
            let plane_dv = 2.0 * v_earth * (inc_rad * 0.5).sin();
            let phase_penalty = 6.8 * (d_theta / PI).powi(2) / tof_ratio.max(0.35);
            let tof_penalty = 3.1 * (tof_ratio - 1.0).powi(2);

            // Include subtle Type-I / Type-II dual-lobe porkchop asymmetry
            let lobe = if transfer_angle < PI { 0.0 } else { 0.45 };
            let total_dv = (base_hoh.total_delta_v_km_s * 0.88
                + phase_penalty
                + tof_penalty
                + plane_dv * 0.35
                + lobe
                + (v_leo * 0.0))
                .clamp(base_hoh.total_delta_v_km_s * 0.82, base_hoh.total_delta_v_km_s * 3.4);

            dv_km_s[r_idx * cols + c_idx] = total_dv;
            if total_dv < min_dv {
                min_dv = total_dv;
                best_c = c_idx;
                best_r = r_idx;
            }
            max_dv = max_dv.max(total_dv);
        }
    }

    PorkchopGrid {
        cols,
        rows,
        best_dep_offset_days: dep_offsets_days[best_c],
        best_tof_days: tof_days[best_r],
        dep_offsets_days,
        tof_days,
        dv_km_s,
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

/// Evaluates a patched-conic 2-leg Gravity-Assist Slingshot (`Earth -> Assist -> Target`).
pub fn compute_gravity_assist(
    assist: &CelestialBody,
    target: &CelestialBody,
    catalog: &[CelestialBody],
) -> Option<GravityAssistSolution> {
    let eff_target = effective_heliocentric_body(target, catalog);
    if assist.id == eff_target.id || assist.id == "earth" || target.id == "earth" {
        return None;
    }
    let direct = compute_transfer_from_earth(target, catalog, 0.20);
    let leg1 = compute_transfer_from_earth(assist, catalog, 0.05);

    // Hyperbolic flyby periapsis clearance & patched-conic bending angle
    let peri_alt = if assist.radius_km > 20_000.0 { 45_000.0 } else { 650.0 };
    let r_p = assist.radius_km + peri_alt;
    let v_inf_est = (leg1.c3_km2_s2.abs().sqrt()).clamp(3.2, 12.5);
    let mu_assist = assist.gm_km3_s2().max(1_000.0);
    let sin_half = 1.0 / (1.0 + (r_p * v_inf_est * v_inf_est) / mu_assist);
    let turn_deg = 2.0 * sin_half.clamp(0.0, 0.999).asin().to_degrees();

    // Gravity assist shines when stepping toward outer giants/Kuiper/interstellar via Jupiter/Saturn
    // or stepping inward to Mercury/Sun via Venus, or Belt targets via Mars!
    let efficiency = if eff_target.distance_ly > 0.0 && (assist.id == "jupiter" || assist.id == "saturn") {
        0.16
    } else if eff_target.semi_major_axis_au > 6.0 && assist.id == "jupiter" {
        0.31
    } else if eff_target.semi_major_axis_au > 15.0 && assist.id == "saturn" {
        0.28
    } else if eff_target.semi_major_axis_au < 0.7 && assist.id == "venus" {
        0.26
    } else if eff_target.semi_major_axis_au > 1.8 && eff_target.semi_major_axis_au < 4.5 && assist.id == "mars" {
        0.19
    } else {
        0.07
    };

    let dv_saved = (direct.total_delta_v_km_s * efficiency).clamp(0.25, 4200.0);
    let assisted_dv = (direct.total_delta_v_km_s - dv_saved).max(3.2);
    let flight_days = if eff_target.distance_ly > 0.0 {
        direct.physical_flight_days * 0.92
    } else {
        leg1.physical_flight_days + direct.physical_flight_days * 0.78
    };

    Some(GravityAssistSolution {
        assist_id: assist.id.clone(),
        assist_name: assist.name.clone(),
        direct_dv_km_s: direct.total_delta_v_km_s,
        assisted_dv_km_s: assisted_dv,
        dv_saved_km_s: direct.total_delta_v_km_s - assisted_dv,
        turn_angle_deg: turn_deg,
        periapsis_alt_km: peri_alt,
        total_flight_days: flight_days,
    })
}
