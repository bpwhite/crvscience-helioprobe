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

//! Space probe launch, fast 3D curved Lambert/Bezier Earth-to-target transit,
//! 2-leg patched-conic Gravity-Assist Slingshot flybys, Tech Tree cruise upgrades,
//! automatic orbital capture, and 3D Keplerian target orbit propagation.

use crate::astrodynamics_lab::compute_gravity_assist;
use crate::catalog::CelestialBody;
use crate::physics::{
    body_scene_position, body_scene_position_with_offsets, compute_transfer_from_earth,
    probe_orbit_radius_scene, Vec3d,
};
use crate::tech_tree::tech_speed_multiplier;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::TAU;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropulsionBus {
    ChemicalCentaur,
    SolarElectricIon,
    RtgNuclearElectric,
    LaserSailFusion,
}

impl PropulsionBus {
    pub const ALL: [PropulsionBus; 4] = [
        PropulsionBus::ChemicalCentaur,
        PropulsionBus::SolarElectricIon,
        PropulsionBus::RtgNuclearElectric,
        PropulsionBus::LaserSailFusion,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::ChemicalCentaur => "Chemical Centaur-V (Inner / Cis-Lunar)",
            Self::SolarElectricIon => "Solar-Electric Hall Ion (Belt / Jovian)",
            Self::RtgNuclearElectric => "RTG Nuclear-Electric (Outer Giants / Kuiper)",
            Self::LaserSailFusion => "Directed-Energy Laser Sail / Fusion Relay (Interstellar)",
        }
    }

    /// Base real-time transit duration in seconds (at `1.0x` warp) so flights are fast and engaging.
    pub fn base_transit_seconds(self, target: &CelestialBody) -> f64 {
        let dist_factor = if target.distance_ly > 0.0 {
            1.35 + (1.0 + target.distance_ly / 100.0).ln() * 0.22
        } else {
            0.75 + (1.0 + target.semi_major_axis_au).ln() * 0.35
        };
        let bus_speed = match self {
            Self::ChemicalCentaur => 1.0,
            Self::SolarElectricIon => 0.88,
            Self::RtgNuclearElectric => 0.78,
            Self::LaserSailFusion => 0.58,
        };
        (4.8 * dist_factor * bus_speed).clamp(2.4, 9.5)
    }

    pub fn relativistic_beta(self) -> f64 {
        match self {
            Self::LaserSailFusion => 0.20,
            Self::RtgNuclearElectric => 0.05,
            _ => 0.05,
        }
    }

    pub fn effective_beta(self, unlocked_techs: &BTreeSet<String>) -> f64 {
        if self == Self::LaserSailFusion && unlocked_techs.contains("beamed_sail_core") {
            0.38
        } else {
            self.relativistic_beta()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProbePhase {
    EnRoute,
    Orbiting,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpaceProbe {
    pub id: u64,
    pub callsign: String,
    pub bus: PropulsionBus,
    pub target_id: String,
    pub target_name: String,
    pub phase: ProbePhase,
    /// Transit progress `0.0..=1.0` while `EnRoute` (`0.0..0.48` is Leg 1 to assist planet when `assist_body_id` is `Some`).
    pub transit_progress: f64,
    pub transit_duration_sec: f64,
    pub launch_origin_scene: [f64; 3],
    /// Optional gravity-assist flyby body (`venus`, `mars`, `jupiter`, `saturn`).
    #[serde(default)]
    pub assist_body_id: Option<String>,
    #[serde(default)]
    pub assist_body_name: Option<String>,
    #[serde(default)]
    pub assist_flyby_completed: bool,
    #[serde(default)]
    pub assist_flyby_notified: bool,
    #[serde(default)]
    pub launched_from_l2: bool,
    /// Orbital phase angle in radians around target once `Orbiting`.
    pub orbit_phase_rad: f64,
    pub orbit_inclination_rad: f64,
    pub total_delta_v_km_s: f64,
    pub physical_flight_days: f64,
    pub capture_orbit_vel_km_s: f64,
    pub capture_altitude_km: f64,
    pub lorentz_gamma: f64,
}

fn quadratic_arc(start: Vec3d, end: Vec3d, raw_t: f64, bend_sign: f64) -> Vec3d {
    let t = raw_t.clamp(0.0, 1.0);
    let s = t * t * (3.0 - 2.0 * t);
    let chord = end.sub(start);
    let dist = chord.length();
    let up = Vec3d::new(0.0, 1.0, 0.0);
    let mut side = chord.cross(up).normalized();
    if side.length() < 0.1 {
        side = Vec3d::new(0.0, 0.0, 1.0);
    }
    let ctrl = start
        .lerp(end, 0.5)
        .add(side.scale(dist * 0.22 * bend_sign))
        .add(Vec3d::new(0.0, (dist * 0.07).min(7.5), 0.0));
    let u = 1.0 - s;
    start
        .scale(u * u)
        .add(ctrl.scale(2.0 * u * s))
        .add(end.scale(s * s))
}

impl SpaceProbe {
    pub fn launch_from_earth(
        id: u64,
        callsign: String,
        bus: PropulsionBus,
        target: &CelestialBody,
        catalog: &[CelestialBody],
        days_since_j2000: f64,
    ) -> Self {
        Self::launch_with_options(
            id,
            callsign,
            bus,
            target,
            None,
            &BTreeSet::new(),
            catalog,
            days_since_j2000,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_with_options(
        id: u64,
        callsign: String,
        bus: PropulsionBus,
        target: &CelestialBody,
        assist: Option<&CelestialBody>,
        unlocked_techs: &BTreeSet<String>,
        catalog: &[CelestialBody],
        days_since_j2000: f64,
    ) -> Self {
        let earth_pos = catalog
            .iter()
            .find(|b| b.id == "earth")
            .map(|e| body_scene_position(e, catalog, days_since_j2000))
            .unwrap_or(Vec3d::new(18.5, 0.0, 0.0));

        let beta = bus.effective_beta(unlocked_techs);
        let xfer = compute_transfer_from_earth(target, catalog, beta);

        let is_ion_or_rtg = matches!(
            bus,
            PropulsionBus::SolarElectricIon | PropulsionBus::RtgNuclearElectric
        );
        let is_sail = bus == PropulsionBus::LaserSailFusion;
        let speed_mult = tech_speed_multiplier(unlocked_techs, is_ion_or_rtg, is_sail);

        let mut total_dv = xfer.total_delta_v_km_s;
        let mut flight_days = xfer.physical_flight_days;
        let mut assist_id = None;
        let mut assist_name = None;
        let mut base_dur = bus.base_transit_seconds(target) / speed_mult.max(1.0);

        if let Some(a_body) = assist {
            if let Some(sol) = compute_gravity_assist(a_body, target, catalog) {
                total_dv = sol.assisted_dv_km_s;
                flight_days = sol.total_flight_days;
                assist_id = Some(sol.assist_id);
                assist_name = Some(sol.assist_name);
                base_dur = (base_dur * 1.18).clamp(2.2, 10.5);
            }
        }

        Self {
            id,
            callsign,
            bus,
            target_id: target.id.clone(),
            target_name: target.name.clone(),
            phase: ProbePhase::EnRoute,
            transit_progress: 0.0,
            transit_duration_sec: base_dur.clamp(1.6, 10.5),
            launch_origin_scene: [earth_pos.x, earth_pos.y, earth_pos.z],
            assist_body_id: assist_id,
            assist_body_name: assist_name,
            assist_flyby_completed: false,
            assist_flyby_notified: false,
            launched_from_l2: false,
            orbit_phase_rad: (id as f64 * 1.17).rem_euclid(TAU),
            orbit_inclination_rad: 0.42 + ((id % 5) as f64) * 0.14,
            total_delta_v_km_s: total_dv,
            physical_flight_days: flight_days,
            capture_orbit_vel_km_s: xfer.capture_orbit_vel_km_s,
            capture_altitude_km: xfer.capture_altitude_km,
            lorentz_gamma: xfer.lorentz_gamma,
        }
    }

    /// Launches a high-speed 3-spacecraft Comet Interceptor sortie directly from the Sun-Earth L2 halo station!
    pub fn launch_from_l2_interceptor(
        id: u64,
        callsign: String,
        target: &CelestialBody,
        unlocked_techs: &BTreeSet<String>,
        catalog: &[CelestialBody],
        days_since_j2000: f64,
    ) -> Self {
        let mut p = Self::launch_with_options(
            id,
            callsign,
            PropulsionBus::SolarElectricIon,
            target,
            None,
            unlocked_techs,
            catalog,
            days_since_j2000,
        );
        let earth_pos = catalog
            .iter()
            .find(|b| b.id == "earth")
            .map(|e| body_scene_position(e, catalog, days_since_j2000))
            .unwrap_or(Vec3d::new(16.45, 0.0, 0.0));
        let l2_pos =
            crate::l2_comets::sun_earth_l2_halo_position(earth_pos, days_since_j2000, 240.0, 0.48);
        p.launch_origin_scene = [l2_pos.x, l2_pos.y, l2_pos.z];
        p.transit_duration_sec = (p.transit_duration_sec * 0.52).clamp(1.2, 5.2);
        p.total_delta_v_km_s = (p.total_delta_v_km_s - 3.05).max(0.65);
        p.launched_from_l2 = true;
        p
    }

    /// Steps the probe state machine forward by `dt_sec` (scaled by `transit_warp`) and returns `true`
    /// on the exact frame the probe completes orbital insertion!
    pub fn step(&mut self, dt_sec: f64, transit_warp: f64) -> bool {
        match self.phase {
            ProbePhase::EnRoute => {
                let rate = transit_warp.max(0.25) / self.transit_duration_sec.max(0.5);
                self.transit_progress += dt_sec * rate;
                if self.assist_body_id.is_some() && self.transit_progress >= 0.48 {
                    self.assist_flyby_completed = true;
                }
                if self.transit_progress >= 1.0 {
                    self.transit_progress = 1.0;
                    self.phase = ProbePhase::Orbiting;
                    return true;
                }
                false
            }
            ProbePhase::Orbiting => {
                self.orbit_phase_rad = (self.orbit_phase_rad + dt_sec * 0.55).rem_euclid(TAU);
                false
            }
        }
    }

    /// Returns `Some(assist_name)` once when a 2-leg probe crosses periapsis of its gravity-assist body.
    pub fn take_flyby_notification(&mut self) -> Option<String> {
        if self.assist_flyby_completed && !self.assist_flyby_notified {
            self.assist_flyby_notified = true;
            return self.assist_body_name.clone().or_else(|| self.assist_body_id.clone());
        }
        None
    }

    /// Evaluates either a 1-leg Hohmann curve, a 2-leg patched-conic Gravity-Assist Slingshot curve,
    /// or an inclined 3D Keplerian capture orbit around the target body once `Orbiting`.
    pub fn scene_position(&self, catalog: &[CelestialBody], days_since_j2000: f64) -> Vec3d {
        self.scene_position_with_offsets(catalog, days_since_j2000, &BTreeMap::new())
    }

    /// Evaluates `scene_position` while honoring any user-triggered comet perihelion offsets.
    pub fn scene_position_with_offsets(
        &self,
        catalog: &[CelestialBody],
        days_since_j2000: f64,
        comet_offsets: &BTreeMap<String, f64>,
    ) -> Vec3d {
        let target = catalog.iter().find(|b| b.id == self.target_id);
        let target_pos = target
            .map(|t| {
                body_scene_position_with_offsets(t, catalog, days_since_j2000, comet_offsets)
            })
            .unwrap_or_default();

        match self.phase {
            ProbePhase::EnRoute => {
                let start = Vec3d::new(
                    self.launch_origin_scene[0],
                    self.launch_origin_scene[1],
                    self.launch_origin_scene[2],
                );
                let t = self.transit_progress.clamp(0.0, 1.0);
                if let Some(aid) = self.assist_body_id.as_deref() {
                    if let Some(assist_body) = catalog.iter().find(|b| b.id == aid) {
                        let assist_pos = body_scene_position_with_offsets(
                            assist_body,
                            catalog,
                            days_since_j2000,
                            comet_offsets,
                        );
                        if t < 0.48 {
                            return quadratic_arc(start, assist_pos, t / 0.48, 1.0);
                        } else {
                            return quadratic_arc(assist_pos, target_pos, (t - 0.48) / 0.52, -0.85);
                        }
                    }
                }
                quadratic_arc(start, target_pos, t, 1.08)
            }
            ProbePhase::Orbiting => {
                let orbit_r = target.map(probe_orbit_radius_scene).unwrap_or(1.55);
                let ang = self.orbit_phase_rad;
                let inc = self.orbit_inclination_rad;
                let local = Vec3d::new(
                    orbit_r * ang.cos(),
                    orbit_r * ang.sin() * inc.sin(),
                    orbit_r * ang.sin() * inc.cos(),
                );
                target_pos.add(local)
            }
        }
    }
}
