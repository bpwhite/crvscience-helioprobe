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

//! Planetary Surface Landers, Ocean-World Cryobots, Galileo-Class Atmospheric Entry / Crash Probes,
//! & Earth Sample-Return Capsules (`src/surface_ops.rs`):
//! - Aero-Shell / Sky-Crane Surface Landers (3-axis seismometry, core drill mineralogy, heat-flow) for solid worlds.
//! - Pu-238 GPHS Copper-Dome Thermal Melt Cryobots with Ice-Sheath Acoustic/RF Relay Pucks for 12 icy ocean worlds.
//! - Galileo / Cassini / Deep-Space-2 Atmospheric Entry & Kinetic Crash-Landing Probes descending from orbit to crush-depth or impact!
//! - Earth Sample-Return Capsules (`SRC`) launching back from target orbit to Earth splashdown & cleanroom curation.

use crate::catalog::{BodyCategory, CelestialBody};
use crate::physics::{body_scene_position, probe_orbit_radius_scene, visual_body_radius_scene, Vec3d};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SurfaceUnitKind {
    AeroShellLander,
    OceanCryobot,
    AtmosphericEntryProbe,
}

impl SurfaceUnitKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::AeroShellLander => "Aero-Shell / Sky-Crane Surface Lander",
            Self::OceanCryobot => "GPHS-Pu238 Thermal Melt Ocean Cryobot",
            Self::AtmosphericEntryProbe => "Galileo-Class Atmospheric Entry / Crash Probe",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurfaceDeployment {
    pub body_id: String,
    pub body_name: String,
    pub kind: SurfaceUnitKind,
    pub telemetry_headline: String,
    pub telemetry_report: String,
    pub relay_pucks_deployed: u32,
    pub ice_melt_depth_km: f64,
    #[serde(default)]
    pub active_experiment_done: bool,
}

pub const ICY_CRYOBOT_WORLDS: [&str; 12] = [
    "europa",
    "enceladus",
    "titan",
    "ganymede",
    "callisto",
    "triton",
    "ceres",
    "pluto",
    "charon",
    "haumea",
    "makemake",
    "eris",
];

pub const ATMOSPHERE_WORLDS: [&str; 14] = [
    "venus",
    "earth",
    "mars",
    "jupiter",
    "saturn",
    "uranus",
    "neptune",
    "titan",
    "triton",
    "proxima_b",
    "trappist_1e",
    "cancri_55e",
    "k2_18b",
    "kepler_186f",
];

pub fn supports_surface_lander(body: &CelestialBody) -> bool {
    match body.category {
        BodyCategory::Star | BodyCategory::Nebula | BodyCategory::LagrangePoint => false,
        BodyCategory::Planet | BodyCategory::Exoplanet => body.radius_km < 18_000.0,
        BodyCategory::DwarfPlanet
        | BodyCategory::Moon
        | BodyCategory::Asteroid
        | BodyCategory::Comet => true,
    }
}

pub fn supports_ocean_cryobot(body: &CelestialBody) -> bool {
    ICY_CRYOBOT_WORLDS.contains(&body.id.as_str())
}

pub fn supports_atmospheric_or_crash_probe(body: &CelestialBody) -> bool {
    !matches!(
        body.category,
        BodyCategory::Star | BodyCategory::Nebula | BodyCategory::LagrangePoint
    ) && body.id != "earth"
}

pub fn has_substantial_atmosphere(body: &CelestialBody) -> bool {
    ATMOSPHERE_WORLDS.contains(&body.id.as_str()) || body.radius_km >= 18_000.0
}

pub fn generate_lander_deployment(body: &CelestialBody) -> SurfaceDeployment {
    let (headline, report) = match body.id.as_str() {
        "mars" => (
            "Sky-Crane Touchdown & Broadband Seismometer Locked",
            "Recorded marsquake P/S-wave arrivals constraining liquid Fe-S-O core radius (~1,660 km) and drilled regolith perchlorate/clay cores.",
        ),
        "titan" => (
            "Parachute Descent Through Orange Tholin Haze (1.45 Bar N2)",
            "Touchdown on cryogenic water-ice cobbles beside liquid methane-ethane drainage channels at -179 °C.",
        ),
        "venus" => (
            "Titanium Pressure-Vessel Touchdown at 92 Bars & 464 °C",
            "High-pressure X-ray fluorescence confirmed basaltic plain composition beneath supercritical CO2 winds.",
        ),
        "bennu" | "apophis" | "psyche" | "vesta" => (
            "Micro-Gravity Harpoon / Thruster-Hold Regolith Contact",
            "Measured ultra-low surface cohesion, thermal inertia, and unweathered chondritic/metallic grain mineralogy.",
        ),
        _ if body.category == BodyCategory::Comet => (
            "Philae-Class Anchor Harpoon & Ice-Screw Comet Touchdown",
            "Anchored into ultra-dark porous cometary dust/ice mantle; sampled sublimating H2O/CO2 jets, glycine-bearing organics, and CONSERT radar interior echoes.",
        ),
        _ => (
            "Autonomous Soft Touchdown & Surface Seismometry Active",
            "Deployed 3-axis broadband seismometer, thermal heat-flow probe, and laser-induced breakdown spectrometer across local regolith.",
        ),
    };

    SurfaceDeployment {
        body_id: body.id.clone(),
        body_name: body.name.clone(),
        kind: SurfaceUnitKind::AeroShellLander,
        telemetry_headline: headline.into(),
        telemetry_report: report.into(),
        relay_pucks_deployed: 1,
        ice_melt_depth_km: 0.0,
        active_experiment_done: false,
    }
}

pub fn generate_cryobot_deployment(
    body: &CelestialBody,
    deep_vent_upgrade: bool,
) -> SurfaceDeployment {
    let depth_km = match body.id.as_str() {
        "enceladus" => if deep_vent_upgrade { 8.5 } else { 4.2 },
        "europa" => if deep_vent_upgrade { 18.4 } else { 9.5 },
        "titan" => if deep_vent_upgrade { 40.0 } else { 19.0 },
        "ganymede" | "callisto" => if deep_vent_upgrade { 34.0 } else { 15.0 },
        _ => if deep_vent_upgrade { 16.0 } else { 7.5 },
    };
    let pucks = ((depth_km / 2.5_f64).ceil() as u32).max(3);
    let mode_tag = if deep_vent_upgrade {
        "Deep-Ocean Hydrothermal Interface Reached"
    } else {
        "Sub-Ice Brine & Clathrate Melt-Channel Active"
    };

    SurfaceDeployment {
        body_id: body.id.clone(),
        body_name: body.name.clone(),
        kind: SurfaceUnitKind::OceanCryobot,
        telemetry_headline: format!(
            "{} ({:.1} km Melt Depth • {} Relay Pucks)",
            mode_tag, depth_km, pucks
        ),
        telemetry_report: format!(
            "Hemispherical copper-alloy GPHS-Pu238 thermal nose cone descended {:.1} km below {}'s refreezing ice/clathrate crust without Leidenfrost vapor lock, linking data to orbit via {} wireless acoustic/RF ice-sheath relay pucks.",
            depth_km, body.name, pucks
        ),
        relay_pucks_deployed: pucks,
        ice_melt_depth_km: depth_km,
        active_experiment_done: false,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntryProbePhase {
    HypersonicEntry,
    ParachuteCloudDrift,
    TerminalCrushOrImpact,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtmosphericProbeRun {
    pub id: u64,
    pub body_id: String,
    pub body_name: String,
    pub has_atmosphere: bool,
    pub phase: EntryProbePhase,
    pub progress: f64,
    pub duration_sec: f64,
    pub entry_speed_km_s: f64,
    pub start_alt_km: f64,
    pub current_alt_km: f64,
    pub current_pressure_bar: f64,
    pub current_temp_c: f64,
    pub terminal_pressure_bar: f64,
    pub terminal_temp_c: f64,
    pub terminal_headline: String,
    pub terminal_report: String,
}

impl AtmosphericProbeRun {
    pub fn launch_from_orbit(id: u64, body: &CelestialBody) -> Self {
        let has_atmo = has_substantial_atmosphere(body);
        let (entry_v, start_alt, term_bar, term_temp, headline, report) = match body.id.as_str() {
            "jupiter" => (
                47.2,
                450.0,
                22.4,
                153.0,
                "Galileo-Class Crush-Depth Loss of Signal at 22.4 Bars (+153 °C)",
                "Entered Jupiter at 170,000 km/h (47.2 km/s), deployed main parachute below ammonia cloud tops, and relayed 1387 MHz He/H2 mass-spectrometry & Doppler wind profiles across a 150 km sub-cloud descent before 22.4-bar thermal/pressure destruction.",
            ),
            "saturn" => (
                35.5,
                420.0,
                25.0,
                138.0,
                "Cassini Finale-Class Deep Troposphere Crush at 25.0 Bars (+138 °C)",
                "Sliced through 500 m/s equatorial jet winds (94% H2 / 6% He) from -139 °C upper hazes into ammonium-hydrosulfide & water cloud decks before supercritical pressure crushed the descent hull.",
            ),
            "uranus" | "neptune" => (
                23.6,
                380.0,
                32.0,
                182.0,
                "Ice-Giant Deep Methane/Sulfide Troposphere Crush at 32.0 Bars",
                "Pierced -200 °C cryogenic methane-ice clouds and H2S decks, measuring D/H and noble-gas enrichment before 32-bar pressure loss of signal.",
            ),
            "venus" => (
                11.2,
                180.0,
                92.0,
                464.0,
                "Supercritical CO2 Atmosphere & 92-Bar Basaltic Impact (+464 °C)",
                "Survived sulfuric-acid cloud droplets (45–65 km) and transmitted sub-cloud noble gas & IR radiative flux profiles down to hard impact on 464 °C volcanic plains.",
            ),
            "titan" => (
                6.1,
                220.0,
                1.45,
                -179.0,
                "Tholin Haze Parachute Descent & Cryogenic Cobble Impact (1.45 Bar)",
                "Sampled complex C-H-N organic aerosols during subsonic parachute drift through dense N2-CH4 haze before impact at -179 °C.",
            ),
            "k2_18b" => (
                21.8,
                350.0,
                28.5,
                165.0,
                "Hycean H2-Rich Cloud Plunge & Deep Pressure Termination (28.5 Bar)",
                "Measured CH4, CO2, and water-vapor mixing ratios beneath H2-dominated upper scale heights before deep-envelope loss of signal.",
            ),
            _ if has_atmo => (
                (body.escape_velocity_km_s.max(5.5)),
                160.0,
                4.5,
                body.mean_temp_c + 65.0,
                "Atmospheric Entry Profile & High-Speed Surface Crash Telemetry",
                "Recorded upper-atmosphere density scale height, peak entry g-load, and boundary-layer composition prior to terminal surface impact.",
            ),
            _ => (
                (body.escape_velocity_km_s.max(2.4)),
                120.0,
                0.0,
                body.mean_temp_c + 180.0,
                "Hypervelocity Kinetic Impactor Plunge & Ejecta Plume Flash",
                "Executed targeted orbital de-orbit crash (LCROSS/DS2 class), excavating unweathered subsurface regolith & volatile ice grains for overhead orbital spectroscopy.",
            ),
        };

        Self {
            id,
            body_id: body.id.clone(),
            body_name: body.name.clone(),
            has_atmosphere: has_atmo,
            phase: EntryProbePhase::HypersonicEntry,
            progress: 0.0,
            duration_sec: 6.2,
            entry_speed_km_s: entry_v,
            start_alt_km: start_alt,
            current_alt_km: start_alt,
            current_pressure_bar: if has_atmo { 0.001 } else { 0.0 },
            current_temp_c: body.mean_temp_c,
            terminal_pressure_bar: term_bar,
            terminal_temp_c: term_temp,
            terminal_headline: headline.into(),
            terminal_report: report.into(),
        }
    }

    /// Advances atmospheric descent or kinetic impact plunge; returns `true` on the frame of crush-depth / crash termination!
    pub fn step(&mut self, dt_sec: f64, transit_warp: f64) -> bool {
        if self.phase == EntryProbePhase::TerminalCrushOrImpact {
            return false;
        }
        let rate = transit_warp.clamp(0.4, 3.5) / self.duration_sec.max(1.5);
        self.progress = (self.progress + dt_sec * rate).min(1.0);

        let p = self.progress;
        self.current_alt_km = if self.has_atmosphere {
            self.start_alt_km * (1.0 - p * 1.32)
        } else {
            self.start_alt_km * (1.0 - p)
        };
        if self.has_atmosphere {
            // Exponential barometric increase toward crush depth
            let exp_f = ((p * 4.2).exp() - 1.0) / ((4.2_f64).exp() - 1.0);
            self.current_pressure_bar = (0.001 + self.terminal_pressure_bar * exp_f).max(0.001);
            self.current_temp_c =
                self.current_temp_c * (1.0 - dt_sec * 1.5).max(0.0)
                    + (-130.0 + (self.terminal_temp_c + 130.0) * p.powf(1.4))
                        * (dt_sec * 1.5).min(1.0);
            self.phase = if p < 0.34 {
                EntryProbePhase::HypersonicEntry
            } else {
                EntryProbePhase::ParachuteCloudDrift
            };
        }

        if self.progress >= 1.0 {
            self.current_pressure_bar = self.terminal_pressure_bar;
            self.current_temp_c = self.terminal_temp_c;
            self.phase = EntryProbePhase::TerminalCrushOrImpact;
            return true;
        }
        false
    }

    pub fn scene_position(&self, catalog: &[CelestialBody], days_since_j2000: f64) -> Vec3d {
        let Some(body) = catalog.iter().find(|b| b.id == self.body_id) else {
            return Vec3d::default();
        };
        let center = body_scene_position(body, catalog, days_since_j2000);
        let r_surf = visual_body_radius_scene(body);
        let r_orb = probe_orbit_radius_scene(body);
        let p = self.progress.clamp(0.0, 1.0);
        let r_cur = r_orb + (r_surf * 1.02 - r_orb) * p;
        let spiral_ang = (1.0 - p) * 1.85 + 0.6;
        center.add(Vec3d::new(
            r_cur * spiral_ang.cos(),
            r_cur * 0.28 * (1.0 - p),
            r_cur * spiral_ang.sin(),
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapsulePhase {
    ReturningToEarth,
    RecoveredAtEarth,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleReturnCapsule {
    pub id: u64,
    pub origin_body_id: String,
    pub origin_body_name: String,
    pub sample_mass_grams: f64,
    pub phase: CapsulePhase,
    pub progress: f64,
    pub duration_sec: f64,
    pub origin_scene_pos: [f64; 3],
}

impl SampleReturnCapsule {
    pub fn launch_to_earth(
        id: u64,
        origin: &CelestialBody,
        catalog: &[CelestialBody],
        days_since_j2000: f64,
    ) -> Self {
        let p = body_scene_position(origin, catalog, days_since_j2000);
        Self {
            id,
            origin_body_id: origin.id.clone(),
            origin_body_name: origin.name.clone(),
            sample_mass_grams: if origin.category == BodyCategory::Asteroid {
                121.6
            } else {
                250.0
            },
            phase: CapsulePhase::ReturningToEarth,
            progress: 0.0,
            duration_sec: 4.8,
            origin_scene_pos: [p.x, p.y, p.z],
        }
    }

    pub fn step(&mut self, dt_sec: f64, transit_warp: f64) -> bool {
        if self.phase == CapsulePhase::ReturningToEarth {
            self.progress += (dt_sec * transit_warp.max(0.4)) / self.duration_sec.max(1.0);
            if self.progress >= 1.0 {
                self.progress = 1.0;
                self.phase = CapsulePhase::RecoveredAtEarth;
                return true;
            }
        }
        false
    }

    pub fn scene_position(&self, catalog: &[CelestialBody], days_since_j2000: f64) -> Vec3d {
        let earth_pos = catalog
            .iter()
            .find(|b| b.id == "earth")
            .map(|e| body_scene_position(e, catalog, days_since_j2000))
            .unwrap_or(Vec3d::new(18.5, 0.0, 0.0));
        if self.phase == CapsulePhase::RecoveredAtEarth {
            return earth_pos;
        }
        let start = Vec3d::new(
            self.origin_scene_pos[0],
            self.origin_scene_pos[1],
            self.origin_scene_pos[2],
        );
        let t = self.progress.clamp(0.0, 1.0);
        let s = t * t * (3.0 - 2.0 * t);
        let chord = earth_pos.sub(start);
        let dist = chord.length();
        let mut side = chord.cross(Vec3d::new(0.0, 1.0, 0.0)).normalized();
        if side.length() < 0.1 {
            side = Vec3d::new(0.0, 0.0, 1.0);
        }
        let ctrl = start
            .lerp(earth_pos, 0.5)
            .sub(side.scale(dist * 0.22))
            .add(Vec3d::new(0.0, (dist * 0.07).min(6.0), 0.0));
        let u = 1.0 - s;
        start
            .scale(u * u)
            .add(ctrl.scale(2.0 * u * s))
            .add(earth_pos.scale(s * s))
    }
}
