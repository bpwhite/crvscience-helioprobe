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

//! Gamification engine: Science Points (SP), Available Tech Lab SP, 5 Agency Ranks,
//! 19 Scientific Achievements, 5-Band Instrument Scanner, Surface Landers / Ocean Cryobots,
//! Galileo Atmospheric Entry / Crash Probes, Earth Sample-Return Capsules, and SQLite WAL persistence.

use crate::catalog::{BodyCategory, CelestialBody, ScanBand};
use crate::probes::{ProbePhase, SpaceProbe};
use crate::surface_ops::{
    generate_cryobot_deployment, generate_lander_deployment, supports_atmospheric_or_crash_probe,
    supports_ocean_cryobot, supports_surface_lander, AtmosphericProbeRun, SampleReturnCapsule,
    SurfaceDeployment, SurfaceUnitKind,
};
use crate::tech_tree::{
    cryobot_points, sample_return_points, scan_band_points, TechNodeDef, TECH_TREE,
};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AchievementDef {
    pub id: &'static str,
    pub title: &'static str,
    pub description: &'static str,
    pub bonus_points: u32,
}

pub const ACHIEVEMENTS: [AchievementDef; 21] = [
    AchievementDef {
        id: "first_launch",
        title: "First Pale Blue Departure",
        description: "Launch your first space probe from Earth into deep space.",
        bonus_points: 100,
    },
    AchievementDef {
        id: "cislunar_pathfinder",
        title: "Cis-Lunar Pathfinder",
        description: "Insert a space probe into stable orbit around Earth's Moon.",
        bonus_points: 100,
    },
    AchievementDef {
        id: "inner_rocky_triad",
        title: "Inner Rocky Triad",
        description: "Orbit and scan Mercury, Venus, and Mars.",
        bonus_points: 250,
    },
    AchievementDef {
        id: "osiris_bennu",
        title: "OSIRIS Sample Legacy",
        description: "Complete a Thermal IR & Mass Spectrometer scan of asteroid 101955 Bennu.",
        bonus_points: 200,
    },
    AchievementDef {
        id: "ocean_worlds",
        title: "Ocean Worlds Investigator",
        description: "Scan Europa, Enceladus, and Ceres to characterize subsurface oceans and brines.",
        bonus_points: 350,
    },
    AchievementDef {
        id: "huygens_titan",
        title: "Huygens' Successor",
        description: "Orbit Saturn's moon Titan and scan its dense nitrogen-methane atmosphere.",
        bonus_points: 250,
    },
    AchievementDef {
        id: "ice_giant_voyager",
        title: "Ice Giant Voyager",
        description: "Establish spacecraft orbit around both Uranus and Neptune.",
        bonus_points: 350,
    },
    AchievementDef {
        id: "kuiper_horizon",
        title: "Kuiper Belt Horizon",
        description: "Insert an orbiter around Pluto at 39.5 AU.",
        bonus_points: 300,
    },
    AchievementDef {
        id: "exoplanet_relay",
        title: "Nearest Neighbor Relay",
        description: "Reach and scan Proxima Centauri b (4 ly) or TRAPPIST-1 e (41 ly).",
        bonus_points: 400,
    },
    AchievementDef {
        id: "stellar_nursery",
        title: "Stellar Nursery Cartographer",
        description: "Scan the star-forming pillars of Orion (M42), Eagle (M16), and Carina (NGC 3372).",
        bonus_points: 400,
    },
    AchievementDef {
        id: "crab_pulsar_1054",
        title: "1054 Supernova Pulsar",
        description: "Detect the 33-millisecond pulsar dynamo inside the Crab Nebula (M1).",
        bonus_points: 300,
    },
    AchievementDef {
        id: "open_data_archivist",
        title: "Open-Data Archivist",
        description: "Import a live target from NASA JPL SBDB, Exoplanet Archive, or CDS SIMBAD.",
        bonus_points: 250,
    },
    AchievementDef {
        id: "galilean_tour",
        title: "Galilean Grand Tour",
        description: "Orbit and scan all four Galilean moons of Jupiter (Io, Europa, Ganymede, and Callisto).",
        bonus_points: 350,
    },
    AchievementDef {
        id: "dwarf_pentad",
        title: "Plutoid & Dwarf Explorer",
        description: "Scan at least three of the five recognized dwarf planets (Ceres, Pluto, Haumea, Makemake, Eris).",
        bonus_points: 400,
    },
    AchievementDef {
        id: "metal_and_lava",
        title: "Heavy Metal & Lava Worlds",
        description: "Scan metal-rich asteroid 16 Psyche and ultra-hot lava exoplanet 55 Cancri e.",
        bonus_points: 350,
    },
    AchievementDef {
        id: "gravity_slingshot",
        title: "Oberth Gravity Slingshot",
        description: "Execute a planetary gravity-assist flyby (Venus, Mars, Jupiter, or Saturn) en route to a target.",
        bonus_points: 300,
    },
    AchievementDef {
        id: "subsurface_cryobot",
        title: "Sub-Ice Ocean Cryobot",
        description: "Melt through an icy moon or dwarf-planet crust with a Pu-238 GPHS Ocean Cryobot and ice-relay pucks.",
        bonus_points: 400,
    },
    AchievementDef {
        id: "earth_sample_return",
        title: "Cleanroom Sample Curatory",
        description: "Launch a Sample-Return Capsule from target orbit and recover extraterrestrial samples at Earth.",
        bonus_points: 450,
    },
    AchievementDef {
        id: "atmospheric_plunge",
        title: "Galileo Crush-Depth Plunge",
        description: "Drop an Atmospheric Entry / Crash Probe from orbit into a gas giant, atmosphere, or target body until crush-depth or impact loss-of-signal.",
        bonus_points: 350,
    },
    AchievementDef {
        id: "l2_halo_array",
        title: "Sun-Earth L2 Halo Array",
        description: "Commission JWST, Nancy Grace Roman, or Comet Interceptor into a Sun-Earth L2 halo orbit (~1.5 million km anti-sunward).",
        bonus_points: 350,
    },
    AchievementDef {
        id: "comet_tail_chaser",
        title: "Solar-Wind Tail Chaser",
        description: "Orbit or intercept an eccentric comet and scan its C2 coma, CO+ ion tail, or silicate dust jets.",
        bonus_points: 350,
    },
];

fn default_one() -> u64 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CampaignState {
    pub campaign_name: String,
    pub science_points: u64,
    #[serde(default)]
    pub spent_sp: u64,
    pub days_since_j2000: f64,
    pub next_probe_id: u64,
    #[serde(default = "default_one")]
    pub next_capsule_id: u64,
    #[serde(default = "default_one")]
    pub next_atmo_id: u64,
    pub probes: Vec<SpaceProbe>,
    pub scanned_bands: BTreeMap<String, BTreeSet<String>>,
    pub orbited_bodies: BTreeSet<String>,
    pub unlocked_achievements: BTreeSet<String>,
    #[serde(default)]
    pub unlocked_techs: BTreeSet<String>,
    #[serde(default)]
    pub dsn_auto_scan: bool,
    #[serde(default)]
    pub surface_deployments: BTreeMap<String, Vec<SurfaceDeployment>>,
    #[serde(default)]
    pub atmospheric_probes: Vec<AtmosphericProbeRun>,
    #[serde(default)]
    pub sample_capsules: Vec<SampleReturnCapsule>,
    #[serde(default)]
    pub completed_slingshots: u32,
    #[serde(default)]
    pub recovered_samples_count: u32,
    #[serde(default)]
    pub completed_atmo_plunges: u32,
    #[serde(default)]
    pub deployed_l2_observatories: BTreeSet<String>,
    #[serde(default)]
    pub roman_survey_completed: bool,
    #[serde(default)]
    pub l2_intercepts_launched: u32,
    #[serde(default)]
    pub comet_perihelion_offsets: BTreeMap<String, f64>,
    pub imported_bodies: Vec<CelestialBody>,
    pub mission_log: Vec<String>,
}

impl Default for CampaignState {
    fn default() -> Self {
        Self {
            campaign_name: "HelioProbe Deep Space Command".into(),
            science_points: 0,
            spent_sp: 0,
            days_since_j2000: 9814.5,
            next_probe_id: 1,
            next_capsule_id: 1,
            next_atmo_id: 1,
            probes: Vec::new(),
            scanned_bands: BTreeMap::new(),
            orbited_bodies: BTreeSet::new(),
            unlocked_achievements: BTreeSet::new(),
            unlocked_techs: BTreeSet::new(),
            dsn_auto_scan: false,
            surface_deployments: BTreeMap::new(),
            atmospheric_probes: Vec::new(),
            sample_capsules: Vec::new(),
            completed_slingshots: 0,
            recovered_samples_count: 0,
            completed_atmo_plunges: 0,
            deployed_l2_observatories: BTreeSet::new(),
            roman_survey_completed: false,
            l2_intercepts_launched: 0,
            comet_perihelion_offsets: BTreeMap::new(),
            imported_bodies: Vec::new(),
            mission_log: vec![
                "Mission Control initialized at Earth LEO launch complex (400 km altitude).".into(),
            ],
        }
    }
}

impl CampaignState {
    pub fn available_sp(&self) -> u64 {
        self.science_points.saturating_sub(self.spent_sp)
    }

    pub fn agency_rank(&self) -> &'static str {
        match self.science_points {
            0..=299 => "Cadet Flight Controller",
            300..=899 => "Willamette Deep Space Navigator",
            900..=1999 => "Planetary Principal Investigator",
            2000..=3999 => "Exoplanet & Nebula Cartographer",
            _ => "Interstellar Mission Director",
        }
    }

    pub fn has_orbiting_probe_at(&self, body_id: &str) -> bool {
        self.probes
            .iter()
            .any(|p| p.target_id == body_id && p.phase == ProbePhase::Orbiting)
    }

    pub fn is_band_scanned(&self, body_id: &str, band: ScanBand) -> bool {
        self.scanned_bands
            .get(body_id)
            .map(|set| set.contains(band.code()))
            .unwrap_or(false)
    }

    pub fn scanned_count_for(&self, body_id: &str) -> usize {
        self.scanned_bands.get(body_id).map(|s| s.len()).unwrap_or(0)
    }

    pub fn has_surface_unit(&self, body_id: &str, kind: SurfaceUnitKind) -> bool {
        self.surface_deployments
            .get(body_id)
            .map(|list| list.iter().any(|d| d.kind == kind))
            .unwrap_or(false)
    }

    pub fn total_cryobots_deployed(&self) -> usize {
        self.surface_deployments
            .values()
            .flat_map(|v| v.iter())
            .filter(|d| d.kind == SurfaceUnitKind::OceanCryobot)
            .count()
    }

    pub fn push_log(&mut self, entry: impl Into<String>) {
        self.mission_log.push(entry.into());
        if self.mission_log.len() > 80 {
            let excess = self.mission_log.len() - 80;
            self.mission_log.drain(0..excess);
        }
    }

    pub fn unlock_tech(&mut self, tech_id: &str) -> Result<&'static TechNodeDef, String> {
        let Some(node) = TECH_TREE.iter().find(|n| n.id == tech_id) else {
            return Err(format!("Unknown tech node '{tech_id}'."));
        };
        if self.unlocked_techs.contains(node.id) {
            return Err(format!("{} is already researched.", node.title));
        }
        if self.available_sp() < node.cost_sp {
            return Err(format!(
                "Need {} Available SP to unlock {} (currently {} Available SP).",
                node.cost_sp,
                node.title,
                self.available_sp()
            ));
        }
        self.spent_sp += node.cost_sp;
        self.unlocked_techs.insert(node.id.to_string());
        if node.id == "dsn_ka_band" {
            self.dsn_auto_scan = true;
        }
        self.push_log(format!(
            "[TECH LAB] Researched {} (-{} Available SP; Lifetime Rank unchanged).",
            node.title, node.cost_sp
        ));
        Ok(node)
    }

    pub fn record_launch(&mut self, probe: SpaceProbe) -> Vec<&'static AchievementDef> {
        self.science_points += 25;
        if let Some(assist_name) = &probe.assist_body_name {
            self.push_log(format!(
                "Launched {} -> {} via {} Gravity Assist (dV = {:.2} km/s, flight = {:.1} d). +25 SP",
                probe.callsign,
                probe.target_name,
                assist_name,
                probe.total_delta_v_km_s,
                probe.physical_flight_days
            ));
        } else {
            self.push_log(format!(
                "Launched {} -> {} (dV = {:.2} km/s, physical transit = {:.1} d). +25 SP",
                probe.callsign, probe.target_name, probe.total_delta_v_km_s, probe.physical_flight_days
            ));
        }
        self.probes.push(probe);
        self.evaluate_achievements()
    }

    pub fn record_slingshot_flyby(
        &mut self,
        callsign: &str,
        assist_name: &str,
    ) -> Vec<&'static AchievementDef> {
        self.completed_slingshots += 1;
        let bonus = if self.unlocked_techs.contains("oberth_nav") {
            135
        } else {
            75
        };
        self.science_points += bonus;
        self.push_log(format!(
            "[SLINGSHOT] {} completed hyperbolic periapsis flyby at {}! +{} SP",
            callsign, assist_name, bonus
        ));
        self.evaluate_achievements()
    }

    pub fn record_orbit_insertion(
        &mut self,
        target: &CelestialBody,
    ) -> Vec<&'static AchievementDef> {
        let first_time = self.orbited_bodies.insert(target.id.clone());
        if first_time {
            let pts = match target.category {
                BodyCategory::Moon | BodyCategory::LagrangePoint => 120,
                BodyCategory::Planet | BodyCategory::Asteroid | BodyCategory::DwarfPlanet => 150,
                BodyCategory::Comet => 210,
                BodyCategory::Star => 200,
                BodyCategory::Exoplanet | BodyCategory::Nebula => 300,
            };
            self.science_points += pts;
            self.push_log(format!(
                "Orbit insertion confirmed around {}! +{} SP",
                target.name, pts
            ));
        } else {
            self.science_points += 35;
            self.push_log(format!(
                "Additional relay probe entered orbit around {}. +35 SP",
                target.name
            ));
        }

        if self.unlocked_techs.contains("dsn_ka_band") && self.dsn_auto_scan {
            for band in ScanBand::ALL {
                let _ = self.perform_scan(target, band);
            }
            self.push_log(format!(
                "[DSN Ka-BAND] Automated 5-band capture sweep completed at {}.",
                target.name
            ));
        }

        self.evaluate_achievements()
    }

    pub fn perform_scan(
        &mut self,
        target: &CelestialBody,
        band: ScanBand,
    ) -> Result<(bool, Vec<&'static AchievementDef>), String> {
        let jwst_remote = self.deployed_l2_observatories.contains("jwst_l2")
            && matches!(band, ScanBand::OpticalNir | ScanBand::ThermalMassSpec)
            && matches!(
                target.category,
                BodyCategory::Comet | BodyCategory::Exoplanet | BodyCategory::Nebula | BodyCategory::DwarfPlanet
            );
        let roman_remote = self.deployed_l2_observatories.contains("roman_l2")
            && band == ScanBand::OpticalNir;
        let l2_self_scan = target.id == "sun_earth_l2" && !self.deployed_l2_observatories.is_empty();
        let remote_allowed = (band == ScanBand::OpticalNir
            && self.unlocked_techs.contains("coronagraph_jwst"))
            || jwst_remote
            || roman_remote
            || l2_self_scan;
        if target.id != "earth" && !remote_allowed && !self.has_orbiting_probe_at(&target.id) {
            return Err(format!(
                "Launch a probe from Earth and establish orbit around {} before running orbital {} scans.",
                target.name,
                band.label()
            ));
        }
        let band_pts = scan_band_points(&self.unlocked_techs);
        let has_coronagraph = self.unlocked_techs.contains("coronagraph_jwst");
        let set = self.scanned_bands.entry(target.id.clone()).or_default();
        let newly_discovered = set.insert(band.code().to_string());
        let completed_all = set.len() == ScanBand::ALL.len();

        if newly_discovered {
            self.science_points += band_pts;
            self.push_log(format!(
                "[{}] Scanned {}: +{} SP",
                band.code(),
                target.name,
                band_pts
            ));
            if completed_all {
                let mut survey_bonus = 150_u64;
                if has_coronagraph
                    && matches!(target.category, BodyCategory::Exoplanet | BodyCategory::Nebula)
                {
                    survey_bonus += 200;
                }
                self.science_points += survey_bonus;
                self.push_log(format!(
                    "Completed all 5 instrument bands at {}! Full-Survey Bonus +{} SP",
                    target.name, survey_bonus
                ));
            }
        }
        let unlocked = self.evaluate_achievements();
        Ok((newly_discovered, unlocked))
    }

    pub fn deploy_surface_unit(
        &mut self,
        target: &CelestialBody,
        kind: SurfaceUnitKind,
    ) -> Result<(SurfaceDeployment, Vec<&'static AchievementDef>), String> {
        if target.id != "earth" && !self.has_orbiting_probe_at(&target.id) {
            return Err(format!(
                "Establish spacecraft orbit around {} before releasing a {}.",
                target.name,
                kind.label()
            ));
        }
        if self.has_surface_unit(&target.id, kind) {
            return Err(format!(
                "{} is already active on {}.",
                kind.label(),
                target.name
            ));
        }

        let deployment = match kind {
            SurfaceUnitKind::AeroShellLander => {
                if !supports_surface_lander(target) {
                    return Err(format!(
                        "{} lacks a solid planetary crust — drop an Atmospheric Entry Probe instead!",
                        target.name
                    ));
                }
                self.science_points += 180;
                generate_lander_deployment(target)
            }
            SurfaceUnitKind::OceanCryobot => {
                if !supports_ocean_cryobot(target) {
                    return Err(format!(
                        "{} is not an icy ocean/brine world suited for a thermal melt Cryobot.",
                        target.name
                    ));
                }
                let pts = cryobot_points(&self.unlocked_techs);
                self.science_points += pts;
                generate_cryobot_deployment(target, self.unlocked_techs.contains("cryobot_gphs"))
            }
            SurfaceUnitKind::AtmosphericEntryProbe => {
                return Err("Use launch_atmospheric_probe() for dynamic descent.".into());
            }
        };

        self.push_log(format!(
            "[SURFACE OPS] {} deployed on {} — {}",
            kind.label(),
            target.name,
            deployment.telemetry_headline
        ));
        self.surface_deployments
            .entry(target.id.clone())
            .or_default()
            .push(deployment.clone());
        let achs = self.evaluate_achievements();
        Ok((deployment, achs))
    }

    pub fn launch_atmospheric_probe(
        &mut self,
        target: &CelestialBody,
    ) -> Result<AtmosphericProbeRun, String> {
        if !supports_atmospheric_or_crash_probe(target) {
            return Err(format!(
                "Cannot drop an atmospheric or crash probe into {}.",
                target.name
            ));
        }
        if !self.has_orbiting_probe_at(&target.id) {
            return Err(format!(
                "Requires an active orbiter around {} to release an Atmospheric / Crash Probe.",
                target.name
            ));
        }
        let aid = self.next_atmo_id.max(1);
        self.next_atmo_id = aid + 1;
        let run = AtmosphericProbeRun::launch_from_orbit(aid, target);
        self.science_points += 60;
        self.push_log(format!(
            "[ENTRY PROBE] Released AEP-{aid} into {} at {:.1} km/s (+60 SP).",
            target.name, run.entry_speed_km_s
        ));
        self.atmospheric_probes.push(run.clone());
        Ok(run)
    }

    pub fn record_atmo_probe_termination(
        &mut self,
        run: &AtmosphericProbeRun,
    ) -> Vec<&'static AchievementDef> {
        self.completed_atmo_plunges += 1;
        self.science_points += 200;
        self.push_log(format!(
            "[LOSS OF SIGNAL] AEP-{} @ {}: {} (+200 SP Black-Box Relay)",
            run.id, run.body_name, run.terminal_headline
        ));
        if !self.has_surface_unit(&run.body_id, SurfaceUnitKind::AtmosphericEntryProbe) {
            self.surface_deployments
                .entry(run.body_id.clone())
                .or_default()
                .push(SurfaceDeployment {
                    body_id: run.body_id.clone(),
                    body_name: run.body_name.clone(),
                    kind: SurfaceUnitKind::AtmosphericEntryProbe,
                    telemetry_headline: run.terminal_headline.clone(),
                    telemetry_report: run.terminal_report.clone(),
                    relay_pucks_deployed: 0,
                    ice_melt_depth_km: 0.0,
                    active_experiment_done: true,
                    descent_progress: 1.0,
                });
        }
        self.evaluate_achievements()
    }

    pub fn run_active_surface_experiment(
        &mut self,
        body_id: &str,
        kind: SurfaceUnitKind,
    ) -> Result<u64, String> {
        let Some(list) = self.surface_deployments.get_mut(body_id) else {
            return Err("No surface unit deployed on this body.".into());
        };
        let Some(unit) = list.iter_mut().find(|u| u.kind == kind) else {
            return Err("Requested unit is not deployed on this body.".into());
        };
        if unit.active_experiment_done {
            return Err("In-situ active experiment already completed for this unit.".into());
        }
        unit.active_experiment_done = true;
        let bonus = match kind {
            SurfaceUnitKind::AeroShellLander => 55,
            SurfaceUnitKind::OceanCryobot => 75,
            SurfaceUnitKind::AtmosphericEntryProbe => 0,
        };
        let body_name = unit.body_name.clone();
        self.science_points += bonus;
        self.push_log(format!(
            "[IN-SITU EXPERIMENT] Executed active {} pulse on {} (+{} SP).",
            kind.label(),
            body_name,
            bonus
        ));
        Ok(bonus)
    }

    pub fn launch_sample_return(
        &mut self,
        target: &CelestialBody,
        catalog: &[CelestialBody],
    ) -> Result<SampleReturnCapsule, String> {
        if target.id == "earth"
            || target.id == "sun"
            || matches!(target.category, BodyCategory::Nebula | BodyCategory::LagrangePoint)
        {
            return Err(format!(
                "Cannot launch an Earth Sample-Return Capsule from {}.",
                target.name
            ));
        }
        if !self.has_orbiting_probe_at(&target.id) {
            return Err(format!(
                "Requires an active orbiter at {} to ascend and seal a Sample-Return Capsule.",
                target.name
            ));
        }
        let cid = self.next_capsule_id.max(1);
        self.next_capsule_id = cid + 1;
        let cap =
            SampleReturnCapsule::launch_to_earth(cid, target, catalog, self.days_since_j2000);
        self.science_points += 40;
        self.push_log(format!(
            "[SAMPLE RETURN] Capsule SRC-{cid} departed {} with {:.1} g sample toward Earth (+40 SP).",
            target.name, cap.sample_mass_grams
        ));
        self.sample_capsules.push(cap.clone());
        Ok(cap)
    }

    pub fn record_capsule_recovery(
        &mut self,
        origin_name: &str,
        mass_grams: f64,
    ) -> Vec<&'static AchievementDef> {
        self.recovered_samples_count += 1;
        let pts = sample_return_points(&self.unlocked_techs);
        self.science_points += pts;
        self.push_log(format!(
            "[CURATORY] Recovered {:.1} g from {} at Utah/Pacific splashdown! +{} SP",
            mass_grams, origin_name, pts
        ));
        self.evaluate_achievements()
    }

    pub fn register_imported_body(
        &mut self,
        body: CelestialBody,
    ) -> Vec<&'static AchievementDef> {
        self.science_points += 100;
        self.push_log(format!(
            "Imported public astronomy target '{}' ({}). +100 SP",
            body.name, body.source_title
        ));
        if !self.imported_bodies.iter().any(|b| b.id == body.id) {
            self.imported_bodies.push(body);
        }
        self.evaluate_achievements()
    }

    pub fn deploy_l2_observatory(
        &mut self,
        obs_id: &str,
    ) -> Result<(&'static crate::l2_comets::L2ObservatoryDef, Vec<&'static AchievementDef>), String> {
        let Some(def) = crate::l2_comets::L2_OBSERVATORIES
            .iter()
            .find(|o| o.id == obs_id)
        else {
            return Err(format!("Unknown L2 observatory '{obs_id}'."));
        };
        if !self.deployed_l2_observatories.insert(def.id.to_string()) {
            return Err(format!("{} is already operating in Sun-Earth L2 halo orbit.", def.name));
        }
        self.orbited_bodies.insert("sun_earth_l2".to_string());
        self.science_points += def.deploy_reward_sp;
        self.push_log(format!(
            "[SUN-EARTH L2] Commissioned {} into 1.5M km halo orbit! +{} SP",
            def.name, def.deploy_reward_sp
        ));
        let achs = self.evaluate_achievements();
        Ok((def, achs))
    }

    pub fn run_roman_wide_field_survey(&mut self) -> Result<u64, String> {
        if !self.deployed_l2_observatories.contains("roman_l2") {
            return Err("Deploy the Nancy Grace Roman Space Telescope to Sun-Earth L2 first.".into());
        }
        if self.roman_survey_completed {
            return Err("Roman Wide-Field High-Latitude & Microlensing Survey already completed.".into());
        }
        self.roman_survey_completed = true;
        let bonus = 180;
        self.science_points += bonus;
        self.push_log(format!(
            "[ROMAN L2 WFI] Completed 100x-Hubble-FOV Wide-Field Sky Survey across comets, KBOs & microlensed exoplanets (+{bonus} SP)."
        ));
        let _ = self.evaluate_achievements();
        Ok(bonus)
    }

    pub fn warp_comet_to_perihelion(&mut self, comet: &CelestialBody) -> Result<f64, String> {
        if comet.category != BodyCategory::Comet || comet.orbital_period_days <= 0.0 {
            return Err(format!("{} is not a periodic solar-system comet.", comet.name));
        }
        let cur_m_deg = comet.mean_anomaly_j2000_deg
            + 360.0 * (self.days_since_j2000 / comet.orbital_period_days);
        let desired_m_deg = 6.5;
        let offset = (desired_m_deg - cur_m_deg).rem_euclid(360.0);
        self.comet_perihelion_offsets.insert(comet.id.clone(), offset);
        let q_au = comet.semi_major_axis_au * (1.0 - comet.eccentricity);
        self.push_log(format!(
            "[COMET EPHEMERIS] Synchronized {} to inner-system perihelion pass (q = {:.3} AU)!",
            comet.name, q_au
        ));
        Ok(q_au)
    }

    pub fn evaluate_achievements(&mut self) -> Vec<&'static AchievementDef> {
        let mut newly_unlocked = Vec::new();
        let has_any_scan = |id: &str| self.scanned_count_for(id) > 0;

        let dwarf_count = ["ceres", "pluto", "haumea", "makemake", "eris"]
            .iter()
            .filter(|id| has_any_scan(id))
            .count();
        let comet_scanned = ["encke_2p", "churyumov_67p", "halley_1p", "hale_bopp_c1995"]
            .iter()
            .any(|id| self.orbited_bodies.contains(*id) && has_any_scan(id))
            || self.imported_bodies.iter().any(|b| {
                b.category == BodyCategory::Comet
                    && self.orbited_bodies.contains(&b.id)
                    && has_any_scan(&b.id)
            });

        let checks: [(&str, bool); 21] = [
            ("first_launch", !self.probes.is_empty()),
            ("cislunar_pathfinder", self.orbited_bodies.contains("moon")),
            (
                "inner_rocky_triad",
                has_any_scan("mercury") && has_any_scan("venus") && has_any_scan("mars"),
            ),
            (
                "osiris_bennu",
                self.is_band_scanned("bennu", ScanBand::ThermalMassSpec),
            ),
            (
                "ocean_worlds",
                has_any_scan("europa") && has_any_scan("enceladus") && has_any_scan("ceres"),
            ),
            (
                "huygens_titan",
                self.orbited_bodies.contains("titan")
                    && self.is_band_scanned("titan", ScanBand::ThermalMassSpec),
            ),
            (
                "ice_giant_voyager",
                self.orbited_bodies.contains("uranus") && self.orbited_bodies.contains("neptune"),
            ),
            ("kuiper_horizon", self.orbited_bodies.contains("pluto")),
            (
                "exoplanet_relay",
                has_any_scan("proxima_b") || has_any_scan("trappist_1e"),
            ),
            (
                "stellar_nursery",
                has_any_scan("m42_orion")
                    && has_any_scan("m16_eagle")
                    && has_any_scan("carina_ngc3372"),
            ),
            (
                "crab_pulsar_1054",
                self.is_band_scanned("m1_crab", ScanBand::RadarGravity)
                    || self.is_band_scanned("m1_crab", ScanBand::MagnetometerPlasma),
            ),
            ("open_data_archivist", !self.imported_bodies.is_empty()),
            (
                "galilean_tour",
                has_any_scan("io")
                    && has_any_scan("europa")
                    && has_any_scan("ganymede")
                    && has_any_scan("callisto"),
            ),
            ("dwarf_pentad", dwarf_count >= 3),
            (
                "metal_and_lava",
                has_any_scan("psyche") && has_any_scan("cancri_55e"),
            ),
            ("gravity_slingshot", self.completed_slingshots > 0),
            ("subsurface_cryobot", self.total_cryobots_deployed() > 0),
            ("earth_sample_return", self.recovered_samples_count > 0),
            ("atmospheric_plunge", self.completed_atmo_plunges > 0),
            ("l2_halo_array", !self.deployed_l2_observatories.is_empty()),
            ("comet_tail_chaser", comet_scanned),
        ];

        for (id, passed) in checks {
            if passed && self.unlocked_achievements.insert(id.to_string()) {
                if let Some(def) = ACHIEVEMENTS.iter().find(|a| a.id == id) {
                    self.science_points += def.bonus_points as u64;
                    self.push_log(format!(
                        "ACHIEVEMENT UNLOCKED: {} (+{} SP)",
                        def.title, def.bonus_points
                    ));
                    newly_unlocked.push(def);
                }
            }
        }
        newly_unlocked
    }
}

#[derive(Debug, Clone)]
pub struct SaveSlotSummary {
    pub slot_id: i64,
    pub campaign_name: String,
    pub science_points: u64,
    pub updated_utc: String,
}

pub struct SaveDatabase {
    conn: Connection,
}

impl SaveDatabase {
    pub fn open(path: &Path) -> Result<Self, String> {
        let conn = Connection::open(path).map_err(|e| format!("SQLite open error: {e}"))?;
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             CREATE TABLE IF NOT EXISTS save_slots (
                 slot_id INTEGER PRIMARY KEY,
                 campaign_name TEXT NOT NULL,
                 science_points INTEGER NOT NULL,
                 updated_utc TEXT NOT NULL,
                 payload_json TEXT NOT NULL
             );",
        )
        .map_err(|e| format!("SQLite schema init error: {e}"))?;
        Ok(Self { conn })
    }

    pub fn save_slot(&self, slot_id: i64, state: &CampaignState) -> Result<(), String> {
        let payload =
            serde_json::to_string_pretty(state).map_err(|e| format!("Save JSON serialize: {e}"))?;
        let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string();
        self.conn
            .execute(
                "INSERT INTO save_slots (slot_id, campaign_name, science_points, updated_utc, payload_json)
                 VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(slot_id) DO UPDATE SET
                     campaign_name = excluded.campaign_name,
                     science_points = excluded.science_points,
                     updated_utc = excluded.updated_utc,
                     payload_json = excluded.payload_json",
                params![
                    slot_id,
                    state.campaign_name,
                    state.science_points as i64,
                    now,
                    payload
                ],
            )
            .map_err(|e| format!("SQLite save slot error: {e}"))?;
        Ok(())
    }

    pub fn load_slot(&self, slot_id: i64) -> Result<CampaignState, String> {
        let payload: String = self
            .conn
            .query_row(
                "SELECT payload_json FROM save_slots WHERE slot_id = ?1",
                params![slot_id],
                |row| row.get(0),
            )
            .map_err(|e| format!("Save slot {slot_id} not found ({e})"))?;
        serde_json::from_str(&payload).map_err(|e| format!("Save JSON decode error: {e}"))
    }

    pub fn list_slots(&self) -> Result<Vec<SaveSlotSummary>, String> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT slot_id, campaign_name, science_points, updated_utc
                 FROM save_slots ORDER BY slot_id ASC",
            )
            .map_err(|e| format!("SQLite list slots error: {e}"))?;
        let rows = stmt
            .query_map([], |row| {
                Ok(SaveSlotSummary {
                    slot_id: row.get(0)?,
                    campaign_name: row.get(1)?,
                    science_points: row.get::<_, i64>(2)?.max(0) as u64,
                    updated_utc: row.get(3)?,
                })
            })
            .map_err(|e| format!("SQLite query slots error: {e}"))?;
        let mut out = Vec::new();
        for r in rows {
            if let Ok(item) = r {
                out.push(item);
            }
        }
        Ok(out)
    }
}
