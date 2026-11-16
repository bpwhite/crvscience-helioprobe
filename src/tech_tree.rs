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

//! Science Point (`SP`) Spacecraft Tech Tree & Payload Upgrade Lab (`src/tech_tree.rs`):
//! - Tracks 8 researchable engineering & instrument upgrades across 3 branches
//!   (`Propulsion & Navigation`, `Orbital & Cryobot Payloads`, `Deep Space Network & Curatory`).
//! - Uses `spent_sp` against `science_points` (`available_sp = science_points - spent_sp`) so unlocking
//!   upgrades never reduces lifetime Agency Rank!

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TechBranch {
    PropulsionNav,
    PayloadsCryo,
    DsnCuratory,
}

impl TechBranch {
    pub fn label(self) -> &'static str {
        match self {
            Self::PropulsionNav => "Propulsion & Astrodynamics",
            Self::PayloadsCryo => "Orbital & Cryobot Payloads",
            Self::DsnCuratory => "Deep Space Network & Curatory",
        }
    }
}

#[derive(Debug, Clone)]
pub struct TechNodeDef {
    pub id: &'static str,
    pub branch: TechBranch,
    pub title: &'static str,
    pub cost_sp: u64,
    pub effect_summary: &'static str,
}

pub const TECH_TREE: [TechNodeDef; 8] = [
    TechNodeDef {
        id: "oberth_nav",
        branch: TechBranch::PropulsionNav,
        title: "Autonomous Oberth & Lambert Guidance",
        cost_sp: 150,
        effect_summary: "+25% probe cruise speed and +60 SP bonus on every Gravity-Assist Slingshot flyby.",
    },
    TechNodeDef {
        id: "next_ion_thruster",
        branch: TechBranch::PropulsionNav,
        title: "NEXT-C High-Isp Xenon Hall Array",
        cost_sp: 300,
        effect_summary: "+40% faster transit on Solar-Electric Hall Ion and RTG Nuclear-Electric buses.",
    },
    TechNodeDef {
        id: "beamed_sail_core",
        branch: TechBranch::PropulsionNav,
        title: "0.38c Beamed-Core Relativistic Sail",
        cost_sp: 600,
        effect_summary: "Boosts Laser-Sail cruise from 0.20c to 0.38c and halves transit time to exoplanets & nebulae.",
    },
    TechNodeDef {
        id: "quantum_gradio",
        branch: TechBranch::PayloadsCryo,
        title: "Cold-Atom Quantum Gravity Gradiometer",
        cost_sp: 200,
        effect_summary: "Increases every 5-band orbital scan reward from +60 SP to +95 SP (+58% science yield).",
    },
    TechNodeDef {
        id: "cryobot_gphs",
        branch: TechBranch::PayloadsCryo,
        title: "Pu-238 GPHS Deep-Vent Cryobot Nose",
        cost_sp: 350,
        effect_summary: "Doubles Ocean-World Cryobot ice-melt depth and boosts Cryobot reward from +240 SP to +420 SP.",
    },
    TechNodeDef {
        id: "coronagraph_jwst",
        branch: TechBranch::PayloadsCryo,
        title: "Segmented Starshade Coronagraph",
        cost_sp: 450,
        effect_summary: "Unlocks Remote Telescope Pre-Scan of Optical/NIR from Earth and +200 SP Exoplanet/Nebula survey bonus.",
    },
    TechNodeDef {
        id: "dsn_ka_band",
        branch: TechBranch::DsnCuratory,
        title: "Optical Laser & Ka-Band DSN Array",
        cost_sp: 250,
        effect_summary: "Unlocks automated Full 5-Band Auto-Scan immediately upon spacecraft orbital capture.",
    },
    TechNodeDef {
        id: "astromaterials_curatory",
        branch: TechBranch::DsnCuratory,
        title: "Cleanroom Isotope Sample Curatory",
        cost_sp: 400,
        effect_summary: "Boosts Earth Sample-Return Capsule recovery yield from +320 SP to +550 SP.",
    },
];

pub fn tech_speed_multiplier(unlocked: &BTreeSet<String>, bus_is_ion_or_rtg: bool, bus_is_sail: bool) -> f64 {
    let mut mult = 1.0;
    if unlocked.contains("oberth_nav") {
        mult *= 1.25;
    }
    if bus_is_ion_or_rtg && unlocked.contains("next_ion_thruster") {
        mult *= 1.40;
    }
    if bus_is_sail && unlocked.contains("beamed_sail_core") {
        mult *= 1.85;
    }
    mult
}

pub fn scan_band_points(unlocked: &BTreeSet<String>) -> u64 {
    if unlocked.contains("quantum_gradio") {
        95
    } else {
        60
    }
}

pub fn cryobot_points(unlocked: &BTreeSet<String>) -> u64 {
    if unlocked.contains("cryobot_gphs") {
        420
    } else {
        240
    }
}

pub fn sample_return_points(unlocked: &BTreeSet<String>) -> u64 {
    if unlocked.contains("astromaterials_curatory") {
        550
    } else {
        320
    }
}
