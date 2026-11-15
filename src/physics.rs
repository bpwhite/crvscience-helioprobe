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
                "moon" => 2.55,
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
        _ => (0.0062 * body.radius_km.max(0.2).powf(0.56)).clamp(0.045, 3.25),
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
