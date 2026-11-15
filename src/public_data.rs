//! Live public astronomical data harvesters and pure-Rust parsers for:
//! 1. NASA JPL Small-Body Database (SBDB) API (`https://ssd-api.jpl.nasa.gov/sbdb.api`)
//! 2. NASA JPL Horizons Ephemeris API (`https://ssd.jpl.nasa.gov/api/horizons.api`)
//! 3. NASA Exoplanet Archive TAP Service (`https://exoplanetarchive.ipac.caltech.edu/TAP/sync`)
//! 4. CDS SIMBAD TAP Astronomical Database (`https://simbad.cds.unistra.fr/simbad/sim-tap/sync`)

use crate::catalog::{BandFinding, BodyCategory, CelestialBody, ScanBand};
use serde_json::Value;
use std::time::Duration;

fn percent_encode_query(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len() * 2);
    for b in raw.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            b' ' => out.push_str("%20"),
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}

/// Fetches and parses a real asteroid or comet from the NASA JPL Small-Body Database (SBDB) API.
pub fn fetch_jpl_sbdb_small_body(query: &str) -> Result<CelestialBody, String> {
    let encoded = percent_encode_query(query.trim());
    let url = format!(
        "https://ssd-api.jpl.nasa.gov/sbdb.api?sstr={}&phys-par=1&discovery=1",
        encoded
    );
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(12))
        .user_agent("HelioProbe-Rust-Sim/1.0 (https://crvscience.com)")
        .build();

    let json: Value = agent
        .get(&url)
        .call()
        .map_err(|e| format!("JPL SBDB HTTP error: {e}"))?
        .into_json()
        .map_err(|e| format!("JPL SBDB JSON decode error: {e}"))?;

    parse_jpl_sbdb_json(&json, query)
}

pub fn parse_jpl_sbdb_json(json: &Value, fallback_query: &str) -> Result<CelestialBody, String> {
    if let Some(code) = json.get("code").and_then(|v| v.as_i64()) {
        if code == 300 {
            let candidates = json
                .get("list")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .take(5)
                        .filter_map(|item| item.get("name").and_then(|n| n.as_str()))
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();
            return Err(format!(
                "Multiple SBDB matches found ({candidates}). Try an exact IAU number or designation."
            ));
        }
    }
    let obj = json
        .get("object")
        .ok_or_else(|| "SBDB response missing 'object' record".to_string())?;
    let fullname = obj
        .get("fullname")
        .or_else(|| obj.get("shortname"))
        .and_then(|v| v.as_str())
        .unwrap_or(fallback_query)
        .to_string();
    let des = obj
        .get("des")
        .and_then(|v| v.as_str())
        .unwrap_or(fallback_query)
        .to_string();
    let spkid = obj
        .get("spkid")
        .and_then(|v| v.as_str())
        .unwrap_or("SBDB")
        .to_string();
    let orbit_class = obj
        .get("orbit_class")
        .and_then(|oc| oc.get("name"))
        .and_then(|v| v.as_str())
        .unwrap_or("Small Solar System Body");

    let mut e = 0.15;
    let mut a = 2.4;
    let mut i = 5.0;
    let mut om = 80.0;
    let mut w = 110.0;
    let mut ma = 45.0;
    let mut per = 1350.0;

    if let Some(elements) = json
        .get("orbit")
        .and_then(|o| o.get("elements"))
        .and_then(|el| el.as_array())
    {
        for el in elements {
            let name = el.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let val = el
                .get("value")
                .and_then(|v| v.as_str().and_then(|s| s.parse::<f64>().ok()).or_else(|| v.as_f64()));
            if let Some(val) = val {
                match name {
                    "e" => e = val,
                    "a" => a = val,
                    "i" => i = val,
                    "om" => om = val,
                    "w" => w = val,
                    "ma" => ma = val,
                    "per" => per = val,
                    _ => {}
                }
            }
        }
    }

    let mut diameter_km = 5.0;
    let mut rot_per_hr = 8.0;
    let mut abs_mag_h = 15.0;
    if let Some(phys) = json.get("phys_par").and_then(|v| v.as_array()) {
        for p in phys {
            let pname = p.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let pval = p
                .get("value")
                .and_then(|v| v.as_str().and_then(|s| s.parse::<f64>().ok()).or_else(|| v.as_f64()));
            if let Some(val) = pval {
                match pname {
                    "diameter" => diameter_km = val,
                    "rot_per" => rot_per_hr = val,
                    "H" => abs_mag_h = val,
                    _ => {}
                }
            }
        }
    }

    let disc_summary = json
        .get("discovery")
        .map(|d| {
            let who = d.get("who").and_then(|v| v.as_str()).unwrap_or("MPC/JPL observers");
            let date = d.get("date").and_then(|v| v.as_str()).unwrap_or("archived epoch");
            let loc = d.get("location").and_then(|v| v.as_str()).unwrap_or("ground/space survey");
            format!("Discovered {date} by {who} at {loc}.")
        })
        .unwrap_or_else(|| "Cataloged in the JPL/MPC Small-Body Database.".to_string());

    let id = format!(
        "sbdb_{}",
        des.chars()
            .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '_' })
            .collect::<String>()
    );
    let lookup_url = format!(
        "https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr={}",
        percent_encode_query(&des)
    );

    let oc_lower = orbit_class.to_ascii_lowercase();
    let is_comet = oc_lower.contains("comet")
        || oc_lower.contains("encke")
        || oc_lower.contains("halley")
        || oc_lower.contains("jupiter-family")
        || oc_lower.contains("chiron")
        || oc_lower.contains("parabolic")
        || oc_lower.contains("hyperbolic")
        || des.starts_with("C/")
        || des.contains("P/");

    Ok(CelestialBody {
        id,
        name: fullname.clone(),
        designation: format!("SPK {spkid} ({orbit_class})"),
        category: if is_comet { BodyCategory::Comet } else { BodyCategory::Asteroid },
        parent_id: None,
        semi_major_axis_au: a,
        eccentricity: e,
        inclination_deg: i,
        lon_ascending_node_deg: om,
        arg_periapsis_deg: w,
        mean_anomaly_j2000_deg: ma,
        orbital_period_days: per,
        distance_ly: 0.0,
        radius_km: (diameter_km * 0.5).max(0.1),
        span_ly: 0.0,
        mass_10e24_kg: 0.0,
        density_kg_m3: 2000.0,
        gravity_m_s2: 0.005,
        escape_velocity_km_s: 0.02,
        mean_temp_c: (278.0 / a.max(0.1).sqrt()) - 273.15,
        color_rgb: [0.66, 0.72, 0.80],
        has_rings: false,
        has_global_mag_field: false,
        known_moons: 0,
        source_url: lookup_url,
        source_title: "NASA JPL Small-Body Database (SBDB)".into(),
        imported_from_api: true,
        findings: vec![
            BandFinding {
                band: ScanBand::OpticalNir,
                headline: format!("{orbit_class} Orbit (H = {abs_mag_h:.2} mag)"),
                detail: format!("{disc_summary} Semi-major axis a = {a:.4} AU, eccentricity e = {e:.4}, inclination i = {i:.2}°."),
            },
            BandFinding {
                band: ScanBand::RadarGravity,
                headline: format!("Estimated Diameter {diameter_km:.2} km & {rot_per_hr:.2} h Spin"),
                detail: format!("Orbital period is {per:.1} days with ascending node {om:.2}° and argument of perihelion {w:.2}°."),
            },
            BandFinding {
                band: ScanBand::MagnetometerPlasma,
                headline: "Solar-Wind Unmagnetized Regolith Wake".into(),
                detail: "Direct interplanetary magnetic field draping measured across the small-body wake.".into(),
            },
            BandFinding {
                band: ScanBand::ThermalMassSpec,
                headline: "Radiative Equilibrium Regolith Thermophysics".into(),
                detail: format!("Equilibrium blackbody temperature near {:.1} °C at {a:.3} AU heliocentric distance.", (278.0 / a.max(0.1).sqrt()) - 273.15),
            },
            BandFinding {
                band: ScanBand::HighEnergyUvXray,
                headline: "Solar X-Ray & Cosmic-Ray Regolith Sputtering".into(),
                detail: "Surface silicate and carbonaceous grains exhibit space-weathering spectral reddening.".into(),
            },
        ],
    })
}

/// Fetches a confirmed exoplanet from the NASA Exoplanet Archive TAP ADQL service.
pub fn fetch_nasa_exoplanet(planet_name: &str) -> Result<CelestialBody, String> {
    let sanitized = planet_name.replace('\'', "");
    let adql = format!(
        "select top 1 pl_name,hostname,sy_dist,pl_orbper,pl_orbsmax,pl_orbeccen,pl_bmasse,pl_rade,discoverymethod,disc_year from pscomppars where pl_name like '%{}%'",
        sanitized.trim()
    );
    let url = format!(
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query={}&format=json",
        percent_encode_query(&adql)
    );
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(12))
        .user_agent("HelioProbe-Rust-Sim/1.0 (https://crvscience.com)")
        .build();

    let rows: Value = agent
        .get(&url)
        .call()
        .map_err(|e| format!("NASA Exoplanet Archive TAP HTTP error: {e}"))?
        .into_json()
        .map_err(|e| format!("NASA Exoplanet Archive JSON error: {e}"))?;

    parse_exoplanet_tap_json(&rows)
}

pub fn parse_exoplanet_tap_json(rows: &Value) -> Result<CelestialBody, String> {
    let first = rows
        .as_array()
        .and_then(|a| a.first())
        .ok_or_else(|| "No matching exoplanet returned by NASA Exoplanet Archive TAP.".to_string())?;

    let pl_name = first
        .get("pl_name")
        .and_then(|v| v.as_str())
        .unwrap_or("Imported Exoplanet");
    let host = first
        .get("hostname")
        .and_then(|v| v.as_str())
        .unwrap_or("Host Star");
    let dist_pc = first.get("sy_dist").and_then(|v| v.as_f64()).unwrap_or(15.0);
    let dist_ly = dist_pc * 3.26156;
    let orbper = first.get("pl_orbper").and_then(|v| v.as_f64()).unwrap_or(12.0);
    let orbsmax = first.get("pl_orbsmax").and_then(|v| v.as_f64()).unwrap_or(0.1);
    let ecc = first.get("pl_orbeccen").and_then(|v| v.as_f64()).unwrap_or(0.02);
    let bmasse = first.get("pl_bmasse").and_then(|v| v.as_f64()).unwrap_or(1.5);
    let rade = first.get("pl_rade").and_then(|v| v.as_f64()).unwrap_or(1.15);
    let method = first
        .get("discoverymethod")
        .and_then(|v| v.as_str())
        .unwrap_or("Transit / RV");
    let disc_year = first.get("disc_year").and_then(|v| v.as_i64()).unwrap_or(2020);

    let id = format!(
        "exo_{}",
        pl_name
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '_' })
            .collect::<String>()
    );
    let hash = pl_name.bytes().fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
    let lon_node = (hash % 360) as f64;
    let inc = ((hash / 7) % 50) as f64 - 25.0;

    Ok(CelestialBody {
        id,
        name: pl_name.to_string(),
        designation: format!("{host} System ({disc_year})"),
        category: BodyCategory::Exoplanet,
        parent_id: None,
        semi_major_axis_au: 56.0,
        eccentricity: ecc,
        inclination_deg: inc,
        lon_ascending_node_deg: lon_node,
        arg_periapsis_deg: 45.0,
        mean_anomaly_j2000_deg: (hash % 360) as f64,
        orbital_period_days: orbper,
        distance_ly: dist_ly,
        radius_km: rade * 6371.0,
        span_ly: 0.0,
        mass_10e24_kg: bmasse * 5.972,
        density_kg_m3: (5514.0 * bmasse / rade.powi(3).max(0.05)).clamp(400.0, 12_000.0),
        gravity_m_s2: 9.8 * bmasse / (rade * rade).max(0.05),
        escape_velocity_km_s: 11.2 * (bmasse / rade.max(0.05)).sqrt(),
        mean_temp_c: 15.0,
        color_rgb: [0.42, 0.78, 0.92],
        has_rings: false,
        has_global_mag_field: false,
        known_moons: 0,
        source_url: "https://caltech-ipac.github.io/nexsciTAP/intro.html".into(),
        source_title: "NASA Exoplanet Archive TAP Service".into(),
        imported_from_api: true,
        findings: vec![
            BandFinding {
                band: ScanBand::OpticalNir,
                headline: format!("{method} Discovery ({disc_year}) at {dist_ly:.1} Light-Years"),
                detail: format!("Orbits host star {host} at {orbsmax:.4} AU with an orbital period of {orbper:.2} days (e = {ecc:.3})."),
            },
            BandFinding {
                band: ScanBand::RadarGravity,
                headline: format!("{bmasse:.2} Earth Masses & {rade:.2} Earth Radii"),
                detail: format!("Derived bulk density is {:.0} kg/m^3 with surface gravity {:.2} m/s^2.", (5514.0 * bmasse / rade.powi(3).max(0.05)).clamp(400.0, 12_000.0), 9.8 * bmasse / (rade * rade).max(0.05)),
            },
            BandFinding {
                band: ScanBand::MagnetometerPlasma,
                headline: "Host-Star Stellar Wind Bow-Shock Envelope".into(),
                detail: format!("At {orbsmax:.4} AU from {host}, magnetic pressure balances stellar coronal wind ram pressure."),
            },
            BandFinding {
                band: ScanBand::ThermalMassSpec,
                headline: "Atmospheric Scale-Height & Equilibrium Spectroscopy".into(),
                detail: "Infrared emission and transit limb spectroscopy constrain atmospheric mean molecular weight.".into(),
            },
            BandFinding {
                band: ScanBand::HighEnergyUvXray,
                headline: "Exospheric Escape & Stellar XUV Irradiation".into(),
                detail: "High-energy sensor monitors Lyman-alpha transit depth and upper-atmosphere photoevaporation.".into(),
            },
        ],
    })
}

/// Fetches a deep-sky nebula, cluster, or stellar remnant from the CDS SIMBAD TAP service.
pub fn fetch_simbad_deep_sky(identifier: &str) -> Result<CelestialBody, String> {
    let clean = identifier.replace('\'', "");
    let adql = format!(
        "SELECT TOP 1 basic.main_id, basic.ra, basic.dec, basic.otype_txt, basic.plx_value FROM basic JOIN ident ON ident.oidref = basic.oid WHERE ident.id = '{}'",
        clean.trim()
    );
    let url = format!(
        "https://simbad.cds.unistra.fr/simbad/sim-tap/sync?request=doQuery&lang=adql&format=json&query={}",
        percent_encode_query(&adql)
    );
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(12))
        .user_agent("HelioProbe-Rust-Sim/1.0 (https://crvscience.com)")
        .build();

    let json: Value = agent
        .get(&url)
        .call()
        .map_err(|e| format!("CDS SIMBAD TAP HTTP error: {e}"))?
        .into_json()
        .map_err(|e| format!("CDS SIMBAD TAP JSON error: {e}"))?;

    parse_simbad_tap_json(&json, identifier)
}

pub fn parse_simbad_tap_json(json: &Value, fallback_id: &str) -> Result<CelestialBody, String> {
    let row = json
        .get("data")
        .and_then(|d| d.as_array())
        .and_then(|arr| arr.first())
        .and_then(|r| r.as_array())
        .ok_or_else(|| format!("SIMBAD returned no match for '{fallback_id}'."))?;

    let main_id = row
        .first()
        .and_then(|v| v.as_str())
        .unwrap_or(fallback_id)
        .trim()
        .to_string();
    let ra_deg = row.get(1).and_then(|v| v.as_f64()).unwrap_or(83.8);
    let dec_deg = row.get(2).and_then(|v| v.as_f64()).unwrap_or(-5.4);
    let otype = row
        .get(3)
        .and_then(|v| v.as_str())
        .unwrap_or("Deep-Sky Object");
    let plx_mas = row.get(4).and_then(|v| v.as_f64()).unwrap_or(0.5);
    let dist_ly = if plx_mas > 0.01 {
        (1000.0 / plx_mas) * 3.26156
    } else {
        2500.0
    };

    let id = format!(
        "simbad_{}",
        main_id
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '_' })
            .collect::<String>()
    );

    Ok(CelestialBody {
        id,
        name: main_id.clone(),
        designation: format!("SIMBAD {otype} (RA {ra_deg:.2}°, Dec {dec_deg:.2}°)"),
        category: BodyCategory::Nebula,
        parent_id: None,
        semi_major_axis_au: 80.0,
        eccentricity: 0.0,
        inclination_deg: dec_deg.clamp(-65.0, 65.0),
        lon_ascending_node_deg: ra_deg.rem_euclid(360.0),
        arg_periapsis_deg: 0.0,
        mean_anomaly_j2000_deg: ra_deg.rem_euclid(360.0),
        orbital_period_days: 0.0,
        distance_ly: dist_ly,
        radius_km: 2.0e13,
        span_ly: 6.0,
        mass_10e24_kg: 0.0,
        density_kg_m3: 0.0,
        gravity_m_s2: 0.0,
        escape_velocity_km_s: 0.0,
        mean_temp_c: 8500.0,
        color_rgb: [0.76, 0.48, 0.96],
        has_rings: false,
        has_global_mag_field: false,
        known_moons: 0,
        source_url: format!(
            "https://simbad.cds.unistra.fr/simbad/sim-id?Ident={}",
            percent_encode_query(&main_id)
        ),
        source_title: "CDS SIMBAD Astronomical Database".into(),
        imported_from_api: true,
        findings: vec![
            BandFinding {
                band: ScanBand::OpticalNir,
                headline: format!("SIMBAD Cataloged {otype} ({main_id})"),
                detail: format!("ICRS coordinates RA = {ra_deg:.4}°, Dec = {dec_deg:.4}°, estimated distance ~{dist_ly:.0} light-years."),
            },
            BandFinding {
                band: ScanBand::RadarGravity,
                headline: "3D Interstellar Astrometric Frame Resolved".into(),
                detail: "Parallax and proper-motion reference tied to Gaia/SIMBAD astrometric catalog.".into(),
            },
            BandFinding {
                band: ScanBand::MagnetometerPlasma,
                headline: "Interstellar Medium Magnetic Filament Coupling".into(),
                detail: "Ionized plasma density fluctuations mapped across the halo observation orbit.".into(),
            },
            BandFinding {
                band: ScanBand::ThermalMassSpec,
                headline: "Interstellar Dust & Molecular Emission Continuum".into(),
                detail: "Mid-infrared thermal dust continuum and emission lines recorded by onboard spectrometer.".into(),
            },
            BandFinding {
                band: ScanBand::HighEnergyUvXray,
                headline: "High-Energy Ionization & Synchrotron Survey".into(),
                detail: "UV and X-ray flux calibrated against CDS SIMBAD multi-wavelength cross-identifications.".into(),
            },
        ],
    })
}

/// Queries live osculating heliocentric range (`delta`) from the NASA JPL Horizons API (`horizons.api`)
/// for a major body ID (e.g. `"499"` for Mars) and returns a formatted live ephemeris summary line.
pub fn fetch_jpl_horizons_ephemeris_note(command_id: &str) -> Result<String, String> {
    let url = format!(
        "https://ssd.jpl.nasa.gov/api/horizons.api?format=json&COMMAND='{}'&OBJ_DATA='YES'&MAKE_EPHEM='NO'",
        percent_encode_query(command_id)
    );
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(12))
        .user_agent("HelioProbe-Rust-Sim/1.0 (https://crvscience.com)")
        .build();
    let json: Value = agent
        .get(&url)
        .call()
        .map_err(|e| format!("JPL Horizons HTTP error: {e}"))?
        .into_json()
        .map_err(|e| format!("JPL Horizons JSON error: {e}"))?;
    let result_text = json
        .get("result")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "JPL Horizons payload missing 'result' text".to_string())?;
    let snippet = result_text
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.contains("***"))
        .take(5)
        .collect::<Vec<_>>()
        .join(" | ");
    Ok(snippet)
}
