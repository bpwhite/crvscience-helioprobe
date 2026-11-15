#![forbid(unsafe_code)]

use clap::{Parser, Subcommand};
use eframe::egui;
use helioprobe::astrodynamics_lab::{
    compute_gravity_assist, compute_porkchop_grid, ASSIST_CANDIDATE_IDS,
};
use helioprobe::catalog::bootstrap_catalog;
use helioprobe::gamification::SaveDatabase;
use helioprobe::gui::HelioProbeApp;
use helioprobe::public_data::{
    fetch_jpl_sbdb_small_body, fetch_nasa_exoplanet, fetch_simbad_deep_sky,
};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "helioprobe",
    version = "2.3.0",
    about = "Native Rust 3D Space Probe, Porkchop Gravity-Assist, Lander/Cryobot POV, Atmospheric Entry Probe & Public Astronomy Simulator"
)]
struct Cli {
    /// Path to SQLite WAL campaign database
    #[arg(long, default_value = "helioprobe_saves.db")]
    db: PathBuf,

    /// Render an offscreen 1600x1000 RGBA PNG screenshot without opening a GUI window
    #[arg(long)]
    headless_screenshot: Option<PathBuf>,

    /// Select preset view (1, 2, or 3) for --headless-screenshot
    #[arg(long, default_value_t = 1)]
    screenshot_view: u8,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Query and print a small body from the live NASA JPL Small-Body Database (SBDB) API
    SyncSbdb { query: String },
    /// Query and print a confirmed exoplanet from the NASA Exoplanet Archive TAP service
    SyncExoplanet { name: String },
    /// Query and print a nebula or deep-sky target from CDS SIMBAD TAP
    SyncSimbad { identifier: String },
    /// Compute optimal Porkchop launch window and Gravity-Assist Slingshot solutions for a catalog target ID
    Porkchop {
        #[arg(default_value = "saturn")]
        target_id: String,
    },
    /// List saved campaigns stored in the local SQLite WAL database
    Slots,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    if let Some(cmd) = cli.command {
        match cmd {
            Commands::SyncSbdb { query } => {
                let body = fetch_jpl_sbdb_small_body(&query)?;
                println!("{}", serde_json::to_string_pretty(&body)?);
            }
            Commands::SyncExoplanet { name } => {
                let body = fetch_nasa_exoplanet(&name)?;
                println!("{}", serde_json::to_string_pretty(&body)?);
                }
            Commands::SyncSimbad { identifier } => {
                let body = fetch_simbad_deep_sky(&identifier)?;
                println!("{}", serde_json::to_string_pretty(&body)?);
            }
            Commands::Porkchop { target_id } => {
                let catalog = bootstrap_catalog();
                let target = catalog
                    .iter()
                    .find(|b| b.id == target_id.to_lowercase())
                    .ok_or_else(|| format!("Unknown catalog target_id '{target_id}'"))?;
                let grid = compute_porkchop_grid(target, &catalog, 9814.5, 28, 20);
                println!(
                    "Target: {} ({}) | Synodic Period: {:.1} d | Phase Error: {:+.1} deg",
                    target.name, target.designation, grid.synodic_period_days, grid.phase_error_deg
                );
                println!(
                    "Optimal Porkchop Basin: Departure +{:.1} d | Flight {:.1} d | Min dV {:.2} km/s",
                    grid.best_dep_offset_days, grid.best_tof_days, grid.min_dv_km_s
                );
                for aid in ASSIST_CANDIDATE_IDS {
                    if let Some(ab) = catalog.iter().find(|b| b.id == aid) {
                        if let Some(sol) = compute_gravity_assist(ab, target, &catalog) {
                            println!(
                                "  Slingshot via {:7}: Assisted dV {:.2} km/s (Saved {:.2} km/s, {:.1} deg bend @ {:.0} km)",
                                sol.assist_name,
                                sol.assisted_dv_km_s,
                                sol.dv_saved_km_s,
                                sol.turn_angle_deg,
                                sol.periapsis_alt_km
                            );
                        }
                    }
                }
            }
            Commands::Slots => {
                let db = SaveDatabase::open(&cli.db)?;
                let slots = db.list_slots()?;
                if slots.is_empty() {
                    println!("No saved campaigns in {}", cli.db.display());
                } else {
                    for s in slots {
                        println!(
                            "Slot {}: {} | {} SP | {}",
                            s.slot_id, s.campaign_name, s.science_points, s.updated_utc
                        );
                    }
                }
            }
        }
        return Ok(());
    }

    if let Some(ref out_path) = cli.headless_screenshot {
        helioprobe::gui::render_headless_screenshot(&cli.db, out_path, cli.screenshot_view)?;
        return Ok(());
    }

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("HelioProbe v2.3.0 — 3D Space Probe, Sun-Earth L2 Telescopes, Dynamic Comets & Public Astronomy Simulator")
            .with_inner_size([1520.0, 920.0])
            .with_min_inner_size([1024.0, 680.0]),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };

    let db_path = cli.db.clone();
    eframe::run_native(
        "HelioProbe 3D",
        native_options,
        Box::new(move |cc| Ok(Box::new(HelioProbeApp::new(cc, db_path)))),
    )
    .map_err(|e| format!("eframe launch error: {e}").into())
}
