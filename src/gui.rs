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

//! `eframe` / `egui` 3D Mission Control GUI for HelioProbe v2.1:
//! - Left Dock: Celestial Catalog filter, Probe Launch Pad with Gravity-Assist Slingshot selector, 6 Camera Modes including `Lander / Bot POV`
//! - Center 3D Viewport: `wgpu` WGSL 3D space/nebula renderer + live Lander / Cryobot / Entry-Probe POV Telemetry HUD
//! - Right Dock: 7-tab Science & Engineering Console (`Scanner`, `Windows`, `Landers`, `Tech Lab`, `Awards`, `Public Data`, `Saves`)

use crate::astrodynamics_lab::{
    compute_gravity_assist, compute_porkchop_grid_with_offsets, ASSIST_CANDIDATE_IDS,
};
use crate::catalog::{bootstrap_catalog, BodyCategory, CelestialBody, ScanBand};
use crate::gamification::{CampaignState, SaveDatabase, ACHIEVEMENTS};
use crate::l2_comets::{compute_comet_tail_state, L2_OBSERVATORIES};
use crate::physics::{
    body_scene_position_with_offsets, compute_transfer_from_earth, probe_orbit_radius_scene,
    Vec3d,
};
use crate::probes::{ProbePhase, PropulsionBus, SpaceProbe};
use crate::public_data::{
    fetch_jpl_horizons_ephemeris_note, fetch_jpl_sbdb_small_body, fetch_nasa_exoplanet,
    fetch_simbad_deep_sky,
};
use crate::renderer_3d::{
    build_cosmos_frame, unit_world_pos, CosmosGpuPipeline, CosmosPaintCallback, OrbitCamera3D,
    ProjectedLabel,
};
use crate::surface_ops::{
    has_substantial_atmosphere, supports_atmospheric_or_crash_probe, supports_ocean_cryobot,
    supports_surface_lander, CapsulePhase, EntryProbePhase, SurfaceUnitKind,
};
use crate::tech_tree::{
    cryobot_points, sample_return_points, scan_band_points, TechBranch, TECH_TREE,
};
use chrono::{TimeZone, Utc};
use eframe::egui;
use eframe::egui_wgpu;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RightTab {
    ScannerFacts,
    L2AndComets,
    LaunchWindows,
    SurfaceOps,
    TechTree,
    Achievements,
    PublicDataImport,
    SaveLoad,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CameraTrackMode {
    FreeSolarOverview,
    LockSelectedBody,
    RideLatestProbe,
    OrbitView,
    LanderBotView,
}

fn porkchop_colormap(norm: f32) -> egui::Color32 {
    let t = norm.clamp(0.0, 1.0);
    if t < 0.25 {
        let u = t / 0.25;
        egui::Color32::from_rgb(
            (12.0 + u * 22.0) as u8,
            (95.0 + u * 135.0) as u8,
            (185.0 + u * 50.0) as u8,
        )
    } else if t < 0.55 {
        let u = (t - 0.25) / 0.30;
        egui::Color32::from_rgb(
            (34.0 + u * 195.0) as u8,
            (230.0 - u * 45.0) as u8,
            (235.0 - u * 175.0) as u8,
        )
    } else {
        let u = (t - 0.55) / 0.45;
        egui::Color32::from_rgb(
            (229.0 + u * 20.0) as u8,
            (185.0 - u * 145.0) as u8,
            (60.0 + u * 35.0) as u8,
        )
    }
}

#[derive(Debug, Clone)]
pub struct ScheduledPorkchopWindow {
    pub target_id: String,
    pub col: usize,
    pub row: usize,
    pub dep_offset_days: f64,
    pub tof_days: f64,
    pub dv_km_s: f64,
    pub c3_km2_s2: f64,
}

#[derive(Debug, Clone)]
pub enum PublicDataFetchResult {
    ImportedBody {
        source: &'static str,
        result: Result<CelestialBody, String>,
    },
    HorizonsNote(Result<String, String>),
}

/// Maps a pointer position inside `p_rect` to `(col, row)` in `PorkchopGrid`,
/// accounting for the bottom-to-top Y-axis inversion (`y_inv = rows - 1 - r`).
pub fn porkchop_cell_from_pointer(
    p_rect: egui::Rect,
    pos: egui::Pos2,
    cols: usize,
    rows: usize,
) -> Option<(usize, usize)> {
    if cols == 0 || rows == 0 || !p_rect.contains(pos) {
        return None;
    }
    let nx = ((pos.x - p_rect.left()) / p_rect.width().max(1.0)).clamp(0.0, 0.999_999);
    let ny = ((pos.y - p_rect.top()) / p_rect.height().max(1.0)).clamp(0.0, 0.999_999);
    let col = ((nx * cols as f32).floor() as usize).min(cols - 1);
    let row_from_top = ((ny * rows as f32).floor() as usize).min(rows - 1);
    let row = rows - 1 - row_from_top;
    Some((col, row))
}

/// Filters the catalog by case-insensitive substring (name, id, designation) and optional `BodyCategory`.
pub fn filter_catalog_bodies<'a>(
    catalog: &'a [CelestialBody],
    query: &str,
    category: Option<BodyCategory>,
) -> Vec<&'a CelestialBody> {
    let q = query.trim().to_ascii_lowercase();
    catalog
        .iter()
        .filter(|b| category.map_or(true, |cat| b.category == cat))
        .filter(|b| {
            q.is_empty()
                || b.name.to_ascii_lowercase().contains(&q)
                || b.id.to_ascii_lowercase().contains(&q)
                || b.designation.to_ascii_lowercase().contains(&q)
        })
        .collect()
}

pub fn export_campaign_json(state: &CampaignState, path: &Path) -> Result<(), String> {
    let txt = serde_json::to_string_pretty(state).map_err(|e| format!("JSON serialize error: {e}"))?;
    std::fs::write(path, txt).map_err(|e| format!("Failed writing {}: {e}", path.display()))
}

pub fn import_campaign_json(path: &Path) -> Result<CampaignState, String> {
    let txt = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed reading {}: {e}", path.display()))?;
    serde_json::from_str::<CampaignState>(&txt)
        .map_err(|e| format!("Invalid campaign JSON in {}: {e}", path.display()))
}

fn j2000_days_to_utc_string(days_since_j2000: f64) -> String {
    let secs = (days_since_j2000 * 86_400.0).round() as i64;
    let j2000_epoch = Utc.with_ymd_and_hms(2000, 1, 1, 12, 0, 0).single();
    if let Some(base) = j2000_epoch {
        let dt = base + chrono::Duration::seconds(secs);
        dt.format("%Y-%m-%d %H:%M UTC").to_string()
    } else {
        format!("J2000 +{days_since_j2000:.1} d")
    }
}

/// Prioritizes and de-overlaps 3D projected screen labels inside `rect`.
pub fn deoverlap_screen_labels<'a>(
    labels: &'a [ProjectedLabel],
    selected_id: &str,
    rect: egui::Rect,
) -> Vec<(&'a ProjectedLabel, egui::Pos2)> {
    let mut indexed: Vec<(u8, f64, &'a ProjectedLabel)> = labels
        .iter()
        .map(|l| {
            let prio = if l.body_id == selected_id && !l.text.starts_with("Probe ") {
                0
            } else if l.text.starts_with("Probe ")
                || matches!(l.body_id.as_str(), "sun" | "earth" | "moon" | "sun_earth_l2")
            {
                1
            } else if matches!(
                l.body_id.as_str(),
                "mercury" | "venus" | "mars" | "jupiter" | "saturn" | "uranus" | "neptune" | "pluto"
            ) || l.text.starts_with("Comet ")
            {
                2
            } else {
                3
            };
            (prio, l.depth, l)
        })
        .collect();
    indexed.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));

    let mut placed_boxes: Vec<egui::Rect> = Vec::with_capacity(indexed.len());
    let mut out: Vec<(&'a ProjectedLabel, egui::Pos2)> = Vec::with_capacity(indexed.len());

    for (prio, _, lbl) in indexed {
        let approx_w = (lbl.text.len() as f32 * 6.4 + 10.0).clamp(36.0, 210.0);
        let approx_h = 15.0_f32;
        let half_w = approx_w * 0.5;
        let cx = lbl
            .screen_pos
            .x
            .clamp(rect.left() + half_w + 4.0, (rect.right() - half_w - 4.0).max(rect.left() + half_w + 4.0));
        let base_y = (lbl.screen_pos.y - lbl.radius_px - 6.0)
            .clamp(rect.top() + approx_h + 4.0, (rect.bottom() - 8.0).max(rect.top() + approx_h + 4.0));

        let candidates = [
            base_y,
            (base_y - 16.0).max(rect.top() + approx_h + 4.0),
            (lbl.screen_pos.y + lbl.radius_px + 16.0).min(rect.bottom() - 8.0),
        ];

        let mut chosen: Option<egui::Pos2> = None;
        for cand_y in candidates {
            let anchor = egui::pos2(cx, cand_y);
            let box_rect = egui::Rect::from_min_size(
                egui::pos2(cx - half_w, cand_y - approx_h),
                egui::vec2(approx_w, approx_h),
            );
            let overlaps = placed_boxes.iter().any(|b| b.intersects(box_rect));
            if !overlaps || prio == 0 {
                placed_boxes.push(box_rect);
                chosen = Some(anchor);
                break;
            }
        }
        if let Some(pos) = chosen {
            out.push((lbl, pos));
        }
    }
    out
}

pub struct HelioProbeApp {
    catalog_base: Vec<CelestialBody>,
    pub state: CampaignState,
    db: Option<SaveDatabase>,
    db_path: PathBuf,
    db_error: Option<String>,
    camera: OrbitCamera3D,
    desired_cam_distance: f64,
    camera_mode: CameraTrackMode,
    lander_cam_initialized: bool,
    tracked_probe_id: Option<u64>,
    selected_body_id: String,
    catalog_search: String,
    catalog_category_filter: Option<BodyCategory>,
    selected_bus: PropulsionBus,
    selected_assist_id: String,
    probe_callsign_input: String,
    transit_warp: f64,
    ephemeris_days_per_sec: f64,
    paused: bool,
    show_orbits: bool,
    show_labels: bool,
    right_tab: RightTab,
    last_frame: Instant,
    anim_time_sec: f64,
    active_scan_pulse: Option<(String, Instant)>,
    toast_banner: Option<(String, Instant)>,
    sbdb_query: String,
    exoplanet_query: String,
    simbad_query: String,
    horizons_cmd: String,
    api_status_msg: String,
    json_export_path: String,
    scheduled_window: Option<ScheduledPorkchopWindow>,
    pending_fetch_rx: Option<mpsc::Receiver<PublicDataFetchResult>>,
    pending_fetch_label: Option<String>,
    mission_log_open: bool,
    show_about_window: bool,
}

impl HelioProbeApp {
    pub fn new(cc: &eframe::CreationContext<'_>, db_path: PathBuf) -> Self {
        if let Some(wgpu_state) = cc.wgpu_render_state.as_ref() {
            let gpu_pipeline =
                CosmosGpuPipeline::new(&wgpu_state.device, wgpu_state.target_format);
            wgpu_state
                .renderer
                .write()
                .callback_resources
                .insert(gpu_pipeline);
        }

        let mut visuals = egui::Visuals::dark();
        visuals.window_fill = egui::Color32::from_rgb(11, 15, 25);
        visuals.panel_fill = egui::Color32::from_rgb(12, 17, 29);
        cc.egui_ctx.set_visuals(visuals);

        let (db, db_error) = match SaveDatabase::open(&db_path) {
            Ok(d) => (Some(d), None),
            Err(e) => (None, Some(e)),
        };
        let state = db
            .as_ref()
            .and_then(|d| d.load_slot(1).ok())
            .unwrap_or_default();

        Self {
            catalog_base: bootstrap_catalog(),
            state,
            db,
            db_path,
            db_error,
            camera: OrbitCamera3D::default(),
            desired_cam_distance: 78.0,
            camera_mode: CameraTrackMode::FreeSolarOverview,
            lander_cam_initialized: false,
            tracked_probe_id: None,
            selected_body_id: "mars".into(),
            catalog_search: String::new(),
            catalog_category_filter: None,
            selected_bus: PropulsionBus::ChemicalCentaur,
            selected_assist_id: "NONE".into(),
            probe_callsign_input: "CRV-Pathfinder-1".into(),
            transit_warp: 2.0,
            ephemeris_days_per_sec: 0.03,
            paused: false,
            show_orbits: true,
            show_labels: true,
            right_tab: RightTab::ScannerFacts,
            last_frame: Instant::now(),
            anim_time_sec: 0.0,
            active_scan_pulse: None,
            toast_banner: None,
            sbdb_query: "Eros".into(),
            exoplanet_query: "Kepler-186 f".into(),
            simbad_query: "M 8".into(),
            horizons_cmd: "499".into(),
            api_status_msg:
                "Ready to query live NASA JPL, NASA Exoplanet Archive, and CDS SIMBAD endpoints."
                    .into(),
            json_export_path: "helioprobe_campaign_save.json".into(),
            scheduled_window: None,
            pending_fetch_rx: None,
            pending_fetch_label: None,
            mission_log_open: false,
            show_about_window: false,
        }
    }

    pub fn new_headless(db_path: PathBuf) -> Self {
        Self {
            catalog_base: bootstrap_catalog(),
            state: CampaignState::default(),
            db: None,
            db_path,
            db_error: None,
            camera: OrbitCamera3D::default(),
            desired_cam_distance: 78.0,
            camera_mode: CameraTrackMode::FreeSolarOverview,
            lander_cam_initialized: false,
            tracked_probe_id: None,
            selected_body_id: "mars".into(),
            catalog_search: String::new(),
            catalog_category_filter: None,
            selected_bus: PropulsionBus::ChemicalCentaur,
            selected_assist_id: "NONE".into(),
            probe_callsign_input: "CRV-Pathfinder-1".into(),
            transit_warp: 2.0,
            ephemeris_days_per_sec: 0.03,
            paused: false,
            show_orbits: true,
            show_labels: true,
            right_tab: RightTab::ScannerFacts,
            last_frame: Instant::now(),
            anim_time_sec: 0.0,
            active_scan_pulse: None,
            toast_banner: None,
            sbdb_query: "Eros".into(),
            exoplanet_query: "Kepler-186 f".into(),
            simbad_query: "M 8".into(),
            horizons_cmd: "499".into(),
            api_status_msg:
                "Ready to query live NASA JPL, NASA Exoplanet Archive, and CDS SIMBAD endpoints."
                    .into(),
            json_export_path: "helioprobe_campaign_save.json".into(),
            scheduled_window: None,
            pending_fetch_rx: None,
            pending_fetch_label: None,
            mission_log_open: false,
            show_about_window: false,
        }
    }

    fn full_catalog(&self) -> Vec<CelestialBody> {
        let mut all = self.catalog_base.clone();
        for imp in &self.state.imported_bodies {
            if !all.iter().any(|b| b.id == imp.id) {
                all.push(imp.clone());
            }
        }
        all
    }

    fn set_toast(&mut self, msg: impl Into<String>) {
        self.toast_banner = Some((msg.into(), Instant::now()));
    }

    pub fn poll_pending_fetch(&mut self) -> bool {
        let Some(rx) = self.pending_fetch_rx.as_ref() else {
            return false;
        };
        match rx.try_recv() {
            Ok(PublicDataFetchResult::ImportedBody { source, result }) => {
                self.pending_fetch_rx = None;
                self.pending_fetch_label = None;
                match result {
                    Ok(body) => {
                        self.selected_body_id = body.id.clone();
                        self.api_status_msg = format!("Imported {} from {source}.", body.name);
                        self.set_toast(self.api_status_msg.clone());
                        self.state.register_imported_body(body);
                    }
                    Err(e) => {
                        self.api_status_msg = e.clone();
                        self.set_toast(format!("{source} error: {e}"));
                    }
                }
                true
            }
            Ok(PublicDataFetchResult::HorizonsNote(res)) => {
                self.pending_fetch_rx = None;
                self.pending_fetch_label = None;
                match res {
                    Ok(note) => {
                        self.api_status_msg = note;
                        self.set_toast("JPL Horizons ephemeris note updated.");
                    }
                    Err(e) => {
                        self.api_status_msg = e.clone();
                        self.set_toast(format!("Horizons error: {e}"));
                    }
                }
                true
            }
            Err(mpsc::TryRecvError::Empty) => false,
            Err(mpsc::TryRecvError::Disconnected) => {
                self.pending_fetch_rx = None;
                self.pending_fetch_label = None;
                self.api_status_msg = "Background public-data worker terminated unexpectedly.".into();
                true
            }
        }
    }

    pub fn inject_fetch_receiver_for_test(
        &mut self,
        label: impl Into<String>,
        rx: mpsc::Receiver<PublicDataFetchResult>,
    ) {
        self.pending_fetch_label = Some(label.into());
        self.pending_fetch_rx = Some(rx);
    }

    pub fn is_public_fetch_pending(&self) -> bool {
        self.pending_fetch_rx.is_some()
    }

    pub fn api_status_message(&self) -> &str {
        &self.api_status_msg
    }

    fn spawn_public_fetch<F>(&mut self, ctx: &egui::Context, label: impl Into<String>, work: F)
    where
        F: FnOnce() -> PublicDataFetchResult + Send + 'static,
    {
        if self.pending_fetch_rx.is_some() {
            return;
        }
        let label_str = label.into();
        let (tx, rx) = mpsc::channel();
        self.pending_fetch_rx = Some(rx);
        self.pending_fetch_label = Some(label_str.clone());
        self.api_status_msg = format!("Fetching {label_str} in background...");
        let ctx_clone = ctx.clone();
        std::thread::spawn(move || {
            let res = work();
            let _ = tx.send(res);
            ctx_clone.request_repaint();
        });
    }

    fn launch_probe_to_selected(&mut self, catalog: &[CelestialBody]) {
        let Some(target) = catalog.iter().find(|b| b.id == self.selected_body_id) else {
            return;
        };
        let assist_body = if self.selected_assist_id != "NONE" {
            catalog
                .iter()
                .find(|b| b.id == self.selected_assist_id)
                .filter(|ab| compute_gravity_assist(ab, target, catalog).is_some())
        } else {
            None
        };

        let id = self.state.next_probe_id;
        self.state.next_probe_id += 1;
        let callsign = if self.probe_callsign_input.trim().is_empty() {
            format!("CRV-Probe-{id}")
        } else {
            self.probe_callsign_input.trim().to_string()
        };
        self.probe_callsign_input = format!("CRV-Pathfinder-{}", self.state.next_probe_id);

        let mut probe = SpaceProbe::launch_with_options(
            id,
            callsign.clone(),
            self.selected_bus,
            target,
            assist_body,
            &self.state.unlocked_techs,
            catalog,
            self.state.days_since_j2000,
        );
        if assist_body.is_none() {
            if let Some(win) = &self.scheduled_window {
                if win.target_id == target.id {
                    probe.total_delta_v_km_s = win.dv_km_s;
                    probe.physical_flight_days = win.tof_days;
                }
            }
        }
        let assist_label = probe.assist_body_name.clone();
        let unlocked = self.state.record_launch(probe);
        self.tracked_probe_id = Some(id);
        self.camera_mode = CameraTrackMode::RideLatestProbe;
        self.desired_cam_distance = 0.36;
        if let Some(first) = unlocked.first() {
            self.set_toast(format!(
                "Following {callsign} • Achievement: {} (+{} SP)",
                first.title, first.bonus_points
            ));
        } else if let Some(via) = assist_label {
            self.set_toast(format!(
                "Following {callsign} via {via} Gravity-Assist Slingshot -> {}!",
                target.name
            ));
        } else {
            self.set_toast(format!(
                "Following {callsign} from Earth toward {}...",
                target.name
            ));
        }
    }
}

impl eframe::App for HelioProbeApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.ui_step(ctx);
    }
}

impl HelioProbeApp {
    pub fn ui_step(&mut self, ctx: &egui::Context) {
        self.poll_pending_fetch();
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f64().clamp(0.0, 0.1);
        self.last_frame = now;
        self.anim_time_sec += dt;

        let catalog = self.full_catalog();

        if !self.paused {
            self.state.days_since_j2000 += dt * self.ephemeris_days_per_sec;
            let mut newly_arrived = Vec::new();
            let mut flyby_events = Vec::new();

            for probe in &mut self.state.probes {
                let inserted = probe.step(dt, self.transit_warp);
                if let Some(assist_name) = probe.take_flyby_notification() {
                    flyby_events.push((probe.callsign.clone(), assist_name));
                }
                if inserted {
                    newly_arrived.push((probe.id, probe.target_id.clone()));
                }
            }

            for (cs, aname) in flyby_events {
                let _ = self.state.record_slingshot_flyby(&cs, &aname);
                self.set_toast(format!(
                    "{cs} executed hyperbolic Oberth slingshot past {aname}!"
                ));
            }

            for (pid, tid) in newly_arrived {
                if let Some(target) = catalog.iter().find(|b| b.id == tid) {
                    if self.camera_mode == CameraTrackMode::RideLatestProbe
                        && (self.tracked_probe_id == Some(pid) || self.tracked_probe_id.is_none())
                    {
                        self.selected_body_id = target.id.clone();
                        self.camera_mode = CameraTrackMode::OrbitView;
                        self.desired_cam_distance = match target.category {
                            BodyCategory::Nebula => 22.0,
                            BodyCategory::Star => 17.5,
                            _ => (probe_orbit_radius_scene(target) * 2.75).max(1.45),
                        };
                        self.right_tab = RightTab::ScannerFacts;
                    }
                    let unlocked = self.state.record_orbit_insertion(target);
                    if let Some(ach) = unlocked.first() {
                        self.set_toast(format!(
                            "Switched to Orbit View at {} & Achievement: {}!",
                            target.name, ach.title
                        ));
                    } else {
                        self.set_toast(format!(
                            "Switched to Orbit View around {} — Scanner & Landers ready!",
                            target.name
                        ));
                    }
                }
            }

            // Step Atmospheric Entry / Crash Probes
            let mut terminated_atmo = Vec::new();
            for ap in &mut self.state.atmospheric_probes {
                if ap.step(dt, self.transit_warp) {
                    terminated_atmo.push(ap.clone());
                }
            }
            for fin in terminated_atmo {
                let _ = self.state.record_atmo_probe_termination(&fin);
                self.active_scan_pulse = Some((fin.body_id.clone(), Instant::now()));
                self.set_toast(format!(
                    "AEP-{} Loss of Signal @ {}: {} (+200 SP)!",
                    fin.id, fin.body_name, fin.terminal_headline
                ));
            }

            // Step returning Earth Sample-Return Capsules
            let mut recovered_events = Vec::new();
            for cap in &mut self.state.sample_capsules {
                if cap.step(dt, self.transit_warp) {
                    recovered_events.push((cap.origin_body_name.clone(), cap.sample_mass_grams));
                }
            }
            for (origin_name, grams) in recovered_events {
                let _ = self.state.record_capsule_recovery(&origin_name, grams);
                self.active_scan_pulse = Some(("earth".into(), Instant::now()));
                self.set_toast(format!(
                    "Recovered {:.1} g sample from {} at Earth Curatory!",
                    grams, origin_name
                ));
            }

            // Step powered orbital-to-surface Lander / Cryobot descent (~4 s)
            let mut landed_events = Vec::new();
            for units in self.state.surface_deployments.values_mut() {
                for u in units {
                    if u.step(dt) {
                        landed_events.push((u.body_id.clone(), u.body_name.clone(), u.kind.label()));
                    }
                }
            }
            for (bid, bname, klabel) in landed_events {
                self.active_scan_pulse = Some((bid, Instant::now()));
                self.set_toast(format!(
                    "Touchdown confirmed on {} — {} anchored on surface!",
                    bname, klabel
                ));
            }
        }

        // Camera tracking update
        match self.camera_mode {
            CameraTrackMode::FreeSolarOverview => {
                self.camera.up_override = None;
                self.lander_cam_initialized = false;
            }
            CameraTrackMode::LockSelectedBody | CameraTrackMode::OrbitView => {
                self.camera.up_override = None;
                self.lander_cam_initialized = false;
                if let Some(sel) = catalog.iter().find(|b| b.id == self.selected_body_id) {
                    let pos = body_scene_position_with_offsets(
                        sel,
                        &catalog,
                        self.state.days_since_j2000,
                        &self.state.comet_perihelion_offsets,
                    );
                    self.camera.target = self.camera.target.lerp(pos, (dt * 6.0).min(1.0));
                }
                if self.camera_mode == CameraTrackMode::OrbitView && !self.paused {
                    self.camera.yaw_rad += dt * 0.14;
                }
            }
            CameraTrackMode::RideLatestProbe => {
                self.camera.up_override = None;
                self.lander_cam_initialized = false;
                let tracked = self
                    .tracked_probe_id
                    .and_then(|id| self.state.probes.iter().find(|p| p.id == id))
                    .or_else(|| self.state.probes.last());
                if let Some(p) = tracked {
                    let pos = p.scene_position_with_offsets(
                        &catalog,
                        self.state.days_since_j2000,
                        &self.state.comet_perihelion_offsets,
                    );
                    self.camera.target = self.camera.target.lerp(pos, (dt * 8.0).min(1.0));
                }
            }
            CameraTrackMode::LanderBotView => {
                // Ride an active atmospheric plunge if descending on this body; otherwise lock tightly to the surface unit!
                if let Some(active_ap) = self.state.atmospheric_probes.iter().rev().find(|ap| {
                    ap.body_id == self.selected_body_id
                        && ap.phase != EntryProbePhase::TerminalCrushOrImpact
                }) {
                    self.camera.up_override = None;
                    let ap_pos = active_ap.scene_position_with_offsets(
                        &catalog,
                        self.state.days_since_j2000,
                        &self.state.comet_perihelion_offsets,
                    );
                    self.camera.target = ap_pos;
                    self.desired_cam_distance = 0.11;
                    self.camera.distance = 0.11;
                } else if let Some(sel) = catalog.iter().find(|b| b.id == self.selected_body_id) {
                    let center = body_scene_position_with_offsets(
                        sel,
                        &catalog,
                        self.state.days_since_j2000,
                        &self.state.comet_perihelion_offsets,
                    );
                    let (u_idx, descent_prog, is_cryobot) = self
                        .state
                        .surface_deployments
                        .get(&sel.id)
                        .and_then(|units| {
                            units.last().map(|d| {
                                (
                                    units.len().saturating_sub(1),
                                    d.descent_progress,
                                    d.kind == SurfaceUnitKind::OceanCryobot,
                                )
                            })
                        })
                        .unwrap_or((0, 1.0, false));
                    let anchor = unit_world_pos(
                        sel,
                        center,
                        u_idx,
                        self.anim_time_sec,
                        descent_prog,
                    );
                    // Zero camera-target lerp lag relative to the moon so the lander stays planted
                    self.camera.up_override = Some(anchor.normal);
                    if is_cryobot && descent_prog >= 1.0 {
                        self.camera.target = anchor
                            .surface_pt
                            .sub(anchor.normal.scale(anchor.rig_scale * 0.65));
                    } else {
                        self.camera.target = anchor
                            .unit_pos
                            .add(anchor.normal.scale(anchor.rig_scale * 0.45));
                    }
                    if !self.lander_cam_initialized {
                        if is_cryobot && descent_prog >= 1.0 {
                            self.desired_cam_distance =
                                (anchor.rig_scale * 6.4).clamp(0.05, 0.44);
                            self.camera.pitch_rad = 0.14;
                        } else {
                            self.desired_cam_distance =
                                (anchor.rig_scale * 4.8).clamp(0.032, 0.36);
                            self.camera.pitch_rad = 0.40;
                        }
                        self.camera.distance = self.desired_cam_distance;
                        self.camera.yaw_rad = 0.62;
                        self.lander_cam_initialized = true;
                    } else {
                        self.camera.distance = self.desired_cam_distance;
                    }
                }
            }
        }
        if self.camera_mode != CameraTrackMode::LanderBotView {
            self.camera.distance +=
                (self.desired_cam_distance - self.camera.distance) * (dt * 5.0).min(1.0);
        }

        // Top HUD Bar
        egui::TopBottomPanel::top("top_mission_bar").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.heading(
                    egui::RichText::new("HELIOPROBE 3D v2.3")
                        .strong()
                        .color(egui::Color32::from_rgb(95, 225, 255)),
                );
                ui.separator();
                ui.label(
                    egui::RichText::new(format!("{} SP", self.state.science_points))
                        .strong()
                        .color(egui::Color32::from_rgb(110, 245, 175)),
                );
                ui.colored_label(
                    egui::Color32::from_rgb(145, 210, 255),
                    format!("(Lab Avail: {} SP)", self.state.available_sp()),
                );
                ui.label(format!("Rank: {}", self.state.agency_rank()));
                ui.separator();
                ui.label(format!(
                    "Awards: {}/{}",
                    self.state.unlocked_achievements.len(),
                    ACHIEVEMENTS.len()
                ));
                ui.separator();
                if ui
                    .button(if self.paused { "Resume Sim" } else { "Pause Sim" })
                    .clicked()
                {
                    self.paused = !self.paused;
                }
                ui.add(
                    egui::Slider::new(&mut self.ephemeris_days_per_sec, 0.002..=1.0)
                        .logarithmic(true)
                        .text("Orbit Clock (d/s)"),
                );
                ui.add(
                    egui::Slider::new(&mut self.transit_warp, 0.5..=10.0)
                        .text("Probe Transit Warp"),
                );
                ui.checkbox(&mut self.show_orbits, "3D Orbits");
                ui.checkbox(&mut self.show_labels, "Labels");
                if ui.button("Quick Save (Slot 1)").clicked() {
                    if let Some(db) = &self.db {
                        match db.save_slot(1, &self.state) {
                            Ok(()) => self.set_toast("Saved campaign to SQLite Slot 1."),
                            Err(e) => self.set_toast(e),
                        }
                    } else {
                        let msg = self
                            .db_error
                            .clone()
                            .unwrap_or_else(|| "SQLite database unavailable.".to_string());
                        self.set_toast(format!("Quick Save failed: {msg}"));
                    }
                }
            });
        });

        // Left Mission Launch & Target Selector Dock
        egui::SidePanel::left("left_launch_dock")
            .default_width(320.0)
            .resizable(true)
            .show(ctx, |ui| {
                ui.heading("1. Celestial Target Catalog");
                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.catalog_search)
                            .desired_width(145.0)
                            .hint_text("Search targets..."),
                    );
                    if !self.catalog_search.is_empty() && ui.button("Clear").clicked() {
                        self.catalog_search.clear();
                    }
                    let cat_label = self
                        .catalog_category_filter
                        .map_or("All Types", |c| c.label());
                    egui::ComboBox::from_id_salt("catalog_cat_filter")
                        .width(115.0)
                        .selected_text(cat_label)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.catalog_category_filter, None, "All Types");
                            for cat in [
                                BodyCategory::Star,
                                BodyCategory::Planet,
                                BodyCategory::DwarfPlanet,
                                BodyCategory::Moon,
                                BodyCategory::Asteroid,
                                BodyCategory::Comet,
                                BodyCategory::LagrangePoint,
                                BodyCategory::Exoplanet,
                                BodyCategory::Nebula,
                            ] {
                                ui.selectable_value(
                                    &mut self.catalog_category_filter,
                                    Some(cat),
                                    cat.label(),
                                );
                            }
                        });
                });
                let filtered_catalog = filter_catalog_bodies(
                    &catalog,
                    &self.catalog_search,
                    self.catalog_category_filter,
                );
                ui.small(format!(
                    "Showing {} / {} bodies",
                    filtered_catalog.len(),
                    catalog.len()
                ));
                ui.push_id("catalog_scroll", |ui| {
                    egui::ScrollArea::vertical()
                        .max_height(225.0)
                        .show(ui, |ui| {
                            if filtered_catalog.is_empty() {
                                ui.weak("No targets match current search/filter.");
                            }
                            for body in filtered_catalog {
                                let scanned_n = self.state.scanned_count_for(&body.id);
                                let orbited = self.state.has_orbiting_probe_at(&body.id);
                                let badge = if orbited {
                                    format!("[ORBIT {scanned_n}/5]")
                                } else {
                                    format!("[{scanned_n}/5]")
                                };
                                let label =
                                    format!("{}  {}  {}", body.name, body.category.label(), badge);
                                let selected = self.selected_body_id == body.id;
                                if ui.selectable_label(selected, label).clicked() {
                                    self.selected_body_id = body.id.clone();
                                    self.lander_cam_initialized = false;
                                    if body.distance_ly > 0.0 {
                                        self.selected_bus = PropulsionBus::LaserSailFusion;
                                    } else if body.semi_major_axis_au > 8.0 {
                                        self.selected_bus = PropulsionBus::RtgNuclearElectric;
                                    } else if body.semi_major_axis_au > 2.0 {
                                        self.selected_bus = PropulsionBus::SolarElectricIon;
                                    } else {
                                        self.selected_bus = PropulsionBus::ChemicalCentaur;
                                    }
                                    if self.selected_assist_id == body.id {
                                        self.selected_assist_id = "NONE".into();
                                    }
                                }
                            }
                        });
                });

                ui.separator();
                ui.heading("2. Earth LEO Probe Launcher");
                if let Some(target) = catalog.iter().find(|b| b.id == self.selected_body_id) {
                    let beta = self.selected_bus.effective_beta(&self.state.unlocked_techs);
                    let xfer = compute_transfer_from_earth(target, &catalog, beta);
                    let assist_sol = if self.selected_assist_id != "NONE" {
                        catalog
                            .iter()
                            .find(|b| b.id == self.selected_assist_id)
                            .and_then(|ab| compute_gravity_assist(ab, target, &catalog))
                    } else {
                        None
                    };
                    if self.selected_assist_id != "NONE" && assist_sol.is_none() {
                        self.selected_assist_id = "NONE".into();
                    }

                    ui.label(format!("Target: {} ({})", target.name, target.designation));
                    if let Some(sol) = &assist_sol {
                        ui.colored_label(
                            egui::Color32::from_rgb(255, 205, 95),
                            format!(
                                "Slingshot via {}: {:.2} km/s (Saved {:.2} km/s • {:.1}° bend)",
                                sol.assist_name,
                                sol.assisted_dv_km_s,
                                sol.dv_saved_km_s,
                                sol.turn_angle_deg
                            ),
                        );
                    } else {
                        ui.label(format!(
                            "Mission dV: {:.2} km/s | Capture: {:.2} km/s @ {:.0} km",
                            xfer.total_delta_v_km_s,
                            xfer.capture_orbit_vel_km_s,
                            xfer.capture_altitude_km
                        ));
                    }
                    if let Some(win) = self.scheduled_window.clone() {
                        if win.target_id == target.id {
                            ui.horizontal(|ui| {
                                ui.colored_label(
                                    egui::Color32::from_rgb(95, 235, 255),
                                    format!(
                                        "Porkchop Scheduled: +{:.0} d | TOF {:.0} d | {:.2} km/s",
                                        win.dep_offset_days, win.tof_days, win.dv_km_s
                                    ),
                                );
                                if ui.small_button("Clear").clicked() {
                                    self.scheduled_window = None;
                                }
                            });
                        }
                    }

                    ui.horizontal(|ui| {
                        ui.label("Callsign:");
                        ui.text_edit_singleline(&mut self.probe_callsign_input);
                    });
                    egui::ComboBox::from_label("Bus")
                        .selected_text(self.selected_bus.label())
                        .show_ui(ui, |ui| {
                            for bus in PropulsionBus::ALL {
                                ui.selectable_value(&mut self.selected_bus, bus, bus.label());
                            }
                        });
                    egui::ComboBox::from_label("Gravity Assist")
                        .selected_text(if self.selected_assist_id == "NONE" {
                            "Direct Transfer (No Slingshot)".to_string()
                        } else {
                            format!("Slingshot via {}", self.selected_assist_id.to_uppercase())
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut self.selected_assist_id,
                                "NONE".to_string(),
                                "Direct Transfer (No Slingshot)",
                            );
                            for aid in ASSIST_CANDIDATE_IDS {
                                if aid != target.id {
                                    if let Some(ab) = catalog.iter().find(|b| b.id == aid) {
                                        if let Some(sol) =
                                            compute_gravity_assist(ab, target, &catalog)
                                        {
                                            let label = format!(
                                                "Via {} (-{:.2} km/s, {:.1}° bend)",
                                                ab.name, sol.dv_saved_km_s, sol.turn_angle_deg
                                            );
                                            ui.selectable_value(
                                                &mut self.selected_assist_id,
                                                aid.to_string(),
                                                label,
                                            );
                                        }
                                    }
                                }
                            }
                        });

                    let launch_btn = egui::Button::new(
                        egui::RichText::new(format!("Launch Probe from Earth -> {}", target.name))
                            .strong()
                            .color(egui::Color32::BLACK),
                    )
                    .fill(egui::Color32::from_rgb(75, 225, 165));
                    if ui.add_sized([ui.available_width(), 32.0], launch_btn).clicked() {
                        self.launch_probe_to_selected(&catalog);
                    }
                }

                ui.separator();
                ui.heading("3. Camera & Active Fleet");
                ui.horizontal_wrapped(|ui| {
                    if ui
                        .selectable_label(
                            self.camera_mode == CameraTrackMode::FreeSolarOverview,
                            "Solar Overview",
                        )
                        .clicked()
                    {
                        self.camera_mode = CameraTrackMode::FreeSolarOverview;
                        self.camera.target = Vec3d::default();
                        self.desired_cam_distance = 82.0;
                    }
                    if ui
                        .selectable_label(
                            self.camera_mode == CameraTrackMode::LockSelectedBody,
                            "Lock Target",
                        )
                        .clicked()
                    {
                        self.camera_mode = CameraTrackMode::LockSelectedBody;
                        self.desired_cam_distance = 16.0;
                    }
                    if ui
                        .selectable_label(
                            self.camera_mode == CameraTrackMode::OrbitView,
                            "Orbit View",
                        )
                        .clicked()
                    {
                        self.camera_mode = CameraTrackMode::OrbitView;
                        self.desired_cam_distance = 10.5;
                    }
                    if ui
                        .selectable_label(
                            self.camera_mode == CameraTrackMode::RideLatestProbe,
                            "Ride Probe",
                        )
                        .clicked()
                    {
                        self.camera_mode = CameraTrackMode::RideLatestProbe;
                        self.desired_cam_distance = 0.36;
                    }
                    if ui
                        .selectable_label(
                            self.camera_mode == CameraTrackMode::LanderBotView,
                            "Lander / Bot POV",
                        )
                        .clicked()
                    {
                        self.camera_mode = CameraTrackMode::LanderBotView;
                        self.lander_cam_initialized = false;
                    }
                });

                ui.push_id("fleet_scroll", |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        if self.state.probes.is_empty() {
                            ui.weak("No probes launched yet. Select a target and launch from Earth!");
                        }
                        for p in self.state.probes.iter().rev() {
                            let status = match p.phase {
                                ProbePhase::EnRoute => {
                                    if let Some(aname) = &p.assist_body_name {
                                        format!(
                                            "Slingshot via {} ({:.0}%)",
                                            aname,
                                            p.transit_progress * 100.0
                                        )
                                    } else {
                                        format!("En Route ({:.0}%)", p.transit_progress * 100.0)
                                    }
                                }
                                ProbePhase::Orbiting => {
                                    format!("Orbiting ({:.2} km/s)", p.capture_orbit_vel_km_s)
                                }
                            };
                            let is_tracked = self.tracked_probe_id == Some(p.id);
                            let resp = ui
                                .selectable_label(
                                    is_tracked,
                                    format!("• {} -> {}: {}", p.callsign, p.target_name, status),
                                )
                                .on_hover_text("Click to focus camera and select probe target");
                            if resp.clicked() {
                                self.tracked_probe_id = Some(p.id);
                                self.selected_body_id = p.target_id.clone();
                                self.lander_cam_initialized = false;
                                match p.phase {
                                    ProbePhase::EnRoute => {
                                        self.camera_mode = CameraTrackMode::RideLatestProbe;
                                        self.desired_cam_distance = 0.36;
                                    }
                                    ProbePhase::Orbiting => {
                                        self.camera_mode = CameraTrackMode::OrbitView;
                                        if let Some(targ) =
                                            catalog.iter().find(|b| b.id == p.target_id)
                                        {
                                            self.desired_cam_distance = match targ.category {
                                                BodyCategory::Nebula => 22.0,
                                                BodyCategory::Star => 17.5,
                                                _ => (probe_orbit_radius_scene(targ) * 2.75)
                                                    .max(1.45),
                                            };
                                        }
                                    }
                                }
                            }
                        }
                    });
                });
            });

        // Right Dock: 7-Tab Science & Engineering Console
        egui::SidePanel::right("right_science_dock")
            .default_width(420.0)
            .resizable(true)
            .show(ctx, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.selectable_value(&mut self.right_tab, RightTab::ScannerFacts, "Scanner");
                    ui.selectable_value(&mut self.right_tab, RightTab::L2AndComets, "L2 & Comets");
                    ui.selectable_value(&mut self.right_tab, RightTab::LaunchWindows, "Windows");
                    ui.selectable_value(&mut self.right_tab, RightTab::SurfaceOps, "Landers");
                    ui.selectable_value(&mut self.right_tab, RightTab::TechTree, "Tech Lab");
                    ui.selectable_value(&mut self.right_tab, RightTab::Achievements, "Awards");
                    ui.selectable_value(&mut self.right_tab, RightTab::PublicDataImport, "Public Data");
                    ui.selectable_value(&mut self.right_tab, RightTab::SaveLoad, "Saves");
                });
                ui.separator();

                match self.right_tab {
                    RightTab::ScannerFacts => {
                        if let Some(target) =
                            catalog.iter().find(|b| b.id == self.selected_body_id).cloned()
                        {
                            ui.heading(&target.name);
                            ui.label(format!(
                                "{} • {}",
                                target.category.label(),
                                target.designation
                            ));
                            ui.hyperlink_to(&target.source_title, &target.source_url);

                            ui.add_space(4.0);
                            egui::Grid::new("phys_metrics_grid")
                                .striped(true)
                                .show(ui, |ui| {
                                    if target.distance_ly > 0.0 {
                                        ui.label("Distance:");
                                        ui.label(format!("{:.1} light-years", target.distance_ly));
                                        ui.end_row();
                                    } else {
                                        ui.label("Semi-Major Axis:");
                                        ui.label(format!("{:.4} AU", target.semi_major_axis_au));
                                        ui.end_row();
                                    }
                                    if target.span_ly > 0.0 {
                                        ui.label("Nebula Span:");
                                        ui.label(format!("{:.1} light-years", target.span_ly));
                                        ui.end_row();
                                    } else {
                                        ui.label("Mean Radius:");
                                        ui.label(format!("{:.1} km", target.radius_km));
                                        ui.end_row();
                                    }
                                    ui.label("Mean Temperature:");
                                    ui.label(format!("{:.0} °C", target.mean_temp_c));
                                    ui.end_row();
                                });

                            ui.separator();
                            let can_scan_all = self.state.can_scan_all_bands(&target);
                            let has_starshade =
                                self.state.unlocked_techs.contains("coronagraph_jwst");
                            let has_jwst_l2 =
                                self.state.deployed_l2_observatories.contains("jwst_l2");
                            let has_roman_l2 =
                                self.state.deployed_l2_observatories.contains("roman_l2");
                            let band_pts = scan_band_points(&self.state.unlocked_techs);

                            if !can_scan_all {
                                let remote_msg = if has_jwst_l2
                                    && matches!(
                                        target.category,
                                        BodyCategory::Comet
                                            | BodyCategory::Exoplanet
                                            | BodyCategory::Nebula
                                            | BodyCategory::DwarfPlanet
                                    ) {
                                    "JWST L2 Telescope active: Optical/NIR & Thermal/Mass-Spec remote scans unlocked!"
                                } else if has_roman_l2 {
                                    "Roman L2 Telescope active: Optical/NIR wide-field remote scan unlocked!"
                                } else if has_starshade {
                                    "Starshade Coronagraph active: Optical/NIR remote pre-scan unlocked from Earth!"
                                } else {
                                    "Requires an orbiting probe around this body (or Sun-Earth L2 observatory deployment) to unlock orbital instrument telemetry."
                                };
                                ui.colored_label(
                                    egui::Color32::from_rgb(255, 195, 95),
                                    remote_msg,
                                );
                            } else if ui
                                .button("Run Full 5-Band Instrument Sweep (+All Remaining SP)")
                                .clicked()
                            {
                                for band in ScanBand::ALL {
                                    let _ = self.state.perform_scan(&target, band);
                                }
                                self.active_scan_pulse =
                                    Some((target.id.clone(), Instant::now()));
                                self.set_toast(format!(
                                    "Completed 5-band scientific survey of {}!",
                                    target.name
                                ));
                            }

                            ui.push_id("scanner_bands_scroll", |ui| {
                                egui::ScrollArea::vertical().show(ui, |ui| {
                                    for band in ScanBand::ALL {
                                        let unlocked =
                                            self.state.is_band_scanned(&target.id, band);
                                        let band_allowed = self.state.can_scan_band(&target, band);
                                        egui::Frame::group(ui.style()).show(ui, |ui| {
                                            ui.horizontal(|ui| {
                                                ui.strong(band.label());
                                                if unlocked {
                                                    ui.colored_label(
                                                        egui::Color32::from_rgb(95, 240, 165),
                                                        "SCANNED",
                                                    );
                                                } else if ui
                                                    .add_enabled(
                                                        band_allowed,
                                                        egui::Button::new(format!(
                                                            "Scan (+{band_pts} SP)"
                                                        )),
                                                    )
                                                    .clicked()
                                                {
                                                    match self.state.perform_scan(&target, band) {
                                                        Ok((_, achs)) => {
                                                            self.active_scan_pulse = Some((
                                                                target.id.clone(),
                                                                Instant::now(),
                                                            ));
                                                            if let Some(a) = achs.first() {
                                                                self.set_toast(format!(
                                                                    "Achievement Unlocked: {}!",
                                                                    a.title
                                                                ));
                                                            }
                                                        }
                                                        Err(e) => self.set_toast(e),
                                                    }
                                                }
                                            });
                                            if unlocked {
                                                if let Some(finding) = target.finding_for(band) {
                                                    ui.colored_label(
                                                        egui::Color32::from_rgb(140, 215, 255),
                                                        &finding.headline,
                                                    );
                                                    ui.label(&finding.detail);
                                                }
                                            } else {
                                                ui.weak("Telemetry locked — scan this band to decode scientific findings.");
                                            }
                                        });
                                    }
                                });
                            });
                        }
                    }

                    RightTab::L2AndComets => {
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            ui.heading("Sun-Earth L2 Space Telescopes (~1.5M km)");
                            ui.label("Station space observatories in anti-sunward Lissajous halo orbits behind Earth & Moon.");
                            if ui.button("Focus Camera on Sun-Earth L2 Halo").clicked() {
                                self.selected_body_id = "sun_earth_l2".into();
                                self.camera_mode = CameraTrackMode::OrbitView;
                                self.desired_cam_distance = 2.4;
                            }
                            ui.add_space(4.0);
                            for obs in &L2_OBSERVATORIES {
                                let active = self.state.deployed_l2_observatories.contains(obs.id);
                                egui::Frame::group(ui.style()).show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        ui.strong(obs.name);
                                        if active {
                                            ui.colored_label(egui::Color32::from_rgb(95, 240, 165), "L2 HALO ACTIVE");
                                        } else if ui
                                            .button(format!("Deploy to L2 (+{} SP)", obs.deploy_reward_sp))
                                            .clicked()
                                        {
                                            match self.state.deploy_l2_observatory(obs.id) {
                                                Ok((def, _)) => {
                                                    self.selected_body_id = "sun_earth_l2".into();
                                                    self.camera_mode = CameraTrackMode::OrbitView;
                                                    self.desired_cam_distance = 2.2;
                                                    self.set_toast(format!("{} stationed in Sun-Earth L2 halo orbit!", def.name));
                                                }
                                                Err(e) => self.set_toast(e),
                                            }
                                        }
                                    });
                                    ui.colored_label(egui::Color32::from_rgb(140, 215, 255), obs.role_headline);
                                    ui.label(obs.role_detail);
                                    ui.hyperlink_to(obs.agency, obs.source_url);
                                    if obs.id == "roman_l2" && active && !self.state.roman_survey_completed {
                                        if ui.button("Run Roman 100x-FOV Wide-Field Sky Survey (+180 SP)").clicked() {
                                            if let Ok(pts) = self.state.run_roman_wide_field_survey() {
                                                self.set_toast(format!("Roman WFI Sky Survey completed (+{pts} SP)!"));
                                            }
                                        }
                                    }
                                });
                            }

                            ui.separator();
                            ui.heading("Eccentric Comets & Solar-Wind Ion/Dust Tails");
                            let interceptor_ready = self.state.deployed_l2_observatories.contains("interceptor_l2");
                            for comet in catalog.iter().filter(|b| b.category == BodyCategory::Comet) {
                                let off = self.state.comet_perihelion_offsets.get(&comet.id).copied().unwrap_or(0.0);
                                let tail = compute_comet_tail_state(comet, self.state.days_since_j2000, off);
                                egui::Frame::group(ui.style()).show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        ui.strong(&comet.name);
                                        ui.colored_label(
                                            egui::Color32::from_rgb(95, 235, 200),
                                            format!("r = {:.2} AU • {:.1} km/s", tail.r_helio_au, tail.orbital_speed_km_s),
                                        );
                                    });
                                    ui.label(format!(
                                        "e = {:.4} | i = {:.1}° | Outgassing Q ~ {:.2e} mol/s | Activity {:.2}",
                                        comet.eccentricity, comet.inclination_deg, tail.gas_production_q_s, tail.activity
                                    ));
                                    ui.horizontal_wrapped(|ui| {
                                        if ui.button("Track Comet").clicked() {
                                            self.selected_body_id = comet.id.clone();
                                            self.camera_mode = CameraTrackMode::OrbitView;
                                            self.desired_cam_distance = 4.2;
                                        }
                                        if ui.button("Warp to Perihelion Pass").clicked() {
                                            if let Ok(q) = self.state.warp_comet_to_perihelion(comet) {
                                                self.selected_body_id = comet.id.clone();
                                                self.camera_mode = CameraTrackMode::OrbitView;
                                                self.desired_cam_distance = 4.2;
                                                self.set_toast(format!("Warped {} to perihelion pass ({q:.3} AU)!", comet.name));
                                            }
                                        }
                                        if ui
                                            .add_enabled(
                                                interceptor_ready,
                                                egui::Button::new("Rapid L2 Intercept (+90 SP)"),
                                            )
                                            .clicked()
                                        {
                                            let pid = self.state.next_probe_id;
                                            self.state.next_probe_id += 1;
                                            let cs = format!("ESA-CometInt-{pid}");
                                            let p = SpaceProbe::launch_from_l2_interceptor(
                                                pid,
                                                cs.clone(),
                                                comet,
                                                &self.state.unlocked_techs,
                                                &catalog,
                                                self.state.days_since_j2000,
                                            );
                                            self.state.l2_intercepts_launched += 1;
                                            self.state.science_points += 90;
                                            let _ = self.state.record_launch(p);
                                            self.tracked_probe_id = Some(pid);
                                            self.selected_body_id = comet.id.clone();
                                            self.camera_mode = CameraTrackMode::RideLatestProbe;
                                            self.desired_cam_distance = 0.32;
                                            self.set_toast(format!("{cs} departed Sun-Earth L2 toward {} (+90 SP)!", comet.name));
                                        }
                                    });
                                });
                            }
                        });
                    }

                    RightTab::LaunchWindows => {
                        if let Some(target) =
                            catalog.iter().find(|b| b.id == self.selected_body_id).cloned()
                        {
                            ui.heading(format!("Porkchop & Gravity-Assist Lab — {}", target.name));
                            let grid = compute_porkchop_grid_with_offsets(
                                &target,
                                &catalog,
                                self.state.days_since_j2000,
                                &self.state.comet_perihelion_offsets,
                                28,
                                20,
                            );

                            ui.label(format!(
                                "Synodic Period: {:.1} d | Current Phase Offset: {:+.1}°",
                                grid.synodic_period_days, grid.phase_error_deg
                            ));
                            ui.colored_label(
                                egui::Color32::from_rgb(110, 245, 185),
                                format!(
                                    "Optimal Window: +{:.1} d • Flight {:.1} d • Min dV {:.2} km/s • C3 {:.1} km^2/s^2",
                                    grid.best_dep_offset_days,
                                    grid.best_tof_days,
                                    grid.min_dv_km_s,
                                    grid.best_c3_km2_s2
                                ),
                            );

                            if grid.best_dep_offset_days > 0.5
                                && ui
                                    .button(format!(
                                        "Snap Sim Clock to Optimal Launch Window (+{:.1} d)",
                                        grid.best_dep_offset_days
                                    ))
                                    .clicked()
                            {
                                self.state.days_since_j2000 += grid.best_dep_offset_days;
                                self.set_toast(format!(
                                    "Advanced ephemeris by +{:.1} d to optimal {} Lambert alignment!",
                                    grid.best_dep_offset_days, target.name
                                ));
                            }

                            ui.add_space(4.0);
                            let plot_w = ui.available_width().max(240.0);
                            let plot_h = 175.0;
                            let (p_rect, p_resp) = ui.allocate_exact_size(
                                egui::vec2(plot_w, plot_h),
                                egui::Sense::click_and_drag(),
                            );
                            let painter = ui.painter_at(p_rect);
                            let cell_w = p_rect.width() / (grid.cols as f32);
                            let cell_h = p_rect.height() / (grid.rows as f32);
                            let span = (grid.max_dv_km_s - grid.min_dv_km_s).max(0.05);

                            for r in 0..grid.rows {
                                for c in 0..grid.cols {
                                    let val = grid.dv_km_s[r * grid.cols + c];
                                    let norm = ((val - grid.min_dv_km_s) / span) as f32;
                                    let y_inv = (grid.rows - 1 - r) as f32;
                                    let cell_rect = egui::Rect::from_min_size(
                                        egui::pos2(
                                            p_rect.left() + (c as f32) * cell_w,
                                            p_rect.top() + y_inv * cell_h,
                                        ),
                                        egui::vec2(cell_w + 0.6, cell_h + 0.6),
                                    );
                                    painter.rect_filled(cell_rect, 0.0, porkchop_colormap(norm));
                                }
                            }
                            let bx =
                                p_rect.left() + (grid.best_col as f32 + 0.5) * cell_w;
                            let by = p_rect.top()
                                + ((grid.rows - 1 - grid.best_row) as f32 + 0.5) * cell_h;
                            painter.circle_stroke(
                                egui::pos2(bx, by),
                                6.5,
                                egui::Stroke::new(2.0_f32, egui::Color32::WHITE),
                            );
                            if let Some(win) = &self.scheduled_window {
                                if win.target_id == target.id && win.col < grid.cols && win.row < grid.rows {
                                    let sx = p_rect.left() + (win.col as f32 + 0.5) * cell_w;
                                    let sy = p_rect.top()
                                        + ((grid.rows - 1 - win.row) as f32 + 0.5) * cell_h;
                                    let d_sz = 6.0_f32;
                                    painter.add(egui::Shape::closed_line(
                                        vec![
                                            egui::pos2(sx, sy - d_sz),
                                            egui::pos2(sx + d_sz, sy),
                                            egui::pos2(sx, sy + d_sz),
                                            egui::pos2(sx - d_sz, sy),
                                        ],
                                        egui::Stroke::new(2.0_f32, egui::Color32::from_rgb(95, 235, 255)),
                                    ));
                                }
                            }

                            if let Some(hover_pos) = p_resp.hover_pos() {
                                if let Some((hc, hr)) =
                                    porkchop_cell_from_pointer(p_rect, hover_pos, grid.cols, grid.rows)
                                {
                                    let hx = p_rect.left() + (hc as f32 + 0.5) * cell_w;
                                    let hy = p_rect.top()
                                        + ((grid.rows - 1 - hr) as f32 + 0.5) * cell_h;
                                    let crosshair_stroke = egui::Stroke::new(
                                        1.0_f32,
                                        egui::Color32::from_rgba_unmultiplied(255, 255, 255, 165),
                                    );
                                    painter.line_segment(
                                        [egui::pos2(hx, p_rect.top()), egui::pos2(hx, p_rect.bottom())],
                                        crosshair_stroke,
                                    );
                                    painter.line_segment(
                                        [egui::pos2(p_rect.left(), hy), egui::pos2(p_rect.right(), hy)],
                                        crosshair_stroke,
                                    );
                                    let idx = hr * grid.cols + hc;
                                    let dep_off = grid.dep_offsets_days[hc];
                                    let tof = grid.tof_days[hr];
                                    let dv = grid.dv_km_s[idx];
                                    let c3 = grid.c3_km2_s2[idx];
                                    let dep_utc =
                                        j2000_days_to_utc_string(self.state.days_since_j2000 + dep_off);
                                    p_resp.clone().on_hover_ui_at_pointer(|ui| {
                                        ui.strong(format!("Dep: {dep_utc} (+{dep_off:.1} d)"));
                                        ui.label(format!("TOF: {tof:.1} d | Total dV: {dv:.2} km/s"));
                                        ui.label(format!("Departure C3: {c3:.1} km^2/s^2"));
                                        ui.weak("Click plot to schedule this transfer window");
                                    });
                                }
                            }

                            if p_resp.clicked() {
                                if let Some(click_pos) = p_resp.interact_pointer_pos() {
                                    if let Some((cc, cr)) = porkchop_cell_from_pointer(
                                        p_rect,
                                        click_pos,
                                        grid.cols,
                                        grid.rows,
                                    ) {
                                        let idx = cr * grid.cols + cc;
                                        let win = ScheduledPorkchopWindow {
                                            target_id: target.id.clone(),
                                            col: cc,
                                            row: cr,
                                            dep_offset_days: grid.dep_offsets_days[cc],
                                            tof_days: grid.tof_days[cr],
                                            dv_km_s: grid.dv_km_s[idx],
                                            c3_km2_s2: grid.c3_km2_s2[idx],
                                        };
                                        self.set_toast(format!(
                                            "Scheduled {} window: +{:.1} d | TOF {:.1} d | dV {:.2} km/s",
                                            target.name, win.dep_offset_days, win.tof_days, win.dv_km_s
                                        ));
                                        self.scheduled_window = Some(win);
                                    }
                                }
                            }

                            painter.rect_stroke(
                                p_rect,
                                2.0,
                                egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(70, 95, 130)),
                                egui::StrokeKind::Inside,
                            );

                            let dep_end = grid.dep_offsets_days.last().copied().unwrap_or(0.0);
                            let tof_min = grid.tof_days.first().copied().unwrap_or(0.0);
                            let tof_max = grid.tof_days.last().copied().unwrap_or(0.0);
                            ui.small(format!(
                                "X: Dep +0..+{dep_end:.0} d  |  Y: TOF {tof_min:.0}..{tof_max:.0} d  |  dV {min_dv:.1}..{max_dv:.1} km/s",
                                min_dv = grid.min_dv_km_s,
                                max_dv = grid.max_dv_km_s
                            ));
                            ui.horizontal(|ui| {
                                ui.small("dV Scale:");
                                let (bar_rect, _) =
                                    ui.allocate_exact_size(egui::vec2(120.0, 10.0), egui::Sense::hover());
                                let bar_painter = ui.painter_at(bar_rect);
                                for step in 0..24 {
                                    let t0 = step as f32 / 24.0;
                                    let x0 = bar_rect.left() + t0 * bar_rect.width();
                                    let w = bar_rect.width() / 24.0 + 0.5;
                                    bar_painter.rect_filled(
                                        egui::Rect::from_min_size(
                                            egui::pos2(x0, bar_rect.top()),
                                            egui::vec2(w, bar_rect.height()),
                                        ),
                                        0.0,
                                        porkchop_colormap(t0),
                                    );
                                }
                                ui.small(format!("{:.1}..{:.1} km/s", grid.min_dv_km_s, grid.max_dv_km_s));
                            });
                            if let Some(win) = self.scheduled_window.clone() {
                                if win.target_id == target.id {
                                    ui.horizontal_wrapped(|ui| {
                                        ui.colored_label(
                                            egui::Color32::from_rgb(95, 235, 255),
                                            format!(
                                                "Scheduled: +{:.1} d | TOF {:.1} d | {:.2} km/s | C3 {:.1}",
                                                win.dep_offset_days, win.tof_days, win.dv_km_s, win.c3_km2_s2
                                            ),
                                        );
                                        if ui.small_button("Apply Dep to Clock").clicked() {
                                            self.state.days_since_j2000 += win.dep_offset_days;
                                            self.set_toast(format!(
                                                "Applied scheduled departure (+{:.1} d) to ephemeris clock.",
                                                win.dep_offset_days
                                            ));
                                        }
                                        if ui.small_button("Clear").clicked() {
                                            self.scheduled_window = None;
                                        }
                                    });
                                }
                            }

                            ui.separator();
                            ui.strong("Patched-Conic Gravity-Assist Slingshot Candidates");
                            for aid in ASSIST_CANDIDATE_IDS {
                                if let Some(ab) = catalog.iter().find(|b| b.id == aid) {
                                    if let Some(sol) =
                                        compute_gravity_assist(ab, &target, &catalog)
                                    {
                                        egui::Frame::group(ui.style()).show(ui, |ui| {
                                            ui.horizontal(|ui| {
                                                ui.strong(format!("Via {}", sol.assist_name));
                                                ui.colored_label(
                                                    egui::Color32::from_rgb(110, 245, 175),
                                                    format!("Save {:.2} km/s", sol.dv_saved_km_s),
                                                );
                                                let armed = self.selected_assist_id == sol.assist_id;
                                                if ui
                                                    .button(if armed {
                                                        "Armed"
                                                    } else {
                                                        "Arm Slingshot"
                                                    })
                                                    .clicked()
                                                {
                                                    self.selected_assist_id =
                                                        sol.assist_id.clone();
                                                }
                                            });
                                            ui.label(format!(
                                                "Assisted dV: {:.2} km/s | Deflection: {:.1}° @ {:.0} km",
                                                sol.assisted_dv_km_s,
                                                sol.turn_angle_deg,
                                                sol.periapsis_alt_km
                                            ));
                                        });
                                    }
                                }
                            }
                        }
                    }

                    RightTab::SurfaceOps => {
                        if let Some(target) =
                            catalog.iter().find(|b| b.id == self.selected_body_id).cloned()
                        {
                            ui.horizontal(|ui| {
                                ui.heading(format!("Landers, Cryobots & Entry — {}", target.name));
                                if ui.button("6. Lander / Bot POV").clicked() {
                                    self.camera_mode = CameraTrackMode::LanderBotView;
                                    self.lander_cam_initialized = false;
                                }
                            });
                            let in_orbit = self.state.has_orbiting_probe_at(&target.id);
                            if !in_orbit && target.id != "earth" {
                                ui.colored_label(
                                    egui::Color32::from_rgb(255, 195, 95),
                                    format!(
                                        "Insert a spacecraft into orbit around {} first to deploy landers, cryobots, or atmospheric crash probes.",
                                        target.name
                                    ),
                                );
                            }

                            ui.separator();
                            // 1. Aero-Shell / Sky-Crane Surface Lander
                            let lander_ok = supports_surface_lander(&target);
                            let lander_done = self
                                .state
                                .has_surface_unit(&target.id, SurfaceUnitKind::AeroShellLander);
                            egui::Frame::group(ui.style()).show(ui, |ui| {
                                ui.strong("1. Aero-Shell / Sky-Crane Surface Lander (+180 SP)");
                                ui.label("Soft-lands on rocky/icy surfaces to run 3-axis broadband seismometry, core-drill X-ray mineralogy, and surface heat-flow experiments.");
                                if lander_done {
                                    ui.horizontal(|ui| {
                                        ui.colored_label(
                                            egui::Color32::from_rgb(95, 240, 165),
                                            "ACTIVE ON SURFACE",
                                        );
                                        if ui.button("Seismic Thumper / Drill (+55 SP)").clicked() {
                                            match self.state.run_active_surface_experiment(
                                                &target.id,
                                                SurfaceUnitKind::AeroShellLander,
                                            ) {
                                                Ok(pts) => self.set_toast(format!(
                                                    "Executed active seismic core-drill pulse (+{pts} SP)!"
                                                )),
                                                Err(e) => self.set_toast(e),
                                            }
                                        }
                                    });
                                } else if !lander_ok {
                                    ui.weak("No solid crust — drop an Atmospheric Entry / Crush Probe below!");
                                } else if ui
                                    .add_enabled(
                                        in_orbit || target.id == "earth",
                                        egui::Button::new("Deploy Surface Lander (+180 SP)"),
                                    )
                                    .clicked()
                                {
                                    match self
                                        .state
                                        .deploy_surface_unit(&target, SurfaceUnitKind::AeroShellLander)
                                    {
                                        Ok((dep, _)) => {
                                            self.camera_mode = CameraTrackMode::LanderBotView;
                                            self.lander_cam_initialized = false;
                                            self.active_scan_pulse =
                                                Some((target.id.clone(), Instant::now()));
                                            self.set_toast(dep.telemetry_headline);
                                        }
                                        Err(e) => self.set_toast(e),
                                    }
                                }
                            });

                            // 2. Pu-238 GPHS Ocean-World Thermal Melt Cryobot
                            let cryo_ok = supports_ocean_cryobot(&target);
                            let cryo_done = self
                                .state
                                .has_surface_unit(&target.id, SurfaceUnitKind::OceanCryobot);
                            let cryo_pts = cryobot_points(&self.state.unlocked_techs);
                            egui::Frame::group(ui.style()).show(ui, |ui| {
                                ui.strong(format!(
                                    "2. GPHS-Pu238 Ocean-World Thermal Cryobot (+{cryo_pts} SP)"
                                ));
                                ui.label("Copper-dome Pu-238 nose melts kilometers through ice into subsurface liquid oceans, spooling acoustic/RF relay pucks to profile hydrothermal vents and brine chemistry.");
                                if cryo_done {
                                    ui.horizontal(|ui| {
                                        ui.colored_label(
                                            egui::Color32::from_rgb(95, 235, 255),
                                            "SUB-ICE MELT CHANNEL ACTIVE",
                                        );
                                        if ui.button("Ping Deep-Ocean Sonar (+75 SP)").clicked() {
                                            match self.state.run_active_surface_experiment(
                                                &target.id,
                                                SurfaceUnitKind::OceanCryobot,
                                            ) {
                                                Ok(pts) => self.set_toast(format!(
                                                    "Sub-ice acoustic bathymetry & organic plume ping complete (+{pts} SP)!"
                                                )),
                                                Err(e) => self.set_toast(e),
                                            }
                                        }
                                    });
                                } else if !cryo_ok {
                                    ui.weak("Eligible on 12 icy worlds (Europa, Enceladus, Titan, Ganymede, Callisto, Triton, Ceres, Pluto, Charon, Haumea, Makemake, Eris).");
                                } else if ui
                                    .add_enabled(
                                        in_orbit,
                                        egui::Button::new(format!(
                                            "Deploy Ocean Cryobot (+{cryo_pts} SP)"
                                        )),
                                    )
                                    .clicked()
                                {
                                    match self
                                        .state
                                        .deploy_surface_unit(&target, SurfaceUnitKind::OceanCryobot)
                                    {
                                        Ok((dep, _)) => {
                                            self.camera_mode = CameraTrackMode::LanderBotView;
                                            self.lander_cam_initialized = false;
                                            self.active_scan_pulse =
                                                Some((target.id.clone(), Instant::now()));
                                            self.set_toast(dep.telemetry_headline);
                                        }
                                        Err(e) => self.set_toast(e),
                                    }
                                }
                            });

                            // 3. Galileo-Class Atmospheric Entry / Kinetic Crash Probe
                            let atmo_ok = supports_atmospheric_or_crash_probe(&target);
                            let is_atmo = has_substantial_atmosphere(&target);
                            egui::Frame::group(ui.style()).show(ui, |ui| {
                                ui.strong(if is_atmo {
                                    "3. Galileo-Class Atmospheric Entry / Crush Probe (+260 SP)"
                                } else {
                                    "3. Hypervelocity Kinetic Impactor Crash Probe (+260 SP)"
                                });
                                ui.label(if is_atmo {
                                    "Carbon-phenolic entry cone plunges into cloud decks under parachute, streaming live pressure/temperature/wind profiles until crush-depth or impact destruction!"
                                } else {
                                    "De-orbits a high-speed kinetic penetrator into airless regolith (LCROSS/DS2 class) to excavate a fresh subsurface volatile ejecta plume!"
                                });
                                if ui
                                    .add_enabled(
                                        in_orbit && atmo_ok,
                                        egui::Button::new(if is_atmo {
                                            "Drop Atmospheric Crush Probe (+60 / +200 SP)"
                                        } else {
                                            "Drop Kinetic Crash Probe (+60 / +200 SP)"
                                        }),
                                    )
                                    .clicked()
                                {
                                    match self.state.launch_atmospheric_probe(&target) {
                                        Ok(run) => {
                                            self.camera_mode = CameraTrackMode::LanderBotView;
                                            self.set_toast(format!(
                                                "Riding AEP-{} plunging into {} at {:.1} km/s!",
                                                run.id, target.name, run.entry_speed_km_s
                                            ));
                                        }
                                        Err(e) => self.set_toast(e),
                                    }
                                }
                            });

                            // 4. Earth Sample-Return Capsule (SRC)
                            let src_pts = sample_return_points(&self.state.unlocked_techs);
                            egui::Frame::group(ui.style()).show(ui, |ui| {
                                ui.strong(format!(
                                    "4. Earth Sample-Return Capsule (+40 / +{src_pts} SP)"
                                ));
                                if ui
                                    .add_enabled(
                                        in_orbit && atmo_ok,
                                        egui::Button::new("Launch Sample-Return Capsule to Earth"),
                                    )
                                    .clicked()
                                {
                                    match self.state.launch_sample_return(&target, &catalog) {
                                        Ok(cap) => self.set_toast(format!(
                                            "Launched SRC-{} ({:.1} g) back toward Earth!",
                                            cap.id, cap.sample_mass_grams
                                        )),
                                        Err(e) => self.set_toast(e),
                                    }
                                }
                            });

                            ui.separator();
                            ui.push_id("surface_telemetry_scroll", |ui| {
                                egui::ScrollArea::vertical().show(ui, |ui| {
                                    if let Some(units) =
                                        self.state.surface_deployments.get(&target.id)
                                    {
                                        for u in units {
                                            egui::Frame::group(ui.style()).show(ui, |ui| {
                                                ui.colored_label(
                                                    egui::Color32::from_rgb(130, 230, 255),
                                                    format!(
                                                        "{} — {}",
                                                        u.kind.label(),
                                                        u.telemetry_headline
                                                    ),
                                                );
                                                ui.label(&u.telemetry_report);
                                            });
                                        }
                                    }
                                    for cap in self.state.sample_capsules.iter().rev() {
                                        let st = match cap.phase {
                                            CapsulePhase::ReturningToEarth => {
                                                format!(
                                                    "Returning to Earth ({:.0}%)",
                                                    cap.progress * 100.0
                                                )
                                            }
                                            CapsulePhase::RecoveredAtEarth => {
                                                "Recovered in Earth Cleanroom".to_string()
                                            }
                                        };
                                        ui.label(format!(
                                            "• SRC-{} ({} • {:.1} g): {}",
                                            cap.id, cap.origin_body_name, cap.sample_mass_grams, st
                                        ));
                                    }
                                });
                            });
                        }
                    }

                    RightTab::TechTree => {
                        ui.heading("Spacecraft Tech Tree & Payload Lab");
                        ui.label(format!(
                            "Available Lab Budget: {} SP  (Lifetime Rank Score: {} SP)",
                            self.state.available_sp(),
                            self.state.science_points
                        ));
                        if self.state.unlocked_techs.contains("dsn_ka_band") {
                            ui.checkbox(
                                &mut self.state.dsn_auto_scan,
                                "Enable Optical/Ka-Band DSN Automated 5-Band Scan on Orbit Capture",
                            );
                        }
                        ui.separator();

                        ui.push_id("tech_tree_scroll", |ui| {
                            egui::ScrollArea::vertical().show(ui, |ui| {
                                for branch in [
                                    TechBranch::PropulsionNav,
                                    TechBranch::PayloadsCryo,
                                    TechBranch::DsnCuratory,
                                ] {
                                    ui.colored_label(
                                        egui::Color32::from_rgb(125, 220, 255),
                                        branch.label(),
                                    );
                                    for node in TECH_TREE.iter().filter(|n| n.branch == branch) {
                                        let done = self.state.unlocked_techs.contains(node.id);
                                        let can_buy =
                                            !done && self.state.available_sp() >= node.cost_sp;
                                        egui::Frame::group(ui.style()).show(ui, |ui| {
                                            ui.horizontal(|ui| {
                                                ui.strong(node.title);
                                                if done {
                                                    ui.colored_label(
                                                        egui::Color32::from_rgb(95, 240, 165),
                                                        "[RESEARCHED]",
                                                    );
                                                } else if ui
                                                    .add_enabled(
                                                        can_buy,
                                                        egui::Button::new(format!(
                                                            "Unlock ({} SP)",
                                                            node.cost_sp
                                                        )),
                                                    )
                                                    .clicked()
                                                {
                                                    match self.state.unlock_tech(node.id) {
                                                        Ok(def) => self.set_toast(format!(
                                                            "Unlocked Tech: {}!",
                                                            def.title
                                                        )),
                                                        Err(e) => self.set_toast(e),
                                                    }
                                                }
                                            });
                                            ui.label(node.effect_summary);
                                        });
                                    }
                                    ui.add_space(6.0);
                                }
                            });
                        });
                    }

                    RightTab::Achievements => {
                        ui.heading(format!(
                            "Mission Achievements ({}/{})",
                            self.state.unlocked_achievements.len(),
                            ACHIEVEMENTS.len()
                        ));
                        ui.push_id("ach_scroll", |ui| {
                            egui::ScrollArea::vertical().show(ui, |ui| {
                                for ach in ACHIEVEMENTS {
                                    let done = self.state.unlocked_achievements.contains(ach.id);
                                    egui::Frame::group(ui.style()).show(ui, |ui| {
                                        ui.horizontal(|ui| {
                                            let color = if done {
                                                egui::Color32::from_rgb(95, 240, 165)
                                            } else {
                                                egui::Color32::GRAY
                                            };
                                            ui.colored_label(
                                                color,
                                                if done { "[UNLOCKED]" } else { "[LOCKED]" },
                                            );
                                            ui.strong(ach.title);
                                            ui.label(format!("+{} SP", ach.bonus_points));
                                        });
                                        ui.label(ach.description);
                                    });
                                }
                            });
                        });
                    }

                    RightTab::PublicDataImport => {
                        ui.heading("Live Public Astronomy Harvesters");
                        let pending = self.is_public_fetch_pending();
                        if pending {
                            ui.horizontal(|ui| {
                                ui.spinner();
                                ui.colored_label(
                                    egui::Color32::from_rgb(255, 210, 100),
                                    format!(
                                        "Fetching {} on background thread...",
                                        self.pending_fetch_label.as_deref().unwrap_or("public data")
                                    ),
                                );
                            });
                        }
                        ui.separator();

                        ui.strong("1. NASA JPL Small-Body Database (SBDB)");
                        ui.hyperlink_to(
                            "JPL SBDB API v1.3 Docs",
                            "https://ssd-api.jpl.nasa.gov/doc/sbdb.html",
                        );
                        ui.horizontal(|ui| {
                            ui.text_edit_singleline(&mut self.sbdb_query);
                            if ui
                                .add_enabled(!pending, egui::Button::new("Import Asteroid/Comet"))
                                .clicked()
                            {
                                let q = self.sbdb_query.clone();
                                self.spawn_public_fetch(ctx, format!("JPL SBDB ({q})"), move || {
                                    PublicDataFetchResult::ImportedBody {
                                        source: "JPL SBDB",
                                        result: fetch_jpl_sbdb_small_body(&q),
                                    }
                                });
                            }
                        });

                        ui.separator();
                        ui.strong("2. NASA Exoplanet Archive TAP (ADQL)");
                        ui.hyperlink_to(
                            "NExScI TAP Documentation",
                            "https://caltech-ipac.github.io/nexsciTAP/intro.html",
                        );
                        ui.horizontal(|ui| {
                            ui.text_edit_singleline(&mut self.exoplanet_query);
                            if ui
                                .add_enabled(!pending, egui::Button::new("Import Exoplanet"))
                                .clicked()
                            {
                                let q = self.exoplanet_query.clone();
                                self.spawn_public_fetch(
                                    ctx,
                                    format!("NASA Exoplanet Archive ({q})"),
                                    move || PublicDataFetchResult::ImportedBody {
                                        source: "NASA Exoplanet Archive",
                                        result: fetch_nasa_exoplanet(&q),
                                    },
                                );
                            }
                        });

                        ui.separator();
                        ui.strong("3. CDS SIMBAD Deep-Sky TAP (Nebulae / Pulsars)");
                        ui.hyperlink_to(
                            "SIMBAD TAP Reference",
                            "https://simbad.cds.unistra.fr/simbad/sim-tap",
                        );
                        ui.horizontal(|ui| {
                            ui.text_edit_singleline(&mut self.simbad_query);
                            if ui
                                .add_enabled(!pending, egui::Button::new("Import Nebula/Object"))
                                .clicked()
                            {
                                let q = self.simbad_query.clone();
                                self.spawn_public_fetch(ctx, format!("CDS SIMBAD ({q})"), move || {
                                    PublicDataFetchResult::ImportedBody {
                                        source: "CDS SIMBAD",
                                        result: fetch_simbad_deep_sky(&q),
                                    }
                                });
                            }
                        });

                        ui.separator();
                        ui.strong("4. NASA JPL Horizons Object Ephemeris Header");
                        ui.hyperlink_to(
                            "JPL Horizons API v1.3 Docs",
                            "https://ssd-api.jpl.nasa.gov/doc/horizons.html",
                        );
                        ui.horizontal(|ui| {
                            ui.text_edit_singleline(&mut self.horizons_cmd);
                            if ui
                                .add_enabled(!pending, egui::Button::new("Query Horizons"))
                                .clicked()
                            {
                                let cmd = self.horizons_cmd.clone();
                                self.spawn_public_fetch(
                                    ctx,
                                    format!("JPL Horizons ({cmd})"),
                                    move || {
                                        PublicDataFetchResult::HorizonsNote(
                                            fetch_jpl_horizons_ephemeris_note(&cmd),
                                        )
                                    },
                                );
                            }
                        });

                        ui.separator();
                        ui.colored_label(
                            egui::Color32::from_rgb(135, 220, 255),
                            &self.api_status_msg,
                        );
                    }

                    RightTab::SaveLoad => {
                        ui.heading("SQLite WAL & Portable JSON Saves");
                        ui.label(format!("Database: {}", self.db_path.display()));
                        if let Some(err) = &self.db_error {
                            ui.colored_label(
                                egui::Color32::from_rgb(255, 130, 120),
                                format!("SQLite Error: {err}"),
                            );
                        }
                        ui.horizontal_wrapped(|ui| {
                            if ui.button("Open / Switch DB...").clicked() {
                                if let Some(path) = rfd::FileDialog::new()
                                    .add_filter("SQLite Database", &["db", "sqlite", "sqlite3"])
                                    .pick_file()
                                {
                                    match SaveDatabase::open(&path) {
                                        Ok(new_db) => {
                                            self.db_path = path.clone();
                                            self.db = Some(new_db);
                                            self.db_error = None;
                                            self.set_toast(format!(
                                                "Opened SQLite DB: {}",
                                                path.display()
                                            ));
                                        }
                                        Err(e) => {
                                            self.db_error = Some(e.clone());
                                            self.set_toast(format!("Failed opening DB: {e}"));
                                        }
                                    }
                                }
                            }
                            if ui.button("Save DB As...").clicked() {
                                if let Some(path) = rfd::FileDialog::new()
                                    .add_filter("SQLite Database", &["db", "sqlite", "sqlite3"])
                                    .set_file_name("helioprobe_saves.db")
                                    .save_file()
                                {
                                    match SaveDatabase::open(&path) {
                                        Ok(new_db) => match new_db.save_slot(1, &self.state) {
                                            Ok(()) => {
                                                self.db_path = path.clone();
                                                self.db = Some(new_db);
                                                self.db_error = None;
                                                self.set_toast(format!(
                                                    "Saved campaign DB to {}",
                                                    path.display()
                                                ));
                                            }
                                            Err(e) => {
                                                self.db_error = Some(e.clone());
                                                self.set_toast(format!("Failed saving DB: {e}"));
                                            }
                                        },
                                        Err(e) => {
                                            self.db_error = Some(e.clone());
                                            self.set_toast(format!("Failed creating DB: {e}"));
                                        }
                                    }
                                }
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("Campaign:");
                            ui.text_edit_singleline(&mut self.state.campaign_name);
                        });
                        ui.separator();
                        for slot in 1..=3 {
                            ui.horizontal(|ui| {
                                ui.strong(format!("Slot {slot}"));
                                if ui.button("Save").clicked() {
                                    if let Some(db) = &self.db {
                                        match db.save_slot(slot, &self.state) {
                                            Ok(()) => {
                                                self.set_toast(format!("Saved to Slot {slot}."))
                                            }
                                            Err(e) => self.set_toast(e),
                                        }
                                    } else {
                                        let msg = self
                                            .db_error
                                            .clone()
                                            .unwrap_or_else(|| "SQLite database unavailable.".into());
                                        self.set_toast(format!("Slot {slot} save failed: {msg}"));
                                    }
                                }
                                if ui.button("Load").clicked() {
                                    if let Some(db) = &self.db {
                                        match db.load_slot(slot) {
                                            Ok(loaded) => {
                                                self.state = loaded;
                                                self.set_toast(format!("Loaded Slot {slot}."));
                                            }
                                            Err(e) => self.set_toast(e),
                                        }
                                    } else {
                                        let msg = self
                                            .db_error
                                            .clone()
                                            .unwrap_or_else(|| "SQLite database unavailable.".into());
                                        self.set_toast(format!("Slot {slot} load failed: {msg}"));
                                    }
                                }
                            });
                        }
                        if let Some(db) = &self.db {
                            match db.list_slots() {
                                Ok(slots) => {
                                    ui.separator();
                                    for s in slots {
                                        ui.label(format!(
                                            "Slot {}: {} ({} SP) — {}",
                                            s.slot_id, s.campaign_name, s.science_points, s.updated_utc
                                        ));
                                    }
                                }
                                Err(e) => {
                                    ui.colored_label(
                                        egui::Color32::from_rgb(255, 130, 120),
                                        format!("Slot list error: {e}"),
                                    );
                                }
                            }
                        }
                        ui.separator();
                        ui.strong("Portable JSON Campaign Export / Import");
                        ui.label(format!("Active JSON Path: {}", self.json_export_path));
                        ui.horizontal_wrapped(|ui| {
                            if ui.button("Export JSON...").clicked() {
                                if let Some(path) = rfd::FileDialog::new()
                                    .add_filter("JSON Campaign", &["json"])
                                    .set_file_name("helioprobe_campaign_save.json")
                                    .save_file()
                                {
                                    self.json_export_path = path.display().to_string();
                                    match export_campaign_json(&self.state, &path) {
                                        Ok(()) => self.set_toast(format!(
                                            "Exported campaign JSON to {}",
                                            path.display()
                                        )),
                                        Err(e) => self.set_toast(format!("Export failed: {e}")),
                                    }
                                }
                            }
                            if ui.button("Import JSON...").clicked() {
                                if let Some(path) = rfd::FileDialog::new()
                                    .add_filter("JSON Campaign", &["json"])
                                    .pick_file()
                                {
                                    self.json_export_path = path.display().to_string();
                                    match import_campaign_json(&path) {
                                        Ok(loaded) => {
                                            self.state = loaded;
                                            self.set_toast(format!(
                                                "Imported campaign JSON from {}",
                                                path.display()
                                            ));
                                        }
                                        Err(e) => self.set_toast(format!("Import failed: {e}")),
                                    }
                                }
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.text_edit_singleline(&mut self.json_export_path);
                            if ui.small_button("Write").clicked() {
                                let path = PathBuf::from(&self.json_export_path);
                                match export_campaign_json(&self.state, &path) {
                                    Ok(()) => self.set_toast(format!(
                                        "Exported campaign JSON to {}",
                                        path.display()
                                    )),
                                    Err(e) => self.set_toast(format!("Export failed: {e}")),
                                }
                            }
                            if ui.small_button("Read").clicked() {
                                let path = PathBuf::from(&self.json_export_path);
                                match import_campaign_json(&path) {
                                    Ok(loaded) => {
                                        self.state = loaded;
                                        self.set_toast("Imported campaign JSON successfully.");
                                    }
                                    Err(e) => self.set_toast(format!("Import failed: {e}")),
                                }
                            }
                        });
                    }
                }
            });

        // Bottom Copyright, About & Flight Director Mission Log Bar/Drawer (must precede CentralPanel)
        egui::TopBottomPanel::bottom("bottom_copyright_bar").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.small("Copyright CRV Science Network  •  GPL-3.0-or-later  •  HelioProbe v2.3.0");
                ui.separator();
                if ui.small_button("About HelioProbe").clicked() {
                    self.show_about_window = true;
                }
                ui.hyperlink_to("crvscience.com", "https://crvscience.com");
                ui.separator();
                let log_count = self.state.mission_log.len();
                let log_btn_lbl = if self.mission_log_open {
                    format!("Hide Flight Director Log ({log_count})")
                } else {
                    format!("Flight Director Log ({log_count})")
                };
                if ui.small_button(log_btn_lbl).clicked() {
                    self.mission_log_open = !self.mission_log_open;
                }
                if let Some(latest) = self.state.mission_log.last() {
                    ui.weak(format!("Latest: {latest}"));
                }
            });
        });

        if self.mission_log_open {
            egui::TopBottomPanel::bottom("flight_director_log_drawer")
                .resizable(true)
                .default_height(145.0)
                .min_height(90.0)
                .max_height(260.0)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.strong(format!(
                            "Flight Director Telemetry & Event Log ({} entries)",
                            self.state.mission_log.len()
                        ));
                        if ui.small_button("Close").clicked() {
                            self.mission_log_open = false;
                        }
                    });
                    ui.separator();
                    ui.push_id("mission_log_scroll", |ui| {
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            if self.state.mission_log.is_empty() {
                                ui.weak("No Flight Director telemetry logged yet.");
                            }
                            for entry in self.state.mission_log.iter().rev() {
                                let color = if entry.contains("[SLINGSHOT]") {
                                    egui::Color32::from_rgb(255, 205, 95)
                                } else if entry.contains("[SURFACE OPS]") {
                                    egui::Color32::from_rgb(95, 235, 255)
                                } else if entry.contains("[SUN-EARTH L2]") {
                                    egui::Color32::from_rgb(165, 195, 255)
                                } else if entry.contains("[LOSS OF SIGNAL]") {
                                    egui::Color32::from_rgb(255, 135, 120)
                                } else if entry.contains("ACHIEVEMENT") {
                                    egui::Color32::from_rgb(110, 245, 175)
                                } else {
                                    egui::Color32::from_rgb(205, 220, 240)
                                };
                                ui.colored_label(color, entry);
                            }
                        });
                    });
                });
        }

        if self.show_about_window {
            let mut open = self.show_about_window;
            egui::Window::new("About HelioProbe")
                .open(&mut open)
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
                .show(ctx, |ui| {
                    ui.heading("HelioProbe v2.3.0");
                    ui.label("Copyright CRV Science Network");
                    ui.label("License: GNU General Public License v3.0 or later (GPL-3.0-or-later)");
                    ui.separator();
                    ui.label("Native Rust 3D Space Probe, Sun-Earth L2 Space Telescopes,");
                    ui.label("3D Universal-Variable Lambert Porkchop & Gravity-Assist Lab,");
                    ui.label("Dynamic Comets, and Ocean-World Cryobot Simulator.");
                    ui.add_space(6.0);
                    ui.hyperlink_to(
                        "CRV Science Network Website (https://crvscience.com)",
                        "https://crvscience.com",
                    );
                    ui.hyperlink_to(
                        "CRV Science Public Apps (https://crvscience.com/apps)",
                        "https://crvscience.com/apps",
                    );
                    ui.add_space(6.0);
                    if ui.button("Close").clicked() {
                        self.show_about_window = false;
                    }
                });
            if !open {
                self.show_about_window = false;
            }
        }

        // Central 3D wgpu Viewport
        egui::CentralPanel::default()
            .frame(
                egui::Frame::default()
                    .inner_margin(0.0)
                    .fill(egui::Color32::from_rgb(4, 7, 15)),
            )
            .show(ctx, |ui| {
                let (rect, response) =
                    ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());

                if response.dragged() {
                    let delta = response.drag_delta();
                    self.camera.yaw_rad -= (delta.x as f64) * 0.0065;
                    let is_cryobot_active = self
                        .state
                        .surface_deployments
                        .get(&self.selected_body_id)
                        .and_then(|units| units.last())
                        .is_some_and(|u| u.kind == SurfaceUnitKind::OceanCryobot);
                    let min_pitch = if self.camera_mode == CameraTrackMode::LanderBotView {
                        if is_cryobot_active { -0.45 } else { 0.05 }
                    } else {
                        -1.45
                    };
                    self.camera.pitch_rad =
                        (self.camera.pitch_rad + (delta.y as f64) * 0.0065).clamp(min_pitch, 1.45);
                }
                if response.hovered() {
                    let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                    if scroll.abs() > 0.01 {
                        let factor = (1.0 - (scroll as f64) * 0.0022).clamp(0.82, 1.22);
                        self.desired_cam_distance =
                            (self.desired_cam_distance * factor).clamp(0.015, 260.0);
                        self.camera.distance = self.desired_cam_distance;
                    }
                }

                let pulse_arg = self.active_scan_pulse.as_ref().and_then(|(id, t0)| {
                    let elapsed = now.duration_since(*t0).as_secs_f64();
                    if elapsed < 2.6 {
                        Some((id.as_str(), elapsed))
                    } else {
                        None
                    }
                });

                let frame_build = build_cosmos_frame(
                    rect,
                    &self.camera,
                    &catalog,
                    &self.state.probes,
                    &self.state.surface_deployments,
                    &self.state.atmospheric_probes,
                    &self.state.sample_capsules,
                    &self.state.deployed_l2_observatories,
                    &self.state.comet_perihelion_offsets,
                    self.state.days_since_j2000,
                    self.anim_time_sec,
                    &self.selected_body_id,
                    pulse_arg,
                    self.show_orbits,
                );

                if response.clicked() {
                    if let Some(click_pos) = response.interact_pointer_pos() {
                        let mut best: Option<(&str, f32)> = None;
                        for lbl in &frame_build.labels {
                            if lbl.text.starts_with("Probe ") {
                                continue;
                            }
                            let d = click_pos.distance(lbl.screen_pos);
                            let hit_r = (lbl.radius_px + 14.0).max(20.0);
                            if d <= hit_r && best.map_or(true, |(_, bd)| d < bd) {
                                best = Some((lbl.body_id.as_str(), d));
                            }
                        }
                        if let Some((hit_id, _)) = best {
                            self.selected_body_id = hit_id.to_string();
                        }
                    }
                }

                ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                    rect,
                    CosmosPaintCallback {
                        vertex_bytes: frame_build.vertex_bytes,
                        vertex_count: frame_build.vertex_count,
                    },
                ));

                if self.show_labels {
                    let painter = ui.painter_at(rect);
                    let placed_labels =
                        deoverlap_screen_labels(&frame_build.labels, &self.selected_body_id, rect);
                    for (lbl, text_pos) in placed_labels {
                        let is_sel =
                            lbl.body_id == self.selected_body_id && !lbl.text.starts_with("Probe ");
                        if is_sel {
                            painter.circle_stroke(
                                lbl.screen_pos,
                                lbl.radius_px + 5.0,
                                egui::Stroke::new(1.6_f32, egui::Color32::from_rgb(95, 240, 200)),
                            );
                        }
                        painter.text(
                            text_pos,
                            egui::Align2::CENTER_BOTTOM,
                            &lbl.text,
                            egui::FontId::proportional(if is_sel { 13.0 } else { 11.0 }),
                            if is_sel {
                                egui::Color32::WHITE
                            } else {
                                lbl.color
                            },
                        );
                    }
                }

                // Live Lander / Cryobot / Entry-Probe POV HUD Overlay
                if self.camera_mode == CameraTrackMode::LanderBotView {
                    let hud_w = 438.0_f32.min((rect.width() - 28.0).max(240.0));
                    let hud_rect = egui::Rect::from_min_size(
                        egui::pos2(rect.right() - hud_w - 14.0, rect.top() + 14.0),
                        egui::vec2(hud_w, 86.0),
                    );
                    let p = ui.painter_at(rect);
                    p.rect_filled(
                        hud_rect,
                        6.0,
                        egui::Color32::from_rgba_unmultiplied(8, 20, 34, 225),
                    );
                    p.rect_stroke(
                        hud_rect,
                        6.0,
                        egui::Stroke::new(1.2_f32, egui::Color32::from_rgb(56, 189, 248)),
                        egui::StrokeKind::Inside,
                    );
                    let active_ap = self
                        .state
                        .atmospheric_probes
                        .iter()
                        .rev()
                        .find(|ap| ap.body_id == self.selected_body_id);
                    let (line1, line2) = if let Some(ap) = active_ap {
                        (
                            format!("AEP-{} @ {} (Atmospheric Descent)", ap.id, ap.body_name),
                            format!(
                                "Alt {:.0} km | {:.2} bar | {:.0} °C",
                                ap.current_alt_km, ap.current_pressure_bar, ap.current_temp_c
                            ),
                        )
                    } else if let Some(units) =
                        self.state.surface_deployments.get(&self.selected_body_id)
                    {
                        if let Some(u) = units.last() {
                            let status = if u.descent_progress < 1.0 {
                                format!("POWERED DESCENT {:.0}%", u.descent_progress * 100.0)
                            } else {
                                "SURFACE ANCHORED • ACTIVE".to_string()
                            };
                            (
                                format!("{} [{}]", u.kind.label(), status),
                                u.telemetry_headline.clone(),
                            )
                        } else {
                            (
                                "Surface Recon Horizon POV".to_string(),
                                "Deploy Lander or Cryobot in Landers Tab".to_string(),
                            )
                        }
                    } else {
                        (
                            "Surface Recon Horizon POV".to_string(),
                            "Deploy Lander or Cryobot in Landers Tab".to_string(),
                        )
                    };
                    p.text(
                        egui::pos2(hud_rect.left() + 12.0, hud_rect.top() + 18.0),
                        egui::Align2::LEFT_CENTER,
                        "6. LANDER / CRYOBOT / ENTRY POV TELEMETRY",
                        egui::FontId::proportional(11.5),
                        egui::Color32::from_rgb(95, 235, 255),
                    );
                    p.text(
                        egui::pos2(hud_rect.left() + 12.0, hud_rect.top() + 42.0),
                        egui::Align2::LEFT_CENTER,
                        line1,
                        egui::FontId::proportional(11.0),
                        egui::Color32::WHITE,
                    );
                    p.text(
                        egui::pos2(hud_rect.left() + 12.0, hud_rect.top() + 66.0),
                        egui::Align2::LEFT_CENTER,
                        line2,
                        egui::FontId::proportional(10.5),
                        egui::Color32::from_rgb(165, 215, 240),
                    );
                }

                if let Some((msg, t0)) = &self.toast_banner {
                    if now.duration_since(*t0).as_secs_f64() < 4.5 {
                        ui.painter_at(rect).rect_filled(
                            egui::Rect::from_min_size(
                                egui::pos2(rect.left() + 16.0, rect.top() + 14.0),
                                egui::vec2(540.0_f32.min(rect.width() - 32.0), 34.0),
                            ),
                            6.0,
                            egui::Color32::from_rgba_unmultiplied(14, 38, 48, 230),
                        );
                        ui.painter_at(rect).text(
                            egui::pos2(rect.left() + 28.0, rect.top() + 31.0),
                            egui::Align2::LEFT_CENTER,
                            msg,
                            egui::FontId::proportional(13.5),
                            egui::Color32::from_rgb(120, 250, 195),
                        );
                    }
                }
            });

        ctx.request_repaint();
    }
}

fn block_on_future<F: std::future::Future>(fut: F) -> F::Output {
    struct NoopWake;
    impl std::task::Wake for NoopWake {
        fn wake(self: std::sync::Arc<Self>) {}
    }
    let waker = std::task::Waker::from(std::sync::Arc::new(NoopWake));
    let mut cx = std::task::Context::from_waker(&waker);
    let mut fut = std::pin::pin!(fut);
    loop {
        match fut.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(val) => return val,
            std::task::Poll::Pending => std::thread::yield_now(),
        }
    }
}

fn write_rgba_png(
    path: &std::path::Path,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let file = std::fs::File::create(path)?;
    let w = std::io::BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(rgba)?;
    Ok(())
}

/// Render an offscreen 1600x1000 RGBA PNG screenshot without opening any OS GUI window.
pub fn render_headless_screenshot(
    db_path: &std::path::Path,
    out_path: &std::path::Path,
    view: u8,
) -> Result<(), Box<dyn std::error::Error>> {
    use eframe::egui_wgpu::{wgpu, Renderer, ScreenDescriptor};

    let width: u32 = 1600;
    let height: u32 = 1000;
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = block_on_future(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or("No GPU adapter available for offscreen rendering")?;
    let (device, queue) = block_on_future(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("helioprobe-offscreen-device"),
            ..Default::default()
        },
        None,
    ))?;

    let format = wgpu::TextureFormat::Rgba8Unorm;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("helioprobe-offscreen-target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let bytes_per_row = width * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("helioprobe-offscreen-staging"),
        size: (bytes_per_row * height) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut renderer = Renderer::new(&device, format, None, 1, false);
    let gpu_pipeline = CosmosGpuPipeline::new(&device, format);
    renderer.callback_resources.insert(gpu_pipeline);

    let egui_ctx = egui::Context::default();
    egui_ctx.set_pixels_per_point(1.0);
    let mut visuals = egui::Visuals::dark();
    visuals.window_fill = egui::Color32::from_rgb(11, 15, 25);
    visuals.panel_fill = egui::Color32::from_rgb(12, 17, 29);
    egui_ctx.set_visuals(visuals);

    let mut app = HelioProbeApp::new_headless(db_path.to_path_buf());
    let catalog = app.full_catalog();

    if view == 1 {
        // View 1: Default 3D Solar System & Spacecraft Orbit view
        app.right_tab = RightTab::ScannerFacts;
        app.selected_body_id = "mars".into();
        app.launch_probe_to_selected(&catalog);
        if let Some(p) = app.state.probes.last_mut() {
            p.transit_progress = 0.36;
        }
        app.camera_mode = CameraTrackMode::FreeSolarOverview;
        app.camera.target = Vec3d::default();
        app.camera.yaw_rad = 0.72;
        app.camera.pitch_rad = 0.44;
        app.desired_cam_distance = 54.0;
        app.camera.distance = 54.0;
        app.paused = true;
    } else if view == 2 {
        // View 2: Sun-Earth L2 Observatories & Comet Tail view
        app.right_tab = RightTab::L2AndComets;
        for obs_id in ["jwst_l2", "roman_l2", "interceptor_l2"] {
            let _ = app.state.deploy_l2_observatory(obs_id);
        }
        if let Some(halley) = catalog.iter().find(|b| b.id == "halley_1p") {
            let _ = app.state.warp_comet_to_perihelion(halley);
        }
        app.selected_body_id = "sun_earth_l2".into();
        if let Some(l2) = catalog.iter().find(|b| b.id == "sun_earth_l2") {
            app.camera.target = body_scene_position_with_offsets(
                l2,
                &catalog,
                app.state.days_since_j2000,
                &app.state.comet_perihelion_offsets,
            );
        }
        app.camera_mode = CameraTrackMode::OrbitView;
        app.desired_cam_distance = 2.10;
        app.camera.distance = 2.10;
        app.camera.yaw_rad = 0.88;
        app.camera.pitch_rad = 0.32;
        app.paused = true;
    } else if view == 3 {
        // View 3: Surface Ops / Grounded Cryobot POV view on Europa
        app.right_tab = RightTab::SurfaceOps;
        app.selected_body_id = "europa".into();
        if let Some(europa) = catalog.iter().find(|b| b.id == "europa") {
            let pid = app.state.next_probe_id;
            app.state.next_probe_id += 1;
            let mut probe = SpaceProbe::launch_with_options(
                pid,
                "CRV-Europa-Cryo".into(),
                PropulsionBus::RtgNuclearElectric,
                europa,
                None,
                &app.state.unlocked_techs,
                &catalog,
                app.state.days_since_j2000,
            );
            probe.phase = ProbePhase::Orbiting;
            let _ = app.state.record_launch(probe);
            let _ = app.state.deploy_surface_unit(europa, SurfaceUnitKind::OceanCryobot);
            if let Some(units) = app.state.surface_deployments.get_mut("europa") {
                for u in units {
                    u.descent_progress = 1.0;
                }
            }
        }
        app.camera_mode = CameraTrackMode::LanderBotView;
        app.lander_cam_initialized = false;
        app.paused = true;
    }

    let warmup_delays_ms = [50u64, 30, 20, 20];
    for (frame_idx, delay_ms) in warmup_delays_ms.into_iter().enumerate() {
        std::thread::sleep(std::time::Duration::from_millis(delay_ms));
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(width as f32, height as f32),
            )),
            viewports: std::iter::once((
                egui::ViewportId::ROOT,
                egui::ViewportInfo {
                    native_pixels_per_point: Some(1.0),
                    ..Default::default()
                },
            ))
            .collect(),
            time: Some(frame_idx as f64 / 60.0),
            predicted_dt: 1.0 / 60.0,
            ..Default::default()
        };
        let full_output = egui_ctx.run(raw_input, |ctx| {
            app.ui_step(ctx);
        });
        for (id, image_delta) in &full_output.textures_delta.set {
            renderer.update_texture(&device, &queue, *id, image_delta);
        }
        let paint_jobs = egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [width, height],
            pixels_per_point: 1.0,
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("helioprobe-offscreen-encoder"),
        });
        let user_cmd_bufs = renderer.update_buffers(
            &device,
            &queue,
            &mut encoder,
            &paint_jobs,
            &screen_descriptor,
        );
        {
            let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("helioprobe-offscreen-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            let mut render_pass = render_pass.forget_lifetime();
            renderer.render(&mut render_pass, &paint_jobs, &screen_descriptor);
        }
        if frame_idx == 3 {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &staging,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row),
                        rows_per_image: Some(height),
                    },
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }
        queue.submit(user_cmd_bufs.into_iter().chain(std::iter::once(encoder.finish())));
        device.poll(wgpu::Maintain::Wait);
        for id in &full_output.textures_delta.free {
            renderer.free_texture(id);
        }
    }

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()??;
    let mapped = slice.get_mapped_range();
    write_rgba_png(out_path, width, height, &mapped)?;
    drop(mapped);
    staging.unmap();
    Ok(())
}
