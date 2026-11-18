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

use helioprobe::astrodynamics_lab::{compute_gravity_assist, compute_porkchop_grid};
use helioprobe::catalog::{bootstrap_catalog, ScanBand};
use helioprobe::gamification::{CampaignState, SaveDatabase};
use helioprobe::physics::{
    body_scene_position, compute_transfer_from_earth, kepler_position_au,
    solve_kepler_eccentric_anomaly, visual_body_radius_scene, Vec3d,
};
use helioprobe::probes::{ProbePhase, PropulsionBus, SpaceProbe};
use helioprobe::public_data::parse_jpl_sbdb_json;
use helioprobe::renderer_3d::{build_cosmos_frame, unit_world_pos, MeshBuilder3D, OrbitCamera3D};
use helioprobe::surface_ops::{CapsulePhase, EntryProbePhase, SurfaceUnitKind};
use serde_json::json;

#[test]
fn kepler_earth_one_au_solver_accuracy() {
    let e = 0.0167;
    let m = 1.25;
    let ecc = solve_kepler_eccentric_anomaly(m, e);
    let residual = (ecc - e * ecc.sin() - m).abs();
    assert!(residual < 1e-10, "Kepler residual too large: {residual}");

    let earth_pos = kepler_position_au(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 365.256, 0.0);
    assert!((earth_pos.length() - 1.0).abs() < 1e-9);
}

#[test]
fn probe_fast_transit_inserts_into_orbit_and_unlocks_scans() {
    let catalog = bootstrap_catalog();
    assert_eq!(catalog.len(), 46);
    let bennu = catalog.iter().find(|b| b.id == "bennu").expect("bennu");
    let mut state = CampaignState::default();

    let mut probe = SpaceProbe::launch_from_earth(
        1,
        "OSIRIS-II".into(),
        PropulsionBus::ChemicalCentaur,
        bennu,
        &catalog,
        state.days_since_j2000,
    );
    assert_eq!(probe.phase, ProbePhase::EnRoute);

    let mut arrived = false;
    for _ in 0..40 {
        if probe.step(0.5, 2.0) {
            arrived = true;
            break;
        }
    }
    assert!(arrived && probe.phase == ProbePhase::Orbiting);

    state.record_launch(probe);
    state.record_orbit_insertion(bennu);
    let (newly_scanned, _) = state
        .perform_scan(bennu, ScanBand::ThermalMassSpec)
        .expect("scan succeeds once orbiting");
    assert!(newly_scanned);
    assert!(state.unlocked_achievements.contains("osiris_bennu"));
}

#[test]
fn jupiter_galileo_atmospheric_entry_crush_probe_verified() {
    let catalog = bootstrap_catalog();
    let jupiter = catalog.iter().find(|b| b.id == "jupiter").expect("jupiter");
    let mut state = CampaignState::default();

    let mut probe = SpaceProbe::launch_from_earth(
        1,
        "Galileo-II".into(),
        PropulsionBus::SolarElectricIon,
        jupiter,
        &catalog,
        state.days_since_j2000,
    );
    while !probe.step(0.5, 3.0) {}
    state.record_launch(probe);
    state.record_orbit_insertion(jupiter);

    let mut entry = state
        .launch_atmospheric_probe(jupiter)
        .expect("drop jupiter atmospheric probe");
    assert!((entry.entry_speed_km_s - 47.2).abs() < 0.2);

    while !entry.step(0.4, 2.5) {}
    assert_eq!(entry.phase, EntryProbePhase::TerminalCrushOrImpact);
    assert!(entry.current_pressure_bar > 20.0);
    state.record_atmo_probe_termination(&entry);
    assert!(state.unlocked_achievements.contains("atmospheric_plunge"));
}

#[test]
fn porkchop_and_jupiter_gravity_assist_slingshot_verified() {
    let catalog = bootstrap_catalog();
    let saturn = catalog.iter().find(|b| b.id == "saturn").expect("saturn");
    let jupiter = catalog.iter().find(|b| b.id == "jupiter").expect("jupiter");

    let grid = compute_porkchop_grid(saturn, &catalog, 9814.5, 24, 16);
    assert_eq!(grid.dv_km_s.len(), 24 * 16);
    assert!(grid.min_dv_km_s > 6.0 && grid.min_dv_km_s <= grid.max_dv_km_s);

    let sol = compute_gravity_assist(jupiter, saturn, &catalog).expect("jupiter slingshot");
    assert!(sol.dv_saved_km_s > 1.5);
    assert!(sol.turn_angle_deg > 10.0);

    let mut state = CampaignState::default();
    let mut probe = SpaceProbe::launch_with_options(
        1,
        "Cassini-Next".into(),
        PropulsionBus::RtgNuclearElectric,
        saturn,
        Some(jupiter),
        &state.unlocked_techs,
        &catalog,
        state.days_since_j2000,
    );
    let mut notified_flyby = false;
    for _ in 0..60 {
        let done = probe.step(0.3, 2.5);
        if let Some(aname) = probe.take_flyby_notification() {
            notified_flyby = true;
            state.record_slingshot_flyby(&probe.callsign, &aname);
        }
        if done {
            break;
        }
    }
    assert!(notified_flyby);
    assert!(state.unlocked_achievements.contains("gravity_slingshot"));
}

#[test]
fn cryobot_sample_return_and_tech_tree_preserve_lifetime_rank() {
    let catalog = bootstrap_catalog();
    let europa = catalog.iter().find(|b| b.id == "europa").expect("europa");
    let mut state = CampaignState::default();
    state.science_points = 800;

    state.unlock_tech("cryobot_gphs").expect("unlock cryobot");
    state
        .unlock_tech("astromaterials_curatory")
        .expect("unlock curatory");
    assert_eq!(state.available_sp(), 50);
    assert_eq!(state.science_points, 800);

    let mut probe = SpaceProbe::launch_from_earth(
        1,
        "Clipper-Cryo".into(),
        PropulsionBus::SolarElectricIon,
        europa,
        &catalog,
        state.days_since_j2000,
    );
    while !probe.step(0.5, 3.0) {}
    state.record_launch(probe);
    state.record_orbit_insertion(europa);

    let (dep, _) = state
        .deploy_surface_unit(europa, SurfaceUnitKind::OceanCryobot)
        .expect("deploy europa cryobot");
    assert!(dep.ice_melt_depth_km > 15.0);
    assert!(state.unlocked_achievements.contains("subsurface_cryobot"));

    let bonus = state
        .run_active_surface_experiment("europa", SurfaceUnitKind::OceanCryobot)
        .expect("run active sonar");
    assert_eq!(bonus, 75);

    let mut cap = state
        .launch_sample_return(europa, &catalog)
        .expect("launch sample return");
    while !cap.step(0.5, 3.0) {}
    assert_eq!(cap.phase, CapsulePhase::RecoveredAtEarth);
    state.record_capsule_recovery(&cap.origin_body_name, cap.sample_mass_grams);
    assert!(state.unlocked_achievements.contains("earth_sample_return"));
}

#[test]
fn transfer_delta_v_and_sbdb_parser_verified() {
    let catalog = bootstrap_catalog();
    let mars = catalog.iter().find(|b| b.id == "mars").expect("mars");
    let xfer = compute_transfer_from_earth(mars, &catalog, 0.05);
    assert!(xfer.total_delta_v_km_s > 4.0 && xfer.total_delta_v_km_s < 9.0);
    assert!((xfer.physical_flight_days - 258.9).abs() < 5.0);

    let mock_eros = json!({
        "object": {
            "fullname": "433 Eros (1898 DQ)",
            "des": "433",
            "spkid": "2000433",
            "orbit_class": { "name": "Amor" }
        },
        "orbit": {
            "elements": [
                { "name": "e", "value": "0.223" },
                { "name": "a", "value": "1.458" },
                { "name": "i", "value": "10.83" },
                { "name": "om", "value": "304.3" },
                { "name": "w", "value": "178.8" },
                { "name": "ma", "value": "183.1" },
                { "name": "per", "value": "643.2" }
            ]
        },
        "phys_par": [
            { "name": "diameter", "value": "16.84" },
            { "name": "H", "value": "10.31" }
        ]
    });
    let parsed = parse_jpl_sbdb_json(&mock_eros, "Eros").expect("parse Eros");
    assert!((parsed.semi_major_axis_au - 1.458).abs() < 1e-4);
    assert!(parsed.imported_from_api);
}

#[test]
fn sqlite_wal_save_and_load_roundtrip() {
    let tmp = std::env::temp_dir().join("helioprobe_unit_test.db");
    let _ = std::fs::remove_file(&tmp);
    let db = SaveDatabase::open(&tmp).expect("open sqlite");
    let mut state = CampaignState::default();
    state.science_points = 1250;
    state.spent_sp = 350;
    db.save_slot(1, &state).expect("save slot 1");
    let loaded = db.load_slot(1).expect("load slot 1");
    assert_eq!(loaded.science_points, 1250);
    assert_eq!(loaded.available_sp(), 900);
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn sun_earth_l2_observatories_and_dynamic_comet_tails_verified() {
    use helioprobe::catalog::BodyCategory;
    use helioprobe::l2_comets::{compute_comet_tail_state, sun_earth_l2_center_scene};
    use helioprobe::physics::body_scene_position;

    let catalog = bootstrap_catalog();
    let earth = catalog.iter().find(|b| b.id == "earth").expect("earth");
    let l2 = catalog.iter().find(|b| b.id == "sun_earth_l2").expect("sun_earth_l2");
    assert_eq!(l2.category, BodyCategory::LagrangePoint);

    let e_pos = body_scene_position(earth, &catalog, 9814.5);
    let l2_pos = body_scene_position(l2, &catalog, 9814.5);
    assert!((l2_pos.sub(sun_earth_l2_center_scene(e_pos))).length() < 1e-9);
    assert!(l2_pos.length() > e_pos.length() + 3.0);
    assert!(e_pos.normalized().dot(l2_pos.normalized()) > 0.999999);

    let ecc = solve_kepler_eccentric_anomaly(0.0025, 0.9951);
    let residual = (ecc - 0.9951 * ecc.sin() - 0.0025).abs();
    assert!(residual < 1e-11, "High-e Kepler residual: {residual}");

    let encke = catalog.iter().find(|b| b.id == "encke_2p").expect("encke_2p");
    let tail = compute_comet_tail_state(encke, 9814.5, 0.0);
    assert!(tail.r_helio_au < 0.75);
    assert!(tail.activity > 1.2);
    assert!(tail.ion_tail_dir_scene.dot(tail.anti_sun_dir_scene) > 0.96);

    let mut state = CampaignState::default();
    state.deploy_l2_observatory("jwst_l2").expect("deploy jwst");
    state.deploy_l2_observatory("interceptor_l2").expect("deploy interceptor");
    assert!(state.unlocked_achievements.contains("l2_halo_array"));

    let c67p = catalog.iter().find(|b| b.id == "churyumov_67p").expect("67P");
    let (scanned, _) = state
        .perform_scan(c67p, ScanBand::ThermalMassSpec)
        .expect("JWST remote IR scan of comet");
    assert!(scanned);

    let mut intercept = SpaceProbe::launch_from_l2_interceptor(
        1,
        "ESA-Interceptor-A".into(),
        c67p,
        &state.unlocked_techs,
        &catalog,
        state.days_since_j2000,
    );
    assert!(intercept.launched_from_l2);
    while !intercept.step(0.4, 3.0) {}
    state.record_launch(intercept);
    state.record_orbit_insertion(c67p);
    assert!(state.unlocked_achievements.contains("comet_tail_chaser"));
}

#[test]
fn campaign_command_name_cryobot_descent_and_3d_surface_anchoring_verified() {
    use std::collections::{BTreeMap, BTreeSet};

    let catalog = bootstrap_catalog();
    let europa = catalog.iter().find(|b| b.id == "europa").expect("europa");
    let mut state = CampaignState::default();
    assert_eq!(state.campaign_name, "HelioProbe Deep Space Command");

    let mut probe = SpaceProbe::launch_from_earth(
        1,
        "Europa-Relay".into(),
        PropulsionBus::RtgNuclearElectric,
        europa,
        &catalog,
        state.days_since_j2000,
    );
    while !probe.step(0.5, 4.0) {}
    state.record_launch(probe);
    state.record_orbit_insertion(europa);

    let (dep, _) = state
        .deploy_surface_unit(europa, SurfaceUnitKind::OceanCryobot)
        .expect("deploy cryobot");
    assert!((dep.descent_progress - 0.0).abs() < 1e-9);

    let unit = &mut state
        .surface_deployments
        .get_mut("europa")
        .expect("europa unit")[0];
    let mut touched_down = false;
    for _ in 0..50 {
        if unit.step(0.1) {
            touched_down = true;
            break;
        }
    }
    assert!(touched_down);
    assert!((unit.descent_progress - 1.0).abs() < 1e-9);

    let body_pos = body_scene_position(europa, &catalog, state.days_since_j2000);
    let anchor = unit_world_pos(europa, body_pos, 0, 0.0, unit.descent_progress);
    let surface_dist = anchor.unit_pos.sub(body_pos).length();
    assert!((surface_dist - visual_body_radius_scene(europa)).abs() < 1e-6);

    let mut cam = OrbitCamera3D::default();
    cam.up_override = Some(anchor.normal);
    cam.target = anchor.unit_pos.add(anchor.normal.scale(anchor.rig_scale * 0.45));
    cam.distance = (anchor.rig_scale * 4.4).clamp(0.032, 0.34);
    cam.pitch_rad = 0.24;
    cam.yaw_rad = 0.62;

    let frame = build_cosmos_frame(
        eframe::egui::Rect::from_min_size(eframe::egui::Pos2::ZERO, eframe::egui::vec2(1280.0, 900.0)),
        &cam,
        &catalog,
        &state.probes,
        &state.surface_deployments,
        &state.atmospheric_probes,
        &state.sample_capsules,
        &BTreeSet::new(),
        &BTreeMap::new(),
        state.days_since_j2000,
        0.0,
        "europa",
        None,
        true,
    );
    assert!(
        frame.vertex_count > 1200,
        "Expected curved surface patch + 3D cryobot + billboard vertices, got {}",
        frame.vertex_count
    );

    // Exercise MeshBuilder3D primitive helpers directly
    let mut mb = MeshBuilder3D::new(&cam, 1280.0 / 900.0, Vec3d::new(0.0, 1.0, 0.0));
    mb.add_prism(
        anchor.unit_pos,
        anchor.normal,
        anchor.east,
        8,
        0.01,
        0.01,
        0.01,
        [0.8, 0.85, 0.9],
        1.0,
        24.0,
        0.3,
        0.05,
        true,
        true,
    );
    mb.add_box(
        anchor.unit_pos,
        anchor.east,
        anchor.normal,
        anchor.north,
        0.01,
        0.01,
        0.01,
        [0.9, 0.7, 0.2],
        1.0,
        24.0,
        0.3,
        0.05,
    );
    mb.add_parabolic_dish(
        anchor.unit_pos,
        anchor.normal,
        anchor.east,
        10,
        0.01,
        0.004,
        [0.95, 0.95, 1.0],
        [0.3, 0.3, 0.4],
        32.0,
    );
    mb.add_solar_wing(
        anchor.unit_pos,
        anchor.east,
        anchor.north,
        anchor.normal,
        3,
        0.01,
        0.005,
        0.002,
        [0.1, 0.4, 0.9],
        [0.7, 0.75, 0.8],
    );
    assert!(mb.tri_count() > 20);
}

#[test]
fn r1_contracts_icon_profile_ci_and_json_error_surfacing() {
    use helioprobe::gui::{export_campaign_json, import_campaign_json};
    use helioprobe::icon::helioprobe_window_icon;
    use std::path::Path;

    let icon = helioprobe_window_icon();
    assert_eq!(icon.width, 64);
    assert_eq!(icon.height, 64);
    assert_eq!(icon.rgba.len(), 64 * 64 * 4);
    assert!(icon.rgba.iter().any(|&b| b > 0));

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let cargo_toml = std::fs::read_to_string(manifest_dir.join("Cargo.toml")).expect("Cargo.toml");
    assert!(cargo_toml.contains("strip = \"symbols\""));
    assert!(cargo_toml.contains("lto = \"thin\""));
    assert!(cargo_toml.contains("codegen-units = 1"));
    assert!(cargo_toml.contains("rfd = \"0.15\""));
    assert!(!cargo_toml.contains("apps/private"));
    assert!(manifest_dir.join("assets/helioprobe.ico").is_file());
    assert!(manifest_dir.join(".github/workflows/ci.yml").is_file());

    let state = CampaignState::default();
    let json_path = std::env::temp_dir().join("helioprobe_r1_roundtrip_test.json");
    let _ = std::fs::remove_file(&json_path);
    export_campaign_json(&state, &json_path).expect("export valid json");
    let imported = import_campaign_json(&json_path).expect("import valid json");
    assert_eq!(imported.campaign_name, state.campaign_name);
    let _ = std::fs::remove_file(&json_path);

    let missing_err = import_campaign_json(&std::env::temp_dir().join("nonexistent_hp_test_12345.json"))
        .expect_err("missing file must surface Err");
    assert!(!missing_err.is_empty());
}

#[test]
fn r2_1_l2_observatory_scan_predicates_match_perform_scan() {
    let catalog = bootstrap_catalog();
    let mars = catalog.iter().find(|b| b.id == "mars").expect("mars");
    let halley = catalog.iter().find(|b| b.id == "halley_1p").expect("halley_1p");
    let l2 = catalog.iter().find(|b| b.id == "sun_earth_l2").expect("sun_earth_l2");

    let mut state = CampaignState::default();
    assert!(!state.can_scan_all_bands(mars));
    assert!(!state.can_scan_band(halley, ScanBand::OpticalNir));

    state.deploy_l2_observatory("jwst_l2").expect("deploy jwst_l2");
    assert!(state.can_scan_band(halley, ScanBand::OpticalNir));
    assert!(state.can_scan_band(halley, ScanBand::ThermalMassSpec));
    assert!(!state.can_scan_band(halley, ScanBand::RadarGravity));
    assert!(!state.can_scan_band(mars, ScanBand::OpticalNir));
    assert!(state.can_scan_all_bands(l2));
    for band in ScanBand::ALL {
        assert!(state.can_scan_band(l2, band));
        state.perform_scan(l2, band).expect("L2 self-scan succeeds");
    }

    state.deploy_l2_observatory("roman_l2").expect("deploy roman_l2");
    assert!(state.can_scan_band(mars, ScanBand::OpticalNir));
    assert!(!state.can_scan_band(mars, ScanBand::ThermalMassSpec));
    state
        .perform_scan(mars, ScanBand::OpticalNir)
        .expect("Roman remote optical scan succeeds");
}

#[test]
fn r2_2_comet_perihelion_offsets_propagate_to_probes_capsules_and_halley_id() {
    use helioprobe::l2_comets::compute_comet_tail_state;
    use helioprobe::physics::{body_scene_position_with_offsets, probe_orbit_radius_scene};

    let catalog = bootstrap_catalog();
    assert!(catalog.iter().all(|b| b.id != "1p_halley"));
    let halley = catalog.iter().find(|b| b.id == "halley_1p").expect("halley_1p");

    let mut state = CampaignState::default();
    state.warp_comet_to_perihelion(halley).expect("warp halley");
    let off_deg = state.comet_perihelion_offsets.get("halley_1p").copied().unwrap_or(0.0);
    let tail = compute_comet_tail_state(halley, state.days_since_j2000, off_deg);
    assert!(
        tail.r_helio_au < 1.2 && tail.activity > 0.8,
        "Expected Halley perihelion warp to bring r_helio_au < 1.2 AU with active coma: r={}, act={}",
        tail.r_helio_au,
        tail.activity
    );

    let unwarped = body_scene_position(halley, &catalog, state.days_since_j2000);
    let warped = body_scene_position_with_offsets(
        halley,
        &catalog,
        state.days_since_j2000,
        &state.comet_perihelion_offsets,
    );
    assert!(
        warped.sub(unwarped).length() > 5.0,
        "Expected Halley perihelion warp to shift scene position significantly"
    );

    let mut probe = SpaceProbe::launch_from_earth(
        1,
        "Giotto-Next".into(),
        PropulsionBus::SolarElectricIon,
        halley,
        &catalog,
        state.days_since_j2000,
    );
    probe.phase = ProbePhase::Orbiting;
    let p_pos = probe.scene_position_with_offsets(
        &catalog,
        state.days_since_j2000,
        &state.comet_perihelion_offsets,
    );
    let dist_to_warped = p_pos.sub(warped).length();
    let expected_r = probe_orbit_radius_scene(halley);
    assert!(
        (dist_to_warped - expected_r).abs() < 1e-4,
        "Orbiting probe must track warped comet position: got {dist_to_warped} vs {expected_r}"
    );

    state.record_launch(probe);
    state.record_orbit_insertion(halley);
    let cap = state
        .launch_sample_return(halley, &catalog)
        .expect("launch sample return from warped Halley");
    let cap_origin = Vec3d::new(
        cap.origin_scene_pos[0],
        cap.origin_scene_pos[1],
        cap.origin_scene_pos[2],
    );
    assert!(cap_origin.sub(warped).length() < 1e-6);
}

#[test]
fn r2_3_background_public_fetch_polling_nonblocking() {
    use helioprobe::catalog::BodyCategory;
    use helioprobe::gui::{HelioProbeApp, PublicDataFetchResult};
    use std::path::PathBuf;
    use std::sync::mpsc;

    let mut app = HelioProbeApp::new_headless(PathBuf::from("unused_test.db"));
    let (tx, rx) = mpsc::channel();
    app.inject_fetch_receiver_for_test("Mock SBDB", rx);
    assert!(app.is_public_fetch_pending());
    assert!(!app.poll_pending_fetch());

    let mut mock_body = bootstrap_catalog()[0].clone();
    mock_body.id = "mock_eros_433".into();
    mock_body.name = "433 Eros Mock".into();
    mock_body.category = BodyCategory::Asteroid;
    mock_body.imported_from_api = true;

    tx.send(PublicDataFetchResult::ImportedBody {
        source: "JPL SBDB",
        result: Ok(mock_body),
    })
    .expect("send mock result");

    assert!(app.poll_pending_fetch());
    assert!(!app.is_public_fetch_pending());
    assert!(app.state.imported_bodies.iter().any(|b| b.id == "mock_eros_433"));
    assert!(app.api_status_message().contains("433 Eros Mock"));

    let (tx2, rx2) = mpsc::channel();
    app.inject_fetch_receiver_for_test("Mock Err", rx2);
    tx2.send(PublicDataFetchResult::HorizonsNote(Err("network timeout".into())))
        .expect("send err");
    assert!(app.poll_pending_fetch());
    assert_eq!(app.api_status_message(), "network timeout");
}

#[test]
fn r2_4_universal_variable_lambert_porkchop_and_gravity_assist_filter() {
    use helioprobe::astrodynamics_lab::solve_lambert_branch;
    use helioprobe::gui::porkchop_cell_from_pointer;
    use helioprobe::physics::{
        kepler_state_ecliptic_km, propagate_kepler_ecliptic_km, AU_KM, GM_SUN_KM3_S2,
    };

    let catalog = bootstrap_catalog();
    let earth = catalog.iter().find(|b| b.id == "earth").expect("earth");
    let mars = catalog.iter().find(|b| b.id == "mars").expect("mars");
    let ceres = catalog.iter().find(|b| b.id == "ceres").expect("ceres");
    let venus = catalog.iter().find(|b| b.id == "venus").expect("venus");
    let mercury = catalog.iter().find(|b| b.id == "mercury").expect("mercury");
    let jupiter = catalog.iter().find(|b| b.id == "jupiter").expect("jupiter");
    let saturn = catalog.iter().find(|b| b.id == "saturn").expect("saturn");
    let neptune = catalog.iter().find(|b| b.id == "neptune").expect("neptune");

    // 1. Right-handed state vector & vis-viva conservation
    let (r_e, v_e) = kepler_state_ecliptic_km(earth, 9814.5, 0.0);
    let h_e = r_e.cross(v_e);
    assert!(h_e.z > 0.0, "Prograde orbit must have positive ecliptic +Z angular momentum");
    let eps_e = 0.5 * v_e.dot(v_e) - GM_SUN_KM3_S2 / r_e.length();
    let expected_eps = -GM_SUN_KM3_S2 / (2.0 * earth.semi_major_axis_au * AU_KM);
    assert!((eps_e - expected_eps).abs() / expected_eps.abs() < 1e-4);

    // 2. 3D Lambert solver round-trip propagation accuracy on inclined Earth->Ceres transfer
    let dt_days = 260.0;
    let dt_sec = dt_days * 86_400.0;
    let (r_c, _) = kepler_state_ecliptic_km(ceres, 9814.5 + dt_days, 0.0);
    let (v1, _v2) = solve_lambert_branch(r_e, r_c, dt_sec, GM_SUN_KM3_S2, 1.0)
        .expect("Earth->Ceres short-way Lambert solution");
    let (r_prop, _) = propagate_kepler_ecliptic_km(r_e, v1, dt_sec, GM_SUN_KM3_S2);
    let miss_km = r_prop.sub(r_c).length();
    assert!(
        miss_km < 5.0,
        "Lambert propagation miss distance too large: {miss_km} km"
    );

    // 3. Porkchop grid populates C3 and realistic delta-v for inclined targets
    let grid = compute_porkchop_grid(ceres, &catalog, 9814.5, 16, 12);
    assert_eq!(grid.c3_km2_s2.len(), 16 * 12);
    assert!(grid.best_c3_km2_s2 > 5.0 && grid.best_c3_km2_s2.is_finite());

    // 4. Gravity-assist feasibility filter
    assert!(compute_gravity_assist(jupiter, saturn, &catalog).is_some());
    assert!(compute_gravity_assist(jupiter, neptune, &catalog).is_some());
    assert!(compute_gravity_assist(venus, mercury, &catalog).is_some());
    assert!(compute_gravity_assist(saturn, venus, &catalog).is_none());
    assert!(compute_gravity_assist(mars, saturn, &catalog).is_none());
    assert!(compute_gravity_assist(jupiter, venus, &catalog).is_none());
    assert!(compute_gravity_assist(jupiter, earth, &catalog).is_none());

    // 5. Inverted-Y Porkchop cell pointer mapping
    let p_rect = eframe::egui::Rect::from_min_size(
        eframe::egui::pos2(100.0, 200.0),
        eframe::egui::vec2(280.0, 200.0),
    );
    assert_eq!(
        porkchop_cell_from_pointer(p_rect, eframe::egui::pos2(101.0, 201.0), 28, 20),
        Some((0, 19))
    );
    assert_eq!(
        porkchop_cell_from_pointer(p_rect, eframe::egui::pos2(101.0, 399.0), 28, 20),
        Some((0, 0))
    );
    assert_eq!(
        porkchop_cell_from_pointer(p_rect, eframe::egui::pos2(379.0, 399.0), 28, 20),
        Some((27, 0))
    );
}

#[test]
fn r2_5_catalog_filter_label_deoverlap_cislunar_scaling_and_cryobot_cutaway() {
    use helioprobe::catalog::BodyCategory;
    use helioprobe::gui::{deoverlap_screen_labels, filter_catalog_bodies};
    use helioprobe::renderer_3d::ProjectedLabel;
    use std::collections::{BTreeMap, BTreeSet};

    let catalog = bootstrap_catalog();
    assert_eq!(filter_catalog_bodies(&catalog, "", None).len(), 46);
    assert_eq!(
        filter_catalog_bodies(&catalog, "", Some(BodyCategory::Comet)).len(),
        4
    );
    let halley_hits = filter_catalog_bodies(&catalog, "halley", None);
    assert_eq!(halley_hits.len(), 1);
    assert_eq!(halley_hits[0].id, "halley_1p");

    // De-overlap keeps selected body and resolves identical screen positions
    let rect = eframe::egui::Rect::from_min_size(
        eframe::egui::pos2(0.0, 0.0),
        eframe::egui::vec2(800.0, 600.0),
    );
    let labels = vec![
        ProjectedLabel {
            body_id: "earth".into(),
            text: "Earth".into(),
            screen_pos: eframe::egui::pos2(400.0, 300.0),
            radius_px: 20.0,
            color: eframe::egui::Color32::LIGHT_BLUE,
            depth: 2.0,
        },
        ProjectedLabel {
            body_id: "moon".into(),
            text: "The Moon".into(),
            screen_pos: eframe::egui::pos2(402.0, 300.0),
            radius_px: 10.0,
            color: eframe::egui::Color32::GRAY,
            depth: 2.1,
        },
    ];
    let placed = deoverlap_screen_labels(&labels, "earth", rect);
    assert_eq!(placed.len(), 2);
    assert!((placed[0].1.y - placed[1].1.y).abs() >= 10.0);

    // Cis-lunar View 2 projection keeps Earth & Moon separated with bounded radius
    let l2 = catalog.iter().find(|b| b.id == "sun_earth_l2").expect("sun_earth_l2");
    let mut cam = OrbitCamera3D::default();
    cam.target = body_scene_position(l2, &catalog, 9814.5);
    cam.distance = 2.10;
    cam.yaw_rad = 0.88;
    cam.pitch_rad = 0.32;
    let l2_frame = build_cosmos_frame(
        eframe::egui::Rect::from_min_size(eframe::egui::Pos2::ZERO, eframe::egui::vec2(1600.0, 1000.0)),
        &cam,
        &catalog,
        &[],
        &BTreeMap::new(),
        &[],
        &[],
        &BTreeSet::new(),
        &BTreeMap::new(),
        9814.5,
        0.0,
        "sun_earth_l2",
        None,
        true,
    );
    let earth_lbl = l2_frame.labels.iter().find(|l| l.body_id == "earth");
    let moon_lbl = l2_frame.labels.iter().find(|l| l.body_id == "moon");
    if let (Some(el), Some(ml)) = (earth_lbl, moon_lbl) {
        let sep = el.screen_pos.distance(ml.screen_pos);
        assert!(
            sep > el.radius_px + ml.radius_px,
            "Earth and Moon spheres overlap in L2 view: sep={sep}, sum_r={}",
            el.radius_px + ml.radius_px
        );
    }

    // Sub-ice Cryobot cutaway on Europa produces rich geometry
    let europa = catalog.iter().find(|b| b.id == "europa").expect("europa");
    let mut state = CampaignState::default();
    let mut probe = SpaceProbe::launch_from_earth(
        1,
        "Europa-Cutaway-Test".into(),
        PropulsionBus::RtgNuclearElectric,
        europa,
        &catalog,
        state.days_since_j2000,
    );
    probe.phase = ProbePhase::Orbiting;
    state.record_launch(probe);
    state.record_orbit_insertion(europa);
    state
        .deploy_surface_unit(europa, SurfaceUnitKind::OceanCryobot)
        .expect("deploy cryobot");
    if let Some(units) = state.surface_deployments.get_mut("europa") {
        units[0].descent_progress = 1.0;
    }
    let body_pos = body_scene_position(europa, &catalog, state.days_since_j2000);
    let anchor = unit_world_pos(europa, body_pos, 0, 0.0, 1.0);
    let mut cryo_cam = OrbitCamera3D::default();
    cryo_cam.up_override = Some(anchor.normal);
    cryo_cam.target = anchor.surface_pt.sub(anchor.normal.scale(anchor.rig_scale * 0.65));
    cryo_cam.distance = (anchor.rig_scale * 6.4).clamp(0.05, 0.44);
    cryo_cam.pitch_rad = 0.14;
    cryo_cam.yaw_rad = 0.62;
    let cryo_frame = build_cosmos_frame(
        eframe::egui::Rect::from_min_size(eframe::egui::Pos2::ZERO, eframe::egui::vec2(1600.0, 1000.0)),
        &cryo_cam,
        &catalog,
        &state.probes,
        &state.surface_deployments,
        &[],
        &[],
        &BTreeSet::new(),
        &BTreeMap::new(),
        state.days_since_j2000,
        0.0,
        "europa",
        None,
        true,
    );
    assert!(
        cryo_frame.vertex_count > 2200,
        "Expected full Cryobot cutaway + ocean + tether mesh (>2200 verts), got {}",
        cryo_frame.vertex_count
    );
}

fn rk4_two_body(r0: Vec3d, v0: Vec3d, dt_sec: f64, mu: f64, steps: usize) -> (Vec3d, Vec3d) {
    let h = dt_sec / steps as f64;
    let mut r = r0;
    let mut v = v0;
    let accel = |pos: Vec3d| -> Vec3d {
        let rm = pos.length();
        pos.scale(-mu / (rm * rm * rm))
    };
    for _ in 0..steps {
        let k1_r = v;
        let k1_v = accel(r);

        let k2_r = v.add(k1_v.scale(0.5 * h));
        let k2_v = accel(r.add(k1_r.scale(0.5 * h)));

        let k3_r = v.add(k2_v.scale(0.5 * h));
        let k3_v = accel(r.add(k2_r.scale(0.5 * h)));

        let k4_r = v.add(k3_v.scale(h));
        let k4_v = accel(r.add(k3_r.scale(h)));

        r = r.add(
            k1_r.add(k2_r.scale(2.0))
                .add(k3_r.scale(2.0))
                .add(k4_r)
                .scale(h / 6.0),
        );
        v = v.add(
            k1_v.add(k2_v.scale(2.0))
                .add(k3_v.scale(2.0))
                .add(k4_v)
                .scale(h / 6.0),
        );
    }
    (r, v)
}

#[test]
fn lambert_and_kepler_porkchop_full_grid_roundtrip_zero_nans() {
    use helioprobe::astrodynamics_lab::solve_lambert_branch;
    use helioprobe::physics::{
        kepler_state_ecliptic_km, propagate_kepler_ecliptic_km, AU_KM, GM_SUN_KM3_S2,
    };
    use std::f64::consts::PI;

    let catalog = bootstrap_catalog();
    let earth = catalog.iter().find(|b| b.id == "earth").expect("earth");
    let bodies = [
        "mercury",
        "venus",
        "mars",
        "ceres",
        "jupiter",
        "saturn",
        "pluto",
        "halley_1p",
        "encke_2p",
    ];
    let j2000 = 9500.0;
    let cols = 20usize;
    let rows = 20usize;

    for id in bodies {
        let target = catalog.iter().find(|b| b.id == id).expect(id);
        let grid = compute_porkchop_grid(target, &catalog, j2000, cols, rows);
        assert!(grid.dv_km_s.iter().all(|v| v.is_finite()));
        assert!(grid.c3_km2_s2.iter().all(|v| v.is_finite()));

        let t_earth = 365.256;
        let t_targ = target.orbital_period_days.max(30.0);
        let synodic = (1.0 / ((1.0 / t_earth) - (1.0 / t_targ)).abs()).clamp(90.0, 900.0);
        let a2 = target.semi_major_axis_au.max(0.2);
        let t_hohmann_days =
            PI * (((0.5 * (1.0 + a2) * AU_KM).powi(3)) / GM_SUN_KM3_S2).sqrt() / 86_400.0;
        let span_dep = (synodic * 1.15).clamp(160.0, 820.0);

        let mut solved_cells = 0usize;
        for r_idx in 0..rows {
            let tof = t_hohmann_days * (0.48 + 1.15 * (r_idx as f64 / (rows - 1) as f64));
            let dt_sec = tof * 86_400.0;
            for c_idx in 0..cols {
                let dep_off = span_dep * (c_idx as f64 / (cols - 1) as f64);
                let t_dep = j2000 + dep_off;
                let t_arr = t_dep + tof;
                let (r_e, _) = kepler_state_ecliptic_km(earth, t_dep, 0.0);
                let (r_t, _) = kepler_state_ecliptic_km(target, t_arr, 0.0);

                for dm in [1.0_f64, -1.0_f64] {
                    if let Some((v1, v2)) =
                        solve_lambert_branch(r_e, r_t, dt_sec, GM_SUN_KM3_S2, dm)
                    {
                        solved_cells += 1;
                        assert!(
                            v1.length().is_finite() && v2.length().is_finite(),
                            "NaN/Inf Lambert velocity for {id} r={r_idx} c={c_idx} dm={dm}"
                        );
                        let (r_prop, v_prop) =
                            propagate_kepler_ecliptic_km(r_e, v1, dt_sec, GM_SUN_KM3_S2);
                        assert!(
                            r_prop.length().is_finite() && v_prop.length().is_finite(),
                            "NaN/Inf propagated state for {id} r={r_idx} c={c_idx} dm={dm}"
                        );
                        let pos_err = r_prop.sub(r_t).length();
                        let vel_err = v_prop.sub(v2).length();
                        assert!(
                            pos_err < 10.0,
                            "Lambert round-trip miss > 10 km for {id} r={r_idx} c={c_idx} dm={dm}: err={pos_err} km (|v1|={}, r_prop_len={}, r_t_len={}, dt_days={})",
                            v1.length(),
                            r_prop.length(),
                            r_t.length(),
                            tof
                        );
                        assert!(
                            vel_err < 1e-2,
                            "Lambert velocity mismatch for {id} r={r_idx} c={c_idx} dm={dm}: err={vel_err} km/s"
                        );
                    }
                }
            }
        }
        assert!(
            solved_cells > 350,
            "Expected majority of 800 branch evaluations solved for {id}, got {solved_cells}"
        );
    }
}

#[test]
fn hyperbolic_kepler_propagation_matches_rk4_zero_nans() {
    use helioprobe::astrodynamics_lab::solve_lambert_branch;
    use helioprobe::physics::{
        propagate_kepler_ecliptic_km, stumpff_c_s, AU_KM, GM_SUN_KM3_S2,
    };

    // Fast hyperbolic 3D Lambert transfer (~107-111 km/s, alpha < 0)
    let r1 = Vec3d::new(AU_KM, 0.0, 0.0);
    let r2 = Vec3d::new(-1.2 * AU_KM, 0.8 * AU_KM, 0.9 * AU_KM);
    let dt_sec = 40.0 * 86_400.0;

    for dm in [1.0_f64, -1.0_f64] {
        let (v1, v2) = solve_lambert_branch(r1, r2, dt_sec, GM_SUN_KM3_S2, dm)
            .expect("fast hyperbolic Lambert solution");
        let alpha = 2.0 / r1.length() - v1.dot(v1) / GM_SUN_KM3_S2;
        assert!(alpha < 0.0, "Expected hyperbolic alpha < 0, got {alpha}");

        let (r_prop, v_prop) = propagate_kepler_ecliptic_km(r1, v1, dt_sec, GM_SUN_KM3_S2);
        assert!(r_prop.length().is_finite() && v_prop.length().is_finite());

        let (r_rk4, v_rk4) = rk4_two_body(r1, v1, dt_sec, GM_SUN_KM3_S2, 8000);
        let err_vs_rk4 = r_prop.sub(r_rk4).length();
        let err_vs_r2 = r_prop.sub(r2).length();
        let vel_err_rk4 = v_prop.sub(v_rk4).length();
        let vel_err_v2 = v_prop.sub(v2).length();
        assert!(
            err_vs_rk4 < 1.0 && err_vs_r2 < 1.0,
            "Hyperbolic dm={dm} pos error too large: vs_rk4={err_vs_rk4} km, vs_r2={err_vs_r2} km"
        );
        assert!(
            vel_err_rk4 < 1e-4 && vel_err_v2 < 1e-4,
            "Hyperbolic dm={dm} vel error too large: vs_rk4={vel_err_rk4}, vs_v2={vel_err_v2}"
        );

        // Reverse propagation back to r1
        let (r_back, v_back) =
            propagate_kepler_ecliptic_km(r_prop, v_prop, -dt_sec, GM_SUN_KM3_S2);
        assert!(r_back.sub(r1).length() < 1e-3);
        assert!(v_back.sub(v1).length() < 1e-6);
    }

    // Extreme negative-z Stumpff overflow guard check
    for z in [-1e4, -1e6, -1e8, -1e12] {
        let (c, s) = stumpff_c_s(z);
        assert!(c.is_finite() && s.is_finite(), "Non-finite Stumpff at z={z}");
    }
    let r_ext = Vec3d::new(AU_KM, 0.0, 0.0);
    let v_ext = Vec3d::new(20.0, 140.0, 15.0);
    let (r_far, v_far) = propagate_kepler_ecliptic_km(r_ext, v_ext, 1200.0 * 86_400.0, GM_SUN_KM3_S2);
    assert!(r_far.length().is_finite() && v_far.length().is_finite());
    let (r_far_rk4, _) = rk4_two_body(r_ext, v_ext, 1200.0 * 86_400.0, GM_SUN_KM3_S2, 12000);
    assert!(
        r_far.sub(r_far_rk4).length() < 1.0,
        "Extreme hyperbolic mismatch vs RK4: {} km",
        r_far.sub(r_far_rk4).length()
    );
}

#[test]
fn gravity_assist_consistent_3d_accounting_on_halley_and_all_candidates() {
    use helioprobe::astrodynamics_lab::ASSIST_CANDIDATE_IDS;

    let catalog = bootstrap_catalog();
    let jupiter = catalog.iter().find(|b| b.id == "jupiter").expect("jupiter");
    let saturn = catalog.iter().find(|b| b.id == "saturn").expect("saturn");
    let neptune = catalog.iter().find(|b| b.id == "neptune").expect("neptune");
    let venus = catalog.iter().find(|b| b.id == "venus").expect("venus");
    let mercury = catalog.iter().find(|b| b.id == "mercury").expect("mercury");
    let mars = catalog.iter().find(|b| b.id == "mars").expect("mars");
    let earth = catalog.iter().find(|b| b.id == "earth").expect("earth");
    let pluto = catalog.iter().find(|b| b.id == "pluto").expect("pluto");
    let halley = catalog.iter().find(|b| b.id == "halley_1p").expect("halley_1p");

    // Outer-planet assists to inclined retrograde comets must never collapse to an unphysical 3.20 km/s floor
    for assist in [jupiter, saturn] {
        if let Some(sol) = compute_gravity_assist(assist, halley, &catalog) {
            assert!(
                sol.assisted_dv_km_s >= 5.5,
                "Unphysical assisted_dv {} km/s for {} -> halley_1p",
                sol.assisted_dv_km_s,
                assist.id
            );
            assert!((sol.direct_dv_km_s - sol.assisted_dv_km_s - sol.dv_saved_km_s).abs() < 1e-9);
        }
    }

    // Valid assists still succeed with expected savings & turn angles
    let jup_sat = compute_gravity_assist(jupiter, saturn, &catalog).expect("jupiter -> saturn");
    assert!(jup_sat.dv_saved_km_s > 1.5, "jup->sat saved={}", jup_sat.dv_saved_km_s);
    assert!(jup_sat.turn_angle_deg > 10.0);
    assert!(jup_sat.assisted_dv_km_s >= 5.5);

    let jup_nep = compute_gravity_assist(jupiter, neptune, &catalog).expect("jupiter -> neptune");
    assert!(jup_nep.assisted_dv_km_s >= 5.5);
    assert!(jup_nep.dv_saved_km_s > 0.5);

    let ven_mer = compute_gravity_assist(venus, mercury, &catalog).expect("venus -> mercury");
    assert!(ven_mer.assisted_dv_km_s >= 3.5);
    assert!(ven_mer.dv_saved_km_s > 1.0);

    // Infeasible pairs remain None
    for (a, t) in [
        (mars, saturn),
        (saturn, venus),
        (jupiter, venus),
        (jupiter, earth),
        (earth, mars),
        (venus, pluto),
        (mars, jupiter),
        (saturn, jupiter),
        (jupiter, mercury),
    ] {
        assert!(
            compute_gravity_assist(a, t, &catalog).is_none(),
            "Expected None for infeasible {} -> {}",
            a.id,
            t.id
        );
    }

    // Global physical invariant across all 46 bodies x 4 assist candidates
    for aid in ASSIST_CANDIDATE_IDS {
        let ab = catalog.iter().find(|b| b.id == aid).expect(aid);
        for target in &catalog {
            if let Some(sol) = compute_gravity_assist(ab, target, &catalog) {
                assert!(sol.direct_dv_km_s.is_finite() && sol.assisted_dv_km_s.is_finite());
                assert!(
                    (sol.direct_dv_km_s - sol.assisted_dv_km_s - sol.dv_saved_km_s).abs() < 1e-9,
                    "Inconsistent dV accounting for {} -> {}",
                    ab.id,
                    target.id
                );
                assert!(sol.dv_saved_km_s > 0.25);
                if aid == "jupiter" || aid == "saturn" {
                    assert!(
                        sol.assisted_dv_km_s >= 5.5,
                        "Outer assist {} -> {} below LEO->Jupiter escape floor: {}",
                        aid,
                        target.id,
                        sol.assisted_dv_km_s
                    );
                }
            }
        }
    }
}
