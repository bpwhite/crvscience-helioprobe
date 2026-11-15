use helioprobe::astrodynamics_lab::{compute_gravity_assist, compute_porkchop_grid};
use helioprobe::catalog::{bootstrap_catalog, ScanBand};
use helioprobe::gamification::{CampaignState, SaveDatabase};
use helioprobe::physics::{
    compute_transfer_from_earth, kepler_position_au, solve_kepler_eccentric_anomaly,
};
use helioprobe::probes::{ProbePhase, PropulsionBus, SpaceProbe};
use helioprobe::public_data::parse_jpl_sbdb_json;
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
