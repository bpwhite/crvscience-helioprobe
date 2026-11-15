//! Safe-Rust (`#![forbid(unsafe_code)]`) 3D perspective camera, depth-sorted geometry builder,
//! 2-leg Slingshot trajectory ribbons, Surface Lander/Cryobot beacons, Galileo Atmospheric Entry Plunges,
//! Earth Sample-Return Capsules, ray-picking hit tester, and `eframe::egui_wgpu` hardware WGSL render pipeline.

use crate::catalog::{BodyCategory, CelestialBody};
use crate::l2_comets::{
    compute_comet_tail_state, sample_dust_tail_point, sun_earth_l2_halo_position, L2_OBSERVATORIES,
};
use crate::physics::{
    body_scene_position, body_scene_position_with_offsets, compress_au_to_scene, kepler_position_au,
    visual_body_radius_scene, Vec3d,
};
use crate::probes::{ProbePhase, PropulsionBus, SpaceProbe};
use crate::surface_ops::{
    AtmosphericProbeRun, CapsulePhase, EntryProbePhase, SampleReturnCapsule, SurfaceDeployment,
    SurfaceUnitKind,
};
use eframe::egui;
use eframe::egui_wgpu::{self, wgpu};
use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::TAU;

#[derive(Debug, Clone)]
pub struct OrbitCamera3D {
    pub target: Vec3d,
    pub yaw_rad: f64,
    pub pitch_rad: f64,
    pub distance: f64,
    pub fov_y_rad: f64,
}

impl Default for OrbitCamera3D {
    fn default() -> Self {
        Self {
            target: Vec3d::default(),
            yaw_rad: 0.65,
            pitch_rad: 0.46,
            distance: 78.0,
            fov_y_rad: 50.0_f64.to_radians(),
        }
    }
}

impl OrbitCamera3D {
    pub fn eye_position(&self) -> Vec3d {
        let cp = self.pitch_rad.cos();
        let offset = Vec3d::new(
            self.distance * cp * self.yaw_rad.sin(),
            self.distance * self.pitch_rad.sin(),
            self.distance * cp * self.yaw_rad.cos(),
        );
        self.target.add(offset)
    }

    pub fn project(&self, world: Vec3d, aspect: f64) -> Option<(f32, f32, f64)> {
        let eye = self.eye_position();
        let forward = self.target.sub(eye).normalized();
        let world_up = Vec3d::new(0.0, 1.0, 0.0);
        let right = forward.cross(world_up).normalized();
        let up = right.cross(forward).normalized();

        let rel = world.sub(eye);
        let z_cam = rel.dot(forward);
        if z_cam <= 0.004 {
            return None;
        }
        let x_cam = rel.dot(right);
        let y_cam = rel.dot(up);

        let f = 1.0 / (self.fov_y_rad * 0.5).tan();
        let ndc_x = (x_cam * f / (aspect.max(0.1) * z_cam)) as f32;
        let ndc_y = (y_cam * f / z_cam) as f32;
        Some((ndc_x, ndc_y, z_cam))
    }
}

#[derive(Clone, Copy)]
struct GpuVertex {
    clip_pos: [f32; 2],
    uv: [f32; 2],
    color: [f32; 4],
    kind_param: [f32; 2],
}

impl GpuVertex {
    const BYTE_STRIDE: u64 = 40;

    fn write_le_bytes(self, out: &mut Vec<u8>) {
        for v in [
            self.clip_pos[0],
            self.clip_pos[1],
            self.uv[0],
            self.uv[1],
            self.color[0],
            self.color[1],
            self.color[2],
            self.color[3],
            self.kind_param[0],
            self.kind_param[1],
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
}

struct DepthQuad {
    depth: f64,
    verts: [GpuVertex; 6],
}

fn make_billboard_quad(
    ndc_x: f32,
    ndc_y: f32,
    half_w: f32,
    half_h: f32,
    color: [f32; 4],
    kind: f32,
    param: f32,
) -> [GpuVertex; 6] {
    let v0 = GpuVertex {
        clip_pos: [ndc_x - half_w, ndc_y - half_h],
        uv: [-1.0, -1.0],
        color,
        kind_param: [kind, param],
    };
    let v1 = GpuVertex {
        clip_pos: [ndc_x + half_w, ndc_y - half_h],
        uv: [1.0, -1.0],
        color,
        kind_param: [kind, param],
    };
    let v2 = GpuVertex {
        clip_pos: [ndc_x + half_w, ndc_y + half_h],
        uv: [1.0, 1.0],
        color,
        kind_param: [kind, param],
    };
    let v3 = GpuVertex {
        clip_pos: [ndc_x - half_w, ndc_y + half_h],
        uv: [-1.0, 1.0],
        color,
        kind_param: [kind, param],
    };
    [v0, v1, v2, v0, v2, v3]
}

fn make_line_ribbon(
    a: (f32, f32),
    b: (f32, f32),
    half_thickness_ndc: f32,
    aspect: f32,
    color: [f32; 4],
) -> Option<[GpuVertex; 6]> {
    let dx = (b.0 - a.0) * aspect;
    let dy = b.1 - a.1;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-5 {
        return None;
    }
    let nx = (-dy / len) * (half_thickness_ndc / aspect.max(0.1));
    let ny = (dx / len) * half_thickness_ndc;
    let v0 = GpuVertex {
        clip_pos: [a.0 - nx, a.1 - ny],
        uv: [0.0, -1.0],
        color,
        kind_param: [4.0, 0.0],
    };
    let v1 = GpuVertex {
        clip_pos: [b.0 - nx, b.1 - ny],
        uv: [1.0, -1.0],
        color,
        kind_param: [4.0, 0.0],
    };
    let v2 = GpuVertex {
        clip_pos: [b.0 + nx, b.1 + ny],
        uv: [1.0, 1.0],
        color,
        kind_param: [4.0, 0.0],
    };
    let v3 = GpuVertex {
        clip_pos: [a.0 + nx, a.1 + ny],
        uv: [0.0, 1.0],
        color,
        kind_param: [4.0, 0.0],
    };
    Some([v0, v1, v2, v0, v2, v3])
}

#[derive(Debug, Clone)]
pub struct ProjectedLabel {
    pub body_id: String,
    pub text: String,
    pub screen_pos: egui::Pos2,
    pub radius_px: f32,
    pub color: egui::Color32,
    pub depth: f64,
}

pub struct SceneFrameBuild {
    pub vertex_bytes: Vec<u8>,
    pub vertex_count: u32,
    pub labels: Vec<ProjectedLabel>,
}

#[allow(clippy::too_many_arguments)]
pub fn build_cosmos_frame(
    rect: egui::Rect,
    camera: &OrbitCamera3D,
    catalog: &[CelestialBody],
    probes: &[SpaceProbe],
    surface_deployments: &BTreeMap<String, Vec<SurfaceDeployment>>,
    atmospheric_probes: &[AtmosphericProbeRun],
    sample_capsules: &[SampleReturnCapsule],
    deployed_l2_observatories: &BTreeSet<String>,
    comet_perihelion_offsets: &BTreeMap<String, f64>,
    days_since_j2000: f64,
    anim_time_sec: f64,
    selected_id: &str,
    active_scan_pulse: Option<(&str, f64)>,
    show_orbits: bool,
) -> SceneFrameBuild {
    let width = rect.width().max(1.0);
    let height = rect.height().max(1.0);
    let aspect = (width / height) as f64;
    let aspect_f32 = aspect as f32;

    let mut quads: Vec<DepthQuad> = Vec::with_capacity(2600);
    let mut labels: Vec<ProjectedLabel> = Vec::with_capacity(catalog.len() + probes.len() + 12);

    // 1. Procedural J2000 background starfield sphere (650 stars)
    for idx in 0..650u32 {
        let u = ((idx.wrapping_mul(2654435761)) as f64) / (u32::MAX as f64);
        let v = ((idx.wrapping_mul(1597334677)) as f64) / (u32::MAX as f64);
        let theta = u * TAU;
        let z = v * 2.0 - 1.0;
        let r_xy = (1.0 - z * z).max(0.0).sqrt();
        let star_pos = Vec3d::new(r_xy * theta.cos(), z, r_xy * theta.sin()).scale(420.0);
        if let Some((nx, ny, depth)) = camera.project(star_pos, aspect) {
            if nx.abs() <= 1.05 && ny.abs() <= 1.05 {
                let sz = (2.2 + (idx % 3) as f32 * 0.9) / height;
                let tint = if idx % 5 == 0 {
                    [0.75, 0.86, 1.0, 0.75]
                } else if idx % 7 == 0 {
                    [1.0, 0.88, 0.72, 0.70]
                } else {
                    [0.92, 0.95, 1.0, 0.62]
                };
                quads.push(DepthQuad {
                    depth,
                    verts: make_billboard_quad(nx, ny, sz / aspect_f32, sz, tint, 0.0, 0.0),
                });
            }
        }
    }

    // 2. Heliocentric Keplerian 3D orbital paths
    if show_orbits {
        for body in catalog {
            if matches!(
                body.category,
                BodyCategory::Planet
                    | BodyCategory::DwarfPlanet
                    | BodyCategory::Asteroid
                    | BodyCategory::Comet
            ) && body.semi_major_axis_au > 0.0
            {
                let steps = if body.category == BodyCategory::Comet { 144usize } else { 72usize };
                let is_sel = body.id == selected_id;
                let alpha = if is_sel { 0.62 } else { 0.22 };
                let col = [body.color_rgb[0], body.color_rgb[1], body.color_rgb[2], alpha];
                let mut prev: Option<(f32, f32, f64)> = None;
                for s in 0..=steps {
                    let frac = (s as f64) / (steps as f64);
                    let pos_au = kepler_position_au(
                        body.semi_major_axis_au,
                        body.eccentricity,
                        body.inclination_deg,
                        body.lon_ascending_node_deg,
                        body.arg_periapsis_deg,
                        frac * 360.0,
                        0.0,
                        0.0,
                    );
                    let world = compress_au_to_scene(pos_au);
                    let cur = camera.project(world, aspect);
                    if let (Some(p0), Some(p1)) = (prev, cur) {
                        if p0.0.abs() < 1.6 && p1.0.abs() < 1.6 && p0.1.abs() < 1.6 && p1.1.abs() < 1.6 {
                            if let Some(verts) = make_line_ribbon(
                                (p0.0, p0.1),
                                (p1.0, p1.1),
                                if is_sel { 3.2 / height } else { 1.8 / height },
                                aspect_f32,
                                col,
                            ) {
                                quads.push(DepthQuad {
                                    depth: 0.5 * (p0.2 + p1.2),
                                    verts,
                                });
                            }
                        }
                    }
                    prev = cur;
                }
            }
        }
    }

    let earth_scene_now = catalog
        .iter()
        .find(|b| b.id == "earth")
        .map(|e| body_scene_position(e, catalog, days_since_j2000))
        .unwrap_or(Vec3d::new(16.45, 0.0, 0.0));

    // 2B. Sun-Earth L2 Lissajous Halo Loop & Deployed Space Telescopes (JWST, Roman, Comet Interceptor)
    {
        let mut prev_halo: Option<(f32, f32, f64)> = None;
        for s in 0..=48 {
            let hp = sun_earth_l2_halo_position(
                earth_scene_now,
                days_since_j2000,
                (s as f64 / 48.0) * 360.0,
                0.48,
            );
            let cur = camera.project(hp, aspect);
            if show_orbits {
                if let (Some(p0), Some(p1)) = (prev_halo, cur) {
                    if let Some(verts) = make_line_ribbon(
                        (p0.0, p0.1),
                        (p1.0, p1.1),
                        2.2 / height,
                        aspect_f32,
                        [0.98, 0.82, 0.25, 0.62],
                    ) {
                        quads.push(DepthQuad {
                            depth: 0.5 * (p0.2 + p1.2),
                            verts,
                        });
                    }
                }
            }
            prev_halo = cur;
        }

        for obs in &L2_OBSERVATORIES {
            if deployed_l2_observatories.contains(obs.id) {
                let opos = sun_earth_l2_halo_position(
                    earth_scene_now,
                    days_since_j2000 + anim_time_sec * 1.4,
                    obs.halo_phase_deg,
                    0.48,
                );
                if let Some((ox, oy, odepth)) = camera.project(opos, aspect) {
                    let o_proj = ((1.0 / (camera.fov_y_rad * 0.5).tan()) / odepth) as f32;
                    let oh = (0.028 * o_proj).clamp(4.0 / height, 0.045);
                    let ow = oh / aspect_f32;
                    let shield_col = match obs.id {
                        "jwst_l2" => [0.98, 0.80, 0.22, 0.98],
                        "roman_l2" => [0.62, 0.84, 1.0, 0.98],
                        _ => [0.28, 0.96, 0.78, 0.98],
                    };
                    quads.push(DepthQuad {
                        depth: odepth - 0.005,
                        verts: make_billboard_quad(ox, oy, ow * 1.35, oh * 0.65, [0.85, 0.72, 0.94, 0.85], 1.0, 0.7),
                    });
                    quads.push(DepthQuad {
                        depth: odepth - 0.006,
                        verts: make_billboard_quad(ox, oy, ow * 0.85, oh * 0.85, shield_col, 0.0, anim_time_sec as f32),
                    });
                }
            }
        }
    }

    // 3. Celestial bodies, moons, comets, exoplanets, volumetric nebulae & deployed surface beacons
    for body in catalog {
        let world = body_scene_position_with_offsets(body, catalog, days_since_j2000, comet_perihelion_offsets);
        if body.category == BodyCategory::Comet {
            let off_deg = comet_perihelion_offsets.get(&body.id).copied().unwrap_or(0.0);
            let tail = compute_comet_tail_state(body, days_since_j2000, off_deg);
            if tail.activity > 0.03 {
                if let Some((cx, cy, cdepth)) = camera.project(world, aspect) {
                    let c_proj = ((1.0 / (camera.fov_y_rad * 0.5).tan()) / cdepth) as f32;
                    let ch = ((tail.coma_radius_scene as f32) * c_proj).clamp(6.0 / height, 0.45);
                    quads.push(DepthQuad {
                        depth: cdepth + 0.01,
                        verts: make_billboard_quad(
                            cx,
                            cy,
                            ch / aspect_f32,
                            ch,
                            [0.20, 0.96, 0.78, (0.32 + tail.activity as f32 * 0.22).min(0.82)],
                            2.0,
                            anim_time_sec as f32,
                        ),
                    });
                }
                let ion_end = world.add(tail.ion_tail_dir_scene.scale(tail.ion_tail_len_scene));
                if let (Some(p0), Some(p1)) = (camera.project(world, aspect), camera.project(ion_end, aspect)) {
                    if let Some(iv) = make_line_ribbon(
                        (p0.0, p0.1),
                        (p1.0, p1.1),
                        (4.2 / height) * (tail.activity as f32).clamp(0.6, 1.8),
                        aspect_f32,
                        [0.24, 0.68, 1.0, (0.35 + tail.activity as f32 * 0.26).min(0.88)],
                    ) {
                        quads.push(DepthQuad {
                            depth: 0.5 * (p0.2 + p1.2) + 0.02,
                            verts: iv,
                        });
                    }
                }
                for fan in [-0.75_f64, 0.0, 0.75] {
                    let mut prev_d: Option<(f32, f32, f64)> = None;
                    for seg in 0..=14 {
                        let s = seg as f64 / 14.0;
                        let dp = sample_dust_tail_point(world, &tail, s, fan);
                        let cur_d = camera.project(dp, aspect);
                        if let (Some(a), Some(b)) = (prev_d, cur_d) {
                            let fade = ((1.0 - s * 0.82) * (0.28 + tail.activity * 0.22)).clamp(0.05, 0.78) as f32;
                            if let Some(dv) = make_line_ribbon(
                                (a.0, a.1),
                                (b.0, b.1),
                                (5.5 / height) * (1.0 + s as f32 * 1.4),
                                aspect_f32,
                                [0.98, 0.84, 0.52, fade],
                            ) {
                                quads.push(DepthQuad {
                                    depth: 0.5 * (a.2 + b.2) + 0.025,
                                    verts: dv,
                                });
                            }
                        }
                        prev_d = cur_d;
                    }
                }
            }
        }
        if let Some((nx, ny, depth)) = camera.project(world, aspect) {
            if nx.abs() > 1.65 || ny.abs() > 1.65 {
                continue;
            }
            let base_world_size = visual_body_radius_scene(body);
            let proj_scale = (1.0 / (camera.fov_y_rad * 0.5).tan()) / depth;
            let min_px = if matches!(body.category, BodyCategory::Moon | BodyCategory::Asteroid) {
                3.2 / height
            } else {
                5.5 / height
            };
            let half_h = ((base_world_size * proj_scale) as f32).clamp(min_px, 1.35);
            let half_w = half_h / aspect_f32;

            let sun_angle = world.z.atan2(world.x) as f32 + 3.14159;
            let kind = match body.category {
                BodyCategory::Star => 0.0,
                BodyCategory::Nebula => 2.0,
                _ => 1.0,
            };
            let param = if body.category == BodyCategory::Star || body.category == BodyCategory::Nebula {
                anim_time_sec as f32
            } else {
                sun_angle
            };

            quads.push(DepthQuad {
                depth,
                verts: make_billboard_quad(
                    nx,
                    ny,
                    half_w,
                    half_h,
                    [body.color_rgb[0], body.color_rgb[1], body.color_rgb[2], 1.0],
                    kind,
                    param,
                ),
            });

            if body.has_rings && body.category != BodyCategory::Nebula {
                let (ring_mul, ring_alpha) = if body.id == "saturn" {
                    (2.22, 0.85)
                } else if body.id == "haumea" {
                    (1.85, 0.48)
                } else {
                    (1.48, 0.30)
                };
                quads.push(DepthQuad {
                    depth: depth - 0.01,
                    verts: make_billboard_quad(
                        nx,
                        ny,
                        half_w * ring_mul,
                        half_h * ring_mul,
                        [0.92, 0.86, 0.70, ring_alpha],
                        3.0,
                        anim_time_sec as f32,
                    ),
                });
            }

            if let Some(units) = surface_deployments.get(&body.id) {
                for (u_idx, unit) in units.iter().enumerate() {
                    let ang = anim_time_sec * 0.10 + (u_idx as f64) * 1.9;
                    let n_dir = Vec3d::new(0.92 * ang.cos(), 0.38, 0.92 * ang.sin()).normalized();
                    let surf_pos = world.add(n_dir.scale(base_world_size * 1.004));
                    if let Some((ux, uy, u_depth)) = camera.project(surf_pos, aspect) {
                        let u_proj = ((1.0 / (camera.fov_y_rad * 0.5).tan()) / u_depth) as f32;
                        let rig_h = (0.012 * u_proj).clamp(3.0 / height, 0.065);
                        let rig_w = rig_h / aspect_f32;
                        let col = match unit.kind {
                            SurfaceUnitKind::AeroShellLander => [0.96, 0.76, 0.22, 0.98],
                            SurfaceUnitKind::OceanCryobot => [0.18, 0.92, 1.0, 0.98],
                            SurfaceUnitKind::AtmosphericEntryProbe => [1.0, 0.42, 0.28, 0.90],
                        };
                        // Miniature surface deck / cryobot headframe
                        quads.push(DepthQuad {
                            depth: u_depth - 0.008,
                            verts: make_billboard_quad(ux, uy, rig_w, rig_h * 0.65, col, 1.0, 0.9),
                        });
                        // When zoomed in close (Lander / Bot POV), render articulated legs, RSM mast, SEIS dome or Cryobot melt-bore!
                        if u_depth < 1.8 {
                            let mast_top = surf_pos.add(n_dir.scale(0.018));
                            if let Some(mt) = camera.project(mast_top, aspect) {
                                if let Some(mv) = make_line_ribbon((ux, uy), (mt.0, mt.1), (1.8 / height).max(rig_h * 0.14), aspect_f32, [0.90, 0.95, 1.0, 0.96]) {
                                    quads.push(DepthQuad { depth: u_depth - 0.009, verts: mv });
                                }
                            }
                            let tang = n_dir.cross(Vec3d::new(0.0, 1.0, 0.0)).normalized();
                            for sgn in [-1.0, 1.0] {
                                let foot = surf_pos.add(tang.scale(sgn * 0.018)).sub(n_dir.scale(0.005));
                                if let Some(fp) = camera.project(foot, aspect) {
                                    if let Some(lv) = make_line_ribbon((ux, uy), (fp.0, fp.1), (1.5 / height).max(rig_h * 0.11), aspect_f32, [0.55, 0.64, 0.76, 0.95]) {
                                        quads.push(DepthQuad { depth: u_depth - 0.007, verts: lv });
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if let Some((scan_id, pulse_age)) = active_scan_pulse {
                if scan_id == body.id {
                    quads.push(DepthQuad {
                        depth: depth - 0.03,
                        verts: make_billboard_quad(
                            nx,
                            ny,
                            (half_w * 2.8).min(1.4),
                            (half_h * 2.8).min(1.4),
                            [0.20, 0.96, 0.82, 0.90],
                            5.0,
                            pulse_age as f32,
                        ),
                    });
                }
            }

            let sx = rect.left() + (nx * 0.5 + 0.5) * width;
            let sy = rect.top() + (0.5 - ny * 0.5) * height;
            labels.push(ProjectedLabel {
                body_id: body.id.clone(),
                text: body.name.clone(),
                screen_pos: egui::pos2(sx, sy),
                radius_px: (half_h * height * 0.5).max(8.0),
                color: egui::Color32::from_rgb(
                    (body.color_rgb[0] * 255.0) as u8,
                    (body.color_rgb[1] * 255.0) as u8,
                    (body.color_rgb[2] * 255.0) as u8,
                ),
                depth,
            });
        }
    }

    // 4. Space probes, 1-leg or 2-leg Slingshot trajectories & 3D capture orbit rings
    for probe in probes {
        if probe.phase == ProbePhase::EnRoute {
            let steps = if probe.assist_body_id.is_some() { 36 } else { 24 };
            let ribbon_col = if probe.assist_body_id.is_some() {
                [1.0, 0.78, 0.24, 0.88]
            } else {
                [0.22, 0.98, 0.78, 0.85]
            };
            let mut prev_pt: Option<(f32, f32, f64)> = None;
            for step in 0..=steps {
                let mut sample = probe.clone();
                sample.transit_progress =
                    (step as f64 / steps as f64) * probe.transit_progress.max(0.04);
                let p_world = sample.scene_position(catalog, days_since_j2000);
                let proj = camera.project(p_world, aspect);
                if let (Some(a), Some(b)) = (prev_pt, proj) {
                    if let Some(verts) = make_line_ribbon(
                        (a.0, a.1),
                        (b.0, b.1),
                        3.4 / height,
                        aspect_f32,
                        ribbon_col,
                    ) {
                        quads.push(DepthQuad {
                            depth: 0.5 * (a.2 + b.2),
                            verts,
                        });
                    }
                }
                prev_pt = proj;
            }
        } else {
            let mut prev_pt: Option<(f32, f32, f64)> = None;
            for step in 0..=48 {
                let mut sample = probe.clone();
                sample.orbit_phase_rad = (step as f64 / 48.0) * TAU;
                let p_world = sample.scene_position(catalog, days_since_j2000);
                let proj = camera.project(p_world, aspect);
                if let (Some(a), Some(b)) = (prev_pt, proj) {
                    if let Some(verts) = make_line_ribbon(
                        (a.0, a.1),
                        (b.0, b.1),
                        2.4 / height,
                        aspect_f32,
                        [0.25, 0.74, 0.98, 0.58],
                    ) {
                        quads.push(DepthQuad {
                            depth: 0.5 * (a.2 + b.2),
                            verts,
                        });
                    }
                }
                prev_pt = proj;
            }
        }

        let probe_world = probe.scene_position(catalog, days_since_j2000);
        let mut ahead = probe.clone();
        if ahead.phase == ProbePhase::EnRoute {
            ahead.transit_progress = (ahead.transit_progress + 0.025).min(1.0);
        } else {
            ahead.orbit_phase_rad = (ahead.orbit_phase_rad + 0.08).rem_euclid(TAU);
        }
        let ahead_world = ahead.scene_position(catalog, days_since_j2000);
        let fwd = ahead_world.sub(probe_world).normalized();
        let right_axis = fwd.cross(Vec3d::new(0.0, 1.0, 0.0)).normalized();
        let up_axis = right_axis.cross(fwd).normalized();

        if let Some((nx, ny, depth)) = camera.project(probe_world, aspect) {
            if nx.abs() <= 1.25 && ny.abs() <= 1.25 {
                let proj_scale = ((1.0 / (camera.fov_y_rad * 0.5).tan()) / depth) as f32;
                // True miniature spacecraft hull scale (~12x smaller than planetary radii)
                let craft_sc = if probe.phase == ProbePhase::EnRoute { 0.032 } else { 0.018 };
                let bus_h = ((craft_sc as f32) * proj_scale).clamp(3.2 / height, 0.036);
                let bus_w = bus_h / aspect_f32;

                match probe.bus {
                    PropulsionBus::ChemicalCentaur | PropulsionBus::SolarElectricIon => {
                        let span = if probe.bus == PropulsionBus::SolarElectricIon {
                            craft_sc * 3.2
                        } else {
                            craft_sc * 2.1
                        };
                        let lw = camera.project(probe_world.sub(right_axis.scale(span)), aspect);
                        let rw = camera.project(probe_world.add(right_axis.scale(span)), aspect);
                        if let (Some(l), Some(r)) = (lw, rw) {
                            if let Some(wing_verts) = make_line_ribbon(
                                (l.0, l.1),
                                (r.0, r.1),
                                (bus_h * 0.42).max(2.0 / height),
                                aspect_f32,
                                [0.14, 0.52, 0.96, 0.95],
                            ) {
                                quads.push(DepthQuad {
                                    depth: depth - 0.004,
                                    verts: wing_verts,
                                });
                            }
                        }
                    }
                    PropulsionBus::RtgNuclearElectric => {
                        for sgn in [-1.0, 1.0] {
                            let fin_end = probe_world
                                .add(right_axis.scale(sgn * craft_sc * 1.5))
                                .sub(up_axis.scale(craft_sc * 0.65));
                            if let Some(fe) = camera.project(fin_end, aspect) {
                                if let Some(fin_verts) = make_line_ribbon(
                                    (nx, ny),
                                    (fe.0, fe.1),
                                    (bus_h * 0.26).max(1.8 / height),
                                    aspect_f32,
                                    [0.95, 0.42, 0.16, 0.95],
                                ) {
                                    quads.push(DepthQuad {
                                        depth: depth - 0.004,
                                        verts: fin_verts,
                                    });
                                }
                            }
                        }
                    }
                    PropulsionBus::LaserSailFusion => {
                        let sail_scale = if probe.phase == ProbePhase::EnRoute {
                            2.4
                        } else {
                            1.35
                        };
                        quads.push(DepthQuad {
                            depth: depth - 0.003,
                            verts: make_billboard_quad(
                                nx,
                                ny,
                                bus_w * sail_scale,
                                bus_h * sail_scale,
                                [0.72, 0.38, 0.98, 0.68],
                                3.0,
                                anim_time_sec as f32 * 2.0,
                            ),
                        });
                    }
                }

                let boom_tip = probe_world.add(right_axis.scale(craft_sc * 1.9)).add(up_axis.scale(craft_sc * 0.5));
                if let Some(bt) = camera.project(boom_tip, aspect) {
                    if let Some(b_verts) = make_line_ribbon(
                        (nx, ny),
                        (bt.0, bt.1),
                        (1.2 / height).max(bus_h * 0.11),
                        aspect_f32,
                        [0.75, 0.82, 0.90, 0.90],
                    ) {
                        quads.push(DepthQuad {
                            depth: depth - 0.0045,
                            verts: b_verts,
                        });
                    }
                }

                quads.push(DepthQuad {
                    depth: depth - 0.005,
                    verts: make_billboard_quad(
                        nx,
                        ny,
                        bus_w,
                        bus_h,
                        [0.96, 0.74, 0.22, 1.0],
                        1.0,
                        0.8,
                    ),
                });
                let dish_pos = probe_world.add(up_axis.scale(craft_sc * 0.55)).add(fwd.scale(craft_sc * 0.28));
                if let Some((dx, dy, d_depth)) = camera.project(dish_pos, aspect) {
                    quads.push(DepthQuad {
                        depth: d_depth - 0.0055,
                        verts: make_billboard_quad(
                            dx,
                            dy,
                            bus_w * 0.82,
                            bus_h * 0.52,
                            [0.95, 0.98, 1.0, 0.98],
                            1.0,
                            1.2,
                        ),
                    });
                }

                let plume_pos = probe_world.sub(fwd.scale(craft_sc * 1.1));
                if let Some((px, py, p_depth)) = camera.project(plume_pos, aspect) {
                    let pulse =
                        0.85 + 0.30 * ((anim_time_sec * 14.0 + probe.id as f64).sin() as f32);
                    let p_scale = if probe.phase == ProbePhase::EnRoute {
                        1.45 * pulse
                    } else {
                        0.55
                    };
                    let plume_col = match probe.bus {
                        PropulsionBus::ChemicalCentaur => [1.0, 0.76, 0.22, 0.92],
                        PropulsionBus::SolarElectricIon => [0.15, 0.90, 1.0, 0.95],
                        PropulsionBus::RtgNuclearElectric => [0.52, 0.58, 1.0, 0.92],
                        PropulsionBus::LaserSailFusion => [0.92, 0.45, 1.0, 0.95],
                    };
                    quads.push(DepthQuad {
                        depth: p_depth - 0.048,
                        verts: make_billboard_quad(
                            px,
                            py,
                            bus_w * p_scale,
                            bus_h * p_scale,
                            plume_col,
                            0.0,
                            anim_time_sec as f32 * 4.0,
                        ),
                    });
                }

                let sx = rect.left() + (nx * 0.5 + 0.5) * width;
                let sy = rect.top() + (0.5 - ny * 0.5) * height;
                labels.push(ProjectedLabel {
                    body_id: probe.target_id.clone(),
                    text: format!("Probe {}", probe.callsign),
                    screen_pos: egui::pos2(sx, sy - (bus_h * height * 0.6) - 8.0),
                    radius_px: 7.0,
                    color: egui::Color32::from_rgb(90, 245, 210),
                    depth,
                });
            }
        }
    }

    // 5. Active Atmospheric Entry / Kinetic Crash Probes
    for ap in atmospheric_probes {
        if ap.phase != EntryProbePhase::TerminalCrushOrImpact {
            let mut prev_pt: Option<(f32, f32, f64)> = None;
            for step in 0..=16 {
                let mut sample = ap.clone();
                sample.progress = (step as f64 / 16.0) * ap.progress.max(0.05);
                let w = sample.scene_position(catalog, days_since_j2000);
                let proj = camera.project(w, aspect);
                if let (Some(a), Some(b)) = (prev_pt, proj) {
                    if let Some(verts) = make_line_ribbon(
                        (a.0, a.1),
                        (b.0, b.1),
                        3.2 / height,
                        aspect_f32,
                        [1.0, 0.38, 0.18, 0.92],
                    ) {
                        quads.push(DepthQuad {
                            depth: 0.5 * (a.2 + b.2),
                            verts,
                        });
                    }
                }
                prev_pt = proj;
            }
            let w_cur = ap.scene_position(catalog, days_since_j2000);
            if let Some((ax, ay, a_depth)) = camera.project(w_cur, aspect) {
                let sz = (10.5 / height).clamp(0.01, 0.055);
                let glow_col = if ap.phase == EntryProbePhase::HypersonicEntry {
                    [1.0, 0.45, 0.15, 1.0]
                } else {
                    [0.95, 0.88, 0.65, 0.96]
                };
                quads.push(DepthQuad {
                    depth: a_depth - 0.05,
                    verts: make_billboard_quad(
                        ax,
                        ay,
                        sz / aspect_f32,
                        sz,
                        glow_col,
                        0.0,
                        anim_time_sec as f32 * 8.0,
                    ),
                });
                let sx = rect.left() + (ax * 0.5 + 0.5) * width;
                let sy = rect.top() + (0.5 - ay * 0.5) * height;
                labels.push(ProjectedLabel {
                    body_id: ap.body_id.clone(),
                    text: format!(
                        "Probe AEP-{} ({:.1} bar)",
                        ap.id, ap.current_pressure_bar
                    ),
                    screen_pos: egui::pos2(sx, sy - 10.0),
                    radius_px: 6.0,
                    color: egui::Color32::from_rgb(255, 145, 95),
                    depth: a_depth,
                });
            }
        }
    }

    // 6. Returning Earth Sample-Return Capsules (SRC)
    for cap in sample_capsules {
        if cap.phase == CapsulePhase::ReturningToEarth {
            let mut prev_pt: Option<(f32, f32, f64)> = None;
            for step in 0..=20 {
                let mut sample = cap.clone();
                sample.progress = (step as f64 / 20.0) * cap.progress.max(0.05);
                let w = sample.scene_position(catalog, days_since_j2000);
                let proj = camera.project(w, aspect);
                if let (Some(a), Some(b)) = (prev_pt, proj) {
                    if let Some(verts) = make_line_ribbon(
                        (a.0, a.1),
                        (b.0, b.1),
                        2.8 / height,
                        aspect_f32,
                        [0.98, 0.62, 0.22, 0.86],
                    ) {
                        quads.push(DepthQuad {
                            depth: 0.5 * (a.2 + b.2),
                            verts,
                        });
                    }
                }
                prev_pt = proj;
            }

            let c_world = cap.scene_position(catalog, days_since_j2000);
            if let Some((cx, cy, c_depth)) = camera.project(c_world, aspect) {
                let sz = (8.0 / height).clamp(0.008, 0.04);
                quads.push(DepthQuad {
                    depth: c_depth - 0.05,
                    verts: make_billboard_quad(
                        cx,
                        cy,
                        sz / aspect_f32,
                        sz,
                        [1.0, 0.84, 0.32, 1.0],
                        0.0,
                        anim_time_sec as f32 * 5.0,
                    ),
                });
                let sx = rect.left() + (cx * 0.5 + 0.5) * width;
                let sy = rect.top() + (0.5 - cy * 0.5) * height;
                labels.push(ProjectedLabel {
                    body_id: "earth".into(),
                    text: format!("Probe SRC-{} -> Earth", cap.id),
                    screen_pos: egui::pos2(sx, sy - 10.0),
                    radius_px: 6.0,
                    color: egui::Color32::from_rgb(255, 205, 95),
                    depth: c_depth,
                });
            }
        }
    }

    quads.sort_by(|a, b| b.depth.total_cmp(&a.depth));

    let mut vertex_bytes = Vec::with_capacity(quads.len() * 6 * 40);
    for q in &quads {
        for v in q.verts {
            v.write_le_bytes(&mut vertex_bytes);
        }
    }

    SceneFrameBuild {
        vertex_bytes,
        vertex_count: (quads.len() * 6) as u32,
        labels,
    }
}

pub struct CosmosGpuPipeline {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    buffer_capacity: u64,
    active_vertex_count: u32,
}

impl CosmosGpuPipeline {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("helioprobe_cosmos_wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/cosmos.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("helioprobe_pipeline_layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let attributes = [
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 8,
                shader_location: 1,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 16,
                shader_location: 2,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 32,
                shader_location: 3,
            },
        ];

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("helioprobe_cosmos_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: GpuVertex::BYTE_STRIDE,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &attributes,
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let initial_cap = 512 * 1024;
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("helioprobe_cosmos_vbo"),
            size: initial_cap,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            vertex_buffer,
            buffer_capacity: initial_cap,
            active_vertex_count: 0,
        }
    }

    pub fn upload_vertices(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: &[u8],
        vertex_count: u32,
    ) {
        if bytes.is_empty() || vertex_count == 0 {
            self.active_vertex_count = 0;
            return;
        }
        let needed = bytes.len() as u64;
        if needed > self.buffer_capacity {
            let next_cap = needed.next_power_of_two().max(512 * 1024);
            self.vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("helioprobe_cosmos_vbo_resized"),
                size: next_cap,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.buffer_capacity = next_cap;
        }
        queue.write_buffer(&self.vertex_buffer, 0, bytes);
        self.active_vertex_count = vertex_count;
    }

    pub fn paint(&self, render_pass: &mut wgpu::RenderPass<'static>) {
        if self.active_vertex_count == 0 {
            return;
        }
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(0..self.active_vertex_count, 0..1);
    }
}

pub struct CosmosPaintCallback {
    pub vertex_bytes: Vec<u8>,
    pub vertex_count: u32,
}

impl egui_wgpu::CallbackTrait for CosmosPaintCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(gpu) = resources.get_mut::<CosmosGpuPipeline>() {
            gpu.upload_vertices(device, queue, &self.vertex_bytes, self.vertex_count);
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        if let Some(gpu) = resources.get::<CosmosGpuPipeline>() {
            gpu.paint(render_pass);
        }
    }
}
