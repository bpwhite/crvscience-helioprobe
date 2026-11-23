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

//! Safe-Rust (`#![forbid(unsafe_code)]`) 3D perspective camera, depth-sorted geometry builder,
//! 2-leg Slingshot trajectory ribbons, Surface Lander/Cryobot beacons, Galileo Atmospheric Entry Plunges,
//! Earth Sample-Return Capsules, ray-picking hit tester, and `eframe::egui_wgpu` hardware WGSL render pipeline.

use crate::catalog::{BodyCategory, CelestialBody};
use crate::l2_comets::{
    compute_comet_tail_state, sample_dust_tail_point, sun_earth_l2_halo_position, L2_OBSERVATORIES,
};
use crate::physics::{
    body_scene_position_with_offsets, compress_au_to_scene,
    kepler_position_from_eccentric_anomaly_au, visual_body_radius_scene, Vec3d,
};
use crate::probes::{ProbePhase, PropulsionBus, SpaceProbe};
use crate::surface_ops::{
    AtmosphericProbeRun, CapsulePhase, EntryProbePhase, SampleReturnCapsule, SurfaceDeployment,
    SurfaceUnitKind, ICY_CRYOBOT_WORLDS,
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
    pub up_override: Option<Vec3d>,
}

impl Default for OrbitCamera3D {
    fn default() -> Self {
        Self {
            target: Vec3d::default(),
            yaw_rad: 0.65,
            pitch_rad: 0.46,
            distance: 78.0,
            fov_y_rad: 50.0_f64.to_radians(),
            up_override: None,
        }
    }
}

impl OrbitCamera3D {
    pub fn local_basis(&self) -> (Vec3d, Vec3d, Vec3d) {
        if let Some(u_raw) = self.up_override {
            let up = u_raw.normalized();
            let ref_up = if up.y.abs() > 0.92 {
                Vec3d::new(0.0, 0.0, 1.0)
            } else {
                Vec3d::new(0.0, 1.0, 0.0)
            };
            let east = ref_up.cross(up).normalized();
            let north = up.cross(east).normalized();
            (east, up, north)
        } else {
            (
                Vec3d::new(1.0, 0.0, 0.0),
                Vec3d::new(0.0, 1.0, 0.0),
                Vec3d::new(0.0, 0.0, 1.0),
            )
        }
    }

    pub fn eye_position(&self) -> Vec3d {
        let (east, up, north) = self.local_basis();
        let cp = self.pitch_rad.cos();
        let offset = east
            .scale(self.distance * cp * self.yaw_rad.sin())
            .add(up.scale(self.distance * self.pitch_rad.sin()))
            .add(north.scale(self.distance * cp * self.yaw_rad.cos()));
        self.target.add(offset)
    }

    /// Returns the orthonormal camera basis `(right, up, forward)` in world coordinates.
    pub fn view_basis(&self) -> (Vec3d, Vec3d, Vec3d) {
        let eye = self.eye_position();
        let forward = self.target.sub(eye).normalized();
        let world_up = self
            .up_override
            .unwrap_or(Vec3d::new(0.0, 1.0, 0.0))
            .normalized();
        let mut right = forward.cross(world_up);
        if right.length() < 1e-6 {
            right = forward.cross(Vec3d::new(0.0, 0.0, 1.0));
        }
        let right = right.normalized();
        let up = right.cross(forward).normalized();
        (right, up, forward)
    }

    pub fn project(&self, world: Vec3d, aspect: f64) -> Option<(f32, f32, f64)> {
        let eye = self.eye_position();
        let (right, up, forward) = self.view_basis();

        let rel = world.sub(eye);
        let z_cam = rel.dot(forward);
        if z_cam <= 0.0015 {
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

/// Transforms the true world-space body-to-Sun vector into camera view space `(lx, ly, lz)`
/// matching the billboard sphere normal basis `(right, up, -forward)`.
pub fn body_sun_light_view_space(camera: &OrbitCamera3D, body_world: Vec3d) -> [f32; 3] {
    let to_sun = if body_world.length() > 1e-6 {
        body_world.scale(-1.0).normalized()
    } else {
        Vec3d::new(0.0, 0.0, 1.0)
    };
    let (right, up, forward) = camera.view_basis();
    let lx = to_sun.dot(right) as f32;
    let ly = to_sun.dot(up) as f32;
    let lz = to_sun.dot(forward.scale(-1.0)) as f32;
    let len = (lx * lx + ly * ly + lz * lz).sqrt().max(1e-6);
    [lx / len, ly / len, lz / len]
}

#[derive(Debug, Clone, Copy)]
pub struct PlanetaryRingSpec {
    pub obliquity_deg: f64,
    pub pole_az_deg: f64,
    pub ring_mul: f64,
    pub r_inner_norm: f64,
    pub ring_alpha: f32,
    pub color_rgb: [f32; 3],
    pub cassini_strength: f32,
}

/// Returns physical obliquity, ring extent, inner/outer radii, and albedo tint for ringed Solar System bodies.
pub fn planetary_ring_spec(body_id: &str) -> PlanetaryRingSpec {
    match body_id {
        "saturn" => PlanetaryRingSpec {
            obliquity_deg: 26.73,
            pole_az_deg: 79.5,
            ring_mul: 2.28,
            r_inner_norm: 0.52,
            ring_alpha: 0.86,
            color_rgb: [0.92, 0.86, 0.70],
            cassini_strength: 1.0,
        },
        "uranus" => PlanetaryRingSpec {
            obliquity_deg: 97.77,
            pole_az_deg: 77.3,
            ring_mul: 1.72,
            r_inner_norm: 0.65,
            ring_alpha: 0.44,
            color_rgb: [0.64, 0.80, 0.86],
            cassini_strength: 0.45,
        },
        "neptune" => PlanetaryRingSpec {
            obliquity_deg: 28.32,
            pole_az_deg: 40.6,
            ring_mul: 1.62,
            r_inner_norm: 0.67,
            ring_alpha: 0.34,
            color_rgb: [0.56, 0.68, 0.88],
            cassini_strength: 0.35,
        },
        "jupiter" => PlanetaryRingSpec {
            obliquity_deg: 3.13,
            pole_az_deg: 248.0,
            ring_mul: 1.50,
            r_inner_norm: 0.70,
            ring_alpha: 0.28,
            color_rgb: [0.78, 0.68, 0.56],
            cassini_strength: 0.20,
        },
        "haumea" => PlanetaryRingSpec {
            obliquity_deg: 28.20,
            pole_az_deg: 122.1,
            ring_mul: 1.85,
            r_inner_norm: 0.62,
            ring_alpha: 0.52,
            color_rgb: [0.85, 0.90, 0.95],
            cassini_strength: 0.30,
        },
        _ => PlanetaryRingSpec {
            obliquity_deg: 23.0,
            pole_az_deg: 60.0,
            ring_mul: 1.55,
            r_inner_norm: 0.64,
            ring_alpha: 0.34,
            color_rgb: [0.88, 0.84, 0.74],
            cassini_strength: 0.30,
        },
    }
}

/// Returns orthonormal 3D ring-plane basis `(ring_u, ring_v, ring_normal)` tilted by planetary obliquity.
pub fn planetary_ring_basis(body_id: &str) -> (Vec3d, Vec3d, Vec3d) {
    let spec = planetary_ring_spec(body_id);
    let obliq = spec.obliquity_deg.to_radians();
    let az = spec.pole_az_deg.to_radians();
    let ring_normal = Vec3d::new(
        obliq.sin() * az.cos(),
        obliq.cos(),
        obliq.sin() * az.sin(),
    )
    .normalized();
    let ref_vec = if ring_normal.y.abs() > 0.92 {
        Vec3d::new(0.0, 0.0, 1.0)
    } else {
        Vec3d::new(0.0, 1.0, 0.0)
    };
    let ring_u = ref_vec.cross(ring_normal).normalized();
    let ring_v = ring_normal.cross(ring_u).normalized();
    (ring_u, ring_v, ring_normal)
}

/// Samples a closed Keplerian orbit uniformly in eccentric anomaly `E` in `[0, TAU]`,
/// preserving dense curvature around perihelion for high-eccentricity comets.
pub fn sample_orbit_ribbon_points_au(body: &CelestialBody, steps: usize) -> Vec<Vec3d> {
    let n = steps.max(12);
    (0..=n)
        .map(|s| {
            let ecc_anom = (s as f64 / n as f64) * TAU;
            kepler_position_from_eccentric_anomaly_au(
                body.semi_major_axis_au,
                body.eccentricity,
                body.inclination_deg,
                body.lon_ascending_node_deg,
                body.arg_periapsis_deg,
                ecc_anom,
            )
        })
        .collect()
}

#[derive(Clone, Copy)]
pub(crate) struct GpuVertex {
    clip_pos: [f32; 2],
    uv: [f32; 2],
    color: [f32; 4],
    kind_param: [f32; 2],
    extra: [f32; 4],
}

impl GpuVertex {
    const BYTE_STRIDE: u64 = 56;

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
            self.extra[0],
            self.extra[1],
            self.extra[2],
            self.extra[3],
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
}

pub(crate) enum DepthGeom {
    Quad { depth: f64, verts: [GpuVertex; 6] },
    Tri { depth: f64, verts: [GpuVertex; 3] },
}

impl DepthGeom {
    fn depth(&self) -> f64 {
        match self {
            Self::Quad { depth, .. } | Self::Tri { depth, .. } => *depth,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SurfaceAnchor {
    pub normal: Vec3d,
    pub east: Vec3d,
    pub north: Vec3d,
    pub surface_pt: Vec3d,
    pub unit_pos: Vec3d,
    pub rig_scale: f64,
}

/// Shared deterministic surface anchor used identically by `LanderBotView` camera tracking
/// and 3D lander/cryobot geometry placement so units stay firmly planted on the body crust.
pub fn unit_world_pos(
    body: &CelestialBody,
    body_world: Vec3d,
    u_idx: usize,
    anim_time_sec: f64,
    descent_progress: f64,
) -> SurfaceAnchor {
    let r_vis = visual_body_radius_scene(body);
    let sun_az = if body_world.length() > 1e-6 {
        (-body_world.z).atan2(-body_world.x)
    } else {
        0.0
    };
    let ang = sun_az + 0.35 + (u_idx as f64) * 1.9 + anim_time_sec * 0.02;
    let normal = Vec3d::new(0.84 * ang.cos(), 0.54, 0.84 * ang.sin()).normalized();
    let mut east = Vec3d::new(0.0, 1.0, 0.0).cross(normal);
    if east.length() < 1e-6 {
        east = Vec3d::new(1.0, 0.0, 0.0);
    }
    let east = east.normalized();
    let north = normal.cross(east).normalized();
    let surface_pt = body_world.add(normal.scale(r_vis));
    let alt_frac = (1.0 - descent_progress.clamp(0.0, 1.0)).powi(2);
    let unit_pos = surface_pt.add(normal.scale(r_vis * 0.42 * alt_frac));
    let rig_scale = (r_vis * 0.058).clamp(0.008, 0.036);
    SurfaceAnchor {
        normal,
        east,
        north,
        surface_pt,
        unit_pos,
        rig_scale,
    }
}

/// Immediate-mode 3D polyhedral mesh builder emitting Lambert + Blinn-Phong solar-lit
/// triangles (`kind == 6`) sorted back-to-front per model.
pub struct MeshBuilder3D<'a> {
    camera: &'a OrbitCamera3D,
    aspect: f64,
    light_dir: Vec3d,
    eye_pos: Vec3d,
    tris: Vec<DepthGeom>,
}

impl<'a> MeshBuilder3D<'a> {
    pub fn new(camera: &'a OrbitCamera3D, aspect: f64, light_dir: Vec3d) -> Self {
        let ld = if light_dir.length() > 1e-6 {
            light_dir.normalized()
        } else {
            Vec3d::new(0.4, 0.8, 0.4).normalized()
        };
        Self {
            camera,
            aspect,
            light_dir: ld,
            eye_pos: camera.eye_position(),
            tris: Vec::with_capacity(96),
        }
    }

    pub fn tri_count(&self) -> usize {
        self.tris.len()
    }

    fn shade_triangle(
        &self,
        center: Vec3d,
        raw_normal: Vec3d,
        base_rgb: [f32; 3],
        shininess: f32,
        spec_strength: f32,
        emissive: f32,
        double_sided: bool,
    ) -> Option<([f32; 3], f32)> {
        let view_dir = self.eye_pos.sub(center).normalized();
        let mut n = raw_normal.normalized();
        let facing = n.dot(view_dir);
        if !double_sided && facing <= -0.02 {
            return None;
        }
        if double_sided && facing < 0.0 {
            n = n.scale(-1.0);
        }
        let ndotl = n.dot(self.light_dir).max(0.0) as f32;
        let sky_fill = ((n.dot(Vec3d::new(0.0, 1.0, 0.0)) * 0.5 + 0.5) as f32) * 0.12;
        let half_vec = self.light_dir.add(view_dir).normalized();
        let spec = if spec_strength > 0.0 {
            (n.dot(half_vec).max(0.0) as f32).powf(shininess) * spec_strength
        } else {
            0.0
        };
        let shade = 0.22 + sky_fill + 0.86 * ndotl + emissive;
        let rgb = [
            (base_rgb[0] * shade).clamp(0.0, 1.0),
            (base_rgb[1] * shade).clamp(0.0, 1.0),
            (base_rgb[2] * shade).clamp(0.0, 1.0),
        ];
        Some((rgb, (spec + emissive * 0.35).clamp(0.0, 1.0)))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_tri_uv(
        &mut self,
        p0: Vec3d,
        p1: Vec3d,
        p2: Vec3d,
        uvs: [[f32; 2]; 3],
        base_rgb: [f32; 3],
        alpha: f32,
        shininess: f32,
        spec_strength: f32,
        emissive: f32,
        double_sided: bool,
        depth_bias: f64,
    ) {
        let e1 = p1.sub(p0);
        let e2 = p2.sub(p0);
        let raw_n = e1.cross(e2);
        if raw_n.length() < 1e-12 {
            return;
        }
        let center = p0.add(p1).add(p2).scale(1.0 / 3.0);
        let Some((lit_rgb, sheen)) = self.shade_triangle(
            center,
            raw_n,
            base_rgb,
            shininess,
            spec_strength,
            emissive,
            double_sided,
        ) else {
            return;
        };
        let (Some(q0), Some(q1), Some(q2)) = (
            self.camera.project(p0, self.aspect),
            self.camera.project(p1, self.aspect),
            self.camera.project(p2, self.aspect),
        ) else {
            return;
        };
        if q0.0.abs() > 4.0 && q1.0.abs() > 4.0 && q2.0.abs() > 4.0 {
            return;
        }
        if q0.1.abs() > 4.0 && q1.1.abs() > 4.0 && q2.1.abs() > 4.0 {
            return;
        }
        let depth = (q0.2 + q1.2 + q2.2) / 3.0 + depth_bias;
        let col = [lit_rgb[0], lit_rgb[1], lit_rgb[2], alpha];
        self.tris.push(DepthGeom::Tri {
            depth,
            verts: [
                GpuVertex {
                    clip_pos: [q0.0, q0.1],
                    uv: uvs[0],
                    color: col,
                    kind_param: [6.0, sheen],
                    extra: [0.0; 4],
                },
                GpuVertex {
                    clip_pos: [q1.0, q1.1],
                    uv: uvs[1],
                    color: col,
                    kind_param: [6.0, sheen],
                    extra: [0.0; 4],
                },
                GpuVertex {
                    clip_pos: [q2.0, q2.1],
                    uv: uvs[2],
                    color: col,
                    kind_param: [6.0, sheen],
                    extra: [0.0; 4],
                },
            ],
        });
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_quad(
        &mut self,
        p0: Vec3d,
        p1: Vec3d,
        p2: Vec3d,
        p3: Vec3d,
        base_rgb: [f32; 3],
        alpha: f32,
        shininess: f32,
        spec_strength: f32,
        emissive: f32,
        double_sided: bool,
        depth_bias: f64,
    ) {
        self.add_tri_uv(
            p0,
            p1,
            p2,
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            base_rgb,
            alpha,
            shininess,
            spec_strength,
            emissive,
            double_sided,
            depth_bias,
        );
        self.add_tri_uv(
            p0,
            p2,
            p3,
            [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            base_rgb,
            alpha,
            shininess,
            spec_strength,
            emissive,
            double_sided,
            depth_bias,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_prism(
        &mut self,
        center: Vec3d,
        axis_dir: Vec3d,
        ref_dir: Vec3d,
        sides: usize,
        r_bot: f64,
        r_top: f64,
        half_len: f64,
        base_rgb: [f32; 3],
        alpha: f32,
        shininess: f32,
        spec_strength: f32,
        emissive: f32,
        cap_bot: bool,
        cap_top: bool,
    ) {
        let axis = axis_dir.normalized();
        let mut u = ref_dir.sub(axis.scale(ref_dir.dot(axis)));
        if u.length() < 1e-6 {
            let fallback = if axis.y.abs() < 0.9 {
                Vec3d::new(0.0, 1.0, 0.0)
            } else {
                Vec3d::new(1.0, 0.0, 0.0)
            };
            u = fallback.cross(axis);
        }
        u = u.normalized();
        let v = axis.cross(u).normalized();
        let n = sides.max(3);
        let double_sided = alpha < 0.99 || !cap_bot || !cap_top;
        let c_bot = center.sub(axis.scale(half_len));
        let c_top = center.add(axis.scale(half_len));

        for i in 0..n {
            let a0 = (i as f64 / n as f64) * TAU;
            let a1 = ((i + 1) as f64 / n as f64) * TAU;
            let dir0 = u.scale(a0.cos()).add(v.scale(a0.sin()));
            let dir1 = u.scale(a1.cos()).add(v.scale(a1.sin()));
            let b0 = c_bot.add(dir0.scale(r_bot));
            let b1 = c_bot.add(dir1.scale(r_bot));
            let t1 = c_top.add(dir1.scale(r_top));
            let t0 = c_top.add(dir0.scale(r_top));

            // Cross(b1-b0, t0-b0) points outward when (u, v, axis) is right-handed
            self.add_quad(
                b0,
                b1,
                t1,
                t0,
                base_rgb,
                alpha,
                shininess,
                spec_strength,
                emissive,
                double_sided,
                0.0,
            );
            if cap_top && r_top > 1e-7 {
                self.add_tri_uv(
                    c_top,
                    t0,
                    t1,
                    [[0.5, 0.5], [1.0, 0.5], [0.5, 1.0]],
                    base_rgb,
                    alpha,
                    shininess,
                    spec_strength,
                    emissive,
                    double_sided,
                    0.0,
                );
            }
            if cap_bot && r_bot > 1e-7 {
                self.add_tri_uv(
                    c_bot,
                    b1,
                    b0,
                    [[0.5, 0.5], [1.0, 0.5], [0.5, 1.0]],
                    base_rgb,
                    alpha,
                    shininess,
                    spec_strength,
                    emissive,
                    double_sided,
                    0.0,
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_box(
        &mut self,
        center: Vec3d,
        right: Vec3d,
        up: Vec3d,
        fwd: Vec3d,
        hx: f64,
        hy: f64,
        hz: f64,
        base_rgb: [f32; 3],
        alpha: f32,
        shininess: f32,
        spec_strength: f32,
        emissive: f32,
    ) {
        let rx = right.normalized().scale(hx);
        let uy = up.normalized().scale(hy);
        let fz = fwd.normalized().scale(hz);

        let p = |sx: f64, sy: f64, sz: f64| -> Vec3d {
            center.add(rx.scale(sx)).add(uy.scale(sy)).add(fz.scale(sz))
        };
        let ds = alpha < 0.99;
        // +Z, -Z, +X, -X, +Y, -Y faces (CCW outward)
        self.add_quad(p(-1.0, -1.0, 1.0), p(1.0, -1.0, 1.0), p(1.0, 1.0, 1.0), p(-1.0, 1.0, 1.0), base_rgb, alpha, shininess, spec_strength, emissive, ds, 0.0);
        self.add_quad(p(1.0, -1.0, -1.0), p(-1.0, -1.0, -1.0), p(-1.0, 1.0, -1.0), p(1.0, 1.0, -1.0), base_rgb, alpha, shininess, spec_strength, emissive, ds, 0.0);
        self.add_quad(p(1.0, -1.0, 1.0), p(1.0, -1.0, -1.0), p(1.0, 1.0, -1.0), p(1.0, 1.0, 1.0), base_rgb, alpha, shininess, spec_strength, emissive, ds, 0.0);
        self.add_quad(p(-1.0, -1.0, -1.0), p(-1.0, -1.0, 1.0), p(-1.0, 1.0, 1.0), p(-1.0, 1.0, -1.0), base_rgb, alpha, shininess, spec_strength, emissive, ds, 0.0);
        self.add_quad(p(-1.0, 1.0, 1.0), p(1.0, 1.0, 1.0), p(1.0, 1.0, -1.0), p(-1.0, 1.0, -1.0), base_rgb, alpha, shininess, spec_strength, emissive, ds, 0.0);
        self.add_quad(p(-1.0, -1.0, -1.0), p(1.0, -1.0, -1.0), p(1.0, -1.0, 1.0), p(-1.0, -1.0, 1.0), base_rgb, alpha, shininess, spec_strength, emissive, ds, 0.0);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_parabolic_dish(
        &mut self,
        apex: Vec3d,
        axis_dir: Vec3d,
        ref_dir: Vec3d,
        sectors: usize,
        radius: f64,
        dish_depth: f64,
        inner_rgb: [f32; 3],
        outer_rgb: [f32; 3],
        shininess: f32,
    ) {
        let axis = axis_dir.normalized();
        let mut u = ref_dir.sub(axis.scale(ref_dir.dot(axis)));
        if u.length() < 1e-6 {
            u = Vec3d::new(1.0, 0.0, 0.0).cross(axis);
        }
        u = u.normalized();
        let v = axis.cross(u).normalized();
        let n = sectors.max(6);

        let ring_pt = |r_frac: f64, ang: f64| -> Vec3d {
            let rad = radius * r_frac;
            let z = dish_depth * r_frac * r_frac;
            apex.add(axis.scale(z))
                .add(u.scale(rad * ang.cos()))
                .add(v.scale(rad * ang.sin()))
        };

        for i in 0..n {
            let a0 = (i as f64 / n as f64) * TAU;
            let a1 = ((i + 1) as f64 / n as f64) * TAU;
            let m0 = ring_pt(0.52, a0);
            let m1 = ring_pt(0.52, a1);
            let o0 = ring_pt(1.0, a0);
            let o1 = ring_pt(1.0, a1);

            // Inner concave reflector & outer parabolic shell (double-sided)
            self.add_tri_uv(
                apex,
                m0,
                m1,
                [[0.5, 0.5], [0.2, 0.8], [0.8, 0.8]],
                inner_rgb,
                1.0,
                shininess,
                0.55,
                0.03,
                true,
                0.0,
            );
            self.add_quad(
                m0,
                o0,
                o1,
                m1,
                inner_rgb,
                1.0,
                shininess,
                0.52,
                0.02,
                true,
                0.0,
            );
            // Outer rim backing subtle dark ring
            self.add_quad(
                o0,
                m0,
                m1,
                o1,
                outer_rgb,
                1.0,
                18.0,
                0.20,
                0.0,
                false,
                0.0002,
            );
        }

        // Tripod feed struts + central feed horn
        let feed_pos = apex.add(axis.scale(dish_depth + radius * 0.62));
        for k in 0..3 {
            let ang = (k as f64 / 3.0) * TAU + 0.35;
            let base = ring_pt(0.76, ang);
            let strut_dir = feed_pos.sub(base);
            let mid = base.lerp(feed_pos, 0.5);
            self.add_prism(
                mid,
                strut_dir,
                axis,
                4,
                radius * 0.035,
                radius * 0.028,
                strut_dir.length() * 0.5,
                [0.72, 0.78, 0.85],
                1.0,
                20.0,
                0.3,
                0.02,
                true,
                true,
            );
        }
        self.add_prism(
            feed_pos,
            axis,
            u,
            6,
            radius * 0.13,
            radius * 0.09,
            radius * 0.10,
            [0.95, 0.76, 0.25],
            1.0,
            30.0,
            0.45,
            0.05,
            true,
            true,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_solar_wing(
        &mut self,
        root: Vec3d,
        span_dir: Vec3d,
        chord_dir: Vec3d,
        normal_dir: Vec3d,
        num_panels: usize,
        panel_len: f64,
        panel_half_w: f64,
        gap: f64,
        cell_rgb: [f32; 3],
        frame_rgb: [f32; 3],
    ) {
        let span = span_dir.normalized();
        let chord = chord_dir.normalized();
        let norm = normal_dir.normalized();
        let total_span = gap + (num_panels as f64) * (panel_len + gap);
        let boom_mid = root.add(span.scale(total_span * 0.5));
        self.add_prism(
            boom_mid,
            span,
            norm,
            4,
            panel_half_w * 0.10,
            panel_half_w * 0.08,
            total_span * 0.5,
            frame_rgb,
            1.0,
            24.0,
            0.30,
            0.02,
            true,
            true,
        );
        for idx in 0..num_panels {
            let dist = gap + (idx as f64) * (panel_len + gap) + panel_len * 0.5;
            let center = root.add(span.scale(dist));
            self.add_box(
                center,
                span,
                norm,
                chord,
                panel_len * 0.48,
                panel_half_w * 0.06,
                panel_half_w,
                cell_rgb,
                1.0,
                34.0,
                0.48,
                0.04,
            );
        }
    }

    pub(crate) fn flush_into(mut self, out: &mut Vec<DepthGeom>) {
        self.tris.sort_by(|a, b| b.depth().total_cmp(&a.depth()));
        out.append(&mut self.tris);
    }
}

fn build_l2_observatory_mesh(
    obs_id: &str,
    opos: Vec3d,
    earth_scene_now: Vec3d,
    camera: &OrbitCamera3D,
    aspect: f64,
    sc: f64,
    out: &mut Vec<DepthGeom>,
) {
    let sun_dir = if opos.length() > 1e-6 {
        opos.scale(-1.0).normalized()
    } else {
        Vec3d::new(-1.0, 0.0, 0.0)
    };
    let boresight = opos.sub(earth_scene_now).normalized();
    let mut right = boresight.cross(Vec3d::new(0.0, 1.0, 0.0));
    if right.length() < 1e-6 {
        right = Vec3d::new(0.0, 0.0, 1.0);
    }
    right = right.normalized();
    let up = right.cross(boresight).normalized();

    let mut mb = MeshBuilder3D::new(camera, aspect, sun_dir);
    match obs_id {
        "jwst_l2" => {
            // 5-layer diamond Kapton sunshield oriented toward Sun
            let layers = [0.0, -0.09, -0.18];
            for (li, &off) in layers.iter().enumerate() {
                let c = opos.add(sun_dir.scale(off * sc));
                let span_r = sc * (1.75 - li as f64 * 0.07);
                let span_f = sc * (1.25 - li as f64 * 0.05);
                let p0 = c.add(boresight.scale(span_f));
                let p1 = c.add(right.scale(span_r));
                let p2 = c.sub(boresight.scale(span_f * 0.85));
                let p3 = c.sub(right.scale(span_r));
                let shield_col = if li == 0 {
                    [0.86, 0.72, 0.94]
                } else {
                    [0.72, 0.78, 0.88]
                };
                mb.add_quad(p0, p1, p2, p3, shield_col, 0.96, 28.0, 0.45, 0.04, true, 0.0);
            }
            // Gold hexagonal primary mirror assembly above sunshield
            let mirror_c = opos.add(up.scale(sc * 0.55));
            mb.add_prism(
                mirror_c,
                boresight,
                up,
                6,
                sc * 0.74,
                sc * 0.74,
                sc * 0.08,
                [0.98, 0.78, 0.18],
                1.0,
                42.0,
                0.65,
                0.08,
                true,
                true,
            );
            // Secondary mirror + 3 tripod struts
            let sec_pos = mirror_c.add(boresight.scale(sc * 0.95));
            for k in 0..3 {
                let ang = (k as f64 / 3.0) * TAU + 0.4;
                let base = mirror_c
                    .add(right.scale(sc * 0.65 * ang.cos()))
                    .add(up.scale(sc * 0.65 * ang.sin()));
                let dir = sec_pos.sub(base);
                mb.add_prism(
                    base.lerp(sec_pos, 0.5),
                    dir,
                    boresight,
                    4,
                    sc * 0.03,
                    sc * 0.025,
                    dir.length() * 0.5,
                    [0.24, 0.28, 0.34],
                    1.0,
                    18.0,
                    0.2,
                    0.02,
                    true,
                    true,
                );
            }
            mb.add_prism(
                sec_pos,
                boresight,
                up,
                6,
                sc * 0.16,
                sc * 0.12,
                sc * 0.06,
                [0.95, 0.80, 0.26],
                1.0,
                36.0,
                0.5,
                0.06,
                true,
                true,
            );
        }
        "roman_l2" => {
            mb.add_prism(
                opos,
                boresight,
                up,
                10,
                sc * 0.46,
                sc * 0.48,
                sc * 0.85,
                [0.82, 0.88, 0.96],
                1.0,
                26.0,
                0.42,
                0.05,
                true,
                true,
            );
            mb.add_solar_wing(
                opos.add(right.scale(sc * 0.45)),
                right,
                boresight,
                up,
                2,
                sc * 0.55,
                sc * 0.28,
                sc * 0.08,
                [0.12, 0.42, 0.92],
                [0.72, 0.78, 0.85],
            );
            mb.add_solar_wing(
                opos.sub(right.scale(sc * 0.45)),
                right.scale(-1.0),
                boresight,
                up,
                2,
                sc * 0.55,
                sc * 0.28,
                sc * 0.08,
                [0.12, 0.42, 0.92],
                [0.72, 0.78, 0.85],
            );
        }
        _ => {
            mb.add_prism(
                opos,
                up,
                boresight,
                8,
                sc * 0.48,
                sc * 0.45,
                sc * 0.36,
                [0.90, 0.76, 0.24],
                1.0,
                28.0,
                0.45,
                0.05,
                true,
                true,
            );
            mb.add_parabolic_dish(
                opos.add(up.scale(sc * 0.38)),
                up,
                boresight,
                10,
                sc * 0.56,
                sc * 0.18,
                [0.92, 0.96, 1.0],
                [0.32, 0.38, 0.48],
                36.0,
            );
            mb.add_solar_wing(
                opos.add(right.scale(sc * 0.48)),
                right,
                boresight,
                up,
                2,
                sc * 0.48,
                sc * 0.22,
                sc * 0.06,
                [0.10, 0.46, 0.92],
                [0.72, 0.78, 0.85],
            );
            mb.add_solar_wing(
                opos.sub(right.scale(sc * 0.48)),
                right.scale(-1.0),
                boresight,
                up,
                2,
                sc * 0.48,
                sc * 0.22,
                sc * 0.06,
                [0.10, 0.46, 0.92],
                [0.72, 0.78, 0.85],
            );
        }
    }
    mb.flush_into(out);
}

#[allow(clippy::too_many_arguments)]
fn build_space_probe_mesh(
    probe: &SpaceProbe,
    probe_world: Vec3d,
    fwd: Vec3d,
    right_axis: Vec3d,
    up_axis: Vec3d,
    camera: &OrbitCamera3D,
    aspect: f64,
    sc: f64,
    anim_time_sec: f64,
    out: &mut Vec<DepthGeom>,
) {
    let sun_dir = if probe_world.length() > 1e-6 {
        probe_world.scale(-1.0).normalized()
    } else {
        Vec3d::new(-1.0, 0.2, 0.0).normalized()
    };
    let mut mb = MeshBuilder3D::new(camera, aspect, sun_dir);

    // 1. Octagonal gold-MLI spacecraft bus
    mb.add_prism(
        probe_world,
        up_axis,
        fwd,
        8,
        sc * 0.62,
        sc * 0.58,
        sc * 0.42,
        [0.94, 0.73, 0.19],
        1.0,
        36.0,
        0.50,
        0.05,
        true,
        true,
    );
    // Forward titanium avionics & optical star-tracker deck
    mb.add_box(
        probe_world.add(fwd.scale(sc * 0.50)),
        right_axis,
        up_axis,
        fwd,
        sc * 0.36,
        sc * 0.24,
        sc * 0.24,
        [0.76, 0.82, 0.90],
        1.0,
        24.0,
        0.30,
        0.04,
    );
    // Magnetometer science boom
    let boom_root = probe_world.add(right_axis.scale(sc * 0.55)).add(up_axis.scale(sc * 0.22));
    let boom_dir = right_axis.scale(0.86).add(up_axis.scale(0.50)).normalized();
    let boom_len = sc * 1.45;
    mb.add_prism(
        boom_root.add(boom_dir.scale(boom_len * 0.5)),
        boom_dir,
        fwd,
        4,
        sc * 0.045,
        sc * 0.032,
        boom_len * 0.5,
        [0.74, 0.80, 0.88],
        1.0,
        20.0,
        0.25,
        0.03,
        true,
        true,
    );
    mb.add_box(
        boom_root.add(boom_dir.scale(boom_len)),
        right_axis,
        up_axis,
        fwd,
        sc * 0.12,
        sc * 0.10,
        sc * 0.10,
        [0.32, 0.88, 0.98],
        1.0,
        24.0,
        0.35,
        0.18,
    );

    // 2. 10-sector concave parabolic High-Gain Antenna (HGA) + tripod + feed horn
    let dish_axis = up_axis.scale(0.92).add(fwd.scale(0.38)).normalized();
    let dish_apex = probe_world.add(up_axis.scale(sc * 0.46));
    mb.add_parabolic_dish(
        dish_apex,
        dish_axis,
        right_axis,
        10,
        sc * 0.78,
        sc * 0.26,
        [0.93, 0.96, 1.0],
        [0.28, 0.34, 0.44],
        40.0,
    );

    // 3. Propulsion-specific 3D wings / finned GPHS-RTGs / 8-sector solar sail
    match probe.bus {
        PropulsionBus::ChemicalCentaur | PropulsionBus::SolarElectricIon => {
            let panel_len = if probe.bus == PropulsionBus::SolarElectricIon {
                sc * 0.86
            } else {
                sc * 0.60
            };
            let half_w = sc * 0.34;
            let wing_norm = up_axis.scale(0.86).add(fwd.scale(0.50)).normalized();
            let wing_chord = right_axis.cross(wing_norm).normalized();
            for sgn in [-1.0_f64, 1.0] {
                let root = probe_world.add(right_axis.scale(sgn * sc * 0.60));
                mb.add_solar_wing(
                    root,
                    right_axis.scale(sgn),
                    wing_chord,
                    wing_norm,
                    3,
                    panel_len,
                    half_w,
                    sc * 0.12,
                    [0.09, 0.40, 0.94],
                    [0.76, 0.82, 0.90],
                );
            }
        }
        PropulsionBus::RtgNuclearElectric => {
            for sgn in [-1.0_f64, 1.0] {
                let rtg_dir = right_axis.scale(sgn * 0.88).sub(up_axis.scale(0.46)).normalized();
                let rtg_c = probe_world
                    .add(right_axis.scale(sgn * sc * 0.92))
                    .sub(up_axis.scale(sc * 0.28));
                mb.add_prism(
                    rtg_c,
                    rtg_dir,
                    fwd,
                    8,
                    sc * 0.16,
                    sc * 0.15,
                    sc * 0.56,
                    [0.24, 0.27, 0.32],
                    1.0,
                    22.0,
                    0.35,
                    0.05,
                    true,
                    true,
                );
                // Glowing amber heat-rejection radiator fins
                let fin_perp1 = rtg_dir.cross(fwd).normalized();
                let fin_perp2 = rtg_dir.cross(fin_perp1).normalized();
                for fdir in [fin_perp1, fin_perp2] {
                    mb.add_box(
                        rtg_c,
                        rtg_dir,
                        fdir,
                        rtg_dir.cross(fdir),
                        sc * 0.48,
                        sc * 0.32,
                        sc * 0.03,
                        [0.95, 0.45, 0.14],
                        1.0,
                        18.0,
                        0.25,
                        0.28,
                    );
                }
            }
        }
        PropulsionBus::LaserSailFusion => {
            let sail_r = if probe.phase == ProbePhase::EnRoute {
                sc * 2.55
            } else {
                sc * 1.55
            };
            let sail_c = probe_world.sub(fwd.scale(sc * 0.28));
            for i in 0..8 {
                let a0 = (i as f64 / 8.0) * TAU;
                let a1 = ((i + 1) as f64 / 8.0) * TAU;
                let p0 = sail_c
                    .add(right_axis.scale(sail_r * a0.cos()))
                    .add(up_axis.scale(sail_r * a0.sin()));
                let p1 = sail_c
                    .add(right_axis.scale(sail_r * a1.cos()))
                    .add(up_axis.scale(sail_r * a1.sin()));
                mb.add_tri_uv(
                    sail_c,
                    p0,
                    p1,
                    [[0.5, 0.5], [0.0, 1.0], [1.0, 1.0]],
                    [0.78, 0.56, 0.98],
                    0.82,
                    54.0,
                    0.72,
                    0.12,
                    true,
                    0.0002,
                );
                let spar_dir = p0.sub(sail_c);
                mb.add_prism(
                    sail_c.lerp(p0, 0.5),
                    spar_dir,
                    fwd,
                    4,
                    sc * 0.024,
                    sc * 0.018,
                    spar_dir.length() * 0.5,
                    [0.86, 0.90, 0.96],
                    1.0,
                    28.0,
                    0.4,
                    0.04,
                    true,
                    true,
                );
            }
        }
    }

    // 4. Aft thruster bell + glowing exhaust cone
    let bell_c = probe_world.sub(fwd.scale(sc * 0.56));
    mb.add_prism(
        bell_c,
        fwd.scale(-1.0),
        up_axis,
        8,
        sc * 0.18,
        sc * 0.30,
        sc * 0.16,
        [0.24, 0.26, 0.31],
        1.0,
        30.0,
        0.42,
        0.02,
        true,
        false,
    );
    let pulse = 0.88 + 0.22 * ((anim_time_sec * 12.0 + probe.id as f64).sin());
    let plume_len = if probe.phase == ProbePhase::EnRoute {
        sc * 0.95 * pulse
    } else {
        sc * 0.32
    };
    let plume_col = match probe.bus {
        PropulsionBus::ChemicalCentaur => [1.0, 0.72, 0.22],
        PropulsionBus::SolarElectricIon => [0.22, 0.90, 1.0],
        PropulsionBus::RtgNuclearElectric => [0.46, 0.62, 1.0],
        PropulsionBus::LaserSailFusion => [0.92, 0.48, 1.0],
    };
    mb.add_prism(
        bell_c.sub(fwd.scale(sc * 0.16 + plume_len * 0.5)),
        fwd.scale(-1.0),
        up_axis,
        8,
        sc * 0.22,
        sc * 0.03,
        plume_len * 0.5,
        plume_col,
        0.78,
        10.0,
        0.10,
        0.82,
        true,
        true,
    );

    mb.flush_into(out);
}

#[allow(clippy::too_many_arguments)]
fn build_surface_patch_and_unit(
    body: &CelestialBody,
    body_world: Vec3d,
    unit: &SurfaceDeployment,
    u_idx: usize,
    camera: &OrbitCamera3D,
    aspect: f64,
    anim_time_sec: f64,
    out: &mut Vec<DepthGeom>,
) {
    let anchor = unit_world_pos(
        body,
        body_world,
        u_idx,
        anim_time_sec,
        unit.descent_progress,
    );
    let Some((_, _, u_depth)) = camera.project(anchor.unit_pos, aspect) else {
        return;
    };
    let r_vis = visual_body_radius_scene(body);
    let sun_dir = if body_world.length() > 1e-6 {
        body_world.scale(-1.0).normalized()
    } else {
        Vec3d::new(0.4, 0.8, 0.4).normalized()
    };
    let mut mb = MeshBuilder3D::new(camera, aspect, sun_dir);

    let is_icy = ICY_CRYOBOT_WORLDS.contains(&body.id.as_str());
    let sc = anchor.rig_scale;
    let is_cryo_cutaway = unit.kind == SurfaceUnitKind::OceanCryobot && unit.descent_progress >= 1.0;

    // 1. Tessellated 3D curved surface patch directly beneath the lander / cryobot when close
    if u_depth < (r_vis * 8.0).max(2.2) {
        let patch_r = (sc * 9.5).min(r_vis * 0.78);
        let r_inner = if unit.kind == SurfaceUnitKind::OceanCryobot {
            sc * 0.80
        } else {
            0.0
        };
        let radial_rings = 7usize;
        let sectors = 24usize;
        // For OceanCryobot, open a 110-deg front cutaway sector centered at theta = 0.95 rad
        // (facing the default yaw = 0.62 rad camera) so the 250-deg ice shell & sub-ice ocean are visible!
        let (theta_start, theta_span) = if is_cryo_cutaway {
            (1.91_f64, TAU * (250.0 / 360.0))
        } else {
            (0.0_f64, TAU)
        };
        let surf_pt = |r: f64, theta: f64| -> Vec3d {
            let u = r * theta.cos();
            let v = r * theta.sin();
            let dist2 = r * r;
            let sphere_y = (r_vis * r_vis - dist2).max(0.0).sqrt() - r_vis;
            let flat_mask = ((r / (sc * 1.45)) - 0.85).clamp(0.0, 1.0);
            let ridge = flat_mask
                * sc
                * 0.24
                * ((u * 42.0).sin() * (v * 34.0).cos() + 0.5 * ((u + v) * 82.0).sin());
            anchor
                .surface_pt
                .add(anchor.east.scale(u))
                .add(anchor.north.scale(v))
                .add(anchor.normal.scale(sphere_y + ridge))
        };

        for ri in 0..radial_rings {
            let f0 = ri as f64 / radial_rings as f64;
            let f1 = (ri + 1) as f64 / radial_rings as f64;
            let r0 = r_inner + (patch_r - r_inner) * f0;
            let r1 = r_inner + (patch_r - r_inner) * f1;
            let alpha = if ri == radial_rings - 1 {
                0.70
            } else if ri == radial_rings - 2 {
                0.90
            } else {
                0.98
            };
            for si in 0..sectors {
                let t0 = theta_start + (si as f64 / sectors as f64) * theta_span;
                let t1 = theta_start + ((si + 1) as f64 / sectors as f64) * theta_span;
                let p00 = surf_pt(r0, t0);
                let p10 = surf_pt(r1, t0);
                let p11 = surf_pt(r1, t1);
                let p01 = surf_pt(r0, t1);
                let mid_r = 0.5 * (r0 + r1);
                let mid_t = 0.5 * (t0 + t1);
                let u = mid_r * mid_t.cos();
                let v = mid_r * mid_t.sin();
                let tile_rgb = if is_icy {
                    // Glacial blue-white crust with warm Europa/Enceladus tectonic lineae bands
                    let lineae = ((u * 26.0 + v * 41.0).sin()).abs();
                    if lineae < 0.24 {
                        [0.70, 0.65, 0.60]
                    } else {
                        [0.76, 0.89, 0.98]
                    }
                } else {
                    let mod_f = 0.90 + 0.14 * ((u * 38.0).cos() * (v * 38.0).sin()) as f32;
                    [
                        (body.color_rgb[0] * mod_f).clamp(0.0, 1.0),
                        (body.color_rgb[1] * mod_f).clamp(0.0, 1.0),
                        (body.color_rgb[2] * mod_f).clamp(0.0, 1.0),
                    ]
                };
                mb.add_quad(
                    p00,
                    p10,
                    p11,
                    p01,
                    tile_rgb,
                    alpha,
                    if is_icy { 28.0 } else { 10.0 },
                    if is_icy { 0.28 } else { 0.08 },
                    0.02,
                    true,
                    0.006,
                );
            }
        }

        if is_cryo_cutaway {
            // Geological cutaway: stratified ice-shell cross-section walls + deep liquid brine ocean slab
            let ice_top_z = 0.0_f64;
            let ice_mid_z = -sc * 1.15;
            let ice_bot_z = -sc * 2.35;
            let ocean_bot_z = -sc * 3.15;
            let wall_outer_r = patch_r * 0.84;

            // Radial cutaway cross-section walls at theta_start and theta_end
            for &cut_theta in &[theta_start, theta_start + theta_span] {
                let rad_dir = anchor
                    .east
                    .scale(cut_theta.cos())
                    .add(anchor.north.scale(cut_theta.sin()));
                let p_in_top = surf_pt(r_inner, cut_theta);
                let p_out_top = surf_pt(wall_outer_r, cut_theta);
                let p_in_mid = anchor
                    .surface_pt
                    .add(rad_dir.scale(r_inner))
                    .add(anchor.normal.scale(ice_mid_z));
                let p_out_mid = anchor
                    .surface_pt
                    .add(rad_dir.scale(wall_outer_r))
                    .add(anchor.normal.scale(ice_mid_z));
                let p_in_bot = anchor
                    .surface_pt
                    .add(rad_dir.scale(r_inner))
                    .add(anchor.normal.scale(ice_bot_z));
                let p_out_bot = anchor
                    .surface_pt
                    .add(rad_dir.scale(wall_outer_r))
                    .add(anchor.normal.scale(ice_bot_z));
                let p_in_oc = anchor
                    .surface_pt
                    .add(rad_dir.scale(r_inner * 0.4))
                    .add(anchor.normal.scale(ocean_bot_z));
                let p_out_oc = anchor
                    .surface_pt
                    .add(rad_dir.scale(wall_outer_r))
                    .add(anchor.normal.scale(ocean_bot_z));

                // Upper crisp glacial ice layer
                mb.add_quad(
                    p_in_top,
                    p_out_top,
                    p_out_mid,
                    p_in_mid,
                    [0.74, 0.88, 0.98],
                    0.96,
                    24.0,
                    0.22,
                    0.06,
                    true,
                    0.004,
                );
                // Lower fractured briny-ice shell
                mb.add_quad(
                    p_in_mid,
                    p_out_mid,
                    p_out_bot,
                    p_in_bot,
                    [0.38, 0.66, 0.86],
                    0.96,
                    20.0,
                    0.18,
                    0.05,
                    true,
                    0.004,
                );
                // Subsurface abyssal brine ocean wall
                mb.add_quad(
                    p_in_bot,
                    p_out_bot,
                    p_out_oc,
                    p_in_oc,
                    [0.06, 0.34, 0.60],
                    0.94,
                    30.0,
                    0.35,
                    0.12,
                    true,
                    0.004,
                );
            }

            // Curved inner borehole ice wall & subsurface ocean cavern floor across the 250-deg arc
            for si in 0..sectors {
                let t0 = theta_start + (si as f64 / sectors as f64) * theta_span;
                let t1 = theta_start + ((si + 1) as f64 / sectors as f64) * theta_span;
                let d0 = anchor.east.scale(t0.cos()).add(anchor.north.scale(t0.sin()));
                let d1 = anchor.east.scale(t1.cos()).add(anchor.north.scale(t1.sin()));

                let w0_top = anchor.surface_pt.add(d0.scale(r_inner)).add(anchor.normal.scale(ice_top_z));
                let w1_top = anchor.surface_pt.add(d1.scale(r_inner)).add(anchor.normal.scale(ice_top_z));
                let w0_mid = anchor.surface_pt.add(d0.scale(r_inner)).add(anchor.normal.scale(ice_mid_z));
                let w1_mid = anchor.surface_pt.add(d1.scale(r_inner)).add(anchor.normal.scale(ice_mid_z));
                let w0_bot = anchor.surface_pt.add(d0.scale(r_inner * 1.15)).add(anchor.normal.scale(ice_bot_z));
                let w1_bot = anchor.surface_pt.add(d1.scale(r_inner * 1.15)).add(anchor.normal.scale(ice_bot_z));

                mb.add_quad(
                    w0_top,
                    w1_top,
                    w1_mid,
                    w0_mid,
                    [0.54, 0.82, 0.96],
                    0.82,
                    28.0,
                    0.28,
                    0.08,
                    true,
                    0.003,
                );
                mb.add_quad(
                    w0_mid,
                    w1_mid,
                    w1_bot,
                    w0_bot,
                    [0.24, 0.56, 0.80],
                    0.85,
                    28.0,
                    0.30,
                    0.10,
                    true,
                    0.003,
                );
            }

            // Subsurface liquid brine ocean floor slab bounded strictly inside the 250-deg cutaway wedge
            let oc_rings = 4usize;
            let oc_sectors = 20usize;
            let c_oc = anchor.surface_pt.add(anchor.normal.scale(ocean_bot_z));
            for ri in 0..oc_rings {
                let r0 = wall_outer_r * (ri as f64 / oc_rings as f64);
                let r1 = wall_outer_r * ((ri + 1) as f64 / oc_rings as f64);
                for si in 0..oc_sectors {
                    let t0 = theta_start + (si as f64 / oc_sectors as f64) * theta_span;
                    let t1 = theta_start + ((si + 1) as f64 / oc_sectors as f64) * theta_span;
                    let d0 = anchor.east.scale(t0.cos()).add(anchor.north.scale(t0.sin()));
                    let d1 = anchor.east.scale(t1.cos()).add(anchor.north.scale(t1.sin()));
                    let p00 = c_oc.add(d0.scale(r0));
                    let p10 = c_oc.add(d0.scale(r1));
                    let p11 = c_oc.add(d1.scale(r1));
                    let p01 = c_oc.add(d1.scale(r0));
                    if ri == 0 {
                        mb.add_tri_uv(
                            c_oc,
                            p10,
                            p11,
                            [[0.5, 0.5], [1.0, 0.5], [0.5, 1.0]],
                            [0.05, 0.34, 0.62],
                            0.95,
                            36.0,
                            0.45,
                            0.18,
                            true,
                            0.002,
                        );
                    } else {
                        mb.add_quad(
                            p00,
                            p10,
                            p11,
                            p01,
                            [0.05, 0.34, 0.62],
                            0.95,
                            36.0,
                            0.45,
                            0.18,
                            true,
                            0.002,
                        );
                    }
                }
            }
        }
    }

    // 2. Model geometry per SurfaceUnitKind
    let pos = anchor.unit_pos;
    let n = anchor.normal;
    let e = anchor.east;
    let nr = anchor.north;

    match unit.kind {
        SurfaceUnitKind::OceanCryobot => {
            // Glowing cyan melt borehole collar & translucent sub-ice column planted on surface_pt
            if u_depth < (r_vis * 8.0).max(2.2) {
                // Glowing cyan borehole rim on surface (250-deg cutaway or full collar)
                mb.add_prism(
                    anchor.surface_pt.add(n.scale(sc * 0.04)),
                    n,
                    e,
                    14,
                    sc * 0.88,
                    sc * 0.74,
                    sc * 0.06,
                    [0.14, 0.92, 1.0],
                    0.96,
                    24.0,
                    0.35,
                    0.58,
                    false,
                    false,
                );
                // Translucent sub-ice melt shaft extending deep into the ice shell cutaway
                let shaft_half = if is_cryo_cutaway { sc * 1.15 } else { sc * 0.68 };
                mb.add_prism(
                    anchor.surface_pt.sub(n.scale(shaft_half)),
                    n,
                    e,
                    12,
                    sc * 0.68,
                    sc * 0.76,
                    shaft_half,
                    [0.20, 0.74, 0.98],
                    0.22,
                    32.0,
                    0.30,
                    0.24,
                    true,
                    false,
                );
            }

            // 4-legged titanium ice-anchor gantry planted at pos
            let deck_y = sc * 0.94;
            let deck_c = pos.add(n.scale(deck_y));
            mb.add_prism(
                deck_c,
                n,
                e,
                8,
                sc * 0.86,
                sc * 0.80,
                sc * 0.14,
                [0.74, 0.82, 0.90],
                1.0,
                32.0,
                0.45,
                0.05,
                true,
                true,
            );
            for k in 0..4 {
                let ang = (k as f64) * (TAU * 0.25) + 0.785;
                let dir = e.scale(ang.cos()).add(nr.scale(ang.sin()));
                let top_j = deck_c.add(dir.scale(sc * 0.70));
                let foot = pos.add(dir.scale(sc * 1.32)).add(n.scale(sc * 0.04));
                let leg_v = foot.sub(top_j);
                mb.add_prism(
                    top_j.lerp(foot, 0.5),
                    leg_v,
                    n,
                    4,
                    sc * 0.08,
                    sc * 0.065,
                    leg_v.length() * 0.5,
                    [0.64, 0.72, 0.82],
                    1.0,
                    28.0,
                    0.40,
                    0.03,
                    true,
                    true,
                );
                // Wide circular ice-anchor footpad flush on ice crust
                mb.add_prism(
                    foot,
                    n,
                    e,
                    8,
                    sc * 0.22,
                    sc * 0.18,
                    sc * 0.045,
                    [0.92, 0.74, 0.24],
                    1.0,
                    26.0,
                    0.35,
                    0.05,
                    true,
                    true,
                );
            }

            // Tether spool winch + 3D Ka-band dish on gantry deck
            let winch_c = deck_c.add(n.scale(sc * 0.28)).sub(e.scale(sc * 0.26));
            mb.add_prism(
                winch_c,
                nr,
                e,
                10,
                sc * 0.24,
                sc * 0.24,
                sc * 0.28,
                [0.88, 0.62, 0.24],
                1.0,
                30.0,
                0.42,
                0.05,
                true,
                true,
            );
            let dish_apex = deck_c.add(n.scale(sc * 0.22)).add(e.scale(sc * 0.34));
            let dish_axis = n.scale(0.84).add(e.scale(0.54)).normalized();
            mb.add_parabolic_dish(
                dish_apex,
                dish_axis,
                nr,
                10,
                sc * 0.56,
                sc * 0.19,
                [0.94, 0.97, 1.0],
                [0.32, 0.38, 0.48],
                42.0,
            );

            // Vertical brushed-titanium cylindrical thermal melt probe + glowing Pu-238 nosecone + relay pucks
            let probe_top = if is_cryo_cutaway {
                anchor.surface_pt.sub(n.scale(sc * 1.18))
            } else {
                pos.sub(n.scale(sc * 0.04))
            };
            let probe_len = if is_cryo_cutaway { sc * 0.68 } else { sc * 0.42 };
            let probe_c = probe_top.sub(n.scale(probe_len * 0.5));
            mb.add_prism(
                probe_c,
                n,
                e,
                12,
                sc * 0.24,
                sc * 0.24,
                probe_len * 0.5,
                [0.82, 0.88, 0.94],
                1.0,
                40.0,
                0.55,
                0.10,
                true,
                true,
            );
            // Glowing orange-red Pu-238 hemispherical/conical thermal nosecone melting into ocean interface
            let nose_half = sc * 0.22;
            let nose_c = probe_top.sub(n.scale(probe_len + nose_half));
            mb.add_prism(
                nose_c,
                n,
                e,
                12,
                sc * 0.05,
                sc * 0.24,
                nose_half,
                [1.0, 0.42, 0.10],
                1.0,
                16.0,
                0.25,
                0.92,
                true,
                true,
            );

            if unit.descent_progress >= 1.0 {
                // Hydrothermal melt plume beneath Pu-238 nosecone in the brine ocean
                mb.add_prism(
                    nose_c.sub(n.scale(sc * 0.36)),
                    n,
                    e,
                    10,
                    sc * 0.48,
                    sc * 0.12,
                    sc * 0.22,
                    [0.18, 0.88, 1.0],
                    0.62,
                    12.0,
                    0.15,
                    0.75,
                    true,
                    true,
                );
                // Optical/fiber tether + 4 submerged wireless acoustic/RF relay pucks
                let tether_top = deck_c.sub(n.scale(sc * 0.10));
                let tether_v = probe_top.sub(tether_top);
                mb.add_prism(
                    tether_top.lerp(probe_top, 0.5),
                    tether_v,
                    e,
                    4,
                    sc * 0.036,
                    sc * 0.036,
                    tether_v.length() * 0.5,
                    [0.95, 0.78, 0.28],
                    1.0,
                    22.0,
                    0.3,
                    0.14,
                    true,
                    true,
                );
                for pk in 0..4 {
                    let frac = 0.16 + 0.23 * (pk as f64);
                    let ppos = anchor.surface_pt.add(n.scale(sc * 0.08)).lerp(probe_top, frac);
                    mb.add_prism(
                        ppos,
                        n,
                        e,
                        8,
                        sc * 0.13,
                        sc * 0.13,
                        sc * 0.055,
                        [0.24, 0.96, 1.0],
                        0.98,
                        24.0,
                        0.35,
                        0.78,
                        true,
                        true,
                    );
                }
            } else {
                // Powered retro-braking descent plume beneath descending gantry
                let plume_c = pos.sub(n.scale(sc * 0.35));
                mb.add_prism(
                    plume_c,
                    n,
                    e,
                    8,
                    sc * 0.04,
                    sc * 0.45,
                    sc * 0.38,
                    [1.0, 0.65, 0.22],
                    0.80,
                    8.0,
                    0.1,
                    0.85,
                    true,
                    true,
                );
            }
        }
        SurfaceUnitKind::AeroShellLander => {
            // Hexagonal gold-MLI lander deck
            let deck_c = pos.add(n.scale(sc * 0.68));
            mb.add_prism(
                deck_c,
                n,
                e,
                6,
                sc * 0.94,
                sc * 0.88,
                sc * 0.16,
                [0.94, 0.72, 0.20],
                1.0,
                36.0,
                0.50,
                0.05,
                true,
                true,
            );
            // Upper white avionics module
            mb.add_box(
                deck_c.add(n.scale(sc * 0.26)),
                e,
                n,
                nr,
                sc * 0.42,
                sc * 0.14,
                sc * 0.36,
                [0.86, 0.90, 0.95],
                1.0,
                24.0,
                0.30,
                0.04,
            );
            // Twin solar wings
            for sgn in [-1.0_f64, 1.0] {
                mb.add_solar_wing(
                    deck_c.add(e.scale(sgn * sc * 0.84)),
                    e.scale(sgn),
                    nr,
                    n,
                    2,
                    sc * 0.48,
                    sc * 0.30,
                    sc * 0.06,
                    [0.09, 0.40, 0.92],
                    [0.74, 0.80, 0.88],
                );
            }
            // 4 A-frame shock legs + landing footpads
            for k in 0..4 {
                let ang = (k as f64) * (TAU * 0.25) + 0.785;
                let dir = e.scale(ang.cos()).add(nr.scale(ang.sin()));
                let tang = n.cross(dir).normalized();
                let foot = pos.add(dir.scale(sc * 1.32)).add(n.scale(sc * 0.04));
                for brace in [-0.26_f64, 0.26] {
                    let top_j = deck_c.add(dir.scale(sc * 0.74)).add(tang.scale(sc * brace));
                    let strut_v = foot.sub(top_j);
                    mb.add_prism(
                        top_j.lerp(foot, 0.5),
                        strut_v,
                        n,
                        4,
                        sc * 0.055,
                        sc * 0.045,
                        strut_v.length() * 0.5,
                        [0.64, 0.72, 0.82],
                        1.0,
                        24.0,
                        0.38,
                        0.03,
                        true,
                        true,
                    );
                }
                mb.add_prism(
                    foot,
                    n,
                    e,
                    8,
                    sc * 0.22,
                    sc * 0.16,
                    sc * 0.045,
                    [0.86, 0.74, 0.34],
                    1.0,
                    24.0,
                    0.35,
                    0.04,
                    true,
                    true,
                );
            }
            // Stereo Pancam mast + optical head
            let mast_base = deck_c.add(n.scale(sc * 0.38)).add(nr.scale(sc * 0.35));
            let mast_top = mast_base.add(n.scale(sc * 0.72));
            mb.add_prism(
                mast_base.lerp(mast_top, 0.5),
                n,
                e,
                6,
                sc * 0.06,
                sc * 0.055,
                sc * 0.36,
                [0.82, 0.86, 0.92],
                1.0,
                28.0,
                0.40,
                0.04,
                true,
                true,
            );
            mb.add_box(
                mast_top.add(n.scale(sc * 0.09)),
                e,
                n,
                nr,
                sc * 0.22,
                sc * 0.09,
                sc * 0.10,
                [0.92, 0.94, 0.98],
                1.0,
                28.0,
                0.42,
                0.05,
            );
            // Core-drill tower
            let drill_c = deck_c.sub(e.scale(sc * 0.52)).sub(nr.scale(sc * 0.28));
            mb.add_prism(
                drill_c.sub(n.scale(sc * 0.22)),
                n,
                e,
                6,
                sc * 0.06,
                sc * 0.09,
                sc * 0.44,
                [0.54, 0.62, 0.74],
                1.0,
                28.0,
                0.45,
                0.03,
                true,
                true,
            );
            // SEIS copper dome on regolith beside lander
            let seis_c = pos.add(e.scale(sc * 1.56)).add(nr.scale(sc * 0.38)).add(n.scale(sc * 0.14));
            mb.add_prism(
                seis_c,
                n,
                e,
                10,
                sc * 0.32,
                sc * 0.16,
                sc * 0.14,
                [0.88, 0.48, 0.20],
                1.0,
                34.0,
                0.48,
                0.06,
                true,
                true,
            );
            if unit.descent_progress < 1.0 {
                mb.add_prism(
                    pos.sub(n.scale(sc * 0.28)),
                    n,
                    e,
                    8,
                    sc * 0.05,
                    sc * 0.40,
                    sc * 0.32,
                    [1.0, 0.68, 0.24],
                    0.82,
                    8.0,
                    0.1,
                    0.85,
                    true,
                    true,
                );
            }
        }
        SurfaceUnitKind::AtmosphericEntryProbe => {
            mb.add_prism(
                pos.add(n.scale(sc * 0.18)),
                n,
                e,
                8,
                sc * 0.42,
                sc * 0.12,
                sc * 0.20,
                [0.98, 0.46, 0.24],
                1.0,
                26.0,
                0.35,
                0.24,
                true,
                true,
            );
        }
    }

    mb.flush_into(out);
}

fn make_billboard_quad_extra(
    ndc_x: f32,
    ndc_y: f32,
    half_w: f32,
    half_h: f32,
    color: [f32; 4],
    kind: f32,
    param: f32,
    extra: [f32; 4],
) -> [GpuVertex; 6] {
    let v0 = GpuVertex {
        clip_pos: [ndc_x - half_w, ndc_y - half_h],
        uv: [-1.0, -1.0],
        color,
        kind_param: [kind, param],
        extra,
    };
    let v1 = GpuVertex {
        clip_pos: [ndc_x + half_w, ndc_y - half_h],
        uv: [1.0, -1.0],
        color,
        kind_param: [kind, param],
        extra,
    };
    let v2 = GpuVertex {
        clip_pos: [ndc_x + half_w, ndc_y + half_h],
        uv: [1.0, 1.0],
        color,
        kind_param: [kind, param],
        extra,
    };
    let v3 = GpuVertex {
        clip_pos: [ndc_x - half_w, ndc_y + half_h],
        uv: [-1.0, 1.0],
        color,
        kind_param: [kind, param],
        extra,
    };
    [v0, v1, v2, v0, v2, v3]
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
    make_billboard_quad_extra(ndc_x, ndc_y, half_w, half_h, color, kind, param, [0.0; 4])
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
        extra: [0.0; 4],
    };
    let v1 = GpuVertex {
        clip_pos: [b.0 - nx, b.1 - ny],
        uv: [1.0, -1.0],
        color,
        kind_param: [4.0, 0.0],
        extra: [0.0; 4],
    };
    let v2 = GpuVertex {
        clip_pos: [b.0 + nx, b.1 + ny],
        uv: [1.0, 1.0],
        color,
        kind_param: [4.0, 0.0],
        extra: [0.0; 4],
    };
    let v3 = GpuVertex {
        clip_pos: [a.0 + nx, a.1 + ny],
        uv: [0.0, 1.0],
        color,
        kind_param: [4.0, 0.0],
        extra: [0.0; 4],
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

    let mut quads: Vec<DepthGeom> = Vec::with_capacity(3400);
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
                quads.push(DepthGeom::Quad {
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
                let steps = if body.category == BodyCategory::Comet || body.eccentricity > 0.35 {
                    192usize
                } else {
                    72usize
                };
                let is_sel = body.id == selected_id;
                let alpha = if is_sel { 0.62 } else { 0.22 };
                let col = [body.color_rgb[0], body.color_rgb[1], body.color_rgb[2], alpha];
                let mut prev: Option<(f32, f32, f64)> = None;
                for pos_au in sample_orbit_ribbon_points_au(body, steps) {
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
                                quads.push(DepthGeom::Quad {
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
        .map(|e| body_scene_position_with_offsets(e, catalog, days_since_j2000, comet_perihelion_offsets))
        .unwrap_or(Vec3d::new(16.45, 0.0, 0.0));
    let cislunar_l2_closeup = earth_scene_now.sub(camera.target).length() < 7.5
        && camera.distance > 0.6
        && camera.distance < 12.0;

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
                        quads.push(DepthGeom::Quad {
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
                    if ox.abs() <= 1.3 && oy.abs() <= 1.3 {
                        let o_proj = ((1.0 / (camera.fov_y_rad * 0.5).tan()) / odepth) as f32;
                        let desired_ndc = (0.036 * o_proj).clamp(0.018, 0.16);
                        let sc = (desired_ndc / o_proj) as f64;
                        build_l2_observatory_mesh(
                            obs.id,
                            opos,
                            earth_scene_now,
                            camera,
                            aspect,
                            sc,
                            &mut quads,
                        );
                    }
                }
            }
        }
    }

    // 3. Celestial bodies, moons, comets, exoplanets, volumetric nebulae & grounded surface rigs
    for body in catalog {
        let world = body_scene_position_with_offsets(body, catalog, days_since_j2000, comet_perihelion_offsets);
        if body.category == BodyCategory::Comet {
            let off_deg = comet_perihelion_offsets.get(&body.id).copied().unwrap_or(0.0);
            let tail = compute_comet_tail_state(body, days_since_j2000, off_deg);
            if tail.activity > 0.03 {
                if let Some((cx, cy, cdepth)) = camera.project(world, aspect) {
                    let c_proj = ((1.0 / (camera.fov_y_rad * 0.5).tan()) / cdepth) as f32;
                    let ch = ((tail.coma_radius_scene as f32) * c_proj).clamp(6.0 / height, 0.45);
                    quads.push(DepthGeom::Quad {
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
                        quads.push(DepthGeom::Quad {
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
                                quads.push(DepthGeom::Quad {
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
            let mut base_world_size = visual_body_radius_scene(body);
            if cislunar_l2_closeup {
                if body.id == "earth" || body.id == "moon" {
                    base_world_size *= 0.30;
                } else if body.category == BodyCategory::Planet {
                    base_world_size *= 0.30;
                }
            }
            let proj_scale = (1.0 / (camera.fov_y_rad * 0.5).tan()) / depth;
            let min_px = if matches!(body.category, BodyCategory::Moon | BodyCategory::Asteroid) {
                3.2 / height
            } else {
                5.5 / height
            };
            let max_half_h = if body.id == selected_id && camera.distance <= 1.2 {
                12.0_f32
            } else {
                0.48_f32
            };
            let half_h = ((base_world_size * proj_scale) as f32).clamp(min_px, max_half_h);
            let half_w = half_h / aspect_f32;
            let max_ndc_x = 1.65 + half_w.min(12.0);
            let max_ndc_y = 1.65 + half_h.min(12.0);

            if nx.abs() <= max_ndc_x && ny.abs() <= max_ndc_y {
                let light_view = body_sun_light_view_space(camera, world);
                let sun_angle = world.z.atan2(world.x) as f32 + 3.14159;
                let (kind, draw_hw, draw_hh, body_alpha) = match body.category {
                    BodyCategory::Star => (0.0, half_w, half_h, 1.0),
                    BodyCategory::Nebula => (2.0, half_w, half_h, 1.0),
                    BodyCategory::LagrangePoint => (0.0, half_w * 0.42, half_h * 0.42, 0.48),
                    _ => (1.0, half_w, half_h, 1.0),
                };
                let param = if matches!(body.category, BodyCategory::Star | BodyCategory::Nebula | BodyCategory::LagrangePoint) {
                    anim_time_sec as f32
                } else {
                    sun_angle
                };

                quads.push(DepthGeom::Quad {
                    depth,
                    verts: make_billboard_quad_extra(
                        nx,
                        ny,
                        draw_hw,
                        draw_hh,
                        [body.color_rgb[0], body.color_rgb[1], body.color_rgb[2], body_alpha],
                        kind,
                        param,
                        [light_view[0], light_view[1], light_view[2], 0.0],
                    ),
                });

                if body.has_rings && body.category != BodyCategory::Nebula {
                    let spec = planetary_ring_spec(&body.id);
                    let (ring_u, ring_v, ring_n) = planetary_ring_basis(&body.id);
                    let apparent_world_r = (half_h as f64) / proj_scale.max(1e-6);
                    let r_outer = apparent_world_r * spec.ring_mul;
                    let r_inner = r_outer * spec.r_inner_norm;
                    let to_sun = if world.length() > 1e-6 {
                        world.scale(-1.0).normalized()
                    } else {
                        Vec3d::new(0.0, 1.0, 0.0)
                    };
                    let sun_grazing = (0.78 + 0.28 * ring_n.dot(to_sun).abs()) as f32;
                    let ring_segs = 56usize;
                    let ri = spec.r_inner_norm as f32;
                    let col = [
                        spec.color_rgb[0],
                        spec.color_rgb[1],
                        spec.color_rgb[2],
                        spec.ring_alpha,
                    ];

                    for s in 0..ring_segs {
                        let t0 = (s as f64 / ring_segs as f64) * TAU;
                        let t1 = ((s + 1) as f64 / ring_segs as f64) * TAU;
                        let d0 = ring_u.scale(t0.cos()).add(ring_v.scale(t0.sin()));
                        let d1 = ring_u.scale(t1.cos()).add(ring_v.scale(t1.sin()));
                        let w_in0 = world.add(d0.scale(r_inner));
                        let w_out0 = world.add(d0.scale(r_outer));
                        let w_out1 = world.add(d1.scale(r_outer));
                        let w_in1 = world.add(d1.scale(r_inner));

                        let (Some(q_in0), Some(q_out0), Some(q_out1), Some(q_in1)) = (
                            camera.project(w_in0, aspect),
                            camera.project(w_out0, aspect),
                            camera.project(w_out1, aspect),
                            camera.project(w_in1, aspect),
                        ) else {
                            continue;
                        };

                        let tm = 0.5 * (t0 + t1);
                        let dm = ring_u.scale(tm.cos()).add(ring_v.scale(tm.sin()));
                        let r_mid = 0.5 * (r_inner + r_outer);
                        let mid_rel = dm.scale(r_mid);
                        let along_sun = mid_rel.dot(to_sun);
                        let perp_dist = mid_rel.sub(to_sun.scale(along_sun)).length();
                        let shadow_fac = if along_sun < 0.0 && perp_dist < apparent_world_r * 1.02 {
                            let edge = ((perp_dist / apparent_world_r.max(1e-6)) - 0.85) / 0.17;
                            0.24 + 0.76 * edge.clamp(0.0, 1.0)
                        } else {
                            1.0
                        };
                        let sun_lit = (sun_grazing * (shadow_fac as f32)).clamp(0.18, 1.12);
                        let seg_depth = 0.25 * (q_in0.2 + q_out0.2 + q_out1.2 + q_in1.2);
                        let extra = [ri, sun_lit, spec.cassini_strength, 0.0];
                        let v_in0 = GpuVertex {
                            clip_pos: [q_in0.0, q_in0.1],
                            uv: [ri * t0.cos() as f32, ri * t0.sin() as f32],
                            color: col,
                            kind_param: [3.0, anim_time_sec as f32],
                            extra,
                        };
                        let v_out0 = GpuVertex {
                            clip_pos: [q_out0.0, q_out0.1],
                            uv: [t0.cos() as f32, t0.sin() as f32],
                            color: col,
                            kind_param: [3.0, anim_time_sec as f32],
                            extra,
                        };
                        let v_out1 = GpuVertex {
                            clip_pos: [q_out1.0, q_out1.1],
                            uv: [t1.cos() as f32, t1.sin() as f32],
                            color: col,
                            kind_param: [3.0, anim_time_sec as f32],
                            extra,
                        };
                        let v_in1 = GpuVertex {
                            clip_pos: [q_in1.0, q_in1.1],
                            uv: [ri * t1.cos() as f32, ri * t1.sin() as f32],
                            color: col,
                            kind_param: [3.0, anim_time_sec as f32],
                            extra,
                        };
                        quads.push(DepthGeom::Quad {
                            depth: seg_depth,
                            verts: [v_in0, v_out0, v_out1, v_in0, v_out1, v_in1],
                        });
                    }
                }

                if let Some((scan_id, pulse_age)) = active_scan_pulse {
                    if scan_id == body.id {
                        quads.push(DepthGeom::Quad {
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

                if nx.abs() <= 1.15 && ny.abs() <= 1.15 {
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
        }

        if let Some(units) = surface_deployments.get(&body.id) {
            for (u_idx, unit) in units.iter().enumerate() {
                build_surface_patch_and_unit(
                    body,
                    world,
                    unit,
                    u_idx,
                    camera,
                    aspect,
                    anim_time_sec,
                    &mut quads,
                );
            }
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
                let p_world = sample.scene_position_with_offsets(
                    catalog,
                    days_since_j2000,
                    comet_perihelion_offsets,
                );
                let proj = camera.project(p_world, aspect);
                if let (Some(a), Some(b)) = (prev_pt, proj) {
                    if let Some(verts) = make_line_ribbon(
                        (a.0, a.1),
                        (b.0, b.1),
                        3.4 / height,
                        aspect_f32,
                        ribbon_col,
                    ) {
                        quads.push(DepthGeom::Quad {
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
                let p_world = sample.scene_position_with_offsets(
                    catalog,
                    days_since_j2000,
                    comet_perihelion_offsets,
                );
                let proj = camera.project(p_world, aspect);
                if let (Some(a), Some(b)) = (prev_pt, proj) {
                    if let Some(verts) = make_line_ribbon(
                        (a.0, a.1),
                        (b.0, b.1),
                        2.4 / height,
                        aspect_f32,
                        [0.25, 0.74, 0.98, 0.58],
                    ) {
                        quads.push(DepthGeom::Quad {
                            depth: 0.5 * (a.2 + b.2),
                            verts,
                        });
                    }
                }
                prev_pt = proj;
            }
        }

        let probe_world = probe.scene_position_with_offsets(
            catalog,
            days_since_j2000,
            comet_perihelion_offsets,
        );
        let mut ahead = probe.clone();
        if ahead.phase == ProbePhase::EnRoute {
            ahead.transit_progress = (ahead.transit_progress + 0.025).min(1.0);
        } else {
            ahead.orbit_phase_rad = (ahead.orbit_phase_rad + 0.08).rem_euclid(TAU);
        }
        let ahead_world = ahead.scene_position_with_offsets(
            catalog,
            days_since_j2000,
            comet_perihelion_offsets,
        );
        let fwd = ahead_world.sub(probe_world).normalized();
        let mut right_axis = fwd.cross(Vec3d::new(0.0, 1.0, 0.0));
        if right_axis.length() < 1e-6 {
            right_axis = fwd.cross(Vec3d::new(0.0, 0.0, 1.0));
        }
        let right_axis = right_axis.normalized();
        let up_axis = right_axis.cross(fwd).normalized();

        if let Some((nx, ny, depth)) = camera.project(probe_world, aspect) {
            if nx.abs() <= 1.25 && ny.abs() <= 1.25 {
                let proj_scale = ((1.0 / (camera.fov_y_rad * 0.5).tan()) / depth) as f32;
                let craft_sc = if probe.phase == ProbePhase::EnRoute {
                    0.032
                } else {
                    0.018
                };
                let desired_ndc = (craft_sc as f32 * proj_scale).clamp(0.018, 0.24);
                let model_sc = (desired_ndc / proj_scale) as f64;
                build_space_probe_mesh(
                    probe,
                    probe_world,
                    fwd,
                    right_axis,
                    up_axis,
                    camera,
                    aspect,
                    model_sc,
                    anim_time_sec,
                    &mut quads,
                );

                let sx = rect.left() + (nx * 0.5 + 0.5) * width;
                let sy = rect.top() + (0.5 - ny * 0.5) * height;
                labels.push(ProjectedLabel {
                    body_id: probe.target_id.clone(),
                    text: format!("Probe {}", probe.callsign),
                    screen_pos: egui::pos2(sx, sy - (desired_ndc * height * 0.55) - 8.0),
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
                let w = sample.scene_position_with_offsets(
                    catalog,
                    days_since_j2000,
                    comet_perihelion_offsets,
                );
                let proj = camera.project(w, aspect);
                if let (Some(a), Some(b)) = (prev_pt, proj) {
                    if let Some(verts) = make_line_ribbon(
                        (a.0, a.1),
                        (b.0, b.1),
                        3.2 / height,
                        aspect_f32,
                        [1.0, 0.38, 0.18, 0.92],
                    ) {
                        quads.push(DepthGeom::Quad {
                            depth: 0.5 * (a.2 + b.2),
                            verts,
                        });
                    }
                }
                prev_pt = proj;
            }
            let w_cur = ap.scene_position_with_offsets(
                catalog,
                days_since_j2000,
                comet_perihelion_offsets,
            );
            if let Some((ax, ay, a_depth)) = camera.project(w_cur, aspect) {
                let sz = (10.5 / height).clamp(0.01, 0.055);
                let glow_col = if ap.phase == EntryProbePhase::HypersonicEntry {
                    [1.0, 0.45, 0.15, 1.0]
                } else {
                    [0.95, 0.88, 0.65, 0.96]
                };
                quads.push(DepthGeom::Quad {
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
                        quads.push(DepthGeom::Quad {
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
                quads.push(DepthGeom::Quad {
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

    quads.sort_by(|a, b| b.depth().total_cmp(&a.depth()));

    let mut vertex_bytes = Vec::with_capacity(quads.len() * 6 * (GpuVertex::BYTE_STRIDE as usize));
    let mut vertex_count = 0u32;
    for q in &quads {
        match q {
            DepthGeom::Quad { verts, .. } => {
                for v in verts {
                    v.write_le_bytes(&mut vertex_bytes);
                }
                vertex_count += 6;
            }
            DepthGeom::Tri { verts, .. } => {
                for v in verts {
                    v.write_le_bytes(&mut vertex_bytes);
                }
                vertex_count += 3;
            }
        }
    }

    SceneFrameBuild {
        vertex_bytes,
        vertex_count,
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
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 40,
                shader_location: 4,
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
