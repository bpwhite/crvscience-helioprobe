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

//! Embedded procedural RGBA window icon (`ViewportBuilder::with_icon`) matching `assets/helioprobe.ico`.

use eframe::egui;

fn blend_pixel(buf: &mut [u8], size: usize, x: i32, y: i32, rgba: [f32; 4]) {
    if x < 0 || y < 0 || x >= size as i32 || y >= size as i32 {
        return;
    }
    let idx = ((y as usize) * size + (x as usize)) * 4;
    let a = rgba[3].clamp(0.0, 1.0);
    let inv = 1.0 - a;
    buf[idx] = ((buf[idx] as f32 * inv + rgba[0] * 255.0 * a).clamp(0.0, 255.0)) as u8;
    buf[idx + 1] = ((buf[idx + 1] as f32 * inv + rgba[1] * 255.0 * a).clamp(0.0, 255.0)) as u8;
    buf[idx + 2] = ((buf[idx + 2] as f32 * inv + rgba[2] * 255.0 * a).clamp(0.0, 255.0)) as u8;
    buf[idx + 3] = ((buf[idx + 3] as f32 * inv + 255.0 * a).clamp(0.0, 255.0)) as u8;
}

/// Renders an anti-aliased `size x size` RGBA HelioProbe badge (deep-space rounded tile,
/// golden solar corona, cyan Keplerian arc, gold-MLI probe with solar wings + parabolic dish).
pub fn render_helioprobe_rgba(size: usize) -> Vec<u8> {
    let mut rgba = vec![0u8; size * size * 4];
    let s = size as f32;

    for y in 0..size {
        for x in 0..size {
            let u = (x as f32 + 0.5) / s;
            let v = (y as f32 + 0.5) / s;

            // Rounded-square badge SDF
            let qx = (u - 0.5).abs() - 0.43;
            let qy = (v - 0.5).abs() - 0.43;
            let outside = qx.max(0.0).hypot(qy.max(0.0)) + qx.max(qy).min(0.0) - 0.06;
            let tile_alpha = (1.0 - (outside * s * 0.9 + 0.5)).clamp(0.0, 1.0);
            if tile_alpha <= 0.001 {
                continue;
            }

            // Deep-space radial indigo background
            let bg_r = 0.035 + 0.03 * (1.0 - v);
            let bg_g = 0.055 + 0.04 * (1.0 - u);
            let bg_b = 0.125 + 0.09 * (1.0 - v);
            blend_pixel(&mut rgba, size, x as i32, y as i32, [bg_r, bg_g, bg_b, tile_alpha]);

            // Subtle border ring
            let rim = (1.0 - (outside.abs() * s * 0.9)).clamp(0.0, 1.0) * 0.55;
            if rim > 0.01 {
                blend_pixel(&mut rgba, size, x as i32, y as i32, [0.22, 0.72, 0.95, rim * tile_alpha]);
            }

            // Golden Sun + Corona at upper-left (0.28, 0.30)
            let dx_sun = u - 0.28;
            let dy_sun = v - 0.30;
            let d_sun = dx_sun.hypot(dy_sun);
            let corona = (-d_sun * 7.5).exp() * 0.85;
            if corona > 0.01 {
                blend_pixel(
                    &mut rgba,
                    size,
                    x as i32,
                    y as i32,
                    [1.0, 0.76, 0.24, corona * tile_alpha],
                );
            }
            if d_sun < 0.13 {
                let core = (1.0 - d_sun / 0.13).clamp(0.0, 1.0);
                blend_pixel(
                    &mut rgba,
                    size,
                    x as i32,
                    y as i32,
                    [1.0, 0.94 + 0.06 * core, 0.65 + 0.32 * core, tile_alpha],
                );
            }

            // Sweeping cyan Keplerian orbital arc centered near Sun
            let dx_orb = (u - 0.30) * 0.85;
            let dy_orb = (v - 0.34) * 1.18;
            let r_orb = dx_orb.hypot(dy_orb);
            let arc_dist = (r_orb - 0.42).abs() * s;
            if arc_dist < 1.9 && (u > 0.18 || v > 0.36) {
                let aa = (1.0 - arc_dist / 1.9).clamp(0.0, 1.0) * 0.92;
                blend_pixel(&mut rgba, size, x as i32, y as i32, [0.20, 0.88, 1.0, aa * tile_alpha]);
            }

            // Comet / L2 ion streak in upper-right (0.76, 0.24)
            let dx_c = u - 0.76;
            let dy_c = v - 0.23;
            let d_c = dx_c.hypot(dy_c);
            if d_c < 0.055 {
                let a = (1.0 - d_c / 0.055) * 0.95;
                blend_pixel(&mut rgba, size, x as i32, y as i32, [0.35, 0.98, 0.86, a * tile_alpha]);
            }
            if u > 0.76 && v < 0.24 {
                let tail_err = ((v - 0.23) + (u - 0.76) * 0.42).abs() * s;
                let tail_len = (u - 0.76) * 5.5;
                if tail_err < 1.6 && tail_len < 1.0 {
                    let a = (1.0 - tail_err / 1.6) * (1.0 - tail_len) * 0.75;
                    blend_pixel(&mut rgba, size, x as i32, y as i32, [0.35, 0.78, 1.0, a * tile_alpha]);
                }
            }

            // Space Probe silhouette at (0.60, 0.60) tilted 35 deg
            let px = u - 0.60;
            let py = v - 0.60;
            let cos_t = 0.819_f32;
            let sin_t = 0.574_f32;
            let lx = px * cos_t + py * sin_t;
            let ly = -px * sin_t + py * cos_t;

            // Blue solar wings along lx
            if lx.abs() >= 0.08 && lx.abs() <= 0.27 && ly.abs() <= 0.048 {
                let cell_line = ((lx * 42.0).sin().abs() * 0.25 + 0.75).clamp(0.0, 1.0);
                blend_pixel(
                    &mut rgba,
                    size,
                    x as i32,
                    y as i32,
                    [0.14 * cell_line, 0.56 * cell_line, 0.98, tile_alpha],
                );
            }
            // Gold-MLI central bus
            if lx.abs() <= 0.075 && ly.abs() <= 0.068 {
                blend_pixel(&mut rgba, size, x as i32, y as i32, [0.96, 0.74, 0.18, tile_alpha]);
            }
            // White parabolic High-Gain Dish on +ly side
            let dish_r = (lx * 1.15).hypot(ly + 0.085);
            if dish_r <= 0.085 && ly < -0.04 {
                blend_pixel(&mut rgba, size, x as i32, y as i32, [0.93, 0.96, 1.0, tile_alpha]);
            }
        }
    }

    rgba
}

/// Returns the 64x64 RGBA `egui::IconData` for `ViewportBuilder::with_icon(...)`.
pub fn helioprobe_window_icon() -> egui::IconData {
    let size = 64u32;
    egui::IconData {
        rgba: render_helioprobe_rgba(size as usize),
        width: size,
        height: size,
    }
}
