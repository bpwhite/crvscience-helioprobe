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

// HelioProbe 3D WGSL Shader (`shaders/cosmos.wgsl`)
// Renders sunlit 3D spheres with atmospheric rim scatter, solar coronae, multi-filament volumetric nebulae,
// procedural planetary rings, orbital ribbons, and active 5-band probe scanner waves.

struct VertexInput {
    @location(0) clip_pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) kind_param: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) kind_param: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.clip_pos.x, in.clip_pos.y, 0.0, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    out.kind_param = in.kind_param;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let kind = i32(round(in.kind_param.x));
    let phase = in.kind_param.y;
    let p = in.uv;
    let r2 = dot(p, p);
    let r = sqrt(r2);
    let safe_ang = atan2(p.y, p.x + 0.00001);

    // 0: Solar / Star Corona + Photosphere
    if (kind == 0) {
        if (r > 1.0) {
            discard;
        }
        let core = smoothstep(0.42, 0.12, r);
        let corona = exp(-r * 3.4) * (0.85 + 0.15 * sin(safe_ang * 8.0 + phase * 3.0));
        let col = mix(in.color.rgb, vec3<f32>(1.0, 0.98, 0.90), core);
        let alpha = clamp(core + corona * 0.85, 0.0, 1.0) * in.color.a;
        return vec4<f32>(col, alpha);
    }

    // 1: 3D Lit Planetary / Lunar / Asteroid Sphere
    if (kind == 1) {
        if (r > 1.0) {
            discard;
        }
        let nz = sqrt(max(0.0, 1.0 - r2));
        let normal = normalize(vec3<f32>(p.x, p.y, nz + 0.0001));
        let light_dir = normalize(vec3<f32>(cos(phase), 0.28, sin(phase) * 0.5 + 0.55));
        let ndotl = max(dot(normal, light_dir), 0.0);
        let bands = 0.93 + 0.07 * sin(p.y * 14.0 + p.x * 2.5);
        let one_minus_nz = max(0.0, 1.0 - nz);
        let rim = one_minus_nz * one_minus_nz * one_minus_nz * 0.35;
        let lit = in.color.rgb * bands * (0.16 + 0.88 * ndotl) + in.color.rgb * rim;
        let edge_aa = smoothstep(1.0, 0.94, r);
        return vec4<f32>(lit, edge_aa * in.color.a);
    }

    // 2: Volumetric Deep-Sky Nebula Cloud
    if (kind == 2) {
        if (r > 1.0) {
            discard;
        }
        let swirl = 0.5 + 0.5 * sin(safe_ang * 3.0 + r * 6.0 - phase) * cos(safe_ang * 5.0 - r * 4.0);
        let shell = exp(-r2 * 2.6) * (0.55 + 0.65 * swirl);
        let core_glow = exp(-r2 * 14.0) * 0.55;
        let filament_col = mix(in.color.rgb, vec3<f32>(0.96, 0.88, 1.0), core_glow);
        return vec4<f32>(filament_col, clamp(shell + core_glow, 0.0, 0.88) * in.color.a);
    }

    // 3: Planetary Ring / Nebula Torus SDF
    if (kind == 3) {
        let q = vec2<f32>(p.x, p.y * 2.35);
        let rr = length(q);
        if (rr < 0.46 || rr > 1.0) {
            discard;
        }
        let cassini_gap = smoothstep(0.015, 0.04, abs(rr - 0.74));
        let fine_ringlets = 0.8 + 0.2 * sin(rr * 65.0);
        let fade = smoothstep(0.46, 0.52, rr) * smoothstep(1.0, 0.92, rr);
        return vec4<f32>(in.color.rgb * fine_ringlets, fade * cassini_gap * in.color.a * 0.78);
    }

    // 4: Solid / Ribbon Geometry (Orbits, Trajectories, Probe Bus Hull)
    if (kind == 4) {
        let aa = 1.0 - abs(p.y);
        return vec4<f32>(in.color.rgb, clamp(aa * 1.4, 0.0, 1.0) * in.color.a);
    }

    // 5: Active 5-Band Orbital Scanner Pulse Wave
    if (r > 1.0) {
        discard;
    }
    let wave = fract(r * 3.0 - phase * 1.8);
    let ring = smoothstep(0.18, 0.0, abs(wave - 0.5)) * (1.0 - r);
    return vec4<f32>(in.color.rgb, ring * in.color.a);
}
