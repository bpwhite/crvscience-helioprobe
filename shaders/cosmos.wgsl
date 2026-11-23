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
    @location(4) extra: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) kind_param: vec2<f32>,
    @location(3) extra: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.clip_pos.x, in.clip_pos.y, 0.0, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    out.kind_param = in.kind_param;
    out.extra = in.extra;
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

    // 1: 3D Lit Planetary / Lunar / Asteroid Sphere (body-to-Sun vector in camera view space)
    if (kind == 1) {
        if (r > 1.0) {
            discard;
        }
        let nz = sqrt(max(0.0, 1.0 - r2));
        let normal = normalize(vec3<f32>(p.x, p.y, nz + 0.0001));
        let light_len = length(in.extra.xyz);
        let light_dir = select(
            normalize(vec3<f32>(cos(phase), 0.28, sin(phase) * 0.5 + 0.55)),
            in.extra.xyz / light_len,
            light_len > 0.01
        );
        let ndotl = max(dot(normal, light_dir), 0.0);
        let bands = 0.93 + 0.07 * sin(p.y * 14.0 + p.x * 2.5);
        let one_minus_nz = max(0.0, 1.0 - nz);
        let rim = one_minus_nz * one_minus_nz * one_minus_nz * 0.34 * (0.22 + 0.78 * ndotl);
        let lit = in.color.rgb * bands * (0.08 + 0.92 * ndotl) + in.color.rgb * rim;
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

    // 3: 3D Obliquity-Tilted Planetary Ring Annulus
    if (kind == 3) {
        let rr = length(p);
        let r_in = clamp(in.extra.x, 0.35, 0.90);
        if (rr < r_in || rr > 1.0) {
            discard;
        }
        let cassini_pos = mix(r_in, 1.0, 0.56);
        let cassini_gap = mix(1.0, smoothstep(0.010, 0.032, abs(rr - cassini_pos)), in.extra.z);
        let fine_ringlets = 0.82 + 0.18 * sin(rr * 72.0);
        let fade = smoothstep(r_in, r_in + 0.04, rr) * smoothstep(1.0, 0.94, rr);
        let sun_lit = clamp(in.extra.y, 0.18, 1.15);
        return vec4<f32>(in.color.rgb * fine_ringlets * sun_lit, fade * cassini_gap * in.color.a);
    }

    // 4: Solid / Ribbon Geometry (Orbits, Trajectories, Probe Bus Hull)
    if (kind == 4) {
        let aa = 1.0 - abs(p.y);
        return vec4<f32>(in.color.rgb, clamp(aa * 1.4, 0.0, 1.0) * in.color.a);
    }

    // 6: Shaded 3D Polyhedral Mesh Triangle (pre-lit Lambert + Blinn-Phong solar specular + panel bevel)
    if (kind == 6) {
        let edge_x = min(in.uv.x, 1.0 - in.uv.x);
        let edge_y = min(in.uv.y, 1.0 - in.uv.y);
        let bevel = smoothstep(0.0, 0.075, min(edge_x, edge_y));
        let panel_shade = mix(0.84, 1.03, bevel);
        let sheen = max(phase, 0.0) * (0.25 + 0.75 * bevel);
        let rgb = clamp(
            in.color.rgb * panel_shade + vec3<f32>(sheen, sheen * 0.96, sheen * 0.88),
            vec3<f32>(0.0),
            vec3<f32>(1.0)
        );
        return vec4<f32>(rgb, in.color.a);
    }

    // 5: Active 5-Band Orbital Scanner Pulse Wave
    if (r > 1.0) {
        discard;
    }
    let wave = fract(r * 3.0 - phase * 1.8);
    let ring = smoothstep(0.18, 0.0, abs(wave - 0.5)) * (1.0 - r);
    return vec4<f32>(in.color.rgb, ring * in.color.a);
}
