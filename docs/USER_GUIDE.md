# HelioProbe v2.3.0 — User Guide

**License:** GNU General Public License v3.0 or later (`GPL-3.0-or-later`, see [LICENSE](../LICENSE))  
**Website:** [CRV Science Network](https://crvscience.com) · [CRV Science Public Apps](https://crvscience.com/apps)

---

## 1. Installation & Building

HelioProbe is a standalone pure-Rust application (`#![forbid(unsafe_code)]` in the core library, zero runtime dependencies outside Cargo crates) using `eframe`/`egui` (`0.31`), `wgpu`, bundled `rusqlite`, and `ureq`.

### Prerequisites
- **Rust toolchain** (`1.78+` recommended) via [rustup](https://rustup.rs/)
- **Graphics driver** supporting Vulkan, DirectX 12, Metal, or OpenGL/ANGLE (`wgpu` selects the best available adapter automatically)
- On Linux desktop builds, standard X11/Wayland development headers (`libxkbcommon-dev`, `libwayland-dev`, `libgtk-3-dev` for native file dialogs via `rfd`)

### Build & Run
```bash
# Launch the interactive 3D simulator (release build)
cargo run --release

# Launch against a specific SQLite WAL campaign file
cargo run --release -- --db my_campaign.db

# Run the complete unit and integration test suite
cargo test --all-targets
```

---

## 2. GUI Layout & Mission Workflow

The HelioProbe interface is organized into five coordinated panels:

1. **Top Mission Bar (`top_mission_bar`)**:
   - Displays lifetime **Science Points (`SP`)**, spendable **Lab Available SP**, **Agency Rank** (5 ranks from *Cadet Flight Controller* to *Interstellar Mission Director*), and unlocked **Awards (`X/21`)**.
   - **Pause / Resume Sim** (`Space` when no text input is focused).
   - **Orbit Clock (d/s)** logarithmic slider (`0.002` to `1.0` Julian days per real-time second): advances planetary, lunar, Sun–Earth L2 halo, and cometary Keplerian motion.
   - **Probe Transit Warp** slider (`0.5x` to `10.0x`): accelerates visual animation speed of spacecraft transits, atmospheric plunges, and sample-return capsules without altering physical Δv or flight-day calculations.
   - **3D Orbits** (`O`) and **Labels** (`L`) toggles, plus **Quick Save (Slot 1)** (`F5`).

2. **Left Mission Launch & Target Selector Dock (`left_launch_dock`)**:
   - **1. Celestial Target Catalog**: Search by name, ID, or designation (`Ctrl+F` focuses the search box), filter by `BodyCategory` (`Star`, `Planet`, `Dwarf Planet`, `Moon`, `Asteroid`, `Comet`, `Sun-Earth Lagrange Point`, `Exoplanet`, `Nebula`), and see `[ORBIT N/5]` scan progress badges.
   - **2. Earth LEO Probe Launcher**: Computes live Hohmann / relativistic Δv (`km/s`), capture periapsis velocity & altitude, and optional **Patched-Conic Gravity-Assist Slingshot** (`Venus`, `Mars`, `Jupiter`, `Saturn`). For interstellar exoplanets and nebulae, displays both Earth-frame transit duration (`yr`), ship proper time `τ = t / γ`, and Lorentz factor `γ`.
   - **3. Camera & Active Fleet**: One-click switching across the 5 camera tracking modes (`1`–`5`) and active spacecraft roster. Clicking a probe in the fleet automatically tracks its flight or capture orbit.

3. **Center 3D Viewport**:
   - Hardware-accelerated WGSL shader rendering (`shaders/cosmos.wgsl`) with true body-to-Sun camera-space planetary illumination, 3D obliquity-tilted planetary rings with planet shadow attenuation, uniform eccentric-anomaly (`E`) orbital ribbons, Sun–Earth L2 halo telescopes, dynamic cometary C2 comae + solar-wind aberrated Type I CO+ ion tails + Type II curved dust fans, polyhedral spacecraft buses, and grounded surface/cryobot cutaway geometry.
   - Live **Lander / Cryobot / Entry POV Telemetry HUD** appears in the upper-right viewport corner whenever Mode `5` (`Lander / Bot POV`) is active.

4. **Right 8-Tab Science & Engineering Console (`right_science_dock`)**:
   - **Scanner**: Inspect bulk metrics, agency source links, and run 5-band instrument telemetry (`Optical/NIR`, `Radar/Gravity`, `Magnetometer/Plasma`, `Thermal/Mass-Spec`, `UV/X-Ray`).
   - **L2 & Comets**: Deploy **JWST**, **Nancy Grace Roman**, and **ESA/JAXA Comet Interceptor** into Sun–Earth L2 halo orbits (~1.5M km anti-sunward), view physical Hill radius and CR3BP halo periods (`T_xy`, `T_z`), trigger Roman's 100x-FOV Wide-Field Sky Survey, warp comets to perihelion, inspect live solar-wind ion-tail aberration angles, and launch rapid 2x-speed L2 comet intercepts.
   - **Windows**: Interactive **3D Universal-Variable Lambert Porkchop Plot** (memoized by target, quantized epoch, and comet offset) featuring a perceptually uniform Viridis colormap, marching-squares iso-Δv contour lines, numeric X/Y departure & TOF ticks, hover UTC/C3 tooltips, click-to-schedule departure windows, 1-click ephemeris snapping, and Gravity-Assist Slingshot arming.
   - **Landers**: Deploy **Aero-Shell / Sky-Crane Surface Landers**, **Pu-238 GPHS Ocean-World Thermal Cryobots** (across 12 icy ocean worlds), **Galileo-Class Atmospheric Entry / Kinetic Crash Probes** (across 14 atmospheric worlds plus airless targets), and **Earth Sample-Return Capsules (`SRC`)**.
   - **Tech Lab**: Research 8 upgrades across 3 engineering branches (`Propulsion & Astrodynamics`, `Orbital & Cryobot Payloads`, `Deep Space Network & Curatory`) using `available_sp = science_points - spent_sp` so lifetime rank never drops.
   - **Awards**: Track all **21 Scientific Achievements**.
   - **Public Data**: Non-blocking background harvesters for **NASA JPL SBDB**, **NASA Exoplanet Archive TAP**, **CDS SIMBAD TAP**, and **NASA JPL Horizons**.
   - **Saves**: Manage 3 SQLite WAL campaign slots (`Open / Switch DB...`, `Save DB As...`) and portable JSON campaign export/import.

5. **Bottom Status & Flight Director Log Bar**:
   - Opens the color-coded **Flight Director Telemetry & Event Log** drawer and **About HelioProbe & Mission Reference** modal (`F1`).

---

## 3. Interactive Controls & Hotkeys

All single-key and navigation hotkeys (except `F1` and `Esc`) are guarded by `!ctx.wants_keyboard_input()` so typing in search bars, probe callsigns, or API inputs is never interrupted.

| Input / Hotkey | Action |
| :--- | :--- |
| **Left-Drag (3D Viewport)** | Orbit camera around current target (`yaw` and `pitch`) |
| **Mouse Scroll Wheel** | Smooth exponentially damped zoom (`0.015` to `260.0` scene units) |
| **Left-Click Body (3D)** | Ray-pick and select celestial body in Catalog & Scanner |
| **`1`** | Camera Mode: **Solar Overview** (`FreeSolarOverview`) |
| **`2`** | Camera Mode: **Lock Target** (`LockSelectedBody`) |
| **`3`** | Camera Mode: **Orbit View** (`OrbitView`, auto-orbits target) |
| **`4`** | Camera Mode: **Ride Probe** (`RideLatestProbe`) |
| **`5`** | Camera Mode: **Lander / Bot POV** (`LanderBotView`, surface / cryobot cutaway / atmo plunge) |
| **`R`** | Reset camera to default **Solar Overview** (`yaw=0.65`, `pitch=0.46`, `dist=78.0`) |
| **`O`** | Toggle 3D Keplerian orbital ribbons (`show_orbits`) |
| **`L`** | Toggle de-overlapped 3D body & probe labels (`show_labels`) |
| **`Ctrl+F`** (or `Cmd+F`) | Focus **Celestial Target Catalog** search bar |
| **`Space`** | Pause / Resume ephemeris and probe simulation clock |
| **`F5`** | Quick Save campaign to SQLite Slot 1 |
| **`F1`** | Toggle **About HelioProbe & Mission Reference** modal |
| **`Esc`** | Close active modal (`About` first, then `Flight Director Log`) |

---

## 4. CLI Flags & Subcommands Reference

All CLI operations are defined in `src/main.rs`:

| Flag / Subcommand | Default / Arguments | Description |
| :--- | :--- | :--- |
| `--db <PATH>` | `helioprobe_saves.db` | Path to SQLite WAL campaign database |
| `--headless-screenshot <PATH>` | *(optional)* | Render an offscreen `1600x1000` RGBA PNG without opening a GUI window |
| `--screenshot-view <1\|2\|3>` | `1` | Select preset camera/scene for `--headless-screenshot`: `1` Solar System + Mars Transit, `2` Sun–Earth L2 Halo + Halley Perihelion, `3` Europa Grounded Cryobot Cutaway POV |
| `sync-sbdb <QUERY>` | e.g. `"Eros"` | Query NASA JPL Small-Body Database (SBDB) API and print JSON `CelestialBody` |
| `sync-exoplanet <NAME>` | e.g. `"Kepler-186 f"` | Query NASA Exoplanet Archive TAP service and print JSON `CelestialBody` |
| `sync-simbad <IDENTIFIER>` | e.g. `"M 8"` | Query CDS SIMBAD TAP service and print JSON `CelestialBody` |
| `porkchop [TARGET_ID]` | `saturn` | Print synodic period, phase error, optimal 3D Lambert Porkchop Δv basin, and feasible Gravity-Assist Slingshot routes (`Earth -> Venus/Mars/Jupiter/Saturn -> Target`) |
| `slots` | — | List saved campaign slots in the specified `--db` SQLite WAL database |

### CLI Examples
```bash
# Compute Saturn Porkchop window and Jupiter slingshot savings
cargo run --release -- porkchop saturn

# Render headless validation PNGs without opening a window
cargo run --release -- --headless-screenshot view1.png --screenshot-view 1
cargo run --release -- --headless-screenshot view2.png --screenshot-view 2
cargo run --release -- --headless-screenshot view3.png --screenshot-view 3

# Live NASA JPL Small-Body Database query (requires internet)
cargo run --release -- sync-sbdb "433 Eros"
```

---

## 5. Troubleshooting

- **Offscreen / Headless GPU Adapter Errors**: `--headless-screenshot` requires a valid `wgpu` backend (Vulkan, DX12, Metal, or Mesa `lavapipe` software adapter on headless Linux CI). Ensure graphics drivers or `mesa-vulkan-drivers` are installed.
- **SQLite WAL Sidecar Files (`-wal`, `-shm`)**: SQLite uses Write-Ahead Logging (`PRAGMA journal_mode = WAL`) for safe concurrent reads/writes. Close the application before copying only the `.db` file, or use **Saves -> Export JSON...** for single-file portable backups.
- **Live Public Data Timeouts**: `sync-sbdb`, `sync-exoplanet`, `sync-simbad`, and GUI **Public Data** harvesters query external NASA/CDS APIs with a 12-second timeout. Offline environments or rate-limited networks surface explicit error messages in the status bar without blocking the GUI thread.
- **Hotkeys Not Firing**: Single-key shortcuts (`1`–`5`, `O`, `L`, `R`, `Space`, `F5`) are intentionally suppressed while any `TextEdit` widget has keyboard focus. Click empty viewport space or press `Tab` / click outside text fields to return focus to the 3D scene.
