# HelioProbe (`helioprobe` v2.3.0)

![HelioProbe 3D Solar System & Spacecraft Orbit View](validation/headless_screenshot.png)

Native safe-Rust (`#![forbid(unsafe_code)]`) 3D graphical space probe, orbital mechanics, gravity-assist planner, surface lander & ocean-world cryobot POV, atmospheric entry crash probe, sample-return, and public astronomy simulation game built with `eframe`/`egui` (`0.31`), `wgpu` (`24.0` WGSL custom 3D shader pipeline), `rusqlite` (bundled SQLite WAL persistence), and `ureq` live astronomical harvesters.

## Core Architecture (10 Safe-Rust Modules)

- **Scientifically Grounded 41-Target Bootstrap Catalog (`src/catalog.rs`)**:
  - Pre-loaded with metric bulk and orbital parameters, monotonic physical radius scaling, and 5-band instrument telemetry across **41 real targets** (205 verified fact cards), spanning the [Sun](https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html), all 8 planets ([NASA Planetary Fact Sheet](https://nssdc.gsfc.nasa.gov/planetary/factsheet/)), all 5 IAU dwarf planets, 10 moons, 4 asteroids, 5 exoplanets, and 8 deep-sky nebulae.
- **Gravity-Assist Slingshot & Porkchop Launch-Window Planner (`src/astrodynamics_lab.rs`)**:
  - Computes synodic periods, planetary phase-angle offsets, 2D Lambert-approximated Porkchop $\Delta v$ contour heatmaps (`departure_offset_days` $\times$ `flight_days`) with 1-click epoch snapping, and 2-leg patched-conic Gravity-Assist Slingshot trajectories (`Earth -> Venus/Mars/Jupiter/Saturn -> Target`).
- **Surface Landers, Ocean Cryobots, Galileo Atmospheric Crash Probes & `6. Lander / Bot POV` (`src/surface_ops.rs`, `src/gui.rs`)**:
  - **Aero-Shell / Sky-Crane Surface Landers**: Soft-land on rocky/icy crusts to run 3-axis broadband seismometry, active core-drill X-ray mineralogy, and surface heat-flow telemetry.
  - **Pu-238 GPHS Ocean-World Thermal Cryobots**: Melt `4.2–40.0 km` through refreezing icy crusts on 12 icy ocean worlds ([Europa](https://science.nasa.gov/jupiter/jupiter-moons/europa/europa-facts/), [Enceladus](https://science.nasa.gov/saturn/moons/enceladus/), [Titan](https://science.nasa.gov/saturn/moons/titan/facts/), etc.), spooling wireless acoustic/RF ice-relay pucks and pinging deep-ocean sonar.
  - **Galileo / Cassini Atmospheric Entry & Kinetic Crash Probes**: Drop carbon-phenolic heat-shielded entry cones from orbit into gas/ice giants ([Galileo Probe Entry Timeline](https://nssdc.gsfc.nasa.gov/planetary/galileo_probe_time.html), [Cassini Grand Finale](https://science.nasa.gov/mission/cassini/grand-finale/overview/)) or hypervelocity kinetic penetrators ([Deep Space 2](https://nssdc.gsfc.nasa.gov/nmc/spacecraft/display.action?id=DEEPSP2)) onto airless worlds, streaming live pressure/temperature/wind telemetry down to crush-depth or impact loss-of-signal.
  - **`6. Lander / Bot POV` Camera**: Switch directly to surface-horizon POV or ride plunging atmospheric entry cones live down into the cloud decks.
  - **Earth Sample-Return Capsules (`SRC`)**: Seal regolith or plume condensates and fly a 3D return trajectory back to Earth splashdown.
- **Science Point (`SP`) Spacecraft Tech Tree & Payload Lab (`src/tech_tree.rs`)**:
  - Research **8 engineering & instrument upgrades** using `available_sp = science_points - spent_sp` so unlocking upgrades never lowers lifetime Agency Rank.
- **Live Public Astronomical Data Harvesters (`src/public_data.rs`)**:
  - [NASA JPL Small-Body Database (SBDB) API](https://ssd-api.jpl.nasa.gov/doc/sbdb.html), [NASA JPL Horizons API](https://ssd-api.jpl.nasa.gov/doc/horizons.html), [NASA Exoplanet Archive TAP Service](https://caltech-ipac.github.io/nexsciTAP/intro.html), and [CDS SIMBAD TAP Service](https://scispace.com/ai-agent-seo/page-files/simbad-api-guide-sid4li15.pdf).
- **19 Achievements & SQLite WAL Persistence (`src/gamification.rs`)**:
  - **19 Scientific Achievements** and **3 SQLite WAL slots** (`helioprobe_saves.db`) + portable JSON export/import.

## Build & Run

```bash
cargo run --release
cargo test
cargo run --release -- porkchop saturn
```

## What's New in v2.3.0 (Sun-Earth L2 Telescopes & Dynamic Comets)
- **Sun–Earth L2 Halo Space Telescopes (`src/l2_comets.rs`)**: Co-rotating collinear Lagrange point (`~1.5 million km` / `0.0100 AU` anti-sunward of Earth) with 3D Lissajous quasi-halo station-keeping for the **James Webb Space Telescope (`JWST`)**, **Nancy Grace Roman Space Telescope** (`100x` Hubble IR FOV survey), and **ESA/JAXA Comet Interceptor** standby triad (enabling 2x-speed direct L2 comet intercepts).
- **Eccentric Comets with Dynamic Solar-Wind Ion & Curved Dust Tails**: Added **2P/Encke**, **67P/Churyumov–Gerasimenko**, **1P/Halley** (`e = 0.9672`, retrograde `162.2°`), and **C/1995 O1 (Hale–Bopp)** (`e = 0.9951`, near-polar `89.43°`), bringing the offline catalog to **46 verified NASA/ESA/Hubble/Webb entities** (`230` five-band findings) alongside safeguarded high-eccentricity Kepler propagation, heliocentric C2 coma scaling, solar-wind aberrated Type I CO+ plasma tails, and radiation-pressure curved Type II silicate dust fans.

