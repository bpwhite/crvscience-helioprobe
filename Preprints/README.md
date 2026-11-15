# Preprints for HelioProbe (`crvscience-helioprobe`)

**Author:** Bryan P. White (ORCID: [0000-0003-4191-3511](https://orcid.org/0000-0003-4191-3511))  
**Hub:** https://crvscience.com/apps

This directory contains the self-contained LaTeX source (`.tex`) and single-column PDF (`.pdf`) editions for both:

1. **Companion Application Preprint (`APP-PREPRINT-14`)**
   - **Title:** HelioProbe: Simulating Spacecraft Orbits, Deep-Space Telescopes, and Comet Tails in 3D
   - **Files:** `helioprobe-3d-solar-system-l2.tex`, `helioprobe-3d-solar-system-l2.pdf`
   - **Canonical Zenodo DOI:** https://doi.org/10.5281/zenodo.24244083
   - **Figshare Mirror:** https://figshare.com/articles/preprint/HelioProbe_Simulating_Spacecraft_Orbits_Deep-Space_Telescopes_and_Comet_Tails_in_3D/35013778
   - **Web Preprint:** https://crvscience.com/apps/preprints/helioprobe-3d-solar-system-l2

2. **CRV Science Public Apps Network Overview (`APP-PREPRINT-01`)**
   - **Title:** The CRV Science App Network: Seventeen Open-Source Rust Programs for Science, Data, and Simulation
   - **Files:** `crv-science-public-apps-network.tex`, `crv-science-public-apps-network.pdf`
   - **Canonical Zenodo DOI:** https://doi.org/10.5281/zenodo.24222398
   - **Figshare Mirror:** https://figshare.com/articles/preprint/The_CRV_Science_App_Network_Seventeen_Open-Source_Rust_Programs_for_Science_Data_and_Simulation/35013736
   - **Web Preprint:** https://crvscience.com/apps/preprints/crv-science-public-apps-network

## Compiling Standalone LaTeX

Both `.tex` files inline their bibliography so each compiles standalone with standard `pdflatex`:

```bash
pdflatex helioprobe-3d-solar-system-l2.tex
pdflatex crv-science-public-apps-network.tex
```
