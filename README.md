# Zenith: Ultra-Fast 2.5D Photogrammetry & Orthorectification Engine (v1.0.0)

[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](Cargo.toml)

**Zenith** is a high-performance, open-source photogrammetry engine built in pure Rust. It is engineered specifically for **time-critical aerial operations**—such as wildfire suppression, disaster response, and tactical search & rescue (SAR)—where standard Structure-from-Motion (SfM) pipelines are too slow.

By leveraging **direct 2.5D back-projection**, **continuous digital elevation models (DEM)**, **Brown-Conrady dynamic lens dewarping**, and **massive CPU multithreading with Rayon**, Zenith delivers orthomosaics and thermal hot-spot maps in seconds to minutes, rather than hours.

---

## ⚡ Key Highlights (v1.0)

- **Extreme Speed**: Processing throughput of **>370 Megapixels/second** in-memory compute on standard multicore CPUs.
- **Multi-CRS Geodetic Engine**: Built-in high-precision Transverse Mercator supporting Colombian Origen Nacional (`EPSG:9377`), global UTM north/south zones (`EPSG:32601`–`32660`, `EPSG:32701`–`32760`), European grids (`EPSG:25830`), and automatic zone selection (`--crs auto` / `--crs utm`).
- **3-Tier DEM Elevation Cascade**:
  1. *Tier 1 (High-Res Local)*: Custom GeoTIFF/ENVI DEMs cropped on the flight bounding box.
  2. *Tier 2 (Regional Satellite)*: Automated streaming of Copernicus GLO-30 (30m) from AWS Open Data with GDAL virtual file system (`/vsicurl/`).
  3. *Tier 3 (Contingency Plane)*: Automatic fallback to flight telemetry takeoff datum with MAD (Median Absolute Deviation) statistical outlier filtering.
- **Automated Laser Rangefinder (LRF) Datum Calibration**: Cross-validates UAV GNSS ellipsoidal height against real-time laser rangefinder measurements and terrain elevations to eliminate geoidal undulation offsets ($H = h - N$).
- **Brown-Conrady Lens Dewarping**: Dynamic extraction of DJI XMP `DewarpData` ($f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3$) with forward Brown-Conrady projection and sub-pixel fixed-point ray inversion.
- **Strict Radiometric Thermal Integrity**: Nearest-Neighbor sampling with narrow-band Voronoi feathering and hotspot protection (`--max-blend-temp-diff`), preventing synthetic blurring or temperature dilution of wildfire fronts.
- **Visual RGB Orthomosaics**: Micro-feathering (Top-3 softmin Voronoi-Band), global Brown-Lowe gain compensation across exposures, and atmospheric dehaze.
- **Contract-Compliant Manifests & Real-Time Telemetry**: Emits standard Level 1 JSON manifests and structured progress notifications (`[ZENITH_PROGRESS] stage=... pct=... detail=...`).
- **100% Pure Rust Workspace**: Memory-safe, zero garbage collection pauses, standalone static binary with zero C++ overhead.

---

## 🏗️ Operational Latency Tiers

Zenith complies with tactical emergency response operational tiers:

| Tier | Operation | Latency Target | Zenith Pipeline | Primary Deliverable |
|---|---|---|---|---|
| **Level 0** | Emergency Thermal Triage | **< 2 minutes** | Direct ray-DEM intersection of hot-spot centroids | Ranked hotspot vector GeoJSON |
| **Level 1** | Fast Tactical Orthomosaic | **< 3 minutes** | Direct 2.5D parallel back-projection with DEM & micro-feathering | High-resolution thermal/RGB GeoTIFF + ENVI |
| **Level 2** | Full Rigorous SfM | **30 – 60 minutes** | Asynchronous feature matching & bundle adjustment | 3D mesh, dense point cloud |

---

## 📐 Workspace Architecture

Zenith is organized as a modular Rust Cargo workspace:

```text
zenith/
├── crates/
│   ├── core/      # Vector3, Ray3D, pinhole & Brown-Conrady dewarp, TransverseMercator, ConvexHull2D
│   ├── dem/       # Continuous bilinear DEM sampling, bisection ray-marching & flat planes
│   ├── ortho/     # Direct 2.5D orthorectification engine with Rayon parallelism (thermal & RGB)
│   ├── io/        # DJI XMP metadata parser, geo.txt parser, and canonical v1.0 EngineManifest schema
│   └── cli/       # Zenith CLI binary with ortho-thermal, ortho-rgb, mission, benchmark & validate commands
├── tests/         # Golden cross-validation, DEM bisection precision, and DJI flight scan integration tests
├── Cargo.toml     # Root workspace configuration (v1.0.0)
├── CITATION.cff   # Citation metadata
└── LICENSE        # MIT License
```

---

## 🚀 Quick Start

### Prerequisites

- Rust 1.80+ (recommended: Rust 1.99+)
- Cargo toolchain
- GDAL tools (`gdalwarp`, `gdal_translate` for GeoTIFF generation and cloud DEM streaming)

### Building

```bash
cd zenith
cargo build --release
```

The optimized binary will be located at `target/release/zenith`.

---

## 💻 CLI Usage

### 1. Thermal Tactical Orthomosaic (Level 1)
```bash
zenith ortho-thermal \
  --images /path/to/flight_dji/ \
  --raw-dir /path/to/thermal_floats_celsius/ \
  --output output/thermal_orthomosaic.tif \
  --crs auto \
  --gsd 0.15 \
  --feather-px 6.0 \
  --max-blend-temp-diff 3.5 \
  --hull-geojson output/thermal_hull.geojson \
  --manifest output/manifest_level1.json
```

### 2. Visual RGB Direct Orthomosaic (Level 1 RGB)
```bash
zenith ortho-rgb \
  --images /path/to/flight_dji/ \
  --output output/rgb_orthomosaic.tif \
  --crs EPSG:9377 \
  --blend-mode micro-feather \
  --feather-px 25.0 \
  --gain-compensation \
  --hull-geojson output/rgb_hull.geojson \
  --manifest output/manifest_level1_rgb.json
```

### 3. Mission Footprint & Telemetry Analysis
```bash
zenith mission \
  --images /path/to/flight_dji/ \
  --dem /path/to/local_dtm.tif \
  --output output/footprints.geojson
```

### 4. High-Throughput Compute Benchmark
```bash
zenith benchmark --frames 100 --dem-size 1000 --gsd 0.15
```

### 5. Validate Photogrammetric Manifest
```bash
zenith validate-contract --manifest output/manifest_level1.json
```

---

## 🔬 Scientific & Mathematical Formulation

### 1. Transverse Mercator Forward Projection
Uses high-order series expansions for conformal mapping of longitude ($\lambda$) and latitude ($\phi$):
$$X = X_0 + k_0 N \left( A + (1 - T + C)\frac{A^3}{6} + (5 - 18T + T^2 + 72C - 58e'^2)\frac{A^5}{120} \right)$$
$$Y = Y_0 + k_0 \left( M + N\tan\phi \left( \frac{A^2}{2} + (5 - T + 9C + 4C^2)\frac{A^4}{24} + (61 - 58T + T^2 + 600C - 330e'^2)\frac{A^6}{720} \right) \right)$$
where $A = (\lambda - \lambda_0)\cos\phi$, $N = \frac{a}{\sqrt{1 - e^2\sin^2\phi}}$, and $M$ is the true meridional arc distance.

### 2. Brown-Conrady Distortion & Fixed-Point Inversion
Forward lens distortion converts undistorted normalized coordinates $(x_n, y_n)$ to distorted $(x_d, y_d)$:
$$r^2 = x_n^2 + y_n^2$$
$$x_d = x_n (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + [2 p_1 x_n y_n + p_2 (r^2 + 2 x_n^2)]$$
$$y_d = y_n (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + [p_1 (r^2 + 2 y_n^2) + 2 p_2 x_n y_n]$$

For pixel-to-ray back-projection, undistorted $(x_n, y_n)$ are reconstructed via Newton-Raphson fixed-point iteration until $\| (x_n^{(k+1)}, y_n^{(k+1)}) - (x_n^{(k)}, y_n^{(k)}) \| < 10^{-10}$.

### 3. Ray Marching & Terrain Bisection
Sub-centimeter terrain intersection $\vec{r}(s^*) = \text{DEM}(X, Y)$ is achieved through:
1. Coarse ray marching with bounding steps $\Delta s$.
2. Continuous binary bisection on $[s_{\text{prev}}, s_{\text{cross}}]$ for 16 iterations over the continuous bilinear elevation surface.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use Zenith in academic research or tactical software, please cite:

```bibtex
@software{zenith2026,
  author = {Gómez, Federico},
  title = {Zenith: Ultra-Fast Direct 2.5D Photogrammetry and Radiometric Orthomosaic Engine for Tactical UAV Operations},
  version = {1.0.0},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/federicogmz/zenith}
}
```
