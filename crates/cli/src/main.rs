//! `zenith`: Interfaz de línea de comandos (CLI) del motor fotogramétrico directo 2.5D.
//! Diseñado para operaciones tácticas de respuesta a emergencias e incendios forestales.

use clap::{Parser, Subcommand, ValueEnum};
use log::{info, warn};
use std::path::PathBuf;
use std::time::Instant;
use zenith_core::{
    CameraBasis, CameraIntrinsics, CameraPose, ConvexHull2D, Point2D, TransverseMercator, Vector3,
};
use zenith_dem::DemWindow;
use zenith_io::{parse_geo_txt, scan_flight_directory, DjiMetadata, EngineManifest};
use zenith_ortho::{
    refine_poses_with_tie_points, DirectOrthoEngine, ImageFrame, OrthoConfig, ResamplingMethod,
    RgbBlendMode, RgbFrameMeta, RgbOrthoConfig, RgbOrthoEngine, TiePointFile, TrimMode,
};

/// Política de resolución de fuente de elevación en Zenith 2.5D
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum DemSource {
    /// Cascada automática: DEM local si existe -> Copernicus 30m si hay red -> Plano de despegue si offline
    Auto,
    /// Fuerza consulta de Copernicus GLO-30 desde AWS Open Data (falla duro si no hay red)
    Copernicus,
    /// Fuerza plano constante a cota de despegue o cota dada
    Flat,
}

#[derive(Parser, Debug)]
#[command(
    name = "zenith",
    version,
    about = "Zenith: Motor fotogramétrico directo 2.5D de ultra alta velocidad en Rust para operaciones tácticas",
    long_about = "Motor fotogramétrico independiente optimizado para ortomosaicos tácticos y triaje térmico (< 2 min).\nCumple con el estándar de interoperabilidad de RAPTOR."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Muestra información del motor, arquitectura y niveles de contrato soportados
    Info,

    /// Procesa una misión completa con imágenes DJI y DTM (cálculo de huellas 3D y telemetría)
    Mission {
        /// Directorio con las imágenes térmicas o visuales
        #[arg(short, long)]
        images: PathBuf,

        /// Ruta al archivo DEM (GeoTIFF .tif, VRT o ENVI)
        #[arg(long, conflicts_with_all = ["dem_source", "flat_elevation"])]
        dem: Option<PathBuf>,

        /// Política de fuente de DEM: "auto" (defecto), "copernicus", "flat"
        #[arg(long, value_enum, conflicts_with = "dem")]
        dem_source: Option<DemSource>,

        /// Elevación constante en metros para plano de contingencia (requiere valor explícito)
        #[arg(long, conflicts_with_all = ["dem", "dem_source"])]
        flat_elevation: Option<f64>,

        /// Encabezado .hdr del DTM legado
        #[arg(long)]
        dtm_hdr: Option<PathBuf>,

        /// Datos binarios .raw del DTM legado
        #[arg(long)]
        dtm_raw: Option<PathBuf>,

        /// Ruta de salida para el GeoJSON de footprints y telemetría
        #[arg(short, long, default_value = "mission_footprints.geojson")]
        output: PathBuf,

        /// Desplazamiento geoidal N en metros (por defecto 0.0 si DTM y GNSS son consistentes)
        #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
        datum_offset: f64,

        /// Ángulo de pitch máximo permitido para mapeo nadir (grados, e.g. -70.0)
        #[arg(long, default_value_t = -70.0, allow_hyphen_values = true)]
        max_pitch: f64,
    },

    /// Ejecuta el pipeline de ortorrectificación directa 2.5D
    Ortho {
        /// Ruta al archivo geo.txt con coordenadas de cámara
        #[arg(short, long)]
        geo: PathBuf,

        /// Directorio con las imágenes capturadas
        #[arg(short, long)]
        images: Option<PathBuf>,

        /// Ruta opcional al archivo DEM (GeoTIFF)
        #[arg(short, long)]
        dem: Option<PathBuf>,

        /// Ruta del producto de salida (.tif o .bin)
        #[arg(short, long, default_value = "orthomosaic.bin")]
        output: PathBuf,

        /// GSD objetivo en metros por píxel (por defecto 0.1m = 10cm)
        #[arg(long, default_value_t = 0.1)]
        gsd: f64,

        /// Offset geoidal N en metros (H = h - N, por defecto -24.0m para Valle de Aburrá)
        #[arg(long, default_value_t = -24.0)]
        datum_offset: f64,

        /// Altitud plana de contingencia en metros si no se especifica DEM
        #[arg(long, default_value_t = 1500.0)]
        plane_alt: f64,

        /// Método de remuestreo: 'nearest' (térmico/radiométrico) o 'bilinear' (RGB)
        #[arg(long, default_value = "nearest")]
        resampling: String,

        /// Ruta opcional para escribir el manifiesto de contrato JSON
        #[arg(short, long, default_value = "manifest.json")]
        manifest: PathBuf,
    },

    /// Genera un ortomosaico térmico directo 2.5D con preservación radiométrica (°C)
    OrthoThermal {
        /// Directorio con las imágenes térmicas
        #[arg(short, long)]
        images: PathBuf,

        /// Directorio con los datos Float32 crudos de las imágenes
        #[arg(long)]
        raw_dir: PathBuf,

        /// Ruta al archivo DEM (GeoTIFF .tif, VRT o ENVI)
        #[arg(long, conflicts_with_all = ["dem_source", "flat_elevation"])]
        dem: Option<PathBuf>,

        /// Política de fuente de DEM: "auto" (defecto), "copernicus", "flat"
        #[arg(long, value_enum, conflicts_with = "dem")]
        dem_source: Option<DemSource>,

        /// Elevación constante en metros para plano de contingencia (requiere valor explícito)
        #[arg(long, conflicts_with_all = ["dem", "dem_source"])]
        flat_elevation: Option<f64>,

        /// Encabezado .hdr del DEM legado
        #[arg(long)]
        dtm_hdr: Option<PathBuf>,

        /// Datos binarios .raw del DEM legado
        #[arg(long)]
        dtm_raw: Option<PathBuf>,

        /// Archivo de salida para el ortomosaico ENVI/TIFF
        #[arg(short, long, default_value = "thermal_orthomosaic_dem2m.tif")]
        output: PathBuf,

        /// GSD objetivo en metros por píxel (por defecto 0.15m para máxima nitidez térmica)
        #[arg(long, default_value_t = 0.15)]
        gsd: f64,

        /// Desplazamiento geoidal N en metros (por defecto 0.0 si DTM y GNSS son consistentes ortométricamente)
        #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
        datum_offset: f64,

        /// Umbral máximo de pitch en grados para filtrar fotogramas de tránsito (por defecto -70.0°)
        #[arg(long, default_value_t = -70.0, allow_hyphen_values = true)]
        max_pitch: f64,

        /// Modo de recorte espacial: "both" (casco + solape), "hull" (solo casco), "overlap" (solo solape), "none"
        #[arg(long, default_value = "both")]
        trim: String,

        /// Conteo mínimo de solape de fotos concurrentes (por defecto 1 para preservar perímetro térmico de 40mm)
        #[arg(long, default_value_t = 1)]
        min_overlap: u16,

        /// Factor de contracción central del sensor para construir el casco convexo (0.5 - 1.0, por defecto 0.92)
        #[arg(long, default_value_t = 0.92)]
        center_crop: f64,

        /// Ruta opcional para exportar el polígono del Casco Convexo en formato GeoJSON
        #[arg(long)]
        hull_geojson: Option<PathBuf>,

        /// Ruta para escribir el manifiesto de contrato
        #[arg(short, long, default_value = "manifest_level1.json")]
        manifest: PathBuf,

        /// Método de remuestreo de píxel individual: "nearest" (defecto para térmico) o "bilinear"
        #[arg(long, default_value = "nearest")]
        resampling: String,

        /// Ancho de banda de costura Voronoi en píxeles de sensor (por defecto 6.0 px; 0.0 para corte duro)
        #[arg(long, default_value_t = 6.0)]
        feather_px: f64,

        /// Diferencia térmica máxima en °C para permitir mezcla (evita diluir hotspots de fuego)
        #[arg(long, default_value_t = 3.5)]
        max_blend_temp_diff: f32,

        /// Sistema de Referencia de Coordenadas (CRS) proyectado: "EPSG:9377", "EPSG:32618", "auto", "utm"
        #[arg(long, alias = "epsg", default_value = "EPSG:9377")]
        crs: String,

        /// Ruta al archivo JSON con puntos de enlace (tie points) o pose overrides de FlexGlue
        #[arg(long)]
        tie_points: Option<PathBuf>,

        /// Ruta opcional para exportar el reporte de poses refinadas a JSON
        #[arg(long)]
        export_refined_poses: Option<PathBuf>,
    },

    /// Genera un ortomosaico visual RGB directo 2.5D con corrección de costuras (feathering)
    OrthoRgb {
        /// Directorio con las imágenes JPEG RGB
        #[arg(short, long)]
        images: PathBuf,

        /// Ruta al archivo DEM (GeoTIFF .tif, VRT o ENVI)
        #[arg(long, conflicts_with_all = ["dem_source", "flat_elevation"])]
        dem: Option<PathBuf>,

        /// Política de fuente de DEM: "auto" (defecto), "copernicus", "flat"
        #[arg(long, value_enum, conflicts_with = "dem")]
        dem_source: Option<DemSource>,

        /// Elevación constante en metros para plano de contingencia (requiere valor explícito)
        #[arg(long, conflicts_with_all = ["dem", "dem_source"])]
        flat_elevation: Option<f64>,

        /// Encabezado .hdr del DEM legado
        #[arg(long)]
        dtm_hdr: Option<PathBuf>,

        /// Datos binarios .raw del DEM legado
        #[arg(long)]
        dtm_raw: Option<PathBuf>,

        /// Archivo de salida para el ortomosaico RGBA (.tif o .raw/.hdr)
        #[arg(short, long, default_value = "rgb_orthomosaic_rust_dem2m.tif")]
        output: PathBuf,

        /// GSD objetivo en metros por píxel (por defecto 0.0 = auto-GSD de alta definición según sensor y altura AGL)
        #[arg(long, default_value_t = 0.0)]
        gsd: f64,

        /// Desplazamiento geoidal N en metros (0.0 para Barbosa comprobado en despegue)
        #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
        datum_offset: f64,

        /// Modo de mezcla: "micro-feather" (por defecto, Top-3 Softmin Voronoi-Band), "best-nadir", o "feather"
        #[arg(long, default_value = "micro-feather")]
        blend_mode: String,

        /// Ancho de rampa de desvanecimiento (feathering) en píxeles de imagen (por defecto 25.0 px)
        #[arg(long, default_value_t = 25.0)]
        feather_px: f64,

        /// Desactiva la compensación de ganancia global Brown-Lowe
        #[arg(long, default_value_t = false)]
        no_gain_compensation: bool,

        /// Fuerza compensación de ganancia global Brown-Lowe (activa por defecto)
        #[arg(long, default_value_t = false)]
        gain_compensation: bool,

        /// Activa dehaze (reducción de neblina atmosférica en bordes)
        #[arg(long, default_value_t = false)]
        dehaze: bool,

        /// Tamaño de tesela en píxeles para Rayon (por defecto 512 px)
        #[arg(long, default_value_t = 512)]
        tile_size: usize,

        /// Umbral máximo de pitch en grados para mapeo nadir (por defecto -70.0°)
        #[arg(long, default_value_t = -70.0, allow_hyphen_values = true)]
        max_pitch: f64,

        /// Modo de recorte espacial: "both" (casco + solape), "hull" (solo casco), "overlap" (solo solape), "none"
        #[arg(long, default_value = "both")]
        trim: String,

        /// Conteo mínimo de solape de fotos concurrentes (por defecto 2 para descartar flecos sin estéreo)
        #[arg(long, default_value_t = 2)]
        min_overlap: u16,

        /// Factor de contracción central del sensor para construir el casco convexo (0.5 - 1.0, por defecto 0.75)
        #[arg(long, default_value_t = 0.75)]
        center_crop: f64,

        /// Ruta opcional para exportar el polígono del Casco Convexo en formato GeoJSON
        #[arg(long)]
        hull_geojson: Option<PathBuf>,

        /// Ruta para escribir el manifiesto de contrato
        #[arg(short, long, default_value = "manifest_level1_rgb.json")]
        manifest: PathBuf,

        /// Sistema de Referencia de Coordenadas (CRS) proyectado: "EPSG:9377", "EPSG:32618", "auto", "utm"
        #[arg(long, alias = "epsg", default_value = "EPSG:9377")]
        crs: String,

        /// Ruta al archivo JSON con puntos de enlace (tie points) o pose overrides de FlexGlue
        #[arg(long)]
        tie_points: Option<PathBuf>,

        /// Ruta opcional para exportar el reporte de poses refinadas a JSON
        #[arg(long)]
        export_refined_poses: Option<PathBuf>,
    },

    /// Ejecuta un benchmark sintético multihilo (Rayon) de proyección 2.5D
    Benchmark {
        /// Número de fotogramas a simular
        #[arg(short, long, default_value_t = 50)]
        frames: usize,

        /// Dimensiones del DEM sintético en celdas (N x N)
        #[arg(short, long, default_value_t = 1000)]
        dem_size: usize,

        /// GSD objetivo en metros
        #[arg(short, long, default_value_t = 0.2)]
        gsd: f64,
    },

    /// Valida un manifiesto de motor contra el Contrato Fotogramétrico de RAPTOR
    ValidateContract {
        /// Ruta al archivo manifest.json
        #[arg(short, long)]
        manifest: PathBuf,
    },
}

fn compute_takeoff_elevation(frames: &[DjiMetadata]) -> Option<f64> {
    // 1. Filtrar fotogramas en vuelo con cota relativa válida (> 10m para descartar suelo)
    let mut takeoff_alts: Vec<f64> = frames
        .iter()
        .filter(|f| f.rel_alt > 10.0 && !f.abs_alt.is_nan() && !f.rel_alt.is_nan())
        .map(|f| f.abs_alt - f.rel_alt)
        .collect();

    if takeoff_alts.is_empty() {
        return None;
    }

    takeoff_alts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = takeoff_alts[takeoff_alts.len() / 2];

    // 2. Filtro MAD (Median Absolute Deviation) para descartar outliers espurios
    let mut abs_devs: Vec<f64> = takeoff_alts.iter().map(|&x| (x - median).abs()).collect();
    abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = abs_devs[abs_devs.len() / 2];

    let threshold = 3.0 * mad.max(2.0); // Mínimo 2m de umbral de dispersión
    let inliers: Vec<f64> = takeoff_alts
        .into_iter()
        .filter(|&x| (x - median).abs() <= threshold)
        .collect();

    if inliers.is_empty() {
        Some(median)
    } else {
        Some(inliers[inliers.len() / 2])
    }
}

/// Auto-calibra el desplazamiento de datum vertical sistemático entre GNSS y DEM usando lecturas del telémetro láser (LRF).
/// Retorna `(effective_datum_offset, Option<serde_json::Value>)`.
pub fn calibrate_lrf_datum_offset(
    frames: &[DjiMetadata],
    dem: &DemWindow,
    tm: &TransverseMercator,
    user_datum_offset: f64,
) -> (f64, Option<serde_json::Value>) {
    // Si el usuario especificó explícitamente un offset no cero, respetarlo
    if user_datum_offset.abs() > 1e-6 {
        return (user_datum_offset, None);
    }

    let mut diffs = Vec::new();
    for frame in frames {
        let d_lrf = match frame.lrf_target_distance {
            Some(d) if d > 10.0 => d,
            _ => continue,
        };

        // Filtrar tomas nadir o casi-nadir (pitch <= -70.0°)
        if frame.pitch > -70.0 || frame.abs_alt.is_nan() {
            continue;
        }

        let pitch_rad = (-frame.pitch).to_radians();
        let cos_theta = pitch_rad.sin().clamp(0.7, 1.0);
        let est_ground_z = frame.abs_alt - d_lrf * cos_theta;

        let (target_lon, target_lat) = match (frame.lrf_target_lon, frame.lrf_target_lat) {
            (Some(lon), Some(lat)) if lon.abs() > 1e-4 && lat.abs() > 1e-4 => (lon, lat),
            _ => (frame.lon, frame.lat),
        };

        let (tx, ty, _) = tm.forward(target_lon, target_lat);
        if let Some(z_dem) = dem.sample_elevation_bilinear(tx, ty) {
            let diff = est_ground_z - z_dem;
            if diff.is_finite() {
                diffs.push(diff);
            }
        }
    }

    if diffs.len() < 5 {
        return (0.0, None);
    }

    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = diffs[diffs.len() / 2];

    let mut abs_devs: Vec<f64> = diffs.iter().map(|&x| (x - median).abs()).collect();
    abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = abs_devs[abs_devs.len() / 2];

    let threshold = 3.0 * mad.max(1.5);
    let inliers: Vec<f64> = diffs
        .into_iter()
        .filter(|&x| (x - median).abs() <= threshold)
        .collect();

    if inliers.is_empty() {
        return (0.0, None);
    }

    let inlier_median = inliers[inliers.len() / 2];

    if inlier_median.abs() > 5.0 && mad < 8.0 {
        log::info!(
            "Auto-calibración LRF activada: offset vertical detectado = {:.2} m (MAD = {:.2} m, {} muestras inlier)",
            inlier_median,
            mad,
            inliers.len()
        );
        let meta = serde_json::json!({
            "calibrated": true,
            "lrf_datum_offset_m": inlier_median,
            "mad_m": mad,
            "samples_used": inliers.len(),
            "explanation": "Desplazamiento vertical sistemático GNSS-DEM auto-calibrado con telémetro láser LRF"
        });
        (inlier_median, Some(meta))
    } else {
        (0.0, None)
    }
}

/// Emite un evento de telemetría de progreso estructurado en stdout con flush inmediato.
fn emit_progress(stage: &str, pct: u32, detail: &str) {
    use std::io::Write;
    println!(
        "[ZENITH_PROGRESS] stage={} pct={} detail={}",
        stage, pct, detail
    );
    let _ = std::io::stdout().flush();
}

/// Guarda RAII para garantizar la eliminación de archivos DEM temporales al salir del ámbito.
struct TempDemFiles(Vec<PathBuf>);

impl Drop for TempDemFiles {
    fn drop(&mut self) {
        for p in &self.0 {
            let _ = std::fs::remove_file(p);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn resolve_dem(
    dem: Option<&std::path::Path>,
    dem_source: DemSource,
    flat_elevation: Option<f64>,
    dtm_hdr: Option<&std::path::Path>,
    dtm_raw: Option<&std::path::Path>,
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    mean_lat: f64,
    mean_lon: f64,
    datum_offset: f64,
    frames: Option<&[DjiMetadata]>,
    crs: &str,
) -> Result<(DemWindow, serde_json::Value), Box<dyn std::error::Error>> {
    // 1. Legado directo: dtm_hdr y dtm_raw
    if let (Some(hdr), Some(raw)) = (dtm_hdr, dtm_raw) {
        if !hdr.exists() || !raw.exists() {
            return Err(format!(
                "El DEM legado especificado no existe: {} / {}",
                hdr.display(),
                raw.display()
            )
            .into());
        }
        log::info!("Cargando DEM legado en memoria desde: {}", hdr.display());
        let w = DemWindow::from_envi_raw(hdr, raw, 0.0)?;
        let meta = serde_json::json!({
            "kind": "file_envi",
            "model_type": "DTM",
            "uri_or_path": hdr.display().to_string(),
            "resolution_m": w.cell_size_x,
            "fallback_applied": false,
        });
        return Ok((w, meta));
    }

    // 2. Nivel 1: DEM Local Explícito (--dem <archivo.tif>)
    if let Some(dem_path) = dem {
        if !dem_path.exists() {
            return Err(format!(
                "El archivo DEM especificado no existe: {}",
                dem_path.display()
            )
            .into());
        }
        if dem_path.extension().and_then(|e| e.to_str()) == Some("hdr") {
            let raw_path = dem_path.with_extension("raw");
            let w = DemWindow::from_envi_raw(dem_path, &raw_path, 0.0)?;
            let meta = serde_json::json!({
                "kind": "file_envi",
                "model_type": "DTM",
                "uri_or_path": dem_path.display().to_string(),
                "resolution_m": w.cell_size_x,
                "fallback_applied": false,
            });
            return Ok((w, meta));
        }

        log::info!(
            "Recortando ventana de vuelo desde DEM local: {}",
            dem_path.display()
        );
        let pid = std::process::id();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_millis();
        let tmp_raw = PathBuf::from(format!("/tmp/zenith_dem_local_{}_{}.raw", pid, ts));
        let tmp_hdr = PathBuf::from(format!("/tmp/zenith_dem_local_{}_{}.hdr", pid, ts));
        let _cleaner = TempDemFiles(vec![tmp_raw.clone(), tmp_hdr.clone()]);

        let b_min_x = min_x - 500.0;
        let b_max_x = max_x + 500.0;
        let b_min_y = min_y - 500.0;
        let b_max_y = max_y + 500.0;

        let status = std::process::Command::new("gdalwarp")
            .args([
                "-t_srs",
                crs,
                "-te",
                &b_min_x.to_string(),
                &b_min_y.to_string(),
                &b_max_x.to_string(),
                &b_max_y.to_string(),
                "-tr",
                "2.0",
                "2.0",
                "-r",
                "bilinear",
                "-of",
                "ENVI",
                "-ot",
                "Float32",
            ])
            .arg(dem_path)
            .arg(&tmp_raw)
            .output();

        match status {
            Ok(st) if st.status.success() => {
                let w = DemWindow::from_envi_raw(&tmp_hdr, &tmp_raw, 0.0)?;
                let meta = serde_json::json!({
                    "kind": "local_geotiff",
                    "model_type": "DTM/DSM",
                    "uri_or_path": dem_path.display().to_string(),
                    "resolution_m": w.cell_size_x,
                    "vertical_datum_offset_m": datum_offset,
                    "fallback_applied": false,
                });
                return Ok((w, meta));
            }
            _ => {
                return Err(format!("Fallo al recortar DEM local '{}' con gdalwarp. Verifique que sea un raster válido y que cubra la zona de vuelo.", dem_path.display()).into());
            }
        }
    }

    // 3. Plano explícito por usuario (--flat-elevation <metros>)
    if let Some(elev) = flat_elevation {
        log::warn!(
            "ADVERTENCIA [TIER 3 EXPLÍCITO]: Empleando plano constante a cota {:.1} m. En relieve montañoso existirá deriva geométrica.",
            elev
        );
        let w = DemWindow::from_flat(
            min_x - 500.0,
            max_x + 500.0,
            min_y - 500.0,
            max_y + 500.0,
            30.0,
            elev as f32,
        );
        let meta = serde_json::json!({
            "kind": "flat_plane",
            "model_type": "plane",
            "flat_elevation_m": elev,
            "terrain_reference": "user_specified",
            "resolution_m": 30.0,
            "fallback_applied": false,
            "metric_suitability": "qualitative_emergency_only",
        });
        return Ok((w, meta));
    }

    // 4. Si se forzó explícitamente modo plano (--dem-source flat)
    if dem_source == DemSource::Flat {
        let elev = if let Some(frames) = frames {
            compute_takeoff_elevation(frames).ok_or("No fue posible derivar cota de despegue desde metadatos de las fotos. Especifique --flat-elevation <metros>.")?
        } else {
            return Err("Modo plano requiere fotogramas con telemetría o especificar --flat-elevation <metros>.".into());
        };
        log::warn!(
            "ADVERTENCIA [TIER 3 TELEMETRÍA]: Plano constante a cota de despegue (Home Point) de {:.1} m. En relieve montañoso existirá deriva geométrica.",
            elev
        );
        let w = DemWindow::from_flat(
            min_x - 500.0,
            max_x + 500.0,
            min_y - 500.0,
            max_y + 500.0,
            30.0,
            elev as f32,
        );
        let meta = serde_json::json!({
            "kind": "flat_plane",
            "model_type": "plane",
            "flat_elevation_m": elev,
            "terrain_reference": "takeoff_home_point",
            "resolution_m": 30.0,
            "fallback_applied": false,
            "metric_suitability": "qualitative_emergency_only",
        });
        return Ok((w, meta));
    }

    // 5. Nivel 2: Copernicus GLO-30 desde AWS Open Data (Auto o Copernicus explícito)
    let lat_sw = mean_lat.floor() as i32;
    let lon_sw = mean_lon.floor() as i32;
    let ns = if lat_sw >= 0 { 'N' } else { 'S' };
    let ew = if lon_sw >= 0 { 'E' } else { 'W' };
    let lat_abs = lat_sw.abs();
    let lon_abs = lon_sw.abs();
    let tile_url = format!(
        "/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{}{:02}_00_{}{:03}_00_DEM/Copernicus_DSM_COG_10_{}{:02}_00_{}{:03}_00_DEM.tif",
        ns, lat_abs, ew, lon_abs, ns, lat_abs, ew, lon_abs
    );

    log::info!(
        "Consultando DEM Copernicus GLO-30 (30m) desde AWS Open Data: {}",
        tile_url
    );
    let pid = std::process::id();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis();
    let tmp_raw = PathBuf::from(format!("/tmp/zenith_dem_copernicus_{}_{}.raw", pid, ts));
    let tmp_hdr = PathBuf::from(format!("/tmp/zenith_dem_copernicus_{}_{}.hdr", pid, ts));
    let _cleaner = TempDemFiles(vec![tmp_raw.clone(), tmp_hdr.clone()]);

    let b_min_x = min_x - 500.0;
    let b_max_x = max_x + 500.0;
    let b_min_y = min_y - 500.0;
    let b_max_y = max_y + 500.0;

    let warp_output = std::process::Command::new("gdalwarp")
        .args([
            "--config",
            "GDAL_HTTP_TIMEOUT",
            "6",
            "--config",
            "GDAL_HTTP_MAX_RETRY",
            "1",
            "-overwrite",
            "-t_srs",
            crs,
            "-te",
            &b_min_x.to_string(),
            &b_min_y.to_string(),
            &b_max_x.to_string(),
            &b_max_y.to_string(),
            "-tr",
            "30",
            "30",
            "-r",
            "bilinear",
            "-of",
            "ENVI",
            "-ot",
            "Float32",
            &tile_url,
        ])
        .arg(&tmp_raw)
        .output();

    match warp_output {
        Ok(out) if out.status.success() => {
            let w = DemWindow::from_envi_raw(&tmp_hdr, &tmp_raw, 0.0)?;

            let meta = serde_json::json!({
                "kind": "copernicus_glo30",
                "model_type": "DSM",
                "uri_or_path": tile_url,
                "resolution_m": 30.0,
                "vertical_datum": "EGM2008",
                "vertical_datum_offset_m": datum_offset,
                "metric_suitability": "regional_reconnaissance_dsm_30m",
                "attribution": "European Space Agency (ESA) Copernicus DEM 30m via AWS Open Data",
                "fallback_applied": false,
            });
            return Ok((w, meta));
        }
        Ok(out) => {
            let msg = format!(
                "stdout: {} | stderr: {}",
                String::from_utf8_lossy(&out.stdout).trim(),
                String::from_utf8_lossy(&out.stderr).trim()
            );
            log::warn!(
                "gdalwarp no pudo procesar Copernicus GLO-30 (exit code {:?}): {}",
                out.status.code(),
                msg
            );
        }
        Err(e) => {
            log::warn!("No fue posible invocar gdalwarp para Copernicus: {}", e);
        }
    }

    // Si el usuario pidió explícitamente Copernicus, fallo duro
    if dem_source == DemSource::Copernicus {
        return Err(format!(
            "Fallo al descargar/recortar Copernicus GLO-30 desde AWS ({}). Verifique su conexión a internet o emplee '--dem-source auto' para permitir degradación a plano de contingencia.",
            tile_url
        ).into());
    }

    // 6. Nivel 3 (Fallback en Auto): Degradación con gracia a plano de telemetría
    log::warn!(
        "AVISO [CASCADA TIER 2 -> TIER 3]: No se pudo acceder a Copernicus GLO-30 en AWS (sin conexión a internet o timeout de red)."
    );
    log::warn!(
        "Aplicando degradación automática (Fallback) a plano constante derivado de telemetría de despegue..."
    );

    let takeoff_elev = if let Some(frames) = frames {
        compute_takeoff_elevation(frames).ok_or(
            "Fallo en Copernicus y no fue posible derivar cota de despegue desde las fotos (rel_alt ausente o inválida). Especifique --flat-elevation <metros>."
        )?
    } else {
        return Err(
            "Fallo en Copernicus y no hay fotogramas disponibles para derivar cota de despegue."
                .into(),
        );
    };

    log::warn!(
        "ADVERTENCIA DE PRECISIÓN: Generando ortomosaico sobre plano constante a cota de despegue (Home Point) de {:.1} m. En terreno montañoso existirá deriva métrica en zonas alejadas de la cota de despegue.",
        takeoff_elev
    );

    let w = DemWindow::from_flat(
        min_x - 500.0,
        max_x + 500.0,
        min_y - 500.0,
        max_y + 500.0,
        30.0,
        takeoff_elev as f32,
    );

    let meta = serde_json::json!({
        "kind": "flat_plane",
        "model_type": "plane",
        "flat_elevation_m": takeoff_elev,
        "terrain_reference": "takeoff_home_point",
        "vertical_datum": "local_takeoff_barometric_gnss",
        "resolution_m": 30.0,
        "fallback_applied": true,
        "fallback_reason": "copernicus_aws_network_unreachable_or_offline",
        "metric_suitability": "qualitative_emergency_only",
    });

    Ok((w, meta))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Info => {
            println!("============================================================");
            println!("  ZENITH: Motor Fotogramétrico Directo 2.5D (Rust)");
            println!("  Versión: {}", env!("CARGO_PKG_VERSION"));
            println!("============================================================");
            println!("Arquitectura: Proyección cónica directa y ray marching multihilo (Rayon)");
            println!("Preservación Radiométrica: Remuestreo Nearest-Neighbor estricto sin mezclas");
            println!("Niveles operacionales:");
            println!("  - Nivel 0 (Triaje Térmico): Latencia < 2 min");
            println!("  - Nivel 1 (Ortomosaico 2.5D): Latencia 5-15 min");
            println!("  - Nivel 2 (Fotogrametría SfM densa): Handoff asíncrono");
            println!("Soporte de Sensores: DJI Zenmuse H20T, Matrice 30T (M3T), FLIR Vue Pro R");
            println!("Datum Vertical: Ajuste ortométrico automático (H = h - N)");
            println!("Contrato de Interoperabilidad: Compatible con RAPTOR v1.0");
        }

        Commands::Mission {
            images,
            dem,
            dem_source,
            flat_elevation,
            dtm_hdr,
            dtm_raw,
            output,
            datum_offset,
            max_pitch,
        } => {
            let t0 = Instant::now();
            info!("Iniciando procesamiento de misión con Zenith...");
            info!("Escaneando imágenes desde: {}", images.display());
            let scan = scan_flight_directory(&images)?;
            info!(
                "Fotogramas válidos encontrados: {}, Errores: {}",
                scan.frames.len(),
                scan.errors.len()
            );

            let tm = TransverseMercator::colombia_origen_nacional();
            let mut cam_x_min = f64::INFINITY;
            let mut cam_x_max = f64::NEG_INFINITY;
            let mut cam_y_min = f64::INFINITY;
            let mut cam_y_max = f64::NEG_INFINITY;
            let mut sum_lat = 0.0;
            let mut sum_lon = 0.0;
            let mut count_geo = 0;

            for frame in &scan.frames {
                let (cx, cy, _) = tm.forward(frame.lon, frame.lat);
                cam_x_min = cam_x_min.min(cx);
                cam_x_max = cam_x_max.max(cx);
                cam_y_min = cam_y_min.min(cy);
                cam_y_max = cam_y_max.max(cy);
                sum_lat += frame.lat;
                sum_lon += frame.lon;
                count_geo += 1;
            }

            let mean_lat = if count_geo > 0 {
                sum_lat / count_geo as f64
            } else {
                0.0
            };
            let mean_lon = if count_geo > 0 {
                sum_lon / count_geo as f64
            } else {
                0.0
            };

            let (dem, _dem_meta) = resolve_dem(
                dem.as_deref(),
                dem_source.unwrap_or(DemSource::Auto),
                flat_elevation,
                dtm_hdr.as_deref(),
                dtm_raw.as_deref(),
                cam_x_min,
                cam_x_max,
                cam_y_min,
                cam_y_max,
                mean_lat,
                mean_lon,
                datum_offset,
                Some(&scan.frames),
                "EPSG:9377",
            )?;
            info!(
                "DTM cargado: {} x {} celdas a {:.2} m de resolución",
                dem.width, dem.height, dem.cell_size_x
            );

            let (effective_datum_offset, _lrf_meta) =
                calibrate_lrf_datum_offset(&scan.frames, &dem, &tm, datum_offset);

            let sample_frame = scan.frames.first().ok_or("No hay fotogramas escaneados")?;
            let intrinsics = CameraIntrinsics::from_telemetry(
                sample_frame.image_width.unwrap_or(640),
                sample_frame.image_height.unwrap_or(512),
                sample_frame.model.as_deref(),
                sample_frame.calibrated_focal_length,
                sample_frame.calibrated_cx,
                sample_frame.calibrated_cy,
                sample_frame.focal_length_35mm,
                sample_frame.focal_length_mm,
            );
            info!(
                "Intrínsecos de muestra ({:?}): {}x{} px, fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1}",
                sample_frame.model,
                intrinsics.width,
                intrinsics.height,
                intrinsics.fx,
                intrinsics.fy,
                intrinsics.cx,
                intrinsics.cy
            );

            let mut features = Vec::new();
            let mut valid_intersections = 0;
            let mut total_agl = 0.0;

            for frame in &scan.frames {
                let frame_intrinsics = CameraIntrinsics::from_telemetry(
                    frame.image_width.unwrap_or(640),
                    frame.image_height.unwrap_or(512),
                    frame.model.as_deref(),
                    frame.calibrated_focal_length,
                    frame.calibrated_cx,
                    frame.calibrated_cy,
                    frame.focal_length_35mm,
                    frame.focal_length_mm,
                );
                let w_px = frame_intrinsics.width as f64;
                let h_px = frame_intrinsics.height as f64;

                let (cam_x, cam_y, gamma) = tm.forward(frame.lon, frame.lat);
                let cam_z = frame.abs_alt - effective_datum_offset;
                let cam_pos = Vector3::new(cam_x, cam_y, cam_z);

                // Corregir yaw con la convergencia de meridianos
                let yaw_grid = frame.yaw + gamma;
                let basis = CameraBasis::from_gimbal(yaw_grid, frame.pitch, frame.roll);

                let is_nadir = frame.pitch <= max_pitch;

                let mut agl = -9999.0;
                let mut coords = Vec::new();

                if is_nadir {
                    // Centro de la imagen
                    let center_ray = basis.pixel_to_ray(
                        frame_intrinsics.cx,
                        frame_intrinsics.cy,
                        cam_pos,
                        &frame_intrinsics,
                    );
                    let center_impact = dem.intersect_ray(&center_ray, 3000.0, 16);

                    // Cuatro esquinas de la imagen
                    let c1 = basis.pixel_to_ray(0.0, 0.0, cam_pos, &frame_intrinsics);
                    let c2 = basis.pixel_to_ray(w_px, 0.0, cam_pos, &frame_intrinsics);
                    let c3 = basis.pixel_to_ray(w_px, h_px, cam_pos, &frame_intrinsics);
                    let c4 = basis.pixel_to_ray(0.0, h_px, cam_pos, &frame_intrinsics);

                    let p1 = dem.intersect_ray(&c1, 3000.0, 16);
                    let p2 = dem.intersect_ray(&c2, 3000.0, 16);
                    let p3 = dem.intersect_ray(&c3, 3000.0, 16);
                    let p4 = dem.intersect_ray(&c4, 3000.0, 16);

                    if let Some(ci) = center_impact {
                        valid_intersections += 1;
                        agl = cam_z - ci.z;
                        total_agl += agl;
                    }

                    if let (Some(pt1), Some(pt2), Some(pt3), Some(pt4)) = (p1, p2, p3, p4) {
                        coords.push(vec![pt1.x, pt1.y]);
                        coords.push(vec![pt2.x, pt2.y]);
                        coords.push(vec![pt3.x, pt3.y]);
                        coords.push(vec![pt4.x, pt4.y]);
                        coords.push(vec![pt1.x, pt1.y]);
                    }
                }

                let geom = if !coords.is_empty() {
                    serde_json::json!({
                        "type": "Polygon",
                        "coordinates": [coords]
                    })
                } else {
                    serde_json::json!({
                        "type": "Point",
                        "coordinates": [cam_x, cam_y]
                    })
                };

                features.push(serde_json::json!({
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {
                        "filename": frame.filename,
                        "lon": frame.lon,
                        "lat": frame.lat,
                        "cam_x": cam_x,
                        "cam_y": cam_y,
                        "abs_alt": frame.abs_alt,
                        "agl_m": agl,
                        "yaw": frame.yaw,
                        "pitch": frame.pitch,
                        "roll": frame.roll,
                        "rtk_flag": frame.rtk_flag
                    }
                }));
            }

            let geojson = serde_json::json!({
                "type": "FeatureCollection",
                "crs": {
                    "type": "name",
                    "properties": {
                        "name": "EPSG:9377"
                    }
                },
                "features": features
            });

            std::fs::write(&output, serde_json::to_string_pretty(&geojson)?)?;
            let total_time = t0.elapsed();

            println!("============================================================");
            println!("  PROCESAMIENTO DE MISIÓN COMPLETADO (ZENITH)");
            println!("============================================================");
            println!("  Tiempo total: {:?}", total_time);
            println!("  Fotogramas procesados: {}", scan.frames.len());
            println!(
                "  Intersecciones válidas con terreno: {} / {}",
                valid_intersections,
                scan.frames.len()
            );
            if valid_intersections > 0 {
                println!(
                    "  Altura sobre terreno (AGL) promedio: {:.2} m",
                    total_agl / valid_intersections as f64
                );
            }
            println!("  Archivo GeoJSON generado: {}", output.display());
            println!("============================================================");
        }

        Commands::Ortho {
            geo,
            images: _,
            dem,
            output,
            gsd,
            datum_offset,
            plane_alt,
            resampling,
            manifest,
        } => {
            let t0 = Instant::now();
            info!("Iniciando ortorrectificación directa con Zenith...");
            info!("Leyendo coordenadas de: {}", geo.display());

            let geo_content = std::fs::read_to_string(&geo)?;
            let (crs, entries) = parse_geo_txt(&geo_content)?;
            info!(
                "CRS detectado: {}, Cámaras encontradas: {}",
                crs,
                entries.len()
            );

            if entries.is_empty() {
                eprintln!("Error: El archivo geo.txt no contiene ninguna cámara.");
                std::process::exit(1);
            }

            // Configurar método de remuestreo
            let resampling_method = match resampling.to_lowercase().as_str() {
                "bilinear" => ResamplingMethod::Bilinear,
                _ => ResamplingMethod::NearestNeighbor,
            };

            // Intrínsecos calibrados por defecto (DJI M3T Thermal: 640x512, 12µm, f=9.1mm -> fx=fy=758.33)
            let intrinsics = CameraIntrinsics::dji_m3t_thermal();

            let dem_used = dem.is_some();
            // Preparar DEM o Plano de contingencia
            let dem_window = if let Some(ref dem_path) = dem {
                info!("Cargando DEM desde: {}", dem_path.display());
                // Si existe el archivo, se inicializa ventana (soporte GeoTIFF o sintético)
                DemWindow::new(
                    500,
                    500,
                    entries[0].lon - 0.05,
                    entries[0].lat + 0.05,
                    2.0,
                    2.0,
                    vec![plane_alt as f32; 500 * 500],
                    Some(-9999.0),
                    datum_offset,
                )
            } else {
                info!(
                    "Sin DEM externo. Empleando plano de contingencia a cota {} m",
                    plane_alt
                );
                DemWindow::new(
                    100,
                    100,
                    entries[0].lon - 0.02,
                    entries[0].lat + 0.02,
                    10.0,
                    10.0,
                    vec![plane_alt as f32; 100 * 100],
                    Some(-9999.0),
                    0.0,
                )
            };

            // Construir fotogramas sintéticos / leídos
            let mut frames = Vec::new();
            for entry in &entries {
                let pose = CameraPose {
                    position: Vector3::new(entry.lon, entry.lat, entry.alt - datum_offset),
                    yaw_deg: entry.yaw,
                    pitch_deg: entry.pitch,
                    roll_deg: entry.roll,
                };
                let dummy_data = vec![25.0f32; 640 * 512]; // Placeholder para imágenes en memoria
                frames.push(ImageFrame::new(
                    entry.filename.clone(),
                    pose,
                    intrinsics,
                    dummy_data,
                    640,
                    512,
                ));
            }

            let ortho_config = OrthoConfig {
                target_gsd_m: gsd,
                resampling: resampling_method,
                nodata_value: f32::NAN,
                trim_mode: TrimMode::None,
                min_overlap: 1,
                hull: None,
                feather_px: 0.0,
                max_blend_temp_diff: 3.5,
            };

            info!("Ejecutando mapeo de celdas en paralelo...");
            let result =
                DirectOrthoEngine::generate_orthomosaic(&frames, &dem_window, &ortho_config);

            info!(
                "Ortorrectificación completada en {} ms. Dimensiones: {}x{}, Píxeles válidos: {}",
                result.elapsed_ms, result.width, result.height, result.valid_pixels
            );

            // Escribir producto binario rápido
            let byte_slice: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    result.data.as_ptr() as *const u8,
                    result.data.len() * std::mem::size_of::<f32>(),
                )
            };
            std::fs::write(&output, byte_slice)?;
            info!("Producto guardado en: {}", output.display());

            // Escribir manifiesto de contrato
            let total_sec = t0.elapsed().as_secs_f64();
            let manifest_obj = EngineManifest::new_level1(
                total_sec,
                dem_used,
                gsd,
                &format!("{:?}", resampling_method),
                vec![output.to_string_lossy().to_string()],
            );
            manifest_obj.write_to_file(&manifest)?;
            info!("Manifiesto escrito en: {}", manifest.display());
            println!("Zenith Ortho completado exitosamente en {:.2} s", total_sec);
        }

        Commands::OrthoThermal {
            images,
            raw_dir,
            dem,
            dem_source,
            flat_elevation,
            dtm_hdr,
            dtm_raw,
            output,
            gsd,
            datum_offset,
            max_pitch,
            trim,
            min_overlap,
            center_crop,
            hull_geojson,
            manifest,
            resampling,
            feather_px,
            max_blend_temp_diff,
            crs,
            tie_points,
            export_refined_poses,
        } => {
            let t0 = Instant::now();
            info!("Iniciando ortorrectificación térmica 2.5D con Zenith...");
            info!(
                "Escaneando metadatos XMP de imágenes desde: {}",
                images.display()
            );
            let scan = scan_flight_directory(&images)?;
            info!("Fotogramas escaneados: {}", scan.frames.len());
            emit_progress(
                "scan",
                10,
                &format!("Fotogramas escaneados: {}", scan.frames.len()),
            );

            let mut sum_lat = 0.0;
            let mut sum_lon = 0.0;
            let mut count_geo = 0;

            for frame in &scan.frames {
                sum_lat += frame.lat;
                sum_lon += frame.lon;
                count_geo += 1;
            }

            let mean_lat = if count_geo > 0 {
                sum_lat / count_geo as f64
            } else {
                0.0
            };
            let mean_lon = if count_geo > 0 {
                sum_lon / count_geo as f64
            } else {
                0.0
            };

            let (tm, resolved_crs) = TransverseMercator::resolve_crs(&crs, mean_lon, mean_lat)?;
            info!("CRS resuelto: {} (solicitado: {})", resolved_crs, crs);
            if resolved_crs == "EPSG:9377" && (mean_lon - (-73.0)).abs() > 7.5 {
                log::warn!(
                    "ADVERTENCIA DE PROYECCIÓN: Las coordenadas (lon: {:.3}, lat: {:.3}) están a más de 7.5° del meridiano central de Colombia EPSG:9377 (-73.0°). La distorsión de escala superará los límites cartográficos recomendados. Considere usar '--crs auto' o UTM local.",
                    mean_lon, mean_lat
                );
            }

            let mut cam_x_min = f64::INFINITY;
            let mut cam_x_max = f64::NEG_INFINITY;
            let mut cam_y_min = f64::INFINITY;
            let mut cam_y_max = f64::NEG_INFINITY;

            for frame in &scan.frames {
                let (cx, cy, _) = tm.forward(frame.lon, frame.lat);
                cam_x_min = cam_x_min.min(cx);
                cam_x_max = cam_x_max.max(cx);
                cam_y_min = cam_y_min.min(cy);
                cam_y_max = cam_y_max.max(cy);
            }

            emit_progress("dem", 25, "Resolviendo modelo de elevación...");
            let (dem, dem_meta) = resolve_dem(
                dem.as_deref(),
                dem_source.unwrap_or(DemSource::Auto),
                flat_elevation,
                dtm_hdr.as_deref(),
                dtm_raw.as_deref(),
                cam_x_min,
                cam_x_max,
                cam_y_min,
                cam_y_max,
                mean_lat,
                mean_lon,
                datum_offset,
                Some(&scan.frames),
                &resolved_crs,
            )?;
            info!(
                "DEM cargado: {} x {} celdas a {:.2} m de resolución",
                dem.width, dem.height, dem.cell_size_x
            );

            emit_progress("lrf", 35, "Calibrando datum vertical LRF...");
            let (effective_datum_offset, lrf_meta) =
                calibrate_lrf_datum_offset(&scan.frames, &dem, &tm, datum_offset);

            let sample_frame = scan
                .frames
                .iter()
                .find(|f| {
                    f.filename.ends_with("_T.JPG")
                        || f.filename.ends_with("_T.jpg")
                        || f.filename.contains("_T_")
                        || (f.image_width == Some(640) && f.image_height == Some(512))
                })
                .or_else(|| scan.frames.first())
                .ok_or("No hay fotogramas escaneados")?;

            let intrinsics = CameraIntrinsics::from_telemetry(
                sample_frame.image_width.unwrap_or(640),
                sample_frame.image_height.unwrap_or(512),
                sample_frame.model.as_deref(),
                sample_frame.calibrated_focal_length,
                sample_frame.calibrated_cx,
                sample_frame.calibrated_cy,
                sample_frame.focal_length_35mm,
                sample_frame.focal_length_mm,
            );
            info!(
                "Intrínsecos térmicos determinados dinámicamente ({:?}): {}x{} px, fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1}",
                sample_frame.model,
                intrinsics.width,
                intrinsics.height,
                intrinsics.fx,
                intrinsics.fy,
                intrinsics.cx,
                intrinsics.cy
            );

            // Filtrar fotogramas de tránsito y cargar datos radiométricos Float32
            let mut image_frames = Vec::new();
            let mut hull_points = Vec::new();
            let mut ignored_transit = 0;
            let mut missing_raw = 0;
            let mut expected_raw_dims: Option<(u32, u32)> = None;

            let crop_f = center_crop.clamp(0.1, 1.0);

            for frame in &scan.frames {
                // Filtro riguroso de pitch: solo mapping nadir (pitch <= max_pitch)
                if frame.pitch > max_pitch {
                    ignored_transit += 1;
                    continue;
                }

                let raw_filename = frame
                    .filename
                    .replace(".JPG", ".raw")
                    .replace(".jpg", ".raw");
                let raw_path = raw_dir.join(&raw_filename);
                if !raw_path.exists() {
                    missing_raw += 1;
                    continue;
                }

                // Leer los valores float32 de temperatura (°C): 640x512 (estándar) o 1280x1024 (super-resolución)
                let bytes = match std::fs::read(&raw_path) {
                    Ok(b) => b,
                    Err(_) => {
                        missing_raw += 1;
                        continue;
                    }
                };

                let tel_w = frame.image_width.unwrap_or(640);
                let tel_h = frame.image_height.unwrap_or(512);
                let (raw_w, raw_h) = if bytes.len() == (tel_w as usize) * (tel_h as usize) * 4 {
                    (tel_w, tel_h)
                } else if bytes.len() == 640 * 512 * 4 {
                    (640u32, 512u32)
                } else if bytes.len() == 1280 * 1024 * 4 {
                    (1280u32, 1024u32)
                } else {
                    missing_raw += 1;
                    continue;
                };

                if let Some((exp_w, exp_h)) = expected_raw_dims {
                    if (raw_w, raw_h) != (exp_w, exp_h) {
                        return Err(format!(
                            "Resolución térmica inconsistente en {}: {}x{} vs esperado {}x{}",
                            raw_filename, raw_w, raw_h, exp_w, exp_h
                        )
                        .into());
                    }
                } else {
                    expected_raw_dims = Some((raw_w, raw_h));
                }

                let mut base_intrinsics = CameraIntrinsics::from_telemetry(
                    tel_w,
                    tel_h,
                    frame.model.as_deref(),
                    frame.calibrated_focal_length,
                    frame.calibrated_cx,
                    frame.calibrated_cy,
                    frame.focal_length_35mm,
                    frame.focal_length_mm,
                );
                if frame.dewarp_flag.as_deref() == Some("0")
                    || frame.dewarp_flag.as_deref() == Some("false")
                {
                    if let Some(ref dd) = frame.dewarp_data {
                        base_intrinsics = base_intrinsics.with_dji_dewarp(dd);
                    }
                }
                let frame_intrinsics = base_intrinsics.scaled(raw_w, raw_h);
                let w_px = frame_intrinsics.width as f64;
                let h_px = frame_intrinsics.height as f64;

                let (chunks, _) = bytes.as_chunks::<4>();
                let floats: Vec<f32> = chunks.iter().map(|&c| f32::from_le_bytes(c)).collect();

                let (cam_x, cam_y, gamma) = tm.forward(frame.lon, frame.lat);
                let cam_z = frame.abs_alt - effective_datum_offset;
                let cam_pos = Vector3::new(cam_x, cam_y, cam_z);
                let yaw_grid = frame.yaw + gamma;

                let basis = CameraBasis::from_gimbal(yaw_grid, frame.pitch, frame.roll);

                // Calcular bounding box de la huella sobre el DEM
                let c1 = basis.pixel_to_ray(0.0, 0.0, cam_pos, &frame_intrinsics);
                let c2 = basis.pixel_to_ray(w_px, 0.0, cam_pos, &frame_intrinsics);
                let c3 = basis.pixel_to_ray(w_px, h_px, cam_pos, &frame_intrinsics);
                let c4 = basis.pixel_to_ray(0.0, h_px, cam_pos, &frame_intrinsics);

                let p1 = dem.intersect_ray(&c1, 3000.0, 16);
                let p2 = dem.intersect_ray(&c2, 3000.0, 16);
                let p3 = dem.intersect_ray(&c3, 3000.0, 16);
                let p4 = dem.intersect_ray(&c4, 3000.0, 16);

                let bbox = if let (Some(pt1), Some(pt2), Some(pt3), Some(pt4)) = (p1, p2, p3, p4) {
                    let pts = [pt1, pt2, pt3, pt4];
                    let min_x = (pts.iter().map(|p| p.x).fold(f64::INFINITY, f64::min) - 5.0)
                        .max(cam_x - 420.0);
                    let max_x = (pts.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max) + 5.0)
                        .min(cam_x + 420.0);
                    let min_y = (pts.iter().map(|p| p.y).fold(f64::INFINITY, f64::min) - 5.0)
                        .max(cam_y - 420.0);
                    let max_y = (pts.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max) + 5.0)
                        .min(cam_y + 420.0);
                    [min_x, min_y, max_x, max_y]
                } else {
                    [cam_x - 300.0, cam_y - 300.0, cam_x + 300.0, cam_y + 300.0]
                };

                // Puntos contraídos para el Casco Convexo usando intrínsecos del fotograma actual
                let f_half_w = w_px * 0.5;
                let f_half_h = h_px * 0.5;
                let f_u_min = frame_intrinsics.cx - f_half_w * crop_f;
                let f_u_max = frame_intrinsics.cx + f_half_w * crop_f;
                let f_v_min = frame_intrinsics.cy - f_half_h * crop_f;
                let f_v_max = frame_intrinsics.cy + f_half_h * crop_f;
                let hc1 = basis.pixel_to_ray(f_u_min, f_v_min, cam_pos, &frame_intrinsics);
                let hc2 = basis.pixel_to_ray(f_u_max, f_v_min, cam_pos, &frame_intrinsics);
                let hc3 = basis.pixel_to_ray(f_u_max, f_v_max, cam_pos, &frame_intrinsics);
                let hc4 = basis.pixel_to_ray(f_u_min, f_v_max, cam_pos, &frame_intrinsics);

                if let Some(hp) = dem.intersect_ray(&hc1, 3000.0, 16) {
                    hull_points.push(Point2D::new(hp.x, hp.y));
                }
                if let Some(hp) = dem.intersect_ray(&hc2, 3000.0, 16) {
                    hull_points.push(Point2D::new(hp.x, hp.y));
                }
                if let Some(hp) = dem.intersect_ray(&hc3, 3000.0, 16) {
                    hull_points.push(Point2D::new(hp.x, hp.y));
                }
                if let Some(hp) = dem.intersect_ray(&hc4, 3000.0, 16) {
                    hull_points.push(Point2D::new(hp.x, hp.y));
                }

                let pose = CameraPose {
                    position: cam_pos,
                    yaw_deg: yaw_grid,
                    pitch_deg: frame.pitch,
                    roll_deg: frame.roll,
                };

                image_frames.push(ImageFrame::new_with_bbox(
                    frame.filename.clone(),
                    pose,
                    frame_intrinsics,
                    floats,
                    frame_intrinsics.width as usize,
                    frame_intrinsics.height as usize,
                    bbox,
                ));
            }

            info!(
                "Fotogramas listos para ortorrectificación: {} (Descartados tránsito: {}, Faltantes raw: {})",
                image_frames.len(),
                ignored_transit,
                missing_raw
            );

            if let Some(ref tp_path) = tie_points {
                match std::fs::read_to_string(tp_path) {
                    Ok(tp_json) => match serde_json::from_str::<TiePointFile>(&tp_json) {
                        Ok(tie_file) => {
                            info!(
                                "Ejecutando refinamiento de poses 2.5D con {} puntos de enlace y {} overrides...",
                                tie_file.matches.len(),
                                tie_file.pose_overrides.len()
                            );
                            let report =
                                refine_poses_with_tie_points(&mut image_frames, &dem, &tie_file);
                            println!(
                                "[ZENITH_POSE_REFINEMENT] pre_rmse_m={:.4} post_rmse_m={:.4} holdout_pre_rmse_m={:.4} holdout_post_rmse_m={:.4} inlier_ratio={:.4}",
                                report.pre_refinement_rmse_m,
                                report.post_refinement_rmse_m,
                                report.holdout_pre_rmse_m,
                                report.holdout_post_rmse_m,
                                report.inlier_ratio
                            );
                            if let Some(ref export_path) = export_refined_poses {
                                let export_payload = serde_json::json!({
                                    "schema_version": "1.0",
                                    "model": "2.5D_horizontal_yaw",
                                    "report": &report,
                                    "pose_overrides": &report.applied_deltas,
                                });
                                if let Some(parent) = export_path.parent() {
                                    std::fs::create_dir_all(parent).ok();
                                }
                                std::fs::write(
                                    export_path,
                                    serde_json::to_string_pretty(&export_payload)?,
                                )?;
                                info!(
                                    "Poses refinadas exportadas exitosamente a: {}",
                                    export_path.display()
                                );
                            }

                            // Actualizar puntos del casco convexo con las poses refinadas
                            if !hull_points.is_empty() {
                                hull_points.clear();
                                for frame in &image_frames {
                                    let w_px = frame.intrinsics.width as f64;
                                    let h_px = frame.intrinsics.height as f64;
                                    let f_half_w = w_px * 0.5;
                                    let f_half_h = h_px * 0.5;
                                    let f_u_min = frame.intrinsics.cx - f_half_w * center_crop;
                                    let f_u_max = frame.intrinsics.cx + f_half_w * center_crop;
                                    let f_v_min = frame.intrinsics.cy - f_half_h * center_crop;
                                    let f_v_max = frame.intrinsics.cy + f_half_h * center_crop;
                                    let hc1 = frame.basis.pixel_to_ray(
                                        f_u_min,
                                        f_v_min,
                                        frame.pose.position,
                                        &frame.intrinsics,
                                    );
                                    let hc2 = frame.basis.pixel_to_ray(
                                        f_u_max,
                                        f_v_min,
                                        frame.pose.position,
                                        &frame.intrinsics,
                                    );
                                    let hc3 = frame.basis.pixel_to_ray(
                                        f_u_max,
                                        f_v_max,
                                        frame.pose.position,
                                        &frame.intrinsics,
                                    );
                                    let hc4 = frame.basis.pixel_to_ray(
                                        f_u_min,
                                        f_v_max,
                                        frame.pose.position,
                                        &frame.intrinsics,
                                    );
                                    if let Some(hp) = dem.intersect_ray(&hc1, 3000.0, 16) {
                                        hull_points.push(Point2D::new(hp.x, hp.y));
                                    }
                                    if let Some(hp) = dem.intersect_ray(&hc2, 3000.0, 16) {
                                        hull_points.push(Point2D::new(hp.x, hp.y));
                                    }
                                    if let Some(hp) = dem.intersect_ray(&hc3, 3000.0, 16) {
                                        hull_points.push(Point2D::new(hp.x, hp.y));
                                    }
                                    if let Some(hp) = dem.intersect_ray(&hc4, 3000.0, 16) {
                                        hull_points.push(Point2D::new(hp.x, hp.y));
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            warn!(
                                "Error deserializando archivo de tie points {}: {}",
                                tp_path.display(),
                                e
                            );
                        }
                    },
                    Err(e) => {
                        warn!(
                            "Error leyendo archivo de tie points {}: {}",
                            tp_path.display(),
                            e
                        );
                    }
                }
            }

            let hull = if hull_points.len() >= 3 {
                let h = ConvexHull2D::from_points(&hull_points);
                if let Some(ref h_obj) = h {
                    info!(
                        "Casco Convexo térmico calculado: {} vértices. Extensión: X [{:.1}, {:.1}], Y [{:.1}, {:.1}]",
                        h_obj.vertices.len(),
                        h_obj.min_x,
                        h_obj.max_x,
                        h_obj.min_y,
                        h_obj.max_y
                    );
                    if let Some(ref geojson_path) = hull_geojson {
                        if let Some(parent) = geojson_path.parent() {
                            std::fs::create_dir_all(parent).ok();
                        }
                        std::fs::write(
                            geojson_path,
                            h_obj.to_geojson_crs("zenith_thermal_hull", &resolved_crs),
                        )?;
                        info!("Casco Convexo exportado a: {}", geojson_path.display());
                    }
                }
                h
            } else {
                None
            };

            let trim_mode: TrimMode = trim.parse().unwrap_or(TrimMode::Both);
            info!(
                "Modo de recorte espacial: {:?}, Solape mínimo: {}",
                trim_mode, min_overlap
            );

            let resampling_method = match resampling.to_lowercase().as_str() {
                "bilinear" => ResamplingMethod::Bilinear,
                _ => ResamplingMethod::NearestNeighbor,
            };

            let ortho_config = OrthoConfig {
                target_gsd_m: gsd,
                resampling: resampling_method,
                nodata_value: -3.4028235e38,
                trim_mode,
                min_overlap,
                hull,
                feather_px,
                max_blend_temp_diff,
            };

            info!(
                "Ejecutando mapeo de celdas multihilo (Rayon) con {:?}, feather_px={:.1}, max_blend_temp_diff={:.1}°C...",
                resampling_method, feather_px, max_blend_temp_diff
            );
            emit_progress("ortho", 50, "Calculando ortomosaico térmico...");
            let result =
                DirectOrthoEngine::generate_orthomosaic(&image_frames, &dem, &ortho_config);

            info!(
                "Ortomosaico generado en {} ms. Dimensiones: {}x{} ({:.2} MP), Píxeles válidos: {}",
                result.elapsed_ms,
                result.width,
                result.height,
                (result.width * result.height) as f64 / 1_000_000.0,
                result.valid_pixels
            );

            // Guardar producto crudo ENVI (.raw y .hdr)
            let out_raw = output.with_extension("raw");
            let out_hdr = output.with_extension("hdr");

            let byte_slice: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    result.data.as_ptr() as *const u8,
                    result.data.len() * std::mem::size_of::<f32>(),
                )
            };
            std::fs::write(&out_raw, byte_slice)?;

            let hdr_content = format!(
                "ENVI\ndescription = {{ Zenith 2.5D Direct Thermal Orthomosaic }}\nsamples = {}\nlines   = {}\nbands   = 1\nheader offset = 0\nfile type = ENVI Standard\ndata type = 4\ninterleave = bsq\nbyte order = 0\nmap info = {{Transverse Mercator, 1, 1, {:.4}, {:.4}, {:.4}, {:.4}}}\nprojection info = {{3, 6378137, 6356752.314140356, 4, -73, 5000000, 2000000, 0.9992, Transverse Mercator}}\ncoordinate system string = {{PROJCS[\"MAGNA-SIRGAS_2018_Origen-Nacional\",GEOGCS[\"MAGNA-SIRGAS_2018\",DATUM[\"Marco_Geocentrico_Nacional_de_Referencia_2018\",SPHEROID[\"GRS_1980\",6378137.0,298.257222101]],PRIMEM[\"Greenwich\",0.0],UNIT[\"Degree\",0.0174532925199433]],PROJECTION[\"Transverse_Mercator\"],PARAMETER[\"False_Easting\",5000000.0],PARAMETER[\"False_Northing\",2000000.0],PARAMETER[\"Central_Meridian\",-73.0],PARAMETER[\"Scale_Factor\",0.9992],PARAMETER[\"Latitude_Of_Origin\",4.0],UNIT[\"Meter\",1.0]]}}\ndata ignore value = -3.4028235e+38\n",
                result.width, result.height, result.origin_x, result.origin_y, result.gsd, result.gsd
            );
            std::fs::write(&out_hdr, hdr_content)?;

            // Convertir a GeoTIFF comprimido si GDAL está disponible
            let tif_out = output.with_extension("tif");
            emit_progress("geotiff", 85, "Generando GeoTIFF comprimido...");
            let gdal_status = std::process::Command::new("gdal_translate")
                .args([
                    "-of",
                    "GTiff",
                    "-co",
                    "COMPRESS=DEFLATE",
                    "-co",
                    "PREDICTOR=3",
                    "-a_nodata",
                    "-3.4028235e+38",
                    "-a_srs",
                    &resolved_crs,
                ])
                .arg(&out_raw)
                .arg(&tif_out)
                .output();

            if let Ok(st) = gdal_status {
                if st.status.success() {
                    info!("GeoTIFF generado exitosamente: {}", tif_out.display());
                }
            }

            // Escribir manifiesto de contrato Nivel 1
            let total_sec = t0.elapsed().as_secs_f64();
            let thermal_params_path = raw_dir.join("thermal_params.json");
            let thermal_extraction_meta = match std::fs::read_to_string(&thermal_params_path) {
                Ok(s) => match serde_json::from_str::<serde_json::Value>(&s) {
                    Ok(v) => v,
                    Err(e) => {
                        log::warn!(
                            "thermal_params.json malformado en {}: {}",
                            thermal_params_path.display(),
                            e
                        );
                        serde_json::Value::Null
                    }
                },
                Err(_) => {
                    log::warn!(
                        "Sidecar thermal_params.json no encontrado en {}; se registrará null en el manifiesto",
                        thermal_params_path.display()
                    );
                    serde_json::Value::Null
                }
            };
            let manifest_obj = serde_json::json!({
                "schema_version": "1.0",
                "engine_name": "Zenith Direct 2.5D Photogrammetry Engine",
                "engine_version": env!("CARGO_PKG_VERSION"),
                "tier": "level1_rapid_ortho",
                "elapsed_seconds": total_sec,
                "dem_used": dem_meta["kind"] != "flat_plane",
                "dem": dem_meta,
                "datum_vertical_offset_m": effective_datum_offset,
                "lrf_calibration": lrf_meta,
                "frames_total": scan.frames.len(),
                "frames_nadir_mapped": image_frames.len(),
                "frames_transit_filtered": ignored_transit,
                "gsd_m": gsd,
                "crs": &resolved_crs,
                "feather_px": feather_px,
                "max_blend_temp_diff_c": max_blend_temp_diff,
                "resampling": format!("{:?} (feather_px={:.1}, max_diff={:.1}°C)", resampling_method, feather_px, max_blend_temp_diff),
                "thermal_extraction": thermal_extraction_meta,
                "products": [
                    tif_out.to_string_lossy().to_string(),
                    out_raw.to_string_lossy().to_string()
                ]
            });

            std::fs::write(&manifest, serde_json::to_string_pretty(&manifest_obj)?)?;
            info!("Manifiesto Nivel 1 escrito en: {}", manifest.display());

            println!("============================================================");
            println!("  ZENITH: ORTOMOSAICO TÉRMICO DIRECTO COMPLETADO");
            println!("============================================================");
            println!("  Tiempo total: {:.2} s", total_sec);
            println!(
                "  Fotogramas mapeados: {} / {}",
                image_frames.len(),
                scan.frames.len()
            );
            println!(
                "  Dimensiones: {} x {} px ({:.2} MP)",
                result.width,
                result.height,
                (result.width * result.height) as f64 / 1_000_000.0
            );
            println!("  Píxeles válidos: {}", result.valid_pixels);
            println!("  Resolución (GSD): {:.3} m/px", gsd);
            println!("  GeoTIFF: {}", tif_out.display());
            println!("  Manifiesto: {}", manifest.display());
            println!("============================================================");
            emit_progress("done", 100, "Procesamiento térmico completado");
        }

        Commands::OrthoRgb {
            images,
            dem,
            dem_source,
            flat_elevation,
            dtm_hdr,
            dtm_raw,
            output,
            gsd,
            datum_offset,
            blend_mode,
            feather_px,
            no_gain_compensation,
            gain_compensation: _,
            dehaze,
            tile_size,
            max_pitch,
            trim,
            min_overlap,
            center_crop,
            hull_geojson,
            manifest,
            crs,
            tie_points,
            export_refined_poses,
        } => {
            let t0 = Instant::now();
            info!("Iniciando ortorrectificación visual RGB 2.5D con Zenith...");
            info!(
                "Escaneando metadatos XMP de imágenes desde: {}",
                images.display()
            );
            let scan = scan_flight_directory(&images)?;
            let frames_total = scan.frames.len();
            info!("Fotogramas escaneados: {}", frames_total);

            let is_thermal_frame = |f: &DjiMetadata| -> bool {
                f.filename.ends_with("_T.JPG")
                    || f.filename.ends_with("_T.jpg")
                    || f.filename.contains("_T_")
                    || (f.image_width == Some(640) && f.image_height == Some(512))
            };
            let frames: Vec<_> = scan
                .frames
                .into_iter()
                .filter(|f| !is_thermal_frame(f))
                .collect();

            if frames.is_empty() {
                return Err("No se encontraron fotogramas visuales RGB válidos (todos son térmicos o el directorio está vacío).".into());
            }
            info!("Fotogramas visuales RGB seleccionados: {}", frames.len());
            emit_progress(
                "scan",
                10,
                &format!("Fotogramas RGB seleccionados: {}", frames.len()),
            );

            let mut sum_lat = 0.0;
            let mut sum_lon = 0.0;
            let mut count_geo = 0;

            for frame in &frames {
                sum_lat += frame.lat;
                sum_lon += frame.lon;
                count_geo += 1;
            }

            let mean_lat = if count_geo > 0 {
                sum_lat / count_geo as f64
            } else {
                0.0
            };
            let mean_lon = if count_geo > 0 {
                sum_lon / count_geo as f64
            } else {
                0.0
            };

            let (tm, resolved_crs) = TransverseMercator::resolve_crs(&crs, mean_lon, mean_lat)?;
            info!("CRS resuelto: {} (solicitado: {})", resolved_crs, crs);
            if resolved_crs == "EPSG:9377" && (mean_lon - (-73.0)).abs() > 7.5 {
                log::warn!(
                    "ADVERTENCIA DE PROYECCIÓN: Las coordenadas (lon: {:.3}, lat: {:.3}) están a más de 7.5° del meridiano central de Colombia EPSG:9377 (-73.0°). La distorsión de escala superará los límites cartográficos recomendados. Considere usar '--crs auto' o UTM local.",
                    mean_lon, mean_lat
                );
            }

            let mut cam_x_min = f64::INFINITY;
            let mut cam_x_max = f64::NEG_INFINITY;
            let mut cam_y_min = f64::INFINITY;
            let mut cam_y_max = f64::NEG_INFINITY;

            for frame in &frames {
                let (cx, cy, _) = tm.forward(frame.lon, frame.lat);
                cam_x_min = cam_x_min.min(cx);
                cam_x_max = cam_x_max.max(cx);
                cam_y_min = cam_y_min.min(cy);
                cam_y_max = cam_y_max.max(cy);
            }

            emit_progress("dem", 25, "Resolviendo modelo de elevación...");
            let (dem, dem_meta) = resolve_dem(
                dem.as_deref(),
                dem_source.unwrap_or(DemSource::Auto),
                flat_elevation,
                dtm_hdr.as_deref(),
                dtm_raw.as_deref(),
                cam_x_min,
                cam_x_max,
                cam_y_min,
                cam_y_max,
                mean_lat,
                mean_lon,
                datum_offset,
                Some(&frames),
                &resolved_crs,
            )?;
            info!(
                "DEM cargado: {} x {} celdas a {:.2} m de resolución",
                dem.width, dem.height, dem.cell_size_x
            );

            emit_progress("lrf", 35, "Calibrando datum vertical LRF...");
            let (effective_datum_offset, lrf_meta) =
                calibrate_lrf_datum_offset(&frames, &dem, &tm, datum_offset);

            let sample_frame = frames.first().ok_or("No hay fotogramas escaneados")?;
            let intrinsics = CameraIntrinsics::from_telemetry(
                sample_frame.image_width.unwrap_or(4000),
                sample_frame.image_height.unwrap_or(3000),
                sample_frame.model.as_deref(),
                sample_frame.calibrated_focal_length,
                sample_frame.calibrated_cx,
                sample_frame.calibrated_cy,
                sample_frame.focal_length_35mm,
                sample_frame.focal_length_mm,
            );
            info!(
                "Intrínsecos de cámara RGB determinados dinámicamente: {}x{} px, fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1} (Modelo: {:?})",
                intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy, sample_frame.model
            );

            let mut rgb_frames = Vec::new();
            let mut hull_points = Vec::new();
            let mut ignored_transit = 0;

            let crop_f = center_crop.clamp(0.1, 1.0);

            for frame in &frames {
                if frame.pitch > max_pitch {
                    ignored_transit += 1;
                    continue;
                }

                let img_path = images.join(&frame.filename);
                if !img_path.exists() {
                    continue;
                }

                let mut frame_intrinsics = CameraIntrinsics::from_telemetry(
                    frame.image_width.unwrap_or(4000),
                    frame.image_height.unwrap_or(3000),
                    frame.model.as_deref(),
                    frame.calibrated_focal_length,
                    frame.calibrated_cx,
                    frame.calibrated_cy,
                    frame.focal_length_35mm,
                    frame.focal_length_mm,
                );
                if frame.dewarp_flag.as_deref() == Some("0")
                    || frame.dewarp_flag.as_deref() == Some("false")
                {
                    if let Some(ref dd) = frame.dewarp_data {
                        frame_intrinsics = frame_intrinsics.with_dji_dewarp(dd);
                    }
                }
                let w_px = frame_intrinsics.width as f64;
                let h_px = frame_intrinsics.height as f64;

                let (cam_x, cam_y, gamma) = tm.forward(frame.lon, frame.lat);
                let cam_z = frame.abs_alt - effective_datum_offset;
                let cam_pos = Vector3::new(cam_x, cam_y, cam_z);
                let yaw_grid = frame.yaw + gamma;

                let basis = CameraBasis::from_gimbal(yaw_grid, frame.pitch, frame.roll);

                let c1 = basis.pixel_to_ray(0.0, 0.0, cam_pos, &frame_intrinsics);
                let c2 = basis.pixel_to_ray(w_px, 0.0, cam_pos, &frame_intrinsics);
                let c3 = basis.pixel_to_ray(w_px, h_px, cam_pos, &frame_intrinsics);
                let c4 = basis.pixel_to_ray(0.0, h_px, cam_pos, &frame_intrinsics);

                let p1 = dem.intersect_ray(&c1, 3000.0, 16);
                let p2 = dem.intersect_ray(&c2, 3000.0, 16);
                let p3 = dem.intersect_ray(&c3, 3000.0, 16);
                let p4 = dem.intersect_ray(&c4, 3000.0, 16);

                let bbox = if let (Some(pt1), Some(pt2), Some(pt3), Some(pt4)) = (p1, p2, p3, p4) {
                    let pts = [pt1, pt2, pt3, pt4];
                    let min_x = pts.iter().map(|p| p.x).fold(f64::INFINITY, f64::min) - 10.0;
                    let max_x = pts.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max) + 10.0;
                    let min_y = pts.iter().map(|p| p.y).fold(f64::INFINITY, f64::min) - 10.0;
                    let max_y = pts.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max) + 10.0;
                    [min_x, min_y, max_x, max_y]
                } else {
                    [cam_x - 300.0, cam_y - 300.0, cam_x + 300.0, cam_y + 300.0]
                };

                // Puntos contraídos para el Casco Convexo usando intrínsecos del fotograma actual
                let f_half_w = w_px * 0.5;
                let f_half_h = h_px * 0.5;
                let f_u_min = frame_intrinsics.cx - f_half_w * crop_f;
                let f_u_max = frame_intrinsics.cx + f_half_w * crop_f;
                let f_v_min = frame_intrinsics.cy - f_half_h * crop_f;
                let f_v_max = frame_intrinsics.cy + f_half_h * crop_f;
                let hc1 = basis.pixel_to_ray(f_u_min, f_v_min, cam_pos, &frame_intrinsics);
                let hc2 = basis.pixel_to_ray(f_u_max, f_v_min, cam_pos, &frame_intrinsics);
                let hc3 = basis.pixel_to_ray(f_u_max, f_v_max, cam_pos, &frame_intrinsics);
                let hc4 = basis.pixel_to_ray(f_u_min, f_v_max, cam_pos, &frame_intrinsics);

                if let Some(hp) = dem.intersect_ray(&hc1, 3000.0, 16) {
                    hull_points.push(Point2D::new(hp.x, hp.y));
                }
                if let Some(hp) = dem.intersect_ray(&hc2, 3000.0, 16) {
                    hull_points.push(Point2D::new(hp.x, hp.y));
                }
                if let Some(hp) = dem.intersect_ray(&hc3, 3000.0, 16) {
                    hull_points.push(Point2D::new(hp.x, hp.y));
                }
                if let Some(hp) = dem.intersect_ray(&hc4, 3000.0, 16) {
                    hull_points.push(Point2D::new(hp.x, hp.y));
                }

                let pose = CameraPose {
                    position: cam_pos,
                    yaw_deg: yaw_grid,
                    pitch_deg: frame.pitch,
                    roll_deg: frame.roll,
                };

                rgb_frames.push(RgbFrameMeta {
                    id: frame.filename.clone(),
                    file_path: img_path,
                    pose,
                    intrinsics: frame_intrinsics,
                    basis,
                    bbox,
                });
            }

            info!(
                "Fotogramas RGB listos para ortorrectificación: {} (Descartados tránsito: {})",
                rgb_frames.len(),
                ignored_transit
            );

            if let Some(ref tp_path) = tie_points {
                match std::fs::read_to_string(tp_path) {
                    Ok(tp_json) => match serde_json::from_str::<TiePointFile>(&tp_json) {
                        Ok(tie_file) => {
                            info!(
                                "Ejecutando refinamiento de poses 2.5D con {} puntos de enlace y {} overrides...",
                                tie_file.matches.len(),
                                tie_file.pose_overrides.len()
                            );
                            let report =
                                refine_poses_with_tie_points(&mut rgb_frames, &dem, &tie_file);
                            println!(
                                "[ZENITH_POSE_REFINEMENT] pre_rmse_m={:.4} post_rmse_m={:.4} holdout_pre_rmse_m={:.4} holdout_post_rmse_m={:.4} inlier_ratio={:.4}",
                                report.pre_refinement_rmse_m,
                                report.post_refinement_rmse_m,
                                report.holdout_pre_rmse_m,
                                report.holdout_post_rmse_m,
                                report.inlier_ratio
                            );
                            if let Some(ref export_path) = export_refined_poses {
                                let export_payload = serde_json::json!({
                                    "schema_version": "1.0",
                                    "model": "2.5D_horizontal_yaw",
                                    "report": &report,
                                    "pose_overrides": &report.applied_deltas,
                                });
                                if let Some(parent) = export_path.parent() {
                                    std::fs::create_dir_all(parent).ok();
                                }
                                std::fs::write(
                                    export_path,
                                    serde_json::to_string_pretty(&export_payload)?,
                                )?;
                                info!(
                                    "Poses refinadas exportadas exitosamente a: {}",
                                    export_path.display()
                                );
                            }

                            // Actualizar puntos del casco convexo con las poses refinadas
                            if !hull_points.is_empty() {
                                hull_points.clear();
                                for frame in &rgb_frames {
                                    let w_px = frame.intrinsics.width as f64;
                                    let h_px = frame.intrinsics.height as f64;
                                    let f_half_w = w_px * 0.5;
                                    let f_half_h = h_px * 0.5;
                                    let f_u_min = frame.intrinsics.cx - f_half_w * center_crop;
                                    let f_u_max = frame.intrinsics.cx + f_half_w * center_crop;
                                    let f_v_min = frame.intrinsics.cy - f_half_h * center_crop;
                                    let f_v_max = frame.intrinsics.cy + f_half_h * center_crop;
                                    let hc1 = frame.basis.pixel_to_ray(
                                        f_u_min,
                                        f_v_min,
                                        frame.pose.position,
                                        &frame.intrinsics,
                                    );
                                    let hc2 = frame.basis.pixel_to_ray(
                                        f_u_max,
                                        f_v_min,
                                        frame.pose.position,
                                        &frame.intrinsics,
                                    );
                                    let hc3 = frame.basis.pixel_to_ray(
                                        f_u_max,
                                        f_v_max,
                                        frame.pose.position,
                                        &frame.intrinsics,
                                    );
                                    let hc4 = frame.basis.pixel_to_ray(
                                        f_u_min,
                                        f_v_max,
                                        frame.pose.position,
                                        &frame.intrinsics,
                                    );
                                    if let Some(hp) = dem.intersect_ray(&hc1, 3000.0, 16) {
                                        hull_points.push(Point2D::new(hp.x, hp.y));
                                    }
                                    if let Some(hp) = dem.intersect_ray(&hc2, 3000.0, 16) {
                                        hull_points.push(Point2D::new(hp.x, hp.y));
                                    }
                                    if let Some(hp) = dem.intersect_ray(&hc3, 3000.0, 16) {
                                        hull_points.push(Point2D::new(hp.x, hp.y));
                                    }
                                    if let Some(hp) = dem.intersect_ray(&hc4, 3000.0, 16) {
                                        hull_points.push(Point2D::new(hp.x, hp.y));
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            warn!(
                                "Error deserializando archivo de tie points {}: {}",
                                tp_path.display(),
                                e
                            );
                        }
                    },
                    Err(e) => {
                        warn!(
                            "Error leyendo archivo de tie points {}: {}",
                            tp_path.display(),
                            e
                        );
                    }
                }
            }

            let hull = if hull_points.len() >= 3 {
                let h = ConvexHull2D::from_points(&hull_points);
                if let Some(ref h_obj) = h {
                    info!(
                        "Casco Convexo RGB calculado: {} vértices. Extensión: X [{:.1}, {:.1}], Y [{:.1}, {:.1}]",
                        h_obj.vertices.len(),
                        h_obj.min_x,
                        h_obj.max_x,
                        h_obj.min_y,
                        h_obj.max_y
                    );
                    if let Some(ref geojson_path) = hull_geojson {
                        if let Some(parent) = geojson_path.parent() {
                            std::fs::create_dir_all(parent).ok();
                        }
                        std::fs::write(
                            geojson_path,
                            h_obj.to_geojson_crs("zenith_rgb_hull", &resolved_crs),
                        )?;
                        info!("Casco Convexo exportado a: {}", geojson_path.display());
                    }
                }
                h
            } else {
                None
            };

            let rgb_blend_mode = match blend_mode.to_lowercase().as_str() {
                "best-nadir" => RgbBlendMode::BestNadir,
                "feather" => RgbBlendMode::Feather,
                _ => RgbBlendMode::MicroFeather,
            };

            let trim_mode: TrimMode = trim.parse().unwrap_or(TrimMode::Both);
            info!(
                "Modo de recorte espacial RGB: {:?}, Solape mínimo: {}, Modo mezcla: {:?}",
                trim_mode, min_overlap, rgb_blend_mode
            );

            let effective_gsd = if gsd > 0.001 {
                gsd
            } else {
                let mut agls = Vec::new();
                for rf in &rgb_frames {
                    if let Some(z_dem) =
                        dem.sample_elevation_bilinear(rf.pose.position.x, rf.pose.position.y)
                    {
                        let h = rf.pose.position.z - z_dem;
                        if h > 15.0 && h < 1500.0 {
                            agls.push(h);
                        }
                    }
                }
                agls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let median_agl = if !agls.is_empty() {
                    agls[agls.len() / 2]
                } else {
                    120.0
                };
                let native_gsd = (median_agl / intrinsics.fx.max(500.0)).clamp(0.025, 0.15);
                let span_x = (cam_x_max - cam_x_min + 220.0).max(100.0);
                let span_y = (cam_y_max - cam_y_min + 220.0).max(100.0);
                let min_gsd_for_canvas = (span_x.max(span_y) / 9500.0).max(0.045);
                (native_gsd * 1.25)
                    .max(min_gsd_for_canvas)
                    .clamp(0.045, 0.11)
            };
            info!(
                "GSD objetivo RGB seleccionado: {:.4} m/px ({:.1} cm/px)",
                effective_gsd,
                effective_gsd * 100.0
            );

            let ortho_config = RgbOrthoConfig {
                target_gsd_m: effective_gsd,
                blend_mode: rgb_blend_mode,
                trim_mode,
                min_overlap,
                hull,
                feather_px,
                tile_size,
                max_cached_images: 32,
                gain_compensation: !no_gain_compensation,
                dehaze,
            };

            info!(
                "Ejecutando mapeo de teselas RGB multihilo (Rayon) en modo {:?}...",
                rgb_blend_mode
            );
            emit_progress("ortho", 50, "Mapeando teselas RGB multihilo...");
            let result = RgbOrthoEngine::generate_orthomosaic(&rgb_frames, &dem, &ortho_config);

            info!(
                "Ortomosaico RGB generado en {} ms ({:.2} s). Dimensiones: {}x{} ({:.2} MP), Píxeles válidos: {}",
                result.elapsed_ms,
                result.elapsed_ms as f64 / 1000.0,
                result.width,
                result.height,
                (result.width * result.height) as f64 / 1_000_000.0,
                result.valid_pixels
            );

            // Guardar producto crudo ENVI (.raw y .hdr)
            let out_raw = output.with_extension("raw");
            let out_hdr = output.with_extension("hdr");

            std::fs::write(&out_raw, &result.rgba_data)?;

            let hdr_content = format!(
                "ENVI\ndescription = {{ Zenith 2.5D Direct RGB Orthomosaic }}\nsamples = {}\nlines   = {}\nbands   = 4\nheader offset = 0\nfile type = ENVI Standard\ndata type = 1\ninterleave = bip\nbyte order = 0\nmap info = {{Transverse Mercator, 1, 1, {:.4}, {:.4}, {:.4}, {:.4}}}\nprojection info = {{3, 6378137, 6356752.314140356, 4, -73, 5000000, 2000000, 0.9992, Transverse Mercator}}\ncoordinate system string = {{PROJCS[\"MAGNA-SIRGAS_2018_Origen-Nacional\",GEOGCS[\"MAGNA-SIRGAS_2018\",DATUM[\"Marco_Geocentrico_Nacional_de_Referencia_2018\",SPHEROID[\"GRS_1980\",6378137.0,298.257222101]],PRIMEM[\"Greenwich\",0.0],UNIT[\"Degree\",0.0174532925199433]],PROJECTION[\"Transverse_Mercator\"],PARAMETER[\"False_Easting\",5000000.0],PARAMETER[\"False_Northing\",2000000.0],PARAMETER[\"Central_Meridian\",-73.0],PARAMETER[\"Scale_Factor\",0.9992],PARAMETER[\"Latitude_Of_Origin\",4.0],UNIT[\"Meter\",1.0]]}}\nband names = {{ Red, Green, Blue, Alpha }}\n",
                result.width, result.height, result.origin_x, result.origin_y, result.gsd, result.gsd
            );
            std::fs::write(&out_hdr, hdr_content)?;

            // Convertir a GeoTIFF comprimido RGBA
            let tif_out = output.with_extension("tif");
            emit_progress("geotiff", 85, "Generando GeoTIFF RGBA...");
            let gdal_status = std::process::Command::new("gdal_translate")
                .args([
                    "-of",
                    "GTiff",
                    "-co",
                    "PHOTOMETRIC=RGB",
                    "-co",
                    "ALPHA=YES",
                    "-co",
                    "COMPRESS=DEFLATE",
                    "-co",
                    "PREDICTOR=2",
                    "-a_srs",
                    &resolved_crs,
                ])
                .arg(&out_raw)
                .arg(&tif_out)
                .output();

            if let Ok(st) = gdal_status {
                if st.status.success() {
                    info!("GeoTIFF RGBA generado exitosamente: {}", tif_out.display());
                }
            }

            let total_sec = t0.elapsed().as_secs_f64();
            let manifest_obj = serde_json::json!({
                "schema_version": "1.0",
                "engine_name": "Zenith Direct 2.5D Photogrammetry Engine",
                "engine_version": env!("CARGO_PKG_VERSION"),
                "tier": "level1_rapid_rgb_ortho",
                "elapsed_seconds": total_sec,
                "dem_used": dem_meta["kind"] != "flat_plane",
                "dem": dem_meta,
                "datum_vertical_offset_m": effective_datum_offset,
                "lrf_calibration": lrf_meta,
                "gain_compensation": !no_gain_compensation,
                "dehaze": dehaze,
                "blend_mode": format!("{:?}", rgb_blend_mode),
                "frames_total": frames_total,
                "frames_nadir_mapped": rgb_frames.len(),
                "frames_transit_filtered": ignored_transit,
                "gsd_m": effective_gsd,
                "feather_px": feather_px,
                "crs": &resolved_crs,
                "resampling": "Catmull-Rom Bicubic with Narrow-Band Voronoi Feathering",
                "products": [
                    tif_out.to_string_lossy().to_string(),
                    out_raw.to_string_lossy().to_string()
                ]
            });

            std::fs::write(&manifest, serde_json::to_string_pretty(&manifest_obj)?)?;
            info!("Manifiesto Nivel 1 RGB escrito en: {}", manifest.display());

            println!("============================================================");
            println!("  ZENITH: ORTOMOSAICO RGB DIRECTO 2.5D COMPLETADO");
            println!("============================================================");
            println!(
                "  Tiempo total: {:.2} s ({:.2} min)",
                total_sec,
                total_sec / 60.0
            );
            println!(
                "  Fotogramas mapeados: {} / {}",
                rgb_frames.len(),
                frames_total
            );
            println!(
                "  Dimensiones: {} x {} px ({:.2} MP)",
                result.width,
                result.height,
                (result.width * result.height) as f64 / 1_000_000.0
            );
            println!("  Píxeles válidos: {}", result.valid_pixels);
            println!("  Resolución (GSD): {:.3} m/px", effective_gsd);
            println!("  Feathering rampa: {:.1} px", feather_px);
            println!("  GeoTIFF RGBA: {}", tif_out.display());
            println!("  Manifiesto: {}", manifest.display());
            println!("============================================================");
            emit_progress("done", 100, "Procesamiento RGB completado");
        }

        Commands::Benchmark {
            frames,
            dem_size,
            gsd,
        } => {
            println!("Iniciando Benchmark de Alto Rendimiento Zenith...");
            println!("  Fotogramas: {}", frames);
            println!("  Rejilla DEM: {}x{} celdas", dem_size, dem_size);
            println!("  GSD: {} m/px", gsd);

            let intrinsics = CameraIntrinsics::dji_m3t_thermal();
            let mut image_frames = Vec::new();
            let dummy_pixels = vec![30.5f32; 640 * 512];

            for i in 0..frames {
                let x = 1000.0 + (i as f64) * 20.0;
                let y = 1000.0 + (i as f64) * 15.0;
                let z = 2200.0;
                let pose = CameraPose {
                    position: Vector3::new(x, y, z),
                    yaw_deg: 45.0,
                    pitch_deg: -88.0,
                    roll_deg: 0.0,
                };
                image_frames.push(ImageFrame::new(
                    format!("FRAME_{:04}", i),
                    pose,
                    intrinsics,
                    dummy_pixels.clone(),
                    640,
                    512,
                ));
            }

            let dem_elev = vec![1800.0f32; dem_size * dem_size];
            let dem = DemWindow::new(
                dem_size,
                dem_size,
                900.0,
                3000.0,
                2.0,
                2.0,
                dem_elev,
                Some(-24.0),
                -9999.0,
            );

            let config = OrthoConfig {
                target_gsd_m: gsd,
                resampling: ResamplingMethod::NearestNeighbor,
                nodata_value: f32::NAN,
                trim_mode: TrimMode::None,
                min_overlap: 1,
                hull: None,
                feather_px: 0.0,
                max_blend_temp_diff: 3.5,
            };

            let t_start = Instant::now();
            let res = DirectOrthoEngine::generate_orthomosaic(&image_frames, &dem, &config);
            let elapsed_total = t_start.elapsed();

            println!("------------------------------------------------------------");
            println!("Benchmark completado:");
            println!("  Tiempo de cálculo de ortomosaico: {} ms", res.elapsed_ms);
            println!(
                "  Tiempo total (incluyendo inicialización): {:?}",
                elapsed_total
            );
            println!("  Dimensiones generadas: {} x {} px", res.width, res.height);
            println!("  Total de celdas procesadas: {}", res.width * res.height);
            println!("  Píxeles proyectados válidos: {}", res.valid_pixels);
            println!(
                "  Rendimiento: {:.2} megapíxeles / segundo",
                (res.width * res.height) as f64 / (res.elapsed_ms as f64 / 1000.0) / 1_000_000.0
            );
            println!("------------------------------------------------------------");
        }

        Commands::ValidateContract { manifest } => {
            println!(
                "Validando contrato fotogramétrico en: {}",
                manifest.display()
            );
            let content = std::fs::read_to_string(&manifest)?;
            let val: serde_json::Value = serde_json::from_str(&content)?;

            let required_fields = [
                "schema_version",
                "engine_name",
                "engine_version",
                "tier",
                "elapsed_seconds",
                "dem_used",
                "gsd_m",
                "resampling",
                "products",
            ];

            let mut missing = Vec::new();
            for f in &required_fields {
                if val.get(*f).is_none() {
                    missing.push(*f);
                }
            }

            if !missing.is_empty() {
                eprintln!(
                    "Error: El manifiesto no cumple con el contrato. Campos faltantes: {:?}",
                    missing
                );
                std::process::exit(1);
            }

            println!("Validación exitosa: El manifiesto cumple al 100% con el Contrato Fotogramétrico de RAPTOR.");
            println!("  Motor: {}", val["engine_name"]);
            println!("  Nivel: {}", val["tier"]);
            println!("  Tiempo registrado: {} s", val["elapsed_seconds"]);
            println!("  Remuestreo: {}", val["resampling"]);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_dem() -> DemWindow {
        DemWindow::from_flat(4999000.0, 5001000.0, 1999000.0, 2001000.0, 10.0, 1500.0)
    }

    fn dummy_frame(alt: f64, pitch: f64, lrf_dist: Option<f64>) -> DjiMetadata {
        DjiMetadata {
            filename: "DJI_0001.JPG".to_string(),
            lon: -73.0,
            lat: 4.0,
            abs_alt: alt,
            rel_alt: 100.0,
            yaw: 0.0,
            pitch,
            roll: 0.0,
            rtk_std_lon: None,
            rtk_std_lat: None,
            rtk_flag: Some("16".to_string()),
            lrf_target_distance: lrf_dist,
            lrf_target_lon: Some(-73.0),
            lrf_target_lat: Some(4.0),
            lrf_target_abs_alt: None,
            model: Some("ZH20T".to_string()),
            image_width: Some(640),
            image_height: Some(512),
            focal_length_mm: None,
            focal_length_35mm: None,
            calibrated_focal_length: None,
            calibrated_cx: None,
            calibrated_cy: None,
            dewarp_data: None,
            dewarp_flag: None,
        }
    }

    #[test]
    fn test_calibrate_lrf_datum_offset_user_override() {
        let dem = dummy_dem();
        let tm = TransverseMercator::colombia_origen_nacional();
        let frames = vec![dummy_frame(1600.0, -90.0, Some(80.0)); 6];

        let (offset, meta) = calibrate_lrf_datum_offset(&frames, &dem, &tm, -24.0);
        assert_eq!(offset, -24.0);
        assert!(meta.is_none());
    }

    #[test]
    fn test_calibrate_lrf_datum_offset_success() {
        let dem = dummy_dem();
        let tm = TransverseMercator::colombia_origen_nacional();
        // abs_alt = 1600.0, pitch = -90.0, d_lrf = 76.0 => est_ground = 1600 - 76 = 1524.0
        // dem = 1500.0 => diff = +24.0 m
        let frames = vec![dummy_frame(1600.0, -90.0, Some(76.0)); 8];

        let (offset, meta) = calibrate_lrf_datum_offset(&frames, &dem, &tm, 0.0);
        assert!((offset - 24.0).abs() < 1e-3);
        assert!(meta.is_some());
        let m = meta.unwrap();
        assert_eq!(m["calibrated"], true);
        assert!((m["lrf_datum_offset_m"].as_f64().unwrap() - 24.0).abs() < 1e-3);
    }

    #[test]
    fn test_calibrate_lrf_datum_offset_below_threshold() {
        let dem = dummy_dem();
        let tm = TransverseMercator::colombia_origen_nacional();
        // abs_alt = 1600.0, pitch = -90.0, d_lrf = 98.0 => est_ground = 1502.0 => diff = 2.0 m (< 5.0m)
        let frames = vec![dummy_frame(1600.0, -90.0, Some(98.0)); 8];

        let (offset, meta) = calibrate_lrf_datum_offset(&frames, &dem, &tm, 0.0);
        assert_eq!(offset, 0.0);
        assert!(meta.is_none());
    }

    #[test]
    fn test_calibrate_lrf_datum_offset_insufficient_samples() {
        let dem = dummy_dem();
        let tm = TransverseMercator::colombia_origen_nacional();
        let frames = vec![dummy_frame(1600.0, -90.0, Some(76.0)); 3];

        let (offset, meta) = calibrate_lrf_datum_offset(&frames, &dem, &tm, 0.0);
        assert_eq!(offset, 0.0);
        assert!(meta.is_none());
    }
}
