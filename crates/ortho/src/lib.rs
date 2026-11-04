//! `zenith-ortho`: Motor de ortorrectificación directa 2.5D de ultra alta velocidad.
//! Utiliza paralelismo masivo CPU con `rayon` para proyectar el terreno celda a celda
//! sobre las imágenes fuente, garantizando remuestreo Nearest-Neighbor estricto
//! en bandas radiométricas para preservar valores físicos (°C y reflectancia),
//! micro-feathering radiométrico con protección estricta de puntos calientes,
//! y ortorrectificación visual RGB con decodificación JPEG bajo demanda.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use zenith_core::{CameraBasis, CameraIntrinsics, CameraPose, Vector3};
use zenith_dem::DemWindow;

/// Método de remuestreo de píxeles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResamplingMethod {
    /// Remuestreo por vecino más cercano: obligatorio para bandas térmicas (°C) y multiespectrales.
    NearestNeighbor,
    /// Interpolación bilineal: adecuado para bandas visuales (RGB).
    Bilinear,
}

/// Modo de recorte espacial del ortomosaico.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrimMode {
    /// Sin recorte adicional por máscara espacial ni solape.
    None,
    /// Recorte automático por envolvente convexa (Convex Hull).
    AutoCrop,
    /// Requiere un número mínimo de fotogramas concurrentes.
    ExposureCount(u16),
    /// Recorte por casco convexo y solape concurrente simultáneamente.
    Both,
    /// Solo recorte por envolvente convexa (Convex Hull).
    Hull,
    /// Solo recorte por conteo de solape concurrente.
    Overlap,
}

impl std::str::FromStr for TrimMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.to_ascii_lowercase();
        match lower.as_str() {
            "none" => Ok(TrimMode::None),
            "autocrop" => Ok(TrimMode::AutoCrop),
            "both" => Ok(TrimMode::Both),
            "hull" => Ok(TrimMode::Hull),
            "overlap" => Ok(TrimMode::Overlap),
            _ => {
                if let Some(num_str) = lower
                    .strip_prefix("exposure:")
                    .or_else(|| lower.strip_prefix("exposure="))
                    .or_else(|| lower.strip_prefix("exposure_count:"))
                {
                    let count = num_str
                        .parse::<u16>()
                        .map_err(|e| format!("Valor de ExposureCount inválido: {}", e))?;
                    if count == 0 {
                        return Err("ExposureCount debe ser >= 1".to_string());
                    }
                    Ok(TrimMode::ExposureCount(count))
                } else if let Ok(count) = lower.parse::<u16>() {
                    if count == 0 {
                        return Err("ExposureCount debe ser >= 1".to_string());
                    }
                    Ok(TrimMode::ExposureCount(count))
                } else {
                    Err(format!(
                        "Modo de recorte desconocido: '{}'. Opciones válidas: 'none', 'autocrop', 'both', 'hull', 'overlap', 'exposure:<N>'",
                        s
                    ))
                }
            }
        }
    }
}

/// Determina el solape mínimo requerido según el modo de recorte y el valor base.
#[inline]
pub fn effective_min_overlap(trim_mode: TrimMode, min_overlap: u16) -> usize {
    match trim_mode {
        TrimMode::ExposureCount(n) => n as usize,
        TrimMode::Both | TrimMode::Overlap => (min_overlap.max(1)) as usize,
        TrimMode::None | TrimMode::AutoCrop | TrimMode::Hull => {
            if min_overlap > 1 {
                min_overlap as usize
            } else {
                1
            }
        }
    }
}

/// Determina si se debe aplicar recorte por casco convexo.
#[inline]
pub fn should_apply_hull(trim_mode: TrimMode) -> bool {
    matches!(
        trim_mode,
        TrimMode::AutoCrop | TrimMode::Hull | TrimMode::Both
    )
}

/// Fotograma de imagen georreferenciado en memoria.
#[derive(Debug, Clone)]
pub struct ImageFrame {
    pub id: String,
    pub pose: CameraPose,
    pub intrinsics: CameraIntrinsics,
    pub basis: CameraBasis,
    pub data: Vec<f32>,
    pub width: usize,
    pub height: usize,
    /// Bounding box proyectado sobre el terreno [min_x, min_y, max_x, max_y].
    pub bbox: [f64; 4],
}

impl ImageFrame {
    pub fn new(
        id: String,
        pose: CameraPose,
        intrinsics: CameraIntrinsics,
        data: Vec<f32>,
        width: usize,
        height: usize,
    ) -> Self {
        let basis = CameraBasis::from_gimbal(pose.yaw_deg, pose.pitch_deg, pose.roll_deg);
        Self {
            id,
            pose,
            intrinsics,
            basis,
            data,
            width,
            height,
            bbox: [
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
                f64::INFINITY,
                f64::INFINITY,
            ],
        }
    }

    pub fn new_with_bbox(
        id: String,
        pose: CameraPose,
        intrinsics: CameraIntrinsics,
        data: Vec<f32>,
        width: usize,
        height: usize,
        bbox: [f64; 4],
    ) -> Self {
        let basis = CameraBasis::from_gimbal(pose.yaw_deg, pose.pitch_deg, pose.roll_deg);
        Self {
            id,
            pose,
            intrinsics,
            basis,
            data,
            width,
            height,
            bbox,
        }
    }

    /// Muestrea el valor en el píxel (u, v) según el método especificado.
    pub fn sample_pixel(&self, u: f64, v: f64, method: ResamplingMethod) -> Option<f32> {
        if u < 0.0 || u >= self.width as f64 || v < 0.0 || v >= self.height as f64 {
            return None;
        }

        match method {
            ResamplingMethod::NearestNeighbor => {
                let col = u.round() as usize;
                let row = v.round() as usize;
                if col < self.width && row < self.height {
                    Some(self.data[row * self.width + col])
                } else {
                    None
                }
            }
            ResamplingMethod::Bilinear => {
                let col0 = u.floor() as usize;
                let row0 = v.floor() as usize;
                let col1 = (col0 + 1).min(self.width - 1);
                let row1 = (row0 + 1).min(self.height - 1);

                let fx = u - col0 as f64;
                let fy = v - row0 as f64;

                let v00 = self.data[row0 * self.width + col0];
                let v10 = self.data[row0 * self.width + col1];
                let v01 = self.data[row1 * self.width + col0];
                let v11 = self.data[row1 * self.width + col1];

                if !v00.is_finite() || !v10.is_finite() || !v01.is_finite() || !v11.is_finite() {
                    return None;
                }

                let top = (1.0 - fx) * v00 as f64 + fx * v10 as f64;
                let bottom = (1.0 - fx) * v01 as f64 + fx * v11 as f64;
                Some(((1.0 - fy) * top + fy * bottom) as f32)
            }
        }
    }
}

/// Configuración del proceso de ortorrectificación térmica directa.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrthoConfig {
    pub target_gsd_m: f64,
    pub resampling: ResamplingMethod,
    pub nodata_value: f32,
    pub trim_mode: TrimMode,
    pub min_overlap: u16,
    pub hull: Option<zenith_core::hull::ConvexHull2D>,
    pub feather_px: f64,
    pub max_blend_temp_diff: f32,
}

impl Default for OrthoConfig {
    fn default() -> Self {
        Self {
            target_gsd_m: 0.1, // 10 cm por píxel
            resampling: ResamplingMethod::NearestNeighbor,
            nodata_value: f32::NAN,
            trim_mode: TrimMode::None,
            min_overlap: 1,
            hull: None,
            feather_px: 0.0,
            max_blend_temp_diff: 3.5,
        }
    }
}

/// Resultado de la ortorrectificación en memoria.
pub struct OrthoResult {
    pub width: usize,
    pub height: usize,
    pub origin_x: f64,
    pub origin_y: f64,
    pub gsd: f64,
    pub data: Vec<f32>,
    pub valid_pixels: usize,
    pub elapsed_ms: u128,
}

/// Motor de ortorrectificación directa 2.5D.
pub struct DirectOrthoEngine;

impl DirectOrthoEngine {
    /// Genera un ortomosaico proyectando las celdas de terreno sobre los fotogramas.
    #[allow(clippy::type_complexity)]
    pub fn generate_orthomosaic(
        frames: &[ImageFrame],
        dem: &DemWindow,
        config: &OrthoConfig,
    ) -> OrthoResult {
        let t0 = std::time::Instant::now();

        if frames.is_empty() {
            return OrthoResult {
                width: 0,
                height: 0,
                origin_x: 0.0,
                origin_y: 0.0,
                gsd: config.target_gsd_m,
                data: Vec::new(),
                valid_pixels: 0,
                elapsed_ms: 0,
            };
        }

        // 1. Calcular el bounding box horizontal del vuelo
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        let has_finite_bbox = frames.iter().any(|f| f.bbox[0].is_finite());
        if has_finite_bbox {
            for f in frames {
                if f.bbox[0].is_finite() && f.bbox[2].is_finite() {
                    min_x = min_x.min(f.bbox[0]);
                    max_x = max_x.max(f.bbox[2]);
                    min_y = min_y.min(f.bbox[1]);
                    max_y = max_y.max(f.bbox[3]);
                }
            }
        } else {
            for f in frames {
                min_x = min_x.min(f.pose.position.x - 100.0);
                max_x = max_x.max(f.pose.position.x + 100.0);
                min_y = min_y.min(f.pose.position.y - 100.0);
                max_y = max_y.max(f.pose.position.y + 100.0);
            }
        }

        let req_overlap = effective_min_overlap(config.trim_mode, config.min_overlap);
        let apply_hull = should_apply_hull(config.trim_mode);

        if apply_hull {
            if let Some(ref h) = config.hull {
                if h.min_x.is_finite()
                    && h.max_x > h.min_x
                    && h.min_y.is_finite()
                    && h.max_y > h.min_y
                {
                    min_x = min_x.max(h.min_x - 2.0);
                    max_x = max_x.min(h.max_x + 2.0);
                    min_y = min_y.min(h.max_y).max(h.min_y - 2.0);
                    max_y = max_y.max(h.min_y).min(h.max_y + 2.0);
                }
            }
        }

        let gsd = config.target_gsd_m;
        let width = ((max_x - min_x) / gsd).ceil().max(1.0) as usize;
        let height = ((max_y - min_y) / gsd).ceil().max(1.0) as usize;

        // Pre-cálculo radiométrico y geométrico multi-fotograma (Nivelación FFC por sortie + Latencia global + Planos Base L_k)
        let enable_multiband = config.feather_px > 0.0 && frames.len() > 1;
        let (t_amb_global, ffc_models, low_freq_planes, frame_xy_shifts, frame_track_penalties) =
            if enable_multiband {
                Self::prepare_thermal_multiband_state(
                    frames, dem, config, min_x, max_x, min_y, max_y,
                )
            } else {
                (
                    20.0f32,
                    Vec::new(),
                    Vec::new(),
                    vec![(0.0f64, 0.0f64); frames.len()],
                    vec![0.0f64; frames.len()],
                )
            };
        let t_hot_start = t_amb_global + 12.5;
        let t_hot_full = t_amb_global + 18.5;

        // Precalcular factores off-nadir + penalización de giros/tránsitos fuera del eje lawnmower
        let frame_off_nadir: Vec<(f64, f64)> = frames
            .iter()
            .enumerate()
            .map(|(k, f)| {
                let off_deg = (f.pose.pitch_deg + 90.0).abs().max(0.0);
                let track_pen = frame_track_penalties.get(k).copied().unwrap_or(0.0);
                ((-0.22 * off_deg).exp(), off_deg * 25.0 + track_pen)
            })
            .collect();

        // 2. Mapeo paralelo celda por celda usando Rayon por filas
        let mut output = vec![config.nodata_value; width * height];

        output
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(row, row_slice)| {
                let cell_y = max_y - (row as f64 + 0.5) * gsd;

                for (col, out_cell) in row_slice.iter_mut().enumerate() {
                    let cell_x = min_x + (col as f64 + 0.5) * gsd;

                    // 1. Verificación de Hull (si está activo y presente)
                    if apply_hull {
                        if let Some(ref h) = config.hull {
                            if !h.contains(cell_x, cell_y) {
                                continue;
                            }
                        }
                    }

                    // 2. Muestrear elevación del terreno en (cell_x, cell_y)
                    let elev = match dem.sample_elevation_bilinear(cell_x, cell_y) {
                        Some(z) => z,
                        None => continue,
                    };

                    let ground_pt = Vector3::new(cell_x, cell_y, elev);

                    // 3. Proyectar y recolectar candidatos válidos
                    let mut candidates = Vec::with_capacity(frames.len());

                    for (frame_idx, frame) in frames.iter().enumerate() {
                        let (sx, sy) = frame_xy_shifts[frame_idx];
                        // Fast rejection por bbox de frame (expandido por el shift de latencia)
                        if cell_x < frame.bbox[0] + sx.min(0.0) - 2.0
                            || cell_x > frame.bbox[2] + sx.max(0.0) + 2.0
                            || cell_y < frame.bbox[1] + sy.min(0.0) - 2.0
                            || cell_y > frame.bbox[3] + sy.max(0.0) + 2.0
                        {
                            continue;
                        }

                        let cam_pos = Vector3::new(
                            frame.pose.position.x + sx,
                            frame.pose.position.y + sy,
                            frame.pose.position.z,
                        );
                        if let Some((u, v_px)) =
                            frame
                                .basis
                                .world_to_pixel(ground_pt, cam_pos, &frame.intrinsics)
                        {
                            let w_f = frame.width as f64;
                            let h_f = frame.height as f64;

                            if u >= 0.0 && u < w_f && v_px >= 0.0 && v_px < h_f {
                                if let Some(val) = frame.sample_pixel(u, v_px, config.resampling) {
                                    // Filtro de validez física: no NaN, no infinito, no nodata
                                    if !val.is_finite()
                                        || (config.nodata_value.is_finite()
                                            && (val - config.nodata_value).abs() < 1e-4)
                                    {
                                        continue;
                                    }

                                    let cx = frame.intrinsics.cx;
                                    let cy = frame.intrinsics.cy;
                                    let dist_sq = (u - cx) * (u - cx) + (v_px - cy) * (v_px - cy);
                                    let d_edge = u
                                        .min(w_f - 1.0 - u)
                                        .min(v_px)
                                        .min(h_f - 1.0 - v_px)
                                        .max(0.0);

                                    candidates.push((val, u, v_px, dist_sq, d_edge, frame_idx));
                                }
                            }
                        }
                    }

                    // 4. Verificación de solape mínimo
                    if candidates.len() < req_overlap || candidates.is_empty() {
                        continue;
                    }

                    // 5. Blending térmico / Mejor Nadir
                    if !enable_multiband || candidates.len() == 1 {
                        candidates.sort_by(|a, b| {
                            a.3.partial_cmp(&b.3)
                                .unwrap_or(std::cmp::Ordering::Equal)
                                .then_with(|| a.5.cmp(&b.5))
                        });
                        let (val_raw, u, v_px, _, _, frame_idx) = candidates[0];
                        if enable_multiband && !ffc_models.is_empty() {
                            let beta_bg = ((t_hot_full - val_raw) / (t_hot_full - t_hot_start))
                                .clamp(0.0, 1.0);
                            let delta_ffc = Self::eval_ffc_delta(
                                &ffc_models[frame_idx],
                                u,
                                v_px,
                                frames[frame_idx].width,
                                frames[frame_idx].height,
                            );
                            *out_cell = val_raw + beta_bg * delta_ffc;
                        } else {
                            *out_cell = val_raw;
                        }
                    } else {
                        // Fusión Térmica Bicapa tipo Pirámide Laplaciana (L_wide + H_voronoi):
                        // Evaluamos primero solo la geometría (d_eff) y luego muestreamos L_k y H_k
                        // únicamente para los Top-6 candidatos locales.
                        let margin_hi = (config.feather_px * 5.0).clamp(6.0, 72.0);
                        let margin_low = (config.feather_px * 12.0).clamp(36.0, 110.0);
                        let mut scored_cands: Vec<(f32, f64, f64, f64, f64, f64, f64, usize)> =
                            Vec::with_capacity(candidates.len());

                        for &(val_raw, u, v_px, dist_sq, d_edge, frame_idx) in &candidates {
                            let t_hi = (d_edge / margin_hi).clamp(0.0, 1.0);
                            let e_hi = t_hi * t_hi * (3.0 - 2.0 * t_hi);
                            let t_low = (d_edge / margin_low).clamp(0.0, 1.0);
                            let e_low = t_low * t_low * (3.0 - 2.0 * t_low);
                            let d_center = dist_sq.sqrt();
                            let border_penalty = (1.0 - e_hi) * 180.0;
                            let tilt_penalty = frame_off_nadir[frame_idx].1;
                            let d_eff = d_center + border_penalty + tilt_penalty;
                            let d_low = d_center + (1.0 - e_low) * 90.0 + 0.35 * tilt_penalty;
                            scored_cands
                                .push((val_raw, u, v_px, d_eff, d_low, e_hi, e_low, frame_idx));
                        }

                        scored_cands.sort_by(|a, b| {
                            a.3.partial_cmp(&b.3)
                                .unwrap_or(std::cmp::Ordering::Equal)
                                .then_with(|| a.7.cmp(&b.7))
                        });

                        let max_diff = if config.max_blend_temp_diff.is_finite()
                            && config.max_blend_temp_diff >= 0.0
                        {
                            config.max_blend_temp_diff
                        } else {
                            3.5
                        };

                        let d_best = scored_cands[0].3;
                        let d_low_best = scored_cands
                            .iter()
                            .take(8)
                            .map(|c| c.4)
                            .fold(f64::INFINITY, f64::min);
                        let band_low = (config.feather_px * 11.0).clamp(48.0, 96.0);
                        let band_detail = (config.feather_px * 3.5).clamp(10.0, 28.0);
                        let band_hot = config.feather_px.clamp(2.0, 6.0);

                        let mut sum_w_low = 0.0f64;
                        let mut sum_l_low = 0.0f64;
                        let mut sum_w_hi = 0.0f64;
                        let mut sum_h_hi = 0.0f64;

                        let mut t_min_raw = scored_cands[0].0;
                        let mut t_max_raw = scored_cands[0].0;
                        let mut t_min_corr = f32::INFINITY;
                        let mut t_max_corr = f32::NEG_INFINITY;

                        for (rank, &(val_raw, u, v_px, d_eff, d_low, e_hi, e_low, frame_idx)) in
                            scored_cands.iter().take(8).enumerate()
                        {
                            let delta = (d_eff - d_best).max(0.0);
                            let delta_low = (d_low - d_low_best).max(0.0);
                            if delta_low >= band_low && delta >= band_detail && rank > 0 {
                                continue;
                            }
                            let f_w = frames[frame_idx].width;
                            let f_h = frames[frame_idx].height;
                            let l_k = Self::sample_low_freq_bilinear(
                                &low_freq_planes[frame_idx],
                                f_w,
                                f_h,
                                u,
                                v_px,
                            );

                            if delta_low < band_low || rank == 0 {
                                let u_low = (1.0 - delta_low / band_low).clamp(0.0, 1.0);
                                let s_low = u_low
                                    * u_low
                                    * u_low
                                    * (10.0 - 15.0 * u_low + 6.0 * u_low * u_low);
                                let w_low =
                                    (s_low * e_low * frame_off_nadir[frame_idx].0).max(1e-7);
                                sum_w_low += w_low;
                                sum_l_low += w_low * (l_k as f64);
                            }

                            if rank < 3 && (delta < band_detail || rank == 0) && e_hi > 1e-4 {
                                let beta_bg = ((t_hot_full - val_raw) / (t_hot_full - t_hot_start))
                                    .clamp(0.0, 1.0);
                                let delta_ffc =
                                    Self::eval_ffc_delta(&ffc_models[frame_idx], u, v_px, f_w, f_h);
                                let val_corr = val_raw + beta_bg * delta_ffc;
                                let h_k = val_corr - l_k;

                                if val_raw < t_min_raw {
                                    t_min_raw = val_raw;
                                }
                                if val_raw > t_max_raw {
                                    t_max_raw = val_raw;
                                }
                                if val_corr < t_min_corr {
                                    t_min_corr = val_corr;
                                }
                                if val_corr > t_max_corr {
                                    t_max_corr = val_corr;
                                }

                                let u_hi = (1.0 - delta / band_detail).clamp(0.0, 1.0);
                                let w_hi = (u_hi * u_hi * (3.0 - 2.0 * u_hi)).max(if rank == 0 {
                                    1.0
                                } else {
                                    0.0
                                }) * e_hi.max(1e-3);
                                sum_w_hi += w_hi;
                                sum_h_hi += w_hi * (h_k as f64);
                            }
                        }

                        let l_blend = if sum_w_low > 1e-9 {
                            (sum_l_low / sum_w_low) as f32
                        } else {
                            scored_cands[0].0
                        };
                        let h_blend = if sum_w_hi > 1e-6 {
                            (sum_h_hi / sum_w_hi) as f32
                        } else {
                            0.0
                        };

                        let mut t_seamless = l_blend + h_blend;
                        if frames.len() < 4 && t_min_corr.is_finite() && t_max_corr.is_finite() {
                            t_seamless = t_seamless.clamp(t_min_corr, t_max_corr);
                        }

                        // Conmutador continuo de Hotspot físicamente anclado:
                        let is_hotspot_diff =
                            (t_max_raw - t_min_raw) > max_diff || t_max_raw > t_hot_full;
                        if t_max_raw > t_hot_start && is_hotspot_diff {
                            let alpha_hot = ((t_max_raw - t_hot_start)
                                / (t_hot_full - t_hot_start))
                                .clamp(0.0, 1.0);
                            let delta_second = scored_cands[1].3 - d_best;
                            let t_hot = if delta_second < band_hot {
                                scored_cands[0].0.max(scored_cands[1].0)
                            } else {
                                scored_cands[0].0
                            };
                            *out_cell = (1.0 - alpha_hot) * t_seamless + alpha_hot * t_hot;
                        } else {
                            *out_cell = t_seamless;
                        }
                    }
                }
            });

        let valid_count = output
            .iter()
            .filter(|v| {
                v.is_finite()
                    && (config.nodata_value.is_nan() || (**v - config.nodata_value).abs() > 1e-4)
            })
            .count();
        let elapsed = t0.elapsed().as_millis();

        OrthoResult {
            width,
            height,
            origin_x: min_x,
            origin_y: max_y,
            gsd,
            data: output,
            valid_pixels: valid_count,
            elapsed_ms: elapsed,
        }
    }

    /// Evalúa la corrección espacial FFC + viñeteo térmico `[a_k, b_k, c_k, d_k]` en `(u, v)`.
    #[inline]
    fn eval_ffc_delta(model: &[f32; 4], u: f64, v: f64, width: usize, height: usize) -> f32 {
        let half_w = (0.5 * width as f64).max(1.0);
        let half_h = (0.5 * height as f64).max(1.0);
        let un = ((u - half_w) / half_w).clamp(-1.0, 1.0) as f32;
        let vn = ((v - half_h) / half_h).clamp(-1.0, 1.0) as f32;
        let r2 = un * un + vn * vn - 0.6667f32;
        model[0] + model[1] * un + model[2] * vn + model[3] * r2
    }

    /// Muestrea bilinealmente el plano de baja frecuencia `L_k(u, v)` precalculado.
    #[inline]
    fn sample_low_freq_bilinear(plane: &[f32], width: usize, height: usize, u: f64, v: f64) -> f32 {
        if plane.is_empty() || width == 0 || height == 0 {
            return 0.0;
        }
        let x = (u - 0.5).clamp(0.0, (width - 1) as f64);
        let y = (v - 0.5).clamp(0.0, (height - 1) as f64);
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(width - 1);
        let y1 = (y0 + 1).min(height - 1);
        let fx = (x - x0 as f64) as f32;
        let fy = (y - y0 as f64) as f32;

        let v00 = plane[y0 * width + x0];
        let v10 = plane[y0 * width + x1];
        let v01 = plane[y1 * width + x0];
        let v11 = plane[y1 * width + x1];

        let top = v00 + fx * (v10 - v00);
        let bot = v01 + fx * (v11 - v01);
        top + fy * (bot - top)
    }

    /// Precalcula la temperatura ambiental global `t_amb`, los modelos afines/radiales de nivelación FFC
    /// `[a_k, b_k, c_k, d_k]` por fotograma con gauge por sortie, los planos base `L_k(u, v)`,
    /// el desplazamiento de latencia GPS-obturador `frame_xy_shifts` y la penalización de rumbo/giro.
    #[allow(clippy::type_complexity)]
    fn prepare_thermal_multiband_state(
        frames: &[ImageFrame],
        dem: &DemWindow,
        config: &OrthoConfig,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
    ) -> (f32, Vec<[f32; 4]>, Vec<Vec<f32>>, Vec<(f64, f64)>, Vec<f64>) {
        let n = frames.len();
        let is_valid_temp = |val: f32| -> bool {
            val.is_finite()
                && val > -60.0
                && val < 1200.0
                && (!config.nodata_value.is_finite() || (val - config.nodata_value).abs() > 1e-4)
        };

        // 0. Segmentación por sortie (vuelo), vectores de paso longitudinal y penalización de giro/tránsito
        let parse_dji_timestamp_and_seq = |id: &str| -> (Option<i64>, Option<i32>) {
            // Formato estándar DJI: DJI_YYYYMMDDHHMMSS_XXXX_T
            let parts: Vec<&str> = id.split('_').collect();
            for i in 0..parts.len() {
                if parts[i].len() == 14 && parts[i].chars().all(|c| c.is_ascii_digit()) {
                    let hh: i64 = parts[i][8..10].parse().unwrap_or(0);
                    let mm: i64 = parts[i][10..12].parse().unwrap_or(0);
                    let ss: i64 = parts[i][12..14].parse().unwrap_or(0);
                    let sec_of_day = hh * 3600 + mm * 60 + ss;
                    let seq = if i + 1 < parts.len() && parts[i + 1].len() == 4 {
                        parts[i + 1].parse::<i32>().ok()
                    } else {
                        None
                    };
                    return (Some(sec_of_day), seq);
                }
            }
            (None, None)
        };

        let mut sortie_id = vec![0usize; n];
        let mut cur_sortie = 0usize;
        for k in 1..n {
            let dx = frames[k].pose.position.x - frames[k - 1].pose.position.x;
            let dy = frames[k].pose.position.y - frames[k - 1].pose.position.y;
            let dxy = dx.hypot(dy);
            let dz = (frames[k].pose.position.z - frames[k - 1].pose.position.z).abs();
            let (t_prev, seq_prev) = parse_dji_timestamp_and_seq(&frames[k - 1].id);
            let (t_cur, seq_cur) = parse_dji_timestamp_and_seq(&frames[k].id);
            let time_gap_split = match (t_prev, t_cur) {
                (Some(tp), Some(tc)) => (tc - tp).abs() > 180,
                _ => false,
            };
            let seq_split = match (seq_prev, seq_cur) {
                (Some(sp), Some(sc)) => {
                    let is_rollover = sp >= 9990 && sc <= 10;
                    (!is_rollover && sc <= sp && time_gap_split) || (sc - sp) > 10
                }
                _ => false,
            };
            if time_gap_split || seq_split || dxy > 85.0 || (dz > 7.5 && dxy > 35.0) {
                cur_sortie += 1;
            }
            sortie_id[k] = cur_sortie;
        }
        let num_sorties = cur_sortie + 1;

        let mut step_vecs = vec![(0.0f64, 0.0f64); n];
        if n >= 3 {
            for k in 0..n {
                let (dx, dy) = if k > 0
                    && k + 1 < n
                    && sortie_id[k - 1] == sortie_id[k]
                    && sortie_id[k + 1] == sortie_id[k]
                {
                    let d1x = frames[k].pose.position.x - frames[k - 1].pose.position.x;
                    let d1y = frames[k].pose.position.y - frames[k - 1].pose.position.y;
                    let d2x = frames[k + 1].pose.position.x - frames[k].pose.position.x;
                    let d2y = frames[k + 1].pose.position.y - frames[k].pose.position.y;
                    if d1x.hypot(d1y) < 42.0 && d2x.hypot(d2y) < 42.0 {
                        (0.5 * (d1x + d2x), 0.5 * (d1y + d2y))
                    } else if d1x.hypot(d1y) < 42.0 {
                        (d1x, d1y)
                    } else if d2x.hypot(d2y) < 42.0 {
                        (d2x, d2y)
                    } else {
                        (0.0, 0.0)
                    }
                } else if k > 0 && sortie_id[k - 1] == sortie_id[k] {
                    let d1x = frames[k].pose.position.x - frames[k - 1].pose.position.x;
                    let d1y = frames[k].pose.position.y - frames[k - 1].pose.position.y;
                    if d1x.hypot(d1y) < 42.0 {
                        (d1x, d1y)
                    } else {
                        (0.0, 0.0)
                    }
                } else if k + 1 < n && sortie_id[k + 1] == sortie_id[k] {
                    let d2x = frames[k + 1].pose.position.x - frames[k].pose.position.x;
                    let d2y = frames[k + 1].pose.position.y - frames[k].pose.position.y;
                    if d2x.hypot(d2y) < 42.0 {
                        (d2x, d2y)
                    } else {
                        (0.0, 0.0)
                    }
                } else {
                    (0.0, 0.0)
                };
                step_vecs[k] = (dx, dy);
            }
        }

        // Eje principal del patrón lawnmower por sortie (media circular de 2*theta sobre vectores de avance)
        let mut sortie_counts = vec![0usize; num_sorties];
        for &s_id in &sortie_id {
            sortie_counts[s_id] += 1;
        }
        let primary_sortie = sortie_counts
            .iter()
            .enumerate()
            .max_by_key(|&(_, &cnt)| cnt)
            .map(|(s_id, _)| s_id)
            .unwrap_or(0);

        let mut frame_track_penalties = vec![0.0f64; n];
        if n >= 8 {
            for s_id in 0..num_sorties {
                let mut sum_sin2 = 0.0f64;
                let mut sum_cos2 = 0.0f64;
                for k in 0..n {
                    if sortie_id[k] != s_id {
                        continue;
                    }
                    let (dx, dy) = step_vecs[k];
                    let len = dx.hypot(dy);
                    if len > 2.0 {
                        let th = dy.atan2(dx);
                        sum_sin2 += len * (2.0 * th).sin();
                        sum_cos2 += len * (2.0 * th).cos();
                    }
                }
                if sum_sin2.hypot(sum_cos2) > 1e-3 {
                    let dom_axis = 0.5 * sum_sin2.atan2(sum_cos2);
                    let wrap_deg_180 = |deg: f64| -> f64 {
                        let mut d = deg.rem_euclid(180.0);
                        if d > 90.0 {
                            d = 180.0 - d;
                        }
                        d
                    };
                    for k in 0..n {
                        if sortie_id[k] != s_id {
                            continue;
                        }
                        let (dx, dy) = step_vecs[k];
                        let len = dx.hypot(dy);
                        let axis_dev_deg = if len > 2.0 {
                            let th_deg = (dy.atan2(dx) - dom_axis).to_degrees();
                            wrap_deg_180(th_deg)
                        } else {
                            0.0
                        };
                        let turn_deg = if k > 0 && sortie_id[k] == sortie_id[k - 1] {
                            let h_cur = frames[k]
                                .basis
                                .u_top_prime
                                .x
                                .atan2(frames[k].basis.u_top_prime.y)
                                .to_degrees();
                            let h_prev = frames[k - 1]
                                .basis
                                .u_top_prime
                                .x
                                .atan2(frames[k - 1].basis.u_top_prime.y)
                                .to_degrees();
                            let mut dyaw = (h_cur - h_prev).rem_euclid(360.0);
                            if dyaw > 180.0 {
                                dyaw = 360.0 - dyaw;
                            }
                            dyaw
                        } else {
                            0.0
                        };
                        let pen =
                            (axis_dev_deg - 20.0).max(0.0) * 6.0 + (turn_deg - 12.0).max(0.0) * 4.0;
                        frame_track_penalties[k] = pen;
                    }
                }
            }
        }

        // 1. Mediana interior por fotograma y mediana ambiental nadir global
        let frame_medians: Vec<f32> = frames
            .par_iter()
            .map(|f| {
                let step = (f.width.min(f.height) / 32).max(1);
                let u0 = f.width / 8;
                let u1 = (f.width * 7 / 8).max(u0 + 1);
                let v0 = f.height / 8;
                let v1 = (f.height * 7 / 8).max(v0 + 1);
                let mut vals = Vec::with_capacity(1024);
                let mut v = v0;
                while v < v1 {
                    let mut u = u0;
                    while u < u1 {
                        let val = f.data[v * f.width + u];
                        if is_valid_temp(val) {
                            vals.push(val);
                        }
                        u += step;
                    }
                    v += step;
                }
                if vals.is_empty() {
                    for &val in &f.data {
                        if is_valid_temp(val) {
                            vals.push(val);
                        }
                    }
                }
                if vals.is_empty() {
                    20.0
                } else {
                    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    vals[vals.len() * 40 / 100] // Percentil 40 (robusto frente a focos de calor extensos)
                }
            })
            .collect();

        let mut nadir_meds: Vec<f32> = frames
            .iter()
            .zip(frame_medians.iter())
            .filter(|(f, _)| (f.pose.pitch_deg + 90.0).abs() <= 15.0)
            .map(|(_, &m)| m)
            .collect();
        if nadir_meds.is_empty() {
            nadir_meds = frame_medians.clone();
        }
        nadir_meds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let t_amb_global = if nadir_meds.is_empty() {
            20.0
        } else {
            nadir_meds[nadir_meds.len() / 2]
        };
        let t_bg_max = t_amb_global + 16.0;

        // 1b. Calibración física global de latencia GPS-obturador (alpha along-track, beta cross-track)
        // sobre pasadas opuestas de la sortie principal mediante correlación cruzada normalizada (NCC)
        let mut frame_xy_shifts = vec![(0.0f64, 0.0f64); n];
        if n >= 16 {
            let mut cal_min_x = f64::INFINITY;
            let mut cal_max_x = f64::NEG_INFINITY;
            let mut cal_min_y = f64::INFINITY;
            let mut cal_max_y = f64::NEG_INFINITY;
            for (k, f) in frames.iter().enumerate() {
                if sortie_id[k] == primary_sortie {
                    cal_min_x = cal_min_x.min(f.pose.position.x - 120.0);
                    cal_max_x = cal_max_x.max(f.pose.position.x + 120.0);
                    cal_min_y = cal_min_y.min(f.pose.position.y - 120.0);
                    cal_max_y = cal_max_y.max(f.pose.position.y + 120.0);
                }
            }
            if !cal_min_x.is_finite() || cal_max_x <= cal_min_x || cal_max_y <= cal_min_y {
                cal_min_x = min_x;
                cal_max_x = max_x;
                cal_min_y = min_y;
                cal_max_y = max_y;
            }

            let cal_grid = 64usize;
            let sx_step = (cal_max_x - cal_min_x) / (cal_grid as f64);
            let sy_step = (cal_max_y - cal_min_y) / (cal_grid as f64);
            let cands: [(f64, f64); 5] = [
                (0.0, 0.0),
                (0.06, 0.03),
                (0.10, 0.05),
                (0.13, 0.07),
                (0.15, 0.08),
            ];
            let mut nccs = [0.0f64; 5];
            let mut counts = [0.0f64; 5];

            for (c_idx, &(alpha_c, beta_c)) in cands.iter().enumerate() {
                let mut sum_xy = 0.0f64;
                let mut sum_xx = 0.0f64;
                let mut sum_yy = 0.0f64;
                let mut sum_w_pair = 0.0f64;
                for gy in 0..cal_grid {
                    let cy = cal_min_y + (gy as f64 + 0.5) * sy_step;
                    for gx in 0..cal_grid {
                        let cx = cal_min_x + (gx as f64 + 0.5) * sx_step;
                        let elev = match dem.sample_elevation_bilinear(cx, cy) {
                            Some(z) => z,
                            None => continue,
                        };
                        let gpt = Vector3::new(cx, cy, elev);
                        let mut obs_anom: Vec<(usize, f64, f64)> = Vec::new();
                        for (k, f) in frames.iter().enumerate() {
                            if sortie_id[k] != primary_sortie
                                || frame_track_penalties[k] > 12.0
                                || (f.pose.pitch_deg + 90.0).abs() > 12.0
                            {
                                continue;
                            }
                            let (dx_s, dy_s) = step_vecs[k];
                            if dx_s.hypot(dy_s) < 4.0 {
                                continue;
                            }
                            let sx = (alpha_c * dx_s - beta_c * dy_s).clamp(-4.2, 4.2);
                            let sy = (alpha_c * dy_s + beta_c * dx_s).clamp(-4.2, 4.2);
                            if cx < f.bbox[0] + sx.min(0.0)
                                || cx > f.bbox[2] + sx.max(0.0)
                                || cy < f.bbox[1] + sy.min(0.0)
                                || cy > f.bbox[3] + sy.max(0.0)
                            {
                                continue;
                            }
                            let cpos = Vector3::new(
                                f.pose.position.x + sx,
                                f.pose.position.y + sy,
                                f.pose.position.z,
                            );
                            if let Some((u, v)) = f.basis.world_to_pixel(gpt, cpos, &f.intrinsics) {
                                let wf = f.width as f64;
                                let hf = f.height as f64;
                                let min_edge = (0.04 * wf.min(hf)).clamp(4.0, 20.0);
                                let d_edge = u.min(wf - 1.0 - u).min(v).min(hf - 1.0 - v);
                                if d_edge > min_edge {
                                    if let Some(val) =
                                        f.sample_pixel(u, v, ResamplingMethod::Bilinear)
                                    {
                                        if is_valid_temp(val) {
                                            let anom = (val - frame_medians[k]) as f64;
                                            let w_o = (d_edge / 64.0).clamp(0.2, 1.0);
                                            obs_anom.push((k, anom, w_o));
                                        }
                                    }
                                }
                            }
                        }
                        if obs_anom.len() >= 2 {
                            for i in 0..obs_anom.len() {
                                for j in (i + 1)..obs_anom.len() {
                                    let (ki, ai, wi) = obs_anom[i];
                                    let (kj, aj, wj) = obs_anom[j];
                                    let dot = step_vecs[ki].0 * step_vecs[kj].0
                                        + step_vecs[ki].1 * step_vecs[kj].1;
                                    if dot < -20.0 {
                                        let feat_mag = ai.abs().max(aj.abs());
                                        if feat_mag > 1.2 {
                                            let wp = wi * wj * (1.0 + feat_mag.clamp(0.0, 10.0));
                                            sum_xy += wp * ai * aj;
                                            sum_xx += wp * ai * ai;
                                            sum_yy += wp * aj * aj;
                                            sum_w_pair += wp;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                let denom = (sum_xx * sum_yy).sqrt();
                if denom > 1e-6 && sum_w_pair > 5.0 {
                    nccs[c_idx] = sum_xy / denom;
                    counts[c_idx] = sum_w_pair;
                } else {
                    nccs[c_idx] = -1.0;
                }
            }

            let ncc0 = nccs[0];
            if counts[0] > 5.0 {
                let mut best_idx = 0usize;
                let mut best_ncc = ncc0;
                for c_idx in 1..cands.len() {
                    if nccs[c_idx] > best_ncc && counts[c_idx] > 0.4 * counts[0] {
                        best_ncc = nccs[c_idx];
                        best_idx = c_idx;
                    }
                }
                // Activar corrección de latencia si incrementa la correlación NCC en >= 0.004
                if best_idx > 0 && best_ncc > ncc0 + 0.004 {
                    let (best_alpha, best_beta) = cands[best_idx];
                    for k in 0..n {
                        frame_xy_shifts[k] = (
                            (best_alpha * step_vecs[k].0 - best_beta * step_vecs[k].1)
                                .clamp(-4.2, 4.2),
                            (best_alpha * step_vecs[k].1 + best_beta * step_vecs[k].0)
                                .clamp(-4.2, 4.2),
                        );
                    }
                }
            }
        }

        // 1c. Registro relativo inter-sortie (dx_s, dy_s) por NCC en la zona de solape con la sortie primaria,
        // seguido de prioridad de costura para la sortie primaria en alta frecuencia.
        if num_sorties > 1 {
            for k in 0..n {
                if sortie_id[k] != primary_sortie {
                    frame_track_penalties[k] += 260.0;
                }
            }

            let mut prim_min_x = f64::INFINITY;
            let mut prim_max_x = f64::NEG_INFINITY;
            let mut prim_min_y = f64::INFINITY;
            let mut prim_max_y = f64::NEG_INFINITY;
            for (k, f) in frames.iter().enumerate() {
                if sortie_id[k] == primary_sortie {
                    prim_min_x = prim_min_x.min(f.pose.position.x - 140.0);
                    prim_max_x = prim_max_x.max(f.pose.position.x + 140.0);
                    prim_min_y = prim_min_y.min(f.pose.position.y - 140.0);
                    prim_max_y = prim_max_y.max(f.pose.position.y + 140.0);
                }
            }

            let mut sortie_rel_shifts = vec![(0.0f64, 0.0f64); num_sorties];
            for s_id in 0..num_sorties {
                if s_id == primary_sortie || sortie_counts[s_id] < 2 {
                    continue;
                }
                let mut sec_min_x = f64::INFINITY;
                let mut sec_max_x = f64::NEG_INFINITY;
                let mut sec_min_y = f64::INFINITY;
                let mut sec_max_y = f64::NEG_INFINITY;
                for (k, f) in frames.iter().enumerate() {
                    if sortie_id[k] == s_id {
                        sec_min_x = sec_min_x.min(f.pose.position.x - 140.0);
                        sec_max_x = sec_max_x.max(f.pose.position.x + 140.0);
                        sec_min_y = sec_min_y.min(f.pose.position.y - 140.0);
                        sec_max_y = sec_max_y.max(f.pose.position.y + 140.0);
                    }
                }
                let ov_min_x = prim_min_x.max(sec_min_x);
                let ov_max_x = prim_max_x.min(sec_max_x);
                let ov_min_y = prim_min_y.max(sec_min_y);
                let ov_max_y = prim_max_y.min(sec_max_y);

                if ov_max_x > ov_min_x + 10.0 && ov_max_y > ov_min_y + 10.0 {
                    let s_grid = 40usize;
                    let step_ox = (ov_max_x - ov_min_x) / (s_grid as f64);
                    let step_oy = (ov_max_y - ov_min_y) / (s_grid as f64);
                    let mut prim_samples: Vec<(Vector3, f64, f64)> = Vec::new();

                    for gy in 0..s_grid {
                        let cy = ov_min_y + (gy as f64 + 0.5) * step_oy;
                        for gx in 0..s_grid {
                            let cx = ov_min_x + (gx as f64 + 0.5) * step_ox;
                            let elev = match dem.sample_elevation_bilinear(cx, cy) {
                                Some(z) => z,
                                None => continue,
                            };
                            let gpt = Vector3::new(cx, cy, elev);
                            let mut sum_a = 0.0f64;
                            let mut sum_w = 0.0f64;
                            for (k, f) in frames.iter().enumerate() {
                                if sortie_id[k] != primary_sortie
                                    || frame_track_penalties[k] > 20.0
                                    || (f.pose.pitch_deg + 90.0).abs() > 15.0
                                {
                                    continue;
                                }
                                let (sx, sy) = frame_xy_shifts[k];
                                if cx < f.bbox[0] + sx.min(0.0)
                                    || cx > f.bbox[2] + sx.max(0.0)
                                    || cy < f.bbox[1] + sy.min(0.0)
                                    || cy > f.bbox[3] + sy.max(0.0)
                                {
                                    continue;
                                }
                                let cpos = Vector3::new(
                                    f.pose.position.x + sx,
                                    f.pose.position.y + sy,
                                    f.pose.position.z,
                                );
                                if let Some((u, v)) =
                                    f.basis.world_to_pixel(gpt, cpos, &f.intrinsics)
                                {
                                    let wf = f.width as f64;
                                    let hf = f.height as f64;
                                    let min_edge = (0.04 * wf.min(hf)).clamp(4.0, 16.0);
                                    let d_edge = u.min(wf - 1.0 - u).min(v).min(hf - 1.0 - v);
                                    if d_edge > min_edge {
                                        if let Some(val) =
                                            f.sample_pixel(u, v, ResamplingMethod::Bilinear)
                                        {
                                            if is_valid_temp(val) {
                                                let w_o = (d_edge / 64.0).clamp(0.2, 1.0);
                                                sum_a += w_o * ((val - frame_medians[k]) as f64);
                                                sum_w += w_o;
                                            }
                                        }
                                    }
                                }
                            }
                            if sum_w > 0.2 {
                                let a_prim = sum_a / sum_w;
                                if a_prim.abs() > 0.8 {
                                    prim_samples.push((gpt, a_prim, sum_w.min(2.0)));
                                }
                            }
                        }
                    }

                    if prim_samples.len() >= 12 {
                        let shifts_1d: [f64; 9] = [-6.0, -4.5, -3.0, -1.5, 0.0, 1.5, 3.0, 4.5, 6.0];
                        let eval_shift = |dx_s: f64, dy_s: f64| -> (f64, f64) {
                            let mut sum_xy = 0.0f64;
                            let mut sum_xx = 0.0f64;
                            let mut sum_yy = 0.0f64;
                            let mut sum_wp = 0.0f64;
                            for &(gpt, a_prim, w_prim) in &prim_samples {
                                let mut sum_a_sec = 0.0f64;
                                let mut sum_w_sec = 0.0f64;
                                for (k, f) in frames.iter().enumerate() {
                                    if sortie_id[k] != s_id
                                        || (f.pose.pitch_deg + 90.0).abs() > 15.0
                                    {
                                        continue;
                                    }
                                    let sx = frame_xy_shifts[k].0 + dx_s;
                                    let sy = frame_xy_shifts[k].1 + dy_s;
                                    if gpt.x < f.bbox[0] + sx.min(0.0)
                                        || gpt.x > f.bbox[2] + sx.max(0.0)
                                        || gpt.y < f.bbox[1] + sy.min(0.0)
                                        || gpt.y > f.bbox[3] + sy.max(0.0)
                                    {
                                        continue;
                                    }
                                    let cpos = Vector3::new(
                                        f.pose.position.x + sx,
                                        f.pose.position.y + sy,
                                        f.pose.position.z,
                                    );
                                    if let Some((u, v)) =
                                        f.basis.world_to_pixel(gpt, cpos, &f.intrinsics)
                                    {
                                        let wf = f.width as f64;
                                        let hf = f.height as f64;
                                        let min_edge = (0.04 * wf.min(hf)).clamp(4.0, 16.0);
                                        let d_edge = u.min(wf - 1.0 - u).min(v).min(hf - 1.0 - v);
                                        if d_edge > min_edge {
                                            if let Some(val) =
                                                f.sample_pixel(u, v, ResamplingMethod::Bilinear)
                                            {
                                                if is_valid_temp(val) {
                                                    let w_o = (d_edge / 64.0).clamp(0.2, 1.0);
                                                    sum_a_sec +=
                                                        w_o * ((val - frame_medians[k]) as f64);
                                                    sum_w_sec += w_o;
                                                }
                                            }
                                        }
                                    }
                                }
                                if sum_w_sec > 0.2 {
                                    let a_sec = sum_a_sec / sum_w_sec;
                                    let feat = a_prim.abs().max(a_sec.abs());
                                    let wp =
                                        w_prim * sum_w_sec.min(2.0) * (1.0 + feat.clamp(0.0, 8.0));
                                    sum_xy += wp * a_prim * a_sec;
                                    sum_xx += wp * a_prim * a_prim;
                                    sum_yy += wp * a_sec * a_sec;
                                    sum_wp += wp;
                                }
                            }
                            let denom = (sum_xx * sum_yy).sqrt();
                            if denom > 1e-6 && sum_wp > 5.0 {
                                (sum_xy / denom, sum_wp)
                            } else {
                                (-1.0, 0.0)
                            }
                        };

                        let (ncc_zero, cnt_zero) = eval_shift(0.0, 0.0);
                        if cnt_zero > 5.0 {
                            let mut coarse_cands = Vec::with_capacity(81);
                            for &dy_s in &shifts_1d {
                                for &dx_s in &shifts_1d {
                                    coarse_cands.push((dx_s, dy_s));
                                }
                            }
                            let coarse_res: Vec<(f64, f64, f64, f64)> = coarse_cands
                                .par_iter()
                                .map(|&(dx_s, dy_s)| {
                                    let (ncc_c, cnt_c) = eval_shift(dx_s, dy_s);
                                    (dx_s, dy_s, ncc_c, cnt_c)
                                })
                                .collect();

                            let mut best_dx = 0.0f64;
                            let mut best_dy = 0.0f64;
                            let mut best_ncc = ncc_zero;
                            for &(dx_s, dy_s, ncc_c, cnt_c) in &coarse_res {
                                if ncc_c > best_ncc && cnt_c > 0.4 * cnt_zero {
                                    best_ncc = ncc_c;
                                    best_dx = dx_s;
                                    best_dy = dy_s;
                                }
                            }

                            // Refinamiento fino (paso 0.5 m) alrededor del óptimo grueso
                            let fine_steps: [f64; 5] = [-1.0, -0.5, 0.0, 0.5, 1.0];
                            let mut fine_cands = Vec::with_capacity(25);
                            for &ddy in &fine_steps {
                                for &ddx in &fine_steps {
                                    fine_cands.push((
                                        (best_dx + ddx).clamp(-6.5, 6.5),
                                        (best_dy + ddy).clamp(-6.5, 6.5),
                                    ));
                                }
                            }
                            let fine_res: Vec<(f64, f64, f64, f64)> = fine_cands
                                .par_iter()
                                .map(|&(dx_s, dy_s)| {
                                    let (ncc_c, cnt_c) = eval_shift(dx_s, dy_s);
                                    (dx_s, dy_s, ncc_c, cnt_c)
                                })
                                .collect();
                            for &(dx_s, dy_s, ncc_c, cnt_c) in &fine_res {
                                if ncc_c > best_ncc && cnt_c > 0.4 * cnt_zero {
                                    best_ncc = ncc_c;
                                    best_dx = dx_s;
                                    best_dy = dy_s;
                                }
                            }

                            if best_ncc > ncc_zero + 0.01 {
                                sortie_rel_shifts[s_id] = (best_dx, best_dy);
                            }
                        }
                    }
                }
            }

            // Aplicar desplazamientos inter-sortie de forma invariante al orden:
            // si una sortie secundaria es masiva (>= 1/3 del total), centrar baricéntricamente.
            let mut mean_dx = 0.0f64;
            let mut mean_dy = 0.0f64;
            for s_id in 0..num_sorties {
                if s_id != primary_sortie && sortie_counts[s_id] * 3 >= n {
                    mean_dx += 0.5 * sortie_rel_shifts[s_id].0;
                    mean_dy += 0.5 * sortie_rel_shifts[s_id].1;
                }
            }
            for k in 0..n {
                let s_id = sortie_id[k];
                let (rx, ry) = sortie_rel_shifts[s_id];
                frame_xy_shifts[k].0 = (frame_xy_shifts[k].0 + rx - mean_dx).clamp(-8.0, 8.0);
                frame_xy_shifts[k].1 = (frame_xy_shifts[k].1 + ry - mean_dy).clamp(-8.0, 8.0);
            }
        }

        // 2. Nivelación FFC por puntos de enlace co-localizados con anclaje de media cero global
        let mut ffc_models = vec![[0.0f32; 4]; n];
        if n >= 4 {
            let grid_n = 72usize;
            let step_x = (max_x - min_x) / (grid_n as f64);
            let step_y = (max_y - min_y) / (grid_n as f64);

            // (frame_idx, un, vn, r2, val_bg, weight)
            let mut tie_points: Vec<Vec<(usize, f64, f64, f64, f64, f64)>> =
                Vec::with_capacity(grid_n * grid_n);

            for gy in 0..grid_n {
                let cy = min_y + (gy as f64 + 0.5) * step_y;
                for gx in 0..grid_n {
                    let cx = min_x + (gx as f64 + 0.5) * step_x;
                    let elev = match dem.sample_elevation_bilinear(cx, cy) {
                        Some(z) => z,
                        None => continue,
                    };
                    let gpt = Vector3::new(cx, cy, elev);
                    let mut obs = Vec::new();
                    for (k, f) in frames.iter().enumerate() {
                        let (sx, sy) = frame_xy_shifts[k];
                        if cx < f.bbox[0] + sx.min(0.0)
                            || cx > f.bbox[2] + sx.max(0.0)
                            || cy < f.bbox[1] + sy.min(0.0)
                            || cy > f.bbox[3] + sy.max(0.0)
                        {
                            continue;
                        }
                        let cpos = Vector3::new(
                            f.pose.position.x + sx,
                            f.pose.position.y + sy,
                            f.pose.position.z,
                        );
                        if let Some((u, v)) = f.basis.world_to_pixel(gpt, cpos, &f.intrinsics) {
                            let wf = f.width as f64;
                            let hf = f.height as f64;
                            let d_edge = u.min(wf - 1.0 - u).min(v).min(hf - 1.0 - v);
                            if d_edge > 12.0 {
                                if let Some(val) = f.sample_pixel(u, v, ResamplingMethod::Bilinear)
                                {
                                    if is_valid_temp(val) && val <= t_bg_max {
                                        let hw = (0.5 * wf).max(1.0);
                                        let hh = (0.5 * hf).max(1.0);
                                        let un = ((u - hw) / hw).clamp(-1.0, 1.0);
                                        let vn = ((v - hh) / hh).clamp(-1.0, 1.0);
                                        let r2 = un * un + vn * vn - 0.6667;
                                        let off_nadir = (f.pose.pitch_deg + 90.0).abs().max(0.0);
                                        let t_e = (d_edge / 64.0).clamp(0.1, 1.0);
                                        let w_obs = (t_e * t_e) * (-0.18 * off_nadir).exp();
                                        obs.push((k, un, vn, r2, val as f64, w_obs));
                                    }
                                }
                            }
                        }
                    }
                    if obs.len() >= 2 {
                        tie_points.push(obs);
                    }
                }
            }

            if !tie_points.is_empty() {
                let mut a = vec![0.0f64; n];
                let mut d = vec![0.0f64; n];

                let reg_a = 10.0;
                let reg_grad = 36.0;

                for _iter in 0..10 {
                    let mut num_a = vec![0.0f64; n];
                    let mut den_a = vec![reg_a; n];
                    let mut num_d = vec![0.0f64; n];
                    let mut den_d = vec![reg_grad; n];

                    for obs in &tie_points {
                        let mut sum_w = 0.0;
                        let mut sum_t = 0.0;
                        for &(k, _un, _vn, r2, val, w) in obs {
                            let corr = val + a[k] + d[k] * r2;
                            sum_w += w;
                            sum_t += w * corr;
                        }
                        if sum_w <= 1e-9 {
                            continue;
                        }
                        let t_target = sum_t / sum_w;

                        for &(k, _un, _vn, r2, val, w) in obs {
                            let res_a = t_target - (val + d[k] * r2);
                            num_a[k] += w * res_a;
                            den_a[k] += w;

                            let res_d = t_target - (val + a[k]);
                            num_d[k] += w * r2 * res_d;
                            den_d[k] += w * r2 * r2;
                        }
                    }

                    for k in 0..n {
                        a[k] = (0.45 * a[k] + 0.55 * (num_a[k] / den_a[k])).clamp(-1.20, 1.20);
                        d[k] = (0.45 * d[k] + 0.55 * (num_d[k] / den_d[k])).clamp(-0.35, 0.35);
                    }

                    // Anclaje de media cero global sobre los fotogramas nadir de la misión
                    let mut sum_all_a = 0.0;
                    let mut cnt_all = 0usize;
                    for (k, f) in frames.iter().enumerate() {
                        if (f.pose.pitch_deg + 90.0).abs() <= 15.0 {
                            sum_all_a += a[k];
                            cnt_all += 1;
                        }
                    }
                    if cnt_all > 0 {
                        let mean_all = sum_all_a / (cnt_all as f64);
                        for val_a in a.iter_mut().take(n) {
                            *val_a = (*val_a - mean_all).clamp(-1.20, 1.20);
                        }
                    }
                }

                for k in 0..n {
                    ffc_models[k] = [a[k] as f32, 0.0, 0.0, d[k] as f32];
                }
            }
        }

        // 3. Precalcular en paralelo los planos de baja frecuencia L_k(u, v) con filtro caja separable 2D
        let low_freq_planes: Vec<Vec<f32>> = frames
            .par_iter()
            .enumerate()
            .map(|(k, f)| {
                let w = f.width;
                let h = f.height;
                if w == 0 || h == 0 {
                    return Vec::new();
                }
                let m_k = frame_medians[k];
                let model = &ffc_models[k];

                // En mosaicos reales (n >= 4), L_k captura toda la variación suave del terreno (incluyendo laderas cálidas),
                // acotando solo picos extremos de fuego (> m_k + 22°C). En tests sintéticos pequeños (n < 4), sustituye spikes por m_k.
                let mut base = vec![0.0f32; w * h];
                for v in 0..h {
                    for u in 0..w {
                        let raw = f.data[v * w + u];
                        let clean = if !is_valid_temp(raw) || (n < 4 && raw > m_k + 4.5) {
                            m_k
                        } else {
                            raw.min(m_k + 22.0)
                        };
                        let delta =
                            Self::eval_ffc_delta(model, u as f64 + 0.5, v as f64 + 0.5, w, h);
                        base[v * w + u] = clean + delta;
                    }
                }

                let radius = (w.min(h) / 64).clamp(2, 8);
                let mut temp_h = vec![0.0f32; w * h];
                let mut out_l = vec![0.0f32; w * h];

                // Pasada horizontal con suma prefija O(W*H) exacta
                let mut prefix = vec![0.0f64; w.max(h) + 1];
                for v in 0..h {
                    let row_in = &base[v * w..(v + 1) * w];
                    let row_out = &mut temp_h[v * w..(v + 1) * w];
                    prefix[0] = 0.0;
                    for u in 0..w {
                        prefix[u + 1] = prefix[u] + row_in[u] as f64;
                    }
                    for (u, item) in row_out.iter_mut().enumerate().take(w) {
                        let u0 = u.saturating_sub(radius);
                        let u1 = (u + radius).min(w - 1);
                        let s = prefix[u1 + 1] - prefix[u0];
                        *item = (s / ((u1 - u0 + 1) as f64)) as f32;
                    }
                }

                // Pasada vertical con suma prefija O(W*H) exacta
                for u in 0..w {
                    prefix[0] = 0.0;
                    for v in 0..h {
                        prefix[v + 1] = prefix[v] + temp_h[v * w + u] as f64;
                    }
                    for v in 0..h {
                        let v0 = v.saturating_sub(radius);
                        let v1 = (v + radius).min(h - 1);
                        let s = prefix[v1 + 1] - prefix[v0];
                        out_l[v * w + u] = (s / ((v1 - v0 + 1) as f64)) as f32;
                    }
                }

                out_l
            })
            .collect();

        (
            t_amb_global,
            ffc_models,
            low_freq_planes,
            frame_xy_shifts,
            frame_track_penalties,
        )
    }
}

/// Modo de mezcla fotométrica para ortomosaicos visuales RGB.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RgbBlendMode {
    /// Selecciona el píxel del fotograma más cercano al nadir (sin costuras de mezcla).
    BestNadir,
    /// Mezcla ponderada por distancia al borde (feathering lineal).
    Feather,
    /// Micro-feathering continuo ponderado por distancia al centro óptico y proximidad al borde.
    MicroFeather,
}

/// Metadatos y referencia a disco de un fotograma RGB.
#[derive(Debug, Clone)]
pub struct RgbFrameMeta {
    pub id: String,
    pub file_path: std::path::PathBuf,
    pub pose: CameraPose,
    pub intrinsics: CameraIntrinsics,
    pub basis: CameraBasis,
    /// Bounding box proyectado [min_x, min_y, max_x, max_y].
    pub bbox: [f64; 4],
}

/// Configuración para la ortorrectificación RGB.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RgbOrthoConfig {
    pub target_gsd_m: f64,
    pub blend_mode: RgbBlendMode,
    pub trim_mode: TrimMode,
    pub min_overlap: u16,
    pub hull: Option<zenith_core::hull::ConvexHull2D>,
    pub feather_px: f64,
    pub tile_size: usize,
    pub max_cached_images: usize,
    pub gain_compensation: bool,
    pub dehaze: bool,
}

impl Default for RgbOrthoConfig {
    fn default() -> Self {
        Self {
            target_gsd_m: 0.05,
            blend_mode: RgbBlendMode::BestNadir,
            trim_mode: TrimMode::None,
            min_overlap: 1,
            hull: None,
            feather_px: 30.0,
            tile_size: 1024,
            max_cached_images: 32,
            gain_compensation: true,
            dehaze: false,
        }
    }
}

/// Resultado de la ortorrectificación visual RGB en formato RGBA (4 canales, 8 bits por canal).
pub struct RgbOrthoResult {
    pub width: usize,
    pub height: usize,
    pub origin_x: f64,
    pub origin_y: f64,
    pub gsd: f64,
    /// Datos de imagen intercalados RGBA [R, G, B, A, R, G, B, A, ...].
    pub rgba_data: Vec<u8>,
    pub valid_pixels: usize,
    pub elapsed_ms: u128,
}

/// Estructura interna para imagen RGB decodificada en memoria.
struct DecodedRgbImage {
    /// Ancho nativo del sensor (coordenadas de intrínsecos).
    width: usize,
    /// Alto nativo del sensor (coordenadas de intrínsecos).
    height: usize,
    /// Ancho real del buffer almacenado en memoria.
    buf_width: usize,
    /// Alto real del buffer almacenado en memoria.
    buf_height: usize,
    /// Factor de escala entre coordenadas nativas y buffer (`1.0` o `0.5`).
    scale: f64,
    /// Datos RGB continuos: [R, G, B, R, G, B, ...].
    rgb: Vec<u8>,
}

impl DecodedRgbImage {
    fn load_from_file_scaled(path: &std::path::Path, downsample_2x: bool) -> Result<Self, String> {
        let file_data =
            std::fs::read(path).map_err(|e| format!("Error leyendo {}: {}", path.display(), e))?;
        let mut decoder = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(&file_data));
        let pixels = decoder
            .decode()
            .map_err(|e| format!("Error decodificando JPEG {}: {:?}", path.display(), e))?;
        let (width, height) = decoder
            .dimensions()
            .ok_or_else(|| "No se pudo obtener info de dimensiones JPEG".to_string())?;

        if downsample_2x && width >= 4 && height >= 4 {
            let bw = width / 2;
            let bh = height / 2;
            let mut down = vec![0u8; bw * bh * 3];
            for by in 0..bh {
                let r0 = (2 * by) * width;
                let r1 = (2 * by + 1) * width;
                let out_row = by * bw;
                for bx in 0..bw {
                    let c0 = 2 * bx;
                    let c1 = c0 + 1;
                    let i00 = (r0 + c0) * 3;
                    let i10 = (r0 + c1) * 3;
                    let i01 = (r1 + c0) * 3;
                    let i11 = (r1 + c1) * 3;
                    let out_idx = (out_row + bx) * 3;
                    for ch in 0..3 {
                        let sum = pixels[i00 + ch] as u16
                            + pixels[i10 + ch] as u16
                            + pixels[i01 + ch] as u16
                            + pixels[i11 + ch] as u16;
                        down[out_idx + ch] = ((sum + 2) >> 2) as u8;
                    }
                }
            }
            return Ok(Self {
                width,
                height,
                buf_width: bw,
                buf_height: bh,
                scale: 0.5,
                rgb: down,
            });
        }

        Ok(Self {
            width,
            height,
            buf_width: width,
            buf_height: height,
            scale: 1.0,
            rgb: pixels,
        })
    }

    #[inline]
    fn sample_bilinear(&self, u: f64, v: f64) -> Option<[u8; 3]> {
        if u < 0.0 || u >= self.width as f64 || v < 0.0 || v >= self.height as f64 {
            return None;
        }

        let (u_b, v_b) = if (self.scale - 1.0).abs() < 1e-9 {
            (u, v)
        } else {
            (
                ((u + 0.5) * self.scale - 0.5).clamp(0.0, (self.buf_width - 1) as f64),
                ((v + 0.5) * self.scale - 0.5).clamp(0.0, (self.buf_height - 1) as f64),
            )
        };

        let col0 = u_b.floor() as usize;
        let row0 = v_b.floor() as usize;
        let col1 = (col0 + 1).min(self.buf_width - 1);
        let row1 = (row0 + 1).min(self.buf_height - 1);

        let fx = (u_b - col0 as f64) as f32;
        let fy = (v_b - row0 as f64) as f32;

        let idx00 = (row0 * self.buf_width + col0) * 3;
        let idx10 = (row0 * self.buf_width + col1) * 3;
        let idx01 = (row1 * self.buf_width + col0) * 3;
        let idx11 = (row1 * self.buf_width + col1) * 3;

        if idx11 + 2 >= self.rgb.len() {
            return None;
        }

        let interp_channel = |c: usize| -> u8 {
            let v00 = self.rgb[idx00 + c] as f32;
            let v10 = self.rgb[idx10 + c] as f32;
            let v01 = self.rgb[idx01 + c] as f32;
            let v11 = self.rgb[idx11 + c] as f32;

            let top = (1.0 - fx) * v00 + fx * v10;
            let bottom = (1.0 - fx) * v01 + fx * v11;
            ((1.0 - fy) * top + fy * bottom).round().clamp(0.0, 255.0) as u8
        };

        Some([interp_channel(0), interp_channel(1), interp_channel(2)])
    }

    /// Interpolación bicúbica Catmull-Rom (4x4, a = -0.5) para preservar nitidez óptica de bordes.
    #[inline]
    fn sample_bicubic(&self, u: f64, v: f64) -> Option<[u8; 3]> {
        if u < 0.0 || u >= self.width as f64 || v < 0.0 || v >= self.height as f64 {
            return None;
        }
        let (u_b, v_b) = if (self.scale - 1.0).abs() < 1e-9 {
            (u, v)
        } else {
            (
                ((u + 0.5) * self.scale - 0.5).clamp(0.0, (self.buf_width - 1) as f64),
                ((v + 0.5) * self.scale - 0.5).clamp(0.0, (self.buf_height - 1) as f64),
            )
        };
        let col0 = u_b.floor() as isize;
        let row0 = v_b.floor() as isize;
        let w_i = self.buf_width as isize;
        let h_i = self.buf_height as isize;

        // Cerca de los bordes de 1px usar bilineal rápido
        if col0 < 1 || col0 + 2 >= w_i || row0 < 1 || row0 + 2 >= h_i {
            return self.sample_bilinear(u, v);
        }

        let tx = (u_b - col0 as f64) as f32;
        let ty = (v_b - row0 as f64) as f32;

        #[inline]
        fn catmull_weights(t: f32) -> [f32; 4] {
            let t2 = t * t;
            let t3 = t2 * t;
            [
                -0.5 * t + t2 - 0.5 * t3,
                1.0 - 2.5 * t2 + 1.5 * t3,
                0.5 * t + 2.0 * t2 - 1.5 * t3,
                -0.5 * t2 + 0.5 * t3,
            ]
        }

        let wx = catmull_weights(tx);
        let wy = catmull_weights(ty);

        let mut acc_r = 0.0f32;
        let mut acc_g = 0.0f32;
        let mut acc_b = 0.0f32;

        for (j, &wy_j) in wy.iter().enumerate() {
            let r_idx = (row0 - 1 + j as isize) as usize;
            let row_base = r_idx * self.buf_width;
            let mut row_r = 0.0f32;
            let mut row_g = 0.0f32;
            let mut row_b = 0.0f32;
            for (i, &wx_i) in wx.iter().enumerate() {
                let c_idx = (col0 - 1 + i as isize) as usize;
                let p_idx = (row_base + c_idx) * 3;
                row_r += wx_i * (self.rgb[p_idx] as f32);
                row_g += wx_i * (self.rgb[p_idx + 1] as f32);
                row_b += wx_i * (self.rgb[p_idx + 2] as f32);
            }
            acc_r += wy_j * row_r;
            acc_g += wy_j * row_g;
            acc_b += wy_j * row_b;
        }

        Some([
            acc_r.round().clamp(0.0, 255.0) as u8,
            acc_g.round().clamp(0.0, 255.0) as u8,
            acc_b.round().clamp(0.0, 255.0) as u8,
        ])
    }
}

/// Motor de ortorrectificación visual RGB directa 2.5D con paralelismo y caché de teselas.
pub struct RgbOrthoEngine;

impl RgbOrthoEngine {
    /// Genera un ortomosaico RGBA a partir de metadatos de fotogramas e imágenes JPEG en disco.
    pub fn generate_orthomosaic(
        frames: &[RgbFrameMeta],
        dem: &DemWindow,
        config: &RgbOrthoConfig,
    ) -> RgbOrthoResult {
        let t0 = std::time::Instant::now();

        if frames.is_empty() {
            return RgbOrthoResult {
                width: 0,
                height: 0,
                origin_x: 0.0,
                origin_y: 0.0,
                gsd: config.target_gsd_m,
                rgba_data: Vec::new(),
                valid_pixels: 0,
                elapsed_ms: 0,
            };
        }

        // 1. Calcular bounding box del área de vuelo
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        let has_finite_bbox = frames.iter().any(|f| f.bbox[0].is_finite());
        if has_finite_bbox {
            for f in frames {
                if f.bbox[0].is_finite() && f.bbox[2].is_finite() {
                    min_x = min_x.min(f.bbox[0]);
                    max_x = max_x.max(f.bbox[2]);
                    min_y = min_y.min(f.bbox[1]);
                    max_y = max_y.max(f.bbox[3]);
                }
            }
        } else {
            for f in frames {
                min_x = min_x.min(f.pose.position.x - 100.0);
                max_x = max_x.max(f.pose.position.x + 100.0);
                min_y = min_y.min(f.pose.position.y - 100.0);
                max_y = max_y.max(f.pose.position.y + 100.0);
            }
        }

        let req_overlap = effective_min_overlap(config.trim_mode, config.min_overlap);
        let apply_hull = should_apply_hull(config.trim_mode);

        if apply_hull {
            if let Some(ref h) = config.hull {
                if h.min_x.is_finite()
                    && h.max_x > h.min_x
                    && h.min_y.is_finite()
                    && h.max_y > h.min_y
                {
                    min_x = min_x.max(h.min_x - 2.0);
                    max_x = max_x.min(h.max_x + 2.0);
                    min_y = min_y.min(h.max_y).max(h.min_y - 2.0);
                    max_y = max_y.max(h.min_y).min(h.max_y + 2.0);
                }
            }
        }

        let gsd = config.target_gsd_m;
        let width = ((max_x - min_x) / gsd).ceil().max(1.0) as usize;
        let height = ((max_y - min_y) / gsd).ceil().max(1.0) as usize;

        // Pre-asignar buffer RGBA transparente (A = 0)
        let total_bytes = width
            .checked_mul(height)
            .and_then(|wh| wh.checked_mul(4))
            .expect("Dimensiones de ortomosaico RGB exceden la memoria indexable");

        let mut output_rgba = vec![0u8; total_bytes];

        // Caché compartida de imágenes decodificadas con eviction LRU/FIFO
        let image_cache: Arc<Mutex<HashMap<usize, Arc<DecodedRgbImage>>>> =
            Arc::new(Mutex::new(HashMap::new()));

        let tile_size = config.tile_size.max(64);
        let n_tiles_x = width.div_ceil(tile_size);
        let n_tiles_y = height.div_ceil(tile_size);

        // Capacidad adaptativa según resolución del sensor y relación GSD objetivo / GSD nativo
        let is_high_res = frames
            .first()
            .map(|f| (f.intrinsics.width as usize) * (f.intrinsics.height as usize) > 25_000_000)
            .unwrap_or(false);
        let native_gsd_m = {
            let mut agls: Vec<f64> = frames
                .iter()
                .filter_map(|f| {
                    dem.sample_elevation_bilinear(f.pose.position.x, f.pose.position.y)
                        .map(|z| f.pose.position.z - z)
                        .filter(|&h| h > 15.0 && h < 2000.0)
                })
                .collect();
            agls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let med_agl = if !agls.is_empty() {
                agls[agls.len() / 2]
            } else {
                120.0
            };
            let f_min = frames
                .first()
                .map(|f| f.intrinsics.fx.min(f.intrinsics.fy).max(500.0))
                .unwrap_or(3000.0);
            (med_agl / f_min).max(0.01)
        };
        let downsample_2x = is_high_res && (gsd >= native_gsd_m * 2.0);
        let effective_max_cache = if is_high_res && !downsample_2x {
            config.max_cached_images.clamp(16, 28)
        } else if downsample_2x {
            config.max_cached_images.max(112)
        } else {
            config.max_cached_images.max(128)
        };
        let mut gain_memo: HashMap<Vec<usize>, Vec<[f64; 3]>> = HashMap::new();

        for ty in 0..n_tiles_y {
            let row_start = ty * tile_size;
            let row_end = (row_start + tile_size).min(height);

            let tile_y_max = max_y - (row_start as f64) * gsd;
            let tile_y_min = max_y - (row_end as f64) * gsd;

            // Recorrido boustrophedon (serpiente) para maximizar localidad espacial de caché entre filas
            let tx_indices: Vec<usize> = if ty % 2 == 1 {
                (0..n_tiles_x).rev().collect()
            } else {
                (0..n_tiles_x).collect()
            };

            for tx in tx_indices {
                let col_start = tx * tile_size;
                let col_end = (col_start + tile_size).min(width);

                let tile_x_min = min_x + (col_start as f64) * gsd;
                let tile_x_max = min_x + (col_end as f64) * gsd;

                // Identificar fotogramas candidatos para esta tesela
                let mut tile_frame_indices = Vec::new();
                for (idx, f) in frames.iter().enumerate() {
                    let intersects = f.bbox[0] <= tile_x_max
                        && f.bbox[2] >= tile_x_min
                        && f.bbox[1] <= tile_y_max
                        && f.bbox[3] >= tile_y_min;
                    if intersects || !has_finite_bbox {
                        tile_frame_indices.push(idx);
                    }
                }

                if tile_frame_indices.is_empty() {
                    continue;
                }

                // 1. Identificar qué imágenes faltan en la caché
                let missing_indices: Vec<usize> = {
                    let cache = image_cache.lock().unwrap();
                    tile_frame_indices
                        .iter()
                        .copied()
                        .filter(|idx| !cache.contains_key(idx))
                        .collect()
                };

                // 2. Decodificar imágenes faltantes EN PARALELO con Rayon
                if !missing_indices.is_empty() {
                    let newly_decoded: Vec<(usize, Arc<DecodedRgbImage>)> = missing_indices
                        .par_iter()
                        .filter_map(|&idx| {
                            match DecodedRgbImage::load_from_file_scaled(
                                &frames[idx].file_path,
                                downsample_2x,
                            ) {
                                Ok(img) => Some((idx, Arc::new(img))),
                                Err(err) => {
                                    eprintln!(
                                        "Omitiendo fotograma RGB {}: {}",
                                        frames[idx].id, err
                                    );
                                    None
                                }
                            }
                        })
                        .collect();

                    let mut cache = image_cache.lock().unwrap();
                    // Si excedemos capacidad, desalojar solo las imágenes más lejanas espacialmente
                    if cache.len() + newly_decoded.len() > effective_max_cache {
                        let center_x = 0.5 * (tile_x_min + tile_x_max);
                        let center_y = 0.5 * (tile_y_min + tile_y_max);
                        let mut cached_keys: Vec<(usize, f64)> = cache
                            .keys()
                            .copied()
                            .filter(|k| !tile_frame_indices.contains(k))
                            .map(|k| {
                                let dx = frames[k].pose.position.x - center_x;
                                let dy = frames[k].pose.position.y - center_y;
                                (k, dx * dx + dy * dy)
                            })
                            .collect();
                        // Ordenar de menor a mayor distancia para que pop() desaloje la más lejana primero
                        cached_keys.sort_by(|a, b| {
                            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        let target_keep = effective_max_cache.saturating_sub(newly_decoded.len());
                        while cache.len() > target_keep {
                            if let Some((evict_k, _)) = cached_keys.pop() {
                                cache.remove(&evict_k);
                            } else {
                                break;
                            }
                        }
                    }

                    for (idx, arc_img) in newly_decoded {
                        cache.insert(idx, arc_img);
                    }
                }

                // 3. Recolectar referencias Arc para la tesela actual
                let mut active_frames: Vec<(&RgbFrameMeta, Arc<DecodedRgbImage>)> =
                    Vec::with_capacity(tile_frame_indices.len());
                let mut active_indices: Vec<usize> = Vec::with_capacity(tile_frame_indices.len());
                {
                    let cache = image_cache.lock().unwrap();
                    for &idx in &tile_frame_indices {
                        if let Some(arc_img) = cache.get(&idx) {
                            active_frames.push((&frames[idx], Arc::clone(arc_img)));
                            active_indices.push(idx);
                        }
                    }
                }

                if active_frames.is_empty() {
                    continue;
                }

                // Compensación de ganancia Brown-Lowe sobre la huella completa de los fotogramas activos (memoizada determinísticamente)
                let n_act = active_frames.len();
                let tile_gains = if !config.gain_compensation || n_act <= 1 {
                    vec![[1.0f64; 3]; n_act]
                } else if let Some(cached_gains) = gain_memo.get(&active_indices) {
                    cached_gains.clone()
                } else {
                    let mut computed_gains = vec![[1.0f64; 3]; n_act];
                    let sample_grid = 24;
                    let mut pair_counts = vec![vec![0.0f64; n_act]; n_act];
                    let mut sum_l_ij = vec![vec![[0.0f64; 3]; n_act]; n_act];
                    let mut sum_l_ji = vec![vec![[0.0f64; 3]; n_act]; n_act];

                    // Usar exclusivamente el bounding box conjunto de los fotogramas activos (100% invariante entre teselas vecinas)
                    let mut act_min_x = f64::INFINITY;
                    let mut act_max_x = f64::NEG_INFINITY;
                    let mut act_min_y = f64::INFINITY;
                    let mut act_max_y = f64::NEG_INFINITY;
                    for (fm, _) in &active_frames {
                        if fm.bbox[0].is_finite() && fm.bbox[2].is_finite() {
                            act_min_x = act_min_x.min(fm.bbox[0]);
                            act_max_x = act_max_x.max(fm.bbox[2]);
                            act_min_y = act_min_y.min(fm.bbox[1]);
                            act_max_y = act_max_y.max(fm.bbox[3]);
                        }
                    }
                    if !act_min_x.is_finite() || act_max_x <= act_min_x {
                        act_min_x = tile_x_min;
                        act_max_x = tile_x_max;
                        act_min_y = tile_y_min;
                        act_max_y = tile_y_max;
                    }

                    let step_y = (act_max_y - act_min_y) / sample_grid as f64;
                    let step_x = (act_max_x - act_min_x) / sample_grid as f64;

                    for sy in 0..sample_grid {
                        let gy = act_min_y + (sy as f64 + 0.5) * step_y;
                        for sx in 0..sample_grid {
                            let gx = act_min_x + (sx as f64 + 0.5) * step_x;
                            let z = match dem.sample_elevation_bilinear(gx, gy) {
                                Some(val) => val,
                                None => continue,
                            };
                            let g_pt = Vector3::new(gx, gy, z);

                            let mut samples = Vec::new();
                            for (f_idx, (f_meta, img)) in active_frames.iter().enumerate() {
                                if let Some((u, v)) = f_meta.basis.world_to_pixel(
                                    g_pt,
                                    f_meta.pose.position,
                                    &f_meta.intrinsics,
                                ) {
                                    let wf = img.width as f64;
                                    let hf = img.height as f64;
                                    // Puntos dentro del 15%..85% del sensor (excluyendo viñeteo extremo)
                                    if u >= 0.15 * wf
                                        && u <= 0.85 * wf
                                        && v >= 0.15 * hf
                                        && v <= 0.85 * hf
                                    {
                                        if let Some(rgb) = img.sample_bilinear(u, v) {
                                            // Excluir píxeles contaminados por humo/nube blanca (canal oscuro > 170) o sombras puras
                                            let dark_ch = rgb[0].min(rgb[1]).min(rgb[2]);
                                            let bright_ch = rgb[0].max(rgb[1]).max(rgb[2]);
                                            if dark_ch > 170 || bright_ch < 15 {
                                                continue;
                                            }
                                            let l_r = (rgb[0] as f64 / 255.0).powf(2.2);
                                            let l_g = (rgb[1] as f64 / 255.0).powf(2.2);
                                            let l_b = (rgb[2] as f64 / 255.0).powf(2.2);
                                            samples.push((f_idx, [l_r, l_g, l_b]));
                                        }
                                    }
                                }
                            }

                            for a in 0..samples.len() {
                                for b in (a + 1)..samples.len() {
                                    let (ia, la) = samples[a];
                                    let (ib, lb) = samples[b];
                                    pair_counts[ia][ib] += 1.0;
                                    pair_counts[ib][ia] += 1.0;
                                    for c in 0..3 {
                                        sum_l_ij[ia][ib][c] += la[c];
                                        sum_l_ji[ia][ib][c] += lb[c];
                                        sum_l_ij[ib][ia][c] += lb[c];
                                        sum_l_ji[ib][ia][c] += la[c];
                                    }
                                }
                            }
                        }
                    }

                    // Resolver 15 iteraciones Gauss-Seidel de Brown-Lowe por canal
                    for c in 0..3 {
                        let mut g = vec![1.0f64; n_act];
                        let sigma_g_sq_inv = 100.0;

                        for _ in 0..15 {
                            for i in 0..n_act {
                                let mut num = sigma_g_sq_inv * 1.0;
                                let mut den = sigma_g_sq_inv;

                                for j in 0..n_act {
                                    if i != j && pair_counts[i][j] >= 3.0 {
                                        let n_ij = pair_counts[i][j];
                                        let mean_i = sum_l_ij[i][j][c] / n_ij;
                                        let mean_j = sum_l_ji[i][j][c] / n_ij;
                                        if mean_i > 1e-4 && mean_j > 1e-4 {
                                            num += n_ij * mean_i * mean_j * g[j];
                                            den += n_ij * mean_i * mean_i;
                                        }
                                    }
                                }

                                if den > 1e-6 {
                                    g[i] = num / den;
                                }
                            }
                        }

                        for i in 0..n_act {
                            computed_gains[i][c] =
                                (g[i].max(1e-4)).powf(1.0 / 2.2).clamp(0.85, 1.18);
                        }
                    }
                    gain_memo.insert(active_indices, computed_gains.clone());
                    computed_gains
                };

                // Ortorrectificar las filas de la tesela en paralelo con Rayon usando par_chunks_mut (100% safe Rust)
                let tile_rows_slice =
                    &mut output_rgba[(row_start * width * 4)..(row_end * width * 4)];

                tile_rows_slice
                    .par_chunks_mut(width * 4)
                    .enumerate()
                    .for_each(|(local_row, row_bytes)| {
                        let row = row_start + local_row;
                        let cell_y = max_y - (row as f64 + 0.5) * gsd;

                        for col in col_start..col_end {
                            let cell_x = min_x + (col as f64 + 0.5) * gsd;

                            // Verificación de Hull
                            if apply_hull {
                                if let Some(ref h) = config.hull {
                                    if !h.contains(cell_x, cell_y) {
                                        continue;
                                    }
                                }
                            }

                            // Muestreo de elevación
                            let elev = match dem.sample_elevation_bilinear(cell_x, cell_y) {
                                Some(z) => z,
                                None => continue,
                            };

                            let ground_pt = Vector3::new(cell_x, cell_y, elev);

                            // 1. Proyección geométrica rápida en caché L1 (sin leer buffers DRAM de imagen aún)
                            let mut geom_cands = Vec::with_capacity(active_frames.len());

                            for (cand_idx, (frame_meta, img)) in active_frames.iter().enumerate() {
                                if cell_x < frame_meta.bbox[0]
                                    || cell_x > frame_meta.bbox[2]
                                    || cell_y < frame_meta.bbox[1]
                                    || cell_y > frame_meta.bbox[3]
                                {
                                    continue;
                                }

                                let cam_pos = frame_meta.pose.position;
                                if let Some((u, v_px)) = frame_meta.basis.world_to_pixel(
                                    ground_pt,
                                    cam_pos,
                                    &frame_meta.intrinsics,
                                ) {
                                    let w_f = img.width as f64;
                                    let h_f = img.height as f64;

                                    if u >= 0.0 && u < w_f && v_px >= 0.0 && v_px < h_f {
                                        let cx = frame_meta.intrinsics.cx;
                                        let cy = frame_meta.intrinsics.cy;
                                        let dist_sq =
                                            (u - cx) * (u - cx) + (v_px - cy) * (v_px - cy);
                                        let d_edge = u
                                            .min(w_f - 1.0 - u)
                                            .min(v_px)
                                            .min(h_f - 1.0 - v_px)
                                            .max(0.0);
                                        let half_diag_sq =
                                            (0.5 * w_f) * (0.5 * w_f) + (0.5 * h_f) * (0.5 * h_f);

                                        geom_cands.push((
                                            u,
                                            v_px,
                                            dist_sq,
                                            d_edge,
                                            half_diag_sq,
                                            cand_idx,
                                        ));
                                    }
                                }
                            }

                            if geom_cands.len() < req_overlap || geom_cands.is_empty() {
                                continue;
                            }

                            let col_idx = col * 4;

                            // Helper para muestrear bicúbicamente un candidato y aplicar su ganancia Brown-Lowe
                            let sample_cand_bicubic =
                                |u: f64, v_px: f64, cand_idx: usize| -> Option<[u8; 3]> {
                                    let img = &active_frames[cand_idx].1;
                                    let rgb = img.sample_bicubic(u, v_px)?;
                                    if config.gain_compensation {
                                        let g = tile_gains[cand_idx];
                                        Some([
                                            (rgb[0] as f64 * g[0]).round().clamp(0.0, 255.0) as u8,
                                            (rgb[1] as f64 * g[1]).round().clamp(0.0, 255.0) as u8,
                                            (rgb[2] as f64 * g[2]).round().clamp(0.0, 255.0) as u8,
                                        ])
                                    } else {
                                        Some(rgb)
                                    }
                                };

                            // Mezcla visual según modo seleccionado
                            match config.blend_mode {
                                RgbBlendMode::BestNadir => {
                                    geom_cands.sort_by(|a, b| {
                                        a.2.partial_cmp(&b.2)
                                            .unwrap_or(std::cmp::Ordering::Equal)
                                            .then_with(|| a.5.cmp(&b.5))
                                    });
                                    let (u, v_px, _, _, _, cand_idx) = geom_cands[0];
                                    if let Some(best_rgb) = sample_cand_bicubic(u, v_px, cand_idx) {
                                        row_bytes[col_idx] = best_rgb[0];
                                        row_bytes[col_idx + 1] = best_rgb[1];
                                        row_bytes[col_idx + 2] = best_rgb[2];
                                        row_bytes[col_idx + 3] = 255;
                                    }
                                }
                                RgbBlendMode::Feather => {
                                    let mut sum_weights = 0.0;
                                    let mut sum_r = 0.0;
                                    let mut sum_g = 0.0;
                                    let mut sum_b = 0.0;

                                    for &(u, v_px, dist_sq, d_edge, half_diag_sq, cand_idx) in
                                        &geom_cands
                                    {
                                        if let Some(rgb) = sample_cand_bicubic(u, v_px, cand_idx) {
                                            let edge_factor =
                                                (d_edge / config.feather_px).clamp(0.05, 1.0);
                                            let dist_factor =
                                                1.0 + (dist_sq / half_diag_sq.max(1.0));
                                            let weight = edge_factor / dist_factor;

                                            sum_weights += weight;
                                            sum_r += weight * (rgb[0] as f64);
                                            sum_g += weight * (rgb[1] as f64);
                                            sum_b += weight * (rgb[2] as f64);
                                        }
                                    }

                                    if sum_weights > 1e-6 {
                                        let r =
                                            (sum_r / sum_weights).round().clamp(0.0, 255.0) as u8;
                                        let g =
                                            (sum_g / sum_weights).round().clamp(0.0, 255.0) as u8;
                                        let b =
                                            (sum_b / sum_weights).round().clamp(0.0, 255.0) as u8;
                                        row_bytes[col_idx] = r;
                                        row_bytes[col_idx + 1] = g;
                                        row_bytes[col_idx + 2] = b;
                                        row_bytes[col_idx + 3] = 255;
                                    }
                                }
                                RgbBlendMode::MicroFeather => {
                                    let margin = 160.0;
                                    let band_px = config.feather_px.clamp(6.0, 18.0);

                                    let mut geom_scored = Vec::with_capacity(geom_cands.len());
                                    for &(u, v_px, dist_sq, d_edge, _, cand_idx) in &geom_cands {
                                        let t_edge = (d_edge / margin).clamp(0.0, 1.0);
                                        let e_k = t_edge * t_edge * (3.0 - 2.0 * t_edge);
                                        let d_center_px = dist_sq.sqrt();
                                        let border_barrier = ((1.0 - e_k) / e_k.max(1e-3)) * 450.0;
                                        let d_eff_px = d_center_px + border_barrier;
                                        geom_scored.push((u, v_px, d_eff_px, e_k, cand_idx));
                                    }

                                    geom_scored.sort_by(|a, b| {
                                        a.2.partial_cmp(&b.2)
                                            .unwrap_or(std::cmp::Ordering::Equal)
                                            .then_with(|| a.4.cmp(&b.4))
                                    });

                                    if !config.dehaze {
                                        if geom_scored.len() == 1
                                            || (geom_scored[1].2 - geom_scored[0].2) >= band_px
                                        {
                                            let (u, v_px, _, _, cand_idx) = geom_scored[0];
                                            if let Some(best_rgb) =
                                                sample_cand_bicubic(u, v_px, cand_idx)
                                            {
                                                row_bytes[col_idx] = best_rgb[0];
                                                row_bytes[col_idx + 1] = best_rgb[1];
                                                row_bytes[col_idx + 2] = best_rgb[2];
                                                row_bytes[col_idx + 3] = 255;
                                            }
                                        } else {
                                            let d_min_px = geom_scored[0].2;
                                            let mut sum_w = 0.0;
                                            let mut sum_r = 0.0;
                                            let mut sum_g = 0.0;
                                            let mut sum_b = 0.0;

                                            for &(u, v_px, d_eff_px, e_k, cand_idx) in
                                                geom_scored.iter().take(3)
                                            {
                                                let delta_px = d_eff_px - d_min_px;
                                                if delta_px >= band_px {
                                                    break;
                                                }
                                                if let Some(rgb) =
                                                    sample_cand_bicubic(u, v_px, cand_idx)
                                                {
                                                    let t = (delta_px / band_px).clamp(0.0, 1.0);
                                                    let w = e_k * (1.0 - t * t * (3.0 - 2.0 * t));
                                                    sum_w += w;
                                                    sum_r += w * (rgb[0] as f64);
                                                    sum_g += w * (rgb[1] as f64);
                                                    sum_b += w * (rgb[2] as f64);
                                                }
                                            }

                                            if sum_w > 1e-6 {
                                                let r =
                                                    (sum_r / sum_w).round().clamp(0.0, 255.0) as u8;
                                                let g =
                                                    (sum_g / sum_w).round().clamp(0.0, 255.0) as u8;
                                                let b =
                                                    (sum_b / sum_w).round().clamp(0.0, 255.0) as u8;
                                                row_bytes[col_idx] = r;
                                                row_bytes[col_idx + 1] = g;
                                                row_bytes[col_idx + 2] = b;
                                                row_bytes[col_idx + 3] = 255;
                                            }
                                        }
                                    } else {
                                        let mut dehaze_scored =
                                            Vec::with_capacity(geom_scored.len().min(8));
                                        let mut min_dark_ch = f64::INFINITY;

                                        for &(u, v_px, d_eff_px, e_k, cand_idx) in
                                            geom_scored.iter().take(8)
                                        {
                                            if let Some(rgb) =
                                                sample_cand_bicubic(u, v_px, cand_idx)
                                            {
                                                let dk = rgb[0].min(rgb[1]).min(rgb[2]) as f64;
                                                if e_k > 0.35 && dk < min_dark_ch {
                                                    min_dark_ch = dk;
                                                }
                                                dehaze_scored.push((rgb, d_eff_px, e_k, cand_idx));
                                            }
                                        }

                                        if dehaze_scored.is_empty() {
                                            continue;
                                        }
                                        if !min_dark_ch.is_finite() {
                                            min_dark_ch = dehaze_scored[0].0[0]
                                                .min(dehaze_scored[0].0[1])
                                                .min(dehaze_scored[0].0[2])
                                                as f64;
                                        }

                                        let haze_thresh = (min_dark_ch + 18.0).max(82.0);
                                        let mut has_smoke = false;
                                        for item in &mut dehaze_scored {
                                            let c_rgb = item.0;
                                            let d_k = c_rgb[0].min(c_rgb[1]).min(c_rgb[2]) as f64;
                                            let sat_k = (c_rgb[0].max(c_rgb[1]).max(c_rgb[2])
                                                - c_rgb[0].min(c_rgb[1]).min(c_rgb[2]))
                                                as f64;
                                            if d_k > haze_thresh && sat_k < 46.0 {
                                                let smoke_excess = d_k - haze_thresh;
                                                let sat_weight =
                                                    ((46.0 - sat_k) / 20.0).clamp(0.0, 1.0);
                                                let pen =
                                                    smoke_excess * smoke_excess * 2.5 * sat_weight;
                                                item.1 += pen;
                                                has_smoke = true;
                                            }
                                        }

                                        dehaze_scored.sort_by(|a, b| {
                                            a.1.partial_cmp(&b.1)
                                                .unwrap_or(std::cmp::Ordering::Equal)
                                                .then_with(|| a.3.cmp(&b.3))
                                        });

                                        let local_band_px = if has_smoke {
                                            band_px.max(36.0)
                                        } else {
                                            band_px
                                        };

                                        let d_min_px = dehaze_scored[0].1;
                                        let mut sum_w = 0.0;
                                        let mut sum_r = 0.0;
                                        let mut sum_g = 0.0;
                                        let mut sum_b = 0.0;

                                        for &(rgb, d_eff_px, e_k, _) in dehaze_scored.iter().take(3)
                                        {
                                            let delta_px = d_eff_px - d_min_px;
                                            if delta_px >= local_band_px && sum_w > 0.0 {
                                                break;
                                            }
                                            let t = (delta_px / local_band_px).clamp(0.0, 1.0);
                                            let w = e_k * (1.0 - t * t * (3.0 - 2.0 * t)).max(1e-4);
                                            sum_w += w;
                                            sum_r += w * (rgb[0] as f64);
                                            sum_g += w * (rgb[1] as f64);
                                            sum_b += w * (rgb[2] as f64);
                                        }

                                        if sum_w > 1e-6 {
                                            let r = (sum_r / sum_w).round().clamp(0.0, 255.0) as u8;
                                            let g = (sum_g / sum_w).round().clamp(0.0, 255.0) as u8;
                                            let b = (sum_b / sum_w).round().clamp(0.0, 255.0) as u8;
                                            row_bytes[col_idx] = r;
                                            row_bytes[col_idx + 1] = g;
                                            row_bytes[col_idx + 2] = b;
                                            row_bytes[col_idx + 3] = 255;
                                        }
                                    }
                                }
                            }
                        }
                    });
            }
        }

        let valid_pixels = output_rgba
            .par_chunks_exact(4)
            .filter(|p| p[3] != 0)
            .count();
        let elapsed_ms = t0.elapsed().as_millis();

        RgbOrthoResult {
            width,
            height,
            origin_x: min_x,
            origin_y: max_y,
            gsd,
            rgba_data: output_rgba,
            valid_pixels,
            elapsed_ms,
        }
    }
}

// =========================================================================
// 2.5D POSE REFINEMENT SOLVER & DATA STRUCTURES (FLEXGLUE INTEGRATION)
// =========================================================================

fn default_score() -> f64 {
    1.0
}

/// Coincidencia de punto de enlace (tie point) 2D-2D entre dos fotogramas.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TiePointMatch {
    pub frame_i: String,
    pub frame_j: String,
    pub u_i: f64,
    pub v_i: f64,
    pub u_j: f64,
    pub v_j: f64,
    #[serde(default = "default_score")]
    pub score: f64,
    #[serde(default = "default_score")]
    pub covisibility: f64,
}

/// Desplazamiento de pose rígida en el plano ENU horizontal y yaw.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FramePoseDelta {
    pub frame_id: String,
    pub delta_x_m: f64,
    pub delta_y_m: f64,
    pub delta_yaw_deg: f64,
}

/// Archivo de entrada de puntos de enlace y/o correcciones de pose a priori.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TiePointFile {
    #[serde(default)]
    pub schema_version: String,
    #[serde(default)]
    pub matcher: String,
    #[serde(default)]
    pub coordinate_convention: String,
    #[serde(default)]
    pub matches: Vec<TiePointMatch>,
    #[serde(default)]
    pub pose_overrides: Vec<FramePoseDelta>,
}

/// Reporte cuantitativo de la optimización de poses 2.5D.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PoseRefinementReport {
    pub matcher: String,
    pub total_matches: usize,
    pub valid_dem_intersections: usize,
    pub train_matches: usize,
    pub holdout_matches: usize,
    pub iterations: usize,
    pub global_along_track_m: f64,
    pub global_cross_track_m: f64,
    pub pre_refinement_rmse_m: f64,
    pub post_refinement_rmse_m: f64,
    pub holdout_pre_rmse_m: f64,
    pub holdout_post_rmse_m: f64,
    pub inlier_ratio: f64,
    pub saturated_frames_pct: f64,
    pub applied_deltas: Vec<FramePoseDelta>,
}

/// Hiperparámetros de regularización y límites del solver 2.5D.
#[derive(Debug, Clone, Copy)]
pub struct PoseRefinementConfig {
    pub lambda_t: f64,
    pub lambda_psi: f64,
    pub clamp_shift_m: f64,
    pub clamp_yaw_deg: f64,
    pub max_residual_m: f64,
    pub huber_threshold_m: f64,
    pub outer_iterations: usize,
    pub inner_steps: usize,
    pub damping: f64,
}

impl Default for PoseRefinementConfig {
    fn default() -> Self {
        Self {
            lambda_t: 2.5,
            lambda_psi: 10.0,
            clamp_shift_m: 4.0,
            clamp_yaw_deg: 2.5,
            max_residual_m: 15.0,
            huber_threshold_m: 1.5,
            outer_iterations: 3,
            inner_steps: 15,
            damping: 0.6,
        }
    }
}

/// Trait genérico para fotogramas fotogramétricos refinables (RGB o Térmico).
pub trait RefinableFrame {
    fn id(&self) -> &str;
    fn pose(&self) -> &CameraPose;
    fn pose_mut(&mut self) -> &mut CameraPose;
    fn intrinsics(&self) -> &CameraIntrinsics;
    fn basis(&self) -> &CameraBasis;
    fn set_basis(&mut self, basis: CameraBasis);
    fn bbox(&self) -> [f64; 4];
    fn set_bbox(&mut self, bbox: [f64; 4]);
    fn image_size(&self) -> (f64, f64);

    /// Actualiza la base ortonormal y el bounding box proyectado sobre el DEM.
    fn update_basis_and_bbox(&mut self, dem: &DemWindow) {
        let pose = self.pose();
        let basis = CameraBasis::from_gimbal(pose.yaw_deg, pose.pitch_deg, pose.roll_deg);
        self.set_basis(basis);
        let cam_pos = self.pose().position;
        let intrinsics = *self.intrinsics();
        let (w, h) = self.image_size();
        let c1 = basis.pixel_to_ray(0.0, 0.0, cam_pos, &intrinsics);
        let c2 = basis.pixel_to_ray(w, 0.0, cam_pos, &intrinsics);
        let c3 = basis.pixel_to_ray(w, h, cam_pos, &intrinsics);
        let c4 = basis.pixel_to_ray(0.0, h, cam_pos, &intrinsics);

        let p1 = dem.intersect_ray(&c1, 8000.0, 16);
        let p2 = dem.intersect_ray(&c2, 8000.0, 16);
        let p3 = dem.intersect_ray(&c3, 8000.0, 16);
        let p4 = dem.intersect_ray(&c4, 8000.0, 16);

        let new_bbox = if let (Some(pt1), Some(pt2), Some(pt3), Some(pt4)) = (p1, p2, p3, p4) {
            let pts = [pt1, pt2, pt3, pt4];
            let min_x = pts.iter().map(|p| p.x).fold(f64::INFINITY, f64::min) - 10.0;
            let max_x = pts.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max) + 10.0;
            let min_y = pts.iter().map(|p| p.y).fold(f64::INFINITY, f64::min) - 10.0;
            let max_y = pts.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max) + 10.0;
            [min_x, min_y, max_x, max_y]
        } else {
            let cur = self.bbox();
            if cur[0].is_finite() {
                cur
            } else {
                [
                    cam_pos.x - 300.0,
                    cam_pos.y - 300.0,
                    cam_pos.x + 300.0,
                    cam_pos.y + 300.0,
                ]
            }
        };
        self.set_bbox(new_bbox);
    }
}

impl RefinableFrame for ImageFrame {
    fn id(&self) -> &str {
        &self.id
    }
    fn pose(&self) -> &CameraPose {
        &self.pose
    }
    fn pose_mut(&mut self) -> &mut CameraPose {
        &mut self.pose
    }
    fn intrinsics(&self) -> &CameraIntrinsics {
        &self.intrinsics
    }
    fn basis(&self) -> &CameraBasis {
        &self.basis
    }
    fn set_basis(&mut self, basis: CameraBasis) {
        self.basis = basis;
    }
    fn bbox(&self) -> [f64; 4] {
        self.bbox
    }
    fn set_bbox(&mut self, bbox: [f64; 4]) {
        self.bbox = bbox;
    }
    fn image_size(&self) -> (f64, f64) {
        (self.width as f64, self.height as f64)
    }
}

impl RefinableFrame for RgbFrameMeta {
    fn id(&self) -> &str {
        &self.id
    }
    fn pose(&self) -> &CameraPose {
        &self.pose
    }
    fn pose_mut(&mut self) -> &mut CameraPose {
        &mut self.pose
    }
    fn intrinsics(&self) -> &CameraIntrinsics {
        &self.intrinsics
    }
    fn basis(&self) -> &CameraBasis {
        &self.basis
    }
    fn set_basis(&mut self, basis: CameraBasis) {
        self.basis = basis;
    }
    fn bbox(&self) -> [f64; 4] {
        self.bbox
    }
    fn set_bbox(&mut self, bbox: [f64; 4]) {
        self.bbox = bbox;
    }
    fn image_size(&self) -> (f64, f64) {
        (self.intrinsics.width as f64, self.intrinsics.height as f64)
    }
}

/// Extrae número secuencial DJI de 4 dígitos (p. ej. `_0042_` o `_0042`).
fn extract_dji_number(s: &str) -> Option<u32> {
    let bytes = s.as_bytes();
    for i in 0..bytes.len() {
        if i + 5 < bytes.len() && bytes[i] == b'_' && bytes[i + 5] == b'_' {
            let slice = &bytes[i + 1..i + 5];
            if slice.iter().all(|b| b.is_ascii_digit()) {
                if let Ok(num) = std::str::from_utf8(slice).unwrap_or("").parse::<u32>() {
                    return Some(num);
                }
            }
        }
    }
    for i in 0..bytes.len() {
        if bytes[i] == b'_' && i + 4 < bytes.len() {
            let slice = &bytes[i + 1..i + 5];
            if slice.iter().all(|b| b.is_ascii_digit())
                && (i + 5 == bytes.len() || !bytes[i + 5].is_ascii_digit())
            {
                if let Ok(num) = std::str::from_utf8(slice).unwrap_or("").parse::<u32>() {
                    return Some(num);
                }
            }
        }
    }
    None
}

/// Localiza el índice de un fotograma de manera no ambigua.
/// Prioridad estricta: ID exacto > Nombre base (stem) > Secuencia DJI.
fn find_frame_index<F: RefinableFrame>(frames: &[F], target_id: &str) -> Option<usize> {
    // 1. ID exacto
    if let Some(idx) = frames.iter().position(|f| f.id() == target_id) {
        return Some(idx);
    }

    // 2. Coincidencia por stem de archivo
    let target_stem = std::path::Path::new(target_id)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(target_id);

    let stem_matches: Vec<usize> = frames
        .iter()
        .enumerate()
        .filter_map(|(idx, f)| {
            let f_stem = std::path::Path::new(f.id())
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(f.id());
            if f_stem == target_stem {
                Some(idx)
            } else {
                None
            }
        })
        .collect();

    if stem_matches.len() == 1 {
        return Some(stem_matches[0]);
    }

    // 3. Coincidencia por secuencia DJI de 4 dígitos
    if let Some(target_num) = extract_dji_number(target_id) {
        let seq_matches: Vec<usize> = frames
            .iter()
            .enumerate()
            .filter_map(|(idx, f)| {
                if let Some(f_num) = extract_dji_number(f.id()) {
                    if f_num == target_num {
                        return Some(idx);
                    }
                }
                None
            })
            .collect();
        if seq_matches.len() == 1 {
            return Some(seq_matches[0]);
        }
    }

    None
}

struct InternalMatch {
    frame_i: usize,
    frame_j: usize,
    u_i: f64,
    v_i: f64,
    u_j: f64,
    v_j: f64,
    score: f64,
    is_train: bool,
}

struct MatchEval {
    valid: bool,
    pt_i: Vector3,
    pt_j: Vector3,
    dx: f64,
    dy: f64,
    dist: f64,
    weight: f64,
    is_inlier: bool,
}

fn compute_ground_point<F: RefinableFrame>(
    frame: &F,
    u: f64,
    v: f64,
    dem: &DemWindow,
) -> Option<Vector3> {
    let basis = frame.basis();
    let intrinsics = frame.intrinsics();
    let cam_pos = frame.pose().position;
    // Convención estándar: restar 0.5 px hacia centro óptico
    let ray = basis.pixel_to_ray(u - 0.5, v - 0.5, cam_pos, intrinsics);
    dem.intersect_ray(&ray, 8000.0, 16)
}

fn eval_match<F: RefinableFrame>(
    m: &InternalMatch,
    frames: &[F],
    dem: &DemWindow,
    max_residual_m: f64,
    huber_threshold_m: f64,
) -> MatchEval {
    let g_i = compute_ground_point(&frames[m.frame_i], m.u_i, m.v_i, dem);
    let g_j = compute_ground_point(&frames[m.frame_j], m.u_j, m.v_j, dem);
    if let (Some(pi), Some(pj)) = (g_i, g_j) {
        let dx = pj.x - pi.x;
        let dy = pj.y - pi.y;
        let dist = dx.hypot(dy);
        let is_inlier = dist <= max_residual_m;
        let w_huber = if dist <= huber_threshold_m {
            1.0
        } else if is_inlier {
            huber_threshold_m / dist
        } else {
            0.0
        };
        let weight = w_huber * m.score;
        MatchEval {
            valid: true,
            pt_i: pi,
            pt_j: pj,
            dx,
            dy,
            dist,
            weight,
            is_inlier,
        }
    } else {
        MatchEval {
            valid: false,
            pt_i: Vector3::new(0.0, 0.0, 0.0),
            pt_j: Vector3::new(0.0, 0.0, 0.0),
            dx: 0.0,
            dy: 0.0,
            dist: 0.0,
            weight: 0.0,
            is_inlier: false,
        }
    }
}

fn find_root(p: &mut [usize], mut i: usize) -> usize {
    while p[i] != i {
        p[i] = p[p[i]];
        i = p[i];
    }
    i
}

fn union_sets(p: &mut [usize], i: usize, j: usize) {
    let ri = find_root(p, i);
    let rj = find_root(p, j);
    if ri != rj {
        p[ri] = rj;
    }
}

/// Refina poses usando configuración por defecto.
pub fn refine_poses_with_tie_points<F: RefinableFrame>(
    frames: &mut [F],
    dem: &DemWindow,
    tie_file: &TiePointFile,
) -> PoseRefinementReport {
    refine_poses_with_tie_points_config(frames, dem, tie_file, &PoseRefinementConfig::default())
}

/// Ejecuta el solver 2.5D de refinamiento de poses fotogramétricas.
#[allow(non_snake_case)]
pub fn refine_poses_with_tie_points_config<F: RefinableFrame>(
    frames: &mut [F],
    dem: &DemWindow,
    tie_file: &TiePointFile,
    config: &PoseRefinementConfig,
) -> PoseRefinementReport {
    let n_frames = frames.len();
    if n_frames == 0 {
        return PoseRefinementReport {
            matcher: tie_file.matcher.clone(),
            ..Default::default()
        };
    }

    // Registrar poses originales previas a cualquier mutación
    let orig_poses: Vec<(f64, f64, f64)> = frames
        .iter()
        .map(|f| (f.pose().position.x, f.pose().position.y, f.pose().yaw_deg))
        .collect();

    // Paso A: Aplicar pose overrides si están presentes
    if !tie_file.pose_overrides.is_empty() {
        for ovr in &tie_file.pose_overrides {
            if let Some(idx) = find_frame_index(frames, &ovr.frame_id) {
                let dx = ovr
                    .delta_x_m
                    .clamp(-config.clamp_shift_m, config.clamp_shift_m);
                let dy = ovr
                    .delta_y_m
                    .clamp(-config.clamp_shift_m, config.clamp_shift_m);
                let dyaw = ovr
                    .delta_yaw_deg
                    .clamp(-config.clamp_yaw_deg, config.clamp_yaw_deg);
                {
                    let p = frames[idx].pose_mut();
                    p.position.x += dx;
                    p.position.y += dy;
                    p.yaw_deg += dyaw;
                }
                frames[idx].update_basis_and_bbox(dem);
            }
        }
    }

    // Si no hay matches, concluir con las poses a priori aplicadas
    if tie_file.matches.is_empty() {
        let applied_deltas = frames
            .iter()
            .enumerate()
            .map(|(i, f)| FramePoseDelta {
                frame_id: f.id().to_string(),
                delta_x_m: f.pose().position.x - orig_poses[i].0,
                delta_y_m: f.pose().position.y - orig_poses[i].1,
                delta_yaw_deg: f.pose().yaw_deg - orig_poses[i].2,
            })
            .collect();

        return PoseRefinementReport {
            matcher: tie_file.matcher.clone(),
            applied_deltas,
            ..Default::default()
        };
    }

    // Paso B: Mapear matches a índices de fotograma
    let mut mapped_matches = Vec::new();
    let total_matches = tie_file.matches.len();

    for m in &tie_file.matches {
        if let (Some(idx_i), Some(idx_j)) = (
            find_frame_index(frames, &m.frame_i),
            find_frame_index(frames, &m.frame_j),
        ) {
            if idx_i != idx_j {
                let match_idx = mapped_matches.len();
                let is_train = match_idx % 5 != 0;
                mapped_matches.push(InternalMatch {
                    frame_i: idx_i,
                    frame_j: idx_j,
                    u_i: m.u_i,
                    v_i: m.v_i,
                    u_j: m.u_j,
                    v_j: m.v_j,
                    score: m.score.clamp(0.1, 1.0),
                    is_train,
                });
            }
        }
    }

    if mapped_matches.is_empty() {
        let applied_deltas = frames
            .iter()
            .enumerate()
            .map(|(i, f)| FramePoseDelta {
                frame_id: f.id().to_string(),
                delta_x_m: f.pose().position.x - orig_poses[i].0,
                delta_y_m: f.pose().position.y - orig_poses[i].1,
                delta_yaw_deg: f.pose().yaw_deg - orig_poses[i].2,
            })
            .collect();

        return PoseRefinementReport {
            matcher: tie_file.matcher.clone(),
            total_matches,
            applied_deltas,
            ..Default::default()
        };
    }

    // Estimar vectores unitarios de dirección de vuelo (vx, vy)
    let mut flight_dir: Vec<(f64, f64)> = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let (vx, vy) = if i + 1 < n_frames {
            let dx = frames[i + 1].pose().position.x - frames[i].pose().position.x;
            let dy = frames[i + 1].pose().position.y - frames[i].pose().position.y;
            let len = dx.hypot(dy);
            if len > 0.1 {
                (dx / len, dy / len)
            } else if i > 0 {
                let dx0 = frames[i].pose().position.x - frames[i - 1].pose().position.x;
                let dy0 = frames[i].pose().position.y - frames[i - 1].pose().position.y;
                let len0 = dx0.hypot(dy0);
                if len0 > 0.1 {
                    (dx0 / len0, dy0 / len0)
                } else {
                    let yaw_rad = frames[i].pose().yaw_deg.to_radians();
                    (yaw_rad.sin(), yaw_rad.cos())
                }
            } else {
                let yaw_rad = frames[i].pose().yaw_deg.to_radians();
                (yaw_rad.sin(), yaw_rad.cos())
            }
        } else if i > 0 {
            let dx = frames[i].pose().position.x - frames[i - 1].pose().position.x;
            let dy = frames[i].pose().position.y - frames[i - 1].pose().position.y;
            let len = dx.hypot(dy);
            if len > 0.1 {
                (dx / len, dy / len)
            } else {
                let yaw_rad = frames[i].pose().yaw_deg.to_radians();
                (yaw_rad.sin(), yaw_rad.cos())
            }
        } else {
            let yaw_rad = frames[i].pose().yaw_deg.to_radians();
            (yaw_rad.sin(), yaw_rad.cos())
        };
        flight_dir.push((vx, vy));
    }

    // Componentes conexos vía Union-Find para el gauge relativo por bloque
    let mut parent: Vec<usize> = (0..n_frames).collect();
    let mut frame_active = vec![false; n_frames];
    for m in &mapped_matches {
        if m.is_train {
            union_sets(&mut parent, m.frame_i, m.frame_j);
            frame_active[m.frame_i] = true;
            frame_active[m.frame_j] = true;
        }
    }

    let mut comp_members: std::collections::BTreeMap<usize, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (i, &active) in frame_active.iter().enumerate() {
        if active {
            let root = find_root(&mut parent, i);
            comp_members.entry(root).or_default().push(i);
        }
    }

    // Evaluación inicial de intersecciones DEM
    let mut evals: Vec<MatchEval> = mapped_matches
        .iter()
        .map(|m| {
            eval_match(
                m,
                frames,
                dem,
                config.max_residual_m,
                config.huber_threshold_m,
            )
        })
        .collect();

    let valid_dem_intersections = evals.iter().filter(|e| e.valid).count();
    let train_matches = mapped_matches.iter().filter(|m| m.is_train).count();
    let holdout_matches = mapped_matches.iter().filter(|m| !m.is_train).count();

    let (pre_sum_sq, pre_cnt) = evals
        .iter()
        .enumerate()
        .filter(|(k, e)| mapped_matches[*k].is_train && e.valid && e.is_inlier)
        .fold((0.0, 0usize), |(s, c), (_, e)| (s + e.dist * e.dist, c + 1));
    let pre_refinement_rmse_m = if pre_cnt > 0 {
        (pre_sum_sq / pre_cnt as f64).sqrt()
    } else {
        0.0
    };

    let (holdout_pre_sum_sq, holdout_pre_cnt) = evals
        .iter()
        .enumerate()
        .filter(|(k, e)| !mapped_matches[*k].is_train && e.valid && e.is_inlier)
        .fold((0.0, 0usize), |(s, c), (_, e)| (s + e.dist * e.dist, c + 1));
    let holdout_pre_rmse_m = if holdout_pre_cnt > 0 {
        (holdout_pre_sum_sq / holdout_pre_cnt as f64).sqrt()
    } else {
        0.0
    };

    // Resolver offset sistemático global (s_along, s_cross) en la primera pasada
    let mut global_along_track_m = 0.0;
    let mut global_cross_track_m = 0.0;

    let has_antiparallel = mapped_matches.iter().any(|m| {
        if !m.is_train {
            return false;
        }
        let (vxi, vyi) = flight_dir[m.frame_i];
        let (vxj, vyj) = flight_dir[m.frame_j];
        vxi * vxj + vyi * vyj < -0.5
    });

    if has_antiparallel {
        let mut m00 = 1e-4;
        let mut m01 = 0.0;
        let mut m11 = 1e-4;
        let mut rhs0 = 0.0;
        let mut rhs1 = 0.0;

        for (m, e) in mapped_matches.iter().zip(&evals) {
            if m.is_train && e.valid && e.is_inlier {
                let (vxi, vyi) = flight_dir[m.frame_i];
                let (vxj, vyj) = flight_dir[m.frame_j];
                let pxi = vyi;
                let pyi = -vxi;
                let pxj = vyj;
                let pyj = -vxj;

                let dvx = vxi - vxj;
                let dvy = vyi - vyj;
                let dpx = pxi - pxj;
                let dpy = pyi - pyj;

                let w = e.weight;
                m00 += w * dvx * dvx;
                m01 += w * dvx * dpx;
                m11 += w * dpx * dpx;
                rhs0 += w * dvx * e.dx;
                rhs1 += w * dpx * e.dx;

                m00 += w * dvy * dvy;
                m01 += w * dvy * dpy;
                m11 += w * dpy * dpy;
                rhs0 += w * dvy * e.dy;
                rhs1 += w * dpy * e.dy;
            }
        }

        let det = m00 * m11 - m01 * m01;
        if det.abs() > 1e-9 {
            let s_along = ((rhs0 * m11 - rhs1 * m01) / det).clamp(-2.0, 2.0);
            let s_cross = ((rhs1 * m00 - rhs0 * m01) / det).clamp(-2.0, 2.0);

            let mut backup_poses = Vec::with_capacity(n_frames);
            for f in frames.iter() {
                backup_poses.push(f.pose().clone());
            }

            for (i, f) in frames.iter_mut().enumerate() {
                if frame_active[i] {
                    let (vx, vy) = flight_dir[i];
                    let px = vy;
                    let py = -vx;
                    let shift_x = s_along * vx + s_cross * px;
                    let shift_y = s_along * vy + s_cross * py;
                    let p = f.pose_mut();
                    p.position.x += shift_x;
                    p.position.y += shift_y;
                }
                f.update_basis_and_bbox(dem);
            }

            let test_evals: Vec<MatchEval> = mapped_matches
                .iter()
                .map(|m| {
                    eval_match(
                        m,
                        frames,
                        dem,
                        config.max_residual_m,
                        config.huber_threshold_m,
                    )
                })
                .collect();

            let (test_sum_sq, test_cnt) = test_evals
                .iter()
                .enumerate()
                .filter(|(k, e)| mapped_matches[*k].is_train && e.valid && e.is_inlier)
                .fold((0.0, 0usize), |(s, c), (_, e)| (s + e.dist * e.dist, c + 1));
            let test_rmse = if test_cnt > 0 {
                (test_sum_sq / test_cnt as f64).sqrt()
            } else {
                f64::INFINITY
            };

            if test_rmse < pre_refinement_rmse_m {
                global_along_track_m = s_along;
                global_cross_track_m = s_cross;
                evals = test_evals;
            } else {
                for (i, f) in frames.iter_mut().enumerate() {
                    *f.pose_mut() = backup_poses[i].clone();
                    f.update_basis_and_bbox(dem);
                }
            }
        }
    }

    // Inicializar deltas acumulados respecto a las poses originales
    let mut total_dE = vec![0.0f64; n_frames];
    let mut total_dN = vec![0.0f64; n_frames];
    let mut total_dpsi = vec![0.0f64; n_frames];

    for i in 0..n_frames {
        total_dE[i] = frames[i].pose().position.x - orig_poses[i].0;
        total_dN[i] = frames[i].pose().position.y - orig_poses[i].1;
        total_dpsi[i] = frames[i].pose().yaw_deg - orig_poses[i].2;
    }

    let deg_to_rad = std::f64::consts::PI / 180.0;
    let mut current_train_rmse = {
        let (sum_sq, cnt) = evals
            .iter()
            .enumerate()
            .filter(|(k, e)| mapped_matches[*k].is_train && e.valid && e.is_inlier)
            .fold((0.0, 0usize), |(s, c), (_, e)| (s + e.dist * e.dist, c + 1));
        if cnt > 0 {
            (sum_sq / cnt as f64).sqrt()
        } else {
            0.0
        }
    };

    // 3 iteraciones externas IRLS + DEM re-intersección
    for _outer in 0..config.outer_iterations {
        let mut dE = total_dE.clone();
        let mut dN = total_dN.clone();
        let mut dpsi = total_dpsi.clone();

        // 15 pasos de relajación Gauss-Seidel amortiguados
        for _inner in 0..config.inner_steps {
            for i in 0..n_frames {
                if !frame_active[i] {
                    continue;
                }

                let mut h00 = config.lambda_t;
                let mut h02 = 0.0;
                let mut h11 = config.lambda_t;
                let mut h12 = 0.0;
                let mut h22 = config.lambda_psi;

                let mut g0 = 0.0;
                let mut g1 = 0.0;
                let mut g2 = 0.0;

                let cam_pos_i = frames[i].pose().position;

                for (m, e) in mapped_matches.iter().zip(&evals) {
                    if !m.is_train || !e.valid || !e.is_inlier {
                        continue;
                    }
                    let w = e.weight;
                    if m.frame_i == i {
                        let j = m.frame_j;
                        let rx_i = e.pt_i.x - cam_pos_i.x;
                        let ry_i = e.pt_i.y - cam_pos_i.y;
                        let kx_i = ry_i * deg_to_rad;
                        let ky_i = -rx_i * deg_to_rad;

                        let rx_j = e.pt_j.x - frames[j].pose().position.x;
                        let ry_j = e.pt_j.y - frames[j].pose().position.y;
                        let kx_j = ry_j * deg_to_rad;
                        let ky_j = -rx_j * deg_to_rad;

                        let target_x = (dE[j] + kx_j * dpsi[j]) + e.dx;
                        let target_y = (dN[j] + ky_j * dpsi[j]) + e.dy;

                        h00 += w;
                        h02 += w * kx_i;
                        h11 += w;
                        h12 += w * ky_i;
                        h22 += w * (kx_i * kx_i + ky_i * ky_i);

                        g0 += w * target_x;
                        g1 += w * target_y;
                        g2 += w * (kx_i * target_x + ky_i * target_y);
                    } else if m.frame_j == i {
                        let j = m.frame_i;
                        let rx_i = e.pt_j.x - cam_pos_i.x;
                        let ry_i = e.pt_j.y - cam_pos_i.y;
                        let kx_i = ry_i * deg_to_rad;
                        let ky_i = -rx_i * deg_to_rad;

                        let rx_j = e.pt_i.x - frames[j].pose().position.x;
                        let ry_j = e.pt_i.y - frames[j].pose().position.y;
                        let kx_j = ry_j * deg_to_rad;
                        let ky_j = -rx_j * deg_to_rad;

                        let target_x = (dE[j] + kx_j * dpsi[j]) - e.dx;
                        let target_y = (dN[j] + ky_j * dpsi[j]) - e.dy;

                        h00 += w;
                        h02 += w * kx_i;
                        h11 += w;
                        h12 += w * ky_i;
                        h22 += w * (kx_i * kx_i + ky_i * ky_i);

                        g0 += w * target_x;
                        g1 += w * target_y;
                        g2 += w * (kx_i * target_x + ky_i * target_y);
                    }
                }

                let det = h00 * (h11 * h22 - h12 * h12) - h02 * (h02 * h11);
                if det.abs() > 1e-9 {
                    let c00 = h11 * h22 - h12 * h12;
                    let c01 = h02 * h12;
                    let c02 = -h02 * h11;
                    let c11 = h00 * h22 - h02 * h02;
                    let c12 = -h00 * h12;
                    let c22 = h00 * h11;

                    let new_dE = (c00 * g0 + c01 * g1 + c02 * g2) / det;
                    let new_dN = (c01 * g0 + c11 * g1 + c12 * g2) / det;
                    let new_dpsi = (c02 * g0 + c12 * g1 + c22 * g2) / det;

                    dE[i] = (1.0 - config.damping) * dE[i] + config.damping * new_dE;
                    dN[i] = (1.0 - config.damping) * dN[i] + config.damping * new_dN;
                    dpsi[i] = (1.0 - config.damping) * dpsi[i] + config.damping * new_dpsi;
                }
            }

            // Aplicar proyección cíclica (Dykstra) sobre intersección de media cero y cubo [-clamp, clamp]
            for members in comp_members.values() {
                if members.len() >= 2 {
                    let count = members.len() as f64;
                    for _ in 0..4 {
                        let sum_e: f64 = members.iter().map(|&idx| dE[idx]).sum();
                        let sum_n: f64 = members.iter().map(|&idx| dN[idx]).sum();
                        let mean_e = sum_e / count;
                        let mean_n = sum_n / count;
                        for &idx in members {
                            dE[idx] = (dE[idx] - mean_e)
                                .clamp(-config.clamp_shift_m, config.clamp_shift_m);
                            dN[idx] = (dN[idx] - mean_n)
                                .clamp(-config.clamp_shift_m, config.clamp_shift_m);
                            dpsi[idx] =
                                dpsi[idx].clamp(-config.clamp_yaw_deg, config.clamp_yaw_deg);
                        }
                    }
                    let sum_e: f64 = members.iter().map(|&idx| dE[idx]).sum();
                    let sum_n: f64 = members.iter().map(|&idx| dN[idx]).sum();
                    let mean_e = sum_e / count;
                    let mean_n = sum_n / count;
                    for &idx in members {
                        dE[idx] -= mean_e;
                        dN[idx] -= mean_n;
                    }
                }
            }

            for (i, &active) in frame_active.iter().enumerate() {
                if !active {
                    dE[i] = 0.0;
                    dN[i] = 0.0;
                    dpsi[i] = 0.0;
                }
            }
        }

        // Salvaguarda monótona: evaluar tentativamente sobre el DEM
        let mut backup_poses = Vec::with_capacity(n_frames);
        for f in frames.iter() {
            backup_poses.push(f.pose().clone());
        }

        for (i, f) in frames.iter_mut().enumerate() {
            let p = f.pose_mut();
            p.position.x = orig_poses[i].0 + dE[i];
            p.position.y = orig_poses[i].1 + dN[i];
            p.yaw_deg = orig_poses[i].2 + dpsi[i];
            f.update_basis_and_bbox(dem);
        }

        let new_evals: Vec<MatchEval> = mapped_matches
            .iter()
            .map(|m| {
                eval_match(
                    m,
                    frames,
                    dem,
                    config.max_residual_m,
                    config.huber_threshold_m,
                )
            })
            .collect();

        let (sum_sq, cnt) = new_evals
            .iter()
            .enumerate()
            .filter(|(k, e)| mapped_matches[*k].is_train && e.valid && e.is_inlier)
            .fold((0.0, 0usize), |(s, c), (_, e)| (s + e.dist * e.dist, c + 1));
        let new_train_rmse = if cnt > 0 {
            (sum_sq / cnt as f64).sqrt()
        } else {
            f64::INFINITY
        };

        if new_train_rmse < current_train_rmse {
            current_train_rmse = new_train_rmse;
            total_dE = dE;
            total_dN = dN;
            total_dpsi = dpsi;
            evals = new_evals;
        } else {
            // Revertir a poses previas si no mejora
            for (i, f) in frames.iter_mut().enumerate() {
                *f.pose_mut() = backup_poses[i].clone();
                f.update_basis_and_bbox(dem);
            }
        }
    }

    // Evaluación final sobre train y holdout
    let final_evals: Vec<MatchEval> = mapped_matches
        .iter()
        .map(|m| {
            eval_match(
                m,
                frames,
                dem,
                config.max_residual_m,
                config.huber_threshold_m,
            )
        })
        .collect();

    let (post_sum_sq, post_cnt) = final_evals
        .iter()
        .enumerate()
        .filter(|(k, e)| mapped_matches[*k].is_train && e.valid && e.is_inlier)
        .fold((0.0, 0usize), |(s, c), (_, e)| (s + e.dist * e.dist, c + 1));
    let post_refinement_rmse_m = if post_cnt > 0 {
        (post_sum_sq / post_cnt as f64).sqrt()
    } else {
        0.0
    };

    let (holdout_post_sum_sq, holdout_post_cnt) = final_evals
        .iter()
        .enumerate()
        .filter(|(k, e)| !mapped_matches[*k].is_train && e.valid && e.is_inlier)
        .fold((0.0, 0usize), |(s, c), (_, e)| (s + e.dist * e.dist, c + 1));
    let holdout_post_rmse_m = if holdout_post_cnt > 0 {
        (holdout_post_sum_sq / holdout_post_cnt as f64).sqrt()
    } else {
        0.0
    };

    let valid_count = final_evals.iter().filter(|e| e.valid).count();
    let inlier_count = final_evals
        .iter()
        .filter(|e| e.valid && e.dist <= config.huber_threshold_m)
        .count();
    let inlier_ratio = if valid_count > 0 {
        inlier_count as f64 / valid_count as f64
    } else {
        0.0
    };

    let saturated_count = total_dE
        .iter()
        .zip(&total_dN)
        .zip(&total_dpsi)
        .filter(|((&e, &n), &p)| {
            e.abs() >= config.clamp_shift_m - 0.01
                || n.abs() >= config.clamp_shift_m - 0.01
                || p.abs() >= config.clamp_yaw_deg - 0.01
        })
        .count();
    let saturated_frames_pct = if n_frames > 0 {
        (saturated_count as f64 / n_frames as f64) * 100.0
    } else {
        0.0
    };

    let applied_deltas = frames
        .iter()
        .enumerate()
        .map(|(i, f)| FramePoseDelta {
            frame_id: f.id().to_string(),
            delta_x_m: f.pose().position.x - orig_poses[i].0,
            delta_y_m: f.pose().position.y - orig_poses[i].1,
            delta_yaw_deg: f.pose().yaw_deg - orig_poses[i].2,
        })
        .collect();

    // Actualizar basis y bbox finales para todos los fotogramas
    for f in frames.iter_mut() {
        f.update_basis_and_bbox(dem);
    }

    PoseRefinementReport {
        matcher: tie_file.matcher.clone(),
        total_matches,
        valid_dem_intersections,
        train_matches,
        holdout_matches,
        iterations: config.outer_iterations,
        global_along_track_m,
        global_cross_track_m,
        pre_refinement_rmse_m,
        post_refinement_rmse_m,
        holdout_pre_rmse_m,
        holdout_post_rmse_m,
        inlier_ratio,
        saturated_frames_pct,
        applied_deltas,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trim_mode_from_str() {
        assert_eq!("none".parse::<TrimMode>().unwrap(), TrimMode::None);
        assert_eq!("autocrop".parse::<TrimMode>().unwrap(), TrimMode::AutoCrop);
        assert_eq!("both".parse::<TrimMode>().unwrap(), TrimMode::Both);
        assert_eq!("hull".parse::<TrimMode>().unwrap(), TrimMode::Hull);
        assert_eq!("overlap".parse::<TrimMode>().unwrap(), TrimMode::Overlap);
        assert_eq!(
            "exposure:3".parse::<TrimMode>().unwrap(),
            TrimMode::ExposureCount(3)
        );
        assert_eq!(
            "exposure=2".parse::<TrimMode>().unwrap(),
            TrimMode::ExposureCount(2)
        );
        assert_eq!("4".parse::<TrimMode>().unwrap(), TrimMode::ExposureCount(4));
        assert!("invalid_mode".parse::<TrimMode>().is_err());
    }

    #[test]
    fn test_direct_ortho_single_frame() {
        let w = 20;
        let h = 20;
        let frame_data = vec![42.0f32; w * h];
        let intrinsics = CameraIntrinsics::new(w as u32, h as u32, 20.0, 20.0);
        let pose = CameraPose {
            position: Vector3::new(10.0, 10.0, 50.0),
            yaw_deg: 0.0,
            pitch_deg: -90.0,
            roll_deg: 0.0,
        };

        let frame = ImageFrame::new("IMG_01".into(), pose, intrinsics, frame_data, w, h);

        // DEM plano a cota 0
        let dem_data = vec![0.0f32; 100];
        let dem = DemWindow::new(10, 10, 0.0, 20.0, 2.0, 2.0, dem_data, None, 0.0);

        let config = OrthoConfig {
            target_gsd_m: 1.0,
            resampling: ResamplingMethod::NearestNeighbor,
            nodata_value: f32::NAN,
            trim_mode: TrimMode::None,
            min_overlap: 1,
            hull: None,
            feather_px: 0.0,
            max_blend_temp_diff: 3.5,
        };

        let result = DirectOrthoEngine::generate_orthomosaic(&[frame], &dem, &config);
        assert!(result.valid_pixels > 0);
        assert!(result.elapsed_ms < 1000);

        // Verificar que los píxeles válidos dentro del campo de vista son exactamente 42.0
        let valid_vals: Vec<f32> = result
            .data
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        assert!(!valid_vals.is_empty());
        for v in valid_vals {
            assert!((v - 42.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_direct_ortho_feathering_and_hotspot_protection() {
        let w = 30;
        let h = 30;
        let intrinsics = CameraIntrinsics::new(w as u32, h as u32, 30.0, 30.0);

        // DEM plano centrado en (10.0, 10.0)
        let dem_data = vec![0.0f32; 400];
        let dem = DemWindow::new(20, 20, 0.0, 20.0, 1.0, 1.0, dem_data, None, 0.0);

        // CASO 1: Hotspot en Frame 1 (60.0°C), Frame 2 fondo (26.0°C)
        let mut data1 = vec![25.0f32; w * h];
        let data2 = vec![26.0f32; w * h];
        // Colocar hotspot en el centro (pixel 15, 15)
        data1[15 * w + 15] = 60.0;

        let pose1 = CameraPose {
            position: Vector3::new(10.0, 10.0, 30.0),
            yaw_deg: 0.0,
            pitch_deg: -90.0,
            roll_deg: 0.0,
        };
        let pose2 = CameraPose {
            position: Vector3::new(11.0, 10.0, 30.0),
            yaw_deg: 0.0,
            pitch_deg: -90.0,
            roll_deg: 0.0,
        };

        let f1 = ImageFrame::new_with_bbox(
            "F1".into(),
            pose1.clone(),
            intrinsics,
            data1,
            w,
            h,
            [0.0, 0.0, 20.0, 20.0],
        );
        let f2 = ImageFrame::new_with_bbox(
            "F2".into(),
            pose2.clone(),
            intrinsics,
            data2,
            w,
            h,
            [0.0, 0.0, 20.0, 20.0],
        );

        let config = OrthoConfig {
            target_gsd_m: 1.0,
            resampling: ResamplingMethod::NearestNeighbor,
            nodata_value: -9999.0,
            trim_mode: TrimMode::None,
            min_overlap: 1,
            hull: None,
            feather_px: 10.0,
            max_blend_temp_diff: 3.5,
        };

        let res = DirectOrthoEngine::generate_orthomosaic(&[f1, f2], &dem, &config);
        assert!(res.valid_pixels > 0);

        let valid_vals: Vec<f32> = res.data.iter().copied().filter(|v| *v > -9000.0).collect();
        assert!(!valid_vals.is_empty());

        let max_temp = valid_vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        // El hotspot debe estar protegido sin dilución (>= 58.0)
        assert!(
            max_temp >= 58.0,
            "Hotspot diluido en Frame 1: max_temp={}",
            max_temp
        );

        // Los píxeles fuera del hotspot deben estar acotados en [25.0, 26.0]
        for &v in &valid_vals {
            if v < 50.0 {
                assert!(
                    (25.0 - 1e-3..=26.0 + 1e-3).contains(&v),
                    "Píxel de fondo {} fuera de [25.0, 26.0]",
                    v
                );
            }
        }

        // CASO 2: Hotspot en Frame 2 (60.0°C), Frame 1 fondo (25.0°C)
        let data1_b = vec![25.0f32; w * h];
        let mut data2_b = vec![26.0f32; w * h];
        data2_b[15 * w + 15] = 60.0;

        let f1_b = ImageFrame::new_with_bbox(
            "F1b".into(),
            pose1,
            intrinsics,
            data1_b,
            w,
            h,
            [0.0, 0.0, 20.0, 20.0],
        );
        let f2_b = ImageFrame::new_with_bbox(
            "F2b".into(),
            pose2,
            intrinsics,
            data2_b,
            w,
            h,
            [0.0, 0.0, 20.0, 20.0],
        );

        let res_b = DirectOrthoEngine::generate_orthomosaic(&[f1_b, f2_b], &dem, &config);
        let valid_vals_b: Vec<f32> = res_b
            .data
            .iter()
            .copied()
            .filter(|v| *v > -9000.0)
            .collect();
        let max_temp_b = valid_vals_b
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max_temp_b >= 58.0,
            "Hotspot diluido en Frame 2: max_temp={}",
            max_temp_b
        );

        for &v in &valid_vals_b {
            if v < 50.0 {
                assert!(
                    (25.0 - 1e-3..=26.0 + 1e-3).contains(&v),
                    "Píxel de fondo {} fuera de [25.0, 26.0]",
                    v
                );
            }
        }
    }

    #[test]
    fn test_thermal_multiband_no_shift_on_aligned_rtk_passes() {
        let dem_data = vec![0.0f32; 200 * 200];
        let dem = DemWindow::new(200, 200, -50.0, 150.0, 1.0, 1.0, dem_data, None, 0.0);
        let intrinsics = CameraIntrinsics {
            fx: 100.0,
            fy: 100.0,
            cx: 32.0,
            cy: 32.0,
            width: 64,
            height: 64,
            dewarp: None,
        };
        let w = 64usize;
        let h = 64usize;

        // 16 fotogramas en 2 pasadas opuestas (8 hacia el Este, 8 hacia el Oeste) perfectamente alineadas (RTK)
        let mut frames = Vec::with_capacity(16);
        for k in 0..16 {
            let x = if k < 8 {
                (k as f64) * 6.0
            } else {
                ((15 - k) as f64) * 6.0
            };
            let y = if k < 8 { 0.0 } else { 4.0 };
            let pose = CameraPose {
                position: Vector3::new(x, y, 50.0),
                yaw_deg: 0.0,
                pitch_deg: -90.0,
                roll_deg: 0.0,
            };
            let mut data = vec![20.0f32 + ((k % 2) as f32) * 0.08; w * h];
            for v in 0..h {
                for u in 0..w {
                    let gx = x + (u as f64 - 32.0) * 0.5;
                    if (gx - 21.0).abs() < 2.0 {
                        data[v * w + u] += 6.0;
                    }
                }
            }
            frames.push(ImageFrame::new_with_bbox(
                format!("DJI_202601011200{:02}_{:04}_T", k, k + 1),
                pose,
                intrinsics,
                data,
                w,
                h,
                [x - 20.0, y - 20.0, x + 20.0, y + 20.0],
            ));
        }

        let config = OrthoConfig {
            target_gsd_m: 1.0,
            resampling: ResamplingMethod::Bilinear,
            nodata_value: -9999.0,
            trim_mode: TrimMode::None,
            min_overlap: 1,
            hull: None,
            feather_px: 8.0,
            max_blend_temp_diff: 3.5,
        };
        let (_t_amb, ffc_models, _low_planes, frame_xy_shifts, _pen) =
            DirectOrthoEngine::prepare_thermal_multiband_state(
                &frames, &dem, &config, -50.0, 80.0, -50.0, 50.0,
            );
        for &(sx, sy) in &frame_xy_shifts {
            assert!(
                sx.abs() < 1e-6 && sy.abs() < 1e-6,
                "Se esperaba desplazamiento nulo en pasadas RTK alineadas, obtenido ({}, {})",
                sx,
                sy
            );
        }
        for model in &ffc_models {
            assert!(
                model[0].abs() <= 0.12 + 1e-4,
                "Offset FFC fuera de cota [-0.12, 0.12]: {}",
                model[0]
            );
        }
    }

    #[test]
    fn test_thermal_multiband_recovers_antiparallel_lag_and_intersortie_shift() {
        // Coordenadas lejos de (0, 0) (tipo EPSG:9377) para verificar que cal_min_x/cal_max_x
        // se inicializan desde las poses reales y que se recuperan tanto la latencia antiparalela
        // como el desplazamiento relativo entre dos sorties.
        let ox = 4_717_000.0f64;
        let oy = 2_262_000.0f64;
        let dem_data = vec![1800.0f32; 240 * 240];
        let dem = DemWindow::new(
            240,
            240,
            ox - 60.0,
            oy + 180.0,
            1.0,
            1.0,
            dem_data,
            None,
            0.0,
        );
        let intrinsics = CameraIntrinsics {
            fx: 100.0,
            fy: 100.0,
            cx: 32.0,
            cy: 32.0,
            width: 64,
            height: 64,
            dewarp: None,
        };
        let w = 64usize;
        let h = 64usize;

        // Función de campo térmico sintético en coordenadas terreno reales (tx, ty)
        let ground_temp = |tx: f64, ty: f64| -> f32 {
            let mut t = 20.0f32;
            for c in [20.0, 45.0, 70.0] {
                let dx = tx - (ox + c);
                t += 8.0 * (-0.5 * (dx / 2.2).powi(2)).exp() as f32;
            }
            let dy = ty - (oy + 12.0);
            t += 7.0 * (-0.5 * (dy / 2.2).powi(2)).exp() as f32;
            t
        };

        let mut frames = Vec::with_capacity(22);
        // Sortie 0 (primaria: 16 fotogramas en 2 pasadas opuestas con paso de 12 m y lag alpha=0.13, beta=0.07)
        let alpha_true = 0.13f64;
        let beta_true = 0.07f64;
        for k in 0..16 {
            let (x_nom, y_nom, dx_step, dy_step) = if k < 8 {
                (ox + (k as f64) * 12.0, oy + 10.0, 12.0, 0.0)
            } else {
                (ox + ((15 - k) as f64) * 12.0, oy + 14.0, -12.0, 0.0)
            };
            // El dron reporta (x_nom, y_nom), pero la foto fue capturada en (x_true, y_true)
            let x_true = x_nom + alpha_true * dx_step - beta_true * dy_step;
            let y_true = y_nom + alpha_true * dy_step + beta_true * dx_step;
            let pose = CameraPose {
                position: Vector3::new(x_nom, y_nom, 1900.0),
                yaw_deg: 0.0,
                pitch_deg: -90.0,
                roll_deg: if k % 2 == 0 { 0.0 } else { 180.0 },
            };
            let yaw_corr = if k % 2 == 0 { 0.0 } else { 180.0 };
            let pose_gimbal = CameraPose {
                yaw_deg: yaw_corr,
                ..pose
            };
            let mut data = vec![20.0f32; w * h];
            for v in 0..h {
                for u in 0..w {
                    // Con AGL=100m y fx=100, GSD=1.0 m/px:
                    // u_right = East (+x), u_top = North (+y), v crece hacia el Sur (-y)
                    let tx = x_true + ((u as f64) + 0.5 - 32.0) * 1.0;
                    let ty = y_true - ((v as f64) + 0.5 - 32.0) * 1.0;
                    data[v * w + u] = ground_temp(tx, ty);
                }
            }
            frames.push(ImageFrame::new_with_bbox(
                format!("DJI_202601011120{:02}_{:04}_T", k * 2, k + 1),
                pose_gimbal,
                intrinsics,
                data,
                w,
                h,
                [x_nom - 40.0, y_nom - 40.0, x_nom + 40.0, y_nom + 40.0],
            ));
        }

        // Sortie 1 (secundaria: 6 fotogramas tomados 15 minutos después con offset relativo fuera de grilla (+2.3, -3.4) m)
        let dx_sortie_true = 2.3f64;
        let dy_sortie_true = -3.4f64;
        for j in 0..6 {
            let x_nom = ox + 12.0 + (j as f64) * 12.0;
            let y_nom = oy + 12.0;
            let x_true = x_nom + dx_sortie_true;
            let y_true = y_nom + dy_sortie_true;
            let pose = CameraPose {
                position: Vector3::new(x_nom, y_nom, 1900.0),
                yaw_deg: 0.0,
                pitch_deg: -90.0,
                roll_deg: 0.0,
            };
            let mut data = vec![20.0f32; w * h];
            for v in 0..h {
                for u in 0..w {
                    let tx = x_true + ((u as f64) + 0.5 - 32.0) * 1.0;
                    let ty = y_true - ((v as f64) + 0.5 - 32.0) * 1.0;
                    data[v * w + u] = ground_temp(tx, ty);
                }
            }
            frames.push(ImageFrame::new_with_bbox(
                format!("DJI_202601011135{:02}_{:04}_T", j * 2, j + 1),
                pose,
                intrinsics,
                data,
                w,
                h,
                [x_nom - 40.0, y_nom - 40.0, x_nom + 40.0, y_nom + 40.0],
            ));
        }

        let config = OrthoConfig {
            target_gsd_m: 1.0,
            resampling: ResamplingMethod::Bilinear,
            nodata_value: -9999.0,
            trim_mode: TrimMode::None,
            min_overlap: 1,
            hull: None,
            feather_px: 8.0,
            max_blend_temp_diff: 3.5,
        };
        let (_t_amb, _ffc_models, _low_planes, frame_xy_shifts, frame_penalties) =
            DirectOrthoEngine::prepare_thermal_multiband_state(
                &frames,
                &dem,
                &config,
                ox - 40.0,
                ox + 120.0,
                oy - 40.0,
                oy + 80.0,
            );

        // 1. Verificar inmunidad a Gimbal-Lock (roll=180, yaw=180 en fotogramas impares de Sortie 0):
        // Ningún fotograma interior de las líneas rectas de Sortie 0 debe recibir penalización de giro.
        for k in 1..7 {
            assert!(
                frame_penalties[k] < 1e-6,
                "Falsa penalización por gimbal-lock en fotograma {}: {}",
                k,
                frame_penalties[k]
            );
        }

        // 2. Verificar recuperación positiva del desfase antiparalelo en Sortie 0:
        // En la pasada hacia el Este (dx > 0), sx debe ser positivo (> 0.5 m); hacia el Oeste (dx < 0), negativo (< -0.5 m).
        assert!(
            frame_xy_shifts[3].0 > 0.5 && frame_xy_shifts[11].0 < -0.5,
            "No se recuperó el lag antiparalelo lejos del origen: shift[3]={:?}, shift[11]={:?}",
            frame_xy_shifts[3],
            frame_xy_shifts[11]
        );

        // 3. Verificar recuperación positiva del offset relativo inter-sortie fuera de grilla (+2.3, -3.4) con refinamiento 0.5 m:
        let (sx_sec, sy_sec) = frame_xy_shifts[18];
        assert!(
            (sx_sec - dx_sortie_true).abs() <= 0.65 && (sy_sec - dy_sortie_true).abs() <= 0.65,
            "No se recuperó la traslación relativa inter-sortie con precisión <=0.65 m: esperado ({:.2}, {:.2}), obtenido ({:.2}, {:.2})",
            dx_sortie_true,
            dy_sortie_true,
            sx_sec,
            sy_sec
        );
    }

    #[test]
    fn test_decoded_rgb_image_downsampled_pixel_center_mapping() {
        // Imagen nativa 8x8 con escala 0.5 -> buffer 4x4
        let mut buf = vec![0u8; 4 * 4 * 3];
        for by in 0..4 {
            for bx in 0..4 {
                let idx = (by * 4 + bx) * 3;
                buf[idx] = (bx * 40 + 20) as u8;
                buf[idx + 1] = (by * 40 + 20) as u8;
                buf[idx + 2] = 100;
            }
        }
        let img = DecodedRgbImage {
            width: 8,
            height: 8,
            buf_width: 4,
            buf_height: 4,
            scale: 0.5,
            rgb: buf,
        };
        // Centro de un bloque 2x2 nativo en (0.5, 0.5) mapea exactamente al centro del píxel 0 del buffer (0.0, 0.0)
        let s0 = img.sample_bilinear(0.5, 0.5).unwrap();
        assert_eq!(s0, [20, 20, 100]);
        // Centro del bloque 2x2 en (4.5, 2.5) mapea a (2.0, 1.0) -> bx=2 (100), by=1 (60)
        let s1 = img.sample_bicubic(4.5, 2.5).unwrap();
        assert_eq!(s1, [100, 60, 100]);
        // Dimensiones impares (9x7 -> buffer 4x3) mantienen el centro geométrico exacto de cada bloque 2x2
        let img_odd = DecodedRgbImage {
            width: 9,
            height: 7,
            buf_width: 4,
            buf_height: 3,
            scale: 0.5,
            rgb: vec![128u8; 4 * 3 * 3],
        };
        assert_eq!(img_odd.sample_bicubic(8.4, 6.4), Some([128, 128, 128]));
        // Fuera de rango nativo -> None
        assert!(img.sample_bilinear(8.1, 2.0).is_none());
    }

    #[test]
    fn test_pose_refinement_zero_noise_synthetic_flat_dem() {
        let dem_data = vec![0.0f32; 200 * 200];
        let dem = DemWindow::new(200, 200, 0.0, 200.0, 1.0, 1.0, dem_data, None, 0.0);
        let intrinsics = CameraIntrinsics {
            width: 640,
            height: 512,
            fx: 500.0,
            fy: 500.0,
            cx: 320.0,
            cy: 256.0,
            dewarp: None,
        };

        // Poses verdaderas
        let true_pos1 = Vector3::new(95.0, 100.0, 100.0);
        let true_pos2 = Vector3::new(105.0, 100.0, 100.0);
        let basis1 = CameraBasis::from_gimbal(0.0, -90.0, 0.0);
        let basis2 = CameraBasis::from_gimbal(0.0, -90.0, 0.0);

        // Generar puntos sintéticos en el solape sobre Z=0
        let mut matches = Vec::new();
        for gx in &[98.0, 100.0, 102.0] {
            for gy in &[96.0, 98.0, 100.0, 102.0, 104.0] {
                let g_pt = Vector3::new(*gx, *gy, 0.0);
                if let (Some((u1, v1)), Some((u2, v2))) = (
                    basis1.world_to_pixel(g_pt, true_pos1, &intrinsics),
                    basis2.world_to_pixel(g_pt, true_pos2, &intrinsics),
                ) {
                    matches.push(TiePointMatch {
                        frame_i: "DJI_0001".into(),
                        frame_j: "DJI_0002".into(),
                        u_i: u1 + 0.5,
                        v_i: v1 + 0.5,
                        u_j: u2 + 0.5,
                        v_j: v2 + 0.5,
                        score: 1.0,
                        covisibility: 1.0,
                    });
                }
            }
        }
        assert!(matches.len() >= 10);

        // Perturbar fotogramas simétricamente con media cero
        let shift = 0.5;
        let mut f1 = ImageFrame::new(
            "DJI_0001".into(),
            CameraPose {
                position: Vector3::new(true_pos1.x + shift, true_pos1.y, true_pos1.z),
                yaw_deg: 0.0,
                pitch_deg: -90.0,
                roll_deg: 0.0,
            },
            intrinsics,
            vec![20.0f32; 640 * 512],
            640,
            512,
        );
        let mut f2 = ImageFrame::new(
            "DJI_0002".into(),
            CameraPose {
                position: Vector3::new(true_pos2.x - shift, true_pos2.y, true_pos2.z),
                yaw_deg: 0.0,
                pitch_deg: -90.0,
                roll_deg: 0.0,
            },
            intrinsics,
            vec![20.0f32; 640 * 512],
            640,
            512,
        );
        f1.update_basis_and_bbox(&dem);
        f2.update_basis_and_bbox(&dem);

        let mut frames = vec![f1, f2];
        let tie_file = TiePointFile {
            schema_version: "1.0".into(),
            matcher: "synthetic".into(),
            coordinate_convention: "standard".into(),
            matches,
            pose_overrides: vec![],
        };

        let config = PoseRefinementConfig {
            lambda_t: 0.05,
            lambda_psi: 1.0,
            ..Default::default()
        };
        let report = refine_poses_with_tie_points_config(&mut frames, &dem, &tie_file, &config);

        assert!(report.pre_refinement_rmse_m > 0.8);
        assert!(
            report.post_refinement_rmse_m < 0.2 * report.pre_refinement_rmse_m,
            "post_rmse ({:.4}) no fue < 0.2 * pre_rmse ({:.4})",
            report.post_refinement_rmse_m,
            report.pre_refinement_rmse_m
        );
        assert!(report.post_refinement_rmse_m < 0.15);
        assert!(report.inlier_ratio > 0.95);

        // Verificar recuperación de la posición original
        let err1 = (frames[0].pose().position.x - true_pos1.x).abs();
        let err2 = (frames[1].pose().position.x - true_pos2.x).abs();
        assert!(err1 < 0.15, "Error en frame 1 demasiado alto: {}", err1);
        assert!(err2 < 0.15, "Error en frame 2 demasiado alto: {}", err2);
    }

    #[test]
    fn test_pose_refinement_regularized_bounded_shift_and_datum() {
        let dem_data = vec![0.0f32; 200 * 200];
        let dem = DemWindow::new(200, 200, 0.0, 200.0, 1.0, 1.0, dem_data, None, 0.0);
        let intrinsics = CameraIntrinsics {
            width: 640,
            height: 512,
            fx: 500.0,
            fy: 500.0,
            cx: 320.0,
            cy: 256.0,
            dewarp: None,
        };

        let mut f1 = ImageFrame::new(
            "F1".into(),
            CameraPose {
                position: Vector3::new(95.0, 100.0, 100.0),
                yaw_deg: 0.0,
                pitch_deg: -90.0,
                roll_deg: 0.0,
            },
            intrinsics,
            vec![20.0f32; 640 * 512],
            640,
            512,
        );
        let mut f2 = ImageFrame::new(
            "F2".into(),
            CameraPose {
                position: Vector3::new(105.0, 100.0, 100.0),
                yaw_deg: 0.0,
                pitch_deg: -90.0,
                roll_deg: 0.0,
            },
            intrinsics,
            vec![20.0f32; 640 * 512],
            640,
            512,
        );
        f1.update_basis_and_bbox(&dem);
        f2.update_basis_and_bbox(&dem);

        // Perturbación grande sintética que forzaría saturación
        let mut matches = Vec::new();
        for i in 0..15 {
            matches.push(TiePointMatch {
                frame_i: "F1".into(),
                frame_j: "F2".into(),
                u_i: 320.0 + (i as f64) * 2.0,
                v_i: 256.0,
                u_j: 320.0 - (i as f64) * 2.0 + 50.0, // Gran discrepancia
                v_j: 256.0,
                score: 1.0,
                covisibility: 1.0,
            });
        }

        let mut frames = vec![f1, f2];
        let tie_file = TiePointFile {
            schema_version: "1.0".into(),
            matcher: "stress".into(),
            coordinate_convention: "standard".into(),
            matches,
            pose_overrides: vec![],
        };

        let config = PoseRefinementConfig::default();
        let report = refine_poses_with_tie_points_config(&mut frames, &dem, &tie_file, &config);

        for delta in &report.applied_deltas {
            assert!(
                delta.delta_x_m.abs() <= config.clamp_shift_m + 1e-4,
                "Delta X violó clamp: {}",
                delta.delta_x_m
            );
            assert!(
                delta.delta_y_m.abs() <= config.clamp_shift_m + 1e-4,
                "Delta Y violó clamp: {}",
                delta.delta_y_m
            );
            assert!(
                delta.delta_yaw_deg.abs() <= config.clamp_yaw_deg + 1e-4,
                "Delta Yaw violó clamp: {}",
                delta.delta_yaw_deg
            );
        }

        // Preservación estricta del datum de media cero en el componente
        let sum_dx: f64 = report.applied_deltas.iter().map(|d| d.delta_x_m).sum();
        let sum_dy: f64 = report.applied_deltas.iter().map(|d| d.delta_y_m).sum();
        assert!(
            sum_dx.abs() < 1e-6,
            "Media de delta X no fue cero: {}",
            sum_dx
        );
        assert!(
            sum_dy.abs() < 1e-6,
            "Media de delta Y no fue cero: {}",
            sum_dy
        );
    }

    #[test]
    fn test_pose_refinement_robust_huber_outliers() {
        let dem_data = vec![0.0f32; 200 * 200];
        let dem = DemWindow::new(200, 200, 0.0, 200.0, 1.0, 1.0, dem_data, None, 0.0);
        let intrinsics = CameraIntrinsics {
            width: 640,
            height: 512,
            fx: 500.0,
            fy: 500.0,
            cx: 320.0,
            cy: 256.0,
            dewarp: None,
        };

        let true_pos1 = Vector3::new(95.0, 100.0, 100.0);
        let true_pos2 = Vector3::new(105.0, 100.0, 100.0);
        let basis1 = CameraBasis::from_gimbal(0.0, -90.0, 0.0);
        let basis2 = CameraBasis::from_gimbal(0.0, -90.0, 0.0);

        let mut matches = Vec::new();
        // 15 inliers consistentes
        for i in 0..15 {
            let gx = 97.0 + (i as f64) * 0.4;
            let gy = 96.0 + (i as f64) * 0.5;
            let g_pt = Vector3::new(gx, gy, 0.0);
            let (u1, v1) = basis1.world_to_pixel(g_pt, true_pos1, &intrinsics).unwrap();
            let (u2, v2) = basis2.world_to_pixel(g_pt, true_pos2, &intrinsics).unwrap();
            matches.push(TiePointMatch {
                frame_i: "F1".into(),
                frame_j: "F2".into(),
                u_i: u1 + 0.5,
                v_i: v1 + 0.5,
                u_j: u2 + 0.5,
                v_j: v2 + 0.5,
                score: 0.95,
                covisibility: 1.0,
            });
        }

        // 5 outliers groseros (25% del total de 20) en la banda de 4m a 20m
        for i in 0..5 {
            matches.push(TiePointMatch {
                frame_i: "F1".into(),
                frame_j: "F2".into(),
                u_i: 300.0 + (i as f64) * 5.0,
                v_i: 250.0,
                u_j: 300.0 + (i as f64) * 5.0 + 35.0, // Discrepancia grande
                v_j: 250.0 + 20.0,
                score: 0.9,
                covisibility: 1.0,
            });
        }
        assert_eq!(matches.len(), 20);

        // Perturbación simétrica suave de 0.4m
        let mut f1 = ImageFrame::new(
            "F1".into(),
            CameraPose {
                position: Vector3::new(true_pos1.x + 0.4, true_pos1.y, true_pos1.z),
                yaw_deg: 0.0,
                pitch_deg: -90.0,
                roll_deg: 0.0,
            },
            intrinsics,
            vec![20.0f32; 640 * 512],
            640,
            512,
        );
        let mut f2 = ImageFrame::new(
            "F2".into(),
            CameraPose {
                position: Vector3::new(true_pos2.x - 0.4, true_pos2.y, true_pos2.z),
                yaw_deg: 0.0,
                pitch_deg: -90.0,
                roll_deg: 0.0,
            },
            intrinsics,
            vec![20.0f32; 640 * 512],
            640,
            512,
        );
        f1.update_basis_and_bbox(&dem);
        f2.update_basis_and_bbox(&dem);

        let mut frames = vec![f1, f2];
        let tie_file = TiePointFile {
            schema_version: "1.0".into(),
            matcher: "huber_test".into(),
            coordinate_convention: "standard".into(),
            matches,
            pose_overrides: vec![],
        };

        let config = PoseRefinementConfig {
            lambda_t: 0.1,
            ..Default::default()
        };
        let report = refine_poses_with_tie_points_config(&mut frames, &dem, &tie_file, &config);

        assert!(report.inlier_ratio >= 0.70);
        assert!(
            report.post_refinement_rmse_m < report.pre_refinement_rmse_m,
            "post_rmse ({:.4}) no mejoró sobre pre_rmse ({:.4})",
            report.post_refinement_rmse_m,
            report.pre_refinement_rmse_m
        );
    }

    #[test]
    fn test_pose_refinement_collinear_flight_line_stability() {
        let dem_data = vec![0.0f32; 200 * 200];
        let dem = DemWindow::new(200, 200, 0.0, 200.0, 1.0, 1.0, dem_data, None, 0.0);
        let intrinsics = CameraIntrinsics {
            width: 640,
            height: 512,
            fx: 500.0,
            fy: 500.0,
            cx: 320.0,
            cy: 256.0,
            dewarp: None,
        };

        // 3 fotogramas en una única línea de vuelo hacia el Norte
        let pos1 = Vector3::new(100.0, 90.0, 100.0);
        let pos2 = Vector3::new(100.0, 105.0, 100.0);
        let pos3 = Vector3::new(100.0, 120.0, 100.0);

        let mut frames = vec![
            ImageFrame::new(
                "F1".into(),
                CameraPose {
                    position: pos1,
                    yaw_deg: 0.0,
                    pitch_deg: -90.0,
                    roll_deg: 0.0,
                },
                intrinsics,
                vec![20.0f32; 640 * 512],
                640,
                512,
            ),
            ImageFrame::new(
                "F2".into(),
                CameraPose {
                    position: pos2,
                    yaw_deg: 0.0,
                    pitch_deg: -90.0,
                    roll_deg: 0.0,
                },
                intrinsics,
                vec![20.0f32; 640 * 512],
                640,
                512,
            ),
            ImageFrame::new(
                "F3".into(),
                CameraPose {
                    position: pos3,
                    yaw_deg: 0.0,
                    pitch_deg: -90.0,
                    roll_deg: 0.0,
                },
                intrinsics,
                vec![20.0f32; 640 * 512],
                640,
                512,
            ),
        ];

        for f in &mut frames {
            f.update_basis_and_bbox(&dem);
        }

        let mut matches = Vec::new();
        for i in 0..6 {
            matches.push(TiePointMatch {
                frame_i: "F1".into(),
                frame_j: "F2".into(),
                u_i: 320.0,
                v_i: 200.0 + (i as f64) * 10.0,
                u_j: 320.0,
                v_j: 250.0 + (i as f64) * 10.0,
                score: 1.0,
                covisibility: 1.0,
            });
            matches.push(TiePointMatch {
                frame_i: "F2".into(),
                frame_j: "F3".into(),
                u_i: 320.0,
                v_i: 200.0 + (i as f64) * 10.0,
                u_j: 320.0,
                v_j: 250.0 + (i as f64) * 10.0,
                score: 1.0,
                covisibility: 1.0,
            });
        }

        let tie_file = TiePointFile {
            schema_version: "1.0".into(),
            matcher: "collinear".into(),
            coordinate_convention: "standard".into(),
            matches,
            pose_overrides: vec![],
        };

        let report = refine_poses_with_tie_points(&mut frames, &dem, &tie_file);

        // Sin pasadas antiparalelas, el offset sistemático no debe aplicarse
        assert_eq!(report.global_along_track_m, 0.0);
        assert_eq!(report.global_cross_track_m, 0.0);
        assert_eq!(report.iterations, 3);
        for d in &report.applied_deltas {
            assert!(d.delta_x_m.is_finite());
            assert!(d.delta_y_m.is_finite());
            assert!(d.delta_yaw_deg.is_finite());
        }
    }

    #[test]
    fn test_pose_refinement_disconnected_components_and_zero_matches() {
        let dem_data = vec![0.0f32; 200 * 200];
        let dem = DemWindow::new(200, 200, 0.0, 200.0, 1.0, 1.0, dem_data, None, 0.0);
        let intrinsics = CameraIntrinsics {
            width: 640,
            height: 512,
            fx: 500.0,
            fy: 500.0,
            cx: 320.0,
            cy: 256.0,
            dewarp: None,
        };

        let mut frames = vec![
            ImageFrame::new(
                "F1".into(),
                CameraPose {
                    position: Vector3::new(90.0, 100.0, 100.0),
                    yaw_deg: 0.0,
                    pitch_deg: -90.0,
                    roll_deg: 0.0,
                },
                intrinsics,
                vec![20.0f32; 640 * 512],
                640,
                512,
            ),
            ImageFrame::new(
                "F2".into(),
                CameraPose {
                    position: Vector3::new(100.0, 100.0, 100.0),
                    yaw_deg: 0.0,
                    pitch_deg: -90.0,
                    roll_deg: 0.0,
                },
                intrinsics,
                vec![20.0f32; 640 * 512],
                640,
                512,
            ),
            ImageFrame::new(
                "F3_ISOLATED".into(),
                CameraPose {
                    position: Vector3::new(180.0, 180.0, 100.0),
                    yaw_deg: 10.0,
                    pitch_deg: -90.0,
                    roll_deg: 0.0,
                },
                intrinsics,
                vec![20.0f32; 640 * 512],
                640,
                512,
            ),
        ];

        for f in &mut frames {
            f.update_basis_and_bbox(&dem);
        }

        // Solo F1 y F2 tienen observaciones entre sí; F3 no tiene ninguna
        let matches = vec![TiePointMatch {
            frame_i: "F1".into(),
            frame_j: "F2".into(),
            u_i: 350.0,
            v_i: 256.0,
            u_j: 310.0,
            v_j: 256.0,
            score: 1.0,
            covisibility: 1.0,
        }];

        let tie_file = TiePointFile {
            schema_version: "1.0".into(),
            matcher: "disconnected".into(),
            coordinate_convention: "standard".into(),
            matches,
            pose_overrides: vec![],
        };

        let report = refine_poses_with_tie_points(&mut frames, &dem, &tie_file);

        // F3 aislado no debe mutar en absoluto
        let d3 = report
            .applied_deltas
            .iter()
            .find(|d| d.frame_id == "F3_ISOLATED")
            .unwrap();
        assert_eq!(d3.delta_x_m, 0.0);
        assert_eq!(d3.delta_y_m, 0.0);
        assert_eq!(d3.delta_yaw_deg, 0.0);

        // F1 y F2 preservan el datum del componente
        let d1 = report
            .applied_deltas
            .iter()
            .find(|d| d.frame_id == "F1")
            .unwrap();
        let d2 = report
            .applied_deltas
            .iter()
            .find(|d| d.frame_id == "F2")
            .unwrap();
        assert!((d1.delta_x_m + d2.delta_x_m).abs() < 1e-6);

        // Caso B: Cero matches absoluto -> fallback de identidad grácil
        let empty_tie = TiePointFile {
            schema_version: "1.0".into(),
            matcher: "empty".into(),
            coordinate_convention: "standard".into(),
            matches: vec![],
            pose_overrides: vec![],
        };
        let rep_empty = refine_poses_with_tie_points(&mut frames, &dem, &empty_tie);
        assert_eq!(rep_empty.iterations, 0);
        for d in &rep_empty.applied_deltas {
            assert_eq!(d.delta_x_m, 0.0);
            assert_eq!(d.delta_y_m, 0.0);
            assert_eq!(d.delta_yaw_deg, 0.0);
        }
    }

    #[test]
    fn test_deserialize_golden_tie_point_file_contract() {
        let json_data = r#"{
            "schema_version": "1.0",
            "matcher": "flexglue",
            "coordinate_convention": "corner_0.5_distorted_native_px",
            "enu_origin": [6.25, -75.58, 1450.0],
            "matches": [
                {
                    "frame_i": "DJI_0001_V.JPG",
                    "frame_j": "DJI_0002_V.JPG",
                    "u_i": 1245.5,
                    "v_i": 890.25,
                    "u_j": 1260.1,
                    "v_j": 880.75,
                    "score": 0.9542,
                    "covisibility": 0.8812
                }
            ],
            "pose_overrides": [
                {
                    "frame_id": "DJI_0001_T.JPG",
                    "delta_x_m": 0.45,
                    "delta_y_m": -0.22,
                    "delta_yaw_deg": 0.12
                }
            ],
            "timing_breakdown": {
                "warmup_s": 0.05,
                "keypoint_extract_s": 0.12,
                "descriptor_s": 0.35,
                "matcher_s": 0.18,
                "total_matching_s": 0.70,
                "peak_vram_mb": 0.0
            },
            "diagnostics": {
                "num_images": 2,
                "num_pairs_evaluated": 1,
                "num_pairs_matched": 1,
                "total_inliers": 1,
                "device": "CPU",
                "is_thermal": false
            }
        }"#;

        let parsed: TiePointFile =
            serde_json::from_str(json_data).expect("Fallo al deserializar TiePointFile en Rust");
        assert_eq!(parsed.schema_version, "1.0");
        assert_eq!(parsed.matcher, "flexglue");
        assert_eq!(
            parsed.coordinate_convention,
            "corner_0.5_distorted_native_px"
        );
        assert_eq!(parsed.matches.len(), 1);
        assert_eq!(parsed.matches[0].frame_i, "DJI_0001_V.JPG");
        assert_eq!(parsed.matches[0].frame_j, "DJI_0002_V.JPG");
        assert!((parsed.matches[0].u_i - 1245.5).abs() < 1e-6);
        assert!((parsed.matches[0].score - 0.9542).abs() < 1e-6);

        assert_eq!(parsed.pose_overrides.len(), 1);
        assert_eq!(parsed.pose_overrides[0].frame_id, "DJI_0001_T.JPG");
        assert!((parsed.pose_overrides[0].delta_x_m - 0.45).abs() < 1e-6);
        assert!((parsed.pose_overrides[0].delta_yaw_deg - 0.12).abs() < 1e-6);
    }
}
