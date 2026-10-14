//! `zenith-ortho`: Motor de ortorrectificación directa 2.5D de ultra alta velocidad.
//! Utiliza paralelismo masivo CPU con `rayon` para proyectar el terreno celda a celda
//! sobre las imágenes fuente, garantizando remuestreo Nearest-Neighbor estricto
//! en bandas radiométricas para preservar valores físicos (°C y reflectancia).

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
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

/// Configuración del proceso de ortorrectificación.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrthoConfig {
    pub target_gsd_m: f64,
    pub resampling: ResamplingMethod,
    pub nodata_value: f32,
}

impl Default for OrthoConfig {
    fn default() -> Self {
        Self {
            target_gsd_m: 0.1, // 10 cm por píxel
            resampling: ResamplingMethod::NearestNeighbor,
            nodata_value: f32::NAN,
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
    pub fn generate_orthomosaic(
        frames: &[ImageFrame],
        dem: &DemWindow,
        config: &OrthoConfig,
    ) -> OrthoResult {
        let t0 = std::time::Instant::now();

        // 1. Calcular el bounding box horizontal del vuelo
        let min_x = frames
            .iter()
            .map(|f| f.pose.position.x)
            .fold(f64::INFINITY, f64::min)
            - 100.0;
        let max_x = frames
            .iter()
            .map(|f| f.pose.position.x)
            .fold(f64::NEG_INFINITY, f64::max)
            + 100.0;
        let min_y = frames
            .iter()
            .map(|f| f.pose.position.y)
            .fold(f64::INFINITY, f64::min)
            - 100.0;
        let max_y = frames
            .iter()
            .map(|f| f.pose.position.y)
            .fold(f64::NEG_INFINITY, f64::max)
            + 100.0;

        let gsd = config.target_gsd_m;
        let width = ((max_x - min_x) / gsd).ceil() as usize;
        let height = ((max_y - min_y) / gsd).ceil() as usize;

        // 2. Mapeo paralelo celda por celda usando Rayon
        let mut output = vec![config.nodata_value; width * height];

        output
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(row, row_slice)| {
                let cell_y = max_y - (row as f64 + 0.5) * gsd;

                for (col, out_cell) in row_slice.iter_mut().enumerate() {
                    let cell_x = min_x + (col as f64 + 0.5) * gsd;

                    // Muestrear elevación del terreno en (cell_x, cell_y)
                    let elev = match dem.sample_elevation_bilinear(cell_x, cell_y) {
                        Some(z) => z,
                        None => continue,
                    };

                    let ground_pt = Vector3::new(cell_x, cell_y, elev);

                    // Buscar el fotograma con el mejor ángulo de observación (más cercano a nadir)
                    let mut best_val: Option<f32> = None;
                    let mut best_off_nadir_cos = -1.0;

                    for frame in frames {
                        let cam_pos = frame.pose.position;
                        if let Some((u, v_px)) =
                            frame
                                .basis
                                .world_to_pixel(ground_pt, cam_pos, &frame.intrinsics)
                        {
                            if u >= 0.0
                                && u < frame.width as f64
                                && v_px >= 0.0
                                && v_px < frame.height as f64
                            {
                                let v = ground_pt - cam_pos;
                                let view_dir = v.normalize();
                                let cos_angle = -view_dir.z; // Máximo cos_angle = mirada más vertical

                                if cos_angle > best_off_nadir_cos {
                                    if let Some(val) =
                                        frame.sample_pixel(u, v_px, config.resampling)
                                    {
                                        best_off_nadir_cos = cos_angle;
                                        best_val = Some(val);
                                    }
                                }
                            }
                        }
                    }

                    if let Some(val) = best_val {
                        *out_cell = val;
                    }
                }
            });

        let valid_count = output.iter().filter(|v| v.is_finite()).count();
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
        };

        let result = DirectOrthoEngine::generate_orthomosaic(&[frame], &dem, &config);
        assert!(result.valid_pixels > 0);
        assert!(result.elapsed_ms < 1000);
    }
}
