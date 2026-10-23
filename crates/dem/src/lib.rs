use std::path::Path;
use thiserror::Error;
use zenith_core::{Ray3D, Vector3};

#[derive(Error, Debug)]
pub enum DemError {
    #[error("Error de E/S leyendo DEM: {0}")]
    Io(#[from] std::io::Error),
    #[error("Encabezado HDR inválido: {0}")]
    InvalidHeader(String),
    #[error("Tamaño de archivo de datos no coincide con width * height * 4 bytes")]
    InvalidDataSize,
}

/// Ventana en memoria de un Modelo Digital de Elevación (DEM/DSM).
#[derive(Debug, Clone)]
pub struct DemWindow {
    pub width: usize,
    pub height: usize,
    pub origin_x: f64,
    pub origin_y: f64,
    pub cell_size_x: f64,
    pub cell_size_y: f64,
    pub data: Vec<f32>,
    pub nodata: Option<f32>,
    pub geoid_undulation_m: f64,
}

impl DemWindow {
    /// Carga un DEM directamente desde un raster binario ENU/ENVI Float32 (.hdr y .raw/.bin).
    pub fn from_envi_raw(
        hdr_path: impl AsRef<Path>,
        raw_path: impl AsRef<Path>,
        geoid_undulation_m: f64,
    ) -> Result<Self, DemError> {
        let hdr_text = std::fs::read_to_string(hdr_path)?;
        let mut samples = 0usize;
        let mut lines = 0usize;
        let mut origin_x = 0.0f64;
        let mut origin_y = 0.0f64;
        let mut cell_size_x = 1.0f64;
        let mut cell_size_y = 1.0f64;
        let mut nodata = Some(-9999.0f32);

        for line in hdr_text.lines() {
            let line_trim = line.trim();
            if line_trim.starts_with("samples") {
                if let Some(val) = line_trim.split('=').nth(1) {
                    samples = val.trim().parse().unwrap_or(0);
                }
            } else if line_trim.starts_with("lines") {
                if let Some(val) = line_trim.split('=').nth(1) {
                    lines = val.trim().parse().unwrap_or(0);
                }
            } else if line_trim.starts_with("data ignore value") {
                if let Some(val) = line_trim.split('=').nth(1) {
                    nodata = val.trim().parse().ok();
                }
            } else if line_trim.starts_with("map info") {
                // Formato: map info = {projection, ref_x, ref_y, easting, northing, x_size, y_size}
                if let Some(brace_start) = line_trim.find('{') {
                    if let Some(brace_end) = line_trim.rfind('}') {
                        let inner = &line_trim[brace_start + 1..brace_end];
                        let parts: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();
                        if parts.len() >= 7 {
                            origin_x = parts[3].parse().unwrap_or(0.0);
                            origin_y = parts[4].parse().unwrap_or(0.0);
                            cell_size_x = parts[5].parse().unwrap_or(1.0);
                            cell_size_y = parts[6].parse().unwrap_or(1.0);
                        }
                    }
                }
            }
        }

        if samples == 0 || lines == 0 {
            return Err(DemError::InvalidHeader(
                "Dimensiones samples/lines no encontradas en .hdr".into(),
            ));
        }

        let raw_bytes = std::fs::read(raw_path)?;
        let expected_bytes = samples * lines * std::mem::size_of::<f32>();
        if raw_bytes.len() != expected_bytes {
            return Err(DemError::InvalidDataSize);
        }

        let mut data = vec![0.0f32; samples * lines];
        unsafe {
            std::ptr::copy_nonoverlapping(
                raw_bytes.as_ptr() as *const f32,
                data.as_mut_ptr(),
                samples * lines,
            );
        }

        // Extrapolación de borde (Nearest-Boundary Continuation + Relajación 2D de Laplace)
        // para celdas NoData perimetrales: preserva intacto el 100% de las celdas válidas interiores
        // del DEM, pero evita escalones direccionales cuando el DEM recortado tiene bordes diagonales.
        let is_valid = |v: f32| -> bool {
            v.is_finite() && v > -500.0 && nodata.is_none_or(|nd| (v - nd).abs() > 1e-3)
        };
        let valid_mask: Vec<bool> = data.iter().copied().map(is_valid).collect();
        let has_invalid = valid_mask.iter().any(|&ok| !ok);
        let has_valid = valid_mask.iter().any(|&ok| ok);

        if has_invalid && has_valid {
            // Pasada 1: Propagación horizontal por fila (izquierda -> derecha y derecha -> izquierda)
            for r in 0..lines {
                let row_start = r * samples;
                let row = &mut data[row_start..row_start + samples];
                let mut last = None;
                for item in row.iter_mut() {
                    if is_valid(*item) {
                        last = Some(*item);
                    } else if let Some(val) = last {
                        *item = val;
                    }
                }
                let mut next = None;
                for item in row.iter_mut().rev() {
                    if is_valid(*item) {
                        next = Some(*item);
                    } else if let Some(val) = next {
                        *item = val;
                    }
                }
            }

            // Pasada 2: Propagación vertical para filas completamente vacías en el borde superior/inferior
            let mut last_valid_row: Option<usize> = None;
            for r in 0..lines {
                let row_start = r * samples;
                if is_valid(data[row_start]) {
                    last_valid_row = Some(r);
                } else if let Some(src_r) = last_valid_row {
                    let src_start = src_r * samples;
                    for c in 0..samples {
                        data[row_start + c] = data[src_start + c];
                    }
                }
            }
            let mut next_valid_row: Option<usize> = None;
            for r in (0..lines).rev() {
                let row_start = r * samples;
                if is_valid(data[row_start]) {
                    next_valid_row = Some(r);
                } else if let Some(src_r) = next_valid_row {
                    let src_start = src_r * samples;
                    for c in 0..samples {
                        data[row_start + c] = data[src_start + c];
                    }
                }
            }

            // Pasada 3: Relajación armónica 2D de Laplace (Gauss-Seidel, 36 iteraciones) exclusivamente
            // sobre las celdas originalmente NoData para eliminar discontinuidades de barrido 1D.
            if lines >= 3 && samples >= 3 {
                for _iter in 0..36 {
                    for r in 0..lines {
                        let r_up = r.saturating_sub(1);
                        let r_dn = (r + 1).min(lines - 1);
                        let row_off = r * samples;
                        let up_off = r_up * samples;
                        let dn_off = r_dn * samples;
                        for c in 0..samples {
                            let idx = row_off + c;
                            if !valid_mask[idx] {
                                let c_lt = c.saturating_sub(1);
                                let c_rt = (c + 1).min(samples - 1);
                                data[idx] = 0.25
                                    * (data[up_off + c]
                                        + data[dn_off + c]
                                        + data[row_off + c_lt]
                                        + data[row_off + c_rt]);
                            }
                        }
                    }
                }
            }
        }

        Ok(Self {
            width: samples,
            height: lines,
            origin_x,
            origin_y,
            cell_size_x,
            cell_size_y,
            data,
            nodata,
            geoid_undulation_m,
        })
    }
    /// Crea una ventana de DEM a partir de un arreglo plano de valores de elevación.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        width: usize,
        height: usize,
        origin_x: f64,
        origin_y: f64,
        cell_size_x: f64,
        cell_size_y: f64,
        data: Vec<f32>,
        nodata: Option<f32>,
        geoid_undulation_m: f64,
    ) -> Self {
        assert_eq!(
            data.len(),
            width * height,
            "Dimensiones de datos no coinciden con width x height"
        );
        Self {
            width,
            height,
            origin_x,
            origin_y,
            cell_size_x,
            cell_size_y,
            data,
            nodata,
            geoid_undulation_m,
        }
    }

    /// Crea un plano de elevación constante sobre una extensión geográfica especificada.
    pub fn from_flat(
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
        cell_size: f64,
        elevation: f32,
    ) -> Self {
        let width = ((max_x - min_x) / cell_size).ceil() as usize + 2;
        let height = ((max_y - min_y) / cell_size).ceil() as usize + 2;
        let data = vec![elevation; width * height];
        Self {
            width,
            height,
            origin_x: min_x,
            origin_y: max_y,
            cell_size_x: cell_size,
            cell_size_y: cell_size,
            data,
            nodata: None,
            geoid_undulation_m: 0.0,
        }
    }

    /// Obtiene la resolución espacial mínima del DEM.
    #[inline]
    pub fn resolution(&self) -> f64 {
        self.cell_size_x.abs().min(self.cell_size_y.abs())
    }

    /// Retorna true si la ventana contiene celdas válidas de DEM.
    #[inline]
    pub fn has_dem(&self) -> bool {
        !self.data.is_empty() && self.width > 0 && self.height > 0
    }

    /// Muestrea la elevación continua en coordenadas mundiales (X, Y) mediante interpolación bilineal.
    pub fn sample_elevation_bilinear(&self, x: f64, y: f64) -> Option<f64> {
        let px = (x - self.origin_x) / self.cell_size_x;
        let py = (self.origin_y - y) / self.cell_size_y;

        if px < 0.0
            || px >= (self.width as f64 - 1.0)
            || py < 0.0
            || py >= (self.height as f64 - 1.0)
        {
            return None;
        }

        let c0 = px.floor() as usize;
        let r0 = py.floor() as usize;
        let c1 = c0 + 1;
        let r1 = r0 + 1;

        let fx = px - c0 as f64;
        let fy = py - r0 as f64;

        let idx00 = r0 * self.width + c0;
        let idx10 = r0 * self.width + c1;
        let idx01 = r1 * self.width + c0;
        let idx11 = r1 * self.width + c1;

        let v00 = self.data[idx00];
        let v10 = self.data[idx10];
        let v01 = self.data[idx01];
        let v11 = self.data[idx11];

        if !v00.is_finite() || !v10.is_finite() || !v01.is_finite() || !v11.is_finite() {
            return None;
        }

        if let Some(nd) = self.nodata {
            if (v00 - nd).abs() < 1e-4
                || (v10 - nd).abs() < 1e-4
                || (v01 - nd).abs() < 1e-4
                || (v11 - nd).abs() < 1e-4
            {
                return None;
            }
        }

        let top = (1.0 - fx) * v00 as f64 + fx * v10 as f64;
        let bottom = (1.0 - fx) * v01 as f64 + fx * v11 as f64;
        let elev_raw = (1.0 - fy) * top + fy * bottom;

        // Elevación ajustada por el datum vertical si el modelo es ortométrico o requiere compensación
        Some(elev_raw + self.geoid_undulation_m)
    }

    /// Intersecta un rayo óptico con la superficie del DEM usando ray-marching y bisección continua.
    pub fn intersect_ray(
        &self,
        ray: &Ray3D,
        max_range_m: f64,
        bisection_steps: u32,
    ) -> Option<Vector3> {
        let dz = ray.direction.z;
        if dz >= -1e-4 {
            return None; // Rayo apunta al horizonte o hacia arriba
        }

        // Comprobación de seguridad: la cámara no debe empezar bajo el terreno
        if let Some(init_z) = self.sample_elevation_bilinear(ray.origin.x, ray.origin.y) {
            if ray.origin.z <= init_z {
                return None; // Datum mismatch o colisión inicial
            }
        }

        let off_nadir_rad = (-dz).clamp(0.0, 1.0).acos();
        let sin_off_nadir = off_nadir_rad.sin().max(0.15);

        // Paso adaptativo acotado
        let step_s = ((self.resolution() * 0.5) / sin_off_nadir).clamp(1.0, 15.0);

        let mut s_prev = 0.0;
        let mut s = step_s;
        let mut crossed = false;

        while s <= max_range_m {
            let p = ray.at(s);
            let dem_z = match self.sample_elevation_bilinear(p.x, p.y) {
                Some(z) => z,
                None => break, // Salió de la cobertura del DEM
            };

            if p.z <= dem_z {
                crossed = true;
                break;
            }

            s_prev = s;
            s += step_s;
        }

        if !crossed {
            return None;
        }

        // Bisección continua para precisión submétrica en el intervalo [s_prev, s]
        let mut s_low = s_prev;
        let mut s_high = s;

        for _ in 0..bisection_steps {
            let s_mid = 0.5 * (s_low + s_high);
            let p_mid = ray.at(s_mid);
            let dem_z_mid = match self.sample_elevation_bilinear(p_mid.x, p_mid.y) {
                Some(z) => z,
                None => break,
            };

            if p_mid.z <= dem_z_mid {
                s_high = s_mid;
            } else {
                s_low = s_mid;
            }
        }

        let s_star = 0.5 * (s_low + s_high);
        Some(ray.at(s_star))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dem_sample_bilinear() {
        // Cuadrícula 2x2: esquinas con elevaciones 10.0, 20.0, 30.0, 40.0
        let data = vec![10.0, 20.0, 30.0, 40.0];
        let dem = DemWindow::new(2, 2, 0.0, 10.0, 10.0, 10.0, data, None, 0.0);

        // En el centro exacto (x=5.0, y=5.0), la media debe ser 25.0
        let z = dem.sample_elevation_bilinear(5.0, 5.0).unwrap();
        assert!((z - 25.0).abs() < 1e-4);
    }

    #[test]
    fn test_dem_ray_intersection() {
        let w = 50;
        let h = 50;
        let res = 2.0;
        let mut data = vec![100.0f32; w * h];
        // Terreno inclinado en X: 100 + 0.1 * X
        for r in 0..h {
            for c in 0..w {
                data[r * w + c] = 100.0 + 0.1 * (c as f32 * res as f32);
            }
        }

        let dem = DemWindow::new(w, h, 0.0, 100.0, res, res, data, None, 0.0);

        // Cámara a nadir en x=40, y=50, z=300
        let cam_pos = Vector3::new(40.0, 50.0, 300.0);
        let ray = Ray3D {
            origin: cam_pos,
            direction: Vector3::new(0.0, 0.0, -1.0),
        };

        let impact = dem.intersect_ray(&ray, 2000.0, 16).unwrap();
        // Cota esperada en x=40: 100 + 0.1 * 40 = 104.0
        assert!((impact.x - 40.0).abs() < 1e-3);
        assert!((impact.y - 50.0).abs() < 1e-3);
        assert!((impact.z - 104.0).abs() < 0.1);
    }
}
