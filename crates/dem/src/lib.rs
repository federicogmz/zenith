//! `zenith-dem`: Lector en memoria e interpolación bilineal de modelos de elevación (DEM/DSM)
//! con algoritmo de intersección ray-marching y bisección continua.

use zenith_core::{Ray3D, Vector3};

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

        // Elevación ajustada por el datum vertical si el modelo es ortométrico
        Some(elev_raw)
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
