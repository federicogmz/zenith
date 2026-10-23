//! `zenith-core`: Modelos geométricos fundamentales, álgebra 3D y proyecciones de cámara para Zenith.

pub mod projection;
pub use projection::{Ellipsoid, TransverseMercator};

pub mod hull;
pub use hull::{ConvexHull2D, Point2D};

use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul, Sub};

/// Vector tridimensional en punto flotante de 64 bits.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vector3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vector3 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    #[inline]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[inline]
    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    #[inline]
    pub fn norm_squared(self) -> f64 {
        self.dot(self)
    }

    #[inline]
    pub fn norm(self) -> f64 {
        self.norm_squared().sqrt()
    }

    #[inline]
    pub fn normalize(self) -> Self {
        let n = self.norm();
        if n > 1e-12 {
            Self {
                x: self.x / n,
                y: self.y / n,
                z: self.z / n,
            }
        } else {
            Self::ZERO
        }
    }
}

impl Add for Vector3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Sub for Vector3 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Mul<f64> for Vector3 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl Mul<Vector3> for f64 {
    type Output = Vector3;
    #[inline]
    fn mul(self, rhs: Vector3) -> Vector3 {
        rhs * self
    }
}

/// Rayo 3D definido por origen y dirección unitaria.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Ray3D {
    pub origin: Vector3,
    pub direction: Vector3,
}

impl Ray3D {
    #[inline]
    pub fn at(&self, s: f64) -> Vector3 {
        self.origin + self.direction * s
    }
}

/// Parámetros de distorsión radial y tangencial (modelo Brown-Conrady).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct DewarpParams {
    pub k1: f64,
    pub k2: f64,
    pub p1: f64,
    pub p2: f64,
    pub k3: f64,
}

/// Parámetros intrínsecos de cámara pinhole y desdistorsión Brown-Conrady.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CameraIntrinsics {
    pub width: u32,
    pub height: u32,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dewarp: Option<DewarpParams>,
}

impl CameraIntrinsics {
    /// Crea intrínsecos pinhole estándar asumiendo punto principal en el centro.
    pub fn new(width: u32, height: u32, fx: f64, fy: f64) -> Self {
        Self {
            width,
            height,
            fx,
            fy,
            cx: width as f64 / 2.0,
            cy: height as f64 / 2.0,
            dewarp: None,
        }
    }

    /// Intrínsecos nominales para el sensor térmico DJI Mavic 3 Enterprise Thermal (M3T).
    /// Sensor VOx 640x512, pitch 12 µm, lente 9.1 mm => fx = fy = 640 * (9.1 / 7.68) ≈ 758.333 px.
    pub fn dji_m3t_thermal() -> Self {
        let f = 640.0 * (9.1 / 7.68);
        Self::new(640, 512, f, f)
    }

    /// Intrínsecos nominales para el sensor gran angular visual (RGB) del DJI Mavic 3 Enterprise / Thermal (M3T).
    /// Sensor 1/2" CMOS 4000x3000 (12 MP), pixel pitch 1.6 µm, lente 4.4 mm (24 mm equiv.)
    /// => fx = fy = 4.4 / 0.0016 = 2750.0 px, cx = 2000.0, cy = 1500.0.
    pub fn dji_m3t_rgb_wide() -> Self {
        Self {
            width: 4000,
            height: 3000,
            fx: 2750.0,
            fy: 2750.0,
            cx: 2000.0,
            cy: 1500.0,
            dewarp: None,
        }
    }

    /// Escala los parámetros intrínsecos a una nueva resolución de sensor/imagen.
    pub fn scaled(&self, target_width: u32, target_height: u32) -> Self {
        if target_width == self.width && target_height == self.height {
            return *self;
        }
        let sx = target_width as f64 / self.width.max(1) as f64;
        let sy = target_height as f64 / self.height.max(1) as f64;
        Self {
            width: target_width,
            height: target_height,
            fx: self.fx * sx,
            fy: self.fy * sy,
            cx: (self.cx + 0.5) * sx - 0.5,
            cy: (self.cy + 0.5) * sy - 0.5,
            dewarp: self.dewarp,
        }
    }

    /// Parsea la calibración de distorsión DJI de la etiqueta XMP `DewarpData`.
    /// Formato típico: `"fecha;fx,fy,dx,dy,k1,k2,p1,p2,k3"`.
    pub fn with_dji_dewarp(&self, dewarp_data: &str) -> Self {
        let payload = if let Some(idx) = dewarp_data.find(';') {
            &dewarp_data[idx + 1..]
        } else {
            dewarp_data
        };

        let parts: Vec<&str> = payload.split(',').map(|s| s.trim()).collect();
        if parts.len() < 9 {
            return *self;
        }

        let nums: Vec<f64> = parts.iter().filter_map(|s| s.parse::<f64>().ok()).collect();
        if nums.len() < 9 {
            return *self;
        }

        let fx = nums[0];
        let fy = nums[1];
        let dx = nums[2];
        let dy = nums[3];
        let k1 = nums[4];
        let k2 = nums[5];
        let p1 = nums[6];
        let p2 = nums[7];
        let k3 = nums[8];

        let has_distortion = k1.abs() > 1e-9
            || k2.abs() > 1e-9
            || k3.abs() > 1e-9
            || p1.abs() > 1e-9
            || p2.abs() > 1e-9;
        if !has_distortion && fx <= 0.0 {
            return *self;
        }

        let w = self.width as f64;
        let h = self.height as f64;

        let cx = if dx.abs() > w * 0.25 {
            dx
        } else {
            (w - 1.0) * 0.5 + dx
        };

        let cy = if dy.abs() > h * 0.25 {
            dy
        } else {
            (h - 1.0) * 0.5 + dy
        };

        let mut res = *self;
        if fx > 10.0 {
            res.fx = fx;
        }
        if fy > 10.0 {
            res.fy = fy;
        }
        res.cx = cx;
        res.cy = cy;
        res.dewarp = Some(DewarpParams { k1, k2, p1, p2, k3 });
        res
    }

    /// Deriva intrínsecos de cámara de forma dinámica para cualquier sensor usando la mejor fuente disponible:
    /// 1. Calibración directa del fabricante en XMP (CalibratedFocalLength, CalibratedOpticalCenter).
    /// 2. Focal equivalente 35 mm proyectada sobre la diagonal física real del sensor:
    ///    fx = fy = f_35 * sqrt(W^2 + H^2) / 43.2666153
    /// 3. Modelo conocido si no hay f35 (P1, M3T, etc.)
    /// 4. Fallback angular estándar: fx = fy = W * 0.70.
    #[allow(clippy::too_many_arguments)]
    pub fn from_telemetry(
        width: u32,
        height: u32,
        model: Option<&str>,
        calibrated_focal_length: Option<f64>,
        calibrated_cx: Option<f64>,
        calibrated_cy: Option<f64>,
        focal_length_35mm: Option<f64>,
        focal_length_mm: Option<f64>,
    ) -> Self {
        let w = width as f64;
        let h = height as f64;
        let sanitize_c = |c_opt: Option<f64>, dim: f64| -> f64 {
            match c_opt {
                Some(c) if c > 0.0 && c <= 1.0 => c * dim, // Coordenadas normalizadas 0..1
                Some(c) if c > 1.0 && c < dim * 2.0 => c,  // Coordenadas absolutas en píxeles
                _ => dim / 2.0,
            }
        };
        let cx = sanitize_c(calibrated_cx, w);
        let cy = sanitize_c(calibrated_cy, h);

        // 1. Calibración directa del dron (DJI P1, M3E, etc.)
        if let Some(cf) = calibrated_focal_length {
            if cf > 10.0 {
                return Self {
                    width,
                    height,
                    fx: cf,
                    fy: cf,
                    cx,
                    cy,
                    dewarp: None,
                };
            }
        }

        // 2. Sensores térmicos radiométricos estándar 640x512 y 1280x1024 (pitch 12 µm -> sensor 7.68 x 6.144 mm)
        // Evaluado ANTES de la f35 genérica para evitar distorsiones por redondeos enteros de DJI
        if (width == 640 && height == 512) || (width == 1280 && height == 1024) {
            let m_lower = model.unwrap_or("").to_lowercase();
            let fx_native = if m_lower.contains("zh20") || m_lower.contains("h20") {
                let f_mm = focal_length_mm.unwrap_or(13.5);
                640.0 * (f_mm / 7.68)
            } else {
                let f_mm = focal_length_mm.unwrap_or(9.1);
                640.0 * (f_mm / 7.68)
            };
            let native = Self {
                width: 640,
                height: 512,
                fx: fx_native,
                fy: fx_native,
                cx: 320.0,
                cy: 256.0,
                dewarp: None,
            };
            let scaled = native.scaled(width, height);
            return Self {
                cx: if calibrated_cx.is_some() {
                    cx
                } else {
                    scaled.cx
                },
                cy: if calibrated_cy.is_some() {
                    cy
                } else {
                    scaled.cy
                },
                ..scaled
            };
        }

        // 3. Focal física en mm o geometría conocida del sensor si se especifica el modelo
        if let Some(m) = model {
            let m_lower = m.to_lowercase();
            if m_lower.contains("p1") || m_lower.contains("zenmusep1") {
                // Full-frame 35.9 x 24.0 mm, lente nominal 35.0 mm si no viene especificada
                let f_mm = focal_length_mm.or(focal_length_35mm).unwrap_or(35.0);
                let fx = w * (f_mm / 35.9);
                return Self {
                    width,
                    height,
                    fx,
                    fy: fx,
                    cx,
                    cy,
                    dewarp: None,
                };
            } else if m_lower.contains("m3t")
                || m_lower.contains("m3e")
                || m_lower.contains("fc3588")
            {
                // 1/2" CMOS 6.4 x 4.8 mm, lente nominal 4.4 mm (24 mm equiv.)
                let f_mm = focal_length_mm.unwrap_or(4.4);
                let fx = w * (f_mm / 6.4);
                return Self {
                    width,
                    height,
                    fx,
                    fy: fx,
                    cx,
                    cy,
                    dewarp: None,
                };
            } else if m_lower.contains("zh20") || m_lower.contains("h20") {
                let diag_px = (w * w + h * h).sqrt();
                let f35 = focal_length_35mm.unwrap_or(24.0);
                let fx = f35 * diag_px / 43.2666153;
                return Self {
                    width,
                    height,
                    fx,
                    fy: fx,
                    cx,
                    cy,
                    dewarp: None,
                };
            }
        }

        // 4. Equivalencia de 35 mm usando la diagonal real del sensor (exacto para 3:2, 4:3, 16:9)
        // Diagonal de sensor 35mm (36x24 mm) = sqrt(36^2 + 24^2) = 43.2666153 mm
        if let Some(f35) = focal_length_35mm {
            if f35 > 1.0 {
                let diag_px = (w * w + h * h).sqrt();
                let f_px = f35 * diag_px / 43.2666153;
                return Self {
                    width,
                    height,
                    fx: f_px,
                    fy: f_px,
                    cx,
                    cy,
                    dewarp: None,
                };
            }
        }

        // 5. Fallback si es térmica estándar 640x512 o 1280x1024
        if (width == 640 && height == 512) || (width == 1280 && height == 1024) {
            return Self::dji_m3t_thermal().scaled(width, height);
        }

        // 6. Fallback gran angular estándar: fx = fy = W * 0.70
        let fx = w * 0.70;
        Self {
            width,
            height,
            fx,
            fy: fx,
            cx,
            cy,
            dewarp: None,
        }
    }
}

/// Pose extrínseca de la cámara en coordenadas proyectadas locales (ENU).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraPose {
    pub position: Vector3,
    pub yaw_deg: f64,
    pub pitch_deg: f64,
    pub roll_deg: f64,
}

/// Base ortonormal de la cámara en sistema de coordenadas ENU (East, North, Up).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraBasis {
    pub x_c_prime: Vector3,
    pub u_top_prime: Vector3,
    pub z_c: Vector3,
}

impl CameraBasis {
    /// Construye la base ortonormal a partir de los ángulos de gimbal de DJI.
    /// - `yaw`: grados horario desde el Norte (0° = North, 90° = East).
    /// - `pitch`: grados desde la horizontal (0° = horizontal, -90° = nadir hacia abajo).
    /// - `roll`: rotación sobre el eje óptico z_c.
    pub fn from_gimbal(yaw_deg: f64, pitch_deg: f64, roll_deg: f64) -> Self {
        let psi = yaw_deg.to_radians();
        let theta = pitch_deg.to_radians();
        let phi = roll_deg.to_radians();

        let (sin_psi, cos_psi) = psi.sin_cos();
        let (sin_theta, cos_theta) = theta.sin_cos();
        let (sin_phi, cos_phi) = phi.sin_cos();

        // 1. Eje óptico hacia adelante
        let z_c = Vector3::new(cos_theta * sin_psi, cos_theta * cos_psi, sin_theta).normalize();

        // 2. Vector horizontal a la derecha (sin roll)
        let x_c = Vector3::new(cos_psi, -sin_psi, 0.0);

        // 3. Vector hacia arriba en la imagen (sin roll)
        let u_top = Vector3::new(-sin_theta * sin_psi, -sin_theta * cos_psi, cos_theta);

        // 4. Aplicar rotación de Roll sobre z_c
        let x_c_prime = (x_c * cos_phi + u_top * sin_phi).normalize();
        let u_top_prime = (x_c * (-sin_phi) + u_top * cos_phi).normalize();

        Self {
            x_c_prime,
            u_top_prime,
            z_c,
        }
    }

    /// Convierte coordenadas de píxel (u, v) en un rayo unitario 3D en coordenadas ENU.
    /// Si `intrinsics.dewarp` está configurado, invierte la distorsión Brown-Conrady por punto fijo.
    pub fn pixel_to_ray(
        &self,
        u: f64,
        v: f64,
        camera_pos: Vector3,
        intrinsics: &CameraIntrinsics,
    ) -> Ray3D {
        let (dx, dy) = if let Some(ref d) = intrinsics.dewarp {
            let xd = (u - intrinsics.cx) / intrinsics.fx;
            let yd = (v - intrinsics.cy) / intrinsics.fy;
            let mut xn = xd;
            let mut yn = yd;
            for _ in 0..8 {
                let r2 = xn * xn + yn * yn;
                let rad = 1.0 + d.k1 * r2 + d.k2 * r2 * r2 + d.k3 * r2 * r2 * r2;
                let delta_x = 2.0 * d.p1 * xn * yn + d.p2 * (r2 + 2.0 * xn * xn);
                let delta_y = d.p1 * (r2 + 2.0 * yn * yn) + 2.0 * d.p2 * xn * yn;
                if rad.abs() < 1e-12 {
                    break;
                }
                let xn_next = (xd - delta_x) / rad;
                let yn_next = (yd - delta_y) / rad;
                let diff = (xn_next - xn).abs() + (yn_next - yn).abs();
                xn = xn_next;
                yn = yn_next;
                if diff < 1e-10 {
                    break;
                }
            }
            (xn, yn)
        } else {
            (
                (u - intrinsics.cx) / intrinsics.fx,
                (v - intrinsics.cy) / intrinsics.fy,
            )
        };

        let dir = self.z_c + self.x_c_prime * dx - self.u_top_prime * dy;
        Ray3D {
            origin: camera_pos,
            direction: dir.normalize(),
        }
    }

    /// Proyección de un punto 3D del terreno (X, Y, Z) a coordenadas de píxel (u, v).
    /// Devuelve None si el punto está detrás del plano focal de la cámara.
    pub fn world_to_pixel(
        &self,
        point: Vector3,
        camera_pos: Vector3,
        intrinsics: &CameraIntrinsics,
    ) -> Option<(f64, f64)> {
        let v = point - camera_pos;
        let depth = v.dot(self.z_c);

        if depth <= 1e-4 {
            return None; // Detrás de la cámara
        }

        let proj_x = v.dot(self.x_c_prime);
        let proj_y = v.dot(self.u_top_prime);

        let xn = proj_x / depth;
        let yn = -proj_y / depth;

        if let Some(ref d) = intrinsics.dewarp {
            let r2 = xn * xn + yn * yn;
            let rad = 1.0 + d.k1 * r2 + d.k2 * r2 * r2 + d.k3 * r2 * r2 * r2;
            let xd = xn * rad + 2.0 * d.p1 * xn * yn + d.p2 * (r2 + 2.0 * xn * xn);
            let yd = yn * rad + d.p1 * (r2 + 2.0 * yn * yn) + 2.0 * d.p2 * xn * yn;
            Some((
                intrinsics.cx + xd * intrinsics.fx,
                intrinsics.cy + yd * intrinsics.fy,
            ))
        } else {
            let u = intrinsics.cx + xn * intrinsics.fx;
            let v_px = intrinsics.cy + yn * intrinsics.fy;
            Some((u, v_px))
        }
    }

    /// Alias canónico de proyección de punto 3D a píxel de sensor.
    #[inline]
    pub fn project_point(
        &self,
        point: Vector3,
        camera_pos: Vector3,
        intrinsics: &CameraIntrinsics,
    ) -> Option<(f64, f64)> {
        self.world_to_pixel(point, camera_pos, intrinsics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera_basis_nadir() {
        let basis = CameraBasis::from_gimbal(0.0, -90.0, 0.0);
        assert!((basis.z_c.z - (-1.0)).abs() < 1e-6);
        assert!((basis.x_c_prime.x - 1.0).abs() < 1e-6);
        assert!((basis.u_top_prime.y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_round_trip_pixel_world() {
        let intrinsics = CameraIntrinsics::dji_m3t_thermal();
        let cam_pos = Vector3::new(1000.0, 2000.0, 450.0);
        let ground_pt = Vector3::new(1040.0, 2070.0, 120.0);

        let v = ground_pt - cam_pos;
        let dist = v.norm();
        let dir = v.normalize();

        let yaw = dir.x.atan2(dir.y).to_degrees();
        let pitch = dir.z.asin().to_degrees();

        let basis = CameraBasis::from_gimbal(yaw, pitch, 0.0);
        let (u, v_px) = basis
            .world_to_pixel(ground_pt, cam_pos, &intrinsics)
            .unwrap();

        let ray = basis.pixel_to_ray(u, v_px, cam_pos, &intrinsics);
        let reconstructed_pt = ray.at(dist);

        assert!((reconstructed_pt.x - ground_pt.x).abs() < 1e-3);
        assert!((reconstructed_pt.y - ground_pt.y).abs() < 1e-3);
        assert!((reconstructed_pt.z - ground_pt.z).abs() < 1e-3);
    }

    #[test]
    fn test_from_telemetry_p1_and_m3t() {
        // 1. Zenmuse P1 con calibración directa
        let intr_p1_calib = CameraIntrinsics::from_telemetry(
            8192,
            5460,
            Some("ZenmuseP1"),
            Some(7986.32),
            Some(4096.0),
            Some(2730.0),
            Some(35.0),
            Some(35.0),
        );
        assert_eq!(intr_p1_calib.width, 8192);
        assert_eq!(intr_p1_calib.height, 5460);
        assert!((intr_p1_calib.fx - 7986.32).abs() < 1e-3);
        assert_eq!(intr_p1_calib.cx, 4096.0);
        assert_eq!(intr_p1_calib.cy, 2730.0);

        // 2. Zenmuse P1 sin calibración, solo f35=35mm (fórmula por diagonal)
        let intr_p1_diag =
            CameraIntrinsics::from_telemetry(8192, 5460, None, None, None, None, Some(35.0), None);
        // fx ≈ 35 * sqrt(8192^2 + 5460^2) / 43.2666153 ≈ 7963.85 px
        assert!((intr_p1_diag.fx - 7963.85).abs() < 1.0);

        // 3. Sensor 4:3 (M3E/M3T) 4000x3000, f35=24mm
        let intr_m3e_diag =
            CameraIntrinsics::from_telemetry(4000, 3000, None, None, None, None, Some(24.0), None);
        // fx ≈ 24 * 5000 / 43.2666153 ≈ 2773.5 px
        assert!((intr_m3e_diag.fx - 2773.5).abs() < 1.0);
    }

    #[test]
    fn test_regression_m3t() {
        let nominal_m3t = CameraIntrinsics::dji_m3t_rgb_wide();
        let telemetry_m3t = CameraIntrinsics::from_telemetry(
            4000,
            3000,
            Some("Mavic 3T"),
            None,
            None,
            None,
            Some(24.0),
            Some(4.4),
        );
        // Telemetría con f35=24mm reproduce la focal nominal dentro de < 1.0%
        let rel_err = (telemetry_m3t.fx - nominal_m3t.fx).abs() / nominal_m3t.fx;
        assert!(
            rel_err < 0.01,
            "Error relativo focal M3T {} excede 1%",
            rel_err
        );
        assert_eq!(telemetry_m3t.cx, nominal_m3t.cx);
        assert_eq!(telemetry_m3t.cy, nominal_m3t.cy);

        // Caso térmico 640x512
        let thermal_intr =
            CameraIntrinsics::from_telemetry(640, 512, None, None, None, None, None, None);
        let expected_thermal = CameraIntrinsics::dji_m3t_thermal();
        assert_eq!(thermal_intr.fx, expected_thermal.fx);
        assert_eq!(thermal_intr.cx, expected_thermal.cx);
        assert_eq!(thermal_intr.cy, expected_thermal.cy);
    }

    #[test]
    fn test_optical_center_normalization() {
        // Coordenadas normalizadas 0.5, 0.5 deben convertirse a píxeles absolutos
        let intr = CameraIntrinsics::from_telemetry(
            4000,
            3000,
            None,
            Some(2800.0),
            Some(0.499),
            Some(0.501),
            None,
            None,
        );
        assert!((intr.cx - 0.499 * 4000.0).abs() < 1e-3);
        assert!((intr.cy - 0.501 * 3000.0).abs() < 1e-3);
    }

    #[test]
    fn test_multisensor_intrinsics_telemetry() {
        // 1. ZH20T Térmico (640x512, 13.5 mm) -> fx = 1125.0
        let intr_zh20_t = CameraIntrinsics::from_telemetry(
            640,
            512,
            Some("Zenmuse H20T"),
            None,
            None,
            None,
            Some(58.0), // simulando f35 reportada por DJI que NO debe anular la focal física
            Some(13.5),
        );
        assert_eq!(intr_zh20_t.width, 640);
        assert_eq!(intr_zh20_t.height, 512);
        assert_eq!(intr_zh20_t.fx, 1125.0);
        assert_eq!(intr_zh20_t.fy, 1125.0);

        // 2. M3T Térmico (640x512, 9.1 mm) -> fx ≈ 758.333
        let intr_m3t_t = CameraIntrinsics::from_telemetry(
            640,
            512,
            Some("Mavic 3T"),
            None,
            None,
            None,
            Some(40.0),
            Some(9.1),
        );
        assert!((intr_m3t_t.fx - 758.333).abs() < 0.05);
        assert!((intr_m3t_t.fy - 758.333).abs() < 0.05);

        // 3. ZH20T RGB Visual (4056x3040, f35=24mm) -> fx ≈ 2811.7
        let intr_zh20_rgb = CameraIntrinsics::from_telemetry(
            4056,
            3040,
            Some("ZH20T"),
            None,
            None,
            None,
            Some(24.0),
            Some(4.5),
        );
        assert!((intr_zh20_rgb.fx - 2811.7).abs() < 1.0);
        assert!((intr_zh20_rgb.fy - 2811.7).abs() < 1.0);

        // 4. Zenmuse P1 RGB Visual (8192x5460, 35mm sobre sensor 35.9mm) -> fx ≈ 7986.6
        let intr_p1 = CameraIntrinsics::from_telemetry(
            8192,
            5460,
            Some("ZenmuseP1"),
            None,
            None,
            None,
            Some(35.0),
            Some(35.0),
        );
        assert!((intr_p1.fx - 7986.6).abs() < 1.0);
        assert!((intr_p1.fy - 7986.6).abs() < 1.0);
    }

    #[test]
    fn test_thermal_scaled_and_high_res_telemetry() {
        // 1. M3T nativo 640x512 vs 1280x1024
        let m3t_native = CameraIntrinsics::from_telemetry(
            640,
            512,
            Some("Mavic 3T"),
            None,
            None,
            None,
            None,
            Some(9.1),
        );
        let m3t_1280 = CameraIntrinsics::from_telemetry(
            1280,
            1024,
            Some("Mavic 3T"),
            None,
            None,
            None,
            None,
            Some(9.1),
        );
        let m3t_scaled = m3t_native.scaled(1280, 1024);

        assert_eq!(m3t_1280.width, 1280);
        assert_eq!(m3t_1280.height, 1024);
        assert!((m3t_1280.fx - (640.0 * (9.1 / 7.68) * 2.0)).abs() < 1e-6);
        assert!((m3t_1280.fx - 1516.6666666666667).abs() < 1e-4);
        assert!((m3t_1280.fx - m3t_scaled.fx).abs() < 1e-6);
        assert!((m3t_1280.fy - m3t_scaled.fy).abs() < 1e-6);

        // 2. ZH20T nativo 640x512 vs 1280x1024
        let zh20_native = CameraIntrinsics::from_telemetry(
            640,
            512,
            Some("ZH20T"),
            None,
            None,
            None,
            None,
            Some(13.5),
        );
        let zh20_1280 = CameraIntrinsics::from_telemetry(
            1280,
            1024,
            Some("ZH20T"),
            None,
            None,
            None,
            None,
            Some(13.5),
        );
        let zh20_scaled = zh20_native.scaled(1280, 1024);

        assert_eq!(zh20_1280.width, 1280);
        assert_eq!(zh20_1280.height, 1024);
        assert_eq!(zh20_1280.fx, 2250.0);
        assert_eq!(zh20_1280.fy, 2250.0);
        assert_eq!(zh20_scaled.fx, 2250.0);
        assert_eq!(zh20_scaled.fy, 2250.0);
    }

    #[test]
    fn test_round_trip_with_dewarp_params() {
        let mut intrinsics = CameraIntrinsics::new(4000, 3000, 2750.0, 2750.0);
        intrinsics.dewarp = Some(DewarpParams {
            k1: -0.15,
            k2: 0.08,
            p1: 0.001,
            p2: -0.0005,
            k3: -0.01,
        });

        let cam_pos = Vector3::new(5000000.0, 2000000.0, 300.0);
        let basis = CameraBasis::from_gimbal(45.0, -80.0, 5.0);

        // Probar varios puntos en el sensor (centro, bordes, esquinas extremas)
        let test_pixels = [
            (2000.0, 1500.0), // centro
            (500.0, 600.0),   // cuadrante superior izq
            (3500.0, 2400.0), // cuadrante inferior der
            (100.0, 2900.0),  // borde
            (0.0, 0.0),       // esquina superior izq
            (3999.0, 0.0),    // esquina superior der
            (0.0, 2999.0),    // esquina inferior izq
            (3999.0, 2999.0), // esquina inferior der
        ];

        for &(u, v) in &test_pixels {
            let ray = basis.pixel_to_ray(u, v, cam_pos, &intrinsics);
            let ground_pt = ray.at(250.0); // 250m de profundidad

            let (u_reproj, v_reproj) = basis
                .project_point(ground_pt, cam_pos, &intrinsics)
                .expect("project_point must succeed");

            assert!(
                (u_reproj - u).abs() < 1e-3,
                "u round-trip error: reproj={}, orig={}, diff={}",
                u_reproj,
                u,
                (u_reproj - u).abs()
            );
            assert!(
                (v_reproj - v).abs() < 1e-3,
                "v round-trip error: reproj={}, orig={}, diff={}",
                v_reproj,
                v,
                (v_reproj - v).abs()
            );
        }
    }

    #[test]
    fn test_scaled_intrinsics_preserves_dimensionless_dewarp() {
        let mut base = CameraIntrinsics::new(640, 512, 758.33, 758.33);
        base.dewarp = Some(DewarpParams {
            k1: -0.12,
            k2: 0.05,
            p1: 0.002,
            p2: -0.001,
            k3: -0.005,
        });
        let scaled = base.scaled(1280, 1024);
        assert_eq!(scaled.width, 1280);
        assert_eq!(scaled.height, 1024);
        assert!((scaled.fx - 1516.66).abs() < 1e-2);
        assert!((scaled.fy - 1516.66).abs() < 1e-2);
        // Coeficientes adimensionales no deben ser alterados por el cambio de resolución:
        let d = scaled.dewarp.expect("Dewarp debe preservarse");
        assert_eq!(d.k1, -0.12);
        assert_eq!(d.k2, 0.05);
        assert_eq!(d.p1, 0.002);
        assert_eq!(d.p2, -0.001);
        assert_eq!(d.k3, -0.005);
    }

    #[test]
    fn test_with_dji_dewarp_parsing() {
        let base = CameraIntrinsics::new(640, 512, 758.33, 758.33);
        // Formato típico DJI: "2023-01-01;fx,fy,dx,dy,k1,k2,p1,p2,k3"
        let dewarp_str = "2023-01-01;760.5,761.2,1.5,-2.0,-0.10,0.04,0.001,-0.0005,-0.002";
        let intr = base.with_dji_dewarp(dewarp_str);
        assert_eq!(intr.fx, 760.5);
        assert_eq!(intr.fy, 761.2);
        // cx = (640 - 1)/2 + 1.5 = 319.5 + 1.5 = 321.0
        assert_eq!(intr.cx, 321.0);
        // cy = (512 - 1)/2 - 2.0 = 255.5 - 2.0 = 253.5
        assert_eq!(intr.cy, 253.5);
        let d = intr.dewarp.expect("DewarpParams debe existir");
        assert_eq!(d.k1, -0.10);
        assert_eq!(d.k2, 0.04);
        assert_eq!(d.p1, 0.001);
        assert_eq!(d.p2, -0.0005);
        assert_eq!(d.k3, -0.002);
    }
}
