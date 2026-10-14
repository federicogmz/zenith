//! `zenith-core`: Modelos geométricos fundamentales, álgebra 3D y proyecciones de cámara para Zenith.

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

/// Parámetros intrínsecos de cámara pinhole.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CameraIntrinsics {
    pub width: u32,
    pub height: u32,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
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
        }
    }

    /// Intrínsecos nominales para el sensor térmico DJI Mavic 3 Enterprise Thermal (M3T).
    /// Sensor VOx 640x512, pitch 12 µm, lente 9.1 mm => f ≈ 758.33 px.
    pub fn dji_m3t_thermal() -> Self {
        Self::new(640, 512, 758.33, 758.33)
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
    pub fn pixel_to_ray(
        &self,
        u: f64,
        v: f64,
        camera_pos: Vector3,
        intrinsics: &CameraIntrinsics,
    ) -> Ray3D {
        let dx = (u - intrinsics.cx) / intrinsics.fx;
        let dy = (v - intrinsics.cy) / intrinsics.fy;

        let dir = self.z_c + self.x_c_prime * dx - self.u_top_prime * dy;
        Ray3D {
            origin: camera_pos,
            direction: dir.normalize(),
        }
    }

    /// Proyección inversa: mapea un punto 3D del terreno (X, Y, Z) a coordenadas de píxel (u, v).
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

        let u = intrinsics.cx + (proj_x / depth) * intrinsics.fx;
        let v_px = intrinsics.cy - (proj_y / depth) * intrinsics.fy;

        Some((u, v_px))
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
}
