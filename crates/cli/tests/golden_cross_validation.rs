//! Test de validación cruzada dorada (Golden Cross-Validation) entre Python (RAPTOR) y Rust (Zenith).
//! Comprueba que las transformaciones geométricas, bases ENU, trazado de rayos y cálculo de intersección
//! coincidan estrictamente hasta nivel de milímetros.

use zenith_core::{CameraBasis, CameraIntrinsics, Vector3};
use zenith_dem::DemWindow;

#[test]
fn test_golden_cross_validation_with_raptor_python() {
    let w = 640;
    let h = 512;
    let fx = 758.333333;
    let fy = 758.333333;
    let yaw = 35.0;
    let pitch = -65.0;
    let roll = 5.0;
    let cam_pos = Vector3::new(500050.0, 699950.0, 450.0);
    let plane_z = 150.0;

    let intrinsics = CameraIntrinsics::new(w, h, fx, fy);
    let basis = CameraBasis::from_gimbal(yaw, pitch, roll);

    // 1. Píxel central (320, 256)
    let ray_center = basis.pixel_to_ray(320.0, 256.0, cam_pos, &intrinsics);

    // Valores dorados de RAPTOR (Python):
    // ray: [0.24240388, 0.34618861, -0.90630779]
    // PT: 500130.2389, 700064.5931, 150.0000, s: 331.0134
    assert!(
        (ray_center.direction.x - 0.24240388).abs() < 1e-6,
        "Error en componente X del rayo: {}",
        ray_center.direction.x
    );
    assert!(
        (ray_center.direction.y - 0.34618861).abs() < 1e-6,
        "Error en componente Y del rayo: {}",
        ray_center.direction.y
    );
    assert!(
        (ray_center.direction.z - (-0.90630779)).abs() < 1e-6,
        "Error en componente Z del rayo: {}",
        ray_center.direction.z
    );

    // Intersección analítica con plano Z = 150.0
    let s = (plane_z - cam_pos.z) / ray_center.direction.z;
    assert!((s - 331.0134).abs() < 1e-3, "Error en slant range s: {}", s);

    let impact = ray_center.at(s);
    assert!(
        (impact.x - 500130.2389).abs() < 1e-3,
        "Error milimétrico en X de impacto: {}",
        impact.x
    );
    assert!(
        (impact.y - 700064.5931).abs() < 1e-3,
        "Error milimétrico en Y de impacto: {}",
        impact.y
    );
    assert!(
        (impact.z - 150.0).abs() < 1e-3,
        "Error milimétrico en Z de impacto: {}",
        impact.z
    );

    // 2. Píxel superior izquierdo (100, 100)
    // Python ray: [0.07948711, 0.61770808, -0.78238004]
    let ray_tl = basis.pixel_to_ray(100.0, 100.0, cam_pos, &intrinsics);
    assert!((ray_tl.direction.x - 0.07948711).abs() < 1e-6);
    assert!((ray_tl.direction.y - 0.61770808).abs() < 1e-6);
    assert!((ray_tl.direction.z - (-0.78238004)).abs() < 1e-6);

    // 3. Píxel inferior derecho (540, 412)
    // Python ray: [0.37729323, 0.03464185, -0.92544571]
    let ray_br = basis.pixel_to_ray(540.0, 412.0, cam_pos, &intrinsics);
    assert!((ray_br.direction.x - 0.37729323).abs() < 1e-6);
    assert!((ray_br.direction.y - 0.03464185).abs() < 1e-6);
    assert!((ray_br.direction.z - (-0.92544571)).abs() < 1e-6);
}

#[test]
fn test_dem_bisection_precision() {
    let w = 100;
    let h = 100;
    let res = 2.0;
    let x0 = 500000.0;
    let y0 = 700000.0;

    let mut elev = vec![0.0f32; w * h];
    for r in 0..h {
        for c in 0..w {
            elev[r * w + c] = 100.0 + 0.1 * (c as f32 * res as f32);
        }
    }

    let dem = DemWindow::new(w, h, x0, y0, res, res, elev, Some(-9999.0), 0.0);
    let cam_pos = Vector3::new(x0 + 50.0, y0 - 50.0, 300.0);
    let ray = zenith_core::Ray3D {
        origin: cam_pos,
        direction: Vector3::new(0.0, 0.0, -1.0),
    };

    let impact = dem
        .intersect_ray(&ray, 1000.0, 16)
        .expect("Debe intersectar el terreno");
    let expected_z = 100.0 + 0.1 * 50.0; // 105.0 m
    assert!((impact.x - (x0 + 50.0)).abs() < 1e-3);
    assert!((impact.y - (y0 - 50.0)).abs() < 1e-3);
    assert!((impact.z - expected_z).abs() < 0.05);
}
