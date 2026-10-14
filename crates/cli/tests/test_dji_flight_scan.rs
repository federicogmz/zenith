//! Test de integración con la misión real del incendio El Viento, Barbosa (M3T).
//! Valida el extractor nativo de XMP DJI contra los 378 fotogramas térmicos reales.

use std::path::Path;
use zenith_io::scan_flight_directory;

#[test]
fn test_scan_real_mission_thermal_frames() {
    let mission_dir = Path::new(
        "/mnt/f/Operaciones_UAS2/20260711_Emergencia_Incendio_El_Viento_Barbosa_M3T/termica_mosaico",
    );

    // Solo corre si el volumen de la misión está montado en la máquina
    if !mission_dir.exists() {
        eprintln!("Aviso: Directorio de misión no montado. Saltando test de datos reales.");
        return;
    }

    let t0 = std::time::Instant::now();
    let scan = scan_flight_directory(mission_dir).expect("Error escaneando directorio de vuelo");
    let elapsed = t0.elapsed();

    println!(
        "Escaneo completado en {:?}: {} frames válidos, {} errores.",
        elapsed,
        scan.frames.len(),
        scan.errors.len()
    );

    assert_eq!(
        scan.errors.len(),
        0,
        "No debería haber errores de parsing XMP"
    );
    assert_eq!(
        scan.frames.len(),
        378,
        "Debe encontrar exactamente los 378 fotogramas térmicos"
    );

    // Verificar frame de muestra: DJI_20260711094702_0010_T.JPG
    let sample = scan
        .frames
        .iter()
        .find(|f| f.filename == "DJI_20260711094702_0010_T.JPG")
        .expect("Frame de muestra debe existir");

    assert!((sample.lat - 6.410397430).abs() < 1e-6);
    assert!((sample.lon - (-75.306236299)).abs() < 1e-6);
    assert!((sample.abs_alt - 2350.529).abs() < 1e-3);
    assert!((sample.pitch - (-87.70)).abs() < 1e-3);
    assert!((sample.yaw - (-67.40)).abs() < 1e-3);
    assert_eq!(sample.rtk_flag.as_deref(), Some("50"));
}

#[test]
fn test_projection_378_frames_against_oracle_csv() {
    use zenith_core::TransverseMercator;

    let oracle_path = Path::new(
        "/home/fdgmz/.gemini/antigravity-ide/brain/1166a368-93be-4f04-9265-5df7ca2c7a47/scratch/mission_379_oracle.csv",
    );

    if !oracle_path.exists() {
        eprintln!("Aviso: Tabla oráculo no encontrada. Saltando test de validación cruzada.");
        return;
    }

    let content = std::fs::read_to_string(oracle_path).expect("Error leyendo tabla oráculo");
    let tm = TransverseMercator::colombia_origen_nacional();

    let mut count = 0;
    let mut max_err_x = 0.0f64;
    let mut max_err_y = 0.0f64;

    for line in content.lines().skip(1) {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 11 {
            continue;
        }

        let lon: f64 = parts[1].parse().unwrap();
        let lat: f64 = parts[2].parse().unwrap();
        let expected_x: f64 = parts[8].parse().unwrap();
        let expected_y: f64 = parts[9].parse().unwrap();

        let (x, y, _gamma) = tm.forward(lon, lat);

        let err_x = (x - expected_x).abs();
        let err_y = (y - expected_y).abs();

        if err_x > max_err_x {
            max_err_x = err_x;
        }
        if err_y > max_err_y {
            max_err_y = err_y;
        }

        // Tolerancia submilimétrica estricta (1e-3 metros = 1 mm)
        assert!(
            err_x < 1e-3,
            "Error en X mayor a 1mm: {} m en frame {}",
            err_x,
            parts[0]
        );
        assert!(
            err_y < 1e-3,
            "Error en Y mayor a 1mm: {} m en frame {}",
            err_y,
            parts[0]
        );

        count += 1;
    }

    println!(
        "Validación cruzada de proyección PROJ completada para {} fotogramas:",
        count
    );
    println!("  Máximo error en Easting X: {:.6} mm", max_err_x * 1000.0);
    println!("  Máximo error en Northing Y: {:.6} mm", max_err_y * 1000.0);

    assert_eq!(count, 378);
    assert!(max_err_x < 0.0005); // Error menor a 0.5 mm
    assert!(max_err_y < 0.0005); // Error menor a 0.5 mm
}
