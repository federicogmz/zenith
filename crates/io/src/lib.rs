//! `zenith-io`: Parser de georreferenciación (`geo.txt`), metadatos de cámaras
//! y serialización del manifiesto de conformidad con el Contrato del Motor Fotogramétrico.

use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;
use zenith_core::{CameraPose, Vector3};

#[derive(Error, Debug)]
pub enum IoError {
    #[error("Error de entrada/salida: {0}")]
    Io(#[from] std::io::Error),
    #[error("Formato de geo.txt inválido en línea {0}: {1}")]
    InvalidGeoTxt(usize, String),
    #[error("Error serializando JSON: {0}")]
    Json(#[from] serde_json::Error),
}

/// Registro individual de una cámara en `geo.txt`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeoEntry {
    pub filename: String,
    pub lon: f64,
    pub lat: f64,
    pub alt: f64,
    pub yaw: f64,
    pub pitch: f64,
    pub roll: f64,
    pub h_acc: Option<f64>,
    pub v_acc: Option<f64>,
}

impl GeoEntry {
    pub fn to_camera_pose(&self, projected_xyz: Vector3) -> CameraPose {
        CameraPose {
            position: projected_xyz,
            yaw_deg: self.yaw,
            pitch_deg: self.pitch,
            roll_deg: self.roll,
        }
    }
}

/// Parsea el contenido de un archivo `geo.txt` estándar de fotogrametría.
pub fn parse_geo_txt(content: &str) -> Result<(String, Vec<GeoEntry>), IoError> {
    let mut lines = content.lines().filter(|l| !l.trim().is_empty());
    let crs = lines
        .next()
        .ok_or_else(|| IoError::InvalidGeoTxt(1, "Archivo geo.txt vacío".into()))?
        .trim()
        .to_string();

    let mut entries = Vec::new();

    for (idx, line) in lines.enumerate() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 7 {
            return Err(IoError::InvalidGeoTxt(
                idx + 2,
                format!(
                    "Se esperaban al menos 7 columnas, encontradas {}",
                    parts.len()
                ),
            ));
        }

        let filename = parts[0].to_string();
        let lon: f64 = parts[1]
            .parse()
            .map_err(|_| IoError::InvalidGeoTxt(idx + 2, "Longitud inválida".into()))?;
        let lat: f64 = parts[2]
            .parse()
            .map_err(|_| IoError::InvalidGeoTxt(idx + 2, "Latitud inválida".into()))?;
        let alt: f64 = parts[3]
            .parse()
            .map_err(|_| IoError::InvalidGeoTxt(idx + 2, "Altitud inválida".into()))?;
        let yaw: f64 = parts[4]
            .parse()
            .map_err(|_| IoError::InvalidGeoTxt(idx + 2, "Yaw inválido".into()))?;
        let pitch: f64 = parts[5]
            .parse()
            .map_err(|_| IoError::InvalidGeoTxt(idx + 2, "Pitch inválido".into()))?;
        let roll: f64 = parts[6]
            .parse()
            .map_err(|_| IoError::InvalidGeoTxt(idx + 2, "Roll inválido".into()))?;

        let h_acc = parts.get(7).and_then(|s| s.parse().ok());
        let v_acc = parts.get(8).and_then(|s| s.parse().ok());

        entries.push(GeoEntry {
            filename,
            lon,
            lat,
            alt,
            yaw,
            pitch,
            roll,
            h_acc,
            v_acc,
        });
    }

    Ok((crs, entries))
}

/// Manifiesto de ejecución emitido por Zenith según el Contrato Fotogramétrico.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineManifest {
    pub schema_version: String,
    pub engine_name: String,
    pub engine_version: String,
    pub tier: String,
    pub elapsed_seconds: f64,
    pub dem_used: bool,
    pub gsd_m: f64,
    pub resampling: String,
    pub products: Vec<String>,
}

impl EngineManifest {
    pub fn new_level1(
        elapsed_seconds: f64,
        dem_used: bool,
        gsd_m: f64,
        resampling: &str,
        products: Vec<String>,
    ) -> Self {
        Self {
            schema_version: "1.0".to_string(),
            engine_name: "Zenith".to_string(),
            engine_version: env!("CARGO_PKG_VERSION").to_string(),
            tier: "Level 1 Fast 2.5D".to_string(),
            elapsed_seconds,
            dem_used,
            gsd_m,
            resampling: resampling.to_string(),
            products,
        }
    }

    pub fn write_to_file(&self, path: impl AsRef<Path>) -> Result<(), IoError> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_geo_txt() {
        let sample = "EPSG:4326\nIMG_0001.tif -75.5892 6.1684 2150.4 34.2 -89.5 0.1 0.05 0.08\n";
        let (crs, entries) = parse_geo_txt(sample).unwrap();

        assert_eq!(crs, "EPSG:4326");
        assert_eq!(entries.len(), 1);
        let e = &entries[0];
        assert_eq!(e.filename, "IMG_0001.tif");
        assert!((e.lon - (-75.5892)).abs() < 1e-4);
        assert!((e.pitch - (-89.5)).abs() < 1e-4);
        assert_eq!(e.h_acc, Some(0.05));
    }
}
