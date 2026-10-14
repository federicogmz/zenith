//! Extractor nativo en Rust puro de metadatos XMP de telemetría DJI (APP1 segment).
//! Extrae coordenadas RTK centimétricas, altitud absoluta/relativa y ángulos de gimbal
//! en microsegundos sin requerir exiftool ni dependencias C++.

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum XmpError {
    #[error("Error de E/S leyendo JPEG: {0}")]
    Io(#[from] std::io::Error),
    #[error("Segmento XMP de DJI no encontrado en los primeros 128 KB")]
    XmpNotFound,
    #[error("Campo requerido faltante o inválido: {0}")]
    MissingField(String),
}

/// Metadatos de telemetría y gimbal extraídos directamente del XMP de DJI.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DjiMetadata {
    pub filename: String,
    pub lon: f64,
    pub lat: f64,
    pub abs_alt: f64,
    pub rel_alt: f64,
    pub yaw: f64,
    pub pitch: f64,
    pub roll: f64,
    pub rtk_std_lon: Option<f64>,
    pub rtk_std_lat: Option<f64>,
    pub rtk_flag: Option<String>,
}

/// Extrae los metadatos XMP de un archivo JPEG de dron DJI leyendo únicamente la cabecera APP1.
pub fn extract_dji_xmp_from_file(path: impl AsRef<Path>) -> Result<DjiMetadata, XmpError> {
    let path = path.as_ref();
    let filename = path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();

    let mut file = File::open(path)?;
    // Los segmentos APP1 de XMP residen siempre en los primeros 128 KB
    let mut buffer = vec![0u8; 131072];
    let n = file.read(&mut buffer)?;
    buffer.truncate(n);

    extract_dji_xmp_from_bytes(&buffer, filename)
}

/// Extrae los metadatos XMP desde un buffer de bytes de la cabecera del archivo.
pub fn extract_dji_xmp_from_bytes(bytes: &[u8], filename: String) -> Result<DjiMetadata, XmpError> {
    // Buscar la firma del espacio de nombres XMP de DJI
    let marker = b"drone-dji:";
    let start_idx = bytes
        .windows(marker.len())
        .position(|window| window == marker)
        .ok_or(XmpError::XmpNotFound)?;

    // Tomar un bloque representativo de hasta 8192 bytes a partir del primer marcador
    let end_idx = (start_idx + 8192).min(bytes.len());
    let xmp_chunk = String::from_utf8_lossy(&bytes[start_idx..end_idx]);

    let parse_float_attr = |attr_name: &str| -> Option<f64> {
        let pattern = format!("drone-dji:{}=\"", attr_name);
        if let Some(pos) = xmp_chunk.find(&pattern) {
            let val_start = pos + pattern.len();
            if let Some(val_end) = xmp_chunk[val_start..].find('"') {
                return xmp_chunk[val_start..val_start + val_end]
                    .parse::<f64>()
                    .ok();
            }
        }
        None
    };

    let parse_string_attr = |attr_name: &str| -> Option<String> {
        let pattern = format!("drone-dji:{}=\"", attr_name);
        if let Some(pos) = xmp_chunk.find(&pattern) {
            let val_start = pos + pattern.len();
            if let Some(val_end) = xmp_chunk[val_start..].find('"') {
                return Some(xmp_chunk[val_start..val_start + val_end].to_string());
            }
        }
        None
    };

    let lat = parse_float_attr("GpsLatitude")
        .ok_or_else(|| XmpError::MissingField("GpsLatitude".into()))?;
    let lon = parse_float_attr("GpsLongitude")
        .ok_or_else(|| XmpError::MissingField("GpsLongitude".into()))?;
    let abs_alt = parse_float_attr("AbsoluteAltitude").unwrap_or(0.0);
    let rel_alt = parse_float_attr("RelativeAltitude").unwrap_or(0.0);
    let yaw = parse_float_attr("GimbalYawDegree").unwrap_or(0.0);
    let pitch = parse_float_attr("GimbalPitchDegree").unwrap_or(-90.0);
    let roll = parse_float_attr("GimbalRollDegree").unwrap_or(0.0);

    let rtk_std_lon = parse_float_attr("RtkStdLon");
    let rtk_std_lat = parse_float_attr("RtkStdLat");
    let rtk_flag = parse_string_attr("RtkFlag");

    Ok(DjiMetadata {
        filename,
        lon,
        lat,
        abs_alt,
        rel_alt,
        yaw,
        pitch,
        roll,
        rtk_std_lon,
        rtk_std_lat,
        rtk_flag,
    })
}

/// Resultado del escaneo de un directorio de vuelo.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlightScanResult {
    pub frames: Vec<DjiMetadata>,
    pub errors: Vec<(PathBuf, String)>,
}

/// Escanea de forma tolerante un directorio de imágenes térmicas o visuales,
/// retornando la lista de metadatos válidos y los errores individuales encontrados.
pub fn scan_flight_directory(dir: impl AsRef<Path>) -> std::io::Result<FlightScanResult> {
    let mut valid = Vec::new();
    let mut errors = Vec::new();

    let mut entries = std::fs::read_dir(dir)?
        .filter_map(|res| res.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .map(|ext| {
                    let s = ext.to_string_lossy().to_lowercase();
                    s == "jpg" || s == "jpeg"
                })
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();

    entries.sort();

    for path in entries {
        match extract_dji_xmp_from_file(&path) {
            Ok(meta) => valid.push(meta),
            Err(e) => errors.push((path, e.to_string())),
        }
    }

    Ok(FlightScanResult {
        frames: valid,
        errors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_dji_xmp_synthetic() {
        let sample_xmp = br#"
            <?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
            <x:xmpmeta xmlns:x="adobe:ns:meta/">
                <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
                    <rdf:Description rdf:about=""
                        xmlns:drone-dji="http://www.dji.com/drone/1.0/"
                        drone-dji:GpsLatitude="+6.410397430"
                        drone-dji:GpsLongitude="-75.306236299"
                        drone-dji:AbsoluteAltitude="+2350.529"
                        drone-dji:RelativeAltitude="+488.719"
                        drone-dji:GimbalRollDegree="+0.00"
                        drone-dji:GimbalYawDegree="-67.40"
                        drone-dji:GimbalPitchDegree="-87.70"
                        drone-dji:RtkStdLon="0.01211"
                        drone-dji:RtkStdLat="0.01191"
                        drone-dji:RtkFlag="50"/>
                </rdf:RDF>
            </x:xmpmeta>
            <?xpacket end="w"?>
        "#;

        let meta = extract_dji_xmp_from_bytes(sample_xmp, "TEST_0001.JPG".into()).unwrap();
        assert_eq!(meta.filename, "TEST_0001.JPG");
        assert!((meta.lat - 6.410397430).abs() < 1e-7);
        assert!((meta.lon - (-75.306236299)).abs() < 1e-7);
        assert!((meta.abs_alt - 2350.529).abs() < 1e-3);
        assert!((meta.pitch - (-87.70)).abs() < 1e-3);
        assert_eq!(meta.rtk_flag.as_deref(), Some("50"));
    }
}
