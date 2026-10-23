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

/// Metadatos de telemetría, gimbal y cámara extraídos del archivo JPEG (XMP / EXIF / SOF).
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
    // Telemetría de telémetro láser (LRF)
    pub lrf_target_distance: Option<f64>,
    pub lrf_target_lon: Option<f64>,
    pub lrf_target_lat: Option<f64>,
    pub lrf_target_abs_alt: Option<f64>,
    // Parámetros de cámara dinámicos
    pub model: Option<String>,
    pub image_width: Option<u32>,
    pub image_height: Option<u32>,
    pub focal_length_mm: Option<f64>,
    pub focal_length_35mm: Option<f64>,
    pub calibrated_focal_length: Option<f64>,
    pub calibrated_cx: Option<f64>,
    pub calibrated_cy: Option<f64>,
    pub dewarp_data: Option<String>,
    #[serde(default)]
    pub dewarp_flag: Option<String>,
}

/// Extrae las dimensiones reales de un JPEG escaneando los marcadores SOF (SOF0, SOF1, SOF2).
pub fn extract_jpeg_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    if bytes.len() < 4 || bytes[0] != 0xFF || bytes[1] != 0xD8 {
        return None;
    }
    let mut i = 2;
    while i + 8 < bytes.len() {
        if bytes[i] != 0xFF {
            i += 1;
            continue;
        }
        let marker = bytes[i + 1];
        // 0xC0 (SOF0: Baseline), 0xC1 (SOF1: Extended), 0xC2 (SOF2: Progressive)
        if marker == 0xC0 || marker == 0xC1 || marker == 0xC2 {
            let h = u16::from_be_bytes([bytes[i + 5], bytes[i + 6]]) as u32;
            let w = u16::from_be_bytes([bytes[i + 7], bytes[i + 8]]) as u32;
            if w > 0 && h > 0 {
                return Some((w, h));
            }
        }
        if marker == 0xDA || marker == 0xD9 {
            break;
        }
        let len = u16::from_be_bytes([bytes[i + 2], bytes[i + 3]]) as usize;
        if len < 2 {
            break;
        }
        i += 2 + len;
    }
    None
}

/// Extrae la distancia focal física (mm) y equivalente 35mm directamente de los metadatos binarios EXIF APP1.
pub fn extract_exif_focal_lengths(bytes: &[u8]) -> (Option<f64>, Option<f64>) {
    let limit = bytes.len().min(131072);
    let mut i = 0;
    while i + 14 < limit {
        if bytes[i] == 0xFF && bytes[i + 1] == 0xE1 {
            let seg_len = u16::from_be_bytes([bytes[i + 2], bytes[i + 3]]) as usize;
            if seg_len >= 8 && i + 10 <= bytes.len() && &bytes[i + 4..i + 10] == b"Exif\0\0" {
                let tiff_start = i + 10;
                let tiff_end = (i + 2 + seg_len).min(bytes.len());
                if tiff_start + 8 <= tiff_end {
                    let tiff = &bytes[tiff_start..tiff_end];
                    return parse_tiff_exif_focal_lengths(tiff).unwrap_or((None, None));
                }
            }
            if seg_len >= 2 {
                i += 2 + seg_len;
                continue;
            }
        }
        i += 1;
    }
    (None, None)
}

fn parse_tiff_exif_focal_lengths(tiff: &[u8]) -> Option<(Option<f64>, Option<f64>)> {
    if tiff.len() < 8 {
        return None;
    }
    let is_le = if &tiff[0..2] == b"II" {
        true
    } else if &tiff[0..2] == b"MM" {
        false
    } else {
        return None;
    };

    let read_u16 = |offset: usize| -> Option<u16> {
        if offset + 2 <= tiff.len() {
            if is_le {
                Some(u16::from_le_bytes([tiff[offset], tiff[offset + 1]]))
            } else {
                Some(u16::from_be_bytes([tiff[offset], tiff[offset + 1]]))
            }
        } else {
            None
        }
    };

    let read_u32 = |offset: usize| -> Option<u32> {
        if offset + 4 <= tiff.len() {
            if is_le {
                Some(u32::from_le_bytes([
                    tiff[offset],
                    tiff[offset + 1],
                    tiff[offset + 2],
                    tiff[offset + 3],
                ]))
            } else {
                Some(u32::from_be_bytes([
                    tiff[offset],
                    tiff[offset + 1],
                    tiff[offset + 2],
                    tiff[offset + 3],
                ]))
            }
        } else {
            None
        }
    };

    let magic = read_u16(2)?;
    if magic != 42 {
        return None;
    }

    let ifd0_offset = read_u32(4)? as usize;
    if ifd0_offset >= tiff.len() {
        return None;
    }

    let ifd0_count = read_u16(ifd0_offset)? as usize;
    let mut subifd_offset: Option<usize> = None;

    for idx in 0..ifd0_count.min(256) {
        let entry = ifd0_offset + 2 + idx * 12;
        if entry + 12 > tiff.len() {
            break;
        }
        let tag = read_u16(entry)?;
        if tag == 0x8769 {
            // ExifSubIFD pointer
            subifd_offset = Some(read_u32(entry + 8)? as usize);
            break;
        }
    }

    let subifd = match subifd_offset {
        Some(off) if off < tiff.len() => off,
        _ => return None,
    };

    let subifd_count = read_u16(subifd)? as usize;
    let mut fl_mm = None;
    let mut fl_35 = None;

    for idx in 0..subifd_count.min(256) {
        let entry = subifd + 2 + idx * 12;
        if entry + 12 > tiff.len() {
            break;
        }
        let tag = read_u16(entry)?;
        let typ = read_u16(entry + 2)?;

        if tag == 0x920A {
            // FocalLength (tipo 5 = RATIONAL, dos u32 num/den en el offset apuntado)
            if typ == 5 {
                let val_offset = read_u32(entry + 8)? as usize;
                if val_offset + 8 <= tiff.len() {
                    let num = read_u32(val_offset)?;
                    let den = read_u32(val_offset + 4)?;
                    if den > 0 {
                        fl_mm = Some(num as f64 / den as f64);
                    }
                }
            }
        } else if tag == 0xA405 {
            // FocalLengthIn35mmFilm (tipo 3 = SHORT, u16 en los primeros 2 bytes de value)
            if typ == 3 {
                let val = read_u16(entry + 8)?;
                if val > 0 {
                    fl_35 = Some(val as f64);
                }
            }
        }
    }

    Some((fl_mm, fl_35))
}

/// Extrae los metadatos XMP de un archivo JPEG de dron DJI leyendo únicamente la cabecera APP1.
pub fn extract_dji_xmp_from_file(path: impl AsRef<Path>) -> Result<DjiMetadata, XmpError> {
    let path = path.as_ref();
    let filename = path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();

    let mut file = File::open(path)?;
    // Los segmentos APP1 de XMP residen habitualmente en los primeros 128-256 KB
    let mut buffer = vec![0u8; 262144];
    let n = file.read(&mut buffer)?;
    buffer.truncate(n);

    extract_dji_xmp_from_bytes(&buffer, filename)
}

/// Extrae los metadatos XMP desde un buffer de bytes de la cabecera del archivo.
pub fn extract_dji_xmp_from_bytes(bytes: &[u8], filename: String) -> Result<DjiMetadata, XmpError> {
    // 1. Extraer dimensiones reales decodificando la cabecera JPEG SOF
    let dimensions = extract_jpeg_dimensions(bytes);

    // 2. Buscar la firma del espacio de nombres XMP de DJI
    let dji_marker = b"drone-dji:";
    let dji_pos = bytes
        .windows(dji_marker.len())
        .position(|window| window == dji_marker)
        .ok_or(XmpError::XmpNotFound)?;

    // El bloque XMP puede comenzar antes de drone-dji (p. ej. con tiff: o exif: dentro de rdf:Description)
    let start_idx = bytes[..dji_pos]
        .windows(b"<x:xmpmeta".len())
        .rposition(|w| w == b"<x:xmpmeta")
        .or_else(|| {
            bytes[..dji_pos]
                .windows(b"<rdf:Description".len())
                .rposition(|w| w == b"<rdf:Description")
        })
        .unwrap_or_else(|| dji_pos.saturating_sub(2048));

    let end_idx = (dji_pos + 16384).min(bytes.len());
    let xmp_chunk = String::from_utf8_lossy(&bytes[start_idx..end_idx]);

    let parse_attr_generic = |prefix: &str, attr_name: &str| -> Option<String> {
        let pattern1 = format!("{}:{}=", prefix, attr_name);
        if let Some(pos) = xmp_chunk.find(&pattern1) {
            let rest = &xmp_chunk[pos + pattern1.len()..];
            let quote = rest.chars().next()?;
            if quote == '"' || quote == '\'' {
                let rest_val = &rest[1..];
                if let Some(end_pos) = rest_val.find(quote) {
                    return Some(rest_val[..end_pos].trim().to_string());
                }
            }
        }
        None
    };

    let parse_float_attr = |attr_name: &str| -> Option<f64> {
        parse_attr_generic("drone-dji", attr_name).and_then(|s| s.parse::<f64>().ok())
    };

    let parse_string_attr =
        |attr_name: &str| -> Option<String> { parse_attr_generic("drone-dji", attr_name) };

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

    // Telemetría LRF (Laser Rangefinder)
    let lrf_target_distance = parse_float_attr("LRFTargetDistance");
    let lrf_target_lon = parse_float_attr("LRFTargetLon");
    let lrf_target_lat = parse_float_attr("LRFTargetLat");
    let lrf_target_abs_alt = parse_float_attr("LRFTargetAbsAlt");

    // Parámetros de cámara y calibración
    let model = parse_attr_generic("tiff", "Model")
        .or_else(|| parse_attr_generic("drone-dji", "CameraModel"))
        .or_else(|| parse_attr_generic("drone-dji", "DroneModel"));

    let calibrated_focal_length = parse_float_attr("CalibratedFocalLength");
    let calibrated_cx = parse_float_attr("CalibratedOpticalCenterX");
    let calibrated_cy = parse_float_attr("CalibratedOpticalCenterY");
    let dewarp_data = parse_string_attr("DewarpData");
    let dewarp_flag = parse_string_attr("DewarpFlag");

    let parse_num_or_frac = |s: String| -> Option<f64> {
        if let Some((num_s, den_s)) = s.split_once('/') {
            let num: f64 = num_s.trim().parse().ok()?;
            let den: f64 = den_s.trim().parse().ok()?;
            if den != 0.0 {
                return Some(num / den);
            }
        }
        s.parse::<f64>().ok()
    };

    let focal_length_mm_xmp = parse_attr_generic("exif", "FocalLength")
        .or_else(|| parse_attr_generic("drone-dji", "FocalLength"))
        .and_then(parse_num_or_frac);

    let focal_length_35mm_xmp = parse_attr_generic("exif", "FocalLengthIn35mmFilm")
        .or_else(|| parse_attr_generic("drone-dji", "FocalLengthIn35mmFormat"))
        .or_else(|| parse_attr_generic("exif", "FocalLengthIn35mmFormat"))
        .and_then(parse_num_or_frac);

    // Fallback a parseo binario EXIF si no están en los atributos XMP
    let (exif_fl_mm, exif_fl_35) = extract_exif_focal_lengths(bytes);
    let focal_length_mm = focal_length_mm_xmp.or(exif_fl_mm);
    let focal_length_35mm = focal_length_35mm_xmp.or(exif_fl_35);

    // También buscar en los bytes crudos si hay etiquetas EXIF directas de modelo o focal 35mm
    let raw_text = String::from_utf8_lossy(&bytes[..bytes.len().min(65536)]);
    let model_final = model.or_else(|| {
        for candidate in &["ZenmuseP1", "FC3588", "ZH20T", "M30T", "ZenmuseH20"] {
            if raw_text.contains(candidate) {
                return Some(candidate.to_string());
            }
        }
        None
    });

    // Dimensiones de imagen: prioridad a las obtenidas de SOF; fallback a XMP/EXIF
    let (image_width, image_height) = match dimensions {
        Some((w, h)) => (Some(w), Some(h)),
        None => {
            let w = parse_attr_generic("exif", "PixelXDimension")
                .or_else(|| parse_attr_generic("tiff", "ImageWidth"))
                .and_then(|s| s.parse::<u32>().ok());
            let h = parse_attr_generic("exif", "PixelYDimension")
                .or_else(|| parse_attr_generic("tiff", "ImageHeight"))
                .and_then(|s| s.parse::<u32>().ok());
            (w, h)
        }
    };

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
        lrf_target_distance,
        lrf_target_lon,
        lrf_target_lat,
        lrf_target_abs_alt,
        model: model_final,
        image_width,
        image_height,
        focal_length_mm,
        focal_length_35mm,
        calibrated_focal_length,
        calibrated_cx,
        calibrated_cy,
        dewarp_data,
        dewarp_flag,
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

    #[test]
    fn test_extract_camera_metadata_and_sof() {
        let mut jpeg_bytes = Vec::new();
        // SOI
        jpeg_bytes.extend_from_slice(&[0xFF, 0xD8]);
        // SOF0 (Baseline, 8192 x 5460)
        // Marker 0xFF, 0xC0, len=17 (0x00, 0x11), precision=8, H=5460 (0x15, 0x54), W=8192 (0x20, 0x00), comp=3
        jpeg_bytes.extend_from_slice(&[
            0xFF, 0xC0, 0x00, 0x11, 0x08, 0x15, 0x54, 0x20, 0x00, 0x03, 0x01, 0x11, 0x00, 0x02,
            0x11, 0x01, 0x03, 0x11, 0x01,
        ]);

        let sample_xmp = br#"
            <x:xmpmeta xmlns:x="adobe:ns:meta/">
                <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
                    <rdf:Description rdf:about=""
                        xmlns:tiff="http://ns.adobe.com/tiff/1.0/"
                        xmlns:exif="http://ns.adobe.com/exif/1.0/"
                        xmlns:drone-dji="http://www.dji.com/drone/1.0/"
                        tiff:Model="ZenmuseP1"
                        exif:FocalLength="35/1"
                        exif:FocalLengthIn35mmFilm="35"
                        drone-dji:GpsLatitude="+6.37"
                        drone-dji:GpsLongitude="-75.56"
                        drone-dji:CalibratedFocalLength="7986.32"
                        drone-dji:CalibratedOpticalCenterX="4096.0"
                        drone-dji:CalibratedOpticalCenterY="2730.0"/>
                </rdf:RDF>
            </x:xmpmeta>
        "#;
        jpeg_bytes.extend_from_slice(sample_xmp);

        let dims = extract_jpeg_dimensions(&jpeg_bytes);
        assert_eq!(dims, Some((8192, 5460)));

        let meta = extract_dji_xmp_from_bytes(&jpeg_bytes, "P1_0001.JPG".into()).unwrap();
        assert_eq!(meta.image_width, Some(8192));
        assert_eq!(meta.image_height, Some(5460));
        assert_eq!(meta.model.as_deref(), Some("ZenmuseP1"));
        assert_eq!(meta.focal_length_mm, Some(35.0));
        assert_eq!(meta.focal_length_35mm, Some(35.0));
        assert_eq!(meta.calibrated_focal_length, Some(7986.32));
        assert_eq!(meta.calibrated_cx, Some(4096.0));
        assert_eq!(meta.calibrated_cy, Some(2730.0));
    }

    #[test]
    fn test_extract_lrf_target_telemetry() {
        let sample_xmp = br#"
            <x:xmpmeta xmlns:x="adobe:ns:meta/">
                <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
                    <rdf:Description rdf:about=""
                        xmlns:drone-dji="http://www.dji.com/drone/1.0/"
                        drone-dji:GpsLatitude="+6.1234"
                        drone-dji:GpsLongitude="-75.5678"
                        drone-dji:AbsoluteAltitude="2200.0"
                        drone-dji:RelativeAltitude="100.0"
                        drone-dji:GimbalPitchDegree="-85.0"
                        drone-dji:LRFTargetDistance="45.6"
                        drone-dji:LRFTargetLon="-75.5680"
                        drone-dji:LRFTargetLat="+6.1236"
                        drone-dji:LRFTargetAbsAlt="2154.4"/>
                </rdf:RDF>
            </x:xmpmeta>
        "#;

        let meta = extract_dji_xmp_from_bytes(sample_xmp, "LRF_0001.JPG".into()).unwrap();
        assert_eq!(meta.lrf_target_distance, Some(45.6));
        assert!((meta.lrf_target_lon.unwrap() - (-75.5680)).abs() < 1e-6);
        assert!((meta.lrf_target_lat.unwrap() - 6.1236).abs() < 1e-6);
        assert!((meta.lrf_target_abs_alt.unwrap() - 2154.4).abs() < 1e-3);
    }

    #[test]
    fn test_extract_exif_focal_lengths_binary() {
        let mut jpeg = Vec::new();
        // SOI
        jpeg.extend_from_slice(&[0xFF, 0xD8]);

        // APP1 EXIF
        // TIFF content:
        // offset 0: "II" (little-endian)
        // offset 2: 42 (0x002A)
        // offset 4: offset to IFD0 = 8 (0x00000008)
        // offset 8: IFD0 num_entries = 1 (0x0001)
        // offset 10: tag 0x8769, typ 4 (LONG), count 1, val_offset = 26 (0x0000001A)
        // offset 22: next IFD = 0 (0x00000000)
        // offset 26: ExifSubIFD num_entries = 2 (0x0002)
        // Entry 1: tag 0x920A (FocalLength), typ 5 (RATIONAL), count 1, val_offset = 54 (0x00000036)
        // Entry 2: tag 0xA405 (FocalLengthIn35mmFilm), typ 3 (SHORT), count 1, val = 24 (0x0018, 0x0000)
        // offset 50: next IFD = 0
        // offset 54: Rational data: num = 135 (0x00000087), den = 10 (0x0000000A) => 13.5 mm
        let mut tiff = Vec::new();
        tiff.extend_from_slice(b"II");
        tiff.extend_from_slice(&42u16.to_le_bytes());
        tiff.extend_from_slice(&8u32.to_le_bytes()); // IFD0 at offset 8

        // IFD0
        tiff.extend_from_slice(&1u16.to_le_bytes()); // 1 entry
        tiff.extend_from_slice(&0x8769u16.to_le_bytes()); // tag ExifSubIFD
        tiff.extend_from_slice(&4u16.to_le_bytes()); // typ LONG
        tiff.extend_from_slice(&1u32.to_le_bytes()); // count 1
        tiff.extend_from_slice(&26u32.to_le_bytes()); // offset 26
        tiff.extend_from_slice(&0u32.to_le_bytes()); // next IFD = 0

        // ExifSubIFD at offset 26
        tiff.extend_from_slice(&2u16.to_le_bytes()); // 2 entries
                                                     // Entry 1: FocalLength
        tiff.extend_from_slice(&0x920Au16.to_le_bytes());
        tiff.extend_from_slice(&5u16.to_le_bytes()); // RATIONAL
        tiff.extend_from_slice(&1u32.to_le_bytes());
        tiff.extend_from_slice(&54u32.to_le_bytes()); // offset 54

        // Entry 2: FocalLengthIn35mmFilm
        tiff.extend_from_slice(&0xA405u16.to_le_bytes());
        tiff.extend_from_slice(&3u16.to_le_bytes()); // SHORT
        tiff.extend_from_slice(&1u32.to_le_bytes());
        tiff.extend_from_slice(&24u16.to_le_bytes()); // value 24 mm
        tiff.extend_from_slice(&[0u8, 0u8]); // padding to 4 bytes
        tiff.extend_from_slice(&0u32.to_le_bytes()); // next IFD = 0

        // offset 54: Rational 135 / 10
        tiff.extend_from_slice(&135u32.to_le_bytes());
        tiff.extend_from_slice(&10u32.to_le_bytes());

        // APP1 marker and payload: len = 2 + 6 (Exif\0\0) + tiff.len()
        let app1_len = (2 + 6 + tiff.len()) as u16;
        jpeg.extend_from_slice(&[0xFF, 0xE1]);
        jpeg.extend_from_slice(&app1_len.to_be_bytes());
        jpeg.extend_from_slice(b"Exif\0\0");
        jpeg.extend_from_slice(&tiff);

        let (fl_mm, fl_35) = extract_exif_focal_lengths(&jpeg);
        assert_eq!(fl_mm, Some(13.5));
        assert_eq!(fl_35, Some(24.0));

        // Test fallback in extract_dji_xmp_from_bytes
        let sample_xmp = br#"
            <x:xmpmeta xmlns:x="adobe:ns:meta/">
                <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
                    <rdf:Description rdf:about=""
                        xmlns:drone-dji="http://www.dji.com/drone/1.0/"
                        drone-dji:GpsLatitude="+6.1234"
                        drone-dji:GpsLongitude="-75.5678"/>
                </rdf:RDF>
            </x:xmpmeta>
        "#;
        jpeg.extend_from_slice(sample_xmp);

        let meta = extract_dji_xmp_from_bytes(&jpeg, "TEST_EXIF.JPG".into()).unwrap();
        assert_eq!(meta.focal_length_mm, Some(13.5));
        assert_eq!(meta.focal_length_35mm, Some(24.0));
    }

    #[test]
    fn test_extract_dewarp_flag_semantics() {
        // Caso 1: DewarpFlag="0" (cruda, sin corrección de firmware)
        let xmp_flag_0 = br#"
            <x:xmpmeta xmlns:x="adobe:ns:meta/">
                <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
                    <rdf:Description rdf:about=""
                        xmlns:drone-dji="http://www.dji.com/drone/1.0/"
                        drone-dji:GpsLatitude="+6.0"
                        drone-dji:GpsLongitude="-75.0"
                        drone-dji:DewarpFlag="0"
                        drone-dji:DewarpData="2023-01-01;760.0,760.0,0.0,0.0,-0.1,0.02,0.0,0.0,0.0"/>
                </rdf:RDF>
            </x:xmpmeta>
        "#;
        let m0 = extract_dji_xmp_from_bytes(xmp_flag_0, "RAW.JPG".into()).unwrap();
        assert_eq!(m0.dewarp_flag.as_deref(), Some("0"));
        assert!(m0.dewarp_data.is_some());

        // Caso 2: DewarpFlag="1" (ya desdistorsionada por firmware de cámara)
        let xmp_flag_1 = br#"
            <x:xmpmeta xmlns:x="adobe:ns:meta/">
                <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
                    <rdf:Description rdf:about=""
                        xmlns:drone-dji="http://www.dji.com/drone/1.0/"
                        drone-dji:GpsLatitude="+6.0"
                        drone-dji:GpsLongitude="-75.0"
                        drone-dji:DewarpFlag="1"/>
                </rdf:RDF>
            </x:xmpmeta>
        "#;
        let m1 = extract_dji_xmp_from_bytes(xmp_flag_1, "CORRECTED.JPG".into()).unwrap();
        assert_eq!(m1.dewarp_flag.as_deref(), Some("1"));

        // Caso 3: Ausencia de etiqueta DewarpFlag
        let xmp_none = br#"
            <x:xmpmeta xmlns:x="adobe:ns:meta/">
                <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
                    <rdf:Description rdf:about=""
                        xmlns:drone-dji="http://www.dji.com/drone/1.0/"
                        drone-dji:GpsLatitude="+6.0"
                        drone-dji:GpsLongitude="-75.0"/>
                </rdf:RDF>
            </x:xmpmeta>
        "#;
        let m_none = extract_dji_xmp_from_bytes(xmp_none, "NONE.JPG".into()).unwrap();
        assert_eq!(m_none.dewarp_flag, None);
    }
}
