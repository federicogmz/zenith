//! Proyección cartográfica Transverse Mercator de alta precisión (series de Krüger orden 6)
//! Implementación en Rust puro compatible al milímetro con PROJ (`etmerc`).
//! Soporta EPSG:9377 (MAGNA-SIRGAS 2018 / Origen-Nacional de Colombia) y UTM.

/// Parámetros de elipsoide de referencia.
#[derive(Debug, Clone, Copy)]
pub struct Ellipsoid {
    pub a: f64, // Semieje mayor en metros
    pub f: f64, // Achatamiento
}

impl Ellipsoid {
    /// Elipsoide GRS80 (usado en MAGNA-SIRGAS 2018 y WGS84 moderno).
    pub const GRS80: Self = Self {
        a: 6378137.0,
        f: 1.0 / 298.257222101,
    };

    /// Elipsoide WGS84 estándar.
    pub const WGS84: Self = Self {
        a: 6378137.0,
        f: 1.0 / 298.257223563,
    };
}

/// Parámetros de proyección Transverse Mercator.
#[derive(Debug, Clone, Copy)]
pub struct TransverseMercator {
    pub ellipsoid: Ellipsoid,
    pub lat_origin_rad: f64,
    pub lon_origin_rad: f64,
    pub scale_factor: f64,
    pub false_easting: f64,
    pub false_northing: f64,
    // Coeficientes precalculados de Krüger (n=6)
    a_radius: f64,
    alpha: [f64; 6],
    m0: f64,
}

impl TransverseMercator {
    /// Inicializa la proyección Transverse Mercator con coeficientes precalculados.
    pub fn new(
        ellipsoid: Ellipsoid,
        lat_origin_deg: f64,
        lon_origin_deg: f64,
        scale_factor: f64,
        false_easting: f64,
        false_northing: f64,
    ) -> Self {
        let a = ellipsoid.a;
        let f = ellipsoid.f;
        let n = f / (2.0 - f);
        let n2 = n * n;
        let n3 = n2 * n;
        let n4 = n3 * n;
        let n5 = n4 * n;
        let n6 = n5 * n;

        // Coeficientes directos de Krüger orden 6 (igual a PROJ etmerc)
        let alpha1 = 0.5 * n - (2.0 / 3.0) * n2 + (5.0 / 16.0) * n3 + (41.0 / 180.0) * n4
            - (127.0 / 288.0) * n5
            + (7891.0 / 37800.0) * n6;
        let alpha2 = (13.0 / 48.0) * n2 - 0.6 * n3 + (557.0 / 1440.0) * n4 + (281.0 / 630.0) * n5
            - (1983433.0 / 1935360.0) * n6;
        let alpha3 = (61.0 / 240.0) * n3 - (103.0 / 140.0) * n4
            + (15061.0 / 26880.0) * n5
            + (167603.0 / 181440.0) * n6;
        let alpha4 =
            (49561.0 / 161280.0) * n4 - (179.0 / 168.0) * n5 + (6601661.0 / 7257600.0) * n6;
        let alpha5 = (34729.0 / 80640.0) * n5 - (3418889.0 / 1995840.0) * n6;
        let alpha6 = (212378941.0 / 319334400.0) * n6;

        let alpha = [alpha1, alpha2, alpha3, alpha4, alpha5, alpha6];

        let a_radius = (a / (1.0 + n)) * (1.0 + 0.25 * n2 + (1.0 / 64.0) * n4 + (1.0 / 256.0) * n6);

        let lat_origin_rad = lat_origin_deg.to_radians();
        let lon_origin_rad = lon_origin_deg.to_radians();

        let mut tm = Self {
            ellipsoid,
            lat_origin_rad,
            lon_origin_rad,
            scale_factor,
            false_easting,
            false_northing,
            a_radius,
            alpha,
            m0: 0.0,
        };

        // Calcular M0 (distancia meridiana al origen de latitud)
        let m0 = tm.calculate_meridian_distance(lat_origin_rad);
        tm.m0 = m0;
        tm
    }

    /// Proyección oficial de Colombia EPSG:9377 (MAGNA-SIRGAS 2018 / Origen-Nacional).
    /// Parámetros oficiales IGAC:
    /// Lat 4°N, Lon 73°W, k0 = 0.9992, Falso Este 5,000,000 m, Falso Norte 2,000,000 m.
    pub fn colombia_origen_nacional() -> Self {
        Self::new(
            Ellipsoid::GRS80,
            4.0,
            -73.0,
            0.9992,
            5_000_000.0,
            2_000_000.0,
        )
    }

    /// Inicializa la proyección a partir de un código EPSG soportado.
    pub fn from_epsg(epsg: u32) -> Result<Self, String> {
        match epsg {
            9377 => Ok(Self::colombia_origen_nacional()),
            // WGS 84 / UTM zone 1N..60N
            32601..=32660 => {
                let zone = (epsg - 32600) as f64;
                Ok(Self::new(
                    Ellipsoid::WGS84,
                    0.0,
                    (zone - 1.0) * 6.0 - 177.0,
                    0.9996,
                    500_000.0,
                    0.0,
                ))
            }
            // WGS 84 / UTM zone 1S..60S
            32701..=32760 => {
                let zone = (epsg - 32700) as f64;
                Ok(Self::new(
                    Ellipsoid::WGS84,
                    0.0,
                    (zone - 1.0) * 6.0 - 177.0,
                    0.9996,
                    500_000.0,
                    10_000_000.0,
                ))
            }
            // ETRS89 / UTM zone 28N..38N (Europa)
            25828..=25838 => {
                let zone = (epsg - 25800) as f64;
                Ok(Self::new(
                    Ellipsoid::GRS80,
                    0.0,
                    (zone - 1.0) * 6.0 - 177.0,
                    0.9996,
                    500_000.0,
                    0.0,
                ))
            }
            // SIRGAS 2000 / UTM zone 11N..22N (América del Sur / Caribe Norte)
            31965..=31976 => {
                let zone = (epsg - 31954) as f64;
                Ok(Self::new(
                    Ellipsoid::GRS80,
                    0.0,
                    (zone - 1.0) * 6.0 - 177.0,
                    0.9996,
                    500_000.0,
                    0.0,
                ))
            }
            // SIRGAS 2000 / UTM zone 17S..25S (América del Sur Sur)
            31977..=31985 => {
                let zone = (epsg - 31960) as f64;
                Ok(Self::new(
                    Ellipsoid::GRS80,
                    0.0,
                    (zone - 1.0) * 6.0 - 177.0,
                    0.9996,
                    500_000.0,
                    10_000_000.0,
                ))
            }
            // NAD83 / UTM zone 1N..23N (Norteamérica)
            26901..=26923 => {
                let zone = (epsg - 26900) as f64;
                Ok(Self::new(
                    Ellipsoid::GRS80,
                    0.0,
                    (zone - 1.0) * 6.0 - 177.0,
                    0.9996,
                    500_000.0,
                    0.0,
                ))
            }
            // MAGNA-SIRGAS Colombia zonas históricas (3114: Far West, 3115: West, 3116: Bogota, 3117: East Central, 3118: East)
            3114..=3118 => {
                let idx = (epsg - 3114) as f64;
                Ok(Self::new(
                    Ellipsoid::GRS80,
                    4.59620041666667,
                    -80.0775079166667 + 3.0 * idx,
                    1.0,
                    1_000_000.0,
                    1_000_000.0,
                ))
            }
            _ => Err(format!(
                "Código EPSG:{} no soportado por el motor TransverseMercator interno de Zenith. Use UTM WGS84, ETRS89, SIRGAS2000, NAD83 o EPSG:9377.",
                epsg
            )),
        }
    }

    /// Deriva la zona UTM estándar WGS 84 (EPSG:32601..32660 o 32701..32760) a partir de longitud y latitud en grados.
    pub fn utm_from_lon_lat(lon_deg: f64, lat_deg: f64) -> Result<(Self, u32), String> {
        if !(-80.0..=84.0).contains(&lat_deg) {
            return Err(format!(
                "Latitud {:.4}° fuera del rango estándar UTM (-80° a +84°)",
                lat_deg
            ));
        }
        if !(-180.0..=180.0).contains(&lon_deg) {
            return Err(format!(
                "Longitud {:.4}° fuera del rango estándar (-180° a +180°)",
                lon_deg
            ));
        }
        let zone = ((((lon_deg + 180.0) / 6.0).floor() as i32) + 1).clamp(1, 60) as u32;
        let epsg = if lat_deg >= 0.0 {
            32600 + zone
        } else {
            32700 + zone
        };
        Ok((Self::from_epsg(epsg)?, epsg))
    }

    /// Resuelve una especificación de CRS textual ("auto", "utm", "EPSG:32618", "9377") al objeto `TransverseMercator`
    /// y al identificador canónico "EPSG:XXXX".
    pub fn resolve_crs(
        crs_spec: &str,
        mean_lon: f64,
        mean_lat: f64,
    ) -> Result<(Self, String), String> {
        let trimmed = crs_spec.trim();
        if trimmed.eq_ignore_ascii_case("auto") || trimmed.eq_ignore_ascii_case("utm") {
            let (tm, epsg) = Self::utm_from_lon_lat(mean_lon, mean_lat)?;
            return Ok((tm, format!("EPSG:{}", epsg)));
        }

        let code_str = if let Some(stripped) = trimmed.strip_prefix("EPSG:") {
            stripped
        } else if let Some(stripped) = trimmed.strip_prefix("epsg:") {
            stripped
        } else {
            trimmed
        };

        let epsg: u32 = code_str.parse().map_err(|_| {
            format!(
                "Especificación de CRS no reconocida: '{}'. Use 'auto', 'utm' o 'EPSG:XXXX'",
                crs_spec
            )
        })?;

        let tm = Self::from_epsg(epsg)?;
        Ok((tm, format!("EPSG:{}", epsg)))
    }

    fn calculate_meridian_distance(&self, phi: f64) -> f64 {
        if phi.abs() < 1e-12 {
            return 0.0;
        }
        let e2 = 2.0 * self.ellipsoid.f - self.ellipsoid.f * self.ellipsoid.f;
        let e = e2.sqrt();
        let s = phi.sin();
        let psi = phi.tan().asinh() - e * (e * s).atanh();
        let xi_prime = psi.sinh().atan2(1.0);

        let mut xi = xi_prime;
        for (j, &a_j) in self.alpha.iter().enumerate() {
            let k = 2.0 * (j as f64 + 1.0);
            xi += a_j * (k * xi_prime).sin();
        }
        self.a_radius * xi
    }

    /// Proyecta una coordenada geográfica (lon_deg, lat_deg) a coordenadas métricas proyectadas (Easting, Northing).
    /// Retorna `(Easting_m, Northing_m, meridian_convergence_deg)`.
    pub fn forward(&self, lon_deg: f64, lat_deg: f64) -> (f64, f64, f64) {
        let phi = lat_deg.to_radians();
        let lambda = lon_deg.to_radians();
        let d_lambda = lambda - self.lon_origin_rad;

        let e2 = 2.0 * self.ellipsoid.f - self.ellipsoid.f * self.ellipsoid.f;
        let e = e2.sqrt();
        let sin_phi = phi.sin();

        // Latitud conforme isométrica
        let psi = phi.tan().asinh() - e * (e * sin_phi).atanh();
        let cos_d_lambda = d_lambda.cos();
        let sin_d_lambda = d_lambda.sin();

        let xi_prime = psi.sinh().atan2(cos_d_lambda);
        let eta_prime = (sin_d_lambda / psi.cosh()).atanh();

        let mut xi = xi_prime;
        let mut eta = eta_prime;
        let mut gamma_series = 0.0;

        for (j, &a_j) in self.alpha.iter().enumerate() {
            let k = 2.0 * (j as f64 + 1.0);
            let sin_k_xi = (k * xi_prime).sin();
            let cos_k_xi = (k * xi_prime).cos();
            let sinh_k_eta = (k * eta_prime).sinh();
            let cosh_k_eta = (k * eta_prime).cosh();

            xi += a_j * sin_k_xi * cosh_k_eta;
            eta += a_j * cos_k_xi * sinh_k_eta;
            gamma_series += a_j * k * cos_k_xi * cosh_k_eta;
        }

        let easting = self.false_easting + self.scale_factor * self.a_radius * eta;
        let northing = self.false_northing + self.scale_factor * (self.a_radius * xi - self.m0);

        // Convergencia de meridianos
        let gamma_0 = (sin_d_lambda * psi.tanh()).atan2(cos_d_lambda);
        let gamma_deg = (gamma_0 + gamma_series).to_degrees();

        (easting, northing, gamma_deg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_colombia_origen_nacional_against_gdal_proj() {
        // Punto de referencia real de la misión Barbosa:
        // lon = -75.306236299, lat = 6.410397430
        // Resultado oficial GDAL PROJ EPSG:9377:
        // X = 4745001.9637 m, Y = 2266910.3477 m
        let tm = TransverseMercator::colombia_origen_nacional();
        let (x, y, gamma) = tm.forward(-75.306236299, 6.410397430);

        let expected_x = 4745001.9637;
        let expected_y = 2266910.3477;

        assert!(
            (x - expected_x).abs() < 1e-3,
            "Error en Easting respecto a GDAL/PROJ: diff = {} m",
            (x - expected_x).abs()
        );
        assert!(
            (y - expected_y).abs() < 1e-3,
            "Error en Northing respecto a GDAL/PROJ: diff = {} m",
            (y - expected_y).abs()
        );

        // Convergencia de meridianos para Barbosa: ~ -0.25° a -0.27°
        assert!(gamma < 0.0 && gamma > -1.0);
    }

    #[test]
    fn test_multi_crs_against_proj_references() {
        // 1. EPSG:32618 (WGS 84 / UTM zone 18N) en Barbosa
        let tm_32618 = TransverseMercator::from_epsg(32618).expect("EPSG:32618");
        let (x, y, _) = tm_32618.forward(-75.306236299, 6.410397430);
        assert!(
            (x - 466135.052805).abs() < 1e-3,
            "EPSG:32618 x: diff={}",
            (x - 466135.052805).abs()
        );
        assert!(
            (y - 708581.242891).abs() < 1e-3,
            "EPSG:32618 y: diff={}",
            (y - 708581.242891).abs()
        );

        // 2. EPSG:32719 (WGS 84 / UTM zone 19S) en Santiago de Chile
        let tm_32719 = TransverseMercator::from_epsg(32719).expect("EPSG:32719");
        let (x, y, _) = tm_32719.forward(-70.64827, -33.45694);
        assert!(
            (x - 346815.712682).abs() < 1e-3,
            "EPSG:32719 x: diff={}",
            (x - 346815.712682).abs()
        );
        assert!(
            (y - 6296839.833389).abs() < 1e-3,
            "EPSG:32719 y: diff={}",
            (y - 6296839.833389).abs()
        );

        // 3. EPSG:25830 (ETRS89 / UTM zone 30N) en Madrid
        let tm_25830 = TransverseMercator::from_epsg(25830).expect("EPSG:25830");
        let (x, y, _) = tm_25830.forward(-3.70379, 40.41678);
        assert!(
            (x - 440291.288768).abs() < 1e-3,
            "EPSG:25830 x: diff={}",
            (x - 440291.288768).abs()
        );
        assert!(
            (y - 4474255.155143).abs() < 1e-3,
            "EPSG:25830 y: diff={}",
            (y - 4474255.155143).abs()
        );

        // 4. EPSG:3116 (MAGNA-SIRGAS / Colombia Bogota zone) en Barbosa
        let tm_3116 = TransverseMercator::from_epsg(3116).expect("EPSG:3116");
        let (x, y, _) = tm_3116.forward(-75.306236299, 6.410397430);
        assert!(
            (x - 864057.883160).abs() < 1e-3,
            "EPSG:3116 x: diff={}",
            (x - 864057.883160).abs()
        );
        assert!(
            (y - 1200784.980556).abs() < 1e-3,
            "EPSG:3116 y: diff={}",
            (y - 1200784.980556).abs()
        );

        // 5. resolve_crs("auto", ...)
        let (_, crs_n) =
            TransverseMercator::resolve_crs("auto", -75.306, 6.410).expect("resolve auto N");
        assert_eq!(crs_n, "EPSG:32618");

        let (_, crs_s) =
            TransverseMercator::resolve_crs("auto", -70.648, -33.456).expect("resolve auto S");
        assert_eq!(crs_s, "EPSG:32719");

        // 6. Rechazo explícito de códigos no proyectados o no soportados
        assert!(TransverseMercator::from_epsg(4326).is_err());
        assert!(TransverseMercator::from_epsg(999999).is_err());
        assert!(TransverseMercator::resolve_crs("EPSG:4326", -75.0, 6.0).is_err());
    }
}
