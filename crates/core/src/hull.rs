//! Algoritmo de Envolvente Convexa 2D (Convex Hull) y Point-in-Polygon.
//! Implementa el algoritmo Monotone Chain de Andrew en O(N log N) con alta estabilidad numérica
//! (sustracción de origen local) para coordenadas métricas proyectadas grandes (EPSG:9377).

use serde::{Deserialize, Serialize};

/// Punto bidimensional en punto flotante de 64 bits.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

impl Point2D {
    #[inline]
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// Polígono de envolvente convexa bidimensional.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConvexHull2D {
    /// Vértices en orden antihorario (CCW).
    pub vertices: Vec<Point2D>,
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl ConvexHull2D {
    /// Construye la envolvente convexa de una nube de puntos 2D usando el algoritmo Monotone Chain de Andrew.
    /// Retorna `None` si hay menos de 3 puntos no colineales.
    pub fn from_points(pts: &[Point2D]) -> Option<Self> {
        if pts.len() < 3 {
            return None;
        }

        // 1. Calcular origen local y bounding box para evitar pérdida de precisión con números de 10^6
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for p in pts {
            if p.x < min_x {
                min_x = p.x;
            }
            if p.x > max_x {
                max_x = p.x;
            }
            if p.y < min_y {
                min_y = p.y;
            }
            if p.y > max_y {
                max_y = p.y;
            }
        }

        if (max_x - min_x).abs() < 1e-6 && (max_y - min_y).abs() < 1e-6 {
            return None;
        }

        // 2. Coordenadas relativas al origen local
        let mut local_pts: Vec<(f64, f64, usize)> = pts
            .iter()
            .enumerate()
            .map(|(idx, p)| (p.x - min_x, p.y - min_y, idx))
            .collect();

        // 3. Ordenar lexicográficamente (por x, luego por y)
        local_pts.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        });

        // 4. Eliminar duplicados cercanos
        local_pts.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-6 && (a.1 - b.1).abs() < 1e-6);

        let k = local_pts.len();
        if k < 3 {
            return None;
        }

        // Producto cruzado 2D de vectores (b - a) x (c - a)
        let cross = |a: (f64, f64, usize), b: (f64, f64, usize), c: (f64, f64, usize)| -> f64 {
            (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
        };

        // 5. Construir casco inferior
        let mut lower: Vec<(f64, f64, usize)> = Vec::with_capacity(k);
        for &p in &local_pts {
            while lower.len() >= 2
                && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 1e-9
            {
                lower.pop();
            }
            lower.push(p);
        }

        // 6. Construir casco superior
        let mut upper: Vec<(f64, f64, usize)> = Vec::with_capacity(k);
        for &p in local_pts.iter().rev() {
            while upper.len() >= 2
                && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 1e-9
            {
                upper.pop();
            }
            upper.push(p);
        }

        // 7. Concatenar excluyendo el último punto de cada mitad (está duplicado)
        lower.pop();
        upper.pop();

        let mut hull_tuples = lower;
        hull_tuples.extend(upper);

        if hull_tuples.len() < 3 {
            return None;
        }

        // Reconstruir en coordenadas globales
        let vertices: Vec<Point2D> = hull_tuples
            .into_iter()
            .map(|(rx, ry, _)| Point2D::new(rx + min_x, ry + min_y))
            .collect();

        Some(Self {
            vertices,
            min_x,
            min_y,
            max_x,
            max_y,
        })
    }

    /// Verifica si un punto (x, y) cae dentro o sobre el perímetro de la envolvente convexa en O(K).
    #[inline]
    pub fn contains(&self, x: f64, y: f64) -> bool {
        // Fast rejection con bounding box
        if x < (self.min_x - 1e-4)
            || x > (self.max_x + 1e-4)
            || y < (self.min_y - 1e-4)
            || y > (self.max_y + 1e-4)
        {
            return false;
        }

        let n = self.vertices.len();
        if n < 3 {
            return false;
        }

        // Para un polígono convexo orientado en sentido antihorario (CCW),
        // un punto está dentro si está a la izquierda (cross >= -eps) de cada arista orientada.
        for i in 0..n {
            let p1 = self.vertices[i];
            let p2 = self.vertices[(i + 1) % n];
            let cross = (p2.x - p1.x) * (y - p1.y) - (p2.y - p1.y) * (x - p1.x);
            if cross < -1e-6 {
                return false;
            }
        }
        true
    }

    /// Exporta el polígono del casco a formato GeoJSON en coordenadas proyectadas EPSG:9377.
    pub fn to_geojson(&self, name: &str) -> String {
        self.to_geojson_crs(name, "EPSG:9377")
    }

    /// Exporta el polígono del casco a formato GeoJSON especificando el CRS proyectado.
    pub fn to_geojson_crs(&self, name: &str, crs: &str) -> String {
        let mut coords_str = String::new();
        for (i, p) in self.vertices.iter().enumerate() {
            if i > 0 {
                coords_str.push_str(", ");
            }
            coords_str.push_str(&format!("[{:.4}, {:.4}]", p.x, p.y));
        }
        // Cerrar el anillo
        if let Some(first) = self.vertices.first() {
            coords_str.push_str(&format!(", [{:.4}, {:.4}]", first.x, first.y));
        }

        let crs_urn = if let Some(code) = crs.strip_prefix("EPSG:") {
            format!("urn:ogc:def:crs:EPSG::{}", code)
        } else {
            crs.to_string()
        };

        format!(
            "{{\n  \"type\": \"FeatureCollection\",\n  \"name\": \"{}\",\n  \"crs\": {{\n    \"type\": \"name\",\n    \"properties\": {{\n      \"name\": \"{}\"\n    }}\n  }},\n  \"features\": [\n    {{\n      \"type\": \"Feature\",\n      \"properties\": {{\n        \"name\": \"{}\",\n        \"vertex_count\": {}\n      }},\n      \"geometry\": {{\n        \"type\": \"Polygon\",\n        \"coordinates\": [[\n          {}\n        ]]\n      }}\n    }}\n  ]\n}}\n",
            name,
            crs_urn,
            name,
            self.vertices.len(),
            coords_str
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convex_hull_square() {
        let pts = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(10.0, 0.0),
            Point2D::new(10.0, 10.0),
            Point2D::new(0.0, 10.0),
            Point2D::new(5.0, 5.0), // punto interior
        ];

        let hull = ConvexHull2D::from_points(&pts).expect("Debe generar casco");
        assert_eq!(hull.vertices.len(), 4);
        assert!(hull.contains(5.0, 5.0));
        assert!(hull.contains(0.0, 0.0));
        assert!(hull.contains(10.0, 10.0));
        assert!(!hull.contains(11.0, 5.0));
        assert!(!hull.contains(-1.0, 5.0));
    }

    #[test]
    fn test_convex_hull_collinear() {
        let pts = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(5.0, 5.0),
            Point2D::new(10.0, 10.0),
        ];
        // Colineales no forman un polígono con área positiva
        assert!(ConvexHull2D::from_points(&pts).is_none());
    }

    #[test]
    fn test_convex_hull_offset_precision() {
        // Coordenadas típicas de EPSG:9377 en Colombia
        let x0 = 4_745_000.0;
        let y0 = 2_267_000.0;

        let pts = vec![
            Point2D::new(x0 + 0.0, y0 + 0.0),
            Point2D::new(x0 + 100.0, y0 + 0.0),
            Point2D::new(x0 + 100.0, y0 + 150.0),
            Point2D::new(x0 + 0.0, y0 + 150.0),
            Point2D::new(x0 + 50.0, y0 + 75.0),
        ];

        let hull = ConvexHull2D::from_points(&pts).expect("Casco EPSG:9377 válido");
        assert_eq!(hull.vertices.len(), 4);
        assert!(hull.contains(x0 + 50.0, y0 + 75.0));
        assert!(!hull.contains(x0 + 150.0, y0 + 75.0));
    }
}
