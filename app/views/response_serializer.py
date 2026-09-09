import json
import uuid
from datetime import datetime, timezone
from dataclasses import asdict

import numpy as np


class ResponseSerializer:
    """
    Clase para convertir el resultado de la segmentacion a JSON o GeoJSON.
    """

    API_VERSION = "1.0.0"

    def __init__(self, processed, result, source_file=""):
        self.processed = processed
        self.result = result
        self.source_file = source_file
        self.job_id = str(uuid.uuid4())
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_json(self):
        """Arma el JSON principal que va a devolver la API."""
        m = self.processed.metadata
        r = self.result

        return {
            "job_id": self.job_id,
            "api_version": self.API_VERSION,
            "timestamp_utc": self.timestamp,
            "status": "completed",

            # TODO: Asegurar que estos metadatos cuadren con el inventario de datos del excel
            "data_provenance": {
                "source_file": self.source_file,
                "crs": m.crs,
                "resolution_m": m.resolution_m,
                "area_ha": m.area_ha,
                "image_size_px": f"{m.width_px} x {m.height_px}",
                "band_count": m.band_count,
                "source_software": m.source_software,
                "capture_datetime": m.capture_datetime,
            },

            "analysis_params": r.params,

            "quality_metrics": {
                "kmeans": {
                    "silhouette_score": r.metrics.get("kmeans_silhouette"),
                    "davies_bouldin_index": r.metrics.get("kmeans_davies_bouldin"),
                    "n_clusters": r.metrics.get("kmeans_n_clusters"),
                    "interpretation": self._interpret_silhouette(
                        r.metrics.get("kmeans_silhouette")
                    ),
                },
                "dbscan": {
                    "n_clusters_detected": r.metrics.get("dbscan_n_clusters"),
                    "noise_pixels": r.metrics.get("dbscan_noise_pixels"),
                    "noise_pct": r.metrics.get("dbscan_noise_pct"),
                    "silhouette_score": r.metrics.get("dbscan_silhouette"),
                },
            },

            "vegetation_indices": self.processed.band_stats,

            "kmeans_zones": [self._zone_to_dict(z) for z in r.kmeans_zones],

            "dbscan_patterns": [self._zone_to_dict(z) for z in r.dbscan_zones],

            "disclaimer": (
                "Los resultados identifican patrones espectrales mediante "
                "clustering no supervisado. No constituyen un diagnostico "
                "agronomico definitivo. Se requiere validacion por parte "
                "de un agronomo calificado."
            ),
        }

    def to_geojson(self):
        """
        Arma el GeoJSON para poder pintarlo en QGIS.
        """
        m = self.processed.metadata
        features = []

        for zone in self.result.kmeans_zones:
            bbox = self._estimate_bbox(zone, m)

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [bbox],
                },
                "properties": {
                    "cluster_id": zone.cluster_id,
                    "label": zone.label,
                    "area_ha": zone.area_ha,
                    "pixel_count": zone.pixel_count,
                    "ndvi_mean": zone.ndvi_mean,
                    "ndvi_std": zone.ndvi_std,
                    "savi_mean": zone.savi_mean,
                    "gndvi_mean": zone.gndvi_mean,
                    "crs": m.crs,
                    "job_id": self.job_id,
                },
            }
            features.append(feature)

        return {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": m.crs},
            },
            "metadata": {
                "job_id": self.job_id,
                "timestamp_utc": self.timestamp,
                "source_file": self.source_file,
                "total_zones": len(features),
            },
            "features": features,
        }

    def to_summary(self):
        """Imprime un resumen en texto para ver en consola."""
        r = self.result
        m = self.processed.metadata
        lines = [
            "=" * 60,
            "RESUMEN - Motor GeoAI",
            "=" * 60,
            f"Job ID       : {self.job_id}",
            f"Archivo      : {self.source_file}",
            f"CRS          : {m.crs}",
            f"Resolucion   : {m.resolution_m} m/px",
            f"Area total   : {m.area_ha} ha",
            f"Pixeles val. : {r.params.get('n_valid_pixels', '?'):,}",
            "",
            "--- K-MEANS ---",
            f"Clusters     : {r.metrics.get('kmeans_n_clusters')}",
            f"Silhouette   : {r.metrics.get('kmeans_silhouette')}  "
            f"({self._interpret_silhouette(r.metrics.get('kmeans_silhouette'))})",
            f"Davies-Bould.: {r.metrics.get('kmeans_davies_bouldin')}",
        ]
        for z in r.kmeans_zones:
            lines.append(
                f"  [{z.cluster_id}] {z.label:<40} "
                f"{z.area_ha:>7.4f} ha  NDVI={z.ndvi_mean:.3f}"
            )
        lines += [
            "",
            "--- DBSCAN ---",
            f"Patrones     : {r.metrics.get('dbscan_n_clusters')}",
            f"Ruido        : {r.metrics.get('dbscan_noise_pct')}%",
        ]
        for z in r.dbscan_zones:
            tag = " [RUIDO]" if z.is_noise else ""
            lines.append(
                f"  [{z.cluster_id}] {z.label:<40} "
                f"{z.pixel_count:>8,} px  NDVI={z.ndvi_mean:.3f}{tag}"
            )
        lines += ["", "=" * 60]
        return "\n".join(lines)

    def save_json(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=2,
                      default=self._json_default)

    def save_geojson(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_geojson(), f, ensure_ascii=False, indent=2,
                      default=self._json_default)

    @staticmethod
    def _zone_to_dict(zone):
        return {
            "cluster_id": zone.cluster_id,
            "label": zone.label,
            "pixel_count": zone.pixel_count,
            "area_ha": zone.area_ha,
            "ndvi_mean": zone.ndvi_mean,
            "ndvi_std": zone.ndvi_std,
            "savi_mean": zone.savi_mean,
            "gndvi_mean": zone.gndvi_mean,
            "is_noise": zone.is_noise,
        }

    @staticmethod
    def _interpret_silhouette(score):
        if score is None:
            return "no calculado"
        if score >= 0.7:
            return "estructura fuerte"
        if score >= 0.5:
            return "estructura razonable"
        if score >= 0.25:
            return "estructura debil"
        return "sin estructura clara"

    @staticmethod
    def _estimate_bbox(zone, metadata):
        # FIXME: Esto saca un cuadrado proporcional, no el poligono real. Luego hay que meterle rasterio.features.shapes para sacar el borde exacto
        b = metadata.bounds
        total_px = metadata.width_px * metadata.height_px
        ratio = zone.pixel_count / max(total_px, 1)

        left = b["left"]
        bottom = b["bottom"]
        width = (b["right"] - b["left"]) * ratio
        height = (b["top"] - b["bottom"]) * ratio

        return [
            [left, bottom],
            [left + width, bottom],
            [left + width, bottom + height],
            [left, bottom + height],
            [left, bottom],
        ]

    @staticmethod
    def _json_default(obj):
        """Sirve para que json.dump no explote con los numeros de numpy."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Tipo no serializable: {type(obj)}")