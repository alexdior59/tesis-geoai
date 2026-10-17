import json
import uuid
from datetime import datetime, timezone
from rasterio.features import shapes, sieve
from shapely.geometry import shape, mapping
import numpy as np

class ResponseSerializer:
    """
    Clase para convertir el resultado de la segmentacion a JSON o GeoJSON.
    """
    API_VERSION = "1.1.0"

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
            "data_provenance": {
                "source_file": self.source_file,
                "crs": m.crs,
                "resolution_m": m.resolution_m,
                "area_ha": m.area_ha,
                "image_size_px": f"{m.width_px} x {m.height_px}",
                "band_count": m.band_count,
                "band_mapping": m.band_mapping,
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
            "drift_report": r.drift_report,
            "disclaimer": (
                "Los resultados identifican patrones espectrales mediante "
                "clustering no supervisado. No constituyen un diagnostico "
                "agronomico definitivo. Se requiere validacion por parte "
                "de un agronomo calificado."
            ),
        }

    def to_geojson(self):
        m = self.processed.metadata
        transform = self.processed.transform
        valid_mask = self.processed.valid_mask

        matriz_etiquetas = np.full(valid_mask.shape, -1, dtype=np.int16)
        matriz_etiquetas[valid_mask] = self.result.kmeans_labels

        # Filtramos micro-parches menores a 100 pixeles (~0.3 m2) para evitar explosion de poligonos
        matriz_limpia = sieve(matriz_etiquetas, size=100)

        features = []
        for geom, val in shapes(matriz_limpia, mask=valid_mask, transform=transform):
            cluster_id = int(val)
            if cluster_id == -1:
                continue
            zone_data = next((z for z in self.result.kmeans_zones if z.cluster_id == cluster_id), None)
            if not zone_data:
                continue

            feature = {
                "type": "Feature",
                "geometry": mapping(shape(geom).simplify(0.2, preserve_topology=True)),
                "properties": {
                    "cluster_id": zone_data.cluster_id,
                    "label": zone_data.label,
                    "area_ha": zone_data.area_ha,
                    "ndvi_mean": zone_data.ndvi_mean,
                    "ndre_mean": zone_data.ndre_mean,
                    "gndvi_mean": zone_data.gndvi_mean,
                    "anomaly_score_mean": zone_data.anomaly_score_mean,
                },
            }
            features.append(feature)

        return {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": m.crs}},
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
            f"Bandas       : {m.band_mapping}",
            f"Pixeles val. : {r.params.get('n_valid_pixels', '?'):,}",
            "",
            "--- MACRO-ZONIFICACION (FCM) ---",
            f"Clusters     : {r.metrics.get('kmeans_n_clusters')}",
            f"Silhouette   : {r.metrics.get('kmeans_silhouette')}  "
            f"({self._interpret_silhouette(r.metrics.get('kmeans_silhouette'))})",
            f"Davies-Bould.: {r.metrics.get('kmeans_davies_bouldin')}",
        ]
        for z in r.kmeans_zones:
            lines.append(
                f"  [{z.cluster_id}] {z.label:<35} "
                f"{z.area_ha:>7.4f} ha  NDVI={z.ndvi_mean:.3f}  "
                f"NDRE={z.ndre_mean:.3f}  Anom={z.anomaly_score_mean:.3f}"
            )
        lines += [
            "",
            "--- ANOMALIAS ESPACIALES (DBSCAN) ---",
            f"Patrones     : {r.metrics.get('dbscan_n_clusters')}",
            f"Ruido        : {r.metrics.get('dbscan_noise_pct')}%",
        ]
        for z in r.dbscan_zones:
            tag = " [RUIDO]" if z.is_noise else ""
            lines.append(
                f"  [{z.cluster_id}] {z.label:<35} "
                f"{z.pixel_count:>8,} px  NDVI={z.ndvi_mean:.3f}  "
                f"NDRE={z.ndre_mean:.3f}  Anom={z.anomaly_score_mean:.3f}{tag}"
            )
        if r.drift_report:
            lines += [
                "",
                "--- REPORTE DE DERIVA (DRIFT REPORT) ---",
                f"Ref. ID      : {r.drift_report.get('reference_model_id')}",
                f"Shift L2     : {r.drift_report.get('centroid_shift_l2')}",
            ]
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
            "ndre_mean": zone.ndre_mean,
            "gndvi_mean": zone.gndvi_mean,
            "anomaly_score_mean": zone.anomaly_score_mean,
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