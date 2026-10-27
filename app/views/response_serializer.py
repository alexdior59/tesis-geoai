import json
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from rasterio.features import shapes, sieve
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

API_VERSION = "1.1.0"


class ResponseSerializer:
    """Convierte el resultado de la segmentacion a JSON, GeoJSON y texto."""

    SIEVE_MIN_PX = 100
    SIMPLIFY_TOLERANCE_M = 0.2

    def __init__(self, processed, result, job_id, source_file="", extra_params=None):
        self.processed = processed
        self.result = result
        self.job_id = job_id
        self.source_file = source_file
        self.extra_params = extra_params or {}
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_json(self):
        m = self.processed.metadata
        r = self.result
        return {
            "job_id": self.job_id,
            "api_version": API_VERSION,
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
                "dtype": m.dtype,
                "source_software": m.source_software,
                "capture_datetime": m.capture_datetime,
            },
            "analysis_params": {**r.params, **self.extra_params},
            "quality_metrics": {
                "kmeans": {
                    "silhouette_score": r.metrics.get("kmeans_silhouette"),
                    "davies_bouldin_index": r.metrics.get("kmeans_davies_bouldin"),
                    "fcm_partition_coefficient": r.metrics.get("fcm_partition_coefficient"),
                    "n_clusters": r.metrics.get("kmeans_n_clusters"),
                    "interpretation": self._interpret_silhouette(r.metrics.get("kmeans_silhouette")),
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
                "Los resultados identifican patrones espectrales mediante clustering no "
                "supervisado. No constituyen un diagnostico agronomico definitivo y deben "
                "ser validados por un agronomo."
            ),
        }

    def to_geojson(self):
        features = self._fcm_features() + self._dbscan_features()
        return {
            "type": "FeatureCollection",
            # el miembro "crs" no esta en el RFC 7946, pero QGIS lo lee para ubicar la capa en UTM
            "crs": {"type": "name", "properties": {"name": self.processed.metadata.crs}},
            "features": features,
        }

    def _fcm_features(self):
        valid_mask = self.processed.valid_mask
        label_grid = np.full(valid_mask.shape, -1, dtype=np.int16)
        label_grid[valid_mask] = self.result.kmeans_labels
        label_grid = sieve(label_grid, size=self.SIEVE_MIN_PX)

        zones = {z.cluster_id: z for z in self.result.kmeans_zones}
        features = []
        for geom, value in shapes(label_grid, mask=valid_mask, transform=self.processed.transform):
            zone = zones.get(int(value))
            if zone is None:
                continue
            polygon = shape(geom).simplify(self.SIMPLIFY_TOLERANCE_M, preserve_topology=True)
            features.append({
                "type": "Feature",
                "geometry": mapping(polygon),
                "properties": {
                    "layer": "zona_fcm",
                    "cluster_id": zone.cluster_id,
                    "label": zone.label,
                    "area_ha": round(polygon.area / 10_000, 6),
                    "zone_area_ha": zone.area_ha,
                    "ndvi_mean": zone.ndvi_mean,
                    "ndre_mean": zone.ndre_mean,
                    "gndvi_mean": zone.gndvi_mean,
                    "anomaly_score_mean": zone.anomaly_score_mean,
                },
            })
        return features

    def _dbscan_features(self):
        r = self.result
        if r.outlier_indices.size == 0:
            return []

        valid_mask = self.processed.valid_mask
        pattern_grid = np.full(valid_mask.shape, -1, dtype=np.int32)
        flat_positions = np.flatnonzero(valid_mask)[r.outlier_indices]
        pattern_grid.flat[flat_positions] = r.dbscan_labels

        # un mismo patron puede salir en varios pedazos; se unen en una sola geometria
        pieces = defaultdict(list)
        for geom, value in shapes(pattern_grid, mask=pattern_grid >= 0, transform=self.processed.transform):
            pieces[int(value)].append(shape(geom))

        zones = {z.cluster_id: z for z in r.dbscan_zones if not z.is_noise}
        features = []
        for cluster_id, polygons in pieces.items():
            zone = zones.get(cluster_id)
            if zone is None:
                continue
            geometry = unary_union(polygons).simplify(self.SIMPLIFY_TOLERANCE_M, preserve_topology=True)
            features.append({
                "type": "Feature",
                "geometry": mapping(geometry),
                "properties": {
                    "layer": "patron_dbscan",
                    "cluster_id": zone.cluster_id,
                    "label": zone.label,
                    "area_ha": zone.area_ha,
                    "pixel_count": zone.pixel_count,
                    "ndvi_mean": zone.ndvi_mean,
                    "ndre_mean": zone.ndre_mean,
                    "gndvi_mean": zone.gndvi_mean,
                    "anomaly_score_mean": zone.anomaly_score_mean,
                },
            })
        return features

    def to_summary(self):
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
            f"Pixeles val. : {r.params.get('n_valid_pixels', 0):,}",
            "",
            "--- MACRO-ZONIFICACION (FCM) ---",
            f"Clusters     : {r.metrics.get('kmeans_n_clusters')}",
            f"Silhouette   : {r.metrics.get('kmeans_silhouette')}  "
            f"({self._interpret_silhouette(r.metrics.get('kmeans_silhouette'))})",
            f"Davies-Bould.: {r.metrics.get('kmeans_davies_bouldin')}",
            f"Coef. part.  : {r.metrics.get('fcm_partition_coefficient')}",
        ]
        for z in r.kmeans_zones:
            lines.append(
                f"  [{z.cluster_id}] {z.label:<35} {z.area_ha:>7.4f} ha  "
                f"NDVI={z.ndvi_mean:.3f}  NDRE={z.ndre_mean:.3f}  Anom={z.anomaly_score_mean:.3f}"
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
                f"  [{z.cluster_id}] {z.label:<35} {z.pixel_count:>8,} px  "
                f"NDVI={z.ndvi_mean:.3f}  NDRE={z.ndre_mean:.3f}  Anom={z.anomaly_score_mean:.3f}{tag}"
            )
        if r.drift_report:
            lines += [
                "",
                "--- REPORTE DE DERIVA ---",
                f"Referencia   : {r.drift_report.get('reference_model_id')}",
                f"Shift L2     : {r.drift_report.get('centroid_shift_l2')}",
            ]
        lines += ["", "=" * 60]
        return "\n".join(lines)

    def save_json(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=2, default=self.json_default)

    def save_geojson(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_geojson(), f, ensure_ascii=False, indent=2, default=self.json_default)

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
    def json_default(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Tipo no serializable: {type(obj)}")