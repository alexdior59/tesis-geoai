import json
import uuid
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import rasterio
from rasterio.features import shapes, sieve
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

API_VERSION = "1.2.0"
MAX_PALMAS_EN_JSON = 200


class ResponseSerializer:
    """Convierte el resultado del analisis a JSON, GeoJSON, texto y GeoTIFF."""

    API_VERSION = API_VERSION
    SIEVE_MIN_PX = 100
    SIMPLIFY_TOLERANCE_M = 0.2

    def __init__(self, processed, result, source_file="", condition_report=None, palm_report=None,
                 explainability=None, job_id=None, extra_params=None):
        self.processed = processed
        self.result = result
        self.source_file = source_file
        self.condition_report = condition_report
        self.palm_report = palm_report
        self.explainability = explainability
        self.job_id = job_id or str(uuid.uuid4())
        self.extra_params = extra_params or {}
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_json(self):
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
                "dtype": m.dtype,
                "source_software": m.source_software,
                "capture_datetime": m.capture_datetime,
            },
            "analysis_params": {**r.params, **self.extra_params},
            "condicion_por_rangos": self.condition_report,
            "analisis_palmas": self._palm_summary(),
            "quality_metrics": {
                "kmeans": {
                    "silhouette_score": r.metrics.get("kmeans_silhouette"),
                    "davies_bouldin_index": r.metrics.get("kmeans_davies_bouldin"),
                    "n_clusters": r.metrics.get("kmeans_n_clusters"),
                    "interpretation": self._interpret_silhouette(r.metrics.get("kmeans_silhouette")),
                    "fcm_partition_coefficient": r.metrics.get("fcm_partition_coefficient"),
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
            "explicabilidad": self.explainability,
            "dbscan_patterns": [self._zone_to_dict(z) for z in r.dbscan_zones],
            "drift_report": r.drift_report,
            "disclaimer": (
                "Los resultados identifican zonas con comportamiento espectral distinto mediante "
                "clustering no supervisado y rangos de NDVI. Una zona de inspeccion no implica dano: "
                "puede corresponder a estres, madurez o senescencia. No constituyen un diagnostico "
                "agronomico; la verificacion en campo y la decision de intervenir corresponden al productor."
            ),
        }

    def _palm_summary(self):
        if self.palm_report is None:
            return None
        summary = dict(self.palm_report)
        for key in ("lista_inspeccion", "lista_copa_debil", "lista_faltantes"):
            summary[f"{key}_truncada"] = len(summary[key]) > MAX_PALMAS_EN_JSON
            summary[key] = summary[key][:MAX_PALMAS_EN_JSON]
        return summary

    def to_geojson(self):
        """Zonas FCM. area_ha es el area de cada poligono y zone_area_ha la de toda la zona."""
        valid_mask = self.processed.valid_mask
        label_grid = np.full(valid_mask.shape, -1, dtype=np.int16)
        label_grid[valid_mask] = self.result.kmeans_labels
        # los parches de menos de 100 px (~0.3 m2) se funden con su vecino
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
                    "cluster_id": zone.cluster_id,
                    "label": zone.label,
                    "expert_label": zone.expert_label,
                    "area_ha": round(polygon.area / 10_000, 6),
                    "zone_area_ha": zone.area_ha,
                    "ndvi_mean": zone.ndvi_mean,
                    "ndre_mean": zone.ndre_mean,
                    "gndvi_mean": zone.gndvi_mean,
                    "deviation_score_mean": zone.deviation_score_mean,
                },
            })
        return self._feature_collection(features)

    def to_patterns_geojson(self):
        """Patrones de DBSCAN como poligonos, uno por patron (sin el ruido)."""
        r = self.result
        if r.outlier_indices is None or r.outlier_indices.size == 0:
            return self._feature_collection([])

        valid_mask = self.processed.valid_mask
        pattern_grid = np.full(valid_mask.shape, -1, dtype=np.int32)
        pattern_grid.flat[np.flatnonzero(valid_mask)[r.outlier_indices]] = r.dbscan_labels

        # un mismo patron puede quedar en varios pedazos; se unen en una sola geometria
        pieces = defaultdict(list)
        for geom, value in shapes(pattern_grid, mask=pattern_grid >= 0, transform=self.processed.transform):
            pieces[int(value)].append(shape(geom))

        zones = {z.cluster_id: z for z in r.dbscan_zones if not z.is_noise}
        features = []
        for cluster_id, polygons in sorted(pieces.items()):
            zone = zones.get(cluster_id)
            if zone is None:
                continue
            geometry = unary_union(polygons).simplify(self.SIMPLIFY_TOLERANCE_M, preserve_topology=True)
            features.append({
                "type": "Feature",
                "geometry": mapping(geometry),
                "properties": {
                    "patron_id": zone.cluster_id,
                    "pixel_count": zone.pixel_count,
                    "area_ha": zone.area_ha,
                    "ndvi_mean": zone.ndvi_mean,
                    "ndre_mean": zone.ndre_mean,
                    "gndvi_mean": zone.gndvi_mean,
                    "deviation_score_mean": zone.deviation_score_mean,
                },
            })
        return self._feature_collection(features)

    def _feature_collection(self, features):
        return {
            "type": "FeatureCollection",
            # "crs" no esta en el RFC 7946, pero QGIS lo usa para ubicar la capa en UTM
            "crs": {"type": "name", "properties": {"name": self.processed.metadata.crs}},
            "features": features,
        }

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
        ]
        c = self.condition_report
        if c:
            insp = c["inspeccion"]
            lines += [
                "",
                f"--- CONDICION POR RANGOS DE NDVI ({c['cultivo_nombre']}) ---",
                f"Area con datos: {c['area_con_datos_ha']} ha",
            ]
            for rg in c["rangos"]:
                lines.append(
                    f"  NDVI {rg['ndvi_min']:>5.2f} a {rg['ndvi_max']:<5.2f} {rg['descripcion'][:42]:<42} "
                    f"{rg['area_ha']:>8.3f} ha  {rg['pct_area']:>6.2f}%"
                )
            lines.append(
                f"Inspeccion   : {insp['pct_area_en_rango']}% del area ({insp['area_en_rango_ha']} ha) en NDVI "
                f"{insp['rango_ndvi'][0]}-{insp['rango_ndvi'][1]}; {insp['n_zonas']} zonas >= {insp['zona_minima_m2']} m2"
            )
        p = self.palm_report
        if p:
            lines += [
                "",
                "--- ANALISIS POR PALMA ---",
                f"Palmas       : {p['conteo_palmas']:,}  ({p['densidad_palmas_ha']} palmas/ha; "
                f"esperado {p['densidad_esperada_palmas_ha']})",
                f"A inspeccionar (NDVI): {p['palmas_a_inspeccionar']} ({p['pct_palmas_a_inspeccionar']}%)",
                f"Copa debil (verificar): {p['palmas_copa_debil']} ({p['pct_palmas_copa_debil']}%)",
                f"Posibles faltantes: {p['posibles_faltantes']}",
            ]
        lines += [
            "",
            "--- MACRO-ZONIFICACION (FCM) ---",
            f"Clusters     : {r.metrics.get('kmeans_n_clusters')}",
            f"Silhouette   : {r.metrics.get('kmeans_silhouette')}  "
            f"({self._interpret_silhouette(r.metrics.get('kmeans_silhouette'))})",
            f"Davies-Bould.: {r.metrics.get('kmeans_davies_bouldin')}",
            f"FPC (Bezdek) : {r.metrics.get('fcm_partition_coefficient')}  (1/k = sin estructura, 1 = nitida)",
        ]
        for z in r.kmeans_zones:
            name = f" [{z.expert_label}]" if z.expert_label else ""
            lines.append(
                f"  [{z.cluster_id}] {z.label:<32} {z.area_ha:>7.4f} ha  NDVI={z.ndvi_mean:.3f}  "
                f"NDRE={z.ndre_mean:.3f}  Desv={z.deviation_score_mean:.3f}{name}"
            )
        e = self.explainability
        if e:
            imp = e["importancia_permutacion"]
            sus = e["modelo_sustituto"]
            inc = e["incertidumbre"]
            lines += [
                "",
                "--- EXPLICABILIDAD ---",
                "Importancia (% de pixeles que cambian de zona al desordenar el indice): "
                + ", ".join(f"{k} {v}%" for k, v in imp.items()),
                f"Arbol sustituto (profundidad {sus['profundidad']}): fidelidad {sus['fidelidad_pct']}%",
                f"Incertidumbre: pertenencia maxima media {inc['pertenencia_max_media']}; "
                f"{inc['pct_pixeles_transicion']}% de pixeles de transicion (pertenencia < {inc['umbral_transicion']})",
            ]
        lines += [
            "",
            "--- PATRONES ESPACIALES ATIPICOS (DBSCAN) ---",
            f"Patrones     : {r.metrics.get('dbscan_n_clusters')}",
            f"Ruido        : {r.metrics.get('dbscan_noise_pct')}%",
        ]
        if r.drift_report:
            lines += [
                "",
                "--- CAMBIO RESPECTO A LA LINEA BASE ---",
                f"Ref. ID      : {r.drift_report.get('reference_model_id')}",
                f"Shift L2     : {r.drift_report.get('centroid_shift_l2')}",
            ]
            for zc in r.drift_report.get("zone_comparison", []):
                pct = zc["area_delta_pct"] if zc["area_delta_pct"] is not None else "-"
                lines.append(
                    f"  Zona {zc['cluster_id']} {zc['expert_label'] or '':<24} "
                    f"area {zc['reference_area_ha']:.3f} -> {zc['current_area_ha']:.3f} ha ({pct}%)"
                )
        lines += ["", "=" * 60]
        return "\n".join(lines)

    def write_uncertainty_tif(self, path):
        """Pertenencia de cada pixel a su zona, en %. 100 = asignacion segura, 255 = sin dato."""
        if self.result.max_membership is None:
            return None
        valid = self.processed.valid_mask
        raster = np.full(valid.shape, 255, dtype=np.uint8)
        raster[valid] = np.clip(np.round(self.result.max_membership * 100), 0, 100).astype(np.uint8)
        with rasterio.open(path, "w", driver="GTiff", height=raster.shape[0], width=raster.shape[1], count=1,
                           dtype="uint8", nodata=255, crs=self.processed.metadata.crs,
                           transform=self.processed.transform, compress="deflate", tiled=True) as dst:
            dst.write(raster, 1)
            dst.set_band_description(1, "Pertenencia maxima FCM (%)")
        return path

    def save_json(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=2, default=self._json_default)

    def save_geojson(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_geojson(), f, ensure_ascii=False, indent=2, default=self._json_default)

    @staticmethod
    def _zone_to_dict(zone):
        return {
            "cluster_id": zone.cluster_id,
            "label": zone.label,
            "expert_label": zone.expert_label,
            "pixel_count": zone.pixel_count,
            "area_ha": zone.area_ha,
            "ndvi_mean": zone.ndvi_mean,
            "ndvi_std": zone.ndvi_std,
            "ndre_mean": zone.ndre_mean,
            "gndvi_mean": zone.gndvi_mean,
            "deviation_score_mean": zone.deviation_score_mean,
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
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Tipo no serializable: {type(obj)}")