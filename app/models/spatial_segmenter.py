from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from app.models.crop_profiles import NDVI_RANGES_EXPERTO, range_for
from app.models.explainability import max_membership


@dataclass
class ClusterZone:
    cluster_id: int
    label: str
    pixel_count: int
    area_ha: float
    ndvi_mean: float
    ndvi_std: float
    ndre_mean: float
    gndvi_mean: float
    deviation_score_mean: float = 0.0
    is_noise: bool = False
    expert_label: str | None = None


@dataclass
class SegmentationResult:
    kmeans_labels: np.ndarray
    dbscan_labels: np.ndarray
    kmeans_zones: list
    dbscan_zones: list
    metrics: dict
    params: dict
    drift_report: dict | None = None
    model_artifacts: dict = field(default_factory=dict)
    max_membership: np.ndarray | None = None
    outlier_indices: np.ndarray | None = None


class SpatialSegmenter:
    """
    FCM para las zonas de manejo y DBSCAN sobre (X, Y) para los patrones atipicos.
    Con una linea base, los pixeles se asignan a sus centroides congelados.
    """

    FCM_SAMPLE_SIZE = 100_000
    OUTLIER_PERCENTILE = 95
    MAX_OUTLIERS = 200_000

    def __init__(
            self,
            kmeans_k=4,
            dbscan_eps=0.5,
            dbscan_min_samples=30,
            resolution_m=0.1,
            fcm_m=2.0,
            random_state=42,
            ndvi_ranges=NDVI_RANGES_EXPERTO,
    ):
        if kmeans_k < 2:
            raise ValueError("kmeans_k debe ser >= 2")
        if dbscan_eps <= 0:
            raise ValueError("dbscan_eps debe ser > 0")
        if fcm_m <= 1:
            raise ValueError("fcm_m debe ser > 1")
        self.kmeans_k = kmeans_k
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.resolution_m = resolution_m
        self.fcm_m = fcm_m
        self.random_state = random_state
        self.ndvi_ranges = ndvi_ranges

    def segment(self, processed, reference_model=None):
        features = processed.feature_stack
        coords = processed.spatial_coords
        if features.shape[0] < self.kmeans_k:
            raise ValueError(
                f"Muy pocos pixeles validos ({features.shape[0]}) para {self.kmeans_k} clusters."
            )

        scaler = StandardScaler()
        if reference_model is not None:
            ref_k = int(reference_model["kmeans_k"])
            if ref_k != self.kmeans_k:
                raise ValueError(
                    f"La linea base tiene k={ref_k} y el analisis pidio k={self.kmeans_k}."
                )
            scaler.mean_ = np.array(reference_model["scaler_mean"], dtype=np.float32)
            scaler.scale_ = np.array(reference_model["scaler_scale"], dtype=np.float32)
            scaler.var_ = scaler.scale_ ** 2
            scaler.n_features_in_ = features.shape[1]
            features_scaled = scaler.transform(features)
            fcm_centers = np.array(reference_model["fcm_centers"], dtype=np.float32)
        else:
            features_scaled = scaler.fit_transform(features)
            fcm_centers = self._fit_fcm(features_scaled)

        labels, min_distances, memberships = self._assign(features_scaled, fcm_centers, self.fcm_m)

        # La desviacion se normaliza con el p99 de la linea base para que un
        # empeoramiento general del lote no quede oculto al comparar vuelos.
        p99_current = float(np.percentile(min_distances, 99))
        p99_ref = float(reference_model.get("distance_p99", p99_current)) if reference_model else p99_current
        deviation_scores = np.clip(min_distances / max(p99_ref, 1e-6), 0.0, 1.0).astype(np.float32)

        dbscan_labels, outlier_indices = self._run_dbscan(coords, min_distances)

        metrics = self._compute_metrics(
            features_scaled, labels, coords[outlier_indices], dbscan_labels, fcm_centers
        )

        expert_labels = (reference_model or {}).get("zone_labels") or {}
        kmeans_zones = self._build_zones(
            labels, features, deviation_scores, prefix="zona", expert_labels=expert_labels
        )
        dbscan_zones = self._build_zones(
            dbscan_labels,
            features[outlier_indices],
            deviation_scores[outlier_indices],
            prefix="patron",
            include_noise=True,
        )

        drift_report = None
        if reference_model is not None:
            # El reajuste solo se usa para medir cuanto se movio cada zona;
            # la clasificacion sigue siendo la de los centroides congelados.
            refit_centers = self._fit_fcm(features_scaled, initial_centers=fcm_centers)
            drift_report = self._compute_drift_report(
                reference_model, kmeans_zones, refit_centers, processed.band_stats, p99_current
            )

        params = {
            "algorithm_macro": "Fuzzy C-Means (FCM)",
            "kmeans_k": self.kmeans_k,
            "fcm_fuzziness_m": self.fcm_m,
            "dbscan_eps": self.dbscan_eps,
            "dbscan_min_samples": self.dbscan_min_samples,
            "resolution_m": self.resolution_m,
            "random_state": self.random_state,
            "n_valid_pixels": int(features.shape[0]),
            "n_outlier_pixels": int(outlier_indices.size),
            "reference_model_id": reference_model.get("reference_id") if reference_model else None,
            "classification_mode": "centroides_congelados_referencia" if reference_model else "fcm_libre",
        }

        model_artifacts = {
            "fcm_centers": fcm_centers.tolist(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "kmeans_k": self.kmeans_k,
            "fcm_m": self.fcm_m,
            "distance_p99": p99_ref,
            "feature_names": ["ndvi", "ndre", "gndvi"],
        }

        return SegmentationResult(
            kmeans_labels=labels,
            dbscan_labels=dbscan_labels,
            kmeans_zones=kmeans_zones,
            dbscan_zones=dbscan_zones,
            metrics=metrics,
            params=params,
            drift_report=drift_report,
            model_artifacts=model_artifacts,
            max_membership=memberships,
            outlier_indices=outlier_indices,
        )

    def _fit_fcm(self, features_scaled, initial_centers=None, max_iter=50, tol=1e-4):
        # Los centroides se aprenden con una muestra; la asignacion de todos los
        # pixeles se hace despues, por lotes, en _assign.
        rng = np.random.default_rng(self.random_state)
        n_total = features_scaled.shape[0]
        sample_size = min(self.FCM_SAMPLE_SIZE, n_total)
        sample = features_scaled[rng.choice(n_total, size=sample_size, replace=False)]

        if initial_centers is not None:
            centers = np.array(initial_centers, dtype=np.float64)
        else:
            centers = sample[rng.choice(sample_size, size=self.kmeans_k, replace=False)].copy()

        exp = 2.0 / (self.fcm_m - 1.0)
        for _ in range(max_iter):
            dists = np.fmax(np.linalg.norm(sample[:, None, :] - centers[None, :, :], axis=2), 1e-10)
            inv = (1.0 / dists) ** exp
            u = inv / inv.sum(axis=1, keepdims=True)
            um = u ** self.fcm_m
            new_centers = (um.T @ sample) / um.sum(axis=0)[:, None]
            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if shift < tol:
                break
        return centers

    @staticmethod
    def _assign(features_scaled, centers, fcm_m=2.0, batch_size=500_000):
        n_total = features_scaled.shape[0]
        labels = np.empty(n_total, dtype=np.int16)
        min_distances = np.empty(n_total, dtype=np.float32)
        memberships = np.empty(n_total, dtype=np.float32)
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            d = np.linalg.norm(features_scaled[start:end, None, :] - centers[None, :, :], axis=2)
            # el centroide mas cercano es tambien el de mayor pertenencia
            labels[start:end] = np.argmin(d, axis=1)
            min_distances[start:end] = d.min(axis=1)
            memberships[start:end] = max_membership(d, fcm_m)
        return labels, min_distances, memberships

    def _run_dbscan(self, spatial_coords, min_distances):
        threshold = np.percentile(min_distances, self.OUTLIER_PERCENTILE)
        outliers = np.flatnonzero(min_distances > threshold)

        # Si hay demasiados se conservan los mas alejados en vez de una muestra
        # aleatoria, para no romper la densidad de los parches en terreno.
        if outliers.size > self.MAX_OUTLIERS:
            top = np.argsort(min_distances[outliers])[-self.MAX_OUTLIERS:]
            outliers = np.sort(outliers[top])

        if outliers.size == 0:
            return np.empty(0, dtype=np.int64), outliers

        model = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            algorithm="ball_tree",
            n_jobs=-1,
        )
        return model.fit_predict(spatial_coords[outliers]), outliers

    def _compute_metrics(self, features_scaled, labels, outlier_coords, dbscan_labels, centers):
        rng = np.random.default_rng(42)
        sample_size = min(10_000, features_scaled.shape[0])
        idx = rng.choice(features_scaled.shape[0], size=sample_size, replace=False)
        metrics = {}

        # Coeficiente de particion de Bezdek (1981): 1/k = sin estructura, 1 = particion nitida
        d = np.fmax(np.linalg.norm(features_scaled[idx, None, :] - centers[None, :, :], axis=2), 1e-10)
        inv = (1.0 / d) ** (2.0 / (self.fcm_m - 1.0))
        u = inv / inv.sum(axis=1, keepdims=True)
        metrics["fcm_partition_coefficient"] = round(float((u ** 2).sum() / sample_size), 4)

        metrics["kmeans_silhouette"] = None
        metrics["kmeans_davies_bouldin"] = None
        if len(np.unique(labels[idx])) >= 2:
            metrics["kmeans_silhouette"] = round(float(silhouette_score(features_scaled[idx], labels[idx])), 4)
            metrics["kmeans_davies_bouldin"] = round(
                float(davies_bouldin_score(features_scaled[idx], labels[idx])), 4
            )
        metrics["kmeans_n_clusters"] = self.kmeans_k

        n_clusters = len(set(np.unique(dbscan_labels).tolist()) - {-1})
        n_noise = int((dbscan_labels == -1).sum())
        noise_pct = round(n_noise * 100.0 / max(dbscan_labels.size, 1), 2)
        metrics["dbscan_n_clusters"] = n_clusters
        metrics["dbscan_noise_pixels"] = n_noise
        metrics["dbscan_noise_pct"] = noise_pct

        metrics["dbscan_silhouette"] = None
        if n_clusters >= 2 and noise_pct < 90:
            clustered = np.flatnonzero(dbscan_labels != -1)
            if clustered.size >= sample_size:
                clustered = rng.choice(clustered, size=sample_size, replace=False)
            if len(np.unique(dbscan_labels[clustered])) >= 2:
                metrics["dbscan_silhouette"] = round(
                    float(silhouette_score(outlier_coords[clustered], dbscan_labels[clustered])), 4
                )
        return metrics

    def _build_zones(self, labels, features, deviation_scores, prefix="zona",
                     include_noise=False, expert_labels=None):
        px_area_ha = (self.resolution_m ** 2) / 10_000
        expert_labels = expert_labels or {}
        zones = []
        for cluster_id in np.unique(labels).tolist():
            is_noise = cluster_id == -1
            if is_noise and not include_noise:
                continue

            mask = labels == cluster_id
            cluster_features = features[mask]
            pixel_count = int(mask.sum())
            ndvi_mean = float(cluster_features[:, 0].mean())

            zones.append(ClusterZone(
                cluster_id=cluster_id,
                label=self._assign_label(prefix, cluster_id, ndvi_mean, is_noise),
                pixel_count=pixel_count,
                area_ha=round(pixel_count * px_area_ha, 4),
                ndvi_mean=round(ndvi_mean, 4),
                ndvi_std=round(float(cluster_features[:, 0].std()), 4),
                ndre_mean=round(float(cluster_features[:, 1].mean()), 4),
                gndvi_mean=round(float(cluster_features[:, 2].mean()), 4),
                deviation_score_mean=round(float(deviation_scores[mask].mean()), 4),
                is_noise=is_noise,
                expert_label=expert_labels.get(str(cluster_id)),
            ))

        return sorted(zones, key=lambda z: z.ndvi_mean, reverse=True)

    @staticmethod
    def _compute_drift_report(reference_model, current_zones, refit_centers, current_band_stats, p99_current):
        ref_centers = np.array(reference_model["fcm_centers"], dtype=np.float64)
        shift_by_zone = np.linalg.norm(refit_centers - ref_centers, axis=1)

        ref_stats = reference_model.get("band_stats", {})
        index_deltas = {}
        for name in ("ndvi", "ndre", "gndvi"):
            if name in ref_stats and name in current_band_stats:
                ref_mean = ref_stats[name]["mean"]
                cur_mean = current_band_stats[name]["mean"]
                index_deltas[name] = {
                    "reference_mean": round(float(ref_mean), 4),
                    "current_mean": round(float(cur_mean), 4),
                    "delta_mean": round(float(cur_mean - ref_mean), 4),
                }

        # Con centroides congelados el cluster_id es la misma condicion en ambos
        # vuelos, asi que las zonas se emparejan por id y no por posicion.
        labels = reference_model.get("zone_labels") or {}
        ref_by_id = {int(z["cluster_id"]): z for z in reference_model.get("kmeans_zones", [])}
        cur_by_id = {z.cluster_id: z for z in current_zones}
        zone_comparison = []
        for cid in sorted(set(ref_by_id) | set(cur_by_id)):
            ref, cur = ref_by_id.get(cid), cur_by_id.get(cid)
            ref_area = ref["area_ha"] if ref else 0.0
            cur_area = cur.area_ha if cur else 0.0
            zone_comparison.append({
                "cluster_id": cid,
                "expert_label": labels.get(str(cid)),
                "reference_area_ha": ref_area,
                "current_area_ha": cur_area,
                "area_delta_ha": round(cur_area - ref_area, 4),
                "area_delta_pct": round((cur_area - ref_area) * 100.0 / ref_area, 2) if ref_area else None,
                "ndvi_delta": round(cur.ndvi_mean - ref["ndvi_mean"], 4) if ref and cur else None,
                "ndre_delta": round(cur.ndre_mean - ref["ndre_mean"], 4) if ref and cur else None,
                "centroid_shift_l2": round(float(shift_by_zone[cid]), 4) if cid < len(shift_by_zone) else None,
            })

        p99_ref = reference_model.get("distance_p99")
        return {
            "reference_model_id": reference_model.get("reference_id"),
            "reference_name": reference_model.get("name"),
            "reference_created_at": reference_model.get("created_at"),
            "validated_by": reference_model.get("validated_by"),
            "method": "Clasificacion con centroides congelados de la linea base; "
                      "el desplazamiento se mide re-ajustando FCM desde esos centroides.",
            "centroid_shift_l2": round(float(np.linalg.norm(refit_centers - ref_centers)), 4),
            "distance_p99_reference": round(float(p99_ref), 4) if p99_ref is not None else None,
            "distance_p99_current": round(p99_current, 4),
            "index_deltas": index_deltas,
            "zone_comparison": zone_comparison,
        }

    def _assign_label(self, prefix, cluster_id, ndvi_mean, is_noise):
        if is_noise:
            return "ruido_espacial"
        return f"{prefix}_{range_for(ndvi_mean, self.ndvi_ranges)['clave']}_{cluster_id}"