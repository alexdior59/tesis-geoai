from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


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
    anomaly_score_mean: float = 0.0
    is_noise: bool = False


@dataclass
class SegmentationResult:
    kmeans_labels: np.ndarray
    dbscan_labels: np.ndarray
    outlier_indices: np.ndarray
    kmeans_zones: list
    dbscan_zones: list
    metrics: dict
    params: dict
    drift_report: dict | None = None
    model_artifacts: dict = field(default_factory=dict)


class SpatialSegmenter:
    """FCM para las zonas de manejo y DBSCAN sobre (X, Y) para los focos anomalos."""

    FCM_SAMPLE_SIZE = 100_000
    BATCH_SIZE = 500_000
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

    def segment(self, processed, reference_model=None):
        features = processed.feature_stack
        coords = processed.spatial_coords
        if features.shape[0] < self.kmeans_k:
            raise ValueError(
                f"Muy pocos pixeles validos ({features.shape[0]}) para {self.kmeans_k} clusters."
            )

        # Con un modelo de referencia se reutiliza su escalado para que los
        # centroides de ambos vuelos esten en el mismo espacio.
        scaler = StandardScaler()
        if reference_model and "scaler_mean" in reference_model:
            scaler.mean_ = np.array(reference_model["scaler_mean"], dtype=np.float64)
            scaler.scale_ = np.array(reference_model["scaler_scale"], dtype=np.float64)
            scaler.var_ = scaler.scale_ ** 2
            scaler.n_features_in_ = features.shape[1]
            features_scaled = scaler.transform(features).astype(np.float32)
        else:
            features_scaled = scaler.fit_transform(features).astype(np.float32)

        ref_centers = None
        if reference_model and "fcm_centers" in reference_model:
            ref_centers = np.array(reference_model["fcm_centers"], dtype=np.float32)

        labels, centers, min_distances, fpc = self._run_fcm(features_scaled, ref_centers)

        p99 = float(np.percentile(min_distances, 99))
        anomaly_scores = np.clip(min_distances / max(p99, 1e-6), 0.0, 1.0).astype(np.float32)

        dbscan_labels, outlier_indices = self._run_dbscan(coords, min_distances)

        metrics = self._compute_metrics(
            features_scaled, labels, coords[outlier_indices], dbscan_labels
        )
        metrics["fcm_partition_coefficient"] = round(fpc, 4)

        kmeans_zones = self._build_zones(labels, features, anomaly_scores, prefix="zona")
        dbscan_zones = self._build_zones(
            dbscan_labels,
            features[outlier_indices],
            anomaly_scores[outlier_indices],
            prefix="patron",
            include_noise=True,
        )

        drift_report = None
        if reference_model is not None:
            drift_report = self._compute_drift_report(
                reference_model, kmeans_zones, centers, processed.band_stats
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
        }

        model_artifacts = {
            "fcm_centers": centers.tolist(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "kmeans_k": self.kmeans_k,
            "fcm_m": self.fcm_m,
        }

        return SegmentationResult(
            kmeans_labels=labels,
            dbscan_labels=dbscan_labels,
            outlier_indices=outlier_indices,
            kmeans_zones=kmeans_zones,
            dbscan_zones=dbscan_zones,
            metrics=metrics,
            params=params,
            drift_report=drift_report,
            model_artifacts=model_artifacts,
        )

    def _run_fcm(self, features_scaled, initial_centers=None, max_iter=50, tol=1e-4):
        # Los centroides se entrenan con una muestra; la asignacion de todos los
        # pixeles se hace por lotes para no cargar la matriz completa de distancias.
        rng = np.random.default_rng(self.random_state)
        n_total = features_scaled.shape[0]
        sample_size = min(self.FCM_SAMPLE_SIZE, n_total)
        sample = features_scaled[rng.choice(n_total, size=sample_size, replace=False)]

        if initial_centers is not None and initial_centers.shape[0] == self.kmeans_k:
            centers = initial_centers.copy()
        else:
            centers = sample[rng.choice(sample_size, size=self.kmeans_k, replace=False)].copy()

        exp = 2.0 / (self.fcm_m - 1.0)
        for _ in range(max_iter):
            dists = np.fmax(np.linalg.norm(sample[:, None, :] - centers[None, :, :], axis=2), 1e-10)
            inv = dists ** -exp
            u = inv / inv.sum(axis=1, keepdims=True)
            um = u ** self.fcm_m
            new_centers = (um.T @ sample) / um.sum(axis=0)[:, None]
            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if shift < tol:
                break

        # Coeficiente de particion de Bezdek: 1/C = particion totalmente difusa, 1 = dura
        fpc = float(np.mean(np.sum(u ** 2, axis=1)))

        labels = np.empty(n_total, dtype=np.int16)
        min_distances = np.empty(n_total, dtype=np.float32)
        for start in range(0, n_total, self.BATCH_SIZE):
            batch = features_scaled[start:start + self.BATCH_SIZE]
            d = np.linalg.norm(batch[:, None, :] - centers[None, :, :], axis=2)
            # con distancia euclidiana, el centroide mas cercano es el de mayor pertenencia
            labels[start:start + self.BATCH_SIZE] = np.argmin(d, axis=1)
            min_distances[start:start + self.BATCH_SIZE] = d.min(axis=1)

        return labels, centers.astype(np.float32), min_distances, fpc

    def _run_dbscan(self, spatial_coords, min_distances):
        threshold = np.percentile(min_distances, self.OUTLIER_PERCENTILE)
        outliers = np.flatnonzero(min_distances > threshold)

        # Si hay demasiados se conservan los mas alejados, no una muestra aleatoria,
        # para no romper la densidad de los focos.
        if outliers.size > self.MAX_OUTLIERS:
            top = np.argsort(min_distances[outliers])[-self.MAX_OUTLIERS:]
            outliers = np.sort(outliers[top])

        if outliers.size == 0:
            return np.empty(0, dtype=np.int32), outliers

        model = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            algorithm="ball_tree",
            n_jobs=-1,
        )
        return model.fit_predict(spatial_coords[outliers]), outliers

    def _compute_metrics(self, features_scaled, labels, outlier_coords, dbscan_labels):
        rng = np.random.default_rng(self.random_state)
        sample_size = min(10_000, features_scaled.shape[0])
        idx = rng.choice(features_scaled.shape[0], size=sample_size, replace=False)

        metrics = {"kmeans_silhouette": None, "kmeans_davies_bouldin": None}
        if len(np.unique(labels[idx])) >= 2:
            metrics["kmeans_silhouette"] = round(float(silhouette_score(features_scaled[idx], labels[idx])), 4)
            metrics["kmeans_davies_bouldin"] = round(
                float(davies_bouldin_score(features_scaled[idx], labels[idx])), 4
            )
        metrics["kmeans_n_clusters"] = self.kmeans_k

        n_clusters = len(set(np.unique(dbscan_labels)) - {-1})
        n_noise = int((dbscan_labels == -1).sum())
        noise_pct = round(n_noise * 100.0 / max(dbscan_labels.size, 1), 2)
        metrics["dbscan_n_clusters"] = n_clusters
        metrics["dbscan_noise_pixels"] = n_noise
        metrics["dbscan_noise_pct"] = noise_pct

        metrics["dbscan_silhouette"] = None
        if n_clusters >= 2 and noise_pct < 90:
            clustered = np.flatnonzero(dbscan_labels != -1)
            if clustered.size > sample_size:
                clustered = rng.choice(clustered, size=sample_size, replace=False)
            if len(np.unique(dbscan_labels[clustered])) >= 2:
                metrics["dbscan_silhouette"] = round(
                    float(silhouette_score(outlier_coords[clustered], dbscan_labels[clustered])), 4
                )
        return metrics

    def _build_zones(self, labels, features, anomaly_scores, prefix="zona", include_noise=False):
        px_area_ha = (self.resolution_m ** 2) / 10_000
        zones = []
        for cluster_id in np.unique(labels):
            is_noise = cluster_id == -1
            if is_noise and not include_noise:
                continue

            mask = labels == cluster_id
            cluster_features = features[mask]
            pixel_count = int(mask.sum())
            ndvi_mean = float(cluster_features[:, 0].mean())

            zones.append(ClusterZone(
                cluster_id=int(cluster_id),
                label=self._assign_label(prefix, int(cluster_id), ndvi_mean, is_noise),
                pixel_count=pixel_count,
                area_ha=round(pixel_count * px_area_ha, 4),
                ndvi_mean=round(ndvi_mean, 4),
                ndvi_std=round(float(cluster_features[:, 0].std()), 4),
                ndre_mean=round(float(cluster_features[:, 1].mean()), 4),
                gndvi_mean=round(float(cluster_features[:, 2].mean()), 4),
                anomaly_score_mean=round(float(anomaly_scores[mask].mean()), 4),
                is_noise=bool(is_noise),
            ))

        return sorted(zones, key=lambda z: z.ndvi_mean, reverse=True)

    @staticmethod
    def _compute_drift_report(reference_model, current_zones, current_centers, current_band_stats):
        ref_centers = np.array(reference_model.get("fcm_centers", []), dtype=np.float32)
        centroid_shift = None
        if ref_centers.shape == current_centers.shape:
            # se alinean por la columna de NDVI escalado antes de comparar
            ref_sorted = ref_centers[np.argsort(ref_centers[:, 0])]
            cur_sorted = current_centers[np.argsort(current_centers[:, 0])]
            centroid_shift = round(float(np.linalg.norm(cur_sorted - ref_sorted)), 4)

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

        # Ambas listas vienen ordenadas por NDVI descendente, asi que se comparan por posicion
        ref_zones = reference_model.get("kmeans_zones", [])
        zone_comparison = []
        for rank, cur in enumerate(current_zones):
            ref = ref_zones[rank] if rank < len(ref_zones) else None
            zone_comparison.append({
                "rank": rank + 1,
                "current_label": cur.label,
                "current_area_ha": cur.area_ha,
                "reference_area_ha": ref.get("area_ha") if ref else None,
                "area_delta_ha": round(cur.area_ha - ref.get("area_ha", 0.0), 4) if ref else None,
                "ndvi_delta": round(cur.ndvi_mean - ref.get("ndvi_mean", 0.0), 4) if ref else None,
                "ndre_delta": round(cur.ndre_mean - ref.get("ndre_mean", 0.0), 4) if ref else None,
            })

        return {
            "reference_model_id": reference_model.get("reference_id"),
            "reference_name": reference_model.get("name"),
            "reference_created_at": reference_model.get("created_at"),
            "centroid_shift_l2": centroid_shift,
            "index_deltas": index_deltas,
            "zone_comparison": zone_comparison,
        }

    @staticmethod
    def _assign_label(prefix, cluster_id, ndvi_mean, is_noise):
        if is_noise:
            return "ruido_espacial"
        if ndvi_mean >= 0.5:
            return f"{prefix}_vegetacion_alta_{cluster_id}"
        if ndvi_mean >= 0.3:
            return f"{prefix}_vegetacion_moderada_{cluster_id}"
        if ndvi_mean >= 0.1:
            return f"{prefix}_vegetacion_baja_{cluster_id}"
        return f"{prefix}_sin_vegetacion_{cluster_id}"