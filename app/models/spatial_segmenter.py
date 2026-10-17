import numpy as np
from dataclasses import dataclass, field
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

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
    kmeans_zones: list
    dbscan_zones: list
    metrics: dict
    params: dict
    drift_report: dict | None = None
    model_artifacts: dict = field(default_factory=dict)

class SpatialSegmenter:
    """
    Clase que aplica Fuzzy C-Means (FCM) para macro-zonificacion,
    DBSCAN espacial sobre coordenadas fisicas (X, Y) para anomalias,
    y soporta comparacion contra un modelo de referencia (drift_report).
    """
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
        self.kmeans_k = kmeans_k
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.resolution_m = resolution_m
        self.fcm_m = fcm_m
        self.random_state = random_state

    def segment(self, processed, reference_model: dict | None = None):
        features = processed.feature_stack
        coords = processed.spatial_coords
        if features.shape[0] < self.kmeans_k:
            raise ValueError(
                f"Muy pocos pixeles validos ({features.shape[0]}) para {self.kmeans_k} clusters."
            )

        # Si hay modelo de referencia, usamos su escalador o ajustamos uno nuevo y comparamos
        scaler = StandardScaler()
        if reference_model and "scaler_mean" in reference_model and "scaler_scale" in reference_model:
            scaler.mean_ = np.array(reference_model["scaler_mean"], dtype=np.float32)
            scaler.scale_ = np.array(reference_model["scaler_scale"], dtype=np.float32)
            scaler.var_ = scaler.scale_ ** 2
            scaler.n_features_in_ = features.shape[1]
            features_scaled = scaler.transform(features)
        else:
            features_scaled = scaler.fit_transform(features)

        # 1. Macro-zonificacion con Fuzzy C-Means (FCM)
        ref_centers = None
        if reference_model and "fcm_centers" in reference_model:
            ref_centers = np.array(reference_model["fcm_centers"], dtype=np.float32)

        fcm_labels, fcm_centers, min_distances = self._run_fcm(
            features_scaled, initial_centers=ref_centers
        )

        # Calculamos anomaly_scores normalizados [0, 1] a partir de la distancia al centroide FCM
        p99 = float(np.percentile(min_distances, 99))
        anomaly_scores = np.clip(min_distances / max(p99, 1e-6), 0.0, 1.0).astype(np.float32)

        # 2. Micro-deteccion con DBSCAN Espacial en coordenadas (X, Y)
        dbscan_labels_atipicos, indices_atipicos = self._run_dbscan(
            coords, min_distances
        )

        # 3. Metricas de calidad
        metrics = self._compute_metrics(
            features_scaled, fcm_labels, coords[indices_atipicos], dbscan_labels_atipicos
        )

        # 4. Construccion de Zonas Macro (FCM) y Patrones de Anomalias (DBSCAN)
        kmeans_zones = self._build_zones(
            labels=fcm_labels,
            features=features,
            anomaly_scores=anomaly_scores,
            prefix="zona",
        )
        dbscan_features = features[indices_atipicos]
        dbscan_anomaly_scores = anomaly_scores[indices_atipicos]
        dbscan_zones = self._build_zones(
            labels=dbscan_labels_atipicos,
            features=dbscan_features,
            anomaly_scores=dbscan_anomaly_scores,
            prefix="patron",
            include_noise=True,
        )

        # 5. Reporte de deriva / trazabilidad (drift_report) respecto al modelo de referencia
        drift_report = None
        if reference_model is not None:
            drift_report = self._compute_drift_report(
                reference_model=reference_model,
                current_zones=kmeans_zones,
                current_centers=fcm_centers,
                current_band_stats=processed.band_stats,
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
            "reference_model_id": reference_model.get("reference_id") if reference_model else None,
        }

        model_artifacts = {
            "fcm_centers": fcm_centers.tolist(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "kmeans_k": self.kmeans_k,
            "fcm_m": self.fcm_m,
        }

        return SegmentationResult(
            kmeans_labels=fcm_labels,
            dbscan_labels=dbscan_labels_atipicos,
            kmeans_zones=kmeans_zones,
            dbscan_zones=dbscan_zones,
            metrics=metrics,
            params=params,
            drift_report=drift_report,
            model_artifacts=model_artifacts,
        )

    def _run_fcm(self, features_scaled, initial_centers=None, max_iter=50, tol=1e-4):
        """
        Entrena centroides difusos mediante Fuzzy C-Means sobre una muestra representativa
        y asigna membresias por lotes de 500k pixeles para proteger la memoria RAM.
        """
        rng = np.random.default_rng(self.random_state)
        n_total = features_scaled.shape[0]
        sample_size = min(100_000, n_total)
        sample_idx = rng.choice(n_total, size=sample_size, replace=False)
        X_sample = features_scaled[sample_idx]

        if initial_centers is not None and initial_centers.shape[0] == self.kmeans_k:
            centers = initial_centers.copy()
        else:
            centers = X_sample[rng.choice(sample_size, size=self.kmeans_k, replace=False)].copy()

        exp = 2.0 / (self.fcm_m - 1.0)

        for _ in range(max_iter):
            dists = np.linalg.norm(X_sample[:, None, :] - centers[None, :, :], axis=2)
            dists = np.fmax(dists, 1e-10)
            inv_dists = (1.0 / dists) ** exp
            U = inv_dists / np.sum(inv_dists, axis=1, keepdims=True)

            Um = U ** self.fcm_m
            new_centers = (Um.T @ X_sample) / np.sum(Um, axis=0)[:, None]

            if np.linalg.norm(new_centers - centers) < tol:
                centers = new_centers
                break
            centers = new_centers

        labels = np.empty(n_total, dtype=np.int16)
        min_distances = np.empty(n_total, dtype=np.float32)
        batch_size = 500_000

        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            batch = features_scaled[start:end]
            d_batch = np.linalg.norm(batch[:, None, :] - centers[None, :, :], axis=2)
            labels[start:end] = np.argmin(d_batch, axis=1)
            min_distances[start:end] = np.min(d_batch, axis=1)

        return labels, centers, min_distances

    def _run_dbscan(self, spatial_coords, min_distances):
        """
        Filtra el top 5% de pixeles espectralmente atipicos (mas alejados de los centroides FCM)
        y ejecuta DBSCAN estrictamente sobre sus coordenadas fisicas (X, Y) en metros.
        """
        umbral_distancia = np.percentile(min_distances, 95)
        indices_atipicos = np.where(min_distances > umbral_distancia)[0]

        MAX_ATIPICOS = 200_000
        if len(indices_atipicos) > MAX_ATIPICOS:
            top_order = np.argsort(min_distances[indices_atipicos])[-MAX_ATIPICOS:]
            indices_atipicos = np.sort(indices_atipicos[top_order])

        coords_atipicas = spatial_coords[indices_atipicos]
        model = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            algorithm="ball_tree",
            n_jobs=-1,
        )
        labels_atipicos = model.fit_predict(coords_atipicas)
        return labels_atipicos, indices_atipicos

    def _compute_metrics(self, features_scaled, kmeans_labels, coords_atipicas, dbscan_labels):
        metrics = {}
        sample_size = min(10_000, features_scaled.shape[0])
        rng = np.random.default_rng(42)
        idx = rng.choice(features_scaled.shape[0], size=sample_size, replace=False)

        metrics["kmeans_silhouette"] = round(
            float(silhouette_score(features_scaled[idx], kmeans_labels[idx])), 4
        )
        metrics["kmeans_davies_bouldin"] = round(
            float(davies_bouldin_score(features_scaled[idx], kmeans_labels[idx])), 4
        )
        metrics["kmeans_n_clusters"] = self.kmeans_k

        unique_labels = set(dbscan_labels)
        n_clusters = len(unique_labels - {-1})
        n_noise = int((dbscan_labels == -1).sum())
        noise_pct = round(n_noise * 100.0 / max(len(dbscan_labels), 1), 2)

        metrics["dbscan_n_clusters"] = n_clusters
        metrics["dbscan_noise_pixels"] = n_noise
        metrics["dbscan_noise_pct"] = noise_pct

        if n_clusters >= 2 and noise_pct < 90:
            non_noise = dbscan_labels != -1
            if non_noise.sum() >= sample_size:
                idx_db = rng.choice(np.where(non_noise)[0], size=sample_size, replace=False)
            else:
                idx_db = np.where(non_noise)[0]
            if len(set(dbscan_labels[idx_db])) >= 2:
                metrics["dbscan_silhouette"] = round(
                    float(silhouette_score(
                        coords_atipicas[idx_db], dbscan_labels[idx_db]
                    )), 4
                )
            else:
                metrics["dbscan_silhouette"] = None
        else:
            metrics["dbscan_silhouette"] = None

        return metrics

    def _build_zones(self, labels, features, anomaly_scores=None, prefix="zona", include_noise=False):
        px_area_ha = (self.resolution_m ** 2) / 10_000
        zones = []
        for cluster_id in sorted(set(labels)):
            is_noise = cluster_id == -1
            if is_noise and not include_noise:
                continue

            mask = labels == cluster_id
            cluster_features = features[mask]
            pixel_count = int(mask.sum())
            area_ha = round(pixel_count * px_area_ha, 4)

            ndvi_vals = cluster_features[:, 0]
            ndre_vals = cluster_features[:, 1]
            gndvi_vals = cluster_features[:, 2]

            anom_mean = 0.0
            if anomaly_scores is not None:
                anom_mean = round(float(anomaly_scores[mask].mean()), 4)

            label = self._assign_label(
                prefix=prefix,
                cluster_id=cluster_id,
                ndvi_mean=float(ndvi_vals.mean()),
                is_noise=is_noise,
            )

            zones.append(ClusterZone(
                cluster_id=int(cluster_id),
                label=label,
                pixel_count=pixel_count,
                area_ha=area_ha,
                ndvi_mean=round(float(ndvi_vals.mean()), 4),
                ndvi_std=round(float(ndvi_vals.std()), 4),
                ndre_mean=round(float(ndre_vals.mean()), 4),
                gndvi_mean=round(float(gndvi_vals.mean()), 4),
                anomaly_score_mean=anom_mean,
                is_noise=is_noise,
            ))

        return sorted(zones, key=lambda z: z.ndvi_mean, reverse=True)

    @staticmethod
    def _compute_drift_report(reference_model, current_zones, current_centers, current_band_stats):
        """
        Compara los resultados del analisis actual contra la linea base del modelo de referencia
        aprobado por el experto agronomo para cuantificar la evolucion del cultivo (drift).
        """
        ref_centers = np.array(reference_model.get("fcm_centers", []), dtype=np.float32)
        centroid_shift = None
        if ref_centers.shape == current_centers.shape:
            # Ordenamos ambos por la primera columna (NDVI escalado) para alinearlos
            ref_sorted = ref_centers[np.argsort(ref_centers[:, 0])]
            cur_sorted = current_centers[np.argsort(current_centers[:, 0])]
            centroid_shift = round(float(np.linalg.norm(cur_sorted - ref_sorted)), 4)

        ref_stats = reference_model.get("band_stats", {})
        index_deltas = {}
        for idx_name in ("ndvi", "ndre", "gndvi"):
            if idx_name in ref_stats and idx_name in current_band_stats:
                delta_mean = current_band_stats[idx_name]["mean"] - ref_stats[idx_name]["mean"]
                index_deltas[idx_name] = {
                    "reference_mean": round(float(ref_stats[idx_name]["mean"]), 4),
                    "current_mean": round(float(current_band_stats[idx_name]["mean"]), 4),
                    "delta_mean": round(float(delta_mean), 4),
                }

        ref_zones = reference_model.get("kmeans_zones", [])
        zone_comparison = []
        for idx, cur_z in enumerate(current_zones):
            ref_z = ref_zones[idx] if idx < len(ref_zones) else None
            zone_comparison.append({
                "rank": idx + 1,
                "current_label": cur_z.label,
                "current_area_ha": cur_z.area_ha,
                "reference_area_ha": ref_z.get("area_ha") if ref_z else None,
                "area_delta_ha": (
                    round(cur_z.area_ha - ref_z.get("area_ha", 0.0), 4) if ref_z else None
                ),
                "ndvi_delta": (
                    round(cur_z.ndvi_mean - ref_z.get("ndvi_mean", 0.0), 4) if ref_z else None
                ),
                "ndre_delta": (
                    round(cur_z.ndre_mean - ref_z.get("ndre_mean", 0.0), 4) if ref_z else None
                ),
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