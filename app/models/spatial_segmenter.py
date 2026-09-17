import numpy as np
from dataclasses import dataclass
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans, DBSCAN


@dataclass
class ClusterZone:
    cluster_id: int
    label: str
    pixel_count: int
    area_ha: float
    ndvi_mean: float
    ndvi_std: float
    savi_mean: float
    gndvi_mean: float
    is_noise: bool = False


@dataclass
class SegmentationResult:
    kmeans_labels: np.ndarray
    dbscan_labels: np.ndarray
    kmeans_zones: list
    dbscan_zones: list
    metrics: dict
    params: dict


class SpatialSegmenter:
    """
    Clase que aplica K-Means y DBSCAN sobre los features procesados.
    """

    def __init__(
            self,
            kmeans_k=4,
            dbscan_eps=0.3,
            dbscan_min_samples=50,
            resolution_m=0.1,
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
        self.random_state = random_state

    def segment(self, processed):
        features = processed.feature_stack

        if features.shape[0] < self.kmeans_k:
            raise ValueError(
                f"Muy pocos píxeles válidos ({features.shape[0]}) para {self.kmeans_k} clusters."
            )

        # print("Normalizando datos para los modelos...")
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # 1. K-Means
        kmeans_labels = self._run_kmeans(features_scaled)

        # 2. DBSCAN
        # TODO: A veces el eps es muy sensible, revisar si lo volvemos un parametro dinamico despues
        dbscan_labels, dbscan_indices = self._run_dbscan(features_scaled)

        # 3. Métricas
        metrics = self._compute_metrics(features_scaled, kmeans_labels, dbscan_labels)

        # 4. Zonas
        kmeans_zones = self._build_zones(
            labels=kmeans_labels,
            features=features,
            prefix="zona",
        )

        dbscan_features = features[dbscan_indices] if features.shape[0] > 200_000 else features

        dbscan_zones = self._build_zones(
            labels=dbscan_labels,
            features=dbscan_features,
            prefix="patron",
            include_noise=True,
        )

        params = {
            "kmeans_k": self.kmeans_k,
            "dbscan_eps": self.dbscan_eps,
            "dbscan_min_samples": self.dbscan_min_samples,
            "resolution_m": self.resolution_m,
            "random_state": self.random_state,
            "n_valid_pixels": int(features.shape[0]),
        }

        # print("Segmentacion terminada con exito")

        return SegmentationResult(
            kmeans_labels=kmeans_labels,
            dbscan_labels=dbscan_labels,
            kmeans_zones=kmeans_zones,
            dbscan_zones=dbscan_zones,
            metrics=metrics,
            params=params,
        )

    def _run_kmeans(self, features_scaled):
        model = MiniBatchKMeans(
            n_clusters=self.kmeans_k,
            init="k-means++",
            n_init=3,
            max_iter=100,
            batch_size=10000,
            random_state=self.random_state,
        )
        return model.fit_predict(features_scaled)

    def _run_dbscan(self, features_scaled):
        MAX_PIXELS = 200_000
        n = features_scaled.shape[0]

        if n <= MAX_PIXELS:
            indices = np.arange(n)
            sample = features_scaled
        else:
            rng = np.random.default_rng(self.random_state)
            indices = rng.choice(n, size=MAX_PIXELS, replace=False)
            indices.sort()
            sample = features_scaled[indices]
            print(f"dataset muy grande ({n} px). Subsample a 200k")

        model = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            algorithm="ball_tree",
            n_jobs=-1,
        )
        labels = model.fit_predict(sample)
        return labels, indices

    def _compute_metrics(self, features_scaled, kmeans_labels, dbscan_labels):
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
        n_clusters = len(unique_labels - {-1})  # -1 es ruido
        n_noise = int((dbscan_labels == -1).sum())
        noise_pct = round(n_noise * 100.0 / len(dbscan_labels), 2)

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
                        features_scaled[idx_db], dbscan_labels[idx_db]
                    )), 4
                )
        else:
            metrics["dbscan_silhouette"] = None

        return metrics

    def _build_zones(self, labels, features, prefix, include_noise=False):
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
            savi_vals = cluster_features[:, 1]
            gndvi_vals = cluster_features[:, 2]

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
                savi_mean=round(float(savi_vals.mean()), 4),
                gndvi_mean=round(float(gndvi_vals.mean()), 4),
                is_noise=is_noise,
            ))

        return sorted(zones, key=lambda z: z.ndvi_mean, reverse=True)

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