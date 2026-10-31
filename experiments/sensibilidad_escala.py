"""
Sensibilidad de la zonificacion a la unidad de analisis (MAUP): pixel original frente a celdas
de 0.5, 1 y 2 m, para k de 2 a 6. Reporta Silhouette, Davies-Bouldin y FPC, y la correlacion
entre indices (redundancia de variables).

Uso:
    python experiments/sensibilidad_escala.py <ortomosaico.tif>
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.models.raster_processor import RasterProcessor  # noqa: E402
from app.models.spatial_segmenter import SpatialSegmenter  # noqa: E402


def aggregate(arr, mask, b):
    if b == 1:
        return arr[mask]
    h, w = (arr.shape[0] // b) * b, (arr.shape[1] // b) * b
    a = np.where(mask, arr, np.nan)[:h, :w].reshape(h // b, b, w // b, b)
    frac = mask[:h, :w].reshape(h // b, b, w // b, b).mean(axis=(1, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(a, axis=(1, 3))
    return mean[frac >= 0.5]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tif")
    a = ap.parse_args()

    p = RasterProcessor(a.tif).process()
    res = p.metadata.resolution_m
    corr = np.corrcoef(p.feature_stack[::200].T)
    print(f"Resolucion {res} m/px | correlacion NDVI-NDRE {corr[0, 1]:.2f}, NDVI-GNDVI {corr[0, 2]:.2f}, NDRE-GNDVI {corr[1, 2]:.2f}")
    print(f"{'unidad':>7} {'n':>11} " + " ".join(f"{'k=' + str(k) + ' Sil/DB/FPC':>20}" for k in range(2, 7)))
    idx = {"ndvi": p.ndvi, "ndre": p.ndre, "gndvi": p.gndvi}
    rng = np.random.default_rng(42)
    for cell in (None, 0.5, 1.0, 2.0):
        b = 1 if cell is None else max(1, round(cell / res))
        feats = np.column_stack([aggregate(idx[n], p.valid_mask, b) for n in ("ndvi", "ndre", "gndvi")])
        feats = feats[np.isfinite(feats).all(axis=1)]
        X = StandardScaler().fit_transform(feats).astype(np.float32)
        s = rng.choice(len(X), size=min(10_000, len(X)), replace=False)
        cells = []
        for k in range(2, 7):
            seg = SpatialSegmenter(kmeans_k=k)
            C = seg._fit_fcm(X)
            labels = seg._assign(X, C)[0]
            d = np.fmax(np.linalg.norm(X[s, None, :] - C[None], axis=2), 1e-10)
            inv = (1.0 / d) ** 2.0
            U = inv / inv.sum(axis=1, keepdims=True)
            fpc = float((U ** 2).sum() / len(s))
            cells.append(f"{silhouette_score(X[s], labels[s]):6.3f}/{davies_bouldin_score(X[s], labels[s]):5.2f}/{fpc:4.2f}")
        name = "pixel" if cell is None else f"{cell} m"
        print(f"{name:>7} {len(X):>11,} " + " ".join(f"{c:>20}" for c in cells), flush=True)


if __name__ == "__main__":
    main()
