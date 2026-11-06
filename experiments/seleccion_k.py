"""
Seleccion del numero de zonas (k) con los indices de Fridgen et al. (2004) y Odeh et al. (1992).

Para cada k calcula:
  FPI  indice de difusividad: 1 - (k*F - 1)/(k - 1), con F el coeficiente de particion (menor es mejor)
  MPE  entropia de particion normalizada: H / ln(k) (menor es mejor)
  FPC  coeficiente de particion de Bezdek (1/k = sin estructura, 1 = nitida)
  Silhouette (mayor es mejor) y Davies-Bouldin (menor es mejor)
y el NDVI medio de cada zona con su rango en la escala del codirector.

Uso:
    python experiments/seleccion_k.py <ortomosaico.tif> [--kmin 2] [--kmax 6] [--lote lote.geojson]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.models.aoi import read_aoi  # noqa: E402
from app.models.crop_profiles import range_for  # noqa: E402
from app.models.raster_processor import RasterProcessor  # noqa: E402
from app.models.spatial_segmenter import SpatialSegmenter  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tif")
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--lote", help="poligono del lote (GeoJSON, shapefile o GeoPackage)")
    a = ap.parse_args()

    p = RasterProcessor(a.tif, aoi=read_aoi(a.lote) if a.lote else None).process()
    X = StandardScaler().fit_transform(p.feature_stack).astype(np.float32)
    ndvi = p.feature_stack[:, 0]
    rng = np.random.default_rng(42)
    s = rng.choice(len(X), size=min(20_000, len(X)), replace=False)

    print(f"{'k':>2} {'FPI':>6} {'MPE':>6} {'FPC':>6} {'Silh':>6} {'DB':>6}   zonas: NDVI medio (rango) y % del area")
    for k in range(a.kmin, a.kmax + 1):
        seg = SpatialSegmenter(kmeans_k=k)
        C = seg._fit_fcm(X)
        d = np.fmax(np.linalg.norm(X[s, None, :] - C[None], axis=2), 1e-10)
        inv = (1.0 / d) ** 2.0
        U = inv / inv.sum(axis=1, keepdims=True)
        F = float((U ** 2).sum() / len(s))
        fpi = 1 - (k * F - 1) / (k - 1)
        mpe = float(-(U * np.log(U)).sum() / len(s)) / np.log(k)
        lab = U.argmax(axis=1)
        sil = silhouette_score(X[s][:10_000], lab[:10_000])
        db = davies_bouldin_score(X[s], lab)
        labels_all = seg._assign(X, C)[0]
        zonas = []
        for c in sorted(range(k), key=lambda c: -ndvi[labels_all == c].mean()):
            m = labels_all == c
            v = float(ndvi[m].mean())
            zonas.append(f"{v:.2f} ({range_for(v)['clave']}) {m.mean():.0%}")
        print(f"{k:>2} {fpi:6.3f} {mpe:6.3f} {F:6.3f} {sil:6.3f} {db:6.3f}   " + " | ".join(zonas), flush=True)


if __name__ == "__main__":
    main()
