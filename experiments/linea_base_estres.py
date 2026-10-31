"""
Experimento de la linea base pedido por el director.

Toma un ortomosaico como vuelo 1 y simula un vuelo 2 bajando el NIR de la mitad izquierda del lote.
Clasifica el vuelo 2 de dos formas, reajustando FCM desde la linea base (como antes de la v1.2) y con
los centroides congelados, y mide cuantos pixeles cambian de zona en la mitad que no se toco.

Uso:
    python experiments/linea_base_estres.py <ortomosaico.tif> [--estres 0.85] [--k 4]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.models.raster_processor import ProcessedRaster, RasterProcessor  # noqa: E402
from app.models.spatial_segmenter import SpatialSegmenter  # noqa: E402
from app.views.response_serializer import ResponseSerializer  # noqa: E402


def build(proc, bands, transform, meta):
    idx = proc._calculate_indices(bands)
    mask = proc._build_valid_mask(bands, idx)
    return ProcessedRaster(
        ndvi=idx["ndvi"], ndre=idx["ndre"], gndvi=idx["gndvi"], valid_mask=mask,
        feature_stack=proc._build_feature_stack(idx, mask),
        spatial_coords=proc._extract_coordinates(mask, transform),
        transform=transform, metadata=meta, band_stats=proc._compute_stats(idx, mask),
    )


def full_map(p, labels):
    m = np.full(p.valid_mask.shape, -1, dtype=np.int16)
    m[p.valid_mask] = labels
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tif")
    ap.add_argument("--estres", type=float, default=0.85, help="factor aplicado al NIR de la mitad izquierda")
    ap.add_argument("--k", type=int, default=4)
    a = ap.parse_args()

    proc = RasterProcessor(a.tif)
    with rasterio.open(a.tif) as ds:
        bm = proc._resolve_band_mapping(ds)
        meta = proc._extract_metadata(ds, bm)
        tr = ds.transform
        bands = proc._load_bands(ds, bm)

    p1 = build(proc, bands, tr, meta)
    seg = SpatialSegmenter(kmeans_k=a.k, resolution_m=meta.resolution_m)
    r1 = seg.segment(p1)
    ref = {
        "reference_id": "experimento", "name": "vuelo 1", "created_at": "experimento",
        **{k: r1.model_artifacts[k] for k in ("fcm_centers", "scaler_mean", "scaler_scale", "kmeans_k", "distance_p99")},
        "band_stats": p1.band_stats,
        "kmeans_zones": [ResponseSerializer._zone_to_dict(z) for z in r1.kmeans_zones],
    }
    rank = {z.cluster_id: i for i, z in enumerate(sorted(r1.kmeans_zones, key=lambda z: -z.ndvi_mean))}
    L1 = full_map(p1, r1.kmeans_labels)

    r1b = seg.segment(p1, reference_model=ref)
    print(f"Mismo vuelo contra su linea base: coincidencia {np.mean(r1b.kmeans_labels == r1.kmeans_labels):.2%}")

    w = bands["nir"].shape[1]
    bands["nir"][:, : w // 2] *= a.estres
    p2 = build(proc, bands, tr, meta)
    del bands
    r2 = seg.segment(p2, reference_model=ref)
    L2 = full_map(p2, r2.kmeans_labels)

    X2 = (p2.feature_stack - np.array(ref["scaler_mean"], dtype=np.float32)) / np.array(ref["scaler_scale"], dtype=np.float32)
    refit = seg._fit_fcm(X2, initial_centers=np.array(ref["fcm_centers"]))
    L2r = full_map(p2, seg._assign(X2, refit)[0])

    lut = np.zeros(max(rank) + 1, dtype=np.int8)
    for c, i in rank.items():
        lut[c] = i
    both = (L1 >= 0) & (L2 >= 0)
    cols = np.broadcast_to(np.arange(w)[None, :], L1.shape)
    print(f"\nEstres simulado: NIR x {a.estres} en la mitad izquierda")
    print(f"{'Mitad':12} {'Metodo':36} {'Cambian de zona':>16} {'Bajan de zona':>14}")
    for name, m in (("con estres", both & (cols < w // 2)), ("sin cambio", both & (cols >= w // 2))):
        base = lut[L1[m]]
        for meth, L in (("centroides re-ajustados (anterior)", L2r), ("centroides congelados (v1.2)", L2)):
            nuevo = lut[L[m]]
            print(f"{name:12} {meth:36} {np.mean(base != nuevo):>15.1%} {np.mean(nuevo > base):>13.1%}")

    print("\nReporte de cambio por zona (v1.2):")
    for z in r2.drift_report["zone_comparison"]:
        print(f"  zona {z['cluster_id']}: {z['reference_area_ha']:.2f} ha -> {z['current_area_ha']:.2f} ha "
              f"({z['area_delta_pct']}%)  desplazamiento del centroide {z['centroid_shift_l2']}")


if __name__ == "__main__":
    main()
