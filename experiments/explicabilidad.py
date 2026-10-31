"""
Explicabilidad de la zonificacion FCM sobre un ortomosaico real: prototipos de cada zona,
incertidumbre, tabla de atributos de algunos pixeles, explicacion local de un pixel de transicion,
importancia por permutacion y el arbol sustituto con sus reglas.

Uso:
    python experiments/explicabilidad.py <ortomosaico.tif> [--k 4]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.models.explainability import ZoneExplainer, local_attribution  # noqa: E402
from app.models.raster_processor import RasterProcessor  # noqa: E402
from app.models.spatial_segmenter import SpatialSegmenter  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tif")
    ap.add_argument("--k", type=int, default=4)
    a = ap.parse_args()

    p = RasterProcessor(a.tif).process()
    r = SpatialSegmenter(kmeans_k=a.k, resolution_m=p.metadata.resolution_m).segment(p)
    art = r.model_artifacts
    C = np.array(art["fcm_centers"])
    mean, scale = np.array(art["scaler_mean"]), np.array(art["scaler_scale"])
    rep = ZoneExplainer().explain(p.feature_stack, r.kmeans_labels, C, mean, scale,
                                  fcm_m=art["fcm_m"], memberships=r.max_membership)

    print("1) PROTOTIPO DE CADA ZONA (unidades originales)")
    for z in rep["prototipos"]:
        print(f"  zona {z['cluster_id']}: NDVI {z['ndvi']:.3f}  NDRE {z['ndre']:.3f}  GNDVI {z['gndvi']:.3f}"
              f"  ({z['pct_pixeles']}% de los pixeles)")
    inc = rep["incertidumbre"]
    print(f"\n2) INCERTIDUMBRE: pertenencia maxima media {inc['pertenencia_max_media']}; "
          f"{inc['pct_pixeles_transicion']}% de pixeles de transicion (pertenencia < {inc['umbral_transicion']}); "
          f"minimo teorico {inc['minimo_teorico']}")

    print("\n3) TABLA DE ATRIBUTOS: pixeles reales")
    rows, cols = np.nonzero(p.valid_mask)
    xs, ys = p.transform * (cols + 0.5, rows + 0.5)
    X = p.feature_stack
    Z = (X - mean) / scale
    u = r.max_membership
    labels = r.kmeans_labels
    orden = [z["cluster_id"] for z in rep["prototipos"]]
    ejemplos = []
    for c in orden:
        m = np.nonzero((labels == c) & (u > 0.9))[0]
        if m.size:
            ejemplos.append(m[len(m) // 2])
    trans = np.nonzero((u > 0.45) & (u < 0.52))[0]
    if trans.size:
        ejemplos.append(trans[len(trans) // 2])
    print(f"{'fila':>5} {'col':>6} {'x_utm':>10} {'y_utm':>11} {'NDVI':>6} {'NDRE':>6} {'GNDVI':>6} | "
          + " ".join(f"{'u' + str(c):>5}" for c in orden) + " | zona")
    for i in ejemplos:
        d = np.linalg.norm(Z[i][None, :] - C, axis=1)
        inv = 1.0 / np.fmax(d, 1e-10) ** 2
        U = inv / inv.sum()
        print(f"{rows[i]:>5} {cols[i]:>6} {xs[i]:>10.1f} {ys[i]:>11.1f} {X[i, 0]:>6.3f} {X[i, 1]:>6.3f} {X[i, 2]:>6.3f} | "
              + " ".join(f"{U[c]:>5.2f}" for c in orden) + f" |  {labels[i]}")

    if trans.size:
        i = ejemplos[-1]
        e = local_attribution(Z[i], C)
        print(f"\n4) EXPLICACION LOCAL EXACTA del pixel de transicion (pertenencia {u[i]:.2f})")
        print(f"  zona asignada {e['zona_asignada']}, segunda opcion {e['segunda_zona']}; "
              f"ventaja total {e['ventaja_total']:+.3f} = suma de aportes")
        for f, v in e["aportes"].items():
            print(f"    {f:5}: {v:+.3f}  ({'favorece' if v > 0 else 'va en contra de'} la zona {e['zona_asignada']})")

    print("\n5) IMPORTANCIA GLOBAL POR PERMUTACION")
    for f, v in rep["importancia_permutacion"].items():
        print(f"  al desordenar {f:5} cambian de zona {v}% de los pixeles")

    sus = rep["modelo_sustituto"]
    print(f"\n6) ARBOL SUSTITUTO (profundidad {sus['profundidad']}): fidelidad {sus['fidelidad_pct']}% "
          f"| importancia " + ", ".join(f"{k} {v:.2f}" for k, v in sus["importancia"].items()))
    print(sus["reglas"])


if __name__ == "__main__":
    main()
