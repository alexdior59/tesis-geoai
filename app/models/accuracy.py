import numpy as np
from rasterio.features import sieve
from rasterio.warp import transform as warp_transform


def stratified_sample(processed, labels, n_per_zone=30, min_distance_m=10.0, mmu_px=100, seed=42,
                      max_candidates=20_000):
    """
    Puntos aleatorios estratificados por zona sobre el mapa entregado (ya generalizado con la unidad
    minima de mapeo). La distancia minima entre puntos reduce la autocorrelacion espacial.
    Los identificadores se barajan para que no revelen la zona (evaluacion a ciegas).
    """
    rng = np.random.default_rng(seed)
    full = np.full(processed.valid_mask.shape, -1, dtype=np.int16)
    full[processed.valid_mask] = labels
    mapa = sieve(full, size=mmu_px)
    mapa[~processed.valid_mask] = -1
    res = processed.metadata.resolution_m
    min_px = min_distance_m / res

    puntos = []
    for zona in sorted(int(z) for z in np.unique(mapa) if z >= 0):
        idx = np.flatnonzero(mapa == zona)
        if idx.size > max_candidates:
            idx = rng.choice(idx, size=max_candidates, replace=False)
        else:
            idx = rng.permutation(idx)
        rows, cols = np.unravel_index(idx, mapa.shape)
        elegidos = []
        for r, c in zip(rows, cols):
            if elegidos and np.min(np.hypot(np.array([e[0] for e in elegidos]) - r,
                                            np.array([e[1] for e in elegidos]) - c)) < min_px:
                continue
            if puntos and np.min(np.hypot(np.array([p["fila"] for p in puntos]) - r,
                                          np.array([p["columna"] for p in puntos]) - c)) < min_px:
                continue
            elegidos.append((int(r), int(c)))
            if len(elegidos) == n_per_zone:
                break
        for r, c in elegidos:
            puntos.append({"zona": zona, "fila": r, "columna": c})

    if not puntos:
        return []
    rows = np.array([p["fila"] for p in puntos])
    cols = np.array([p["columna"] for p in puntos])
    xs, ys = processed.transform * (cols + 0.5, rows + 0.5)
    lons, lats = warp_transform(processed.metadata.crs, "EPSG:4326", list(xs), list(ys))
    orden = rng.permutation(len(puntos))
    salida = []
    for nuevo_id, i in enumerate(orden, start=1):
        salida.append({
            "punto_id": nuevo_id,
            "zona": puntos[i]["zona"],
            "x_utm": round(float(xs[i]), 2),
            "y_utm": round(float(ys[i]), 2),
            "lat": round(float(lats[i]), 7),
            "lon": round(float(lons[i]), 7),
        })
    return salida


def blind_geojson(puntos, crs):
    """GeoJSON de puntos de verificacion sin la zona del modelo, para el experto."""
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs}},
        "features": [{
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [p["x_utm"], p["y_utm"]]},
            "properties": {"punto_id": p["punto_id"], "lat": p["lat"], "lon": p["lon"]},
        } for p in sorted(puntos, key=lambda p: p["punto_id"])],
    }


def accuracy_assessment(map_classes, reference_classes):
    """
    Matriz de confusion con filas = clase del mapa y columnas = clase de referencia (Congalton, 1991).
    Exactitud del usuario = acierto / total de la fila; del productor = acierto / total de la columna.
    """
    if len(map_classes) != len(reference_classes) or not map_classes:
        raise ValueError("Se necesitan listas no vacias y del mismo largo de clases del mapa y de referencia.")
    clases = sorted(set(map_classes) | set(reference_classes), key=str)
    pos = {c: i for i, c in enumerate(clases)}
    cm = np.zeros((len(clases), len(clases)), dtype=np.int64)
    for m, r in zip(map_classes, reference_classes):
        cm[pos[m], pos[r]] += 1
    n = cm.sum()
    filas, columnas = cm.sum(axis=1), cm.sum(axis=0)
    po = np.trace(cm) / n
    pe = float((filas * columnas).sum()) / n ** 2
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    por_clase = {}
    for c, i in pos.items():
        por_clase[str(c)] = {
            "exactitud_usuario": round(float(cm[i, i] / filas[i]), 4) if filas[i] else None,
            "exactitud_productor": round(float(cm[i, i] / columnas[i]), 4) if columnas[i] else None,
            "n_mapa": int(filas[i]),
            "n_referencia": int(columnas[i]),
        }
    return {
        "clases": [str(c) for c in clases],
        "matriz_confusion": cm.tolist(),
        "orientacion": "filas = clase del mapa, columnas = clase de referencia",
        "n_puntos": int(n),
        "exactitud_global": round(float(po), 4),
        "kappa": round(float(kappa), 4),
        "por_clase": por_clase,
    }
