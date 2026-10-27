import warnings

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform
from scipy import ndimage
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree


def _block_mean(a, f):
    if f == 1:
        return a.astype(np.float32)
    h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a[:h, :w].reshape(h // f, f, w // f, f), axis=(1, 3)).astype(np.float32)


def _disk(radius_px):
    r = max(1, int(round(radius_px)))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y <= r * r).astype(np.float32)


class PalmAnalyzer:
    """
    Detecta copas de palma como manchas brillantes en el infrarrojo cercano (diferencia de
    gaussianas a la escala de la copa), calcula NDVI y NDRE por copa y marca para inspeccion
    las palmas con NDVI <= ndvi_alert o claramente inferior a sus vecinas. Las copas de bajo
    contraste dentro de la malla de siembra se reportan para verificacion (jovenes o enfermas),
    y los espacios de la malla sin copa detectada como posibles faltantes.
    """

    def __init__(self, spacing_m=9.0, ndvi_alert=0.5, grid_m=0.30, relative_k=3.0, relative_min_drop=0.10,
                 neighbor_radius_m=30.0, strong_contrast=1.5, weak_contrast=0.5):
        self.spacing_m = spacing_m
        self.ndvi_alert = ndvi_alert
        self.grid_m = grid_m
        self.relative_k = relative_k
        self.relative_min_drop = relative_min_drop
        self.neighbor_radius_m = neighbor_radius_m
        self.strong_contrast = strong_contrast
        self.weak_contrast = weak_contrast

    def analyze(self, tif_path, processed):
        meta = processed.metadata
        f = max(1, int(round(self.grid_m / meta.resolution_m)))
        g = f * meta.resolution_m
        s = self.spacing_m

        with rasterio.open(tif_path) as ds:
            nir = ds.read(meta.band_mapping["nir"]).astype(np.float32)
        nir[~np.isfinite(processed.ndvi)] = np.nan
        nir_g = _block_mean(nir, f)
        del nir
        ndvi_g = _block_mean(processed.ndvi, f)
        ndre_g = _block_mean(processed.ndre, f)
        data_g = np.isfinite(nir_g) & np.isfinite(ndvi_g)
        if data_g.sum() == 0:
            raise ValueError("No hay datos suficientes para el analisis por palma.")

        crown = _disk(0.25 * s / g)
        data_frac = self._disk_mean(data_g.astype(np.float32), data_g | True, crown)
        rows, cols, contrast = self._detect_crowns(nir_g, data_g, data_frac, g)
        xs, ys = self._to_utm(processed.transform, rows, cols, f)
        keep, weak = self._filter_by_lattice(np.column_stack([xs, ys]), contrast)
        rows, cols, xs, ys, weak = rows[keep], cols[keep], xs[keep], ys[keep], weak[keep]

        ndvi_c = self._disk_mean(ndvi_g, data_g, crown)[rows, cols]
        ndre_c = self._disk_mean(ndre_g, data_g, crown)[rows, cols]
        lons, lats = self._to_wgs84(meta.crs, xs, ys)
        motivos = self._flag(np.column_stack([xs, ys]), ndvi_c)

        palm_mask = np.zeros_like(data_g)
        palm_mask[rows, cols] = True
        dist = ndimage.distance_transform_edt(~palm_mask) * g
        area_ha = float(((dist <= 0.6 * s) & data_g).sum() * g * g / 10_000)
        missing_rc = self._missing(dist, data_g, data_frac, np.column_stack([xs, ys]), processed.transform, f, g)

        palms = []
        for i in range(len(rows)):
            if motivos[i]:
                estado, motivo = "inspeccionar", "; ".join(motivos[i])
            elif weak[i]:
                estado, motivo = "verificar_copa", "Copa pequena o poco definida: palma joven, resembrada o enferma"
            else:
                estado, motivo = "normal", ""
            palms.append({
                "palma_id": i + 1,
                "lat": round(float(lats[i]), 7),
                "lon": round(float(lons[i]), 7),
                "x_utm": round(float(xs[i]), 2),
                "y_utm": round(float(ys[i]), 2),
                "ndvi_copa": round(float(ndvi_c[i]), 3),
                "ndre_copa": round(float(ndre_c[i]), 3),
                "estado": estado,
                "motivo": motivo,
            })
        if missing_rc:
            mr, mc = np.array(missing_rc).T
            mxs, mys = self._to_utm(processed.transform, mr, mc, f)
            mlons, mlats = self._to_wgs84(meta.crs, mxs, mys)
        faltantes = [
            {"faltante_id": i + 1, "lat": round(float(mlats[i]), 7), "lon": round(float(mlons[i]), 7),
             "x_utm": round(float(mxs[i]), 2), "y_utm": round(float(mys[i]), 2)}
            for i in range(len(missing_rc))
        ]

        n = len(palms)
        inspeccionar = sorted([p for p in palms if p["estado"] == "inspeccionar"], key=lambda p: p["ndvi_copa"])
        verificar = [p for p in palms if p["estado"] == "verificar_copa"]
        valid_ndvi = ndvi_c[np.isfinite(ndvi_c)]
        report = {
            "conteo_palmas": n,
            "area_plantacion_ha": round(area_ha, 3),
            "densidad_palmas_ha": round(n / area_ha, 1) if area_ha > 0 else None,
            "densidad_esperada_palmas_ha": round(2 / (np.sqrt(3) * s * s) * 10_000, 1),
            "ndvi_copa": {
                "media": round(float(valid_ndvi.mean()), 3) if valid_ndvi.size else None,
                "mediana": round(float(np.median(valid_ndvi)), 3) if valid_ndvi.size else None,
                "p5": round(float(np.percentile(valid_ndvi, 5)), 3) if valid_ndvi.size else None,
                "p95": round(float(np.percentile(valid_ndvi, 95)), 3) if valid_ndvi.size else None,
            },
            "palmas_a_inspeccionar": len(inspeccionar),
            "pct_palmas_a_inspeccionar": round(len(inspeccionar) * 100.0 / n, 2) if n else 0.0,
            "palmas_copa_debil": len(verificar),
            "pct_palmas_copa_debil": round(len(verificar) * 100.0 / n, 2) if n else 0.0,
            "posibles_faltantes": len(faltantes),
            "parametros": {
                "distancia_siembra_m": s,
                "ndvi_alerta_copa": self.ndvi_alert,
                "umbral_relativo_mad": self.relative_k,
                "caida_minima_vs_vecinas": self.relative_min_drop,
                "radio_vecindad_m": self.neighbor_radius_m,
                "contraste_copa_fuerte": self.strong_contrast,
                "contraste_copa_debil": self.weak_contrast,
                "resolucion_analisis_m": round(g, 3),
            },
            "lista_inspeccion": inspeccionar,
            "lista_copa_debil": verificar,
            "lista_faltantes": faltantes,
        }
        return report, self._geojson(meta.crs, palms, faltantes)

    def _detect_crowns(self, nir_g, data_g, data_frac, g):
        s = self.spacing_m
        v = np.where(data_g, nir_g, np.nanmedian(nir_g[data_g]))
        dog = ndimage.gaussian_filter(v, 0.11 * s / g) - ndimage.gaussian_filter(v, 0.39 * s / g)
        dog /= max(float(np.std(dog[data_g])), 1e-9)
        sep = _disk(0.55 * s / g / 2).astype(bool)
        peaks = ((dog == ndimage.maximum_filter(dog, footprint=sep)) & (dog > self.weak_contrast)
                 & data_g & (data_frac >= 0.6))
        rows, cols = np.nonzero(peaks)
        keep = self._non_max_suppression(rows, cols, dog[rows, cols], 0.55 * s / g / 2)
        return rows[keep], cols[keep], dog[rows[keep], cols[keep]]

    @staticmethod
    def _non_max_suppression(rows, cols, values, min_dist):
        # Dos maximos con el mismo valor (copa centrada entre dos celdas) pasan el filtro de maximos;
        # se conserva solo el pico mas fuerte dentro de la distancia minima entre copas.
        if len(rows) == 0:
            return np.zeros(0, dtype=bool)
        pts = np.column_stack([rows, cols]).astype(float)
        tree = cKDTree(pts)
        keep = np.zeros(len(rows), dtype=bool)
        suppressed = np.zeros(len(rows), dtype=bool)
        for i in np.argsort(-values, kind="stable"):
            if suppressed[i]:
                continue
            keep[i] = True
            suppressed[tree.query_ball_point(pts[i], min_dist)] = True
        return keep

    def _filter_by_lattice(self, xy, contrast):
        """
        Las copas de contraste alto se aceptan como palmas. Las de contraste bajo solo si estan
        dentro de la malla de siembra (>= 3 palmas fuertes a la distancia de siembra): asi se
        conservan palmas jovenes o con copa cerrada y se descartan parches de pasto o lindero.
        """
        s = self.spacing_m
        strong = contrast >= self.strong_contrast
        weak = ~strong
        keep = strong.copy()
        if strong.any() and weak.any():
            tree = cKDTree(xy[strong])
            for i in np.nonzero(weak)[0]:
                near = tree.query_ball_point(xy[i], 1.3 * s)
                d = np.linalg.norm(xy[strong][near] - xy[i], axis=1) if near else np.array([])
                keep[i] = int(((d >= 0.7 * s) & (d <= 1.3 * s)).sum()) >= 3
        return keep, weak

    @staticmethod
    def _disk_mean(values, valid, kernel):
        filled = np.where(valid & np.isfinite(values), values, 0.0).astype(np.float32)
        w = (valid & np.isfinite(values)).astype(np.float32)
        num = fftconvolve(filled, kernel, mode="same")
        den = fftconvolve(w, kernel, mode="same")
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 1e-3, num / np.maximum(den, 1e-3), np.nan).astype(np.float32)

    def _flag(self, xy, ndvi_c):
        motivos = [[] for _ in range(len(ndvi_c))]
        if len(ndvi_c) == 0:
            return motivos
        tree = cKDTree(xy)
        neigh = tree.query_ball_point(xy, self.neighbor_radius_m)
        med = np.array([
            np.nanmedian(ndvi_c[[j for j in nb if j != i]]) if len(nb) > 1 else np.nan
            for i, nb in enumerate(neigh)
        ])
        resid = ndvi_c - med
        ok = np.isfinite(resid)
        mad = 1.4826 * float(np.median(np.abs(resid[ok] - np.median(resid[ok])))) if ok.any() else 0.0
        for i, v in enumerate(ndvi_c):
            if np.isfinite(v) and v <= self.ndvi_alert:
                motivos[i].append(f"NDVI de copa <= {self.ndvi_alert}")
            if ok[i] and resid[i] <= -max(self.relative_k * mad, self.relative_min_drop):
                motivos[i].append(f"NDVI {abs(resid[i]):.2f} por debajo de la mediana de sus vecinas")
        return motivos

    def _missing(self, dist, data_g, data_frac, palm_xy, transform, f, g):
        s = self.spacing_m
        if len(palm_xy) < 7:
            return []
        local_max = dist == ndimage.maximum_filter(dist, footprint=_disk(0.4 * s / g).astype(bool))
        cand = np.argwhere(local_max & (dist >= 0.8 * s) & data_g & (data_frac >= 0.9))
        if cand.size == 0:
            return []
        cx, cy = self._to_utm(transform, cand[:, 0], cand[:, 1], f)
        cxy = np.column_stack([cx, cy])
        tree = cKDTree(palm_xy)
        kept, kept_xy = [], []
        for k, nb in enumerate(tree.query_ball_point(cxy, 1.6 * s)):
            # En malla triangular una palma faltante tiene 6 vecinas a la distancia de siembra; en una
            # via o fila completa sin palmas solo quedan ~4, por eso se exigen al menos 5.
            d = np.linalg.norm(palm_xy[nb] - cxy[k], axis=1) if nb else np.array([])
            if int(((d >= 0.7 * s) & (d <= 1.3 * s)).sum()) < 5:
                continue
            ang = np.sort(np.arctan2(palm_xy[nb, 1] - cxy[k, 1], palm_xy[nb, 0] - cxy[k, 0]))
            gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * np.pi]]))
            if gaps.max() >= np.deg2rad(150):
                continue
            if kept_xy and np.min(np.linalg.norm(np.array(kept_xy) - cxy[k], axis=1)) < 0.5 * s:
                continue
            kept.append(tuple(cand[k]))
            kept_xy.append(cxy[k])
        return kept

    @staticmethod
    def _to_utm(transform, rows, cols, f):
        return transform * (np.asarray(cols) * f + f / 2.0, np.asarray(rows) * f + f / 2.0)

    @staticmethod
    def _to_wgs84(crs, xs, ys):
        if len(xs) == 0:
            return [], []
        return warp_transform(crs, "EPSG:4326", list(xs), list(ys))

    @staticmethod
    def _geojson(crs, palms, faltantes):
        feats = [{
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [p["x_utm"], p["y_utm"]]},
            "properties": {k: v for k, v in p.items() if k not in ("x_utm", "y_utm")},
        } for p in palms]
        feats += [{
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [m["x_utm"], m["y_utm"]]},
            "properties": {"faltante_id": m["faltante_id"], "lat": m["lat"], "lon": m["lon"],
                           "estado": "posible_faltante",
                           "motivo": "Espacio de siembra sin copa detectada: palma faltante, muy joven o enferma"},
        } for m in faltantes]
        return {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": crs}}, "features": feats}
