import numpy as np
from rasterio.features import shapes, sieve
from rasterio.warp import transform as warp_transform
from scipy import ndimage
from shapely.geometry import mapping, shape

NOTA_INSPECCION = (
    "Una zona de inspeccion no implica dano: puede corresponder a estres, madurez o senescencia "
    "del cultivo. El sistema reporta el area; la decision de intervenir corresponde al productor."
)
MAX_ZONAS_EN_JSON = 50


class ConditionReporter:
    """
    Clasifica cada pixel en los rangos agronomicos de NDVI del perfil del cultivo,
    reporta area y porcentaje por rango y georreferencia las zonas de inspeccion.
    """

    def __init__(self, profile: dict, crop_type: str, min_zone_m2: float = 10.0):
        self.profile = profile
        self.crop_type = crop_type
        self.min_zone_m2 = min_zone_m2

    def build(self, processed):
        ndvi = processed.ndvi
        data = np.isfinite(ndvi)
        res = processed.metadata.resolution_m
        px_m2 = res * res
        total_px = int(data.sum())
        if total_px == 0:
            raise ValueError("La imagen no tiene pixeles con datos para calcular los rangos de NDVI.")

        rangos = []
        for r in self.profile["ndvi_ranges"]:
            n = int((data & (ndvi >= r["min"]) & (ndvi < r["max"])).sum())
            rangos.append({
                "clave": r["clave"],
                "descripcion": r["descripcion"],
                "ndvi_min": r["min"],
                "ndvi_max": min(r["max"], 1.0),
                "area_ha": round(n * px_m2 / 10_000, 4),
                "pct_area": round(n * 100.0 / total_px, 2),
            })

        lo, hi = self.profile["inspection_range"]
        en_rango = data & (ndvi >= lo) & (ndvi < hi)
        n_en_rango = int(en_rango.sum())
        min_px = max(2, int(round(self.min_zone_m2 / px_m2)))
        zonas_mask = sieve(en_rango.astype(np.uint8), size=min_px).astype(bool) & data
        del en_rango

        zonas, geojson = self._zones(zonas_mask, ndvi, processed, px_m2)
        area_zonas_ha = round(sum(z["area_m2"] for z in zonas) / 10_000, 4)

        report = {
            "cultivo": self.crop_type,
            "cultivo_nombre": self.profile["nombre"],
            "fuente_umbrales": self.profile["fuente"],
            "area_con_datos_ha": round(total_px * px_m2 / 10_000, 4),
            "rangos": rangos,
            "inspeccion": {
                "rango_ndvi": [lo, hi],
                "area_en_rango_ha": round(n_en_rango * px_m2 / 10_000, 4),
                "pct_area_en_rango": round(n_en_rango * 100.0 / total_px, 2),
                "zona_minima_m2": self.min_zone_m2,
                "n_zonas": len(zonas),
                "area_zonas_ha": area_zonas_ha,
                "zonas_principales": zonas[:MAX_ZONAS_EN_JSON],
            },
            "nota": NOTA_INSPECCION,
        }
        return report, geojson

    def _zones(self, mask, ndvi, processed, px_m2):
        crs = processed.metadata.crs
        transform = processed.transform
        empty = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": crs}}, "features": []}
        labels, n = ndimage.label(mask)
        if n == 0:
            return [], empty

        idx = np.arange(1, n + 1)
        counts = np.bincount(labels.ravel(), minlength=n + 1)[1:]
        ndvi_means = ndimage.mean(np.nan_to_num(ndvi), labels, idx)
        centers = np.array(ndimage.center_of_mass(mask, labels, idx))
        xs, ys = transform * (centers[:, 1] + 0.5, centers[:, 0] + 0.5)
        lons, lats = warp_transform(crs, "EPSG:4326", list(xs), list(ys))

        order = np.argsort(-counts)
        rank = {int(idx[i]): k + 1 for k, i in enumerate(order)}
        zonas = []
        for k, i in enumerate(order, start=1):
            zonas.append({
                "zona_id": k,
                "area_m2": round(float(counts[i] * px_m2), 1),
                "ndvi_medio": round(float(ndvi_means[i]), 3),
                "lat": round(float(lats[i]), 7),
                "lon": round(float(lons[i]), 7),
                "x_utm": round(float(xs[i]), 2),
                "y_utm": round(float(ys[i]), 2),
            })

        by_id = {z["zona_id"]: z for z in zonas}
        features = []
        for geom, val in shapes(labels.astype(np.int32), mask=mask, transform=transform):
            z = by_id[rank[int(val)]]
            features.append({
                "type": "Feature",
                "geometry": mapping(shape(geom).simplify(0.2, preserve_topology=True)),
                "properties": {**z, "tipo": "zona_de_inspeccion"},
            })
        geojson = {**empty, "features": features}
        return zonas, geojson
