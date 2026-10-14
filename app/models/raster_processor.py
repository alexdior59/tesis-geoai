import numpy as np
import rasterio
from rasterio.crs import CRS
from dataclasses import dataclass, field
from pathlib import Path

BAND_INDEX = {
    "blue":     1,
    "green":    2,
    "red":      3,
    "rededge":  4,
    "nir":      5,
}

EXPECTED_CRS = "EPSG:32618"

@dataclass
class RasterMetadata:
    crs: str
    resolution_m: float
    width_px: int
    height_px: int
    bounds: dict
    area_ha: float
    band_count: int
    dtype: str
    source_software: str
    capture_datetime: str

@dataclass
class ProcessedRaster:
    ndvi: np.ndarray
    ndre: np.ndarray
    gndvi: np.ndarray
    valid_mask: np.ndarray
    feature_stack: np.ndarray
    spatial_coords: np.ndarray
    transform: rasterio.Affine
    metadata: RasterMetadata
    band_stats: dict = field(default_factory=dict)

class RasterProcessor:
    """
    Clase para procesar la imagen, sacar metadatos y calcular los indices.
    """
    VEGETATION_THRESHOLD = 0.1

    def __init__(self, tif_path):
        self.path = Path(tif_path)
        if not self.path.exists():
            raise ValueError(f"Archivo no encontrado: {self.path}")
        if self.path.suffix.lower() not in (".tif", ".tiff"):
            raise ValueError("Ojo, tiene que ser un archivo .tif o .tiff")

    def process(self):
        with rasterio.open(self.path) as ds:
            self._validate(ds)
            metadata = self._extract_metadata(ds)
            transform = ds.transform
            bands = self._load_bands(ds)

        indices = self._calculate_indices(bands)
        valid_mask = self._build_valid_mask(bands, indices)
        feature_stack = self._build_feature_stack(indices, valid_mask)
        spatial_coords = self._extract_coordinates(valid_mask, transform)
        band_stats = self._compute_stats(indices, valid_mask)

        return ProcessedRaster(
            ndvi=indices["ndvi"],
            ndre=indices["ndre"],
            gndvi=indices["gndvi"],
            valid_mask=valid_mask,
            feature_stack=feature_stack,
            spatial_coords=spatial_coords,
            transform=transform,
            metadata=metadata,
            band_stats=band_stats,
        )

    def _validate(self, ds):
        if ds.count < 5:
            raise ValueError(
                f"El archivo tiene {ds.count} bandas, pero necesitamos 5 (Blue/Green/Red/RedEdge/NIR)."
            )
        if ds.crs is None:
            raise ValueError(
                "Error: El GeoTIFF no tiene sistema de coordenadas (CRS). Revisar si es el exportado de Pix4D."
            )
        if CRS.from_user_input(ds.crs) != CRS.from_user_input(EXPECTED_CRS):
            print(f"OJO: El CRS es {ds.crs} pero esperabamos {EXPECTED_CRS}. Puede salir descuadrado.")

    def _extract_metadata(self, ds):
        res_x, res_y = ds.res
        b = ds.bounds
        area_m2 = (b.right - b.left) * (b.top - b.bottom)
        tags = ds.tags()
        return RasterMetadata(
            crs=str(ds.crs),
            resolution_m=round(res_x, 4),
            width_px=ds.width,
            height_px=ds.height,
            bounds={
                "left": round(b.left, 2),
                "bottom": round(b.bottom, 2),
                "right": round(b.right, 2),
                "top": round(b.top, 2),
            },
            area_ha=round(area_m2 / 10_000, 4),
            band_count=ds.count,
            dtype=ds.dtypes[0],
            source_software=tags.get("TIFFTAG_SOFTWARE", "desconocido"),
            capture_datetime=tags.get("TIFFTAG_DATETIME", "desconocida"),
        )

    def _load_bands(self, ds):
        raw = {}
        for name, idx in BAND_INDEX.items():
            data = ds.read(idx).astype(np.float32)
            if ds.nodata is not None:
                data[data == ds.nodata] = np.nan
            if np.nanmax(data) > 1.0:
                data = data / 65535.0
            raw[name] = data
        return raw

    def _calculate_indices(self, bands):
        nir = bands["nir"]
        red = bands["red"]
        green = bands["green"]
        rededge = bands["rededge"]
        ndvi = self._safe_ratio(nir - red, nir + red)
        ndre = self._safe_ratio(nir - rededge, nir + rededge)
        gndvi = self._safe_ratio(nir - green, nir + green)
        return {"ndvi": ndvi, "ndre": ndre, "gndvi": gndvi}

    @staticmethod
    def _safe_ratio(numerator, denominator):
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(
                (denominator == 0) | np.isnan(denominator),
                np.nan,
                numerator / denominator,
            )
        return result.astype(np.float32)

    def _build_valid_mask(self, bands, indices):
        mask = np.ones(bands["nir"].shape, dtype=bool)
        for arr in bands.values():
            mask &= ~np.isnan(arr)
        for arr in indices.values():
            mask &= ~np.isnan(arr)
        mask &= (indices["ndvi"] >= self.VEGETATION_THRESHOLD)
        return mask

    def _build_feature_stack(self, indices, valid_mask):
        features = np.stack([
            indices["ndvi"][valid_mask],
            indices["ndre"][valid_mask],
            indices["gndvi"][valid_mask],
        ], axis=1)
        return features.astype(np.float32)

    def _extract_coordinates(self, valid_mask, transform):
        """
        Calcula coordenadas relativas en metros (X, Y) de forma vectorizada.
        Restamos el origen (transform.c, transform.f) para mantener precision submilimetrica en float32.
        """
        rows, cols = np.where(valid_mask)
        xs = (cols.astype(np.float32) + 0.5) * float(transform.a)
        ys = (rows.astype(np.float32) + 0.5) * float(transform.e)
        return np.column_stack((xs, ys)).astype(np.float32)

    def _compute_stats(self, indices, valid_mask):
        stats = {}
        for name, arr in indices.items():
            valid_vals = arr[valid_mask]
            stats[name] = {
                "min":    float(np.nanmin(valid_vals)),
                "max":    float(np.nanmax(valid_vals)),
                "mean":   float(np.nanmean(valid_vals)),
                "std":    float(np.nanstd(valid_vals)),
                "median": float(np.nanmedian(valid_vals)),
            }
        return stats