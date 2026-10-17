import numpy as np
import rasterio
from rasterio.crs import CRS
from dataclasses import dataclass, field
from pathlib import Path

# Orden por defecto en exportaciones de Pix4Dfields para DJI Mavic 3M (4 bandas + Alpha)
DEFAULT_MAVIC3M_BANDS = {
    "green":   1,
    "red":     2,
    "rededge": 3,
    "nir":     4,
    "alpha":   5,
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
    band_mapping: dict
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
    Clase para procesar el ortomosaico multiespectral (Mavic 3M / Pix4Dfields),
    detectar dinamicamente el orden de las bandas y calcular NDVI, NDRE y GNDVI.
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
            band_map = self._resolve_band_mapping(ds)
            metadata = self._extract_metadata(ds, band_map)
            transform = ds.transform
            bands = self._load_bands(ds, band_map)

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
        if ds.count < 4:
            raise ValueError(
                f"El archivo tiene {ds.count} bandas, pero necesitamos al menos las 4 bandas "
                "multiespectrales (Green, Red, RedEdge, NIR)."
            )
        if ds.crs is None:
            raise ValueError(
                "Error: El GeoTIFF no tiene sistema de coordenadas (CRS). Revisar exportacion de Pix4D."
            )
        if CRS.from_user_input(ds.crs) != CRS.from_user_input(EXPECTED_CRS):
            print(f"OJO: El CRS es {ds.crs} pero esperabamos {EXPECTED_CRS}. Puede salir descuadrado.")

    def _resolve_band_mapping(self, ds) -> dict:
        """
        Inspecciona las descripciones y tags del GeoTIFF (Pix4Dfields) para asignar
        correctamente Green, Red, RedEdge, NIR y Alpha sin confundir NIR con Alpha.
        """
        mapping = {}
        descriptions = list(ds.descriptions or [])

        for idx in range(1, ds.count + 1):
            desc = (descriptions[idx - 1] or "").strip().lower()
            if not desc:
                b_tags = ds.tags(idx)
                desc = (
                    b_tags.get("BAND_NAME", "")
                    or b_tags.get("DESCRIPTION", "")
                    or b_tags.get("Name", "")
                ).strip().lower()

            norm = desc.replace("_", "").replace("-", "").replace(" ", "")
            if "rededge" in norm or norm == "re":
                mapping["rededge"] = idx
            elif "nir" in norm or "nearinfrared" in norm:
                mapping["nir"] = idx
            elif "green" in norm:
                mapping["green"] = idx
            elif "red" in norm:
                mapping["red"] = idx
            elif "alpha" in norm or "mask" in norm:
                mapping["alpha"] = idx

        required = ("green", "red", "rededge", "nir")
        if all(k in mapping for k in required):
            return mapping

        # Fallback seguro para Mavic 3M exportado de Pix4D (Green=1, Red=2, RedEdge=3, NIR=4, Alpha=5)
        fallback = {
            k: v for k, v in DEFAULT_MAVIC3M_BANDS.items() if v <= ds.count
        }
        return fallback

    def _extract_metadata(self, ds, band_map: dict):
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
            band_mapping=band_map,
            dtype=ds.dtypes[0],
            source_software=tags.get("TIFFTAG_SOFTWARE", "Pix4Dfields"),
            capture_datetime=tags.get("TIFFTAG_DATETIME", "desconocida"),
        )

    def _load_bands(self, ds, band_map: dict):
        raw = {}
        # Si existe banda Alpha, la cargamos para enmascarar los bordes vacios del ortomosaico
        alpha_mask = None
        if "alpha" in band_map:
            alpha_data = ds.read(band_map["alpha"]).astype(np.float32)
            alpha_mask = alpha_data > 0

        for name in ("green", "red", "rededge", "nir"):
            idx = band_map[name]
            data = ds.read(idx).astype(np.float32)
            if ds.nodata is not None:
                data[data == ds.nodata] = np.nan
            if alpha_mask is not None:
                data[~alpha_mask] = np.nan

            max_val = np.nanmax(data)
            if max_val > 1.0:
                # Normalizacion radiometrica (16-bit uint -> reflectancia [0, 1])
                divisor = 65535.0 if max_val > 255.0 else 255.0
                data = data / divisor
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