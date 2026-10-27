from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import ColorInterp

# Orden en que Pix4Dfields exporta el Mavic 3M cuando el tif no trae nombres de banda
DEFAULT_MAVIC3M_BANDS = {
    "green": 1,
    "red": 2,
    "rededge": 3,
    "nir": 4,
    "alpha": 5,
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
    """Lee el ortomosaico del Mavic 3M y calcula NDVI, NDRE y GNDVI."""

    def __init__(self, tif_path, vegetation_threshold=0.1):
        self.path = Path(tif_path)
        self.vegetation_threshold = vegetation_threshold
        if not self.path.exists():
            raise ValueError(f"Archivo no encontrado: {self.path}")
        if self.path.suffix.lower() not in (".tif", ".tiff"):
            raise ValueError("El archivo debe ser .tif o .tiff")

    def process(self):
        with rasterio.open(self.path) as ds:
            self._validate(ds)
            band_map = self._resolve_band_mapping(ds)
            metadata = self._extract_metadata(ds, band_map)
            transform = ds.transform
            bands = self._load_bands(ds, band_map)

        indices = self._calculate_indices(bands)
        valid_mask = self._build_valid_mask(bands, indices)
        if not valid_mask.any():
            raise ValueError(
                f"Ningun pixel supera el umbral de NDVI {self.vegetation_threshold}. "
                "Revisar el orden de bandas o el umbral."
            )

        return ProcessedRaster(
            ndvi=indices["ndvi"],
            ndre=indices["ndre"],
            gndvi=indices["gndvi"],
            valid_mask=valid_mask,
            feature_stack=self._build_feature_stack(indices, valid_mask),
            spatial_coords=self._extract_coordinates(valid_mask, transform),
            transform=transform,
            metadata=metadata,
            band_stats=self._compute_stats(indices, valid_mask),
        )

    def _validate(self, ds):
        if ds.count < 4:
            raise ValueError(
                f"El archivo tiene {ds.count} bandas y se necesitan al menos 4 "
                "(Green, Red, RedEdge, NIR)."
            )
        if ds.crs is None:
            raise ValueError("El GeoTIFF no tiene CRS. Revisar la exportacion de Pix4D.")
        if CRS.from_user_input(ds.crs) != CRS.from_user_input(EXPECTED_CRS):
            print(f"Aviso: el CRS es {ds.crs} y se esperaba {EXPECTED_CRS}.")

    def _resolve_band_mapping(self, ds):
        mapping = {}
        descriptions = list(ds.descriptions or [])

        for idx in range(1, ds.count + 1):
            if ds.colorinterp[idx - 1] == ColorInterp.alpha:
                mapping["alpha"] = idx
                continue

            desc = (descriptions[idx - 1] or "").strip().lower()
            if not desc:
                tags = ds.tags(idx)
                desc = (
                    tags.get("BAND_NAME", "")
                    or tags.get("DESCRIPTION", "")
                    or tags.get("Name", "")
                ).strip().lower()

            name = desc.replace("_", "").replace("-", "").replace(" ", "")
            # el orden importa: "rededge" e "infrared" tambien contienen "red"
            if "rededge" in name or name == "re":
                mapping["rededge"] = idx
            elif "nir" in name or "infrared" in name:
                mapping["nir"] = idx
            elif "green" in name:
                mapping["green"] = idx
            elif "red" in name:
                mapping["red"] = idx
            elif "alpha" in name or "mask" in name:
                mapping["alpha"] = idx

        if all(k in mapping for k in ("green", "red", "rededge", "nir")):
            return mapping

        return {k: v for k, v in DEFAULT_MAVIC3M_BANDS.items() if v <= ds.count}

    def _extract_metadata(self, ds, band_map):
        b = ds.bounds
        tags = ds.tags()
        return RasterMetadata(
            crs=str(ds.crs),
            resolution_m=round(ds.res[0], 4),
            width_px=ds.width,
            height_px=ds.height,
            bounds={
                "left": round(b.left, 2),
                "bottom": round(b.bottom, 2),
                "right": round(b.right, 2),
                "top": round(b.top, 2),
            },
            area_ha=round((b.right - b.left) * (b.top - b.bottom) / 10_000, 4),
            band_count=ds.count,
            band_mapping=band_map,
            dtype=ds.dtypes[0],
            source_software=tags.get("TIFFTAG_SOFTWARE", "Pix4Dfields"),
            capture_datetime=tags.get("TIFFTAG_DATETIME", "desconocida"),
        )

    def _load_bands(self, ds, band_map):
        alpha_mask = None
        if "alpha" in band_map:
            alpha_mask = ds.read(band_map["alpha"]) > 0

        # Mismo divisor para todas las bandas: si cada banda se escalara por su
        # propio maximo, los indices normalizados quedarian distorsionados.
        dtype = np.dtype(ds.dtypes[band_map["nir"] - 1])
        divisor = float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1.0

        bands = {}
        for name in ("green", "red", "rededge", "nir"):
            data = ds.read(band_map[name]).astype(np.float32)
            if ds.nodata is not None:
                data[data == ds.nodata] = np.nan
            if alpha_mask is not None:
                data[~alpha_mask] = np.nan
            bands[name] = data / divisor
        return bands

    def _calculate_indices(self, bands):
        nir = bands["nir"]
        return {
            "ndvi": self._safe_ratio(nir - bands["red"], nir + bands["red"]),
            "ndre": self._safe_ratio(nir - bands["rededge"], nir + bands["rededge"]),
            "gndvi": self._safe_ratio(nir - bands["green"], nir + bands["green"]),
        }

    @staticmethod
    def _safe_ratio(numerator, denominator):
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(denominator == 0, np.nan, numerator / denominator)
        return result.astype(np.float32)

    def _build_valid_mask(self, bands, indices):
        mask = np.ones(bands["nir"].shape, dtype=bool)
        for arr in list(bands.values()) + list(indices.values()):
            mask &= ~np.isnan(arr)
        mask &= indices["ndvi"] >= self.vegetation_threshold
        return mask

    @staticmethod
    def _build_feature_stack(indices, valid_mask):
        return np.stack(
            [indices["ndvi"][valid_mask], indices["ndre"][valid_mask], indices["gndvi"][valid_mask]],
            axis=1,
        ).astype(np.float32)

    @staticmethod
    def _extract_coordinates(valid_mask, transform):
        # Coordenadas en metros relativas a la esquina del raster. Sin el offset UTM
        # (~500 km) caben en float32 sin perder precision.
        rows, cols = np.where(valid_mask)
        xs = (cols.astype(np.float32) + 0.5) * np.float32(transform.a)
        ys = (rows.astype(np.float32) + 0.5) * np.float32(transform.e)
        return np.column_stack((xs, ys))

    @staticmethod
    def _compute_stats(indices, valid_mask):
        stats = {}
        for name, arr in indices.items():
            values = arr[valid_mask]
            stats[name] = {
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "median": float(np.median(values)),
            }
        return stats