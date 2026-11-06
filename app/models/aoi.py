"""Poligono del lote: limita el analisis al area sembrada."""
import io

import geopandas as gpd
from rasterio.features import geometry_mask


def read_aoi(source):
    """
    Lee el poligono del lote desde una ruta, bytes (p. ej. un GeoJSON subido) o el dict que
    devuelve aoi_to_dict. Devuelve un GeoDataFrame con geometrias validas y CRS definido.
    """
    if isinstance(source, dict):
        gdf = gpd.GeoDataFrame.from_features(source["geojson"]["features"], crs=source["crs"])
    else:
        try:
            gdf = gpd.read_file(io.BytesIO(source) if isinstance(source, bytes) else source)
        except Exception as e:
            raise ValueError(f"No se pudo leer el poligono del lote: {e}")

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if gdf.empty:
        raise ValueError("El archivo del lote no tiene geometrias.")
    if not gdf.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError("El lote debe estar formado por poligonos.")
    if gdf.crs is None:
        raise ValueError("El poligono del lote no tiene sistema de coordenadas.")
    gdf = gdf.set_geometry(gdf.geometry.make_valid())
    return gdf[["geometry"]]


def aoi_to_dict(gdf):
    """Forma serializable para guardar el lote en el job y en la linea base."""
    return {"crs": gdf.crs.to_string(), "geojson": gdf.__geo_interface__}


def aoi_mask(gdf, crs, transform, shape):
    """Mascara booleana del tamano del raster: True dentro del lote."""
    geom = gdf.to_crs(crs).union_all()
    return geometry_mask([geom], out_shape=shape, transform=transform, invert=True)


def aoi_area_ha(gdf, crs):
    return float(gdf.to_crs(crs).union_all().area / 10_000)
