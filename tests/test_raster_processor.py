import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app.models.raster_processor import RasterProcessor
from conftest import FIRMAS, ORIGIN, escribir_raster, mosaico_por_bloques


def bloque(firma, n=50):
    mapa = np.full((n, n), firma, dtype=object)
    return mosaico_por_bloques(mapa)


def test_mapeo_de_bandas_por_descripcion(lote_cuadrantes):
    p = RasterProcessor(lote_cuadrantes).process()
    assert p.metadata.band_mapping == {"green": 1, "red": 2, "rededge": 3, "nir": 4, "alpha": 5}


def test_sin_descripciones_usa_orden_del_mavic_3m(tmp_path):
    path = escribir_raster(tmp_path / "sin_desc.tif", bloque("vigorosa"), res=0.1, descriptions=None)
    p = RasterProcessor(path).process()
    assert p.metadata.band_mapping["nir"] == 4
    assert p.metadata.band_mapping["alpha"] == 5


def test_regresion_alpha_no_se_usa_como_nir(tmp_path):
    # Antes del arreglo, la banda 5 (Alpha = 255) se leia como NIR y el NDVI salia ~0.3 en vez de ~0.9
    path = escribir_raster(tmp_path / "v.tif", bloque("vigorosa"), res=0.1)
    p = RasterProcessor(path).process()
    assert np.nanmean(p.ndvi) == pytest.approx(0.8966, abs=1e-3)


def test_indices_con_valores_conocidos(tmp_path):
    g, r, re, nir = FIRMAS["vigorosa"]
    path = escribir_raster(tmp_path / "v.tif", bloque("vigorosa"), res=0.1)
    p = RasterProcessor(path).process()
    assert np.nanmean(p.ndvi) == pytest.approx((nir - r) / (nir + r), abs=1e-4)
    assert np.nanmean(p.ndre) == pytest.approx((nir - re) / (nir + re), abs=1e-4)
    assert np.nanmean(p.gndvi) == pytest.approx((nir - g) / (nir + g), abs=1e-4)


def test_mascara_alpha_y_umbral_de_vegetacion(tmp_path):
    mapa = np.full((60, 60), "vigorosa", dtype=object)
    mapa[:, 30:] = "suelo"
    bandas = mosaico_por_bloques(mapa)
    bandas["alpha"] = np.full((60, 60), 255, np.float32)
    bandas["alpha"][:10, :10] = 0
    p = RasterProcessor(escribir_raster(tmp_path / "m.tif", bandas, res=0.1)).process()
    assert np.isnan(p.ndvi[:10, :10]).all()
    assert not p.valid_mask[:, 30:].any()
    assert np.isfinite(p.ndvi[:, 30:]).all()
    assert p.feature_stack.shape == (60 * 30 - 100, 3)


def test_un_solo_factor_de_escala_preserva_los_indices(tmp_path):
    referencia = RasterProcessor(escribir_raster(tmp_path / "a.tif", bloque("buena"), res=0.1)).process()
    escalado = {k: v * 10_000 for k, v in bloque("buena").items()}
    p = RasterProcessor(escribir_raster(tmp_path / "b.tif", escalado, res=0.1)).process()
    assert np.nanmean(p.ndvi) == pytest.approx(np.nanmean(referencia.ndvi), abs=1e-5)
    assert np.nanmean(p.ndre) == pytest.approx(np.nanmean(referencia.ndre), abs=1e-5)


def test_banda_mayor_a_uno_no_distorsiona_el_ndvi(tmp_path):
    bandas = bloque("buena")
    bandas["nir"] = np.full_like(bandas["nir"], 1.2)
    p = RasterProcessor(escribir_raster(tmp_path / "c.tif", bandas, res=0.1)).process()
    red = FIRMAS["buena"][1]
    assert np.nanmean(p.ndvi) == pytest.approx((1.2 - red) / (1.2 + red), abs=1e-4)


def test_menos_de_cuatro_bandas_es_error(tmp_path):
    path = tmp_path / "tres.tif"
    with rasterio.open(path, "w", driver="GTiff", height=10, width=10, count=3, dtype="float32",
                       crs="EPSG:32618", transform=from_origin(*ORIGIN, 0.1, 0.1)) as ds:
        ds.write(np.ones((3, 10, 10), np.float32))
    with pytest.raises(ValueError):
        RasterProcessor(path).process()
