import numpy as np
import pytest

from app.models.palm_analyzer import PalmAnalyzer
from app.models.raster_processor import RasterProcessor
from conftest import ENFERMA, FALTANTE, ORIGIN


def a_utm(pos):
    x, y = pos
    return np.array([ORIGIN[0] + x, ORIGIN[1] - y])


@pytest.fixture
def resultado(lote_palma):
    path, posiciones = lote_palma
    p = RasterProcessor(path).process()
    reporte, geojson = PalmAnalyzer().analyze(path, p)
    return reporte, geojson, posiciones


def cerca(items, punto, radio=2.5):
    return [i for i in items if np.hypot(i["x_utm"] - punto[0], i["y_utm"] - punto[1]) <= radio]


def test_conteo_de_palmas(resultado):
    reporte, _, posiciones = resultado
    sanas = len(posiciones) - 2  # sin la faltante ni la enferma
    assert sanas <= reporte["conteo_palmas"] <= sanas + 1


def test_densidad_coherente_con_siembra_a_9_m(resultado):
    reporte, _, _ = resultado
    assert reporte["densidad_esperada_palmas_ha"] == pytest.approx(142.6, abs=0.1)
    assert reporte["densidad_palmas_ha"] == pytest.approx(142.6, rel=0.15)


def test_cada_palma_sana_se_detecta_una_vez(resultado):
    reporte, geojson, posiciones = resultado
    palmas = [f["properties"] | {"x_utm": f["geometry"]["coordinates"][0], "y_utm": f["geometry"]["coordinates"][1]}
              for f in geojson["features"] if "palma_id" in f["properties"]]
    for k, pos in enumerate(posiciones):
        if k in (FALTANTE, ENFERMA):
            continue
        assert len(cerca(palmas, a_utm(pos))) == 1, f"palma {k}"


def test_palma_faltante_se_reporta(resultado):
    reporte, _, posiciones = resultado
    assert cerca(reporte["lista_faltantes"], a_utm(posiciones[FALTANTE]))


def test_palma_enferma_queda_para_revision(resultado):
    reporte, _, posiciones = resultado
    punto = a_utm(posiciones[ENFERMA])
    revisadas = reporte["lista_inspeccion"] + reporte["lista_copa_debil"] + reporte["lista_faltantes"]
    assert cerca(revisadas, punto)


def test_palmas_sanas_no_se_marcan(resultado):
    reporte, _, posiciones = resultado
    falsas = [p for p in reporte["lista_inspeccion"] + reporte["lista_copa_debil"]
              if not cerca([p], a_utm(posiciones[ENFERMA]))]
    assert len(falsas) <= 1


def test_salida_con_coordenadas_gps(resultado):
    reporte, geojson, _ = resultado
    assert geojson["crs"]["properties"]["name"] == "EPSG:32618"
    for f in geojson["features"]:
        assert 3.5 < f["properties"]["lat"] < 4.5
        assert -74.0 < f["properties"]["lon"] < -73.0
