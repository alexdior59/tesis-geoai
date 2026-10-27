import numpy as np
import pytest

from app.models.condition_report import ConditionReporter
from app.models.crop_profiles import get_profile
from app.models.raster_processor import RasterProcessor
from conftest import escribir_raster, mosaico_por_bloques


def reporte(path):
    p = RasterProcessor(path).process()
    return ConditionReporter(get_profile("palma"), "palma").build(p)


def test_porcentaje_por_rango_de_ndvi(lote_cuadrantes):
    rep, _ = reporte(lote_cuadrantes)
    pct = {r["clave"]: r["pct_area"] for r in rep["rangos"]}
    assert pct == pytest.approx({"suelo_o_senescencia": 25, "estres": 25, "transicion": 25, "buena_condicion": 25})
    assert rep["area_con_datos_ha"] == pytest.approx(0.04)
    assert sum(r["area_ha"] for r in rep["rangos"]) == pytest.approx(0.04)


def test_zona_de_inspeccion_georreferenciada(lote_cuadrantes):
    rep, geojson = reporte(lote_cuadrantes)
    insp = rep["inspeccion"]
    assert insp["pct_area_en_rango"] == pytest.approx(25)
    assert insp["n_zonas"] == 1
    zona = insp["zonas_principales"][0]
    assert zona["area_m2"] == pytest.approx(100.0)
    assert zona["ndvi_medio"] == pytest.approx(0.4, abs=1e-3)
    assert 3.5 < zona["lat"] < 4.5 and -74.0 < zona["lon"] < -73.0
    assert len(geojson["features"]) == 1
    assert geojson["features"][0]["properties"]["tipo"] == "zona_de_inspeccion"


def test_parches_menores_al_minimo_no_son_zonas(tmp_path):
    mapa = np.full((200, 200), "vigorosa", dtype=object)
    mapa[50:70, 50:70] = "estres"  # 2 x 2 m = 4 m2 < 10 m2
    rep, _ = reporte(escribir_raster(tmp_path / "parche.tif", mosaico_por_bloques(mapa), res=0.1))
    assert rep["inspeccion"]["area_en_rango_ha"] > 0
    assert rep["inspeccion"]["n_zonas"] == 0


def test_lenguaje_neutral(lote_cuadrantes):
    rep, _ = reporte(lote_cuadrantes)
    assert "no implica dano" in rep["nota"]
    assert rep["fuente_umbrales"].startswith("Concepto del codirector")
