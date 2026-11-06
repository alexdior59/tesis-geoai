import json

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, box

from app.models.aoi import read_aoi
from app.models.raster_processor import RasterProcessor
from conftest import ORIGIN
from test_api import analizar, esperar, nueva_api, subir

# lote_cuadrantes ocupa 20 x 20 m desde ORIGIN hacia el sur y el este; la mitad
# izquierda tiene palma vigorosa arriba y estres abajo
MITAD_IZQUIERDA = box(ORIGIN[0], ORIGIN[1] - 20, ORIGIN[0] + 10, ORIGIN[1])


@pytest.fixture
def api(tmp_path, monkeypatch):
    return nueva_api(tmp_path / "refs", monkeypatch)


def lote(geometria, crs="EPSG:32618"):
    return gpd.GeoDataFrame(geometry=[geometria], crs=crs)


def geojson_bytes(gdf):
    return gdf.to_json().encode("utf-8")


def test_fuera_del_lote_queda_sin_dato(lote_cuadrantes):
    p = RasterProcessor(lote_cuadrantes, aoi=lote(MITAD_IZQUIERDA)).process()
    assert np.isnan(p.ndvi[:, 100:]).all()
    assert np.isfinite(p.ndvi[:, :100]).all()
    assert p.valid_mask[:, :100].all()
    assert p.metadata.aoi_area_ha == pytest.approx(0.02)


def test_lote_en_otro_crs_se_reproyecta(lote_cuadrantes):
    utm = RasterProcessor(lote_cuadrantes, aoi=lote(MITAD_IZQUIERDA)).process()
    wgs84 = RasterProcessor(lote_cuadrantes, aoi=lote(MITAD_IZQUIERDA).to_crs("EPSG:4326")).process()
    assert wgs84.valid_mask.sum() == pytest.approx(utm.valid_mask.sum(), rel=0.01)


def test_lote_que_no_se_cruza_con_la_imagen(lote_cuadrantes):
    lejos = box(ORIGIN[0] + 1000, ORIGIN[1] - 20, ORIGIN[0] + 1010, ORIGIN[1])
    with pytest.raises(ValueError, match="no se cruza"):
        RasterProcessor(lote_cuadrantes, aoi=lote(lejos)).process()


def test_lote_sin_poligonos(tmp_path):
    puntos = gpd.GeoDataFrame(geometry=[Point(ORIGIN)], crs="EPSG:32618")
    with pytest.raises(ValueError, match="poligonos"):
        read_aoi(geojson_bytes(puntos))
    with pytest.raises(ValueError):
        read_aoi(b"esto no es un GeoJSON")


def enviar_con_lote(client, path, aoi_bytes, **form):
    with open(path, "rb") as fh:
        return client.post("/api/v1/analyze", data=form, files={
            "file": (path.name, fh, "image/tiff"),
            "aoi": ("lote.geojson", aoi_bytes, "application/geo+json"),
        })


def test_analisis_y_linea_base_con_el_lote(api, lote_cuadrantes):
    client, _ = api
    aoi = geojson_bytes(lote(MITAD_IZQUIERDA))
    r = enviar_con_lote(client, lote_cuadrantes, aoi, crop_type="generico", kmeans_k="2")
    assert r.status_code == 202, r.text
    job = r.json()["job_id"]
    assert esperar(client, job)["status"] == "completed"
    res = client.get(f"/api/v1/result/{job}").json()
    assert res["data_provenance"]["lote_area_ha"] == pytest.approx(0.02)
    assert res["analysis_params"]["poligono_lote"] is True
    rangos = {r["clave"]: r["pct_area"] for r in res["condicion_por_rangos"]["rangos"]}
    assert rangos["buena_condicion"] == pytest.approx(50, abs=0.5)
    assert rangos["estres"] == pytest.approx(50, abs=0.5)
    assert rangos["suelo_o_senescencia"] == 0

    # la linea base guarda el lote y el vuelo siguiente lo usa sin volver a enviarlo
    ref_id = client.post(f"/api/v1/reference/{job}", data={"name": "base con lote"}).json()["reference_id"]
    _, res2 = analizar(client, lote_cuadrantes, crop_type="generico", reference_model_id=ref_id)
    assert res2["data_provenance"]["lote_area_ha"] == pytest.approx(0.02)
    r = enviar_con_lote(client, lote_cuadrantes, aoi, crop_type="generico", reference_model_id=ref_id)
    assert r.status_code == 422


def test_lote_invalido_por_api(api, lote_cuadrantes):
    client, _ = api
    vacio = json.dumps({"type": "FeatureCollection", "features": []}).encode()
    assert enviar_con_lote(client, lote_cuadrantes, vacio).status_code == 422
    assert subir(client, lote_cuadrantes).status_code == 202
