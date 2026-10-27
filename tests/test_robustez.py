import json

import numpy as np
import pytest
import rasterio
from rasterio.enums import ColorInterp
from shapely.geometry import shape

from app.models.raster_processor import RasterProcessor
from conftest import escribir_raster, mosaico_por_bloques
from test_api import analizar, nueva_api


@pytest.fixture
def api(tmp_path, monkeypatch):
    return nueva_api(tmp_path / "refs", monkeypatch)


def bloque(firma, n=50):
    return mosaico_por_bloques(np.full((n, n), firma, dtype=object))


def test_alpha_detectada_por_color_interpretation(tmp_path):
    path = escribir_raster(tmp_path / "a.tif", bloque("vigorosa"), res=0.1,
                           descriptions=("Green", "Red", "Red edge", "NIR", "Banda 5"))
    with rasterio.open(path, "r+") as ds:
        ds.colorinterp = [ColorInterp.gray, ColorInterp.undefined, ColorInterp.undefined,
                          ColorInterp.undefined, ColorInterp.alpha]
    assert RasterProcessor(path).process().metadata.band_mapping["alpha"] == 5


def test_infrared_no_se_confunde_con_red(tmp_path):
    path = escribir_raster(tmp_path / "b.tif", bloque("vigorosa"), res=0.1,
                           descriptions=("Green", "Red", "Red edge", "Near infrared", "Alpha"))
    mapping = RasterProcessor(path).process().metadata.band_mapping
    assert mapping["red"] == 2 and mapping["nir"] == 4


def test_umbral_por_constructor(tmp_path):
    path = escribir_raster(tmp_path / "c.tif", bloque("transicion"), res=0.1)
    assert RasterProcessor(path).process().valid_mask.all()
    with pytest.raises(ValueError):
        RasterProcessor(path, vegetation_threshold=0.9).process()


def test_resultado_trazable(api, lote_franjas):
    client, _ = api
    job, res = analizar(client, lote_franjas, crop_type="generico", kmeans_k=3, vegetation_threshold=0.2)
    assert res["job_id"] == job
    assert res["analysis_params"]["vegetation_threshold"] == 0.2
    assert res["data_provenance"]["source_file"] == lote_franjas.name


def test_area_de_cada_poligono_y_de_la_zona(api, lote_franjas):
    client, _ = api
    job, res = analizar(client, lote_franjas, crop_type="generico", kmeans_k=3)
    features = client.get(f"/api/v1/result/{job}/geojson").json()["features"]
    total_poligonos = sum(f["properties"]["area_ha"] for f in features)
    total_zonas = sum(z["area_ha"] for z in res["kmeans_zones"])
    assert total_poligonos == pytest.approx(total_zonas, rel=0.05)
    for f in features:
        assert f["properties"]["area_ha"] == pytest.approx(shape(f["geometry"]).area / 10_000, rel=1e-3)


def test_patrones_dbscan_en_geojson(api, lote_palma):
    client, _ = api
    path, _ = lote_palma
    job, res = analizar(client, path, crop_type="palma")
    patrones = client.get(f"/api/v1/result/{job}/patterns").json()
    assert patrones["type"] == "FeatureCollection"
    assert len(patrones["features"]) == res["quality_metrics"]["dbscan"]["n_clusters_detected"]


def test_linea_base_impone_su_umbral(api, lote_franjas, tmp_path, monkeypatch):
    client, _ = api
    job, _ = analizar(client, lote_franjas, crop_type="generico", kmeans_k=3, vegetation_threshold=0.2)
    ref_id = client.post(f"/api/v1/reference/{job}", data={"name": "base"}).json()["reference_id"]
    guardada = json.loads((tmp_path / "refs" / f"{ref_id}.json").read_text(encoding="utf-8"))
    assert guardada["vegetation_threshold"] == 0.2

    client2, _ = nueva_api(tmp_path / "refs", monkeypatch)
    _, res2 = analizar(client2, lote_franjas, crop_type="generico", reference_model_id=ref_id,
                       vegetation_threshold=0.5)
    assert res2["analysis_params"]["vegetation_threshold"] == 0.2


def test_resultado_pendiente_responde_409(api):
    client, router_mod = api
    router_mod._register_job("pendiente", "x.tif", None, "generico")
    assert client.get("/api/v1/result/pendiente").status_code == 409


def test_nombre_de_archivo_sin_rutas(api, lote_cuadrantes):
    client, router_mod = api
    with open(lote_cuadrantes, "rb") as fh:
        r = client.post("/api/v1/analyze", files={"file": ("../../otra/carpeta/lote.tif", fh, "image/tiff")})
    assert r.status_code == 202
    assert router_mod._jobs[r.json()["job_id"]]["source_file"] == "lote.tif"