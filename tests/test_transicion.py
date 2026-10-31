import numpy as np
import pytest
import rasterio

from app.models.raster_processor import RasterProcessor
from app.models.spatial_segmenter import SpatialSegmenter
from app.views.response_serializer import NODATA_CODE, TRANSITION_CODE
from conftest import FIRMAS, escribir_raster
from test_api import analizar, nueva_api

FILAS, COLUMNAS = 120, 120
DEGRADADO = slice(40, 80)  # filas donde se pasa poco a poco de vigorosa a estres


@pytest.fixture
def api(tmp_path, monkeypatch):
    return nueva_api(tmp_path / "refs", monkeypatch)


def bandas_con_degradado(ruido=0.0):
    """Arriba palma vigorosa, abajo estres y en medio un paso gradual de una condicion a la otra."""
    peso_estres = np.zeros(FILAS, dtype=np.float32)
    n = DEGRADADO.stop - DEGRADADO.start
    peso_estres[DEGRADADO] = np.linspace(0, 1, n + 2)[1:-1]
    peso_estres[DEGRADADO.stop:] = 1.0
    rng = np.random.default_rng(0)
    bandas = {}
    for i, nombre in enumerate(("green", "red", "rededge", "nir")):
        fila = (1 - peso_estres) * FIRMAS["vigorosa"][i] + peso_estres * FIRMAS["estres"][i]
        banda = np.repeat(fila[:, None], COLUMNAS, axis=1)
        if ruido:
            banda = banda + rng.normal(0, ruido, banda.shape).astype(np.float32)
        bandas[nombre] = banda.astype(np.float32)
    return bandas


@pytest.fixture
def lote_degradado(tmp_path):
    return escribir_raster(tmp_path / "degradado.tif", bandas_con_degradado(ruido=0.004), res=0.1)


def segmentar(path, k=2):
    processed = RasterProcessor(path).process()
    result = SpatialSegmenter(kmeans_k=k, resolution_m=0.1).segment(processed)
    return processed, result


def filas_de_cada_pixel(processed):
    filas = np.repeat(np.arange(FILAS)[:, None], COLUMNAS, axis=1)
    return filas[processed.valid_mask]


def test_la_transicion_queda_dentro_del_degradado(lote_degradado):
    processed, r = segmentar(lote_degradado)
    filas = filas_de_cada_pixel(processed)
    transicion = r.max_membership < SpatialSegmenter.TRANSITION_THRESHOLD
    assert transicion.any()
    assert ((filas[transicion] >= DEGRADADO.start) & (filas[transicion] < DEGRADADO.stop)).all()
    # en las zonas puras la asignacion es segura
    puros = (filas < DEGRADADO.start - 5) | (filas >= DEGRADADO.stop + 5)
    assert (r.max_membership[puros] >= 0.9).mean() > 0.99


def test_segunda_zona_es_la_otra_condicion(lote_degradado):
    _, r = segmentar(lote_degradado)
    assert r.second_labels.shape == r.kmeans_labels.shape
    assert (r.second_labels != r.kmeans_labels).all()


def test_reporte_de_transicion(lote_degradado):
    _, r = segmentar(lote_degradado)
    t = r.transition_report
    pct_real = np.mean(r.max_membership < t["umbral_pertenencia"]) * 100
    assert t["pct_pixeles_transicion"] == pytest.approx(pct_real, abs=0.01)
    # el area se redondea a 4 decimales de hectarea
    assert t["area_transicion_ha"] == pytest.approx(pct_real / 100 * FILAS * COLUMNAS * 0.01 / 10_000, abs=1e-4)
    assert {z["cluster_id"] for z in t["por_zona"]} == {0, 1}
    for zona in t["por_zona"]:
        if zona["pct_transicion"] > 0:
            # con dos zonas, un pixel de transicion solo puede mezclarse con la otra
            assert zona["se_mezcla_con"] == {str(1 - zona["cluster_id"]): 100.0}


def test_pixeles_repetidos_no_bloquean_fcm(tmp_path):
    # Sin ruido hay miles de pixeles identicos y el arranque al azar puede tomar el mismo
    # pixel como dos centroides; aun asi deben salir dos zonas distintas.
    path = escribir_raster(tmp_path / "plano.tif", bandas_con_degradado(ruido=0.0), res=0.1)
    processed, r = segmentar(path)
    filas = filas_de_cada_pixel(processed)
    assert len(np.unique(r.kmeans_labels)) == 2
    assert r.kmeans_labels[filas < 10][0] != r.kmeans_labels[filas >= 110][0]


def test_sin_degradado_casi_no_hay_transicion(lote_franjas):
    _, r = segmentar(lote_franjas, k=3)
    assert r.transition_report["pct_pixeles_transicion"] < 2


def test_mapa_de_zonas_con_transicion_por_api(api, lote_degradado, tmp_path):
    client, _ = api
    job, res = analizar(client, lote_degradado, crop_type="generico", kmeans_k=2)
    assert res["transicion"]["pct_pixeles_transicion"] > 0
    assert "TRANSICION" in client.get(f"/api/v1/result/{job}/summary").json()["summary"]

    r = client.get(f"/api/v1/result/{job}/zones-map")
    assert r.status_code == 200 and r.headers["content-type"] == "image/tiff"
    tif = tmp_path / "zonas.tif"
    tif.write_bytes(r.content)
    with rasterio.open(tif) as ds:
        valores = ds.read(1)
        assert ds.crs.to_epsg() == 32618
        assert ds.nodata == NODATA_CODE
    assert set(np.unique(valores).tolist()) <= {0, 1, TRANSITION_CODE, NODATA_CODE}
    con_dato = valores != NODATA_CODE
    pct = np.mean(valores[con_dato] == TRANSITION_CODE) * 100
    assert pct == pytest.approx(res["transicion"]["pct_pixeles_transicion"], abs=0.01)