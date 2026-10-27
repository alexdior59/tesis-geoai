import numpy as np
import pytest

from app.models.accuracy import accuracy_assessment, blind_geojson, stratified_sample
from app.models.raster_processor import RasterProcessor
from app.models.spatial_segmenter import SpatialSegmenter


def test_matriz_de_confusion_con_valores_conocidos():
    mapa = ["A", "A", "A", "B", "B", "C"]
    referencia = ["A", "A", "B", "B", "B", "C"]
    r = accuracy_assessment(mapa, referencia)
    assert r["clases"] == ["A", "B", "C"]
    assert r["matriz_confusion"] == [[2, 1, 0], [0, 2, 0], [0, 0, 1]]
    assert r["exactitud_global"] == pytest.approx(5 / 6, abs=1e-4)
    assert r["kappa"] == pytest.approx(17 / 23, abs=1e-4)
    assert r["por_clase"]["A"]["exactitud_usuario"] == pytest.approx(2 / 3, abs=1e-4)
    assert r["por_clase"]["A"]["exactitud_productor"] == pytest.approx(1.0)
    assert r["por_clase"]["B"]["exactitud_usuario"] == pytest.approx(1.0)
    assert r["por_clase"]["B"]["exactitud_productor"] == pytest.approx(2 / 3, abs=1e-4)


def test_acuerdo_perfecto_da_kappa_uno():
    r = accuracy_assessment(["x", "y", "x"], ["x", "y", "x"])
    assert r["exactitud_global"] == 1.0 and r["kappa"] == 1.0


def test_listas_invalidas():
    with pytest.raises(ValueError):
        accuracy_assessment(["A"], [])


@pytest.fixture
def segmentado(lote_franjas):
    p = RasterProcessor(lote_franjas).process()
    r = SpatialSegmenter(kmeans_k=3, resolution_m=p.metadata.resolution_m).segment(p)
    return p, r


def test_muestreo_estratificado_por_zona(segmentado):
    p, r = segmentado
    puntos = stratified_sample(p, r.kmeans_labels, n_per_zone=5, min_distance_m=2.0)
    por_zona = np.bincount([pt["zona"] for pt in puntos])
    assert list(por_zona) == [5, 5, 5]
    xy = np.array([[pt["x_utm"], pt["y_utm"]] for pt in puntos])
    distancias = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    np.fill_diagonal(distancias, np.inf)
    assert distancias.min() >= 2.0 - 0.15
    assert sorted(pt["punto_id"] for pt in puntos) == list(range(1, 16))


def test_muestreo_reproducible(segmentado):
    p, r = segmentado
    a = stratified_sample(p, r.kmeans_labels, n_per_zone=5, min_distance_m=2.0, seed=1)
    b = stratified_sample(p, r.kmeans_labels, n_per_zone=5, min_distance_m=2.0, seed=1)
    assert a == b


def test_geojson_a_ciegas_no_revela_la_zona(segmentado):
    p, r = segmentado
    puntos = stratified_sample(p, r.kmeans_labels, n_per_zone=3, min_distance_m=2.0)
    gj = blind_geojson(puntos, p.metadata.crs)
    assert len(gj["features"]) == 9
    assert all("zona" not in f["properties"] for f in gj["features"])
