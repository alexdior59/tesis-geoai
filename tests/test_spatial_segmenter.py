import numpy as np
import pytest

from app.models.raster_processor import RasterProcessor
from app.models.spatial_segmenter import SpatialSegmenter
from app.views.response_serializer import ResponseSerializer
from conftest import mapa_franjas

ETIQUETAS_EXPERTO = {"0": "Zona experta A", "1": "Zona experta B", "2": "Zona experta C"}


def segmentar(path, k=3, referencia=None):
    p = RasterProcessor(path).process()
    seg = SpatialSegmenter(kmeans_k=k, resolution_m=p.metadata.resolution_m)
    return p, seg.segment(p, reference_model=referencia)


def linea_base(p, r, etiquetas=None):
    """Mismo contenido que guarda POST /reference/{job_id}."""
    return {
        "reference_id": "prueba",
        "name": "Linea base de prueba",
        "created_at": "2026-10-26",
        "validated_by": "Experto de prueba",
        "zone_labels": etiquetas or {},
        **{k: r.model_artifacts[k] for k in ("fcm_centers", "scaler_mean", "scaler_scale", "kmeans_k", "distance_p99")},
        "band_stats": p.band_stats,
        "kmeans_zones": [ResponseSerializer._zone_to_dict(z) for z in r.kmeans_zones],
    }


def etiquetas_en_mapa(p, labels):
    m = np.full(p.valid_mask.shape, -1, dtype=np.int16)
    m[p.valid_mask] = labels
    return m


def test_fcm_recupera_las_tres_condiciones(lote_franjas):
    p, r = segmentar(lote_franjas)
    mapa = etiquetas_en_mapa(p, r.kmeans_labels)
    verdad = mapa_franjas()
    for condicion in ("vigorosa", "buena", "estres"):
        zona = mapa[verdad == condicion]
        dominante = np.bincount(zona).max() / zona.size
        assert dominante > 0.99, condicion


def test_resultado_reproducible(lote_franjas):
    _, r1 = segmentar(lote_franjas)
    _, r2 = segmentar(lote_franjas)
    assert np.array_equal(r1.kmeans_labels, r2.kmeans_labels)


def test_metricas_de_calidad_en_rango(lote_franjas):
    _, r = segmentar(lote_franjas)
    k = r.metrics["kmeans_n_clusters"]
    assert 1 / k <= r.metrics["fcm_partition_coefficient"] <= 1
    assert r.metrics["kmeans_silhouette"] > 0.5
    assert r.metrics["kmeans_davies_bouldin"] > 0


def test_zonas_ordenadas_y_con_nombre_de_la_escala_del_experto(lote_franjas):
    _, r = segmentar(lote_franjas)
    ndvi = [z.ndvi_mean for z in r.kmeans_zones]
    assert ndvi == sorted(ndvi, reverse=True)
    claves = [z.label.split("_", 1)[1].rsplit("_", 1)[0] for z in r.kmeans_zones]
    assert claves == ["buena_condicion", "buena_condicion", "estres"]
    assert all(not z.label.lower().startswith("anomal") for z in r.kmeans_zones)


def test_linea_base_mismo_vuelo_no_reporta_cambio(lote_franjas):
    p, r = segmentar(lote_franjas)
    ref = linea_base(p, r, ETIQUETAS_EXPERTO)
    _, r2 = segmentar(lote_franjas, referencia=ref)
    assert np.array_equal(r.kmeans_labels, r2.kmeans_labels)
    assert r2.params["classification_mode"] == "centroides_congelados_referencia"
    assert all(abs(z["area_delta_ha"]) < 1e-9 for z in r2.drift_report["zone_comparison"])


def test_linea_base_congelada_detecta_estres_sin_cambios_falsos(lote_franjas, lote_franjas_estresado):
    p1, r1 = segmentar(lote_franjas)
    ref = linea_base(p1, r1)
    p2, r2 = segmentar(lote_franjas_estresado, referencia=ref)
    m1, m2 = etiquetas_en_mapa(p1, r1.kmeans_labels), etiquetas_en_mapa(p2, r2.kmeans_labels)
    w = m1.shape[1]
    ambos = (m1 >= 0) & (m2 >= 0)
    derecha = ambos.copy()
    derecha[:, : w // 2] = False
    izquierda = ambos & ~derecha
    assert np.mean(m1[derecha] != m2[derecha]) == 0.0
    assert np.mean(m1[izquierda] != m2[izquierda]) > 0.05
    assert r2.drift_report["centroid_shift_l2"] > 0


def test_cambio_reportado_por_zona_con_etiquetas_del_experto(lote_franjas, lote_franjas_estresado):
    p1, r1 = segmentar(lote_franjas)
    ref = linea_base(p1, r1, ETIQUETAS_EXPERTO)
    _, r2 = segmentar(lote_franjas_estresado, referencia=ref)
    comp = {z["cluster_id"]: z for z in r2.drift_report["zone_comparison"]}
    assert set(comp) == {0, 1, 2}
    assert {z["expert_label"] for z in comp.values()} == set(ETIQUETAS_EXPERTO.values())
    assert all(z.expert_label for z in r2.kmeans_zones)
    assert r2.drift_report["validated_by"] == "Experto de prueba"


def test_k_distinto_al_de_la_linea_base_es_error(lote_franjas):
    p, r = segmentar(lote_franjas)
    with pytest.raises(ValueError):
        segmentar(lote_franjas, k=4, referencia=linea_base(p, r))


def test_desviacion_normalizada_con_la_linea_base(lote_franjas):
    p, r = segmentar(lote_franjas)
    ref = linea_base(p, r)
    _, r2 = segmentar(lote_franjas, referencia=ref)
    assert r2.model_artifacts["distance_p99"] == pytest.approx(ref["distance_p99"])
    assert all(0 <= z.deviation_score_mean <= 1 for z in r2.kmeans_zones)
