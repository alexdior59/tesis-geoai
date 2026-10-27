import numpy as np
import pytest

from app.models.explainability import ZoneExplainer, local_attribution, max_membership
from app.models.raster_processor import RasterProcessor
from app.models.spatial_segmenter import SpatialSegmenter


@pytest.fixture
def datos_ndre():
    """Dos zonas que solo se diferencian en NDRE: NDVI y GNDVI no aportan informacion."""
    rng = np.random.default_rng(0)
    n = 4_000
    zona = np.repeat([0, 1], n // 2)
    X = np.column_stack([
        rng.normal(0.85, 0.01, n),
        np.where(zona == 0, 0.25, 0.38) + rng.normal(0, 0.01, n),
        rng.normal(0.75, 0.01, n),
    ])
    mean, scale = X.mean(axis=0), X.std(axis=0)
    Z = (X - mean) / scale
    centros = np.array([Z[zona == 0].mean(axis=0), Z[zona == 1].mean(axis=0)])
    return X, zona, centros, mean, scale


def test_pertenencia_maxima_en_rango_y_media_en_la_frontera():
    d = np.array([[0.0, 1.0], [1.0, 1.0], [0.2, 2.0]])
    u = max_membership(d)
    assert u[0] == pytest.approx(1.0, abs=1e-6)
    assert u[1] == pytest.approx(0.5)
    assert 0.5 < u[2] < 1.0


def test_explicacion_local_es_aditiva_y_exacta(datos_ndre):
    _, _, centros, _, _ = datos_ndre
    z = np.array([0.3, 0.1, -0.2])
    e = local_attribution(z, centros)
    d2 = ((z - centros) ** 2).sum(axis=1)
    assert sum(e["aportes"].values()) == pytest.approx(e["ventaja_total"], abs=1e-3)
    assert e["ventaja_total"] == pytest.approx(d2[e["segunda_zona"]] - d2[e["zona_asignada"]], abs=1e-3)


def test_importancia_identifica_el_indice_informativo(datos_ndre):
    X, zona, centros, mean, scale = datos_ndre
    rep = ZoneExplainer(sample_size=4_000).explain(X, zona, centros, mean, scale)
    imp = rep["importancia_permutacion"]
    assert imp["NDRE"] > 30
    assert imp["NDVI"] < 5 and imp["GNDVI"] < 5


def test_modelo_sustituto_fiel_y_con_reglas(datos_ndre):
    X, zona, centros, mean, scale = datos_ndre
    sus = ZoneExplainer(sample_size=4_000).explain(X, zona, centros, mean, scale)["modelo_sustituto"]
    assert sus["fidelidad_pct"] > 95
    assert sus["importancia"]["NDRE"] > 0.9
    assert "NDRE" in sus["reglas"]


def test_prototipos_en_unidades_originales(datos_ndre):
    X, zona, centros, mean, scale = datos_ndre
    proto = ZoneExplainer(sample_size=4_000).explain(X, zona, centros, mean, scale)["prototipos"]
    por_zona = {p["cluster_id"]: p for p in proto}
    assert por_zona[0]["ndre"] == pytest.approx(0.25, abs=0.005)
    assert por_zona[1]["ndre"] == pytest.approx(0.38, abs=0.005)
    assert sum(p["pct_pixeles"] for p in proto) == pytest.approx(100)


def test_explicabilidad_sobre_la_segmentacion_real(lote_franjas):
    p = RasterProcessor(lote_franjas).process()
    r = SpatialSegmenter(kmeans_k=3, resolution_m=p.metadata.resolution_m).segment(p)
    assert r.max_membership.shape == r.kmeans_labels.shape
    assert (r.max_membership >= 1 / 3 - 1e-6).all() and (r.max_membership <= 1 + 1e-6).all()
    a = r.model_artifacts
    rep = ZoneExplainer(sample_size=20_000).explain(
        p.feature_stack, r.kmeans_labels, a["fcm_centers"], a["scaler_mean"], a["scaler_scale"],
        memberships=r.max_membership)
    assert rep["incertidumbre"]["minimo_teorico"] == pytest.approx(1 / 3, abs=1e-4)
    assert rep["incertidumbre"]["pct_pixeles_transicion"] < 5
    tipos = [e["tipo"] for e in rep["explicaciones_locales"]]
    assert tipos.count("tipico") == 3 and tipos.count("transicion") == 3
