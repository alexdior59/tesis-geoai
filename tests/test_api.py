import importlib
import json
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def nueva_api(ref_dir, monkeypatch):
    monkeypatch.setenv("GEOAI_REF_DIR", str(ref_dir))
    monkeypatch.setenv("GEOAI_RESULTS_DIR", str(ref_dir.parent / "results"))
    import app.controllers.analyze_router as router_mod
    router_mod = importlib.reload(router_mod)
    api = FastAPI()
    api.include_router(router_mod.router)
    return TestClient(api), router_mod


@pytest.fixture
def api(tmp_path, monkeypatch):
    return nueva_api(tmp_path / "refs", monkeypatch)


def subir(client, path, **form):
    with open(path, "rb") as fh:
        return client.post("/api/v1/analyze", files={"file": (path.name, fh, "image/tiff")}, data=form)


def esperar(client, job_id, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = client.get(f"/api/v1/status/{job_id}").json()
        if st["status"] in ("completed", "failed"):
            return st
        time.sleep(0.5)
    raise TimeoutError(job_id)


def analizar(client, path, **form):
    r = subir(client, path, **form)
    assert r.status_code == 202, r.text
    job = r.json()["job_id"]
    st = esperar(client, job)
    assert st["status"] == "completed", st.get("error")
    return job, client.get(f"/api/v1/result/{job}").json()


def test_health_y_cultivos(api):
    client, _ = api
    assert client.get("/api/v1/health").json()["status"] == "ok"
    crops = client.get("/api/v1/crops").json()
    assert set(crops) == {"generico", "palma"}
    assert crops["palma"]["analisis_por_palma"] is True


def test_validaciones_de_entrada(api, lote_cuadrantes, tmp_path):
    client, _ = api
    assert subir(client, lote_cuadrantes, crop_type="banano").status_code == 422
    txt = tmp_path / "no_es_tif.txt"
    txt.write_text("x")
    assert subir(client, txt).status_code == 422
    assert subir(client, lote_cuadrantes, reference_model_id="no-existe").status_code == 404


def test_analisis_completo_generico(api, lote_franjas):
    client, _ = api
    job, res = analizar(client, lote_franjas, crop_type="generico", kmeans_k=3)
    assert res["api_version"] == "1.2.0"
    assert res["condicion_por_rangos"]["cultivo"] == "generico"
    assert res["analisis_palmas"] is None
    assert res["quality_metrics"]["kmeans"]["fcm_partition_coefficient"] is not None
    assert "anomal" not in json.dumps(res).lower()
    assert client.get(f"/api/v1/result/{job}/geojson").json()["type"] == "FeatureCollection"
    assert client.get(f"/api/v1/result/{job}/inspection").json()["type"] == "FeatureCollection"
    assert client.get(f"/api/v1/result/{job}/palms").status_code == 404
    assert "RESUMEN" in client.get(f"/api/v1/result/{job}/summary").json()["summary"]


def test_explicabilidad_e_incertidumbre(api, lote_franjas, tmp_path):
    client, _ = api
    job, res = analizar(client, lote_franjas, crop_type="generico", kmeans_k=3)
    e = res["explicabilidad"]
    assert set(e["importancia_permutacion"]) == {"NDVI", "NDRE", "GNDVI"}
    assert e["modelo_sustituto"]["fidelidad_pct"] > 90
    assert len(e["prototipos"]) == 3
    r = client.get(f"/api/v1/result/{job}/uncertainty")
    assert r.status_code == 200 and r.headers["content-type"] == "image/tiff"
    tif = tmp_path / "inc.tif"
    tif.write_bytes(r.content)
    import rasterio
    with rasterio.open(tif) as ds:
        v = ds.read(1)
        assert ds.crs.to_epsg() == 32618
        assert v[v != 255].min() >= 33 and v[v != 255].max() <= 100


def test_exactitud_tematica_por_api(api, lote_franjas):
    client, _ = api
    job, res = analizar(client, lote_franjas, crop_type="generico", kmeans_k=3)
    puntos = client.get(f"/api/v1/result/{job}/accuracy-sample").json()["features"]
    assert puntos and all("zona" not in f["properties"] for f in puntos)
    ids = [str(f["properties"]["punto_id"]) for f in puntos]
    zonas = {str(z["cluster_id"]): f"clase {z['cluster_id']}" for z in res["kmeans_zones"]}
    _, router_mod = api
    clave = {str(p["punto_id"]): str(p["zona"]) for p in router_mod._jobs[job]["accuracy_points"]}
    respuesta_perfecta = {pid: zonas[clave[pid]] for pid in ids}
    r = client.post(f"/api/v1/result/{job}/accuracy",
                    json={"referencias": respuesta_perfecta, "zone_labels": zonas})
    assert r.status_code == 200
    assert r.json()["exactitud_global"] == 1.0 and r.json()["kappa"] == 1.0
    malo = client.post(f"/api/v1/result/{job}/accuracy", json={"referencias": {"9999": "x"}})
    assert malo.status_code == 422


def test_analisis_de_palma(api, lote_palma):
    client, _ = api
    path, _ = lote_palma
    job, res = analizar(client, path, crop_type="palma")
    assert res["analisis_palmas"]["conteo_palmas"] > 50
    palms = client.get(f"/api/v1/result/{job}/palms").json()
    estados = {f["properties"]["estado"] for f in palms["features"]}
    assert "normal" in estados and "posible_faltante" in estados


def test_ciclo_de_vida_de_la_linea_base(api, lote_franjas, tmp_path, monkeypatch):
    client, _ = api
    job, res = analizar(client, lote_franjas, crop_type="generico", kmeans_k=3)
    ids = [str(z["cluster_id"]) for z in res["kmeans_zones"]]

    malo = client.post(f"/api/v1/reference/{job}", data={"zone_labels": json.dumps({"99": "x"})})
    assert malo.status_code == 422

    etiquetas = {cid: f"Zona experta {cid}" for cid in ids}
    ref = client.post(f"/api/v1/reference/{job}", data={
        "name": "Linea base prueba", "validated_by": "Codirector", "zone_labels": json.dumps(etiquetas)})
    assert ref.status_code == 201
    ref_id = ref.json()["reference_id"]
    assert (tmp_path / "refs" / f"{ref_id}.json").exists()

    client2, router2 = nueva_api(tmp_path / "refs", monkeypatch)
    assert ref_id in router2._references
    assert ref_id in [r["reference_id"] for r in client2.get("/api/v1/references").json()["references"]]

    assert subir(client2, lote_franjas, crop_type="palma", reference_model_id=ref_id).status_code == 422

    _, res2 = analizar(client2, lote_franjas, crop_type="generico", reference_model_id=ref_id, kmeans_k=7)
    assert res2["analysis_params"]["kmeans_k"] == 3
    assert res2["analysis_params"]["classification_mode"] == "centroides_congelados_referencia"
    assert all(abs(z["area_delta_ha"]) < 1e-9 for z in res2["drift_report"]["zone_comparison"])
    assert {z["expert_label"] for z in res2["kmeans_zones"]} == set(etiquetas.values())
