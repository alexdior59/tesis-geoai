"""
Ejecuta el pipeline completo sin levantar la API y guarda todas las salidas en una carpeta.

Uso:
    python experiments/analisis_completo.py <ortomosaico.tif> <carpeta_salida> [--cultivo palma] [--k 4]
        [--lote lote.geojson]
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.models.accuracy import blind_geojson, stratified_sample  # noqa: E402
from app.models.aoi import read_aoi  # noqa: E402
from app.models.condition_report import ConditionReporter  # noqa: E402
from app.models.crop_profiles import get_profile  # noqa: E402
from app.models.explainability import ZoneExplainer  # noqa: E402
from app.models.palm_analyzer import PalmAnalyzer  # noqa: E402
from app.models.raster_processor import RasterProcessor  # noqa: E402
from app.models.spatial_segmenter import SpatialSegmenter  # noqa: E402
from app.views.response_serializer import ResponseSerializer  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tif")
    ap.add_argument("salida")
    ap.add_argument("--cultivo", default="palma")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--lote", help="poligono del lote (GeoJSON, shapefile o GeoPackage)")
    a = ap.parse_args()

    out = Path(a.salida)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    profile = get_profile(a.cultivo)
    lote = read_aoi(a.lote) if a.lote else None
    processed = RasterProcessor(a.tif, aoi=lote).process()
    if lote is not None:
        lote.to_crs(processed.metadata.crs).to_file(out / "lote.geojson", driver="GeoJSON")
    else:
        (out / "lote.geojson").unlink(missing_ok=True)
    # resolucion sin redondear, igual que en la API, para que las areas cuadren
    seg = SpatialSegmenter(kmeans_k=a.k, resolution_m=abs(processed.transform.a),
                           ndvi_ranges=profile["ndvi_ranges"]).segment(processed)
    cond, insp = ConditionReporter(profile, a.cultivo).build(processed)
    palm, palms_gj = (PalmAnalyzer(**profile["palm_analysis"]).analyze(a.tif, processed)
                      if profile["palm_analysis"] else (None, None))
    art = seg.model_artifacts
    expl = ZoneExplainer(transition_threshold=SpatialSegmenter.TRANSITION_THRESHOLD).explain(
        processed.feature_stack, seg.kmeans_labels, art["fcm_centers"],
        art["scaler_mean"], art["scaler_scale"], fcm_m=art["fcm_m"],
        memberships=seg.max_membership,
    )
    puntos = stratified_sample(processed, seg.kmeans_labels)
    ser = ResponseSerializer(processed, seg, Path(a.tif).name, cond, palm, expl,
                             extra_params={"crop_type": a.cultivo,
                                           "vegetation_threshold": RasterProcessor.VEGETATION_THRESHOLD,
                                           "poligono_lote": lote is not None})
    nd = ser._json_default
    ser.write_uncertainty_tif(out / "incertidumbre.tif")
    ser.write_zones_tif(out / "zonas_transicion.tif", SpatialSegmenter.TRANSITION_THRESHOLD)
    (out / "patrones_dbscan.geojson").write_text(json.dumps(ser.to_patterns_geojson(), default=nd), encoding="utf-8")
    (out / "puntos_verificacion.geojson").write_text(
        json.dumps(blind_geojson(puntos, processed.metadata.crs), default=nd), encoding="utf-8")
    with open(out / "clave_puntos_zona.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["punto_id", "zona"])
        for p in sorted(puntos, key=lambda p: p["punto_id"]):
            w.writerow([p["punto_id"], p["zona"]])
    (out / "resultado.json").write_text(json.dumps(ser.to_json(), default=nd, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "zonas_fcm.geojson").write_text(json.dumps(ser.to_geojson(), default=nd), encoding="utf-8")
    (out / "zonas_inspeccion.geojson").write_text(json.dumps(insp, default=nd), encoding="utf-8")
    if palms_gj:
        (out / "palmas.geojson").write_text(json.dumps(palms_gj, default=nd), encoding="utf-8")
        (out / "palmas_reporte.json").write_text(json.dumps(palm, default=nd, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "resumen.txt").write_text(ser.to_summary(), encoding="utf-8")
    print(ser.to_summary())
    print(f"Tiempo total: {time.time() - t0:.0f} s. Salidas en {out}")


if __name__ == "__main__":
    main()
