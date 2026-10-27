import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.models.accuracy import accuracy_assessment, blind_geojson, stratified_sample
from app.models.condition_report import ConditionReporter
from app.models.crop_profiles import CROP_PROFILES, get_profile
from app.models.explainability import ZoneExplainer
from app.models.palm_analyzer import PalmAnalyzer
from app.models.raster_processor import RasterProcessor
from app.models.spatial_segmenter import SpatialSegmenter
from app.views.response_serializer import ResponseSerializer

router = APIRouter(prefix="/api/v1", tags=["GeoAI Analysis"])

PROJECT_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = Path(tempfile.gettempdir()) / "geoai_uploads"
REF_DIR = Path(os.environ.get("GEOAI_REF_DIR", PROJECT_DIR / "data" / "references"))
RESULTS_DIR = Path(os.environ.get("GEOAI_RESULTS_DIR", PROJECT_DIR / "data" / "results"))
for _dir in (UPLOAD_DIR, REF_DIR, RESULTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

MAX_FILE_MB = 2000
UPLOAD_CHUNK_BYTES = 16 * 1024 ** 2
# Un ortomosaico completo ocupa varios GB de RAM mientras se procesa; por eso,
# por defecto, corre un analisis a la vez y los demas esperan en cola.
MAX_CONCURRENT_JOBS = int(os.environ.get("GEOAI_MAX_CONCURRENT_JOBS", "1"))

# Los jobs viven en memoria; las lineas base tambien se guardan en disco.
_jobs = {}
_references = {}
_lock = threading.Lock()
_job_slots = threading.Semaphore(MAX_CONCURRENT_JOBS)


def _load_references():
    for path in sorted(REF_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            _references[data["reference_id"]] = data
        except (OSError, ValueError, KeyError) as e:
            print(f"No se pudo cargar la linea base {path.name}: {e}")


_load_references()


class AnalysisParams(BaseModel):
    kmeans_k: int = Field(default=4, ge=2, le=10)
    dbscan_eps: float = Field(default=0.5, gt=0)
    dbscan_min_samples: int = Field(default=30, ge=5)
    vegetation_threshold: float = Field(default=0.1, ge=-1, le=1)
    crop_type: str = "generico"
    reference_model_id: str | None = None


class AccuracyRequest(BaseModel):
    referencias: dict[str, str] = Field(
        ..., description='Clase observada por el experto en cada punto: {"punto_id": "categoria"}')
    zone_labels: dict[str, str] | None = Field(
        default=None, description='Categoria del experto para cada zona: {"cluster_id": "categoria"}')


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    created_at: str
    completed_at: str | None = None
    error: str | None = None
    source_file: str | None = None
    crop_type: str | None = None
    reference_model_id: str | None = None


@router.post(
    "/analyze",
    summary="Analizar un ortomosaico multiespectral GeoTIFF",
    status_code=status.HTTP_202_ACCEPTED,
)
async def analyze(
    file: UploadFile = File(..., description="Ortomosaico GeoTIFF de Pix4Dfields / DJI Mavic 3M"),
    crop_type: str = Form(default="generico", description="Cultivo: ver GET /api/v1/crops"),
    kmeans_k: int = Form(default=4, ge=2, le=10, description="Numero de zonas FCM"),
    dbscan_eps: float = Form(default=0.5, gt=0, description="Radio de vecindad DBSCAN en metros"),
    dbscan_min_samples: int = Form(default=30, ge=5),
    vegetation_threshold: float = Form(default=0.1, ge=-1, le=1, description="NDVI minimo de vegetacion"),
    reference_model_id: str | None = Form(default=None),
):
    """
    Calcula NDVI/NDRE/GNDVI, la condicion por rangos del cultivo, las zonas FCM,
    los patrones DBSCAN y, en palma, el analisis por palma. Con reference_model_id
    se usan el k y el umbral de NDVI de la linea base y se genera el drift_report.
    """
    filename = Path(file.filename or "").name
    if not filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Solo se aceptan archivos .tif o .tiff",
        )
    if crop_type not in CROP_PROFILES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Cultivo no soportado: {crop_type}. Opciones: {', '.join(CROP_PROFILES)}",
        )

    if reference_model_id:
        with _lock:
            ref = _references.get(reference_model_id)
        if ref is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Modelo de referencia no encontrado: {reference_model_id}",
            )
        if ref.get("crop_type", crop_type) != crop_type:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"El modelo de referencia es de cultivo '{ref['crop_type']}', no '{crop_type}'.",
            )
        # Con otro umbral cambiarian los pixeles de vegetacion y el cambio por zona no seria comparable
        kmeans_k = int(ref["kmeans_k"])
        vegetation_threshold = float(ref.get("vegetation_threshold", vegetation_threshold))

    job_id = str(uuid.uuid4())
    tif_path = UPLOAD_DIR / f"{job_id}_{filename}"
    await _save_upload(file, tif_path)

    params = AnalysisParams(
        kmeans_k=kmeans_k,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        vegetation_threshold=vegetation_threshold,
        crop_type=crop_type,
        reference_model_id=reference_model_id,
    )
    _register_job(job_id, filename, reference_model_id, crop_type)
    threading.Thread(target=_run_pipeline, args=(job_id, tif_path, params), daemon=True).start()

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "job_id": job_id,
            "status": "queued",
            "crop_type": crop_type,
            "reference_model_id": reference_model_id,
            "message": "Pipeline iniciado. Consultar GET /api/v1/status/{job_id}",
        },
    )


@router.get("/crops", summary="Cultivos soportados y sus rangos de NDVI")
def list_crops():
    return {
        key: {
            "nombre": p["nombre"],
            "ndvi_ranges": p["ndvi_ranges"],
            "inspection_range": p["inspection_range"],
            "analisis_por_palma": p["palm_analysis"] is not None,
            "fuente": p["fuente"],
        }
        for key, p in CROP_PROFILES.items()
    }


@router.get("/status/{job_id}", response_model=JobStatusResponse, summary="Estado de un job")
def get_status(job_id: str):
    job = _get_job_or_404(job_id)
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
        error=job.get("error"),
        source_file=job.get("source_file"),
        crop_type=job.get("crop_type"),
        reference_model_id=job.get("reference_model_id"),
    )


@router.get("/result/{job_id}", summary="Resultado JSON completo")
def get_result(job_id: str):
    job = _get_completed_job(job_id)
    return JSONResponse(content=job["result_json"])


@router.get("/result/{job_id}/geojson", summary="Zonas FCM en GeoJSON (QGIS)")
def get_geojson(job_id: str):
    job = _get_completed_job(job_id)
    return JSONResponse(content=job["result_geojson"], media_type="application/geo+json")


@router.get("/result/{job_id}/patterns", summary="Patrones espaciales DBSCAN en GeoJSON")
def get_patterns(job_id: str):
    job = _get_completed_job(job_id)
    return JSONResponse(content=job["result_patterns_geojson"], media_type="application/geo+json")


@router.get("/result/{job_id}/inspection", summary="Zonas de inspeccion (rango de atencion de NDVI) en GeoJSON")
def get_inspection(job_id: str):
    job = _get_completed_job(job_id)
    return JSONResponse(content=job["result_inspection_geojson"], media_type="application/geo+json")


@router.get("/result/{job_id}/palms", summary="Palmas detectadas, a inspeccionar y posibles faltantes en GeoJSON")
def get_palms(job_id: str):
    job = _get_completed_job(job_id)
    if job.get("result_palms_geojson") is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Este analisis no incluye conteo de palmas (crop_type distinto de 'palma').",
        )
    return JSONResponse(content=job["result_palms_geojson"], media_type="application/geo+json")


@router.get("/result/{job_id}/uncertainty", summary="Mapa de incertidumbre: pertenencia de cada pixel a su zona (GeoTIFF, %)")
def get_uncertainty(job_id: str):
    job = _get_completed_job(job_id)
    path = job.get("result_uncertainty_path")
    if not path or not Path(path).exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mapa de incertidumbre no disponible.")
    return FileResponse(path, media_type="image/tiff", filename=f"incertidumbre_{job_id}.tif")


@router.get(
    "/result/{job_id}/accuracy-sample",
    summary="Puntos de verificacion estratificados por zona (sin la zona del modelo)",
)
def get_accuracy_sample(job_id: str):
    job = _get_completed_job(job_id)
    return JSONResponse(content=job["accuracy_geojson"], media_type="application/geo+json")


@router.post(
    "/result/{job_id}/accuracy",
    summary="Exactitud tematica: matriz de confusion, exactitudes y Kappa",
)
def post_accuracy(job_id: str, body: AccuracyRequest):
    """
    Compara la clase que el experto observo en cada punto con la zona del mapa.
    Con zone_labels, cada zona se traduce primero a la categoria del experto.
    """
    job = _get_completed_job(job_id)
    points = {str(p["punto_id"]): p for p in job["accuracy_points"]}
    unknown = set(body.referencias) - set(points)
    if unknown or not body.referencias:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Puntos inexistentes o vacios: {sorted(unknown)}",
        )
    labels = body.zone_labels or {}
    map_classes = [
        labels.get(str(points[pid]["zona"]), f"zona {points[pid]['zona']}") for pid in body.referencias
    ]
    return accuracy_assessment(map_classes, list(body.referencias.values()))


@router.get("/result/{job_id}/summary", summary="Resumen en texto plano")
def get_summary(job_id: str):
    job = _get_completed_job(job_id)
    return JSONResponse(content={"summary": job["result_summary"]})


@router.post(
    "/reference/{job_id}",
    summary="Registrar un job como linea base validada por el experto",
    status_code=status.HTTP_201_CREATED,
)
def create_reference_model(
    job_id: str,
    name: str = Form(default="Linea base experto agronomo"),
    validated_by: str | None = Form(default=None, description="Nombre del experto que valido las zonas"),
    zone_labels: str | None = Form(
        default=None,
        description='JSON con el nombre de cada zona segun el experto, p. ej. {"2": "Palma vigorosa"}',
    ),
    notes: str | None = Form(default=None),
):
    """
    Guarda centroides, escalado, umbral de NDVI, estadisticas y etiquetas del experto.
    Los vuelos siguientes del mismo lote se clasifican con estos centroides congelados.
    """
    job = _get_completed_job(job_id)

    cluster_ids = {str(z["cluster_id"]) for z in job["result_json"].get("kmeans_zones", [])}
    labels = {}
    if zone_labels:
        try:
            labels = {str(k): str(v) for k, v in json.loads(zone_labels).items()}
        except (ValueError, AttributeError):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail='zone_labels debe ser un JSON tipo {"2": "Palma vigorosa"}',
            )
        unknown = set(labels) - cluster_ids
        if unknown:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Zonas inexistentes en el job: {sorted(unknown)}. Zonas validas: {sorted(cluster_ids)}",
            )

    artifacts = job["model_artifacts"]
    ref_id = str(uuid.uuid4())
    ref = {
        "reference_id": ref_id,
        "source_job_id": job_id,
        "name": name,
        "notes": notes,
        "validated_by": validated_by,
        "zone_labels": labels,
        "crop_type": job.get("crop_type", "generico"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_file": job.get("source_file"),
        "fcm_centers": artifacts["fcm_centers"],
        "scaler_mean": artifacts["scaler_mean"],
        "scaler_scale": artifacts["scaler_scale"],
        "kmeans_k": artifacts["kmeans_k"],
        "vegetation_threshold": job["result_json"]["analysis_params"]["vegetation_threshold"],
        "distance_p99": artifacts["distance_p99"],
        "feature_names": artifacts["feature_names"],
        "band_stats": job["result_json"].get("vegetation_indices", {}),
        "kmeans_zones": job["result_json"].get("kmeans_zones", []),
    }

    (REF_DIR / f"{ref_id}.json").write_text(json.dumps(ref, indent=2, ensure_ascii=False), encoding="utf-8")
    with _lock:
        _references[ref_id] = ref

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "reference_id": ref_id,
            "name": name,
            "crop_type": ref["crop_type"],
            "validated_by": validated_by,
            "zone_labels": labels,
            "source_job_id": job_id,
            "created_at": ref["created_at"],
            "message": "Linea base registrada.",
        },
    )


@router.get("/references", summary="Listar las lineas base registradas")
def list_reference_models():
    with _lock:
        refs = [
            {
                "reference_id": r["reference_id"],
                "name": r["name"],
                "crop_type": r.get("crop_type"),
                "validated_by": r.get("validated_by"),
                "zone_labels": r.get("zone_labels"),
                "source_job_id": r["source_job_id"],
                "source_file": r["source_file"],
                "created_at": r["created_at"],
            }
            for r in _references.values()
        ]
    return {"total": len(refs), "references": refs}


@router.get("/reference/{reference_id}", summary="Detalle de una linea base")
def get_reference_model(reference_id: str):
    with _lock:
        ref = _references.get(reference_id)
    if ref is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modelo de referencia no encontrado: {reference_id}",
        )
    return ref


@router.get("/health", summary="Verificacion de disponibilidad")
def health():
    with _lock:
        active = sum(1 for j in _jobs.values() if j["status"] == "processing")
        queued = sum(1 for j in _jobs.values() if j["status"] == "queued")
        n_refs = len(_references)
    return {
        "status": "ok",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "active_jobs": active,
        "queued_jobs": queued,
        "registered_references": n_refs,
    }


async def _save_upload(file: UploadFile, dest: Path):
    # Se escribe por bloques para no tener el ortomosaico completo en memoria.
    max_bytes = MAX_FILE_MB * 1024 ** 2
    written = 0
    try:
        with open(dest, "wb") as out:
            while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"El archivo supera el maximo de {MAX_FILE_MB} MB",
                    )
                out.write(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise


def _run_pipeline(job_id: str, tif_path: Path, params: AnalysisParams):
    try:
        with _job_slots:
            _update_job(job_id, status="processing")

            profile = get_profile(params.crop_type)
            ref_model = None
            if params.reference_model_id:
                with _lock:
                    ref_model = _references.get(params.reference_model_id)

            processed = RasterProcessor(tif_path, params.vegetation_threshold).process()

            seg_result = SpatialSegmenter(
                kmeans_k=params.kmeans_k,
                dbscan_eps=params.dbscan_eps,
                dbscan_min_samples=params.dbscan_min_samples,
                # resolucion sin redondear para que las areas cuadren
                resolution_m=abs(processed.transform.a),
                ndvi_ranges=profile["ndvi_ranges"],
            ).segment(processed, reference_model=ref_model)

            condition_report, inspection_geojson = ConditionReporter(profile, params.crop_type).build(processed)

            palm_report, palms_geojson = None, None
            if profile["palm_analysis"]:
                palm_report, palms_geojson = PalmAnalyzer(**profile["palm_analysis"]).analyze(tif_path, processed)

            art = seg_result.model_artifacts
            explainability = ZoneExplainer().explain(
                processed.feature_stack, seg_result.kmeans_labels, art["fcm_centers"],
                art["scaler_mean"], art["scaler_scale"], fcm_m=art["fcm_m"],
                memberships=seg_result.max_membership,
            )
            points = stratified_sample(processed, seg_result.kmeans_labels)

            serializer = ResponseSerializer(
                processed=processed,
                result=seg_result,
                source_file=tif_path.name.removeprefix(f"{job_id}_"),
                condition_report=condition_report,
                palm_report=palm_report,
                explainability=explainability,
                job_id=job_id,
                extra_params={
                    "vegetation_threshold": params.vegetation_threshold,
                    "crop_type": params.crop_type,
                },
            )
            job_dir = RESULTS_DIR / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            uncertainty_path = serializer.write_uncertainty_tif(job_dir / "incertidumbre.tif")

            def plain(obj):
                return json.loads(json.dumps(obj, default=serializer._json_default))

            _update_job(
                job_id,
                status="completed",
                result_json=plain(serializer.to_json()),
                result_geojson=plain(serializer.to_geojson()),
                result_patterns_geojson=plain(serializer.to_patterns_geojson()),
                result_inspection_geojson=plain(inspection_geojson),
                result_palms_geojson=plain(palms_geojson) if palms_geojson else None,
                result_summary=serializer.to_summary(),
                result_uncertainty_path=str(uncertainty_path) if uncertainty_path else None,
                accuracy_points=points,
                accuracy_geojson=blind_geojson(points, processed.metadata.crs),
                model_artifacts=seg_result.model_artifacts,
            )
    except ValueError as e:
        _update_job(job_id, status="failed", error=str(e))
    except Exception as e:
        _update_job(job_id, status="failed", error=f"Error inesperado: {type(e).__name__}: {e}")
    finally:
        tif_path.unlink(missing_ok=True)


def _register_job(job_id: str, source_file: str, reference_model_id: str | None, crop_type: str):
    with _lock:
        _jobs[job_id] = {
            "status": "queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_file": source_file,
            "crop_type": crop_type,
            "reference_model_id": reference_model_id,
        }


def _update_job(job_id: str, **fields):
    with _lock:
        if job_id not in _jobs:
            return
        _jobs[job_id].update(fields)
        if fields.get("status") in ("completed", "failed"):
            _jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()


def _get_job_or_404(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job no encontrado: {job_id}")
    return job


def _get_completed_job(job_id: str):
    job = _get_job_or_404(job_id)
    if job["status"] in ("queued", "processing"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"El job esta en estado '{job['status']}'. Reintentar en unos segundos.",
        )
    if job["status"] == "failed":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"El job fallo: {job.get('error')}",
        )
    return job