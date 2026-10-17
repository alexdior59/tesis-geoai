import json
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.models.raster_processor import RasterProcessor
from app.models.spatial_segmenter import SpatialSegmenter
from app.views.response_serializer import ResponseSerializer

router = APIRouter(prefix="/api/v1", tags=["GeoAI Analysis"])

# Almacen en memoria de jobs y de modelos de referencia (Ground Truth / Linea Base)
_jobs = {}
_references = {}
_jobs_lock = threading.Lock()

# Directorios de trabajo
UPLOAD_DIR = Path(tempfile.gettempdir()) / "geoai_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
REF_DIR = Path(tempfile.gettempdir()) / "geoai_references"
REF_DIR.mkdir(exist_ok=True)

MAX_FILE_MB = 2000

# ------------------------------------------------------------------
# Schemas Pydantic (request / response)
# ------------------------------------------------------------------
class AnalysisParams(BaseModel):
    kmeans_k: int = Field(default=4, ge=2, le=10,
                          description="Numero de clusters FCM (2-10)")
    dbscan_eps: float = Field(default=0.5, gt=0,
                              description="Radio de vecindad DBSCAN en metros")
    dbscan_min_samples: int = Field(default=30, ge=5,
                                    description="Minimo de puntos por cluster DBSCAN")
    vegetation_threshold: float = Field(default=0.1, ge=-1, le=1,
                                        description="NDVI minimo para considerar vegetacion")
    reference_model_id: str | None = Field(default=None,
                                           description="ID del modelo de referencia experto para calcular drift_report")

class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    created_at: str
    completed_at: str | None = None
    error: str | None = None
    source_file: str | None = None
    reference_model_id: str | None = None

# ------------------------------------------------------------------
# POST /analyze
# ------------------------------------------------------------------
@router.post(
    "/analyze",
    summary="Analizar imagen multiespectral GeoTIFF",
    response_description="Job ID para consultar el resultado",
    status_code=status.HTTP_202_ACCEPTED,
)
async def analyze(
    file: UploadFile = File(..., description="Ortomosaico GeoTIFF de Pix4Dfields / DJI Mavic 3M"),
    kmeans_k: int = Form(default=4, ge=2, le=10),
    dbscan_eps: float = Form(default=0.5, gt=0),
    dbscan_min_samples: int = Form(default=30, ge=5),
    vegetation_threshold: float = Form(default=0.1, ge=-1, le=1),
    reference_model_id: str | None = Form(default=None),
):
    """
    Recibe un ortomosaico GeoTIFF multiespectral y ejecuta:
    1. Preprocesamiento raster (NDVI / NDRE / GNDVI) con deteccion dinamica de bandas
    2. Macro-zonificacion con Fuzzy C-Means (FCM)
    3. Deteccion de patrones espaciales DBSCAN en metros
    4. Comparacion contra modelo de referencia (drift_report) si se envia reference_model_id
    """
    if not file.filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Solo se aceptan archivos .tif o .tiff",
        )

    if reference_model_id and reference_model_id not in _references:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modelo de referencia no encontrado: {reference_model_id}",
        )

    contents = await file.read()
    size_mb = len(contents) / (1024 ** 2)
    if size_mb > MAX_FILE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Archivo demasiado grande ({size_mb:.1f} MB). Maximo: {MAX_FILE_MB} MB",
        )

    job_id = _new_job_id()
    tif_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    tif_path.write_bytes(contents)
    del contents

    _register_job(job_id, str(file.filename), reference_model_id)

    params = AnalysisParams(
        kmeans_k=kmeans_k,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        vegetation_threshold=vegetation_threshold,
        reference_model_id=reference_model_id,
    )
    t = threading.Thread(
        target=_run_pipeline,
        args=(job_id, tif_path, params),
        daemon=True,
    )
    t.start()

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "job_id": job_id,
            "status": "processing",
            "reference_model_id": reference_model_id,
            "message": "Pipeline iniciado. Consulta GET /api/v1/status/{job_id}",
        },
    )

# ------------------------------------------------------------------
# GET /status/{job_id}
# ------------------------------------------------------------------
@router.get(
    "/status/{job_id}",
    response_model=JobStatusResponse,
    summary="Consultar estado de un job",
)
def get_status(job_id: str):
    """Retorna el estado actual del job: queued / processing / completed / failed."""
    job = _get_job_or_404(job_id)
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
        error=job.get("error"),
        source_file=job.get("source_file"),
        reference_model_id=job.get("reference_model_id"),
    )

# ------------------------------------------------------------------
# GET /result/{job_id}
# ------------------------------------------------------------------
@router.get(
    "/result/{job_id}",
    summary="Obtener resultado JSON completo",
)
def get_result(job_id: str):
    job = _get_job_or_404(job_id)
    _assert_completed(job)
    return JSONResponse(content=job["result_json"])

# ------------------------------------------------------------------
# GET /result/{job_id}/geojson
# ------------------------------------------------------------------
@router.get(
    "/result/{job_id}/geojson",
    summary="Obtener resultado en formato GeoJSON (compatible QGIS)",
)
def get_geojson(job_id: str):
    job = _get_job_or_404(job_id)
    _assert_completed(job)
    return JSONResponse(
        content=job["result_geojson"],
        media_type="application/geo+json",
    )

# ------------------------------------------------------------------
# GET /result/{job_id}/summary
# ------------------------------------------------------------------
@router.get(
    "/result/{job_id}/summary",
    summary="Obtener resumen en texto plano",
)
def get_summary(job_id: str):
    job = _get_job_or_404(job_id)
    _assert_completed(job)
    return JSONResponse(content={"summary": job["result_summary"]})

# ------------------------------------------------------------------
# ENDPOINTS DE MODELOS DE REFERENCIA (TRAZABILIDAD / EXPERTO)
# ------------------------------------------------------------------
@router.post(
    "/reference/{job_id}",
    summary="Promover un job completado a Modelo de Referencia (Linea Base Experto)",
    status_code=status.HTTP_201_CREATED,
)
def create_reference_model(
    job_id: str,
    name: str = Form(default="Linea Base Experto Agronomo"),
    notes: str | None = Form(default=None),
):
    """
    Guarda los centroides FCM, parametros de escalado y estadisticas de un job
    validado por el experto para usarlo como patron de referencia en futuros vuelos.
    """
    job = _get_job_or_404(job_id)
    _assert_completed(job)

    ref_id = _new_job_id()
    ref_data = {
        "reference_id": ref_id,
        "source_job_id": job_id,
        "name": name,
        "notes": notes,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_file": job.get("source_file"),
        "fcm_centers": job["model_artifacts"]["fcm_centers"],
        "scaler_mean": job["model_artifacts"]["scaler_mean"],
        "scaler_scale": job["model_artifacts"]["scaler_scale"],
        "kmeans_k": job["model_artifacts"]["kmeans_k"],
        "band_stats": job["result_json"].get("vegetation_indices", {}),
        "kmeans_zones": job["result_json"].get("kmeans_zones", []),
    }

    with _jobs_lock:
        _references[ref_id] = ref_data

    ref_file = REF_DIR / f"{ref_id}.json"
    ref_file.write_text(json.dumps(ref_data, indent=2, ensure_ascii=False), encoding="utf-8")

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "reference_id": ref_id,
            "name": name,
            "source_job_id": job_id,
            "created_at": ref_data["created_at"],
            "message": "Modelo de referencia registrado con exito.",
        },
    )

@router.get(
    "/references",
    summary="Listar todos los modelos de referencia registrados",
)
def list_reference_models():
    with _jobs_lock:
        refs = [
            {
                "reference_id": r["reference_id"],
                "name": r["name"],
                "source_job_id": r["source_job_id"],
                "source_file": r["source_file"],
                "created_at": r["created_at"],
            }
            for r in _references.values()
        ]
    return {"total": len(refs), "references": refs}

@router.get(
    "/reference/{reference_id}",
    summary="Consultar detalle de un modelo de referencia especifico",
)
def get_reference_model(reference_id: str):
    with _jobs_lock:
        ref = _references.get(reference_id)
    if not ref:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modelo de referencia no encontrado: {reference_id}",
        )
    return ref

# ------------------------------------------------------------------
# GET /health
# ------------------------------------------------------------------
@router.get("/health", summary="Verificacion de disponibilidad")
def health():
    return {
        "status": "ok",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "active_jobs": len([j for j in _jobs.values() if j["status"] == "processing"]),
        "registered_references": len(_references),
    }

# ------------------------------------------------------------------
# Pipeline (ejecutado en hilo)
# ------------------------------------------------------------------
def _run_pipeline(job_id: str, tif_path: Path, params: AnalysisParams):
    try:
        _update_job(job_id, status="processing")

        ref_model = None
        if params.reference_model_id:
            with _jobs_lock:
                ref_model = _references.get(params.reference_model_id)

        processor = RasterProcessor(tif_path)
        processor.VEGETATION_THRESHOLD = params.vegetation_threshold
        processed = processor.process()

        segmenter = SpatialSegmenter(
            kmeans_k=params.kmeans_k,
            dbscan_eps=params.dbscan_eps,
            dbscan_min_samples=params.dbscan_min_samples,
            resolution_m=processed.metadata.resolution_m,
        )
        seg_result = segmenter.segment(processed, reference_model=ref_model)

        serializer = ResponseSerializer(
            processed=processed,
            result=seg_result,
            source_file=tif_path.name,
        )

        _nd = serializer._json_default
        result_json = json.loads(json.dumps(serializer.to_json(), default=_nd))
        result_geojson = json.loads(json.dumps(serializer.to_geojson(), default=_nd))

        _update_job(
            job_id,
            status="completed",
            result_json=result_json,
            result_geojson=result_geojson,
            result_summary=serializer.to_summary(),
            model_artifacts=seg_result.model_artifacts,
        )

    except ValueError as e:
        _update_job(job_id, status="failed", error=str(e))
    except Exception as e:
        _update_job(job_id, status="failed", error=f"Error inesperado: {e}")
    finally:
        try:
            tif_path.unlink(missing_ok=True)
        except Exception:
            pass

# ------------------------------------------------------------------
# Helpers de gestion de jobs
# ------------------------------------------------------------------
def _new_job_id() -> str:
    return str(uuid.uuid4())

def _register_job(job_id: str, source_file: str, reference_model_id: str | None = None):
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_file": source_file,
            "reference_model_id": reference_model_id,
        }

def _update_job(job_id: str, **kwargs):
    with _jobs_lock:
        if job_id not in _jobs:
            return
        _jobs[job_id].update(kwargs)
        if kwargs.get("status") in ("completed", "failed"):
            _jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()

def _get_job_or_404(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job no encontrado: {job_id}",
        )
    return job

def _assert_completed(job: dict):
    if job["status"] == "processing":
        raise HTTPException(
            status_code=status.HTTP_202_ACCEPTED,
            detail="El analisis aun esta en proceso. Reintenta en unos segundos.",
        )
    if job["status"] == "failed":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"El job fallo: {job.get('error')}",
        )
    if job["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Estado inesperado: {job['status']}",
        )