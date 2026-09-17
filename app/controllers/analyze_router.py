import os
import shutil
import tempfile
import threading
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

# Almacén en memoria de jobs
# TODO: Para el despliegue final, deberíamos cambiar este diccionario en memoria por Redis para no perder los jobs si se reinicia el servidor.
_jobs = {}
_jobs_lock = threading.Lock()

# Directorio temporal
UPLOAD_DIR = Path(tempfile.gettempdir()) / "geoai_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_FILE_MB = 2000


# ------------------------------------------------------------------
# Schemas Pydantic (request / response)
# ------------------------------------------------------------------

class AnalysisParams(BaseModel):
    kmeans_k: int = Field(default=4, ge=2, le=10,
                          description="Número de clusters K-Means (2–10)")
    dbscan_eps: float = Field(default=0.3, gt=0,
                              description="Radio de vecindad DBSCAN")
    dbscan_min_samples: int = Field(default=50, ge=5,
                                    description="Mínimo de puntos por cluster DBSCAN")
    vegetation_threshold: float = Field(default=0.1, ge=-1, le=1,
                                        description="NDVI mínimo para considerar vegetación")


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    created_at: str
    completed_at: str | None = None
    error: str | None = None
    source_file: str | None = None


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
    file: UploadFile = File(..., description="Ortomosaico GeoTIFF de 5 bandas (Pix4Dfields/DJI)"),
    kmeans_k: int = Form(default=4, ge=2, le=10),
    dbscan_eps: float = Form(default=0.3, gt=0),
    dbscan_min_samples: int = Form(default=50, ge=5),
):
    """
    Recibe un ortomosaico GeoTIFF multiespectral y ejecuta:
    1. Preprocesamiento raster (NDVI / SAVI / GNDVI)
    2. Segmentación K-Means
    3. Detección de patrones DBSCAN

    Retorna un job_id para consultar el resultado cuando termine.
    """
    # Validar extensión
    if not file.filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Solo se aceptan archivos .tif o .tiff",
        )

    # Validar tamaño máximo
    contents = await file.read()
    size_mb = len(contents) / (1024 ** 2)
    if size_mb > MAX_FILE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Archivo demasiado grande ({size_mb:.1f} MB). Máximo: {MAX_FILE_MB} MB",
        )

    # Guardar en disco temporal
    tif_path = UPLOAD_DIR / f"{_new_job_id()}_{file.filename}"
    tif_path.write_bytes(contents)
    del contents  # liberar RAM

    # Crear job
    job_id = _new_job_id()
    _register_job(job_id, str(file.filename))

    # Ejecutar pipeline en hilo para no bloquear el event loop
    params = AnalysisParams(
        kmeans_k=kmeans_k,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
    )
    t = threading.Thread(
        target=_run_pipeline,
        args=(job_id, tif_path, params),
        daemon=True,
    )
    print(f"Arrancando hilo para el job {job_id}")
    t.start()

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "job_id": job_id,
            "status": "processing",
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
    )


# ------------------------------------------------------------------
# GET /result/{job_id}
# ------------------------------------------------------------------

@router.get(
    "/result/{job_id}",
    summary="Obtener resultado JSON completo",
)
def get_result(job_id: str):
    """
    Retorna el resultado completo del análisis en formato JSON.
    Solo disponible cuando status == 'completed'.
    """
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
    """
    Retorna las zonas detectadas como FeatureCollection GeoJSON
    con CRS EPSG:32618. Importable directamente en QGIS.
    """
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
    """Resumen legible del análisis. Útil para logs y reportes."""
    job = _get_job_or_404(job_id)
    _assert_completed(job)
    return JSONResponse(content={"summary": job["result_summary"]})


# ------------------------------------------------------------------
# GET /health
# ------------------------------------------------------------------

@router.get("/health", summary="Verificación de disponibilidad")
def health():
    """Endpoint para monitoreo. Retorna 200 si el servicio está activo."""
    return {
        "status": "ok",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "active_jobs": len([j for j in _jobs.values() if j["status"] == "processing"]),
    }


# ------------------------------------------------------------------
# Pipeline (ejecutado en hilo)
# ------------------------------------------------------------------

def _run_pipeline(job_id: str, tif_path: Path, params: AnalysisParams):
    """
    Orquesta los tres componentes del Model-View:
      RasterProcessor → SpatialSegmenter → ResponseSerializer

    Al finalizar libera el archivo temporal y actualiza el job.
    """
    try:
        _update_job(job_id, status="processing")

        # 1. MODEL — Procesamiento raster
        processor = RasterProcessor(tif_path)
        processor.VEGETATION_THRESHOLD = params.vegetation_threshold
        processed = processor.process()

        # 2. MODEL — Segmentación espacial
        segmenter = SpatialSegmenter(
            kmeans_k=params.kmeans_k,
            dbscan_eps=params.dbscan_eps,
            dbscan_min_samples=params.dbscan_min_samples,
            resolution_m=processed.metadata.resolution_m,
        )
        seg_result = segmenter.segment(processed)

        # 3. VIEW — Serialización
        serializer = ResponseSerializer(
            processed=processed,
            result=seg_result,
            source_file=tif_path.name,
        )

        # Serializar a string y volver a parsear para eliminar tipos numpy
        # antes de guardar en el store — JSONResponse usa el encoder estándar
        import json as _json
        _nd = serializer._json_default
        result_json    = _json.loads(_json.dumps(serializer.to_json(),    default=_nd))
        result_geojson = _json.loads(_json.dumps(serializer.to_geojson(), default=_nd))

        _update_job(
            job_id,
            status="completed",
            result_json=result_json,
            result_geojson=result_geojson,
            result_summary=serializer.to_summary(),
        )

    except ValueError as e:
        _update_job(job_id, status="failed", error=str(e))

    except Exception as e:
        _update_job(job_id, status="failed", error=f"Error inesperado: {e}")

    finally:
        # Liberar archivo temporal del disco
        try:
            tif_path.unlink(missing_ok=True)
        except Exception:
            pass


# ------------------------------------------------------------------
# Helpers de gestión de jobs
# ------------------------------------------------------------------

def _new_job_id() -> str:
    import uuid
    return str(uuid.uuid4())


def _register_job(job_id: str, source_file: str):
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_file": source_file,
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
            detail="El análisis aún está en proceso. Reintenta en unos segundos.",
        )
    if job["status"] == "failed":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"El job falló: {job.get('error')}",
        )
    if job["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Estado inesperado: {job['status']}",
        )