import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.models.raster_processor import RasterProcessor
from app.models.spatial_segmenter import SpatialSegmenter
from app.views.response_serializer import ResponseSerializer

router = APIRouter(prefix="/api/v1", tags=["GeoAI Analysis"])

# En Docker GEOAI_DATA_DIR apunta a un volumen montado para que las referencias
# sobrevivan a reinicios del contenedor.
DATA_DIR = Path(os.getenv("GEOAI_DATA_DIR", tempfile.gettempdir()))
UPLOAD_DIR = DATA_DIR / "geoai_uploads"
REF_DIR = DATA_DIR / "geoai_references"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
REF_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_MB = 2000
UPLOAD_CHUNK_BYTES = 16 * 1024 ** 2
# Un ortomosaico de ~1.4 GB ocupa varios GB de RAM al procesarse, por eso por
# defecto solo corre un job a la vez y los demas quedan en cola.
MAX_CONCURRENT_JOBS = int(os.getenv("GEOAI_MAX_CONCURRENT_JOBS", "1"))

_jobs = {}
_references = {}
_lock = threading.Lock()
_job_slots = threading.Semaphore(MAX_CONCURRENT_JOBS)


def _load_references():
    for ref_file in REF_DIR.glob("*.json"):
        try:
            ref = json.loads(ref_file.read_text(encoding="utf-8"))
            _references[ref["reference_id"]] = ref
        except (OSError, ValueError, KeyError) as e:
            print(f"No se pudo cargar la referencia {ref_file.name}: {e}")


_load_references()


class AnalysisParams(BaseModel):
    kmeans_k: int
    dbscan_eps: float
    dbscan_min_samples: int
    vegetation_threshold: float
    reference_model_id: str | None = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    created_at: str
    completed_at: str | None = None
    error: str | None = None
    source_file: str | None = None
    reference_model_id: str | None = None


@router.post(
    "/analyze",
    summary="Analizar un ortomosaico multiespectral GeoTIFF",
    status_code=status.HTTP_202_ACCEPTED,
)
async def analyze(
    file: UploadFile = File(..., description="Ortomosaico GeoTIFF de Pix4Dfields / DJI Mavic 3M"),
    kmeans_k: int = Form(default=4, ge=2, le=10, description="Numero de clusters FCM"),
    dbscan_eps: float = Form(default=0.5, gt=0, description="Radio de vecindad DBSCAN en metros"),
    dbscan_min_samples: int = Form(default=30, ge=5),
    vegetation_threshold: float = Form(default=0.1, ge=-1, le=1, description="NDVI minimo de vegetacion"),
    reference_model_id: str | None = Form(default=None),
):
    """
    Calcula NDVI/NDRE/GNDVI, zonifica con FCM y busca focos con DBSCAN.
    Si se envia reference_model_id se usan el K y el umbral de NDVI de la
    referencia (para que la comparacion sea valida) y se genera el drift_report.
    """
    filename = Path(file.filename or "").name
    if not filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Solo se aceptan archivos .tif o .tiff",
        )

    if reference_model_id:
        with _lock:
            ref = _references.get(reference_model_id)
        if ref is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Modelo de referencia no encontrado: {reference_model_id}",
            )
        kmeans_k = ref["kmeans_k"]
        vegetation_threshold = ref.get("vegetation_threshold", vegetation_threshold)

    job_id = str(uuid.uuid4())
    tif_path = UPLOAD_DIR / f"{job_id}_{filename}"
    await _save_upload(file, tif_path)

    params = AnalysisParams(
        kmeans_k=kmeans_k,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        vegetation_threshold=vegetation_threshold,
        reference_model_id=reference_model_id,
    )
    _register_job(job_id, filename, reference_model_id)
    threading.Thread(target=_run_pipeline, args=(job_id, tif_path, params), daemon=True).start()

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "job_id": job_id,
            "status": "queued",
            "reference_model_id": reference_model_id,
            "message": "Pipeline iniciado. Consultar GET /api/v1/status/{job_id}",
        },
    )


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
        reference_model_id=job.get("reference_model_id"),
    )


@router.get("/result/{job_id}", summary="Resultado JSON completo")
def get_result(job_id: str):
    job = _get_job_or_404(job_id)
    _assert_completed(job)
    return JSONResponse(content=job["result_json"])


@router.get("/result/{job_id}/geojson", summary="Resultado en GeoJSON para QGIS")
def get_geojson(job_id: str):
    job = _get_job_or_404(job_id)
    _assert_completed(job)
    return JSONResponse(content=job["result_geojson"], media_type="application/geo+json")


@router.get("/result/{job_id}/summary", summary="Resumen en texto plano")
def get_summary(job_id: str):
    job = _get_job_or_404(job_id)
    _assert_completed(job)
    return JSONResponse(content={"summary": job["result_summary"]})


@router.post(
    "/reference/{job_id}",
    summary="Registrar un job como modelo de referencia (linea base validada)",
    status_code=status.HTTP_201_CREATED,
)
def create_reference_model(
    job_id: str,
    name: str = Form(default="Linea base experto agronomo"),
    notes: str | None = Form(default=None),
):
    """Guarda centroides, escalado, umbral y estadisticas del job para comparar vuelos futuros."""
    job = _get_job_or_404(job_id)
    _assert_completed(job)

    artifacts = job["model_artifacts"]
    ref_id = str(uuid.uuid4())
    ref = {
        "reference_id": ref_id,
        "source_job_id": job_id,
        "name": name,
        "notes": notes,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_file": job.get("source_file"),
        "fcm_centers": artifacts["fcm_centers"],
        "scaler_mean": artifacts["scaler_mean"],
        "scaler_scale": artifacts["scaler_scale"],
        "kmeans_k": artifacts["kmeans_k"],
        "vegetation_threshold": job["result_json"]["analysis_params"]["vegetation_threshold"],
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
            "source_job_id": job_id,
            "created_at": ref["created_at"],
            "message": "Modelo de referencia registrado.",
        },
    )


@router.get("/references", summary="Listar modelos de referencia")
def list_reference_models():
    with _lock:
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


@router.get("/reference/{reference_id}", summary="Detalle de un modelo de referencia")
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
    # Se copia por bloques para no cargar el ortomosaico completo en memoria.
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

            ref_model = None
            if params.reference_model_id:
                with _lock:
                    ref_model = _references.get(params.reference_model_id)

            processed = RasterProcessor(tif_path, params.vegetation_threshold).process()

            segmenter = SpatialSegmenter(
                kmeans_k=params.kmeans_k,
                dbscan_eps=params.dbscan_eps,
                dbscan_min_samples=params.dbscan_min_samples,
                # se usa la resolucion sin redondear para que las areas cuadren
                resolution_m=abs(processed.transform.a),
            )
            seg_result = segmenter.segment(processed, reference_model=ref_model)

            serializer = ResponseSerializer(
                processed=processed,
                result=seg_result,
                job_id=job_id,
                source_file=tif_path.name.removeprefix(f"{job_id}_"),
                extra_params={"vegetation_threshold": params.vegetation_threshold},
            )
            to_plain = serializer.json_default
            result_json = json.loads(json.dumps(serializer.to_json(), default=to_plain))
            result_geojson = json.loads(json.dumps(serializer.to_geojson(), default=to_plain))

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
        _update_job(job_id, status="failed", error=f"Error inesperado: {type(e).__name__}: {e}")
    finally:
        tif_path.unlink(missing_ok=True)


def _register_job(job_id: str, source_file: str, reference_model_id: str | None):
    with _lock:
        _jobs[job_id] = {
            "status": "queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_file": source_file,
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


def _assert_completed(job: dict):
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