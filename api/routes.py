import logging

from fastapi import APIRouter, HTTPException

from models.schemas import InferenceRequest, InferenceResponse, HealthResponse
from services.model_service import model_service

logger = logging.getLogger("routes")

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        device=model_service.device,
        gpu_name=model_service.gpu_name,
        model_loaded=model_service.is_loaded,
    )


@router.post("/infer", response_model=InferenceResponse)
def infer(request: InferenceRequest) -> InferenceResponse:
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model belum siap, coba lagi sebentar.")

    logger.info("Menerima permintaan inferensi: %.60s...", request.question)
    try:
        result = model_service.generate(request.question)
        logger.info("Inferensi selesai dalam %.2fs", result["elapsed_time"])
    except Exception as exc:
        logger.exception("Inferensi gagal")
        raise HTTPException(status_code=500, detail=f"Inferensi gagal: {exc}") from exc

    return InferenceResponse(**result)