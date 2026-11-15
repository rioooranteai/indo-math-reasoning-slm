"""JSON API: health check and solving one question."""

import logging
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, status

from api.schemas import HealthResponse, SolveRequest, SolveResponse
from services.model_service import model_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["inference"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=model_service.is_loaded,
        device=model_service.device,
        gpu_name=model_service.gpu_name,
    )


# A plain `def` endpoint runs in FastAPI's thread pool, so the blocking
# generation does not stall the event loop.
@router.post("/infer", response_model=SolveResponse)
def infer(request: SolveRequest) -> SolveResponse:
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model belum siap, coba lagi sebentar.",
        )

    try:
        solution = model_service.solve(request.question)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inferensi gagal: {exc}",
        ) from exc

    logger.info("Solved in %.2fs: %.60s", solution.elapsed_time, request.question)
    return SolveResponse(**asdict(solution))
