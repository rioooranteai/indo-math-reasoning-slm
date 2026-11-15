from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Soal matematika yang akan dijawab model")
    model: str | None = Field(default=None, description="Nama model (opsional, saat ini hanya satu model aktif)")


class InferenceResponse(BaseModel):
    answer: str
    chain_of_thought: str
    model: str
    elapsed_time: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    device: str
    gpu_name: str | None = None
    model_loaded: bool
