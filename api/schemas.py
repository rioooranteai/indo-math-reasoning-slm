"""Request and response bodies of the JSON API."""

from pydantic import BaseModel, Field


class SolveRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, max_length=4000,
        description="Soal cerita matematika dalam bahasa Indonesia.",
    )


class SolveResponse(BaseModel):
    answer: str = Field(description="Jawaban akhir; kosong jika tidak ditemukan.")
    chain_of_thought: str = Field(description="Langkah penyelesaian lengkap dari model.")
    model: str
    elapsed_time: float = Field(description="Lama inferensi dalam detik.")
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str | None
    gpu_name: str | None
