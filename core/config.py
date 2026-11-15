"""Application settings, read from environment variables or a `.env` file."""

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "model"

# Same Indonesian system prompt used for SFT data and evaluation (SYSTEM_PROMPT_ID).
SYSTEM_PROMPT_ID = (
    "Anda adalah asisten penalaran matematika. Selesaikan soal dengan cermat "
    "secara bertahap. Pada baris terakhir, tuliskan hanya jawaban numerik "
    "akhir dengan format: \\boxed{nilai}"
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Model files. The merged slerp-sft model is preferred when it exists;
    # otherwise the SLERP base model is loaded with the LoRA adapter on top.
    merged_model_path: Path = MODEL_DIR / "slerp_sft_model"
    base_model_path: Path = MODEL_DIR / "slerp_model"
    adapter_model_path: Path = MODEL_DIR / "lora_model"
    use_merged_model: bool = True
    model_display_name: str = "slerp-sft-model"

    # Hardware. With require_gpu the server refuses to start without CUDA
    # instead of silently running on the (much slower) CPU.
    require_gpu: bool = True
    gpu_index: int = 0
    force_cpu: bool = False

    # Generation. Greedy decoding matches the evaluation protocol.
    system_prompt: str = SYSTEM_PROMPT_ID
    max_new_tokens: int = 1024
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.9

    # Server, used by `python main.py`.
    host: str = "0.0.0.0"
    port: int = 8000

    @field_validator("merged_model_path", "base_model_path", "adapter_model_path")
    @classmethod
    def _relative_to_project(cls, path: Path) -> Path:
        # Relative paths in .env are taken from the project folder, not the
        # folder the server happens to be started from.
        return path if path.is_absolute() else PROJECT_DIR / path

    @property
    def merged_model_available(self) -> bool:
        return self.use_merged_model and (self.merged_model_path / "config.json").exists()

    def validate_model_paths(self) -> None:
        """Fail fast at startup if no usable model is on disk."""
        if self.merged_model_available:
            return

        missing = [
            path for path in (self.base_model_path, self.adapter_model_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Model tidak ditemukan: "
                + ", ".join(str(p) for p in missing)
                + "\nUnduh model dengan: hf download rantetadung/indo-math-slerp-sft-1.5b "
                "--local-dir model/slerp_sft_model"
            )


settings = Settings()
