from pathlib import Path

from pydantic_settings import BaseSettings

# Root project (folder yang berisi main.py), supaya path model tidak
# bergantung pada lokasi folder di disk.
PROJECT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # Path ke model dasar hasil merge SLERP (model utuh)
    base_model_path: str = str(PROJECT_DIR / "model" / "slerp_model")

    # Path ke adapter LoRA
    adapter_model_path: str = str(PROJECT_DIR / "model" / "lora_model")

    # Path ke model full slerp-sft (hasil `python -m utils.merge_lora`).
    # Jika folder ini ada dan use_merged_model=True, model ini yang dimuat
    # langsung tanpa PEFT; jika tidak, kembali ke base + adapter.
    merged_model_path: str = str(PROJECT_DIR / "model" / "slerp_sft_model")
    use_merged_model: bool = True

    # Nama model yang ditampilkan di UI dan response API.
    model_display_name: str = "slerp-sft-model"

    # Wajib GPU: server menolak start jika CUDA tidak tersedia, supaya
    # model tidak diam-diam berjalan (lambat) di CPU.
    require_gpu: bool = True

    # Index GPU yang dipakai (0 = GPU pertama). Seluruh model dimuat ke GPU ini.
    gpu_index: int = 0

    # Paksa pakai CPU walau GPU tersedia (hanya untuk debugging;
    # mengabaikan require_gpu).
    force_cpu: bool = False

    # Batas panjang token yang dihasilkan model.
    max_new_tokens: int = 1024

    # Parameter sampling.
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

    # Host & port server, dibaca main.py saat dijalankan langsung
    # (python main.py), atau diabaikan kalau pakai `uvicorn main:app`.
    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_file = str(PROJECT_DIR / ".env")
        env_file_encoding = "utf-8"


settings = Settings()


def merged_model_available() -> bool:
    return settings.use_merged_model and (Path(settings.merged_model_path) / "config.json").exists()


def validate_paths() -> None:
    """Pastikan folder model benar-benar ada sebelum server dijalankan."""
    if merged_model_available():
        return

    for label, path in [
        ("base_model_path", settings.base_model_path),
        ("adapter_model_path", settings.adapter_model_path),
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Path '{label}' tidak ditemukan: {path}\n"
                "Periksa isi file .env atau core/config.py, atau buat model "
                "gabungan dengan `python -m utils.merge_lora`."
            )
