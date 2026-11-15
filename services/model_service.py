import re
import time
import logging
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from core.config import settings, merged_model_available
from core.gpu import (
    assert_all_on_gpu,
    best_gpu_dtype,
    gpu_name,
    log_gpu_info,
    log_vram_usage,
    require_cuda,
)

logger = logging.getLogger("model_service")

BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]*)\}")


def resolve_device() -> str:
    """Tentukan device tanpa melempar error (dipanggil saat import)."""
    if settings.force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return f"cuda:{settings.gpu_index}"
    return "cpu"


class ModelService:
    """Membungkus tokenizer + model, dan menyediakan satu method generate()."""

    def __init__(self) -> None:
        self.device: str = resolve_device()
        self.tokenizer = None
        self.model = None
        self.is_loaded: bool = False
        self.gpu_name: str | None = None

    def load(self) -> None:
        if settings.force_cpu:
            logger.warning("force_cpu=True: model dijalankan di CPU (mode debugging, lambat).")
            self.device = "cpu"
            dtype = torch.float32
            device_map = None
        elif torch.cuda.is_available() or settings.require_gpu:
            # require_cuda() melempar error yang jelas jika CUDA tidak ada,
            # sehingga server tidak diam-diam jatuh ke CPU.
            gpu = require_cuda(settings.gpu_index)
            self.device = str(gpu)
            self.gpu_name = gpu_name(settings.gpu_index)
            log_gpu_info(settings.gpu_index)
            dtype = best_gpu_dtype()
            # Seluruh layer dipetakan ke satu GPU: tidak ada offload ke CPU/disk.
            device_map = {"": settings.gpu_index}
        else:
            logger.warning("CUDA tidak tersedia dan require_gpu=False: memakai CPU.")
            self.device = "cpu"
            dtype = torch.float32
            device_map = None

        logger.info("Memuat model di device: %s (dtype %s)", self.device, dtype)

        if merged_model_available():
            # Model full slerp-sft: adapter LoRA sudah digabung ke bobot,
            # jadi cukup dimuat langsung tanpa PEFT (lebih cepat saat inferensi).
            logger.info("Memuat model full slerp-sft dari: %s", settings.merged_model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.merged_model_path,
                trust_remote_code=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.merged_model_path,
                dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
        else:
            # Tokenizer diambil dari folder adapter karena menyimpan chat_template.jinja
            # yang dipakai untuk memformat percakapan sesuai gaya training.
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.adapter_model_path,
                trust_remote_code=True,
            )

            logger.info("Memuat base model dari: %s", settings.base_model_path)
            base_model = AutoModelForCausalLM.from_pretrained(
                settings.base_model_path,
                dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
            )

            logger.info("Menempelkan adapter LoRA dari: %s", settings.adapter_model_path)
            self.model = PeftModel.from_pretrained(base_model, settings.adapter_model_path)

        self.model.eval()

        if self.device.startswith("cuda"):
            assert_all_on_gpu(self.model, torch.device(self.device))
            log_vram_usage(settings.gpu_index)
            self._warmup()
        else:
            self.model.to(self.device)

        self.is_loaded = True
        logger.info("Model siap digunakan.")

    def _warmup(self) -> None:
        """Satu generate pendek agar kernel CUDA siap sebelum request pertama."""
        inputs = self.tokenizer("1+1=", return_tensors="pt").to(self.device)
        with torch.inference_mode():
            self.model.generate(**inputs, max_new_tokens=4, do_sample=False,
                                pad_token_id=self.tokenizer.eos_token_id)
        torch.cuda.synchronize()
        logger.info("Warm-up GPU selesai.")

    def generate(self, question: str) -> dict:
        if not self.is_loaded:
            raise RuntimeError("Model belum dimuat. Panggil load() terlebih dahulu.")

        start = time.perf_counter()

        messages = [{"role": "user", "content": question}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=settings.max_new_tokens,
                do_sample=settings.do_sample,
                temperature=settings.temperature,
                top_p=settings.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        if self.device.startswith("cuda"):
            # generate() di GPU berjalan asinkron; sinkronkan agar waktu akurat.
            torch.cuda.synchronize()

        # Ambil hanya token baru (di luar prompt) supaya hasil tidak
        # mengulang pertanyaan.
        generated_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        answer = self._extract_answer(raw_text)
        elapsed = time.perf_counter() - start

        return {
            "answer": answer,
            "chain_of_thought": raw_text,
            "model": settings.model_display_name,
            "elapsed_time": round(elapsed, 2),
            "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        }

    @staticmethod
    def _extract_answer(text: str) -> str:
        match = BOXED_PATTERN.search(text)
        if match:
            return match.group(1).strip()
        # Fallback: kalau model tidak mengeluarkan \boxed{}, pakai baris
        # terakhir yang tidak kosong sebagai perkiraan jawaban akhir.
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[-1] if lines else "(jawaban tidak ditemukan)"


# Instance tunggal yang dipakai di seluruh aplikasi.
model_service = ModelService()
