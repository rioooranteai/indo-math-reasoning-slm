"""Loads the model once and answers questions with it."""

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from core.config import settings
from core.gpu import assert_on_gpu, best_gpu_dtype, log_gpu_info, log_vram_usage, require_cuda
from services.answer import extract_answer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Solution:
    answer: str
    chain_of_thought: str
    model: str
    elapsed_time: float
    timestamp: str


class ModelService:
    """Owns the tokenizer and model. Create one instance per process."""

    def __init__(self) -> None:
        self.tokenizer = None
        self.model: PreTrainedModel | None = None
        self.device: str | None = None
        self.gpu_name: str | None = None
        # One GPU, one generation at a time: concurrent requests would
        # compete for the same VRAM.
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    # ---------- Loading ----------

    def load(self) -> None:
        device_map, dtype = self._select_device()
        logger.info("Loading model on %s (%s)", self.device, dtype)

        if settings.merged_model_available:
            self.tokenizer = AutoTokenizer.from_pretrained(settings.merged_model_path)
            model = self._load_weights(settings.merged_model_path, dtype, device_map)
        else:
            model = self._load_base_with_adapter(dtype, device_map)

        model.eval()
        if self.device.startswith("cuda"):
            assert_on_gpu(model)
            log_vram_usage(settings.gpu_index)
        else:
            model.to(self.device)

        self.model = model
        self._warm_up()
        logger.info("Model ready.")

    def _select_device(self) -> tuple[dict | None, torch.dtype]:
        if settings.force_cpu:
            logger.warning("FORCE_CPU is set: running on CPU (debugging only, slow).")
            self.device = "cpu"
            return None, torch.float32

        if settings.require_gpu or torch.cuda.is_available():
            # Raises with install instructions when CUDA is missing.
            gpu = require_cuda(settings.gpu_index)
            self.device = str(gpu)
            self.gpu_name = torch.cuda.get_device_name(settings.gpu_index)
            log_gpu_info(settings.gpu_index)
            # Every layer on one GPU: no offloading to CPU or disk.
            return {"": settings.gpu_index}, best_gpu_dtype()

        logger.warning("CUDA not available and REQUIRE_GPU is false: running on CPU.")
        self.device = "cpu"
        return None, torch.float32

    @staticmethod
    def _load_weights(path: Path, dtype: torch.dtype, device_map: dict | None) -> PreTrainedModel:
        logger.info("Loading weights from %s", path)
        return AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map=device_map)

    def _load_base_with_adapter(self, dtype: torch.dtype, device_map: dict | None) -> PreTrainedModel:
        # Imported here so PEFT is only needed for this fallback path.
        from peft import PeftModel

        # The adapter folder holds the chat template used during SFT.
        self.tokenizer = AutoTokenizer.from_pretrained(settings.adapter_model_path)
        base = self._load_weights(settings.base_model_path, dtype, device_map)
        logger.info("Applying LoRA adapter from %s", settings.adapter_model_path)
        return PeftModel.from_pretrained(base, settings.adapter_model_path)

    def _warm_up(self) -> None:
        """Run one tiny generation so the first real request is not slow."""
        inputs = self.tokenizer("1+1=", return_tensors="pt").to(self.device)
        with torch.inference_mode():
            self.model.generate(
                **inputs, max_new_tokens=4, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        self._synchronize()

    # ---------- Inference ----------

    def solve(self, question: str) -> Solution:
        if not self.is_loaded:
            raise RuntimeError("Model has not been loaded.")

        with self._lock:
            started = time.perf_counter()
            text = self._generate(question)
            elapsed = time.perf_counter() - started

        return Solution(
            answer=extract_answer(text),
            chain_of_thought=text,
            model=settings.model_display_name,
            elapsed_time=round(elapsed, 2),
            timestamp=datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        )

    def _generate(self, question: str) -> str:
        messages = [
            {"role": "system", "content": settings.system_prompt},
            {"role": "user", "content": question},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        sampling = (
            {"do_sample": True, "temperature": settings.temperature, "top_p": settings.top_p}
            if settings.do_sample else {"do_sample": False}
        )
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=settings.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                **sampling,
            )
        self._synchronize()

        # Decode only the new tokens, not the prompt.
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _synchronize(self) -> None:
        # CUDA runs asynchronously; wait so timings are accurate.
        if self.device and self.device.startswith("cuda"):
            torch.cuda.synchronize()


model_service = ModelService()
