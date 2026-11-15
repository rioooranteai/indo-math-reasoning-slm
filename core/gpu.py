"""GPU helpers shared by the web app and the merge utility."""

import logging

import torch

logger = logging.getLogger(__name__)

_GIB = 1024 ** 3

_CUDA_HELP = (
    "CUDA tidak terdeteksi oleh PyTorch.\n"
    "  - PyTorch terpasang: {torch_version} (build CUDA: {cuda_build})\n"
    "  - Jika build CUDA 'None', yang terpasang adalah PyTorch versi CPU.\n"
    "    Pasang ulang versi CUDA:\n"
    "      pip uninstall -y torch\n"
    "      pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129\n"
    "  - Pastikan driver NVIDIA terpasang: jalankan `nvidia-smi`."
)


def cuda_error_message() -> str:
    return _CUDA_HELP.format(torch_version=torch.__version__, cuda_build=torch.version.cuda)


def require_cuda(gpu_index: int = 0) -> torch.device:
    """Return the selected GPU, or raise a helpful error if CUDA is unusable."""
    if not torch.cuda.is_available():
        raise RuntimeError(cuda_error_message())

    count = torch.cuda.device_count()
    if gpu_index >= count:
        raise RuntimeError(f"gpu_index={gpu_index} tidak valid: hanya ada {count} GPU.")

    torch.cuda.set_device(gpu_index)
    # TF32 speeds up float32 matmuls on Ampere and newer GPUs.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return torch.device(f"cuda:{gpu_index}")


def best_gpu_dtype() -> torch.dtype:
    """bfloat16 on GPUs that support it (Ampere and newer), otherwise float16."""
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def log_gpu_info(gpu_index: int = 0) -> None:
    props = torch.cuda.get_device_properties(gpu_index)
    logger.info(
        "GPU %d: %s, %.1f GB VRAM, compute capability %d.%d, CUDA %s",
        gpu_index, props.name, props.total_memory / _GIB,
        props.major, props.minor, torch.version.cuda,
    )


def log_vram_usage(gpu_index: int = 0) -> None:
    allocated = torch.cuda.memory_allocated(gpu_index) / _GIB
    total = torch.cuda.get_device_properties(gpu_index).total_memory / _GIB
    logger.info("VRAM in use: %.2f of %.1f GB", allocated, total)


def assert_on_gpu(model: torch.nn.Module) -> None:
    """Raise if any weight was left on (or offloaded to) the CPU or disk."""
    tensors = list(model.named_parameters()) + list(model.named_buffers())
    off_gpu = [name for name, tensor in tensors if tensor.device.type != "cuda"]
    if off_gpu:
        raise RuntimeError(
            f"{len(off_gpu)} tensor model tidak berada di GPU (mis. {off_gpu[0]}). "
            "VRAM kemungkinan tidak cukup untuk memuat seluruh model."
        )

    device_map = getattr(model, "hf_device_map", None) or {}
    if any(str(device) in ("cpu", "disk") for device in device_map.values()):
        raise RuntimeError(f"Sebagian model dipindahkan ke CPU/disk: {device_map}")
