"""Helper GPU bersama untuk dashboard dan utils."""

import logging

import torch

logger = logging.getLogger("gpu")

CUDA_HELP = (
    "CUDA tidak terdeteksi oleh PyTorch.\n"
    "  - PyTorch terinstall : {torch_version} (build CUDA: {cuda_build})\n"
    "  - Jika build CUDA 'None', PyTorch yang terpasang adalah versi CPU.\n"
    "    Install ulang versi CUDA, contoh:\n"
    "      pip uninstall -y torch\n"
    "      pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129\n"
    "    (build CUDA 12.9, sesuai driver di mesin ini)\n"
    "  - Pastikan driver NVIDIA terpasang: jalankan `nvidia-smi`."
)


def cuda_error_message() -> str:
    return CUDA_HELP.format(torch_version=torch.__version__, cuda_build=torch.version.cuda)


def require_cuda(gpu_index: int = 0) -> torch.device:
    """Pastikan CUDA tersedia dan kembalikan device GPU yang dipilih."""
    if not torch.cuda.is_available():
        raise RuntimeError(cuda_error_message())
    count = torch.cuda.device_count()
    if gpu_index >= count:
        raise RuntimeError(f"gpu_index={gpu_index} tidak valid: hanya ada {count} GPU.")

    torch.cuda.set_device(gpu_index)
    # TF32 mempercepat matmul float32 di GPU Ampere+ (RTX 30xx ke atas).
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return torch.device(f"cuda:{gpu_index}")


def best_gpu_dtype() -> torch.dtype:
    """bfloat16 jika GPU mendukung (Ampere+), selain itu float16."""
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def gpu_name(gpu_index: int = 0) -> str:
    return torch.cuda.get_device_name(gpu_index)


def log_gpu_info(gpu_index: int = 0) -> None:
    props = torch.cuda.get_device_properties(gpu_index)
    logger.info(
        "GPU %d: %s | VRAM %.1f GB | compute capability %d.%d | CUDA %s",
        gpu_index, props.name, props.total_memory / 1024**3,
        props.major, props.minor, torch.version.cuda,
    )


def log_vram_usage(gpu_index: int = 0) -> None:
    allocated = torch.cuda.memory_allocated(gpu_index) / 1024**3
    reserved = torch.cuda.memory_reserved(gpu_index) / 1024**3
    total = torch.cuda.get_device_properties(gpu_index).total_memory / 1024**3
    logger.info("VRAM terpakai: %.2f GB (reserved %.2f GB) dari %.1f GB", allocated, reserved, total)


def assert_all_on_gpu(model, device: torch.device) -> None:
    """Gagal jika ada parameter/buffer model yang tidak berada di GPU."""
    off = [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
           if t.device.type != "cuda"]
    if off:
        raise RuntimeError(
            f"{len(off)} tensor model tidak berada di GPU (mis. {off[0]}). "
            "VRAM kemungkinan tidak cukup untuk memuat seluruh model."
        )
    hf_map = getattr(model, "hf_device_map", None)
    if hf_map and any(str(d) in ("cpu", "disk") for d in hf_map.values()):
        raise RuntimeError(f"Sebagian model di-offload ke CPU/disk: {hf_map}")
