"""
Gabungkan (merge) adapter LoRA hasil SFT ke dalam model SLERP, lalu simpan
sebagai SATU model utuh "slerp-sft" yang bisa dimuat langsung dengan
AutoModelForCausalLM tanpa PEFT.

    W_baru = W_slerp + scaling * (B @ A)      (dihitung untuk setiap modul LoRA)

Cara pakai (jalankan dari root proyek):

    python -m utils.merge_lora
    python -m utils.merge_lora --verify
    python -m utils.merge_lora --verify-only      # uji model yang sudah di-merge saja
    python -m utils.merge_lora --output model/slerp_sft_model --dtype bfloat16 --overwrite

Default berjalan penuh di GPU (--device cuda). Kebutuhan: sekitar 4 GB VRAM untuk model 1.5B
dalam bfloat16. Opsi --verify memuat ulang model hasil merge, jadi butuh waktu ekstra.
"""

import argparse
import gc
import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.gpu import assert_all_on_gpu, cuda_error_message, log_gpu_info, log_vram_usage  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BASE = PROJECT_DIR / "model" / "slerp_model"
DEFAULT_ADAPTER = PROJECT_DIR / "model" / "lora_model"
DEFAULT_OUTPUT = PROJECT_DIR / "model" / "slerp_sft_model"

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

VERIFY_QUESTION = (
    "Budi memiliki 12 kelereng. Ia memberikan 5 kelereng kepada Ani, "
    "lalu membeli 8 kelereng lagi. Berapa kelereng Budi sekarang?"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("merge_lora")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge adapter LoRA ke model SLERP menjadi satu model full slerp-sft."
    )
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE,
                        help=f"Folder model SLERP (default: {DEFAULT_BASE.relative_to(PROJECT_DIR)})")
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER,
                        help=f"Folder adapter LoRA (default: {DEFAULT_ADAPTER.relative_to(PROJECT_DIR)})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Folder output model gabungan (default: {DEFAULT_OUTPUT.relative_to(PROJECT_DIR)})")
    parser.add_argument("--dtype", choices=DTYPES.keys(), default="bfloat16",
                        help="Presisi bobot saat merge & disimpan (default: bfloat16, sama dengan model SLERP)")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                        help="Device untuk proses merge (default: cuda / GPU penuh)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="Index GPU yang dipakai (default: 0)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Timpa folder output jika sudah berisi file")
    parser.add_argument("--verify", action="store_true",
                        help="Setelah merge, muat ulang model gabungan dan jalankan satu contoh soal")
    parser.add_argument("--verify-only", action="store_true",
                        help="Lewati merge; hanya uji model gabungan yang sudah ada di --output")
    return parser.parse_args()


def resolve_device(args: argparse.Namespace) -> str:
    if args.device == "cpu":
        logger.warning("--device cpu: merge berjalan di CPU (lebih lambat).")
        return "cpu"
    if not torch.cuda.is_available():
        sys.exit(cuda_error_message() + "\n\nAtau jalankan dengan --device cpu.")
    if args.gpu >= torch.cuda.device_count():
        sys.exit(f"--gpu {args.gpu} tidak valid: hanya ada {torch.cuda.device_count()} GPU.")
    torch.cuda.set_device(args.gpu)
    return f"cuda:{args.gpu}"


def check_inputs(args: argparse.Namespace) -> None:
    if not (args.base / "config.json").exists():
        sys.exit(f"Model SLERP tidak ditemukan / tidak lengkap: {args.base}")
    if not (args.adapter / "adapter_config.json").exists():
        sys.exit(f"Adapter LoRA tidak ditemukan / tidak lengkap: {args.adapter}")
    if args.output.resolve() in (args.base.resolve(), args.adapter.resolve()):
        sys.exit("Folder output tidak boleh sama dengan folder base model atau adapter.")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        sys.exit(f"Folder output sudah berisi file: {args.output}\n"
                 "Tambahkan --overwrite untuk menimpanya.")


def merge(args: argparse.Namespace) -> dict:
    device = resolve_device(args)
    dtype = DTYPES[args.dtype]
    if device.startswith("cuda") and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        logger.warning("GPU tidak mendukung bfloat16; merge memakai float16.")
        dtype = torch.float16
    start = time.perf_counter()

    if device.startswith("cuda"):
        log_gpu_info(args.gpu)
    logger.info("Device: %s | dtype: %s", device, dtype)

    logger.info("Memuat model SLERP langsung ke %s dari: %s", device, args.base)
    # device_map ke satu device: bobot langsung ditulis ke GPU, tanpa offload.
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=dtype, device_map={"": device}
    )

    # Tokenizer diambil dari folder adapter karena menyimpan chat template
    # yang dipakai saat SFT.
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)

    logger.info("Menempelkan adapter LoRA dari: %s", args.adapter)
    peft_model = PeftModel.from_pretrained(base_model, args.adapter, torch_device=device)
    if device.startswith("cuda"):
        assert_all_on_gpu(peft_model, torch.device(device))
        log_vram_usage(args.gpu)

    logger.info("Menggabungkan bobot LoRA ke model (merge_and_unload)...")
    # safe_merge=True: cek NaN pada bobot hasil merge sebelum ditulis ke model.
    merged = peft_model.merge_and_unload(safe_merge=True)

    leftover = [name for name, _ in merged.named_parameters() if "lora_" in name]
    if leftover:
        sys.exit(f"Merge gagal: masih ada {len(leftover)} parameter LoRA, mis. {leftover[0]}")

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    args.output.mkdir(parents=True, exist_ok=True)

    logger.info("Menyimpan model gabungan ke: %s", args.output)
    merged.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    # Pastikan chat template hasil SFT ikut tersimpan.
    template_src = args.adapter / "chat_template.jinja"
    template_dst = args.output / "chat_template.jinja"
    if template_src.exists() and not template_dst.exists():
        shutil.copy2(template_src, template_dst)

    info = build_merge_info(args, time.perf_counter() - start)
    (args.output / "merge_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    del merged, peft_model, base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Merge selesai dalam %.1f detik.", info["merge_duration_seconds"])
    return info


def build_merge_info(args: argparse.Namespace, duration: float) -> dict:
    adapter_cfg = json.loads((args.adapter / "adapter_config.json").read_text(encoding="utf-8"))
    slerp_meta_path = args.base / "merge_metadata.json"
    slerp_meta = (json.loads(slerp_meta_path.read_text(encoding="utf-8"))
                  if slerp_meta_path.exists() else None)

    return {
        "model_name": "slerp-sft-model",
        "description": "Model SLERP + adapter LoRA (SFT) yang sudah digabung menjadi satu model utuh.",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "merge_duration_seconds": round(duration, 1),
        "dtype": args.dtype,
        "device": args.device,
        "gpu_name": torch.cuda.get_device_name(args.gpu) if args.device == "cuda" else None,
        "base_model_path": str(args.base),
        "adapter_path": str(args.adapter),
        "slerp": slerp_meta,
        "lora": {
            "r": adapter_cfg.get("r"),
            "lora_alpha": adapter_cfg.get("lora_alpha"),
            "use_rslora": adapter_cfg.get("use_rslora"),
            "target_modules": adapter_cfg.get("target_modules"),
        },
        "versions": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "peft": __import__("peft").__version__,
        },
    }


def verify(args: argparse.Namespace) -> None:
    device = resolve_device(args)
    logger.info("Verifikasi: memuat model gabungan ke %s dari %s", device, args.output)

    tokenizer = AutoTokenizer.from_pretrained(args.output)
    model = AutoModelForCausalLM.from_pretrained(
        args.output, dtype=DTYPES[args.dtype], device_map={"": device}
    )
    model.eval()
    if device.startswith("cuda"):
        assert_all_on_gpu(model, torch.device(device))

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": VERIFY_QUESTION}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    answer = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print("\n" + "=" * 70)
    print("SOAL   :", VERIFY_QUESTION)
    print("-" * 70)
    print(answer.strip())
    print("=" * 70)
    print("(Jawaban yang benar: 15)\n")


def main() -> None:
    args = parse_args()
    args.base = args.base.resolve()
    args.adapter = args.adapter.resolve()
    args.output = args.output.resolve()

    if args.verify_only:
        if not (args.output / "config.json").exists():
            sys.exit(f"Model gabungan belum ada di: {args.output}. Jalankan merge dulu.")
        verify(args)
        return

    check_inputs(args)
    merge(args)

    if args.verify:
        # Verifikasi dijalankan di proses Python BARU. Di Windows, memori
        # virtual (commit) dari proses merge tidak sepenuhnya dilepas walau
        # objek model sudah dihapus, sehingga memuat model lagi di proses
        # yang sama bisa gagal dengan "paging file is too small" (os error 1455).
        logger.info("Menjalankan verifikasi di proses terpisah...")
        cmd = [sys.executable, "-m", "utils.merge_lora", "--verify-only",
               "--output", str(args.output), "--dtype", args.dtype,
               "--device", args.device, "--gpu", str(args.gpu)]
        result = subprocess.run(cmd, cwd=PROJECT_DIR)
        if result.returncode != 0:
            sys.exit(f"Verifikasi gagal (exit code {result.returncode}). Model gabungan "
                     f"tetap tersimpan di {args.output}; coba: python -m utils.merge_lora --verify-only")

    logger.info("Model full slerp-sft siap di: %s", args.output)


if __name__ == "__main__":
    main()
