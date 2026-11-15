"""Merge the SFT LoRA adapter into the SLERP model and save one full model.

For every LoRA module the update is folded into the base weight,
W = W_slerp + scaling * (B @ A), so the result loads with plain
AutoModelForCausalLM, without PEFT.

Run from the project folder:
    python -m utils.merge_lora                 # merge into model/slerp_sft_model
    python -m utils.merge_lora --verify        # merge, then answer one test question
    python -m utils.merge_lora --verify-only   # only test an existing merged model

Runs on the GPU by default and needs about 4 GB of VRAM in bfloat16.
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

import peft
import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.config import MODEL_DIR, PROJECT_DIR, SYSTEM_PROMPT_ID
from core.gpu import assert_on_gpu, cuda_error_message, log_gpu_info, log_vram_usage

logger = logging.getLogger("merge_lora")

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

VERIFY_QUESTION = (
    "Budi memiliki 12 kelereng. Ia memberikan 5 kelereng kepada Ani, "
    "lalu membeli 8 kelereng lagi. Berapa kelereng Budi sekarang?"
)
VERIFY_EXPECTED = "15"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gabungkan adapter LoRA ke model SLERP menjadi satu model utuh.",
    )
    parser.add_argument("--base", type=Path, default=MODEL_DIR / "slerp_model",
                        help="folder model SLERP (default: model/slerp_model)")
    parser.add_argument("--adapter", type=Path, default=MODEL_DIR / "lora_model",
                        help="folder adapter LoRA (default: model/lora_model)")
    parser.add_argument("--output", type=Path, default=MODEL_DIR / "slerp_sft_model",
                        help="folder hasil (default: model/slerp_sft_model)")
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16",
                        help="presisi bobot (default: bfloat16, sama dengan model SLERP)")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                        help="device untuk merge (default: cuda)")
    parser.add_argument("--gpu", type=int, default=0, help="index GPU (default: 0)")
    parser.add_argument("--overwrite", action="store_true",
                        help="timpa folder hasil jika sudah berisi file")
    parser.add_argument("--verify", action="store_true",
                        help="setelah merge, jawab satu soal uji")
    parser.add_argument("--verify-only", action="store_true",
                        help="lewati merge; hanya uji model yang sudah ada di --output")

    args = parser.parse_args()
    for name in ("base", "adapter", "output"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def select_device(args: argparse.Namespace) -> str:
    if args.device == "cpu":
        logger.warning("Running on CPU; this is much slower than the GPU.")
        return "cpu"
    if not torch.cuda.is_available():
        sys.exit(cuda_error_message() + "\n\nAtau jalankan dengan --device cpu.")
    if args.gpu >= torch.cuda.device_count():
        sys.exit(f"--gpu {args.gpu} tidak valid: hanya ada {torch.cuda.device_count()} GPU.")
    torch.cuda.set_device(args.gpu)
    log_gpu_info(args.gpu)
    return f"cuda:{args.gpu}"


def select_dtype(args: argparse.Namespace, device: str) -> torch.dtype:
    dtype = DTYPES[args.dtype]
    if device.startswith("cuda") and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        logger.warning("This GPU has no bfloat16 support; using float16.")
        return torch.float16
    return dtype


def check_inputs(args: argparse.Namespace) -> None:
    if not (args.base / "config.json").exists():
        sys.exit(f"Model SLERP tidak ditemukan atau tidak lengkap: {args.base}")
    if not (args.adapter / "adapter_config.json").exists():
        sys.exit(f"Adapter LoRA tidak ditemukan atau tidak lengkap: {args.adapter}")
    if args.output in (args.base, args.adapter):
        sys.exit("Folder hasil tidak boleh sama dengan folder model SLERP atau adapter.")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        sys.exit(f"Folder hasil sudah berisi file: {args.output}\nTambahkan --overwrite untuk menimpanya.")


def merge(args: argparse.Namespace) -> None:
    device = select_device(args)
    dtype = select_dtype(args, device)
    started = time.perf_counter()

    logger.info("Loading SLERP model from %s onto %s (%s)", args.base, device, dtype)
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype, device_map={"": device})
    # The adapter folder holds the chat template used during SFT.
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)

    logger.info("Applying LoRA adapter from %s", args.adapter)
    model = PeftModel.from_pretrained(base, args.adapter, torch_device=device)
    if device.startswith("cuda"):
        assert_on_gpu(model)
        log_vram_usage(args.gpu)

    logger.info("Merging LoRA weights")
    # safe_merge checks the merged weights for NaNs before writing them.
    merged = model.merge_and_unload(safe_merge=True)
    leftover = [name for name, _ in merged.named_parameters() if "lora_" in name]
    if leftover:
        sys.exit(f"Merge gagal: masih ada {len(leftover)} parameter LoRA, mis. {leftover[0]}.")

    logger.info("Saving merged model to %s", args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    template = args.adapter / "chat_template.jinja"
    if template.exists() and not (args.output / "chat_template.jinja").exists():
        shutil.copy2(template, args.output)

    duration = time.perf_counter() - started
    info = merge_info(args, duration)
    (args.output / "merge_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    del merged, model, base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Merge finished in %.1f s", duration)


def merge_info(args: argparse.Namespace, duration: float) -> dict:
    """Provenance record saved next to the merged weights."""
    adapter_cfg = json.loads((args.adapter / "adapter_config.json").read_text(encoding="utf-8"))
    slerp_meta_path = args.base / "merge_metadata.json"
    slerp_meta = json.loads(slerp_meta_path.read_text(encoding="utf-8")) if slerp_meta_path.exists() else None

    return {
        "model_name": "slerp-sft-model",
        "description": "SLERP model with the SFT LoRA adapter merged into its weights.",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "merge_duration_seconds": round(duration, 1),
        "dtype": args.dtype,
        "device": args.device,
        "gpu_name": torch.cuda.get_device_name(args.gpu) if args.device == "cuda" else None,
        "base_model_path": str(args.base),
        "adapter_path": str(args.adapter),
        "slerp": slerp_meta,
        "lora": {key: adapter_cfg.get(key) for key in ("r", "lora_alpha", "use_rslora", "target_modules")},
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
    }


def verify(args: argparse.Namespace) -> None:
    """Answer one known question with the merged model."""
    if not (args.output / "config.json").exists():
        sys.exit(f"Model gabungan belum ada di {args.output}. Jalankan merge dulu.")

    device = select_device(args)
    logger.info("Loading merged model from %s", args.output)
    tokenizer = AutoTokenizer.from_pretrained(args.output)
    model = AutoModelForCausalLM.from_pretrained(
        args.output, dtype=select_dtype(args, device), device_map={"": device},
    ).eval()
    if device.startswith("cuda"):
        assert_on_gpu(model)

    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT_ID},
         {"role": "user", "content": VERIFY_QUESTION}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs, max_new_tokens=512, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    print("\n" + "=" * 70)
    print("SOAL:", VERIFY_QUESTION)
    print("-" * 70)
    print(answer.strip())
    print("=" * 70)
    print(f"(Jawaban yang benar: {VERIFY_EXPECTED})\n")


def verify_in_new_process(args: argparse.Namespace) -> None:
    # On Windows the memory committed by the merge is not fully released in
    # this process, and loading the model again here can fail with
    # "paging file is too small" (os error 1455). A fresh process avoids it.
    command = [
        sys.executable, "-m", "utils.merge_lora", "--verify-only",
        "--output", str(args.output), "--dtype", args.dtype,
        "--device", args.device, "--gpu", str(args.gpu),
    ]
    result = subprocess.run(command, cwd=PROJECT_DIR, check=False)
    if result.returncode != 0:
        sys.exit(
            f"Verifikasi gagal (kode {result.returncode}). Model gabungan tetap tersimpan di "
            f"{args.output}; coba: python -m utils.merge_lora --verify-only"
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    if args.verify_only:
        verify(args)
        return

    check_inputs(args)
    merge(args)
    if args.verify:
        verify_in_new_process(args)
    logger.info("Merged model ready at %s", args.output)


if __name__ == "__main__":
    main()
