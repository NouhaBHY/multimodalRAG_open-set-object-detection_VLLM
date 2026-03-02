"""
Model Quantization Script
Downloads full-precision models from HuggingFace, quantizes them, 
and saves the quantized versions to disk for efficient inference.

Models:
  - CLIP (openai/clip-vit-base-patch32) → 8-bit quantization
  - LLaVA (llava-hf/llava-1.5-7b-hf) → 4-bit NF4 quantization
  - Grounding DINO (IDEA-Research/grounding-dino-base) → 8-bit quantization

Usage:
  python scripts/quantize_models.py [--models clip llava dino] [--output-dir /models/quantized]
"""

import argparse
import logging
import os
import sys
import shutil

import torch
from transformers import (
    CLIPModel,
    CLIPProcessor,
    LlavaForConditionalGeneration,
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    BitsAndBytesConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Model Definitions
# ═══════════════════════════════════════════════════════════════

MODELS = {
    "clip": {
        "hf_name": "openai/clip-vit-base-patch32",
        "output_name": "clip-vit-base-patch32-8bit",
        "description": "CLIP ViT-B/32 (8-bit quantized)",
    },
    "llava": {
        "hf_name": "llava-hf/llava-1.5-7b-hf",
        "output_name": "llava-1.5-7b-4bit",
        "description": "LLaVA 1.5 7B (4-bit NF4 quantized)",
    },
    "dino": {
        "hf_name": "IDEA-Research/grounding-dino-base",
        "output_name": "grounding-dino-base-8bit",
        "description": "Grounding DINO Base (8-bit quantized)",
    },
}


def quantize_clip(output_dir: str) -> None:
    """Download CLIP, load in fp16, and save (CLIP is small, no heavy quantization needed)."""
    cfg = MODELS["clip"]
    save_path = os.path.join(output_dir, cfg["output_name"])

    if os.path.exists(save_path) and os.listdir(save_path):
        logger.info(f"CLIP already quantized at {save_path}, skipping.")
        return

    logger.info(f"Downloading CLIP: {cfg['hf_name']}")

    # CLIP doesn't support device_map='auto' with bitsandbytes in transformers 4.36.
    # Load in fp16 (CLIP is only ~600MB so this is efficient enough).
    model = CLIPModel.from_pretrained(
        cfg["hf_name"],
        torch_dtype=torch.float16,
    )

    processor = CLIPProcessor.from_pretrained(cfg["hf_name"])

    logger.info(f"Saving CLIP (fp16) to {save_path}")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)

    logger.info(f"CLIP saved successfully.")

    # Free memory
    del model
    torch.cuda.empty_cache()


def quantize_llava(output_dir: str) -> None:
    """Download LLaVA, quantize to 4-bit NF4, and save."""
    cfg = MODELS["llava"]
    save_path = os.path.join(output_dir, cfg["output_name"])

    if os.path.exists(save_path) and os.listdir(save_path):
        logger.info(f"LLaVA already quantized at {save_path}, skipping.")
        return

    logger.info(f"Downloading LLaVA: {cfg['hf_name']}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = LlavaForConditionalGeneration.from_pretrained(
        cfg["hf_name"],
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    processor = AutoProcessor.from_pretrained(cfg["hf_name"])

    logger.info(f"Saving quantized LLaVA to {save_path}")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)

    logger.info(f"LLaVA quantized and saved successfully.")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()


def quantize_dino(output_dir: str) -> None:
    """Download Grounding DINO, quantize to 8-bit, and save."""
    cfg = MODELS["dino"]
    save_path = os.path.join(output_dir, cfg["output_name"])

    if os.path.exists(save_path) and os.listdir(save_path):
        logger.info(f"Grounding DINO already quantized at {save_path}, skipping.")
        return

    logger.info(f"Downloading Grounding DINO: {cfg['hf_name']}")

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        cfg["hf_name"],
        quantization_config=quantization_config,
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(cfg["hf_name"])

    logger.info(f"Saving quantized Grounding DINO to {save_path}")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)

    logger.info(f"Grounding DINO quantized and saved successfully.")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

QUANTIZE_FNS = {
    "clip": quantize_clip,
    "llava": quantize_llava,
    "dino": quantize_dino,
}


def main():
    parser = argparse.ArgumentParser(
        description="Download and quantize HuggingFace models for inference."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["clip", "llava", "dino", "all"],
        default=["all"],
        help="Which models to quantize (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/models/quantized",
        help="Directory to save quantized models (default: /models/quantized)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-quantize even if output already exists",
    )

    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if "all" in args.models:
        targets = ["clip", "llava", "dino"]
    else:
        targets = args.models

    if args.force:
        for t in targets:
            path = os.path.join(output_dir, MODELS[t]["output_name"])
            if os.path.exists(path):
                logger.info(f"Removing existing quantized model: {path}")
                shutil.rmtree(path)

    logger.info(f"Quantizing models: {targets}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

    for target in targets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Quantizing: {MODELS[target]['description']}")
        logger.info(f"{'='*60}")
        try:
            QUANTIZE_FNS[target](output_dir)
        except Exception as e:
            logger.error(f"Failed to quantize {target}: {e}")
            raise

    logger.info("\n" + "=" * 60)
    logger.info("All models quantized successfully!")
    logger.info("=" * 60)

    # Print summary
    for target in targets:
        path = os.path.join(output_dir, MODELS[target]["output_name"])
        if os.path.exists(path):
            size_mb = sum(
                os.path.getsize(os.path.join(path, f))
                for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f))
            )
            logger.info(f"  {MODELS[target]['description']}: {size_mb / 1024**2:.1f} MB")


if __name__ == "__main__":
    main()
