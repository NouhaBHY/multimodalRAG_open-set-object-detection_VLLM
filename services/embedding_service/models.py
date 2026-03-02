"""
Embedding Service - CLIP & LLaVA Model Wrappers
Handles image embedding generation (CLIP) and image description generation (LLaVA).
Both models are loaded from pre-quantized weights saved on disk.
"""

import logging
import os
import re
import torch
import numpy as np
from PIL import Image
from transformers import (
    CLIPModel,
    CLIPProcessor,
    LlavaForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

# Base directory for pre-quantized model weights
QUANTIZED_MODELS_DIR = os.environ.get("QUANTIZED_MODELS_DIR", "/models/quantized")

logger = logging.getLogger(__name__)


class CLIPEmbedder:
    """
    Generates image embeddings using the CLIP model (openai/clip-vit-base-patch32).
    Loads pre-saved fp16 weights from disk (saved by scripts/quantize_models.py).
    CLIP is small (~600MB) so fp16 is efficient enough without heavy quantization.
    """

    HF_MODEL_NAME = "openai/clip-vit-base-patch32"
    QUANTIZED_NAME = "clip-vit-base-patch32-8bit"
    EMBEDDING_DIM = 512

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        quantized_path = os.path.join(QUANTIZED_MODELS_DIR, self.QUANTIZED_NAME)

        if os.path.exists(quantized_path) and os.listdir(quantized_path):
            logger.info(f"Loading pre-saved CLIP from {quantized_path}")
            self.model = CLIPModel.from_pretrained(
                quantized_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            self.processor = CLIPProcessor.from_pretrained(quantized_path)
        else:
            logger.warning(
                f"Saved CLIP not found at {quantized_path}. "
                f"Run 'python scripts/quantize_models.py --models clip' first. "
                f"Falling back to downloading from HuggingFace."
            )
            self.model = CLIPModel.from_pretrained(self.HF_MODEL_NAME)
            self.processor = CLIPProcessor.from_pretrained(self.HF_MODEL_NAME)

        self.model.to(self.device)
        self.model.eval()
        logger.info("CLIP model loaded successfully")

    def generate_embedding(self, image: Image.Image) -> np.ndarray:
        """Generate a 512-dimensional embedding vector for an image."""
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        # L2 normalize the embedding
        embedding = image_features.cpu().numpy().flatten()
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def generate_text_embedding(self, text: str) -> np.ndarray:
        """Generate a 512-dimensional embedding vector for text."""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        embedding = text_features.cpu().numpy().flatten()
        embedding = embedding / np.linalg.norm(embedding)
        return embedding


class LLaVADescriber:
    """
    Generates structured text descriptions of images using LLaVA (llava-hf/llava-1.5-7b-hf).
    Loads pre-quantized 4-bit NF4 weights from disk (saved by scripts/quantize_models.py).
    Output format: "object1.object2.object3"
    """

    HF_MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
    QUANTIZED_NAME = "llava-1.5-7b-4bit"

    PROMPT = (
        "USER: <image>\nList all the distinct objects and items visible in this image. "
        "Output ONLY the object names separated by dots, nothing else. "
        "Example format: apple.banana.table.chair\nASSISTANT:"
    )

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        quantized_path = os.path.join(QUANTIZED_MODELS_DIR, self.QUANTIZED_NAME)

        if os.path.exists(quantized_path) and os.listdir(quantized_path):
            logger.info(f"Loading pre-quantized LLaVA from {quantized_path}")
            self.model = LlavaForConditionalGeneration.from_pretrained(
                quantized_path,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            self.processor = AutoProcessor.from_pretrained(quantized_path)
        else:
            logger.warning(
                f"Quantized LLaVA not found at {quantized_path}. "
                f"Run 'python scripts/quantize_models.py --models llava' first. "
                f"Falling back to on-the-fly quantization."
            )
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            if self.device == "cuda":
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    self.HF_MODEL_NAME,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
            else:
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    self.HF_MODEL_NAME,
                    torch_dtype=torch.float32,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
            self.processor = AutoProcessor.from_pretrained(self.HF_MODEL_NAME)

        self.model.eval()
        logger.info("LLaVA model loaded successfully")

    def generate_description(self, image: Image.Image) -> str:
        """
        Generate a structured text description of the image.
        Returns format: "object1.object2.object3"
        """
        inputs = self.processor(text=self.PROMPT, images=image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                temperature=1.0,
            )

        raw_text = self.processor.decode(output[0], skip_special_tokens=True)
        return self._format_output(raw_text)

    def _format_output(self, raw_text: str) -> str:
        """
        Parse and clean the LLaVA output to structured dot-separated format.
        Extracts the assistant's response and normalizes it.
        """
        # Extract text after ASSISTANT:
        if "ASSISTANT:" in raw_text:
            raw_text = raw_text.split("ASSISTANT:")[-1].strip()

        # Remove common artifacts
        raw_text = raw_text.replace("\n", " ").strip()

        # Try to extract dot-separated words directly
        # If the model already outputs in the right format
        if re.match(r"^[\w\s]+(\.\s*[\w\s]+)*$", raw_text):
            items = [item.strip().lower() for item in raw_text.split(".") if item.strip()]
            return ".".join(items)

        # Otherwise, extract nouns/objects from the text
        # Remove punctuation except dots and spaces
        cleaned = re.sub(r"[^a-zA-Z\s.]", " ", raw_text)
        # Split by common separators
        items = re.split(r"[,.\s]+", cleaned)
        # Filter out common stop words and empty strings
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "and", "or", "but",
            "in", "on", "at", "to", "for", "of", "with", "by", "from", "it",
            "this", "that", "these", "those", "i", "you", "he", "she", "we",
            "they", "there", "here", "has", "have", "had", "be", "been",
            "being", "can", "could", "would", "should", "may", "might",
            "some", "all", "each", "every", "no", "not", "very", "image",
            "picture", "photo", "see", "shows", "showing", "visible",
        }
        items = [
            item.lower().strip()
            for item in items
            if item.strip() and item.lower().strip() not in stop_words and len(item.strip()) > 1
        ]

        # Deduplicate while preserving order
        seen = set()
        unique_items = []
        for item in items:
            if item not in seen:
                seen.add(item)
                unique_items.append(item)

        return ".".join(unique_items) if unique_items else "unknown"
