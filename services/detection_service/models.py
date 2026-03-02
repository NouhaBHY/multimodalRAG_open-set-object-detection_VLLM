"""
Detection Service - Grounding DINO Model Wrapper
Performs zero-shot object detection with text prompts.
Loads pre-quantized weights from disk (saved by scripts/quantize_models.py).
"""

import logging
import io
import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, BitsAndBytesConfig

# Base directory for pre-quantized model weights
QUANTIZED_MODELS_DIR = os.environ.get("QUANTIZED_MODELS_DIR", "/models/quantized")

logger = logging.getLogger(__name__)


class DetectionResult:
    """Encapsulates object detection results."""

    def __init__(self, boxes: list, labels: list, scores: list):
        self.boxes = boxes      # List of [x1, y1, x2, y2]
        self.labels = labels    # List of label strings
        self.scores = scores    # List of confidence scores

    def to_dict(self) -> dict:
        return {
            "boxes": self.boxes,
            "labels": self.labels,
            "scores": self.scores,
            "count": len(self.labels),
        }


class GroundingDINODetector:
    """
    Zero-shot object detection using Grounding DINO (IDEA-Research/grounding-dino-base).
    Loads pre-quantized 8-bit weights from disk (saved by scripts/quantize_models.py).
    """

    HF_MODEL_NAME = "IDEA-Research/grounding-dino-base"
    QUANTIZED_NAME = "grounding-dino-base-8bit"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        quantized_path = os.path.join(QUANTIZED_MODELS_DIR, self.QUANTIZED_NAME)

        if os.path.exists(quantized_path) and os.listdir(quantized_path):
            logger.info(f"Loading pre-quantized Grounding DINO from {quantized_path}")
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                quantized_path,
                device_map="auto" if self.device == "cuda" else None,
            )
            self.processor = AutoProcessor.from_pretrained(quantized_path)
        else:
            logger.warning(
                f"Quantized Grounding DINO not found at {quantized_path}. "
                f"Run 'python scripts/quantize_models.py --models dino' first. "
                f"Falling back to on-the-fly quantization."
            )
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            if self.device == "cuda":
                self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                    self.HF_MODEL_NAME,
                    quantization_config=quantization_config,
                    device_map="auto",
                )
            else:
                self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                    self.HF_MODEL_NAME,
                )
                self.model.to(self.device)
            self.processor = AutoProcessor.from_pretrained(self.HF_MODEL_NAME)

        if self.device == "cpu":
            self.model.to(self.device)
        self.model.eval()
        logger.info("Grounding DINO model loaded successfully")

    def detect(
        self,
        image: Image.Image,
        prompt: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> DetectionResult:
        """
        Run zero-shot object detection on an image with a text prompt.
        
        Args:
            image: PIL Image to run detection on.
            prompt: Text prompt describing objects to detect (e.g., "apple. potato. car.").
            box_threshold: Minimum confidence for bounding boxes.
            text_threshold: Minimum confidence for text matching.
            
        Returns:
            DetectionResult with boxes, labels, and scores.
        """
        # Ensure prompt ends with a period for Grounding DINO format
        if not prompt.endswith("."):
            prompt = prompt + "."

        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],  # (height, width)
        )

        result = results[0]
        boxes = result["boxes"].cpu().numpy().tolist()
        scores = result["scores"].cpu().numpy().tolist()
        labels = result["labels"]

        # Round box coordinates
        boxes = [[round(c, 2) for c in box] for box in boxes]
        scores = [round(s, 4) for s in scores]

        logger.info(f"Detected {len(labels)} objects with prompt: '{prompt[:80]}...'")
        return DetectionResult(boxes=boxes, labels=labels, scores=scores)

    def annotate_image(
        self,
        image: Image.Image,
        result: DetectionResult,
    ) -> bytes:
        """
        Draw bounding boxes and labels on the image.
        Returns annotated image as JPEG bytes.
        """
        draw = ImageDraw.Draw(image)

        # Color palette for different objects
        colors = [
            "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF",
            "#00FFFF", "#FFA500", "#800080", "#008000", "#000080",
        ]

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except (IOError, OSError):
            font = ImageFont.load_default()

        for i, (box, label, score) in enumerate(
            zip(result.boxes, result.labels, result.scores)
        ):
            color = colors[i % len(colors)]
            x1, y1, x2, y2 = box

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Draw label background
            text = f"{label} ({score:.2f})"
            text_bbox = draw.textbbox((x1, y1), text, font=font)
            draw.rectangle(
                [text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2],
                fill=color,
            )
            draw.text((x1, y1), text, fill="white", font=font)

        # Convert to bytes
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)
        return buffer.getvalue()
