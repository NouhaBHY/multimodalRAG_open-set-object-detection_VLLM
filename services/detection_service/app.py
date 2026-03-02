"""
Detection Service - Flask Application
Provides REST API endpoints for object detection using Grounding DINO.
"""

import io
import logging
import base64

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

from models import GroundingDINODetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global model instance (lazy loaded)
detector = None


def get_detector() -> GroundingDINODetector:
    """Lazy-load the Grounding DINO detector singleton."""
    global detector
    if detector is None:
        detector = GroundingDINODetector()
    return detector


def _load_image_from_request() -> Image.Image:
    """Extract and load a PIL Image from the incoming request."""
    if "image" in request.files:
        file = request.files["image"]
        image = Image.open(file.stream).convert("RGB")
    elif request.is_json and "image_base64" in request.json:
        image_data = base64.b64decode(request.json["image_base64"])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
    else:
        raise ValueError("No image provided. Send 'image' file or 'image_base64' JSON field.")
    return image


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "detection_service"})


@app.route("/detect", methods=["POST"])
def detect():
    """
    Run object detection on an image with a text prompt.
    
    Accepts:
        - multipart/form-data: 'image' file + 'prompt' field
        - JSON: 'image_base64' + 'prompt'
    
    Optional fields:
        - box_threshold (float, default 0.25)
        - text_threshold (float, default 0.25)
    
    Returns:
        JSON with detection results + base64 annotated image.
    """
    try:
        # Get the prompt
        if request.content_type and "multipart" in request.content_type:
            prompt = request.form.get("prompt", "")
            box_threshold = float(request.form.get("box_threshold", 0.25))
            text_threshold = float(request.form.get("text_threshold", 0.25))
        elif request.is_json:
            prompt = request.json.get("prompt", "")
            box_threshold = request.json.get("box_threshold", 0.25)
            text_threshold = request.json.get("text_threshold", 0.25)
        else:
            prompt = request.form.get("prompt", "")
            box_threshold = float(request.form.get("box_threshold", 0.25))
            text_threshold = float(request.form.get("text_threshold", 0.25))

        if not prompt:
            return jsonify({"error": "No prompt provided", "status": "error"}), 400

        image = _load_image_from_request()
        det = get_detector()

        # Run detection
        result = det.detect(image, prompt, box_threshold, text_threshold)

        # Annotate image
        annotated_bytes = det.annotate_image(image.copy(), result)
        annotated_b64 = base64.b64encode(annotated_bytes).decode("utf-8")

        response = result.to_dict()
        response["annotated_image"] = annotated_b64
        response["prompt_used"] = prompt
        response["status"] = "success"

        return jsonify(response)

    except ValueError as e:
        return jsonify({"error": str(e), "status": "error"}), 400
    except Exception as e:
        logger.error(f"Error in detection: {e}", exc_info=True)
        return jsonify({"error": str(e), "status": "error"}), 500


if __name__ == "__main__":
    logger.info("Starting Detection Service on port 5003")
    app.run(host="0.0.0.0", port=5003, debug=False)
