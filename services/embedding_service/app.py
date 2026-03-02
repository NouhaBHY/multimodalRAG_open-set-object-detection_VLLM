"""
Embedding Service - Flask Application
Provides REST API endpoints for image embedding generation and description.
"""

import io
import logging
import base64
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

from models import CLIPEmbedder, LLaVADescriber

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global model instances (lazy loaded)
clip_embedder = None
llava_describer = None


def get_clip_embedder() -> CLIPEmbedder:
    """Lazy-load the CLIP embedder singleton."""
    global clip_embedder
    if clip_embedder is None:
        clip_embedder = CLIPEmbedder()
    return clip_embedder


def get_llava_describer() -> LLaVADescriber:
    """Lazy-load the LLaVA describer singleton."""
    global llava_describer
    if llava_describer is None:
        llava_describer = LLaVADescriber()
    return llava_describer


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
    return jsonify({"status": "healthy", "service": "embedding_service"})


@app.route("/embed/image", methods=["POST"])
def embed_image():
    """
    Generate CLIP embedding for an uploaded image.
    
    Accepts: multipart/form-data with 'image' file, or JSON with 'image_base64'.
    Returns: JSON with 'embedding' (list of floats, 512-dim).
    """
    try:
        image = _load_image_from_request()
        embedder = get_clip_embedder()
        embedding = embedder.generate_embedding(image)

        return jsonify({
            "embedding": embedding.tolist(),
            "dimension": len(embedding),
            "status": "success",
        })
    except ValueError as e:
        return jsonify({"error": str(e), "status": "error"}), 400
    except Exception as e:
        logger.error(f"Error generating embedding: {e}", exc_info=True)
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/embed/text", methods=["POST"])
def embed_text():
    """
    Generate CLIP text embedding for a text query.
    
    Accepts: JSON with 'text' field.
    Returns: JSON with 'embedding' (list of floats, 512-dim).
    """
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "No 'text' field provided", "status": "error"}), 400

        embedder = get_clip_embedder()
        embedding = embedder.generate_text_embedding(data["text"])

        return jsonify({
            "embedding": embedding.tolist(),
            "dimension": len(embedding),
            "status": "success",
        })
    except Exception as e:
        logger.error(f"Error generating text embedding: {e}", exc_info=True)
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/describe/image", methods=["POST"])
def describe_image():
    """
    Generate a structured text description of an image using LLaVA.
    
    Accepts: multipart/form-data with 'image' file, or JSON with 'image_base64'.
    Returns: JSON with 'description' in "object1.object2.object3" format.
    """
    try:
        image = _load_image_from_request()
        describer = get_llava_describer()
        description = describer.generate_description(image)

        return jsonify({
            "description": description,
            "status": "success",
        })
    except ValueError as e:
        return jsonify({"error": str(e), "status": "error"}), 400
    except Exception as e:
        logger.error(f"Error generating description: {e}", exc_info=True)
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/embed-and-describe", methods=["POST"])
def embed_and_describe():
    """
    Generate both CLIP embedding and LLaVA description for an image in one call.
    
    Accepts: multipart/form-data with 'image' file, or JSON with 'image_base64'.
    Returns: JSON with 'embedding' and 'description'.
    """
    try:
        image = _load_image_from_request()

        embedder = get_clip_embedder()
        embedding = embedder.generate_embedding(image)

        describer = get_llava_describer()
        description = describer.generate_description(image)

        return jsonify({
            "embedding": embedding.tolist(),
            "dimension": len(embedding),
            "description": description,
            "status": "success",
        })
    except ValueError as e:
        return jsonify({"error": str(e), "status": "error"}), 400
    except Exception as e:
        logger.error(f"Error in embed-and-describe: {e}", exc_info=True)
        return jsonify({"error": str(e), "status": "error"}), 500


if __name__ == "__main__":
    logger.info("Starting Embedding Service on port 5001")
    app.run(host="0.0.0.0", port=5001, debug=False)
