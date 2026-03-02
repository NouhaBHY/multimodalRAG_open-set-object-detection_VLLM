"""
API Gateway - Flask Application
Orchestrates calls between the frontend and microservices.
Implements the two main pipelines:
  1. Image Indexing: Upload → CLIP embed → LLaVA describe → Store in ES + MongoDB
  2. Object Detection: Upload → CLIP embed → ES KNN search → Augment prompt → Grounding DINO
"""

import os
import io
import logging
import base64
import time

import requests as http_requests
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ─── Service URLs ─────────────────────────────────────────────────────────────

EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://embedding_service:5001")
STORAGE_SERVICE_URL = os.getenv("STORAGE_SERVICE_URL", "http://storage_service:5002")
DETECTION_SERVICE_URL = os.getenv("DETECTION_SERVICE_URL", "http://detection_service:5003")

REQUEST_TIMEOUT = 300  # seconds


# ─── Helper Functions ─────────────────────────────────────────────────────────

def _image_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")


def _call_service(method: str, url: str, **kwargs) -> dict:
    """Make an HTTP call to a microservice with error handling."""
    try:
        kwargs.setdefault("timeout", REQUEST_TIMEOUT)
        resp = getattr(http_requests, method)(url, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except http_requests.exceptions.ConnectionError:
        raise ConnectionError(f"Service unavailable: {url}")
    except http_requests.exceptions.Timeout:
        raise TimeoutError(f"Service timeout: {url}")
    except Exception as e:
        raise RuntimeError(f"Service error ({url}): {e}")


# ─── Health Check ─────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    """Check health of all services."""
    services = {
        "embedding_service": f"{EMBEDDING_SERVICE_URL}/health",
        "storage_service": f"{STORAGE_SERVICE_URL}/health",
        "detection_service": f"{DETECTION_SERVICE_URL}/health",
    }
    status = {}
    for name, url in services.items():
        try:
            resp = http_requests.get(url, timeout=5)
            status[name] = "healthy" if resp.status_code == 200 else "unhealthy"
        except Exception:
            status[name] = "unreachable"

    all_healthy = all(s == "healthy" for s in status.values())
    return jsonify({
        "status": "healthy" if all_healthy else "degraded",
        "services": status,
    })


# ─── Pipeline 1: Image Indexing ───────────────────────────────────────────────

@app.route("/api/index", methods=["POST"])
def index_images():
    """
    Index one or more images.
    Pipeline: Store image → Generate embedding → Generate description → Index in ES.
    
    Accepts: multipart/form-data with one or more 'images' files.
    """
    try:
        if "images" not in request.files:
            return jsonify({"error": "No images provided", "status": "error"}), 400

        files = request.files.getlist("images")
        results = []

        for file in files:
            filename = file.filename or "unknown"
            image_bytes = file.read()
            image_b64 = _image_to_base64(image_bytes)

            logger.info(f"Indexing image: {filename}")

            # Step 1: Store image in MongoDB
            store_resp = _call_service(
                "post",
                f"{STORAGE_SERVICE_URL}/store/image",
                json={"image_base64": image_b64, "filename": filename},
            )
            image_id = store_resp["image_id"]
            logger.info(f"  Stored image in MongoDB: {image_id}")

            # Step 2: Generate CLIP embedding
            embed_resp = _call_service(
                "post",
                f"{EMBEDDING_SERVICE_URL}/embed/image",
                json={"image_base64": image_b64},
            )
            embedding = embed_resp["embedding"]
            logger.info(f"  Generated embedding (dim={embed_resp['dimension']})")

            # Step 3: Generate LLaVA description
            desc_resp = _call_service(
                "post",
                f"{EMBEDDING_SERVICE_URL}/describe/image",
                json={"image_base64": image_b64},
            )
            description = desc_resp["description"]
            logger.info(f"  Generated description: {description}")

            # Step 4: Store embedding + description in Elasticsearch
            index_resp = _call_service(
                "post",
                f"{STORAGE_SERVICE_URL}/store/embedding",
                json={
                    "image_id": image_id,
                    "embedding": embedding,
                    "description": description,
                    "filename": filename,
                },
            )
            doc_id = index_resp["doc_id"]
            logger.info(f"  Indexed in Elasticsearch: {doc_id}")

            results.append({
                "filename": filename,
                "image_id": image_id,
                "doc_id": doc_id,
                "description": description,
                "status": "success",
            })

        return jsonify({
            "results": results,
            "indexed_count": len(results),
            "status": "success",
        })

    except (ConnectionError, TimeoutError) as e:
        logger.error(f"Service error during indexing: {e}")
        return jsonify({"error": str(e), "status": "error"}), 503
    except Exception as e:
        logger.error(f"Error indexing images: {e}", exc_info=True)
        return jsonify({"error": str(e), "status": "error"}), 500


# ─── Pipeline 2: Object Detection with Prompt Augmentation ───────────────────

@app.route("/api/detect", methods=["POST"])
def detect_objects():
    """
    Detect objects in a user image with augmented prompt.
    
    Pipeline:
        1. Generate CLIP embedding for user image
        2. KNN search in ES for 2 most similar indexed images
        3. Get text descriptions from similar images
        4. Augment user prompt with descriptions
        5. Run Grounding DINO with augmented prompt
        6. Store and return results
    
    Accepts: multipart/form-data with 'image' file + 'prompt' field.
    """
    try:
        # Get prompt and optional thresholds
        if request.content_type and "multipart" in request.content_type:
            prompt = request.form.get("prompt", "")
            box_threshold = float(request.form.get("box_threshold", 0.25))
            text_threshold = float(request.form.get("text_threshold", 0.25))
        else:
            return jsonify({"error": "Use multipart/form-data", "status": "error"}), 400

        if not prompt:
            return jsonify({"error": "No prompt provided", "status": "error"}), 400

        if "image" not in request.files:
            return jsonify({"error": "No image provided", "status": "error"}), 400

        file = request.files["image"]
        image_bytes = file.read()
        image_b64 = _image_to_base64(image_bytes)

        logger.info(f"Detection request - original prompt: '{prompt}'")

        # Step 1: Generate CLIP embedding for user image
        embed_resp = _call_service(
            "post",
            f"{EMBEDDING_SERVICE_URL}/embed/image",
            json={"image_base64": image_b64},
        )
        query_embedding = embed_resp["embedding"]
        logger.info("  Generated query embedding")

        # Step 2: KNN search in Elasticsearch for 2 most similar images
        search_resp = _call_service(
            "post",
            f"{STORAGE_SERVICE_URL}/search/similar",
            json={"embedding": query_embedding, "k": 2},
        )
        similar_results = search_resp.get("results", [])
        logger.info(f"  Found {len(similar_results)} similar images")

        # Step 3: Augment prompt with descriptions from similar images
        augmented_prompt = prompt
        descriptions_used = []
        for match in similar_results:
            desc = match.get("description", "")
            if desc and desc != "unknown":
                descriptions_used.append(desc)
                # Convert dot-separated to space-separated with dots for Grounding DINO
                desc_items = desc.replace(".", ". ")
                augmented_prompt = augmented_prompt.rstrip(".") + ". " + desc_items

        # Ensure proper Grounding DINO prompt format (period-separated phrases)
        augmented_prompt = augmented_prompt.strip()
        if not augmented_prompt.endswith("."):
            augmented_prompt += "."

        logger.info(f"  Augmented prompt: '{augmented_prompt[:120]}...'")

        # Step 4: Run Grounding DINO detection
        detect_resp = _call_service(
            "post",
            f"{DETECTION_SERVICE_URL}/detect",
            json={
                "image_base64": image_b64,
                "prompt": augmented_prompt,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
            },
        )
        logger.info(f"  Detected {detect_resp.get('count', 0)} objects")

        # Step 5: Store detection result in MongoDB
        result_doc = {
            "original_prompt": prompt,
            "augmented_prompt": augmented_prompt,
            "descriptions_used": descriptions_used,
            "similar_images": [
                {"image_id": m["image_id"], "score": m["score"]}
                for m in similar_results
            ],
            "boxes": detect_resp.get("boxes", []),
            "labels": detect_resp.get("labels", []),
            "scores": detect_resp.get("scores", []),
            "count": detect_resp.get("count", 0),
            "created_at": int(time.time() * 1000),
        }

        store_resp = _call_service(
            "post",
            f"{STORAGE_SERVICE_URL}/store/result",
            json=result_doc,
        )
        result_id = store_resp.get("result_id", "")

        return jsonify({
            "result_id": result_id,
            "original_prompt": prompt,
            "augmented_prompt": augmented_prompt,
            "descriptions_used": descriptions_used,
            "boxes": detect_resp.get("boxes", []),
            "labels": detect_resp.get("labels", []),
            "scores": detect_resp.get("scores", []),
            "count": detect_resp.get("count", 0),
            "annotated_image": detect_resp.get("annotated_image", ""),
            "similar_images": similar_results,
            "status": "success",
        })

    except (ConnectionError, TimeoutError) as e:
        logger.error(f"Service error during detection: {e}")
        return jsonify({"error": str(e), "status": "error"}), 503
    except Exception as e:
        logger.error(f"Error in detection: {e}", exc_info=True)
        return jsonify({"error": str(e), "status": "error"}), 500


# ─── Data Listing Endpoints ──────────────────────────────────────────────────

@app.route("/api/images", methods=["GET"])
def list_images():
    """List all indexed images."""
    try:
        resp = _call_service("get", f"{STORAGE_SERVICE_URL}/list/images")
        return jsonify(resp)
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/api/images/<image_id>", methods=["GET"])
def get_image(image_id: str):
    """Proxy image retrieval from storage service."""
    try:
        resp = http_requests.get(
            f"{STORAGE_SERVICE_URL}/get/image/{image_id}",
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.content, 200, {"Content-Type": "image/jpeg"}
        return jsonify({"error": "Image not found", "status": "error"}), 404
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/api/embeddings", methods=["GET"])
def list_embeddings():
    """List all indexed embeddings/documents."""
    try:
        resp = _call_service("get", f"{STORAGE_SERVICE_URL}/list/embeddings")
        return jsonify(resp)
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/api/results", methods=["GET"])
def list_results():
    """List all detection results."""
    try:
        resp = _call_service("get", f"{STORAGE_SERVICE_URL}/list/results")
        return jsonify(resp)
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/api/results/<result_id>", methods=["GET"])
def get_result(result_id: str):
    """Get a specific detection result."""
    try:
        resp = _call_service("get", f"{STORAGE_SERVICE_URL}/get/result/{result_id}")
        return jsonify(resp)
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


if __name__ == "__main__":
    logger.info("Starting API Gateway on port 5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
