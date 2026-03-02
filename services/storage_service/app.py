"""
Storage Service - Flask Application
Provides REST API endpoints for storing/retrieving images, embeddings, and detection results.
"""

import io
import os
import logging
import base64
import time

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from clients import ElasticsearchClient, MongoDBClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global client instances (lazy loaded)
es_client = None
mongo_client = None


def get_es_client() -> ElasticsearchClient:
    global es_client
    if es_client is None:
        es_client = ElasticsearchClient(
            host=os.getenv("ELASTICSEARCH_HOST", "elasticsearch"),
            port=int(os.getenv("ELASTICSEARCH_PORT", 9200)),
        )
    return es_client


def get_mongo_client() -> MongoDBClient:
    global mongo_client
    if mongo_client is None:
        mongo_client = MongoDBClient(
            host=os.getenv("MONGODB_HOST", "mongodb"),
            port=int(os.getenv("MONGODB_PORT", 27017)),
        )
    return mongo_client


# ─── Health ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "storage_service"})


# ─── Image Endpoints ──────────────────────────────────────────────────────────

@app.route("/store/image", methods=["POST"])
def store_image():
    """Store an image in MongoDB GridFS."""
    try:
        if "image" in request.files:
            file = request.files["image"]
            image_bytes = file.read()
            filename = file.filename or "unknown"
        elif request.is_json and "image_base64" in request.json:
            image_bytes = base64.b64decode(request.json["image_base64"])
            filename = request.json.get("filename", "unknown")
        else:
            return jsonify({"error": "No image provided", "status": "error"}), 400

        mongo = get_mongo_client()
        image_id = mongo.store_image(image_bytes, filename)

        return jsonify({
            "image_id": image_id,
            "filename": filename,
            "status": "success",
        })
    except Exception as e:
        logger.error(f"Error storing image: {e}", exc_info=True)
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/get/image/<image_id>", methods=["GET"])
def get_image(image_id: str):
    """Retrieve an image from MongoDB GridFS."""
    try:
        mongo = get_mongo_client()
        image_bytes = mongo.get_image(image_id)
        if image_bytes is None:
            return jsonify({"error": "Image not found", "status": "error"}), 404
        return send_file(io.BytesIO(image_bytes), mimetype="image/jpeg")
    except Exception as e:
        logger.error(f"Error retrieving image: {e}", exc_info=True)
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/list/images", methods=["GET"])
def list_images():
    """List all stored images."""
    try:
        mongo = get_mongo_client()
        images = mongo.list_images()
        return jsonify({"images": images, "count": len(images), "status": "success"})
    except Exception as e:
        logger.error(f"Error listing images: {e}", exc_info=True)
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/delete/image/<image_id>", methods=["DELETE"])
def delete_image(image_id: str):
    """Delete an image from MongoDB."""
    try:
        mongo = get_mongo_client()
        success = mongo.delete_image(image_id)
        if success:
            return jsonify({"status": "success", "message": "Image deleted"})
        return jsonify({"error": "Delete failed", "status": "error"}), 404
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


# ─── Embedding Endpoints ──────────────────────────────────────────────────────

@app.route("/store/embedding", methods=["POST"])
def store_embedding():
    """
    Store an image embedding and description in Elasticsearch.
    Expects JSON: {image_id, embedding, description, filename}
    """
    try:
        data = request.get_json()
        required = ["image_id", "embedding", "description"]
        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}", "status": "error"}), 400

        es = get_es_client()
        doc_id = es.index_document(
            image_id=data["image_id"],
            embedding=data["embedding"],
            description=data["description"],
            filename=data.get("filename", ""),
        )

        return jsonify({"doc_id": doc_id, "status": "success"})
    except Exception as e:
        logger.error(f"Error storing embedding: {e}", exc_info=True)
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/search/similar", methods=["POST"])
def search_similar():
    """
    Search for similar embeddings using KNN in Elasticsearch.
    Expects JSON: {embedding: [...], k: 2}
    """
    try:
        data = request.get_json()
        if "embedding" not in data:
            return jsonify({"error": "Missing 'embedding' field", "status": "error"}), 400

        k = data.get("k", 2)
        es = get_es_client()
        results = es.search_similar(data["embedding"], k=k)

        return jsonify({"results": results, "count": len(results), "status": "success"})
    except Exception as e:
        logger.error(f"Error searching similar: {e}", exc_info=True)
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/list/embeddings", methods=["GET"])
def list_embeddings():
    """List all indexed documents in Elasticsearch."""
    try:
        es = get_es_client()
        docs = es.get_all_documents()
        return jsonify({"documents": docs, "count": len(docs), "status": "success"})
    except Exception as e:
        logger.error(f"Error listing embeddings: {e}", exc_info=True)
        return jsonify({"error": str(e), "status": "error"}), 500


# ─── Detection Results Endpoints ──────────────────────────────────────────────

@app.route("/store/result", methods=["POST"])
def store_result():
    """Store a detection result in MongoDB."""
    try:
        data = request.get_json()
        data["created_at"] = int(time.time() * 1000)

        mongo = get_mongo_client()
        result_id = mongo.store_detection_result(data)

        return jsonify({"result_id": result_id, "status": "success"})
    except Exception as e:
        logger.error(f"Error storing result: {e}", exc_info=True)
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/get/result/<result_id>", methods=["GET"])
def get_result(result_id: str):
    """Retrieve a detection result by ID."""
    try:
        mongo = get_mongo_client()
        result = mongo.get_detection_result(result_id)
        if result is None:
            return jsonify({"error": "Result not found", "status": "error"}), 404
        return jsonify({"result": result, "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/list/results", methods=["GET"])
def list_results():
    """List all detection results."""
    try:
        mongo = get_mongo_client()
        results = mongo.list_detection_results()
        return jsonify({"results": results, "count": len(results), "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


if __name__ == "__main__":
    logger.info("Starting Storage Service on port 5002")
    app.run(host="0.0.0.0", port=5002, debug=False)
