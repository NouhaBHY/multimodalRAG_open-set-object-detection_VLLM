"""
Storage Service - Elasticsearch & MongoDB Clients
Handles persistence of embeddings, text descriptions, images, and detection results.
"""

import logging
import time
from typing import List, Optional

from elasticsearch import Elasticsearch
from pymongo import MongoClient
from bson.objectid import ObjectId
import gridfs

logger = logging.getLogger(__name__)


class ElasticsearchClient:
    """
    Manages the Elasticsearch index for image embeddings and text descriptions.
    Uses dense_vector for KNN similarity search.
    """

    INDEX_NAME = "image_embeddings"
    EMBEDDING_DIM = 512

    def __init__(self, host: str = "elasticsearch", port: int = 9200):
        self.es_url = f"http://{host}:{port}"
        self.client = None
        self._connect_with_retry(max_retries=15, delay=5)
        self._create_index()

    def _connect_with_retry(self, max_retries: int, delay: int):
        """Connect to Elasticsearch with retry logic for container startup."""
        for attempt in range(max_retries):
            try:
                self.client = Elasticsearch(self.es_url)
                if self.client.ping():
                    logger.info(f"Connected to Elasticsearch at {self.es_url}")
                    return
            except Exception as e:
                logger.warning(
                    f"Elasticsearch connection attempt {attempt + 1}/{max_retries} failed: {e}"
                )
            time.sleep(delay)
        raise ConnectionError(f"Could not connect to Elasticsearch at {self.es_url}")

    def _create_index(self):
        """Create the image_embeddings index with dense_vector mapping."""
        if self.client.indices.exists(index=self.INDEX_NAME):
            logger.info(f"Index '{self.INDEX_NAME}' already exists")
            return

        mapping = {
            "mappings": {
                "properties": {
                    "image_id": {"type": "keyword"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": self.EMBEDDING_DIM,
                        "index": True,
                        "similarity": "cosine",
                    },
                    "description": {"type": "text"},
                    "filename": {"type": "keyword"},
                    "created_at": {"type": "date"},
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
        }

        self.client.indices.create(index=self.INDEX_NAME, body=mapping)
        logger.info(f"Created index '{self.INDEX_NAME}'")

    def index_document(
        self,
        image_id: str,
        embedding: list,
        description: str,
        filename: str = "",
    ) -> str:
        """Index an image embedding and description document."""
        doc = {
            "image_id": image_id,
            "embedding": embedding,
            "description": description,
            "filename": filename,
            "created_at": int(time.time() * 1000),
        }
        result = self.client.index(index=self.INDEX_NAME, body=doc, refresh="wait_for")
        doc_id = result["_id"]
        logger.info(f"Indexed document {doc_id} for image {image_id}")
        return doc_id

    def search_similar(self, query_embedding: list, k: int = 2) -> List[dict]:
        """
        Find the k most similar embeddings using KNN search.
        Returns list of {image_id, description, score}.
        """
        query = {
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": k,
                "num_candidates": max(k * 10, 50),
            },
            "_source": ["image_id", "description", "filename"],
        }
        result = self.client.search(index=self.INDEX_NAME, body=query)
        hits = []
        for hit in result["hits"]["hits"]:
            hits.append({
                "doc_id": hit["_id"],
                "image_id": hit["_source"]["image_id"],
                "description": hit["_source"]["description"],
                "filename": hit["_source"].get("filename", ""),
                "score": hit["_score"],
            })
        logger.info(f"KNN search returned {len(hits)} results")
        return hits

    def get_document(self, doc_id: str) -> Optional[dict]:
        """Retrieve a document by its ID."""
        try:
            result = self.client.get(index=self.INDEX_NAME, id=doc_id)
            return result["_source"]
        except Exception:
            return None

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by its ID."""
        try:
            self.client.delete(index=self.INDEX_NAME, id=doc_id, refresh="wait_for")
            return True
        except Exception:
            return False

    def get_all_documents(self) -> List[dict]:
        """Retrieve all indexed documents."""
        result = self.client.search(
            index=self.INDEX_NAME,
            body={"query": {"match_all": {}}, "size": 1000},
            _source=["image_id", "description", "filename", "created_at"],
        )
        return [
            {
                "doc_id": hit["_id"],
                **hit["_source"],
            }
            for hit in result["hits"]["hits"]
        ]


class MongoDBClient:
    """
    Manages MongoDB storage for raw images using GridFS and detection results.
    """

    DB_NAME = "object_detection"

    def __init__(self, host: str = "mongodb", port: int = 27017):
        self.client = None
        self._connect_with_retry(host, port, max_retries=15, delay=5)
        self.db = self.client[self.DB_NAME]
        self.fs = gridfs.GridFS(self.db)
        self.results_collection = self.db["detection_results"]
        logger.info(f"MongoDB client ready, database: {self.DB_NAME}")

    def _connect_with_retry(self, host: str, port: int, max_retries: int, delay: int):
        """Connect to MongoDB with retry logic."""
        for attempt in range(max_retries):
            try:
                self.client = MongoClient(host, port, serverSelectionTimeoutMS=5000)
                self.client.admin.command("ping")
                logger.info(f"Connected to MongoDB at {host}:{port}")
                return
            except Exception as e:
                logger.warning(
                    f"MongoDB connection attempt {attempt + 1}/{max_retries} failed: {e}"
                )
            time.sleep(delay)
        raise ConnectionError(f"Could not connect to MongoDB at {host}:{port}")

    def store_image(self, image_bytes: bytes, filename: str, metadata: dict = None) -> str:
        """Store an image in GridFS. Returns the file ID as string."""
        file_id = self.fs.put(
            image_bytes,
            filename=filename,
            metadata=metadata or {},
        )
        logger.info(f"Stored image '{filename}' with ID {file_id}")
        return str(file_id)

    def get_image(self, image_id: str) -> Optional[bytes]:
        """Retrieve an image from GridFS by ID."""
        try:
            grid_out = self.fs.get(ObjectId(image_id))
            return grid_out.read()
        except Exception as e:
            logger.error(f"Error retrieving image {image_id}: {e}")
            return None

    def get_image_metadata(self, image_id: str) -> Optional[dict]:
        """Get metadata for a stored image."""
        try:
            grid_out = self.fs.get(ObjectId(image_id))
            return {
                "filename": grid_out.filename,
                "length": grid_out.length,
                "upload_date": str(grid_out.upload_date),
                "metadata": grid_out.metadata,
            }
        except Exception:
            return None

    def list_images(self) -> List[dict]:
        """List all stored images with metadata."""
        images = []
        for grid_out in self.fs.find():
            images.append({
                "image_id": str(grid_out._id),
                "filename": grid_out.filename,
                "length": grid_out.length,
                "upload_date": str(grid_out.upload_date),
            })
        return images

    def delete_image(self, image_id: str) -> bool:
        """Delete an image from GridFS."""
        try:
            self.fs.delete(ObjectId(image_id))
            return True
        except Exception:
            return False

    def store_detection_result(self, result: dict) -> str:
        """Store a detection result document."""
        inserted = self.results_collection.insert_one(result)
        return str(inserted.inserted_id)

    def get_detection_result(self, result_id: str) -> Optional[dict]:
        """Retrieve a detection result by ID."""
        try:
            result = self.results_collection.find_one({"_id": ObjectId(result_id)})
            if result:
                result["_id"] = str(result["_id"])
            return result
        except Exception:
            return None

    def list_detection_results(self) -> List[dict]:
        """List all detection results."""
        results = []
        for doc in self.results_collection.find().sort("created_at", -1).limit(50):
            doc["_id"] = str(doc["_id"])
            results.append(doc)
        return results
