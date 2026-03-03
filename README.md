# 🎯 Agentic Multi-Modal Object Detection System

A microservices-based object detection system that combines **CLIP**, **LLaVA**, and **Grounding DINO** models with **Elasticsearch** vector search and **MongoDB** image storage to deliver prompt-augmented zero-shot object detection.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    React Frontend (:3000)                        │
└──────────────────────┬───────────────────────────────────────────┘
                       │ HTTP REST
┌──────────────────────▼───────────────────────────────────────────┐
│                   API Gateway (:5000)                             │
│              Orchestrates all pipelines                           │
└───────┬──────────────┬──────────────────────┬────────────────────┘
        │              │                      │
┌───────▼──────┐ ┌─────▼──────────┐ ┌────────▼─────────┐
│  Embedding   │ │    Storage     │ │   Detection      │
│  Service     │ │    Service     │ │   Service        │
│  (:5001)     │ │    (:5002)     │ │   (:5003)        │
│  CLIP+LLaVA  │ │  ES+MongoDB   │ │  Grounding DINO  │
└──────────────┘ └──┬─────────┬──┘ └──────────────────┘
                    │         │
           ┌────────▼──┐  ┌──▼──────────┐
           │Elasticsearch│  │  MongoDB    │
           │  (:9200)    │  │  (:27017)   │
           └─────────────┘  └─────────────┘
```

## Features

### Pipeline 1: Image Indexing
1. Upload images through the UI
2. **CLIP** generates 512-dim embeddings for each image
3. **LLaVA** generates structured text descriptions (e.g., `apple.potato.tomato`)
4. Embeddings + descriptions stored in **Elasticsearch** (dense_vector + KNN)
5. Raw images stored in **MongoDB** (GridFS)

### Pipeline 2: Object Detection with Prompt Augmentation
1. User uploads an image + text prompt
2. **CLIP** generates embedding for the query image
3. **Elasticsearch KNN** finds the 2 most similar indexed images
4. Text descriptions from similar images are **added to the prompt**
5. **Grounding DINO** runs zero-shot detection with the augmented prompt
6. Results (bounding boxes + annotated image) stored and displayed

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | React JS |
| API Gateway | Flask (Python) |
| Embedding Service | Flask + PyTorch + HuggingFace Transformers |
| Storage Service | Flask + Elasticsearch + PyMongo |
| Detection Service | Flask + PyTorch + HuggingFace Transformers |
| Vector Search | Elasticsearch 8.x (dense_vector + KNN) |
| Image Storage | MongoDB 7.0 (GridFS) |
| Containerization | Docker + Docker Compose |

## Models

All three models are downloaded from **HuggingFace Hub**, quantized once via `scripts/quantize_models.py`, and saved locally to `./models/quantized/`. Services load only the saved quantized weights at startup — no re-downloading or re-quantizing on each restart.

The original (full-precision) downloads are cached in `./models/cache/` so you keep a backup of the original weights even if HuggingFace removes the models in the future.

| Model (HuggingFace ID) | Service | Purpose | Saved Format | Params | Size on Disk |
|---|---|---|---|---|---|
| `openai/clip-vit-base-patch32` | Embedding Service | Image & text embeddings (512-dim) | FP16 (half-precision) | 151M | 293 MB |
| `llava-hf/llava-1.5-7b-hf` | Embedding Service | Image description generation | 4-bit NF4 (bitsandbytes, double quantization) | 7B | 3.8 GB |
| `IDEA-Research/grounding-dino-base` | Detection Service | Zero-shot object detection | FP16 (half-precision) | 172M | 448 MB |

### Model Details

#### CLIP — `openai/clip-vit-base-patch32`
- **Architecture**: Vision Transformer (ViT-B/32) + Text Transformer
- **Embedding dimension**: 512
- **Quantization**: Saved as FP16 (~300 MB). CLIP is a small model so FP16 provides a good balance between speed and accuracy without needing aggressive quantization.
- **Used for**: Generating image embeddings for KNN similarity search in Elasticsearch, and text embeddings for multimodal matching.

#### LLaVA — `llava-hf/llava-1.5-7b-hf`
- **Architecture**: LLaVA 1.5 (Vicuna-7B language model + CLIP vision encoder)
- **Quantization**: 4-bit NF4 via bitsandbytes with double quantization (`bnb_4bit_use_double_quant=True`), compute dtype FP16
- **Loaded via**: `AutoModelForImageTextToText` + `AutoProcessor` (HuggingFace Transformers 4.47+)
- **Output format**: Dot-separated object list — e.g., `apple.banana.table.chair`
- **Used for**: Generating structured text descriptions of indexed images. These descriptions are later used for prompt augmentation in the detection pipeline.

#### Grounding DINO — `IDEA-Research/grounding-dino-base`
- **Architecture**: DINO (DETR with Improved deNoising anchOr boxes) + grounded pre-training
- **Saved format**: FP16 (half-precision). Grounding DINO does not support `device_map='auto'` with bitsandbytes quantization in Transformers 4.47, so we save it in FP16 instead (~448 MB — small enough that FP16 is efficient).
- **Used for**: Zero-shot object detection — given an image and a text prompt (e.g., `"apple. car. person."`), outputs bounding boxes with labels and confidence scores.
- **Prompt augmentation**: The detection prompt is augmented with descriptions from similar indexed images found via Elasticsearch KNN search.

### Quantization Workflow

```
HuggingFace Hub              Local: ./models/cache/         Local: ./models/quantized/
─────────────────            ──────────────────────         ──────────────────────────────
openai/clip-vit-base-patch32  ──► cache/hub/clip-*           ──► clip-vit-base-patch32-8bit/
  (FP32, ~600 MB)                (original backup)               (FP16, 293 MB)

llava-hf/llava-1.5-7b-hf     ──► cache/hub/llava-*          ──► llava-1.5-7b-4bit/
  (FP16, ~14 GB)                 (original backup)               (4-bit NF4, 3.8 GB)

IDEA-Research/grounding-dino  ──► cache/hub/dino-*           ──► grounding-dino-base-fp16/
  -base (FP32, ~900 MB)          (original backup)               (FP16, 448 MB)
```

> **Both directories are on your local disk.** If HuggingFace ever removes these models, you still have the original weights in `./models/cache/` and the quantized versions in `./models/quantized/`. These directories are excluded from git (via `.gitignore`) since they are too large for GitHub.

### VRAM Requirements

| Configuration | VRAM Needed | Notes |
|---|---|---|
| All 3 models loaded | ~6.5 GB | LLaVA 4-bit (~3.8 GB) + CLIP FP16 (~0.3 GB) + DINO FP16 (~0.9 GB) + overhead |
| Embedding only (CLIP + LLaVA) | ~4.5 GB | Sufficient for indexing pipeline |
| Detection only (DINO) | ~1.5 GB | Sufficient for detection pipeline |

### Disk Space Requirements

| Directory | Size | Contents |
|---|---|---|
| `models/cache/` | ~16 GB | Original full-precision weights (HuggingFace download cache) |
| `models/quantized/` | ~4.5 GB | Quantized/FP16 weights used for inference |
| **Total** | **~20.5 GB** | Both directories combined |

> **Note**: Tested on NVIDIA RTX 4070 Laptop GPU (8 GB VRAM). All 3 models fit in memory simultaneously.

## Prerequisites

- **Docker** + **Docker Compose** v2
- **NVIDIA GPU** with CUDA support (recommended)
- **NVIDIA Container Toolkit** (for GPU access in Docker)
- At least **16 GB RAM** (32 GB recommended for all 3 models)
- At least **~21 GB free disk space** for model weights (cache + quantized)

## Quick Start

### Step 1: Clone the repository

```bash
git clone https://github.com/<your-username>/agentic_multiModalities_ObjectDetection_project_.git
cd agentic_multiModalities_ObjectDetection_project_
```

> After cloning, the `models/cache/` and `models/quantized/` directories will be empty (model weights are not stored in git).

### Step 2: Build Docker images

```bash
sudo docker compose build
```

This builds:
- `ml_base` — shared PyTorch + Transformers + bitsandbytes base image (built once, reused by ML services)
- `embedding_service`, `detection_service` — ML services inheriting from `ml_base`
- `storage_service`, `api_gateway` — lightweight Python services
- `frontend` — React app served via Nginx

### Step 3: Download & quantize models (run once)

This downloads all 3 models from HuggingFace, quantizes them, and saves everything locally:

```bash
sudo docker compose --profile quantize run --rm quantize_models
```

After completion, your local directory will contain:

```
models/
├── cache/                              ← HuggingFace download cache (~16 GB)
│   └── hub/                               Original full-precision weights (your backup)
│       ├── models--openai--clip-vit-base-patch32/
│       ├── models--llava-hf--llava-1.5-7b-hf/
│       └── models--IDEA-Research--grounding-dino-base/
└── quantized/                          ← Quantized weights (~4.5 GB total)
    ├── clip-vit-base-patch32-8bit/        CLIP in FP16 (293 MB)
    ├── llava-1.5-7b-4bit/                 LLaVA in 4-bit NF4 (3.8 GB)
    └── grounding-dino-base-fp16/          Grounding DINO in FP16 (448 MB)
```

**Time estimate**: ~2-4 hours depending on internet speed (LLaVA alone is ~14 GB download).

You can also quantize individual models or force re-quantization:

```bash
# Quantize only CLIP
sudo docker compose --profile quantize run --rm quantize_models --models clip

# Quantize only LLaVA
sudo docker compose --profile quantize run --rm quantize_models --models llava

# Quantize only Grounding DINO
sudo docker compose --profile quantize run --rm quantize_models --models dino

# Re-quantize all (force overwrite existing)
sudo docker compose --profile quantize run --rm quantize_models --models all --force
```

**Verify models are ready:**

```bash
ls models/quantized/
# Expected: clip-vit-base-patch32-8bit  llava-1.5-7b-4bit  grounding-dino-base-fp16
```

### Step 4: Start all services

```bash
sudo docker compose up -d
```

### CPU only (no NVIDIA GPU)

```bash
sudo docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

### Access the application

| Service | URL |
|---------|-----|
| Frontend UI | http://localhost:3000 |
| API Gateway | http://localhost:5000/api/health |
| Embedding Service | http://localhost:5001/health |
| Storage Service | http://localhost:5002/health |
| Detection Service | http://localhost:5003/health |
| Elasticsearch | http://localhost:9200 |
| MongoDB | localhost:27017 |

## Usage

### 1. Index Images
1. Open http://localhost:3000
2. Go to the **Index** tab
3. Drag & drop or select images
4. Click **Index Images** — the system will generate embeddings and descriptions

### 2. Detect Objects
1. Go to the **Detect** tab
2. Upload a query image
3. Enter a prompt (e.g., `apple. car. person.`)
4. Adjust thresholds if needed
5. Click **Run Detection**
6. The system augments your prompt with descriptions from similar indexed images, then runs Grounding DINO

### 3. Browse Gallery
- The **Gallery** tab shows all indexed images with their LLaVA-generated descriptions

### 4. View History
- The **History** tab shows all past detection results with details

## API Reference

### Image Indexing
```
POST /api/index
Content-Type: multipart/form-data
Body: images (file[])

Response: { results: [{filename, image_id, doc_id, description}], indexed_count }
```

### Object Detection
```
POST /api/detect
Content-Type: multipart/form-data
Body: image (file), prompt (str), box_threshold (float), text_threshold (float)

Response: {
  original_prompt, augmented_prompt, descriptions_used,
  boxes, labels, scores, count, annotated_image (base64),
  similar_images
}
```

### List Images
```
GET /api/images
Response: { images: [{image_id, filename, length, upload_date}] }
```

### List Results
```
GET /api/results
Response: { results: [{original_prompt, augmented_prompt, labels, scores, ...}] }
```

### Health Check
```
GET /api/health
Response: { status, services: {embedding_service, storage_service, detection_service} }
```

## Project Structure

```
├── docker-compose.yml              # Main orchestration (GPU)
├── docker-compose.cpu.yml          # CPU override
├── models/                             # Model weights (NOT in git — .gitignore'd)
│   ├── cache/                          #   HuggingFace download cache (original backups)
│   │   └── hub/                        #   Full-precision weights from HuggingFace
│   └── quantized/                      #   Quantized weights (used by services)
│       ├── clip-vit-base-patch32-8bit/ #     CLIP FP16 (293 MB)
│       ├── llava-1.5-7b-4bit/          #     LLaVA 4-bit NF4 (3.8 GB)
│       └── grounding-dino-base-fp16/   #     Grounding DINO FP16 (448 MB)
├── scripts/
│   └── quantize_models.py          # Download → quantize → save models
├── services/
│   ├── ml_base/                    # Shared ML base Docker image
│   │   └── Dockerfile              #   PyTorch + Transformers + bitsandbytes (built once)
│   ├── api_gateway/                # Orchestrator service
│   │   ├── app.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── embedding_service/          # CLIP + LLaVA models
│   │   ├── app.py
│   │   ├── models.py               #   CLIPEmbedder, LLaVADescriber (loads from /models/quantized/)
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── storage_service/            # Elasticsearch + MongoDB
│   │   ├── app.py
│   │   ├── clients.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── detection_service/          # Grounding DINO
│       ├── app.py
│       ├── models.py               #   GroundingDINODetector (loads from /models/quantized/)
│       ├── Dockerfile
│       └── requirements.txt
├── frontend/                       # React JS UI
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ImageUploader.jsx
│   │   │   ├── ObjectDetector.jsx
│   │   │   ├── Gallery.jsx
│   │   │   ├── ResultsHistory.jsx
│   │   │   └── HealthStatus.jsx
│   │   ├── services/
│   │   │   └── api.js
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── index.js
│   ├── Dockerfile
│   ├── nginx.conf
│   └── package.json
├── docs/
│   └── diagrams.md                 # UML diagrams (Mermaid)
└── sample_images/                  # Test images
```

### Local Directories (bind mounts)

| Host Path | Container Path | Purpose |
|---|---|---|
| `./models/cache/` | `/root/.cache/huggingface` | HuggingFace download cache — original full-precision model weights. Acts as a **backup** in case models are removed from HuggingFace. |
| `./models/quantized/` | `/models/quantized` | Quantized model weights used for inference. Created by `quantize_models` service, mounted read-only in ML services. |

### Docker Volumes (managed by Docker)

| Volume | Container Path | Purpose |
|---|---|---|
| `es_data` | `/usr/share/elasticsearch/data` | Elasticsearch index data (embeddings + metadata) |
| `mongo_data` | `/data/db` | MongoDB database files (images stored via GridFS) |

> **Why bind mounts for models?** So you can see, inspect, and back up the model files directly on your filesystem. Docker volumes are hidden inside `/var/lib/docker/volumes/` and are harder to manage. Since the models are ~20 GB total, having them visible ensures you never accidentally lose them.

## Design Principles

- **Single Responsibility**: Each microservice handles one domain concern
- **Loose Coupling**: Services communicate only via REST — no shared state
- **Open/Closed**: New models/services can be added without modifying existing ones
- **Interface Segregation**: Each service exposes only the endpoints it owns
- **Dependency Inversion**: API Gateway depends on abstractions (HTTP interfaces), not concrete implementations

## License

MIT
