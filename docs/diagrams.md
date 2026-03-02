# System Architecture
```mermaid
graph TB
    subgraph "Frontend"
        UI[React JS Frontend<br/>Port 3000]
    end
    
    subgraph "API Layer"
        GW[API Gateway<br/>Flask - Port 5000]
    end
    
    subgraph "Microservices"
        subgraph "ML Base Image<br/>(PyTorch + Transformers — built once)"
            ES[Embedding Service<br/>Flask - Port 5001<br/>CLIP + LLaVA]
            DS[Detection Service<br/>Flask - Port 5003<br/>Grounding DINO]
        end
        SS[Storage Service<br/>Flask - Port 5002<br/>ES + MongoDB ops]
    end
    
    subgraph "Data Stores"
        ELASTIC[(Elasticsearch<br/>Port 9200<br/>Embeddings + Text)]
        MONGO[(MongoDB<br/>Port 27017<br/>Images)]
    end
    
    UI -->|HTTP REST| GW
    GW -->|/embed| ES
    GW -->|/store, /search| SS
    GW -->|/detect| DS
    
    ES -->|Generate embeddings + descriptions| GW
    SS -->|CRUD operations| ELASTIC
    SS -->|CRUD operations| MONGO
    DS -->|Augmented detection| GW
    
    SS -.->|Search similar embeddings| ELASTIC
    SS -.->|Store/retrieve images| MONGO
```

# Use Case Diagram
```mermaid
graph LR
    User((User))
    
    subgraph "Use Cases"
        UC1[Upload Images for Indexing]
        UC2[View Indexed Images]
        UC3[Upload Image + Prompt for Detection]
        UC4[View Detection Results]
        UC5[Search Similar Images]
    end
    
    subgraph "System Actors"
        CLIP[CLIP Model]
        LLAVA[LLaVA Model]
        GDINO[Grounding DINO]
        ESDB[(Elasticsearch)]
        MONGODB[(MongoDB)]
    end
    
    User --> UC1
    User --> UC2
    User --> UC3
    User --> UC4
    User --> UC5
    
    UC1 -.->|includes| CLIP
    UC1 -.->|includes| LLAVA
    UC1 -.->|includes| ESDB
    UC1 -.->|includes| MONGODB
    
    UC3 -.->|includes| CLIP
    UC3 -.->|includes| ESDB
    UC3 -.->|includes| GDINO
    
    UC5 -.->|includes| CLIP
    UC5 -.->|includes| ESDB
```

# Sequence Diagram — Image Indexing Pipeline
```mermaid
sequenceDiagram
    actor User
    participant UI as React Frontend
    participant GW as API Gateway
    participant EMB as Embedding Service
    participant STR as Storage Service
    participant ES as Elasticsearch
    participant MDB as MongoDB

    User->>UI: Upload images
    UI->>GW: POST /api/index (images)
    
    GW->>STR: POST /store/images (images)
    STR->>MDB: Store raw images
    MDB-->>STR: image_ids[]
    STR-->>GW: image_ids[]
    
    loop For each image
        GW->>EMB: POST /embed/image (image)
        Note over EMB: CLIP generates embedding
        EMB-->>GW: embedding vector (512-dim)
        
        GW->>EMB: POST /describe/image (image)
        Note over EMB: LLaVA generates description
        EMB-->>GW: text description "apple.potato.tomato"
        
        GW->>STR: POST /store/embedding
        Note over STR: {image_id, embedding, description}
        STR->>ES: Index document with dense_vector + text
        ES-->>STR: doc_id
        STR-->>GW: success
    end
    
    GW-->>UI: Indexing complete
    UI-->>User: Show success + indexed images
```

# Sequence Diagram — Object Detection Pipeline
```mermaid
sequenceDiagram
    actor User
    participant UI as React Frontend
    participant GW as API Gateway
    participant EMB as Embedding Service
    participant STR as Storage Service
    participant DET as Detection Service
    participant ES as Elasticsearch

    User->>UI: Upload image + text prompt
    UI->>GW: POST /api/detect (image, prompt)
    
    GW->>EMB: POST /embed/image (user_image)
    Note over EMB: CLIP generates query embedding
    EMB-->>GW: query_embedding (512-dim)
    
    GW->>STR: POST /search/similar (query_embedding, k=2)
    STR->>ES: KNN search on dense_vector
    ES-->>STR: Top 2 matches with descriptions
    STR-->>GW: ["apple.potato.tomato", "car.bus.tree"]
    
    Note over GW: Augment prompt: original + desc1 + desc2
    
    GW->>DET: POST /detect (image, augmented_prompt)
    Note over DET: Grounding DINO runs detection
    DET-->>GW: bounding_boxes, labels, scores, annotated_image
    
    GW->>STR: POST /store/result (detection_result)
    STR-->>GW: result_id
    
    GW-->>UI: Detection results + annotated image
    UI-->>User: Display bounding boxes + labels
```

# Class Diagram
```mermaid
classDiagram
    class CLIPEmbedder {
        -model: CLIPModel
        -processor: CLIPProcessor
        -device: str
        +__init__(model_name, quantized)
        +generate_embedding(image) ndarray
        +generate_text_embedding(text) ndarray
    }

    class LLaVADescriber {
        -model: LlavaForConditionalGeneration
        -processor: AutoProcessor
        -device: str
        +__init__(model_name, quantized)
        +generate_description(image) str
        -_format_output(raw_text) str
    }

    class GroundingDINODetector {
        -model: AutoModelForZeroShotObjectDetection
        -processor: AutoProcessor
        -device: str
        +__init__(model_name, quantized)
        +detect(image, prompt, threshold) DetectionResult
        +annotate_image(image, result) Image
    }

    class DetectionResult {
        +boxes: List~BBox~
        +labels: List~str~
        +scores: List~float~
        +annotated_image: bytes
    }

    class ElasticsearchClient {
        -client: Elasticsearch
        -index_name: str
        +__init__(host, port, index_name)
        +create_index()
        +index_document(image_id, embedding, description) str
        +search_similar(embedding, k) List~dict~
        +get_document(doc_id) dict
        +delete_document(doc_id) bool
    }

    class MongoDBClient {
        -client: MongoClient
        -db: Database
        -collection: Collection
        +__init__(host, port, db_name)
        +store_image(image_bytes, metadata) str
        +get_image(image_id) bytes
        +list_images() List~dict~
        +delete_image(image_id) bool
    }

    class EmbeddingService {
        -clip: CLIPEmbedder
        -llava: LLaVADescriber
        +embed_image(image) dict
        +describe_image(image) dict
    }

    class StorageService {
        -es_client: ElasticsearchClient
        -mongo_client: MongoDBClient
        +store_image(image) str
        +store_embedding(image_id, embedding, desc) str
        +search_similar(embedding, k) List
        +get_image(image_id) bytes
        +store_result(result) str
    }

    class DetectionService {
        -detector: GroundingDINODetector
        +detect(image, prompt) DetectionResult
    }

    class APIGateway {
        +index_images(images) dict
        +detect_objects(image, prompt) dict
        +get_images() List
        +get_results() List
    }

    EmbeddingService --> CLIPEmbedder
    EmbeddingService --> LLaVADescriber
    StorageService --> ElasticsearchClient
    StorageService --> MongoDBClient
    DetectionService --> GroundingDINODetector
    GroundingDINODetector --> DetectionResult
    APIGateway --> EmbeddingService
    APIGateway --> StorageService
    APIGateway --> DetectionService
```
