import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 600000, // 10 min for model inference
});

/**
 * Check health of all backend services.
 */
export const checkHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

/**
 * Upload and index one or more images.
 * Pipeline: Store → CLIP embed → LLaVA describe → Index in ES.
 */
export const indexImages = async (files, onProgress) => {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append('images', file);
  });

  const response = await api.post('/index', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: onProgress,
  });
  return response.data;
};

/**
 * Run object detection with prompt augmentation.
 * Pipeline: CLIP embed → ES KNN search → Augment prompt → Grounding DINO.
 */
export const detectObjects = async (file, prompt, boxThreshold = 0.25, textThreshold = 0.25) => {
  const formData = new FormData();
  formData.append('image', file);
  formData.append('prompt', prompt);
  formData.append('box_threshold', boxThreshold);
  formData.append('text_threshold', textThreshold);

  const response = await api.post('/detect', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

/**
 * List all indexed images.
 */
export const listImages = async () => {
  const response = await api.get('/images');
  return response.data;
};

/**
 * Get image URL for display.
 */
export const getImageUrl = (imageId) => {
  return `${API_BASE}/images/${imageId}`;
};

/**
 * List all indexed embeddings.
 */
export const listEmbeddings = async () => {
  const response = await api.get('/embeddings');
  return response.data;
};

/**
 * List all detection results.
 */
export const listResults = async () => {
  const response = await api.get('/results');
  return response.data;
};

/**
 * Get a specific detection result.
 */
export const getResult = async (resultId) => {
  const response = await api.get(`/results/${resultId}`);
  return response.data;
};

export default api;
