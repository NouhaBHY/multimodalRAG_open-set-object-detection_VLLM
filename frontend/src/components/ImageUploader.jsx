import React, { useState, useCallback } from 'react';
import { indexImages } from '../services/api';

const ImageUploader = ({ onIndexComplete }) => {
  const [files, setFiles] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = useCallback((e) => {
    const selectedFiles = Array.from(e.target.files);
    setFiles(selectedFiles);
    setResults(null);
    setError(null);

    // Generate previews
    const newPreviews = selectedFiles.map((file) => ({
      name: file.name,
      url: URL.createObjectURL(file),
    }));
    setPreviews(newPreviews);
  }, []);

  const handleUpload = async () => {
    if (files.length === 0) return;

    setLoading(true);
    setProgress(0);
    setError(null);
    setResults(null);

    try {
      const data = await indexImages(files, (event) => {
        const pct = Math.round((event.loaded * 100) / event.total);
        setProgress(pct);
      });
      setResults(data);
      if (onIndexComplete) onIndexComplete(data);
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Indexing failed');
    } finally {
      setLoading(false);
    }
  };

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    const droppedFiles = Array.from(e.dataTransfer.files).filter((f) =>
      f.type.startsWith('image/')
    );
    setFiles(droppedFiles);
    setResults(null);
    setError(null);
    const newPreviews = droppedFiles.map((file) => ({
      name: file.name,
      url: URL.createObjectURL(file),
    }));
    setPreviews(newPreviews);
  }, []);

  return (
    <div className="card">
      <h2>📥 Index Images</h2>
      <p className="subtitle">
        Upload images to generate CLIP embeddings and LLaVA descriptions. 
        They will be stored in Elasticsearch &amp; MongoDB.
      </p>

      {/* Drop Zone */}
      <div
        className="drop-zone"
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        onClick={() => document.getElementById('file-input').click()}
      >
        <input
          id="file-input"
          type="file"
          multiple
          accept="image/*"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />
        <div className="drop-zone-content">
          <span className="drop-icon">🖼️</span>
          <p>Drag &amp; drop images here, or click to browse</p>
          {files.length > 0 && (
            <p className="file-count">{files.length} file(s) selected</p>
          )}
        </div>
      </div>

      {/* Previews */}
      {previews.length > 0 && (
        <div className="preview-grid">
          {previews.map((p, i) => (
            <div key={i} className="preview-item">
              <img src={p.url} alt={p.name} />
              <span>{p.name}</span>
            </div>
          ))}
        </div>
      )}

      {/* Upload Button */}
      <button
        className="btn btn-primary"
        onClick={handleUpload}
        disabled={loading || files.length === 0}
      >
        {loading ? `Indexing... ${progress}%` : `Index ${files.length} Image(s)`}
      </button>

      {/* Progress Bar */}
      {loading && (
        <div className="progress-bar">
          <div className="progress-fill" style={{ width: `${progress}%` }} />
        </div>
      )}

      {/* Error */}
      {error && <div className="alert alert-error">❌ {error}</div>}

      {/* Results */}
      {results && (
        <div className="alert alert-success">
          <h3>✅ Indexed {results.indexed_count} image(s)</h3>
          {results.results?.map((r, i) => (
            <div key={i} className="result-item">
              <strong>{r.filename}</strong>
              <span className="badge">{r.description}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default ImageUploader;
