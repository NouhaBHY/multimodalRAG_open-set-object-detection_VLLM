import React, { useState } from 'react';
import { detectObjects } from '../services/api';

const ObjectDetector = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prompt, setPrompt] = useState('');
  const [boxThreshold, setBoxThreshold] = useState(0.25);
  const [textThreshold, setTextThreshold] = useState(0.25);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setResult(null);
      setError(null);
    }
  };

  const handleDetect = async () => {
    if (!file || !prompt) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await detectObjects(file, prompt, boxThreshold, textThreshold);
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Detection failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h2>🔍 Object Detection</h2>
      <p className="subtitle">
        Upload an image and provide a prompt. The system will augment the prompt 
        with descriptions from similar indexed images, then run Grounding DINO detection.
      </p>

      <div className="form-grid">
        {/* Image Input */}
        <div className="form-group">
          <label>Upload Image</label>
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="file-input"
          />
          {preview && (
            <div className="preview-single">
              <img src={preview} alt="Query" />
            </div>
          )}
        </div>

        {/* Prompt Input */}
        <div className="form-group">
          <label>Detection Prompt</label>
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="e.g., apple. car. person."
            className="text-input"
          />
          <small>Describe objects to detect, separated by periods</small>
        </div>

        {/* Thresholds */}
        <div className="form-row">
          <div className="form-group half">
            <label>Box Threshold: {boxThreshold}</label>
            <input
              type="range"
              min="0.05"
              max="0.95"
              step="0.05"
              value={boxThreshold}
              onChange={(e) => setBoxThreshold(parseFloat(e.target.value))}
            />
          </div>
          <div className="form-group half">
            <label>Text Threshold: {textThreshold}</label>
            <input
              type="range"
              min="0.05"
              max="0.95"
              step="0.05"
              value={textThreshold}
              onChange={(e) => setTextThreshold(parseFloat(e.target.value))}
            />
          </div>
        </div>
      </div>

      {/* Detect Button */}
      <button
        className="btn btn-detect"
        onClick={handleDetect}
        disabled={loading || !file || !prompt}
      >
        {loading ? '🔄 Detecting...' : '🎯 Run Detection'}
      </button>

      {/* Error */}
      {error && <div className="alert alert-error">❌ {error}</div>}

      {/* Results */}
      {result && (
        <div className="detection-results">
          <h3>Detection Results ({result.count} objects found)</h3>

          {/* Prompt Info */}
          <div className="prompt-info">
            <div className="prompt-row">
              <strong>Original Prompt:</strong>
              <span>{result.original_prompt}</span>
            </div>
            <div className="prompt-row">
              <strong>Augmented Prompt:</strong>
              <span className="augmented">{result.augmented_prompt}</span>
            </div>
            {result.descriptions_used?.length > 0 && (
              <div className="prompt-row">
                <strong>Added from similar images:</strong>
                {result.descriptions_used.map((d, i) => (
                  <span key={i} className="badge">{d}</span>
                ))}
              </div>
            )}
          </div>

          {/* Annotated Image */}
          {result.annotated_image && (
            <div className="annotated-image">
              <h4>Annotated Image</h4>
              <img
                src={`data:image/jpeg;base64,${result.annotated_image}`}
                alt="Detection result"
              />
            </div>
          )}

          {/* Detection Table */}
          {result.labels?.length > 0 && (
            <table className="results-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Label</th>
                  <th>Confidence</th>
                  <th>Bounding Box</th>
                </tr>
              </thead>
              <tbody>
                {result.labels.map((label, i) => (
                  <tr key={i}>
                    <td>{i + 1}</td>
                    <td><span className="label-badge">{label}</span></td>
                    <td>{(result.scores[i] * 100).toFixed(1)}%</td>
                    <td className="bbox">
                      [{result.boxes[i].map((v) => v.toFixed(0)).join(', ')}]
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {/* Similar Images */}
          {result.similar_images?.length > 0 && (
            <div className="similar-section">
              <h4>Similar Indexed Images Used</h4>
              <div className="similar-list">
                {result.similar_images.map((sim, i) => (
                  <div key={i} className="similar-item">
                    <span className="filename">{sim.filename || sim.image_id}</span>
                    <span className="score">Score: {sim.score?.toFixed(4)}</span>
                    <span className="badge">{sim.description}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ObjectDetector;
