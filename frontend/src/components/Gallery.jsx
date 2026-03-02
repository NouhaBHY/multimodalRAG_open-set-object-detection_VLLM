import React, { useState, useEffect } from 'react';
import { listImages, listEmbeddings, getImageUrl } from '../services/api';

const Gallery = () => {
  const [images, setImages] = useState([]);
  const [embeddings, setEmbeddings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [imgResp, embResp] = await Promise.all([
        listImages(),
        listEmbeddings(),
      ]);
      setImages(imgResp.images || []);
      setEmbeddings(embResp.documents || []);
    } catch (err) {
      setError(err.message || 'Failed to load gallery');
    } finally {
      setLoading(false);
    }
  };

  // Merge images with their embedding data
  const enrichedImages = images.map((img) => {
    const embDoc = embeddings.find((e) => e.image_id === img.image_id);
    return {
      ...img,
      description: embDoc?.description || 'No description',
      doc_id: embDoc?.doc_id || '',
    };
  });

  if (loading) return <div className="card"><p>Loading gallery...</p></div>;
  if (error) return <div className="card"><div className="alert alert-error">{error}</div></div>;

  return (
    <div className="card">
      <h2>🖼️ Indexed Images Gallery</h2>
      <p className="subtitle">
        {enrichedImages.length} images indexed with CLIP embeddings and LLaVA descriptions.
      </p>

      <button className="btn btn-secondary" onClick={fetchData}>
        🔄 Refresh
      </button>

      {enrichedImages.length === 0 ? (
        <div className="empty-state">
          <p>No images indexed yet. Go to the Index tab to upload images.</p>
        </div>
      ) : (
        <div className="gallery-grid">
          {enrichedImages.map((img, i) => (
            <div key={i} className="gallery-item">
              <div className="gallery-image">
                <img
                  src={getImageUrl(img.image_id)}
                  alt={img.filename}
                  loading="lazy"
                />
              </div>
              <div className="gallery-info">
                <h4>{img.filename}</h4>
                <div className="description-tags">
                  {img.description.split('.').filter(Boolean).map((tag, j) => (
                    <span key={j} className="tag">{tag.trim()}</span>
                  ))}
                </div>
                <small className="meta">ID: {img.image_id}</small>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default Gallery;
