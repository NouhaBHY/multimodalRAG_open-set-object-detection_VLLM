import React, { useState, useEffect } from 'react';
import { listResults } from '../services/api';

const ResultsHistory = () => {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState(null);

  useEffect(() => {
    fetchResults();
  }, []);

  const fetchResults = async () => {
    setLoading(true);
    try {
      const data = await listResults();
      setResults(data.results || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <div className="card"><p>Loading results...</p></div>;
  if (error) return <div className="card"><div className="alert alert-error">{error}</div></div>;

  return (
    <div className="card">
      <h2>📋 Detection Results History</h2>
      <p className="subtitle">{results.length} detection results stored.</p>

      <button className="btn btn-secondary" onClick={fetchResults}>
        🔄 Refresh
      </button>

      {results.length === 0 ? (
        <div className="empty-state">
          <p>No detection results yet. Go to the Detect tab to run object detection.</p>
        </div>
      ) : (
        <div className="results-list">
          {results.map((r, i) => (
            <div
              key={r._id || i}
              className={`result-card ${expanded === i ? 'expanded' : ''}`}
              onClick={() => setExpanded(expanded === i ? null : i)}
            >
              <div className="result-header">
                <span className="result-num">#{i + 1}</span>
                <span className="result-prompt">{r.original_prompt}</span>
                <span className="result-count">{r.count} objects</span>
                <span className="result-time">
                  {r.created_at ? new Date(r.created_at).toLocaleString() : ''}
                </span>
              </div>

              {expanded === i && (
                <div className="result-details">
                  <div className="detail-row">
                    <strong>Augmented Prompt:</strong>
                    <p>{r.augmented_prompt}</p>
                  </div>
                  <div className="detail-row">
                    <strong>Descriptions Used:</strong>
                    <div className="desc-tags">
                      {r.descriptions_used?.map((d, j) => (
                        <span key={j} className="badge">{d}</span>
                      ))}
                    </div>
                  </div>
                  <div className="detail-row">
                    <strong>Detected Objects:</strong>
                    <ul>
                      {r.labels?.map((label, j) => (
                        <li key={j}>
                          {label} — {(r.scores[j] * 100).toFixed(1)}%
                          <span className="bbox-small">
                            [{r.boxes[j]?.map(v => v.toFixed(0)).join(', ')}]
                          </span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default ResultsHistory;
