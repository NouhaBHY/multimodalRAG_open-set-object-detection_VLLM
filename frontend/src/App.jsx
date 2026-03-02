import React, { useState } from 'react';
import HealthStatus from './components/HealthStatus';
import ImageUploader from './components/ImageUploader';
import ObjectDetector from './components/ObjectDetector';
import Gallery from './components/Gallery';
import ResultsHistory from './components/ResultsHistory';
import './App.css';

const TABS = [
  { id: 'index', label: '📥 Index', component: ImageUploader },
  { id: 'detect', label: '🔍 Detect', component: ObjectDetector },
  { id: 'gallery', label: '🖼️ Gallery', component: Gallery },
  { id: 'history', label: '📋 History', component: ResultsHistory },
];

function App() {
  const [activeTab, setActiveTab] = useState('index');

  const ActiveComponent = TABS.find((t) => t.id === activeTab)?.component;

  return (
    <div className="app">
      <header className="app-header">
        <h1>🎯 Multi-Modal Object Detection</h1>
        <p>Powered by CLIP · LLaVA · Grounding DINO · Elasticsearch · MongoDB</p>
      </header>

      <HealthStatus />

      <nav className="tab-nav">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      <main className="app-main">
        {ActiveComponent && <ActiveComponent />}
      </main>

      <footer className="app-footer">
        <p>Agentic Multimodal Object Detection • Microservices Architecture</p>
      </footer>
    </div>
  );
}

export default App;
