import React, { useState, useEffect } from 'react';
import { checkHealth } from '../services/api';

const HealthStatus = () => {
  const [health, setHealth] = useState(null);

  useEffect(() => {
    const check = async () => {
      try {
        const data = await checkHealth();
        setHealth(data);
      } catch {
        setHealth({ status: 'unreachable', services: {} });
      }
    };
    check();
    const interval = setInterval(check, 30000);
    return () => clearInterval(interval);
  }, []);

  if (!health) return null;

  const statusIcon = (s) => {
    if (s === 'healthy') return '🟢';
    if (s === 'unreachable') return '🔴';
    return '🟡';
  };

  return (
    <div className="health-bar">
      <span className="health-overall">
        {statusIcon(health.status)} System: {health.status}
      </span>
      {health.services && Object.entries(health.services).map(([name, status]) => (
        <span key={name} className="health-service">
          {statusIcon(status)} {name.replace('_', ' ')}
        </span>
      ))}
    </div>
  );
};

export default HealthStatus;
