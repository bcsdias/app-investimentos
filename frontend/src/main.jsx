import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';

const rootEl = document.getElementById('root');
if (rootEl) {
  ReactDOM.createRoot(rootEl).render(
    <React.StrictMode>
      <div className="min-h-screen bg-zinc-950 text-zinc-100 p-4">App Investimentos</div>
    </React.StrictMode>
  );
}

