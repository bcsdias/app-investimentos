import React from 'react';

export function Card({ title, subtitle, action, children, className = '' }) {
  return (
    <div
      className={`bg-zinc-900/90 border border-zinc-800 rounded-xl p-5 shadow-lg shadow-black/20 ${className}`}
    >
      {(title || subtitle || action) && (
        <div className="flex items-start justify-between gap-4 mb-4">
          <div>
            {title && <h3 className="text-base font-semibold text-zinc-100">{title}</h3>}
            {subtitle && <p className="text-xs text-zinc-400 mt-0.5">{subtitle}</p>}
          </div>
          {action && <div className="shrink-0">{action}</div>}
        </div>
      )}
      {children}
    </div>
  );
}

export default Card;
