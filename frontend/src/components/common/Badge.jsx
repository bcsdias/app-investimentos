import React from 'react';

const VARIANTS = {
  success: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
  warning: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
  danger: 'bg-red-500/15 text-red-400 border-red-500/30',
  info: 'bg-blue-500/15 text-blue-400 border-blue-500/30',
  neutral: 'bg-zinc-800 text-zinc-300 border-zinc-700',
};

export function Badge({ children, variant = 'neutral', className = '', ...props }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium border ${VARIANTS[variant] || VARIANTS.neutral} ${className}`}
      {...props}
    >
      {children}
    </span>
  );
}

export default Badge;

