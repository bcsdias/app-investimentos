import React from 'react';

export function Skeleton({ className = 'h-6 w-full', ...props }) {
  return (
    <div
      data-testid="skeleton"
      className={`animate-pulse rounded-lg bg-zinc-800/80 ${className}`}
      {...props}
    />
  );
}

export default Skeleton;
