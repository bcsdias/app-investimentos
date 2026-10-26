import React from 'react';

const VARIANTS = {
  primary:
    'bg-emerald-600 hover:bg-emerald-500 text-white border border-emerald-500/40 shadow-sm shadow-emerald-950/50',
  secondary:
    'bg-zinc-800 hover:bg-zinc-700 text-zinc-100 border border-zinc-700',
  ghost:
    'bg-transparent hover:bg-zinc-800/70 text-zinc-300 hover:text-white border border-transparent',
  danger:
    'bg-red-600/90 hover:bg-red-500 text-white border border-red-500/40',
};

const SIZES = {
  sm: 'px-2.5 py-1.5 text-xs',
  md: 'px-3.5 py-2 text-sm',
  lg: 'px-5 py-2.5 text-base',
};

export function Button({
  children,
  variant = 'primary',
  size = 'md',
  className = '',
  disabled = false,
  isLoading = false,
  type = 'button',
  ...props
}) {
  return (
    <button
      type={type}
      disabled={disabled || isLoading}
      className={`inline-flex items-center justify-center gap-2 font-medium rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-emerald-500/50 disabled:opacity-50 disabled:cursor-not-allowed ${VARIANTS[variant] || VARIANTS.primary} ${SIZES[size] || SIZES.md} ${className}`}
      {...props}
    >
      {isLoading && (
        <span className="h-4 w-4 rounded-full border-2 border-current border-t-transparent animate-spin" />
      )}
      {children}
    </button>
  );
}

export default Button;
