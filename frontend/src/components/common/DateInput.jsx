import React, { useRef } from 'react';
import { CalendarDays } from 'lucide-react';

export function DateInput({
  value,
  onChange,
  ariaLabel,
  min,
  max,
  className = '',
  id,
  name,
  disabled = false,
  ...rest
}) {
  const inputRef = useRef(null);

  const openPicker = () => {
    if (disabled) return;
    try {
      if (inputRef.current && typeof inputRef.current.showPicker === 'function') {
        inputRef.current.showPicker();
      }
    } catch {
      // Ignora erro se navegador não suportar ou restringir showPicker
    }
  };

  const handleInputClick = (e) => {
    rest.onClick?.(e);
    openPicker();
  };

  const handleButtonClick = (e) => {
    e.preventDefault();
    e.stopPropagation();
    inputRef.current?.focus();
    openPicker();
  };

  const label = ariaLabel || rest['aria-label'];

  return (
    <div className="relative inline-flex items-center">
      <input
        ref={inputRef}
        type="date"
        id={id}
        name={name}
        value={value ?? ''}
        onChange={onChange}
        min={min}
        max={max}
        disabled={disabled}
        aria-label={label}
        onClick={handleInputClick}
        className={`bg-zinc-950 border border-zinc-700 rounded-lg pl-2.5 pr-8 py-1 text-xs text-zinc-100 [color-scheme:dark] [&::-webkit-calendar-picker-indicator]:opacity-0 cursor-pointer ${className}`.trim()}
        {...rest}
      />
      <button
        type="button"
        aria-label="Abrir calendário"
        onClick={handleButtonClick}
        disabled={disabled}
        className="absolute right-2 text-zinc-400 hover:text-zinc-200 transition-colors flex items-center justify-center p-0.5 disabled:opacity-50 disabled:pointer-events-none"
      >
        <CalendarDays className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}

export default DateInput;
