import React, { useEffect, useId, useRef } from 'react';
import { X } from 'lucide-react';

const FOCUSABLE_SELECTOR =
  'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"]):not([disabled])';

export function Modal({ isOpen, onClose, title, children, maxWidth = 'max-w-lg' }) {
  const titleId = useId();
  const dialogRef = useRef(null);
  const triggerElementRef = useRef(null);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    if (!isOpen) return undefined;

    triggerElementRef.current = document.activeElement;

    if (dialogRef.current) {
      const focusables = dialogRef.current.querySelectorAll(FOCUSABLE_SELECTOR);
      if (focusables.length > 0) {
        focusables[0].focus();
      } else {
        dialogRef.current.focus?.();
      }
    }

    const onKeyDown = (e) => {
      if (e.key === 'Escape' && onCloseRef.current) {
        e.stopPropagation();
        onCloseRef.current();
        return;
      }

      if (e.key === 'Tab') {
        if (!dialogRef.current) return;
        const focusables = Array.from(dialogRef.current.querySelectorAll(FOCUSABLE_SELECTOR));
        if (focusables.length === 0) {
          e.preventDefault();
          return;
        }

        const first = focusables[0];
        const last = focusables[focusables.length - 1];

        if (e.shiftKey) {
          if (document.activeElement === first || !dialogRef.current.contains(document.activeElement)) {
            e.preventDefault();
            last.focus();
          }
        } else {
          if (document.activeElement === last || !dialogRef.current.contains(document.activeElement)) {
            e.preventDefault();
            first.focus();
          }
        }
      }
    };

    window.addEventListener('keydown', onKeyDown);

    return () => {
      window.removeEventListener('keydown', onKeyDown);
      if (triggerElementRef.current && typeof triggerElementRef.current.focus === 'function') {
        triggerElementRef.current.focus();
      }
    };
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div
      ref={dialogRef}
      role="dialog"
      aria-modal="true"
      aria-labelledby={title ? titleId : undefined}
      aria-label={!title ? 'Modal' : undefined}
      onClick={(e) => {
        if (e.target === e.currentTarget && onClose) {
          onClose();
        }
      }}
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm"
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className={`w-full ${maxWidth} bg-zinc-900 border border-zinc-800 rounded-xl shadow-2xl overflow-hidden`}
      >
        <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-800">
          <h2 id={titleId} className="text-base font-semibold text-zinc-100">
            {title}
          </h2>
          {onClose && (
            <button
              type="button"
              onClick={onClose}
              aria-label="Fechar modal"
              className="p-1 text-zinc-400 hover:text-zinc-100 rounded-lg hover:bg-zinc-800 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
        <div className="p-5">{children}</div>
      </div>
    </div>
  );
}

export default Modal;
