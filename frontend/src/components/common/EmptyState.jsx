import React from 'react';
import { Inbox } from 'lucide-react';

export function EmptyState({ title, description, action, icon: Icon = Inbox }) {
  return (
    <div
      data-testid="empty-state"
      className="flex flex-col items-center justify-center text-center py-12 px-6 bg-zinc-900/50 border border-dashed border-zinc-800 rounded-xl"
    >
      <div className="p-3 rounded-full bg-zinc-800/80 text-zinc-400 mb-3">
        <Icon className="w-6 h-6" />
      </div>
      <h4 className="text-base font-semibold text-zinc-200">{title}</h4>
      {description && <p className="text-sm text-zinc-400 mt-1 max-w-md">{description}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export default EmptyState;

