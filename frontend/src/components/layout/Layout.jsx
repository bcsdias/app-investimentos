import React from 'react';
import { Outlet } from 'react-router-dom';
import { Header } from './Header';

export function Layout({ children }) {
  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col">
      <Header />
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 py-6">
        {children ? children : <Outlet />}
      </main>
    </div>
  );
}

export default Layout;
