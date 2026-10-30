import React, { useMemo, lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AuthProvider, useAuth } from './context/AuthContext';
import { PortfolioFilterProvider } from './context/PortfolioFilterContext';
import { RequireAuth } from './components/common/RequireAuth';
import { Layout } from './components/layout/Layout';
import { Skeleton } from './components/common/Skeleton';
import { LoginPage } from './pages/LoginPage';
import { shouldRetry } from './api/errors';

const PortfolioPage = lazy(() =>
  import('./pages/PortfolioPage').then((m) => ({ default: m.PortfolioPage }))
);
const MigrationPage = lazy(() =>
  import('./pages/MigrationPage').then((m) => ({ default: m.MigrationPage }))
);

export function createQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        refetchOnWindowFocus: false,
        retry: shouldRetry,
      },
    },
  });
}

function AppRoutes() {
  const { user } = useAuth();

  return (
    <PortfolioFilterProvider key={user?.id ?? 'anon'}>
      <Suspense fallback={<Skeleton className="h-64 w-full" />}>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route element={<RequireAuth />}>
            <Route element={<Layout />}>
              <Route path="/" element={<PortfolioPage />} />
              <Route path="/migracao-ir" element={<MigrationPage />} />
            </Route>
          </Route>
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Suspense>
    </PortfolioFilterProvider>
  );
}

export function App() {
  const queryClient = useMemo(createQueryClient, []);

  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AuthProvider queryClient={queryClient}>
          <AppRoutes />
        </AuthProvider>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;

