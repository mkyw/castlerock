'use client';

import { useState, useEffect, useRef } from 'react';
import { SessionProvider } from 'next-auth/react';
import SessionValidator from '../SessionValidator';
import { Alert, Snackbar } from '@mui/material';
import { forceLogout, requiresAuthentication } from '@/lib/auth-utils';

export default function SessionProviderWrapper({
  children,
}: {
  children: React.ReactNode;
}) {
  const [error, setError] = useState<string | null>(null);
  const isPublicPage = typeof window !== 'undefined' && !requiresAuthentication();

  // Determine session refresh strategy based on page type
  const sessionStrategy = isPublicPage ? 'none' : 'jwt';

  return (
    <>
      <SessionProvider
        refetchOnWindowFocus={false}
        refetchInterval={0} // Disable automatic refetching
        refetchWhenOffline={false}
        // @ts-ignore - 'none' is not in the type definition but works to disable session checks
        strategy={sessionStrategy}
      >
        <SessionValidator />
        {children}
      </SessionProvider>

      {/* Display error message if authentication fails */}
      <Snackbar
        open={!!error}
        autoHideDuration={6000}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
        onClose={() => setError(null)}
      >
        <Alert severity="error" variant="filled" onClose={() => setError(null)}>
          {error}
        </Alert>
      </Snackbar>
    </>
  );
}
