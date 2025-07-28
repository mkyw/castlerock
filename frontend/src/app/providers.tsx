'use client';

import { ThemeProvider } from 'next-themes';
import { useEffect, useState } from 'react';
import ErrorBoundary from '@/components/providers/ErrorBoundary';

type ProvidersProps = {
  children: React.ReactNode;
};

export function Providers({ children }: ProvidersProps) {
  const [mounted, setMounted] = useState(false);

  // Prevent hydration mismatch by only rendering the theme provider on the client
  useEffect(() => {
    setMounted(true);

    // Global error handler for uncaught NextAuth errors
    const originalError = console.error;
    console.error = function (...args) {
      // Convert args to string for easier pattern matching
      const errorString = args.join(' ');

      // Suppress specific NextAuth errors that are expected and handled elsewhere
      if (
        errorString.includes('[next-auth][error]') ||
        errorString.includes('CLIENT_FETCH_ERROR') ||
        // Empty error details that come from our error handling
        (errorString.includes('Error details:') && errorString.includes('{}'))
      ) {
        // Just log a simplified version without triggering error reporting
        console.log('Suppressed error:', errorString.substring(0, 100) + (errorString.length > 100 ? '...' : ''));
      } else {
        // Pass through all other errors
        originalError.apply(console, args);
      }
    };

    return () => {
      console.error = originalError;
    };
  }, []);

  // Don't render theme-dependent content on the server
  if (!mounted) {
    return (
      <div style={{ visibility: 'hidden' }}>
        {children}
      </div>
    );
  }

  return (
    <ErrorBoundary>
      <ThemeProvider
        attribute="class"
        defaultTheme="light"
        enableSystem={false}
        disableTransitionOnChange
      >
        {children}
      </ThemeProvider>
    </ErrorBoundary>
  );
}
