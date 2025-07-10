'use client';

import { SessionProvider } from 'next-auth/react';

export default function SessionProviderWrapper({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <SessionProvider 
      refetchOnWindowFocus={true}
      refetchInterval={60 * 5}
    >
      {children}
    </SessionProvider>
  );
}
