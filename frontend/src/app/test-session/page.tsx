'use client';

import { useEffect, useState } from 'react';
import { useSession } from 'next-auth/react';
import { useRouter } from 'next/navigation';

export default function TestSessionPage() {
  const { data: session, status, update } = useSession();
  const router = useRouter();
  const [tokenInfo, setTokenInfo] = useState<any>(null);

  useEffect(() => {
    console.log('Session updated:', { session, status });
    
    if (status === 'unauthenticated') {
      router.push('/auth/signin');
    } else if (session) {
      // Log the raw session data
      console.log('Raw session data:', session);
      
      // Try to get token info from session
      const sessionWithToken = session as any;
      setTokenInfo({
        accessToken: sessionWithToken.accessToken ? 'Token exists' : 'No access token',
        refreshToken: sessionWithToken.refreshToken ? 'Token exists' : 'No refresh token',
        error: sessionWithToken.error || 'No error',
        user: session.user,
      });
    }
  }, [session, status, router]);

  const refreshSession = async () => {
    try {
      console.log('Manually refreshing session...');
      const updatedSession = await update();
      console.log('Updated session:', updatedSession);
    } catch (error) {
      console.error('Error refreshing session:', error);
    }
  };

  if (status === 'loading') {
    return <div>Loading session...</div>;
  }

  if (!session) {
    return <div>Not authenticated. Redirecting to login...</div>;
  }

  return (
    <div className="container mx-auto p-6">
      <h1 className="text-2xl font-bold mb-6">Session Test Page</h1>
      
      <div className="bg-white p-6 rounded-lg shadow-md mb-6">
        <h2 className="text-xl font-semibold mb-4">Session Data</h2>
        <pre className="bg-gray-100 p-4 rounded overflow-auto">
          {JSON.stringify(session, null, 2)}
        </pre>
      </div>

      <div className="bg-white p-6 rounded-lg shadow-md mb-6">
        <h2 className="text-xl font-semibold mb-4">Token Information</h2>
        <pre className="bg-gray-100 p-4 rounded overflow-auto">
          {JSON.stringify(tokenInfo, null, 2)}
        </pre>
      </div>

      <div className="flex gap-4">
        <button
          onClick={refreshSession}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          Refresh Session
        </button>
        
        <button
          onClick={() => router.push('/api/auth/signout')}
          className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
        >
          Sign Out
        </button>
      </div>
    </div>
  );
}
