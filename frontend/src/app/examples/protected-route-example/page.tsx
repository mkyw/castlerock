'use client';

import { useEffect, useState } from 'react';
import { useSession } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import ApiClient from '@/lib/api-client';

export default function ProtectedRouteExample() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (status === 'unauthenticated') {
      router.push('/auth/signin');
    }
  }, [status, router]);

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Example: Fetch protected data
      const result = await ApiClient.get('/api/protected-route');
      setData(result);
      
    } catch (err: any) {
      console.error('Error fetching data:', err);
      setError(err.message || 'Failed to fetch data');
    } finally {
      setLoading(false);
    }
  };

  if (status === 'loading') {
    return <div>Loading session...</div>;
  }

  if (!session) {
    return <div>Redirecting to login...</div>;
  }

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Protected Route Example</h1>
      
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-semibold mb-4">User Info</h2>
        <pre className="bg-gray-100 p-4 rounded overflow-auto">
          {JSON.stringify(session.user, null, 2)}
        </pre>
        
        <div className="mt-6">
          <button
            onClick={fetchData}
            disabled={loading}
            className={`px-4 py-2 rounded-md text-white ${
              loading ? 'bg-gray-400' : 'bg-blue-600 hover:bg-blue-700'
            }`}
          >
            {loading ? 'Loading...' : 'Fetch Protected Data'}
          </button>
          
          {error && (
            <div className="mt-4 p-3 bg-red-100 text-red-700 rounded-md">
              Error: {error}
            </div>
          )}
          
          {data && (
            <div className="mt-4">
              <h3 className="font-semibold mb-2">Protected Data:</h3>
              <pre className="bg-gray-100 p-4 rounded overflow-auto">
                {JSON.stringify(data, null, 2)}
              </pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
