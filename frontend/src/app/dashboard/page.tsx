'use client';

import { useSession } from 'next-auth/react';
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import IndexList from '@/components/IndexList';
import SessionValidator from '@/components/SessionValidator';

export default function Dashboard() {
  const { data: session, status } = useSession();
  const [isLoading, setIsLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    if (status === 'unauthenticated') {
      router.push('/login');
    } else if (status === 'authenticated') {
      setIsLoading(false);
    }
  }, [status, router]);

  if (status === 'loading' || isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <SessionValidator />
      <div className="max-w-4xl mx-auto">
        <div className="bg-white shadow overflow-hidden sm:rounded-lg">
          <div className="px-4 py-5 sm:px-6">
            <h1 className="text-2xl font-bold text-gray-900">
              Knowledge Bases
            </h1>
            <p className="mt-1 max-w-2xl text-sm text-gray-500">
              Manage your knowledge bases and create new ones
            </p>
          </div>

          <div className="border-t border-gray-200 px-4 py-5 sm:p-6">
            <IndexList />
          </div>
        </div>
      </div>
    </div>
  );
}
