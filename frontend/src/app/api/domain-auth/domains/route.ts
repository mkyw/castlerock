import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth/next';
import { authOptions } from '@/lib/auth';

export const dynamic = 'force-dynamic';

function getBaseUrl() {
  if (process.env.NODE_ENV === 'development') {
    return process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000';
  }
  // For production, use the environment variable or default to the current host
  return process.env.NEXT_PUBLIC_APP_URL || `https://${process.env.VERCEL_URL || process.env.HOST || 'localhost:3000'}`;
}

async function fetchWithAuthRetry(url: string, options: RequestInit = {}, retries = 1) {
  const session = await getServerSession(authOptions);
  
  if (!session || !(session as any)?.accessToken) {
    throw new Error('No valid session found');
  }

  const defaultOptions: RequestInit = {
    ...options,
    headers: {
      ...options.headers,
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${(session as any).accessToken}`,
    },
    credentials: 'include' as const,
  };

  let response = await fetch(url, defaultOptions);
  
  // If unauthorized, try to refresh the token once
  if (response.status === 401 && retries > 0) {
    console.log('Token expired, attempting to refresh...');
    
    const baseUrl = getBaseUrl();
    const sessionUrl = new URL('/api/auth/session', baseUrl).toString();
    
    try {
      // Try to refresh the session
      const refreshResponse = await fetch(sessionUrl, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Cookie': `next-auth.session-token=${(session as any).refreshToken}`,
        },
        body: JSON.stringify({ update: true }),
      });
      
      if (refreshResponse.ok) {
        // Retry the original request with the new token
        const newSession = await refreshResponse.json();
        if (newSession?.accessToken) {
          defaultOptions.headers = {
            ...defaultOptions.headers,
            'Authorization': `Bearer ${newSession.accessToken}`,
          };
          response = await fetch(url, defaultOptions);
        }
      } else {
        console.error('Failed to refresh session:', await refreshResponse.text());
      }
    } catch (refreshError) {
      console.error('Error refreshing session:', refreshError);
    }
  }
  
  return response;
}

// GET /api/domain-auth/domains - List all domains for the current user
export async function GET() {
  try {
    const response = await fetchWithAuthRetry(
      `${process.env.BACKEND_URL || 'http://localhost:8000'}/api/domain-auth/domains`,
      { method: 'GET' }
    );

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || 'Failed to fetch domains');
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error in GET /api/domain-auth/domains:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to fetch domains' },
      { status: error instanceof Error && error.message === 'No valid session found' ? 401 : 500 }
    );
  }
}

// POST /api/domain-auth/domains - Create a new domain link
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    const response = await fetchWithAuthRetry(
      `${process.env.BACKEND_URL || 'http://localhost:8000'}/api/domain-auth/domains`,
      {
        method: 'POST',
        body: JSON.stringify(body),
      }
    );

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || 'Failed to create domain link');
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error('Error in POST /api/domain-auth/domains:', error);
    return NextResponse.json(
      { 
        error: error instanceof Error ? error.message : 'Failed to create domain link',
        status: error instanceof Error && error.message === 'No valid session found' ? 401 : 500
      },
      { status: error instanceof Error && error.message === 'No valid session found' ? 401 : 500 }
    );
  }
}
