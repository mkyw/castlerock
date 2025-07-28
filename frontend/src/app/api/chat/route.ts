import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

// Helper function to handle CORS
function corsHeaders(origin: string) {
  return {
    'Access-Control-Allow-Origin': origin,
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization, Origin, X-Requested-With, Accept',
    'Access-Control-Allow-Credentials': 'true',
    'Access-Control-Max-Age': '86400' // 24 hours
  };
}

// Handle OPTIONS requests for CORS preflight
export async function OPTIONS(request: Request) {
  const origin = request.headers.get('origin') || '*';

  return new NextResponse(null, {
    status: 200,
    headers: corsHeaders(origin)
  });
}

export async function POST(request: Request) {
  const session = await getServerSession(authOptions);
  const origin = request.headers.get('origin') || '*';

  try {
    // Get the request body
    const { message, index_name } = await request.json();

    // Get the headers from the request
    const requestOrigin = request.headers.get('origin') || 'http://localhost:3000';
    const referer = request.headers.get('referer') || 'http://localhost:3000';

    // Prepare headers for the backend request
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      'Origin': requestOrigin,
      'Referer': referer
    };

    // Add authorization header if session exists
    if (session?.accessToken) {
      headers['Authorization'] = `Bearer ${session.accessToken}`;
    }

    // Forward the message to your backend
    const response = await fetch(`${BACKEND_URL}/api/rag/query`, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        query: message,
        index_name: index_name || 'default', // Use default if not provided
        k: 5, // Include the number of results to return
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      console.error('Backend error:', error);
      return NextResponse.json(
        { error: 'Failed to process message' },
        {
          status: response.status,
          headers: corsHeaders(origin)
        }
      );
    }

    const data = await response.json();
    return NextResponse.json(
      { response: data.answer || data.response || "I'm sorry, I couldn't process your request." },
      { headers: corsHeaders(origin) }
    );

  } catch (error) {
    console.error('Chat error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      {
        status: 500,
        headers: corsHeaders(origin)
      }
    );
  }
}
