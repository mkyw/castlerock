import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export async function POST(req: NextRequest) {
  try {
    const session = await getServerSession(authOptions);

    if (!session) {
      return NextResponse.json(
        { error: 'Not authenticated' },
        { status: 401 }
      );
    }

    const accessToken = (session as any).accessToken;

    if (!accessToken) {
      console.error('No access token found in session:', session);
      return NextResponse.json(
        { error: 'No authentication token found' },
        { status: 401 }
      );
    }

    const { query, k } = await req.json();

    if (!query) {
      return NextResponse.json(
        { error: 'Query is required' },
        { status: 400 }
      );
    }

    // Debug logging
    console.log('Original accessToken:', accessToken);

    // Make sure we don't add 'Bearer' prefix if it's already there
    const authHeader = accessToken.startsWith('Bearer ')
      ? accessToken
      : `Bearer ${accessToken}`;

    console.log('Sending auth header:', authHeader);

    const response = await fetch(`${BACKEND_URL}/api/rag/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': authHeader,
      },
      body: JSON.stringify({ query, k: k || 5 }),
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(
        { error: data.error || 'Failed to process query' },
        { status: response.status }
      );
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error('Error processing query:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
