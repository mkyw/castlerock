import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export async function POST(req: NextRequest) {
  const session = await getServerSession(authOptions);

  if (!session || !session.user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  try {
    // Parse the request body
    const body = await req.json();
    const { query, k, index_name } = body;

    if (!query) {
      return NextResponse.json(
        { error: 'Query is required' },
        { status: 400 }
      );
    }

    if (!index_name) {
      return NextResponse.json(
        { error: 'index_name is required' },
        { status: 400 }
      );
    }

    // Get user email - backend uses email as the user ID in the JWT sub claim
    const userEmail = session.user.email;

    if (!userEmail) {
      console.error('No user email found in session');
      return NextResponse.json({ error: 'No user email found' }, { status: 401 });
    }

    // Get access token from session
    const accessToken = (session as any).accessToken;

    if (!accessToken) {
      console.error('No access token found in session');
      return NextResponse.json({ error: 'No access token found' }, { status: 401 });
    }

    // Forward the query to the backend
    const response = await fetch(`${BACKEND_URL}/api/rag/query`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${accessToken}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      console.error('Backend error:', response.status);
      return NextResponse.json({ error: 'Failed to process query' }, { status: response.status });
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error processing query:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
