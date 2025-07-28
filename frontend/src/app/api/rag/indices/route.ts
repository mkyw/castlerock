import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export async function GET() {
  const session = await getServerSession(authOptions);

  if (!session || !session.user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  try {
    // Get user email - backend uses email as the user ID in the JWT sub claim
    const userEmail = session.user.email;

    if (!userEmail) {
      console.error('No user email found in session');
      return NextResponse.json({ error: 'No user email found' }, { status: 401 });
    }

    // Get access token from session (check both locations)
    const accessToken = (session as any).accessToken || session.user.accessToken;

    if (!accessToken) {
      console.error('No access token found in session');
      return NextResponse.json({ error: 'No access token found' }, { status: 401 });
    }

    console.log('Fetching indices from backend with token:', accessToken.substring(0, 10) + '...');

    // Fetch indices from the backend
    const response = await fetch(`${BACKEND_URL}/api/rag/indices`, {
      headers: {
        'Authorization': `Bearer ${accessToken}`,
        'Content-Type': 'application/json',
      },
      cache: 'no-store',
    });

    if (!response.ok) {
      console.error('Backend error:', response.status);
      return NextResponse.json({ error: 'Failed to fetch indices' }, { status: response.status });
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error fetching indices:', error);
    return NextResponse.json(
      { error: 'Failed to connect to the server' },
      { status: 500 }
    );
  }
}

export async function POST(request: Request) {
  const session = await getServerSession(authOptions);

  if (!session || !session.user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  try {
    const body = await request.json();

    // Get user email - backend uses email as the user ID in the JWT sub claim
    const userEmail = session.user.email;

    if (!userEmail) {
      console.error('No user email found in session');
      return NextResponse.json({ error: 'No user email found' }, { status: 401 });
    }

    // Get access token from session (check both locations)
    const accessToken = (session as any).accessToken || session.user.accessToken;

    if (!accessToken) {
      console.error('No access token found in session');
      return NextResponse.json({ error: 'No access token found' }, { status: 401 });
    }

    const response = await fetch(`${BACKEND_URL}/api/rag/indices`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${accessToken}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const error = await response.text();
      console.error('Backend error:', error);
      return NextResponse.json({ error: 'Failed to create index' }, { status: response.status });
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error creating index:', error);
    return NextResponse.json(
      { error: 'Failed to connect to the server' },
      { status: 500 }
    );
  }
}
