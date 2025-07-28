import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export async function GET(
  request: Request,
  context: { params: { indexName: string } }
) {
  const session = await getServerSession(authOptions);

  if (!session || !session.user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const { params } = context;
  const decodedIndexName = decodeURIComponent(params.indexName);

  try {
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

    // Fetch index details from the backend
    const response = await fetch(`${BACKEND_URL}/api/rag/indices/${decodedIndexName}`, {
      headers: {
        'Authorization': `Bearer ${accessToken}`,
        'Content-Type': 'application/json',
      },
      cache: 'no-store',
    });

    if (!response.ok) {
      console.error('Backend error:', response.status);
      return NextResponse.json({ error: 'Failed to fetch index details' }, { status: response.status });
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error fetching index details:', error);
    return NextResponse.json(
      { error: 'Failed to connect to the server' },
      { status: 500 }
    );
  }
}

export async function DELETE(
  request: Request,
  context: { params: { indexName: string } }
) {
  const session = await getServerSession(authOptions);

  if (!session || !session.user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const { params } = context;
  const decodedIndexName = decodeURIComponent(params.indexName);

  try {
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

    // Delete index through the backend
    const response = await fetch(`${BACKEND_URL}/api/rag/indices/${decodedIndexName}`, {
      method: 'DELETE',
      headers: {
        'Authorization': `Bearer ${accessToken}`,
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok && response.status !== 404) {
      console.error('Backend error:', response.status);
      return NextResponse.json({ error: 'Failed to delete index' }, { status: response.status });
    }

    return new Response(null, { status: 204 });
  } catch (error) {
    console.error('Error deleting index:', error);
    return NextResponse.json(
      { error: 'Failed to connect to the server' },
      { status: 500 }
    );
  }
}

export async function PATCH(
  request: Request,
  context: { params: { indexName: string } }
) {
  const session = await getServerSession(authOptions);

  if (!session || !session.user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const { params } = context;
  const decodedIndexName = decodeURIComponent(params.indexName);

  try {
    const body = await request.json();

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

    // Update index through the backend
    const response = await fetch(`${BACKEND_URL}/api/rag/indices/${decodedIndexName}`, {
      method: 'PATCH',
      headers: {
        'Authorization': `Bearer ${accessToken}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      console.error('Backend error:', response.status);
      return NextResponse.json({ error: 'Failed to update index' }, { status: response.status });
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error updating index:', error);
    return NextResponse.json(
      { error: 'Failed to update index' },
      { status: 500 }
    );
  }
}
