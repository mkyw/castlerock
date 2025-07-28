import { NextRequest, NextResponse } from 'next/server';
import { getToken } from 'next-auth/jwt';
import { withAuthAppRouter } from '@/lib/api-utils';
import { getServerSession } from 'next-auth/next';
import { authOptions } from '@/lib/auth';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

// Direct handler without withAuthAppRouter to avoid request body consumption issues
export async function POST(req: NextRequest) {
  console.log('API route: Processing PDF request received');

  try {
    // Get session directly instead of using withAuthAppRouter
    const session = await getServerSession(authOptions);
    console.log('Session retrieved:', !!session);
    console.log('Session has accessToken:', !!session?.accessToken);

    if (!session?.accessToken) {
      console.error('No access token found in session');
      return NextResponse.json(
        { error: 'No authentication token found' },
        { status: 401 }
      );
    }

    const formData = await req.formData();
    const file = formData.get('file') as File | null;
    const indexName = formData.get('index_name') as string | null;

    console.log('Received form data with file:', !!file);
    console.log('Received index_name:', indexName);

    if (!file) {
      return NextResponse.json(
        { error: 'File is required' },
        { status: 400 }
      );
    }

    // Create a new FormData to send to the backend
    const form = new FormData();
    form.append('file', file);

    // Add the index_name if provided
    if (indexName) {
      form.append('index_name', indexName);
      console.log('Added index_name to form data:', indexName);
    }

    console.log('Sending request to backend:', `${BACKEND_URL}/api/rag/process/pdf`);
    const response = await fetch(`${BACKEND_URL}/api/rag/process/pdf`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${session.accessToken}`,
      },
      body: form,
    });

    console.log('Backend response status:', response.status);

    let data;
    try {
      const responseText = await response.text();
      console.log('Response text:', responseText ? responseText.substring(0, 100) + '...' : 'empty');
      data = responseText ? JSON.parse(responseText) : {};
    } catch (error) {
      console.error('Error parsing response:', error);
      return NextResponse.json(
        { error: 'Error parsing backend response' },
        { status: 500 }
      );
    }

    if (!response.ok) {
      console.error('Backend error:', data);
      return NextResponse.json(
        { error: 'Error processing PDF', details: data },
        { status: response.status }
      );
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error('Error processing PDF:', error);
    return NextResponse.json(
      { error: 'Internal server error', message: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
}
