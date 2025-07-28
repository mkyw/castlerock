import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth/next';
import { authOptions } from '@/lib/auth';

// DELETE /api/domain-auth/domains/[id] - Delete a domain link
export async function DELETE(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  const session = await getServerSession(authOptions);
  
  if (!session?.user?.email) {
    return NextResponse.json(
      { error: 'Unauthorized' },
      { status: 401 }
    );
  }

  try {
    // Get the access token from the session
    const accessToken = (session as any).accessToken;
    if (!accessToken) {
      console.error('No access token found in session');
      return NextResponse.json({ error: 'No access token found' }, { status: 401 });
    }

    const response = await fetch(
      `${process.env.BACKEND_URL || 'http://localhost:8000'}/api/domain-auth/domains/${params.id}`,
      {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${accessToken}`,
          'Content-Type': 'application/json',
        },
      }
    );

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to delete domain');
    }

    return new Response(null, { status: 204 });
  } catch (error) {
    console.error('Error deleting domain:', error);
    return NextResponse.json(
      { error: 'Failed to delete domain' },
      { status: 500 }
    );
  }
}
