import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth';

export async function GET() {
  try {
    const session = await getServerSession(authOptions);
    
    if (!session?.accessToken) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      );
    }

    // This is where you would typically fetch data from your backend
    // using the session.accessToken for authentication
    
    return NextResponse.json({
      message: 'This is protected data',
      user: session.user,
      // Don't expose sensitive information in production
      tokenInfo: {
        hasToken: !!session.accessToken,
        tokenLength: session.accessToken?.length,
      }
    });
    
  } catch (error) {
    console.error('Protected route error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
