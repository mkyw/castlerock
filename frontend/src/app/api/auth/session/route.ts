import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth/next';
import { authOptions } from '@/lib/auth';
import { getToken } from 'next-auth/jwt';

export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest) {
  try {
    const session = await getServerSession(authOptions);

    if (!session) {
      return NextResponse.json(
        { error: 'Not authenticated', code: 'SESSION_ERROR' },
        { status: 401 }
      );
    }

    // Check if the request includes updated tokens
    const body = await request.json().catch(() => ({}));

    if (body.accessToken) {
      // If we have new tokens, update the session
      (session as any).accessToken = body.accessToken;
      if (session.user) {
        (session.user as any).accessToken = body.accessToken;
      }
    }

    if (body.refreshToken) {
      (session as any).refreshToken = body.refreshToken;
      if (session.user) {
        (session.user as any).refreshToken = body.refreshToken;
      }
    }

    // Return the updated session
    return NextResponse.json({
      ...session,
      // Make sure we don't expose sensitive information
      user: {
        id: session.user?.id,
        name: session.user?.name,
        email: session.user?.email,
        image: session.user?.image,
        accessToken: (session.user as any)?.accessToken,
        refreshToken: (session.user as any)?.refreshToken,
      },
    });
  } catch (error) {
    console.error('Error in /api/auth/session:', error);

    // Check if this is a JWT error
    const errorMessage = error instanceof Error ? error.message : String(error);
    const isJwtError =
      errorMessage.includes('JWT') ||
      errorMessage.includes('JWE') ||
      errorMessage.includes('token') ||
      errorMessage.includes('Invalid Compact') ||
      errorMessage.includes('malformed');

    if (isJwtError) {
      // Create a response that will trigger a logout
      const response = NextResponse.json(
        {
          error: 'Invalid session token',
          code: 'INVALID_JWT',
          message: 'Your session has expired or is invalid. Please log in again.'
        },
        { status: 401 }
      );

      // Clear session cookies
      response.cookies.set('next-auth.session-token', '', {
        expires: new Date(0),
        path: '/'
      });
      response.cookies.set('__Secure-next-auth.session-token', '', {
        expires: new Date(0),
        path: '/',
        secure: true
      });

      return response;
    }

    return NextResponse.json(
      { error: 'Internal server error', code: 'SERVER_ERROR' },
      { status: 500 }
    );
  }
}

export { POST as GET };
