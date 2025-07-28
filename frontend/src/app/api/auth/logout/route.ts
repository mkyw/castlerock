import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth/next';
import { authOptions } from '@/lib/auth';

export const dynamic = 'force-dynamic';

/**
 * API route to handle logout
 * This clears session data and cookies
 */
export async function GET(request: NextRequest) {
    const session = await getServerSession(authOptions);

    const response = NextResponse.json(
        {
            success: true,
            message: 'Logged out successfully',
            wasLoggedIn: !!session
        },
        { status: 200 }
    );

    // Clear all possible session cookies
    const cookiesToClear = [
        'next-auth.session-token',
        'next-auth.csrf-token',
        'next-auth.callback-url',
        '__Secure-next-auth.session-token',
        '__Secure-next-auth.csrf-token',
        '__Host-next-auth.csrf-token',
    ];

    cookiesToClear.forEach(cookieName => {
        response.cookies.set(cookieName, '', {
            expires: new Date(0),
            path: '/',
        });
    });

    return response;
}

export async function POST(request: NextRequest) {
    return GET(request);
} 