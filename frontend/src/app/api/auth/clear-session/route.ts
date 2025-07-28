import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

/**
 * API route to clear all session cookies
 * This is used when we need to force logout a user due to invalid sessions
 */
export async function GET(request: NextRequest) {
    const response = NextResponse.json(
        { success: true, message: 'Session cleared' },
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