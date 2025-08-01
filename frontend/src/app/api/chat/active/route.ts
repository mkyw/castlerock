import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export async function GET(request: Request) {
    const session = await getServerSession(authOptions);

    if (!session?.accessToken) {
        return NextResponse.json(
            { error: 'Unauthorized' },
            { status: 401 }
        );
    }

    try {
        // Get the status query parameter
        const { searchParams } = new URL(request.url);
        const status = searchParams.get('status');

        // Try to forward the request to the backend
        try {
            const url = new URL(`${BACKEND_URL}/api/chat/active`);
            if (status) {
                url.searchParams.append('status', status);
            }

            const response = await fetch(url.toString(), {
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${session.accessToken}`,
                },
                // Short timeout to quickly fall back to mock data if backend is unavailable
                signal: AbortSignal.timeout(2000)
            });

            if (response.ok) {
                const data = await response.json();
                return NextResponse.json(data);
            }
        } catch (error) {
            console.warn('Backend active chats API not available, using mock data');
        }

        // If backend request fails, return empty data
        return NextResponse.json({
            is_mock_data: true
        });

    } catch (error) {
        console.error('Active chats error:', error);
        return NextResponse.json(
            { error: 'Internal server error' },
            { status: 500 }
        );
    }
} 