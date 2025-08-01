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
        // Try to forward the request to the backend
        try {
            const response = await fetch(`${BACKEND_URL}/api/chat/stats`, {
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
            console.warn('Backend chat stats API not available, using mock data');
        }

        // If backend request fails, return mock data
        return NextResponse.json({
            total: 0,
            ai_handling: 0,
            escalation_requested: 0,
            agent_assigned: 0,
            is_mock_data: true
        });

    } catch (error) {
        console.error('Chat stats error:', error);
        return NextResponse.json(
            { error: 'Internal server error' },
            { status: 500 }
        );
    }
} 