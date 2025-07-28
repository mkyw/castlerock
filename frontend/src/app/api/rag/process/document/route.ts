import { NextRequest, NextResponse } from 'next/server';
import { getToken } from 'next-auth/jwt';
import { withAuthAppRouter } from '@/lib/api-utils';
import { getServerSession } from 'next-auth/next';
import { authOptions } from '@/lib/auth';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

// Direct handler without withAuthAppRouter to avoid request body consumption issues
export async function POST(req: NextRequest) {
    console.log('API route: Processing document request received');
    console.log('Request method:', req.method);
    console.log('Request headers:', Object.fromEntries(req.headers.entries()));

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

        // Parse the request body - clone first to avoid consuming it
        const clonedReq = req.clone();
        const bodyText = await clonedReq.text();
        console.log('Request body text:', bodyText);

        let body;
        if (!bodyText) {
            console.error('Empty request body received');
            return NextResponse.json(
                { error: 'Empty request body' },
                { status: 400 }
            );
        }

        try {
            body = JSON.parse(bodyText);
            console.log('Parsed body:', body);
        } catch (jsonError) {
            console.error('JSON parse error:', jsonError);
            return NextResponse.json(
                { error: 'Invalid JSON in request body' },
                { status: 400 }
            );
        }

        const { url, index_name } = body;
        console.log('Extracted URL:', url);
        console.log('Extracted index_name:', index_name);

        if (!url) {
            console.error('URL is missing from request body');
            return NextResponse.json(
                { error: 'URL is required' },
                { status: 400 }
            );
        }

        // Validate URL format
        try {
            new URL(url);
        } catch (e) {
            console.error('Invalid URL format:', url);
            return NextResponse.json(
                { error: 'Invalid URL format. Must include http:// or https://' },
                { status: 400 }
            );
        }

        // Validate index_name is present
        if (!index_name) {
            console.error('index_name is missing from request body');
            return NextResponse.json(
                { error: 'index_name is required' },
                { status: 400 }
            );
        }

        // Check if the URL is for a supported document type
        const supportedExtensions = ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.ppt', '.pptx', '.xls', '.xlsx', '.jpg', '.jpeg', '.png', '.csv'];
        const isDocumentUrl = supportedExtensions.some(ext => url.toLowerCase().endsWith(ext));

        if (!isDocumentUrl) {
            console.error('URL is not for a supported document type:', url);
            return NextResponse.json(
                { error: `URL must point to a supported document type: ${supportedExtensions.join(', ')}` },
                { status: 400 }
            );
        }

        console.log('Sending request to backend:', `${BACKEND_URL}/api/rag/process/document`);
        const response = await fetch(`${BACKEND_URL}/api/rag/process/document`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${session.accessToken}`,
            },
            body: JSON.stringify({
                url,
                index_name
            }),
        });

        console.log('Backend response status:', response.status);

        // Check if the response is OK before trying to parse JSON
        if (!response.ok) {
            let errorData = { detail: `Error: ${response.status} ${response.statusText}` };

            try {
                // Try to parse error details if available
                const errorText = await response.text();
                console.log('Error response text from backend:', errorText);

                if (errorText) {
                    try {
                        errorData = JSON.parse(errorText);
                        console.log('Parsed error data:', errorData);
                    } catch (jsonError) {
                        // If not JSON, use text as detail
                        console.log('Error parsing backend error as JSON:', jsonError);
                        errorData = { detail: errorText };
                    }
                }
            } catch (parseError) {
                console.error('Error parsing error response:', parseError);
            }

            console.error('Backend error:', errorData);
            return NextResponse.json(
                { error: 'Error processing document', details: errorData },
                { status: response.status }
            );
        }

        // For successful responses, safely parse the JSON
        let data;
        try {
            const responseText = await response.text();
            console.log('Success response text from backend:', responseText);

            if (!responseText) {
                console.log('Empty response from backend, returning default success message');
                return NextResponse.json({ message: 'Document processing started' });
            }

            try {
                data = JSON.parse(responseText);
                console.log('Parsed success data:', data);
            } catch (jsonError) {
                console.error('Error parsing JSON response:', jsonError);
                return NextResponse.json(
                    { message: 'Document processing started, but response could not be parsed' }
                );
            }
        } catch (parseError) {
            console.error('Error parsing successful response:', parseError);
            return NextResponse.json(
                { error: 'Error parsing response from backend' },
                { status: 500 }
            );
        }

        return NextResponse.json(data);
    } catch (error) {
        console.error('Error processing document:', error);
        return NextResponse.json(
            { error: 'Internal server error', message: error instanceof Error ? error.message : 'Unknown error' },
            { status: 500 }
        );
    }
} 