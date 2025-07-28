import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

// Public paths that don't require authentication
const PUBLIC_PATHS = ['/', '/login', '/signup', '/logout'];

// Function to handle CORS headers
function corsHeaders(origin: string) {
  return {
    'Access-Control-Allow-Origin': origin,
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization, Origin, X-Requested-With, Accept',
    'Access-Control-Allow-Credentials': 'true',
    'Access-Control-Max-Age': '86400' // 24 hours
  };
}

export function middleware(request: NextRequest) {
  // Skip authentication for public paths
  const path = request.nextUrl.pathname;
  if (PUBLIC_PATHS.includes(path)) {
    return NextResponse.next();
  }

  // Handle CORS preflight requests
  if (request.method === 'OPTIONS' && request.nextUrl.pathname.startsWith('/api/')) {
    const origin = request.headers.get('origin') || '*';
    return NextResponse.json({}, {
      status: 200,
      headers: corsHeaders(origin)
    });
  }

  // Add CORS headers to API responses
  if (request.nextUrl.pathname.startsWith('/api/')) {
    const origin = request.headers.get('origin') || '*';
    const response = NextResponse.next();

    // Add CORS headers to the response
    Object.entries(corsHeaders(origin)).forEach(([key, value]) => {
      response.headers.set(key, value);
    });

    return response;
  }

  // Continue for non-API routes
  return NextResponse.next();
}

// See "Matching Paths" below to learn more
export const config = {
  matcher: [
    // Match all API routes
    '/api/:path*',
    // Match all widget routes
    '/widget/:path*',
    // Match dashboard routes
    '/dashboard/:path*',
    // Exclude public paths
    '/((?!login|signup|logout|_next/static|_next/image|favicon.ico).*)',
  ],
};
