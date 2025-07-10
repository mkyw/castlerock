import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

// Simplified middleware for debugging
export async function middleware(request: NextRequest) {
  console.log('Middleware running for:', request.nextUrl.pathname);
  return NextResponse.next();
}

export const config = {
  // Only run on specific paths for now
  matcher: ['/dashboard/:path*'],
};
