import { getServerSession } from "next-auth/next";
import { NextRequest, NextResponse } from "next/server";
import { authOptions } from "@/lib/auth";
import { getToken } from 'next-auth/jwt';
import axios from './axios';

// Types
type ApiHandler = (
  req: NextRequest,
  params: { [key: string]: string | string[] | undefined }
) => Promise<NextResponse>;

/**
 * Higher-order function to protect API routes with authentication
 * For Next.js App Router
 */
export function withAuthAppRouter(
  handler: ApiHandler
): ApiHandler {
  return async (req, params) => {
    console.log('withAuthAppRouter: Starting authentication check');
    console.log('Request method:', req.method);
    console.log('Request URL:', req.url);

    try {
      const session = await getServerSession(authOptions);
      console.log('Session retrieved:', !!session);
      console.log('Session has accessToken:', !!session?.accessToken);

      if (!session?.accessToken) {
        console.log('No access token in session, returning 401');
        return NextResponse.json(
          { error: "Not authenticated" },
          { status: 401 }
        );
      }

      // Add the access token to the request headers
      const requestHeaders = new Headers(req.headers);
      requestHeaders.set("Authorization", `Bearer ${session.accessToken}`);
      console.log('Added Authorization header to request');

      // Create a new request with the updated headers
      const authReq = new NextRequest(req.url, {
        ...req,
        headers: requestHeaders
      });

      console.log('Calling handler with authenticated request');
      return handler(authReq, params);
    } catch (error) {
      console.error('API Error in withAuthAppRouter:', error);
      return NextResponse.json(
        { error: 'Internal server error' },
        { status: 500 }
      );
    }
  };
}

/**
 * Get the current user from the session
 * For server-side usage
 */
export async function getCurrentUser() {
  const session = await getServerSession(authOptions);
  return session?.user || null;
}

/**
 * Get the access token from the session
 * For client-side usage
 */
export async function getAccessToken(): Promise<string | null> {
  try {
    // For client-side, we can use getSession()
    const { data: session } = await axios.get('/api/auth/session');
    return session?.accessToken || null;
  } catch (error) {
    console.error('Error getting access token:', error);
    return null;
  }
}

/**
 * Helper to make authenticated API requests
 */
export async function authenticatedFetch<T = any>(
  url: string,
  options: RequestInit = {}
): Promise<T> {
  try {
    // Get the access token
    const token = await getAccessToken();

    if (!token) {
      throw new Error('No authentication token found');
    }

    // Set up headers
    const headers = new Headers(options.headers);
    headers.set('Authorization', `Bearer ${token}`);
    headers.set('Content-Type', 'application/json');

    // Make the request
    const response = await fetch(url, {
      ...options,
      headers,
      credentials: 'include', // Important for cookies if using them
    });

    // Handle non-2xx responses
    if (!response.ok) {
      let errorData;
      try {
        errorData = await response.json();
      } catch (e) {
        errorData = { message: 'API request failed' };
      }

      // If unauthorized, try to refresh the token
      if (response.status === 401) {
        try {
          // Try to refresh the token
          await axios.post('/api/auth/session', { update: true });
          // Retry the original request
          return authenticatedFetch<T>(url, options);
        } catch (refreshError) {
          console.error('Token refresh failed:', refreshError);
          // Redirect to login or handle the error as needed
          window.location.href = '/auth/signin';
          throw new Error('Session expired. Please log in again.');
        }
      }

      throw new Error(errorData.message || `API request failed: ${response.statusText}`);
    }

    // Handle successful response
    try {
      return await response.json();
    } catch (e) {
      // If response is not JSON, return as text
      return (await response.text()) as unknown as T;
    }
  } catch (error) {
    console.error('API request error:', error);
    throw error;
  }
}

/**
 * Format an index name for display by removing the prefix
 * 
 * @param fullIndexName The full index name (e.g., castlerock-a744863d-myindex)
 * @returns The formatted name (e.g., myindex)
 */
export function formatIndexName(fullIndexName: string): string {
  if (!fullIndexName) return '';

  // Check if the name follows the pattern castlerock-{hash}-{name}
  const parts = fullIndexName.split('-');
  if (parts.length >= 3 && parts[0] === 'castlerock') {
    // Return everything after the second dash
    return fullIndexName.substring(fullIndexName.indexOf('-', 11) + 1);
  }

  // If it doesn't match the pattern, return the original name
  return fullIndexName;
}
