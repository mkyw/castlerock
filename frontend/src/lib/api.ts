import axios, { AxiosError, AxiosRequestConfig } from 'axios';
import { getSession, signOut } from 'next-auth/react';
import { forceLogout } from './auth-utils';

// Create axios instance with base URL
const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Function to handle session errors by redirecting to logout page
const handleSessionError = (error?: string) => {
  console.log('Detected invalid session, redirecting to logout page', error ? `Error: ${error}` : '');
  // Use direct navigation to bypass any session checks
  window.location.href = '/logout';
};

// Add a request interceptor to add the auth token to requests
api.interceptors.request.use(
  async (config) => {
    try {
      const session = await getSession();
      console.log('Session in request interceptor:', session ? 'Valid session' : 'No session');

      // Check if session has error or is expired
      if (session?.error || !session) {
        console.log('Invalid session detected:', session?.error || 'No session');
        handleSessionError(session?.error as string);
        throw new Error('Session is invalid or expired');
      }

      if (session?.accessToken) {
        console.log('Adding access token to request');
        config.headers.Authorization = `Bearer ${session.accessToken}`;

        // Add Origin header for domain validation
        if (typeof window !== 'undefined') {
          config.headers.Origin = window.location.origin;
          // Also add Referer for older servers
          config.headers.Referer = window.location.href;
        }
      } else {
        console.log('No access token in session');
        handleSessionError('missing_token');
        throw new Error('No access token available');
      }

      return config;
    } catch (error) {
      // Check if this is a JWT-related error
      const errorMessage = error instanceof Error ? error.message : String(error);
      if (
        errorMessage.includes('JWT') ||
        errorMessage.includes('JWE') ||
        errorMessage.includes('token')
      ) {
        console.error('JWT error in request interceptor:', errorMessage);
        handleSessionError('jwt_error');
      } else {
        console.error('Error in request interceptor:', error);
      }

      return Promise.reject(error);
    }
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add a response interceptor to handle token refresh
api.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const originalRequest = error.config as any;

    // Log all 401 errors for debugging
    if (error.response?.status === 401) {
      console.error('401 Unauthorized error:', error.response.data);
      console.error('Request URL:', originalRequest.url);
      console.error('Request headers:', originalRequest.headers);
    }

    // Log all 403 errors for debugging
    if (error.response?.status === 403) {
      console.error('403 Forbidden error:', error.response.data);
      console.error('Request URL:', originalRequest.url);
      console.error('Request headers:', originalRequest.headers);
    }

    // Check for specific error codes that indicate session problems
    if (error.response?.status === 401) {
      // Check if the response contains specific error codes
      const errorData = error.response.data as any;

      if (errorData?.code === 'INVALID_JWT' || errorData?.code === 'SESSION_ERROR') {
        console.log('Received session error from API:', errorData.code);
        handleSessionError(errorData.code);
        return Promise.reject(new Error('Session expired. Redirecting to login.'));
      }

      // If it's a regular 401 and we haven't tried to refresh yet
      if (!originalRequest._retry) {
        originalRequest._retry = true;

        try {
          // Try to refresh the token
          const session = await getSession();

          // Check if session has error
          if (session?.error) {
            console.log('Invalid session detected during refresh:', session.error);
            handleSessionError(session.error as string);
            return Promise.reject(new Error('Session is invalid'));
          }

          if (!session?.refreshToken) {
            console.log('No refresh token available, logging out');
            handleSessionError('no_refresh_token');
            return Promise.reject(new Error('Authentication expired. Please log in again.'));
          }

          console.log('Attempting to refresh token');
          const refreshResponse = await axios.post(
            `${process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'}/auth/refresh-token`,
            {
              refresh_token: session.refreshToken,
            }
          );

          const { access_token, refresh_token } = refreshResponse.data;
          console.log('Token refreshed successfully');

          // Update the session with the new tokens
          const sessionResponse = await fetch('/api/auth/session', {
            method: 'POST',
            body: JSON.stringify({
              accessToken: access_token,
              refreshToken: refresh_token || session.refreshToken,
            }),
            headers: {
              'Content-Type': 'application/json',
            },
          });

          // Check if the session update was successful
          if (!sessionResponse.ok) {
            const sessionError = await sessionResponse.json();
            console.error('Failed to update session:', sessionError);
            handleSessionError(sessionError.code || 'session_update_failed');
            return Promise.reject(new Error('Failed to update session'));
          }

          console.log('Session updated with new token');
          // Update the authorization header
          originalRequest.headers.Authorization = `Bearer ${access_token}`;

          // Ensure Origin header is present
          if (typeof window !== 'undefined') {
            originalRequest.headers.Origin = window.location.origin;
            originalRequest.headers.Referer = window.location.href;
          }

          // Retry the original request
          return api(originalRequest);
        } catch (refreshError) {
          console.error('Token refresh failed:', refreshError);
          handleSessionError('refresh_failed');
          return Promise.reject(new Error('Authentication expired. Please log in again.'));
        }
      }
    }

    return Promise.reject(error);
  }
);

export default api;
