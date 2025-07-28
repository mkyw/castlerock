import { hash } from "bcryptjs";
import { signOut } from "next-auth/react";

export async function hashPassword(password: string): Promise<string> {
  return await hash(password, 12);
}

export function validatePassword(password: string): { valid: boolean; message?: string } {
  if (password.length < 8) {
    return { valid: false, message: "Password must be at least 8 characters long" };
  }
  if (!/[A-Z]/.test(password)) {
    return { valid: false, message: "Password must contain at least one uppercase letter" };
  }
  if (!/[a-z]/.test(password)) {
    return { valid: false, message: "Password must contain at least one lowercase letter" };
  }
  if (!/[0-9]/.test(password)) {
    return { valid: false, message: "Password must contain at least one number" };
  }
  return { valid: true };
}

/**
 * Check if the current page requires authentication
 * @returns boolean indicating if the current page requires authentication
 */
export function requiresAuthentication(): boolean {
  if (typeof window === 'undefined') {
    return false; // During SSR, assume no auth required
  }

  const path = window.location.pathname;

  // List of paths that don't require authentication
  const publicPaths = ['/', '/login', '/signup', '/logout', '/demo'];

  // Check if the current path is in the list of public paths
  const isPublicPath = publicPaths.some(publicPath =>
    path === publicPath || path.startsWith(`${publicPath}/`)
  );

  // Also check for widget paths
  const isWidgetPath = path.startsWith('/widget/');

  return !isPublicPath && !isWidgetPath;
}

/**
 * Handle authentication errors consistently across the app
 * @param response The fetch response object
 * @param customErrorMessage Optional custom error message
 * @returns True if there was an auth error that was handled, false otherwise
 */
export async function handleAuthError(response: Response, customErrorMessage?: string): Promise<boolean> {
  if (response.status === 401) {
    console.error("Authentication error:", response.status, response.statusText);

    // If we're on a page that requires authentication, handle the error
    if (requiresAuthentication()) {
      // If there's a JSON response with more details, log it
      try {
        const errorData = await response.clone().json();
        console.error("Auth error details:", errorData);
      } catch (e) {
        // Ignore JSON parsing errors
      }

      // Force logout after a short delay to allow the user to see the error message
      setTimeout(() => {
        forceLogout();
      }, 2000);

      return true;
    } else {
      console.log("Auth error on public page - not redirecting");
      return true;
    }
  }
  return false;
}

/**
 * Force logout the user and clear session data
 * This is used when authentication errors occur
 */
export async function forceLogout() {
  // Static flag to prevent multiple simultaneous logout attempts
  if ((window as any).__isLoggingOut) {
    console.log("Logout already in progress, skipping duplicate request");
    return;
  }

  try {
    (window as any).__isLoggingOut = true;
    console.log("Force logout initiated");

    // Prevent infinite redirect loops
    // If we're already on the home page and this is a logout request, don't redirect again
    if (typeof window !== 'undefined') {
      // Get current path
      const currentPath = window.location.pathname;

      // Check if we're already on the home page
      if (currentPath === '/' || currentPath === '') {
        // Check if we've recently done a logout to prevent loops
        const lastLogout = localStorage.getItem('last_logout_timestamp');
        const now = Date.now();

        if (lastLogout && (now - parseInt(lastLogout)) < 3000) {
          console.log("Preventing logout loop - already on home page and recently logged out");
          return; // Exit early to prevent loop
        }

        // Set the timestamp for this logout
        localStorage.setItem('last_logout_timestamp', now.toString());
      }
    }

    // Try all methods to clear session state in parallel for speed
    const promises = [];

    // 1. Clear cookies via API
    promises.push(
      fetch('/api/auth/logout', {
        method: 'GET',
        cache: 'no-store',
        headers: {
          'Cache-Control': 'no-cache, no-store, must-revalidate',
          'Pragma': 'no-cache',
        }
      }).catch(error => {
        console.error('Error clearing session cookies via API:', error);
      })
    );

    // 2. Also try the clear-session endpoint
    promises.push(
      fetch('/api/auth/clear-session', {
        method: 'GET',
        cache: 'no-store',
        headers: {
          'Cache-Control': 'no-cache, no-store, must-revalidate',
          'Pragma': 'no-cache',
        }
      }).catch(error => {
        console.error('Error clearing session via clear-session endpoint:', error);
      })
    );

    // 3. Clear localStorage and sessionStorage
    try {
      if (typeof window !== 'undefined') {
        // Clear any auth-related items from storage
        localStorage.removeItem('next-auth.session-token');
        localStorage.removeItem('next-auth.csrf-token');
        localStorage.removeItem('next-auth.callback-url');
        sessionStorage.clear();
      }
    } catch (error) {
      console.error('Error clearing storage:', error);
    }

    // 4. Use NextAuth signOut
    promises.push(
      signOut({ redirect: false }).catch(error => {
        console.error('Error signing out via NextAuth:', error);
      })
    );

    // Wait for all promises to settle (not necessarily resolve)
    await Promise.allSettled(promises);

    // 5. Finally, redirect to home page (only if we're not already there)
    try {
      if (typeof window !== 'undefined' && window.location.pathname !== '/') {
        window.location.href = '/';
      }
    } catch (error) {
      console.error('Error redirecting to home page:', error);
    }
  } finally {
    // Reset the logout flag after a delay to allow for page transition
    setTimeout(() => {
      if (typeof window !== 'undefined') {
        (window as any).__isLoggingOut = false;
      }
    }, 5000);
  }
}
