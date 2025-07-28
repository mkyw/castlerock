'use client';

import { useEffect, useRef } from 'react';
import { useSession } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import { forceLogout, requiresAuthentication } from '@/lib/auth-utils';

/**
 * SessionValidator component
 * 
 * This component detects invalid JWT tokens and other session errors
 * and redirects to the home page when necessary.
 * 
 * It should be added to the layout.tsx file to run on every page.
 */
export default function SessionValidator() {
    const { data: session, status } = useSession();
    const errorDetected = useRef(false);

    // Only run on protected pages and only check for session errors
    useEffect(() => {
        // Skip validation on public pages or if we've already detected an error
        if (!requiresAuthentication() || errorDetected.current) {
            return;
        }

        // Check for session errors
        if (session?.error) {
            errorDetected.current = true;
            console.log('Session error detected:', session.error);

            // Add a small delay before logout to prevent immediate redirect loops
            setTimeout(() => {
                forceLogout();
            }, 100);
        }
    }, [session]);

    return null; // This component doesn't render anything
} 