'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { signOut } from 'next-auth/react';

export default function LogoutPage() {
    const router = useRouter();

    useEffect(() => {
        async function handleLogout() {
            try {
                // First try to call our API logout endpoint
                await fetch('/api/auth/logout', {
                    method: 'POST',
                    cache: 'no-store',
                    headers: {
                        'Cache-Control': 'no-cache, no-store, must-revalidate',
                        'Pragma': 'no-cache',
                    }
                });

                // Also try the clear-session endpoint
                await fetch('/api/auth/clear-session', {
                    method: 'POST',
                    cache: 'no-store',
                    headers: {
                        'Cache-Control': 'no-cache, no-store, must-revalidate',
                        'Pragma': 'no-cache',
                    }
                });
            } catch (e) {
                console.error('Error calling logout APIs:', e);
            }

            // Clear cookies manually as well
            document.cookie.split(';').forEach(cookie => {
                const [name] = cookie.trim().split('=');
                document.cookie = `${name}=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;`;
            });

            // Clear storage
            try {
                localStorage.clear();
                sessionStorage.clear();
            } catch (e) {
                console.error('Error clearing storage:', e);
            }

            // Sign out via NextAuth
            try {
                await signOut({ redirect: false });
            } catch (e) {
                console.error('Error signing out:', e);
            }

            // Redirect to home page instead of login
            setTimeout(() => {
                router.push('/');
            }, 1000);
        }

        handleLogout();
    }, [router]);

    return (
        <div className="min-h-screen flex items-center justify-center bg-gray-50">
            <div className="text-center p-8 bg-white rounded-lg shadow-md">
                <h1 className="text-2xl font-bold mb-4">Logging out...</h1>
                <div className="animate-spin rounded-full h-10 w-10 border-t-2 border-b-2 border-indigo-500 mx-auto mb-4"></div>
                <p className="text-gray-600">Clearing your session data...</p>
            </div>
        </div>
    );
} 