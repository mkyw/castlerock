'use client';

import { useEffect, useState } from 'react';

export default function TestPage() {
    const [message, setMessage] = useState('Loading...');

    useEffect(() => {
        setMessage('Test page loaded successfully!');
    }, []);

    return (
        <div className="min-h-screen flex items-center justify-center bg-gray-50">
            <div className="max-w-md w-full p-6 bg-white rounded-lg shadow-md">
                <h1 className="text-2xl font-bold text-center text-gray-900 mb-4">Test Page</h1>
                <p className="text-gray-700 text-center">{message}</p>
                <div className="mt-6 flex justify-center">
                    <a
                        href="/"
                        className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
                    >
                        Go Home
                    </a>
                </div>
            </div>
        </div>
    );
} 