'use client';

import Navbar from '@/components/Navbar';
import Hero from '@/components/Hero';
import { useEffect } from 'react';

// This is the public landing page that doesn't require authentication
export default function Home() {
  // Ensure this page doesn't require authentication
  useEffect(() => {
    // This is a public page - no authentication check needed
    document.title = "Castlerock AI - AI-Powered Customer Support";
  }, []);

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      <main>
        <Hero />
      </main>
    </div>
  );
}
