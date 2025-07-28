/** @type {import('next').NextConfig} */
const nextConfig = {
  env: {
    BACKEND_URL: process.env.BACKEND_URL || 'http://localhost:8000',
    NEXTAUTH_URL: process.env.NEXTAUTH_URL || 'http://localhost:3000',
    NEXTAUTH_SECRET: process.env.NEXTAUTH_SECRET,
  },
  // Enable server actions (moved to top-level in newer Next.js versions)
  experimental: {
    serverActions: {
      bodySizeLimit: '2mb',
    },
  },
  async rewrites() {
    return [
      // Proxy API requests to the backend
      {
        source: '/api/proxy/:path*',
        destination: `${process.env.BACKEND_URL || 'http://localhost:8000'}/api/:path*`,
      },
      // Proxy auth requests to the backend
      {
        source: '/auth/:path*',
        destination: `${process.env.BACKEND_URL || 'http://localhost:8000'}/auth/:path*`,
      },
    ];
  },
  async headers() {
    return [
      {
        // Apply these headers to all routes
        source: '/:path*',
        headers: [
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-XSS-Protection',
            value: '1; mode=block',
          },
          {
            key: 'Referrer-Policy',
            value: 'strict-origin-when-cross-origin',
          },
          {
            key: 'Permissions-Policy',
            value: 'camera=(), microphone=(), geolocation=()',
          },
        ],
      },
    ];
  },
  // Enable React strict mode
  reactStrictMode: true,
  // Configure images
  images: {
    domains: ['localhost'], // Add your image domains here
  },
};

// Make sure environment variables are set
if (!process.env.NEXTAUTH_SECRET) {
  console.warn('Warning: NEXTAUTH_SECRET is not set. Authentication may not work correctly.');
}

// For debugging environment variables in development
if (process.env.NODE_ENV === 'development') {
  console.log('Environment variables:', {
    NODE_ENV: process.env.NODE_ENV,
    NEXTAUTH_URL: process.env.NEXTAUTH_URL,
    BACKEND_URL: process.env.BACKEND_URL,
  });
}

module.exports = nextConfig;
