import Link from 'next/link';

export default function Hero() {
  return (
    <div className="min-h-screen flex items-center justify-center px-4 sm:px-6 lg:px-8">
      <div className="max-w-3xl text-center">
        <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
          Transform Your Customer Service with AI
        </h1>
        <p className="text-xl text-gray-600 mb-10">
          Our AI-powered chatbot solution helps businesses scale their customer support 
          operations, providing instant, accurate responses to customer inquiries 24/7. 
          Reduce response times, increase customer satisfaction, and free up your team 
          to focus on what matters most.
        </p>
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Link 
            href="/signup" 
            className="px-8 py-3 text-base font-medium text-white bg-indigo-600 rounded-md hover:bg-indigo-700 transition-colors sm:px-10 sm:py-4 sm:text-lg"
          >
            Get Started for Free
          </Link>
          <Link 
            href="/demo" 
            className="px-8 py-3 text-base font-medium text-indigo-600 bg-white border border-indigo-600 rounded-md hover:bg-gray-50 transition-colors sm:px-10 sm:py-4 sm:text-lg"
          >
            See Demo
          </Link>
        </div>
      </div>
    </div>
  );
}
