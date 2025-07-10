'use client';

import { useState, useRef } from 'react';
import { useSession } from 'next-auth/react';

// Type for our session with accessToken
type SessionWithToken = {
  accessToken?: string;
  user: {
    id: string;
    name?: string | null;
    email?: string | null;
    image?: string | null;
  };
} | null;

type SearchResult = {
  answer: string;
};

export default function RAGInterface() {
  const { data: session } = useSession() as { data: SessionWithToken };
  
  // Debug log the session
  console.log('RAGInterface - session:', session);
  if (session) {
    console.log('RAGInterface - accessToken exists:', 'accessToken' in session);
    console.log('RAGInterface - session keys:', Object.keys(session));
  }
  const [activeTab, setActiveTab] = useState<'website' | 'pdf'>('website');
  const [url, setUrl] = useState('');
  const [query, setQuery] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [searchResult, setSearchResult] = useState<string | null>(null);
  const [message, setMessage] = useState<{type: 'success' | 'error', text: string} | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const showMessage = (type: 'success' | 'error', text: string) => {
    setMessage({ type, text });
    setTimeout(() => setMessage(null), 5000);
  };

  const handleProcessWebsite = async () => {
    if (!url) {
      showMessage('error', 'Please enter a valid URL');
      return;
    }

    if (!session?.accessToken) {
      showMessage('error', 'No authentication token found');
      return;
    }

    setIsProcessing(true);
    try {
      const response = await fetch('/api/rag/process/website', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${session.accessToken}`,
        },
        body: JSON.stringify({ url }),
      });

      if (!response.ok) {
        throw new Error('Failed to process website');
      }

      const data = await response.json();
      showMessage('success', data.message || 'Website processed successfully!');
    } catch (error) {
      console.error('Error processing website:', error);
      showMessage('error', 'Failed to process website');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!session?.accessToken) {
      showMessage('error', 'No authentication token found');
      return;
    }

    setIsProcessing(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/rag/process/pdf', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${session.accessToken}`,
        },
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to process PDF');
      }

      const data = await response.json();
      showMessage('success', data.message || 'PDF processed successfully!');
    } catch (error) {
      console.error('Error processing PDF:', error);
      showMessage('error', 'Failed to process PDF');
    } finally {
      setIsProcessing(false);
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleSearch = async () => {
    if (!query.trim()) {
      showMessage('error', 'Please enter a search query');
      return;
    }

    if (!session?.accessToken) {
      showMessage('error', 'No authentication token found');
      console.error('No access token found in session:', session);
      return;
    }

    setIsSearching(true);
    try {
      console.log('Session access token:', session.accessToken);
      console.log('Session keys:', Object.keys(session));
      
      // Make sure the token doesn't already have 'Bearer ' prefix
      const token = session.accessToken.startsWith('Bearer ') 
        ? session.accessToken 
        : `Bearer ${session.accessToken}`;
        
      console.log('Sending search request with token:', token);
      console.log('Sending request to:', '/api/rag/query');
      
      const response = await fetch('/api/rag/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': token,
        },
        body: JSON.stringify({ query, k: 5 }),
      });

      const responseData = await response.json();
      
      if (!response.ok) {
        console.error('Search request failed:', {
          status: response.status,
          statusText: response.statusText,
          response: responseData
        });
        throw new Error(
          responseData.message || 
          `Search failed with status ${response.status}: ${response.statusText}`
        );
      }

      // Set the answer from the response
      setSearchResult(responseData.answer || 'No response from the server');
    } catch (error) {
      console.error('Error searching:', error);
      const errorMessage = error instanceof Error ? error.message : 'Search failed';
      showMessage('error', errorMessage);
      setSearchResult(null);
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-white shadow rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4">Add Knowledge Source</h2>
        
        {/* Tabs */}
        <div className="border-b border-gray-200 mb-6">
          <nav className="-mb-px flex space-x-8">
            <button
              onClick={() => setActiveTab('website')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'website'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Website
            </button>
            <button
              onClick={() => setActiveTab('pdf')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'pdf'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              PDF Document
            </button>
          </nav>
        </div>

        {/* Website Form */}
        {activeTab === 'website' && (
          <div className="space-y-4">
            <div>
              <label htmlFor="url" className="block text-sm font-medium text-gray-700 mb-1">
                Website URL
              </label>
              <div className="flex space-x-2">
                <input
                  type="url"
                  id="url"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="https://example.com"
                  className="flex-1 min-w-0 block w-full px-3 py-2 rounded-md border border-gray-300 shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                />
                <button
                  onClick={handleProcessWebsite}
                  disabled={isProcessing}
                  className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isProcessing ? 'Processing...' : 'Add Website'}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* PDF Form */}
        {activeTab === 'pdf' && (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Upload PDF Document
              </label>
              <div className="mt-1 flex items-center">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf"
                  onChange={handleFileUpload}
                  className="hidden"
                  id="pdf-upload"
                />
                <label
                  htmlFor="pdf-upload"
                  className="cursor-pointer bg-white py-2 px-3 border border-gray-300 rounded-md shadow-sm text-sm leading-4 font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                >
                  Choose File
                </label>
                <span className="ml-3 text-sm text-gray-500">
                  {fileInputRef.current?.files?.[0]?.name || 'No file chosen'}
                </span>
              </div>
              <p className="mt-1 text-xs text-gray-500">
                Upload a PDF document to add to the knowledge base
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Search Section */}
      <div className="bg-white shadow rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4">Search Knowledge Base</h2>
        <div className="flex space-x-2">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
            placeholder="Ask a question or search for information..."
            className="flex-1 min-w-0 block w-full px-3 py-2 rounded-md border border-gray-300 shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
          />
          <button
            onClick={handleSearch}
            disabled={isSearching}
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isSearching ? 'Searching...' : 'Search'}
          </button>
        </div>
      </div>

      {/* Results */}
      {searchResult && (
        <div className="bg-white shadow rounded-lg p-6">
          <div className="prose max-w-none">
            {searchResult.split('\n').map((paragraph, i) => (
              <p key={i} className="mb-4">{paragraph}</p>
            ))}
          </div>
        </div>
      )}

      {/* Message Alert */}
      {message && (
        <div
          className={`p-4 rounded-md ${
            message.type === 'success' ? 'bg-green-50' : 'bg-red-50'
          }`}
        >
          <div className="flex">
            <div className="flex-shrink-0">
              {message.type === 'success' ? (
                <svg
                  className="h-5 w-5 text-green-400"
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                >
                  <path
                    fillRule="evenodd"
                    d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                    clipRule="evenodd"
                  />
                </svg>
              ) : (
                <svg
                  className="h-5 w-5 text-red-400"
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                >
                  <path
                    fillRule="evenodd"
                    d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                    clipRule="evenodd"
                  />
                </svg>
              )}
            </div>
            <div className="ml-3">
              <p
                className={`text-sm font-medium ${
                  message.type === 'success' ? 'text-green-800' : 'text-red-800'
                }`}
              >
                {message.text}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
