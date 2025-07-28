'use client';

import { useState, useRef, useEffect } from 'react';
import { useSession } from 'next-auth/react';
import { handleAuthError } from '@/lib/auth-utils';
import {
  Box,
  Typography,
  Tabs,
  Tab,
  TextField,
  Button,
  CircularProgress,
  Alert,
  Snackbar,
  Paper
} from '@mui/material';

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
  sources?: string[];
  model_used?: string;
};

interface RAGInterfaceProps {
  indexName: string;
}

export default function RAGInterface({ indexName }: RAGInterfaceProps) {
  const { data: session } = useSession() as { data: SessionWithToken };
  const [activeTab, setActiveTab] = useState<'website' | 'pdf' | 'query'>('query');
  const [url, setUrl] = useState('');
  const [query, setQuery] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [searchResult, setSearchResult] = useState<SearchResult | null>(null);
  const [message, setMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Ensure indexName is provided
  if (!indexName) {
    return (
      <Alert severity="error">
        Error: No index specified. Please ensure you're accessing this page through a valid domain link.
      </Alert>
    );
  }

  useEffect(() => {
    // Reset states when index changes
    setSearchResult(null);
    setQuery('');
    setUrl('');
  }, [indexName]);

  const showMessage = (type: 'success' | 'error', text: string) => {
    setMessage({ type, text });
    setTimeout(() => setMessage(null), 5000);
  };

  const handleProcessWebsite = async () => {
    if (!url.trim()) {
      showMessage('error', 'Please enter a valid URL');
      return;
    }

    // Basic URL validation
    try {
      new URL(url.trim());
    } catch (e) {
      showMessage('error', 'Please enter a valid URL with http:// or https://');
      return;
    }

    if (!session?.accessToken) {
      showMessage('error', 'No authentication token found');
      return;
    }

    console.log('Processing website with URL:', url.trim());
    console.log('Index name:', indexName);
    console.log('Session token available:', !!session.accessToken);

    const requestBody = JSON.stringify({
      url: url.trim(),
      index_name: indexName // Pass the index name to the backend
    });

    console.log('Request body:', requestBody);

    setIsProcessing(true);
    try {
      const response = await fetch('/api/rag/process/website', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${session.accessToken}`,
        },
        body: requestBody,
      });

      console.log('Response status:', response.status);
      console.log('Response ok:', response.ok);

      if (!response.ok) {
        // Handle authentication errors
        const isAuthError = await handleAuthError(response);
        if (isAuthError) {
          showMessage('error', 'Authentication expired. Please log in again.');
          return;
        }

        // Try to get detailed error information
        let errorMessage = 'Failed to process website';
        try {
          const errorText = await response.text();
          console.log('Error response text:', errorText);

          if (errorText) {
            try {
              const parsedError = JSON.parse(errorText);
              console.log('Parsed error:', parsedError);
              errorMessage = parsedError.error || parsedError.detail || errorMessage;
              if (parsedError.details) {
                console.error('Error details:', parsedError.details);
              }
            } catch (jsonError) {
              // If not JSON, use the raw text
              console.log('Error parsing JSON:', jsonError);
              errorMessage = errorText || errorMessage;
            }
          }
        } catch (parseError) {
          console.error('Error parsing error response:', parseError);
        }

        throw new Error(errorMessage);
      }

      // Safely parse the successful response
      let data;
      try {
        const responseText = await response.text();
        console.log('Success response text:', responseText);
        data = responseText ? JSON.parse(responseText) : {};
      } catch (parseError) {
        console.error('Error parsing successful response:', parseError);
        throw new Error('Error parsing response from server');
      }

      showMessage('success', data.message || 'Website processed and added to knowledge base!');
      setUrl('');
    } catch (error) {
      console.error('Error processing website:', error);
      showMessage('error', error instanceof Error ? error.message : 'Failed to process website');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleProcessDocument = async () => {
    if (!url.trim()) {
      showMessage('error', 'Please enter a valid document URL');
      return;
    }

    // Basic URL validation
    try {
      new URL(url.trim());
    } catch (e) {
      showMessage('error', 'Please enter a valid URL with http:// or https://');
      return;
    }

    // Check if the URL is for a supported document type
    const supportedExtensions = ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.ppt', '.pptx', '.xls', '.xlsx', '.jpg', '.jpeg', '.png', '.csv'];
    const isDocumentUrl = supportedExtensions.some(ext => url.toLowerCase().trim().endsWith(ext));

    if (!isDocumentUrl) {
      showMessage('error', 'URL must point to a supported document type: ' + supportedExtensions.join(', '));
      return;
    }

    if (!session?.accessToken) {
      showMessage('error', 'No authentication token found');
      return;
    }

    setIsProcessing(true);
    try {
      const response = await fetch('/api/rag/process/document', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${session.accessToken}`,
        },
        body: JSON.stringify({
          url: url.trim(),
          index_name: indexName
        }),
      });

      if (!response.ok) {
        // Handle authentication errors
        const isAuthError = await handleAuthError(response);
        if (isAuthError) {
          showMessage('error', 'Authentication expired. Please log in again.');
          return;
        }

        // Try to get detailed error information
        let errorMessage = 'Failed to process document';
        try {
          const errorText = await response.text();
          if (errorText) {
            try {
              const parsedError = JSON.parse(errorText);
              errorMessage = parsedError.error || parsedError.detail || errorMessage;
              if (parsedError.details) {
                console.error('Error details:', parsedError.details);
              }
            } catch (jsonError) {
              // If not JSON, use the raw text
              errorMessage = errorText || errorMessage;
            }
          }
        } catch (parseError) {
          console.error('Error parsing error response:', parseError);
        }

        throw new Error(errorMessage);
      }

      // Safely parse the successful response
      let data;
      try {
        const responseText = await response.text();
        data = responseText ? JSON.parse(responseText) : {};
      } catch (parseError) {
        console.error('Error parsing successful response:', parseError);
        throw new Error('Error parsing response from server');
      }

      showMessage('success', data.message || 'Document processed and added to knowledge base!');
      setUrl('');
    } catch (error) {
      console.error('Error processing document:', error);
      showMessage('error', error instanceof Error ? error.message : 'Failed to process document');
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
    formData.append('index_name', indexName);

    try {
      const response = await fetch('/api/rag/process/pdf', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${session.accessToken}`,
        },
        body: formData,
      });

      if (!response.ok) {
        // Handle authentication errors
        const isAuthError = await handleAuthError(response);
        if (isAuthError) {
          showMessage('error', 'Authentication expired. Please log in again.');
          return;
        }

        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Failed to process PDF');
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
        body: JSON.stringify({
          query,
          k: 5,
          index_name: indexName // Include the index name in the request
        }),
      });

      if (!response.ok) {
        // Handle authentication errors
        const isAuthError = await handleAuthError(response);
        if (isAuthError) {
          showMessage('error', 'Authentication expired. Please log in again.');
          return;
        }

        const responseData = await response.json();
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

      const responseData = await response.json();
      console.log('Search response data:', responseData);

      // Check if the response has the expected structure
      if (responseData.success) {
        setSearchResult({
          answer: responseData.answer || 'No answer provided',
          sources: responseData.sources || [],
          model_used: responseData.model_used
        });

        if (responseData.model_used) {
          console.log('Model used for response:', responseData.model_used);
        }
      } else {
        // Handle unsuccessful but non-error responses
        const errorMessage = responseData.answer || responseData.error || 'Search returned no results';
        showMessage('error', errorMessage);
        setSearchResult(null);
      }
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
    <Box sx={{ width: '100%' }}>
      <Tabs
        value={activeTab}
        onChange={(_, newValue) => setActiveTab(newValue)}
        sx={{ mb: 3 }}
      >
        <Tab label="Query Knowledge Base" value="query" />
        <Tab label="Add Website" value="website" />
        <Tab label="Upload PDF" value="pdf" />
      </Tabs>

      {activeTab === 'query' && (
        <Box>
          <Box display="flex" gap={2} mb={3}>
            <TextField
              fullWidth
              variant="outlined"
              placeholder="Ask a question about your documents..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
              disabled={isSearching}
              sx={{ flex: 1 }}
            />
            <Button
              variant="contained"
              onClick={handleSearch}
              disabled={isSearching || !query.trim()}
              sx={{ minWidth: 120 }}
            >
              {isSearching ? <CircularProgress size={24} /> : 'Search'}
            </Button>
          </Box>

          {searchResult && (
            <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
              <Typography variant="h6" gutterBottom>Answer:</Typography>
              <Typography component="div" paragraph style={{ whiteSpace: 'pre-wrap' }}>{searchResult.answer}</Typography>

              {searchResult.sources && searchResult.sources.length > 0 && (
                <Box mt={2}>
                  <Typography variant="subtitle2" color="textSecondary" gutterBottom>
                    Sources:
                  </Typography>
                  <Box component="ul" sx={{ pl: 2, mt: 1 }}>
                    {searchResult.sources.map((source, i) => (
                      <Typography component="li" key={i} variant="body2">
                        {source}
                      </Typography>
                    ))}
                  </Box>
                </Box>
              )}

              {searchResult.model_used && (
                <Box mt={2}>
                  <Typography variant="caption" color="textSecondary">
                    Generated by: {searchResult.model_used}
                  </Typography>
                </Box>
              )}
            </Paper>
          )}
        </Box>
      )}

      {activeTab === 'website' && (
        <Box>
          <Typography variant="h6" gutterBottom>Add Website to Knowledge Base</Typography>
          <Box className="flex flex-col md:flex-row gap-4 mb-6">
            <TextField
              fullWidth
              variant="outlined"
              label="Website URL"
              placeholder="https://example.com"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              disabled={isProcessing}
              className="flex-grow"
            />
            <Box className="flex gap-2">
              <Button
                variant="contained"
                color="primary"
                onClick={handleProcessWebsite}
                disabled={isProcessing || !url.trim()}
                startIcon={isProcessing ? <CircularProgress size={20} /> : null}
                className="whitespace-nowrap"
              >
                Process Website
              </Button>
              <Button
                variant="outlined"
                color="secondary"
                onClick={handleProcessDocument}
                disabled={isProcessing || !url.trim()}
                startIcon={isProcessing ? <CircularProgress size={20} /> : null}
                className="whitespace-nowrap"
                title="Process document URL directly (.pdf, .doc, etc.)"
              >
                Process Document
              </Button>
            </Box>
          </Box>
        </Box>
      )}

      {activeTab === 'pdf' && (
        <Box>
          <Typography variant="h6" gutterBottom>Upload PDF to Knowledge Base</Typography>
          <Box
            sx={{
              border: '2px dashed',
              borderColor: 'divider',
              borderRadius: 2,
              p: 4,
              textAlign: 'center',
              bgcolor: 'action.hover',
              '&:hover': {
                bgcolor: 'action.selected',
                cursor: 'pointer'
              },
              mb: 2
            }}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              accept=".pdf"
              style={{ display: 'none' }}
            />
            <Typography>Click to upload or drag and drop</Typography>
            <Typography variant="caption" color="textSecondary">PDF (MAX. 10MB)</Typography>
            {isProcessing && (
              <Box mt={2} display="flex" alignItems="center" justifyContent="center" gap={1}>
                <CircularProgress size={20} />
                <Typography>Processing PDF...</Typography>
              </Box>
            )}
          </Box>
        </Box>
      )}

      <Snackbar
        open={!!message}
        autoHideDuration={5000}
        onClose={() => setMessage(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert
          onClose={() => setMessage(null)}
          severity={message?.type || 'info'}
          sx={{ width: '100%' }}
        >
          {message?.text}
        </Alert>
      </Snackbar>
    </Box>
  );
}
