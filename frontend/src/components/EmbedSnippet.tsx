'use client';

import { useState, useEffect } from 'react';
import { Box, Typography, Paper, IconButton, Button, Alert, CircularProgress } from '@mui/material';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import CheckIcon from '@mui/icons-material/Check';
import { useSession } from 'next-auth/react';
import type { Session } from 'next-auth';
import { generateEmbedCode } from '@/lib/widget-config';

interface EmbedSnippetProps {
  indexName: string;
}

export default function EmbedSnippet({ indexName }: EmbedSnippetProps) {
  const { data: session } = useSession() as { data: Session & { accessToken?: string } };
  const [copied, setCopied] = useState(false);
  const [isClient, setIsClient] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [widgetUrl, setWidgetUrl] = useState('');

  useEffect(() => {
    setIsClient(true);
    if (typeof window !== 'undefined') {
      // Use the public URL or fallback to localhost
      const baseUrl = process.env.NEXT_PUBLIC_FRONTEND_URL || window.location.origin;
      setWidgetUrl(`${baseUrl}/widget/chatbot-widget-new.js`);
    }
  }, []);

  if (loading) {
    return (
      <Box className="flex justify-center p-4">
        <CircularProgress size={24} />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="warning">
        {error}
        {error.includes('Authentication') && (
          <Button
            variant="contained"
            color="primary"
            size="small"
            sx={{ ml: 2 }}
            onClick={() => window.location.href = '/logout'}
          >
            Log in again
          </Button>
        )}
      </Alert>
    );
  }

  // Generate the embed code using the shared utility
  const snippet = isClient ? generateEmbedCode(widgetUrl, {
    ui: {
      title: indexName.split('-').pop() || 'Chat Assistant'
    }
  }) : '';

  const copyToClipboard = async () => {
    if (!isClient) return;

    try {
      await navigator.clipboard.writeText(snippet);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  return (
    <Box sx={{ mt: 4, maxWidth: '800px' }}>
      <Typography variant="h6" gutterBottom>
        Add Chatbot to Your Website
      </Typography>
      <Typography variant="body1" paragraph>
        Copy and paste this code into your website's HTML to add the chatbot widget.
        The widget will automatically connect to your "{indexName}" knowledge base.
      </Typography>

      <Paper
        variant="outlined"
        sx={{
          position: 'relative',
          backgroundColor: '#f8f9fa',
          borderRadius: 1,
          overflow: 'hidden',
          borderColor: 'divider',
          '&:hover': {
            borderColor: 'primary.main',
            boxShadow: '0 0 0 1px #4a90e2'
          }
        }}
      >
        <Box
          component="pre"
          sx={{
            m: 0,
            p: 3,
            overflowX: 'auto',
            fontFamily: '\'Roboto Mono\', monospace',
            fontSize: '0.8rem',
            lineHeight: 1.5,
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            color: 'text.primary',
            backgroundColor: 'transparent'
          }}
        >
          {snippet}
        </Box>
        <Box
          sx={{
            position: 'absolute',
            top: 8,
            right: 8,
            display: 'flex',
            gap: 1
          }}
        >
          {copied ? (
            <Box sx={{
              display: 'flex',
              alignItems: 'center',
              bgcolor: 'success.light',
              color: 'success.contrastText',
              px: 1.5,
              py: 0.5,
              borderRadius: 1,
              fontSize: '0.75rem',
              fontWeight: 500
            }}>
              <CheckIcon sx={{ fontSize: '1rem', mr: 0.5 }} />
              Copied!
            </Box>
          ) : (
            <IconButton
              size="small"
              onClick={copyToClipboard}
              sx={{
                bgcolor: 'background.paper',
                '&:hover': { bgcolor: 'action.hover' },
                boxShadow: 1
              }}
              title="Copy to clipboard"
            >
              <ContentCopyIcon fontSize="small" />
            </IconButton>
          )}
        </Box>
      </Paper>

      <Box sx={{ mt: 2, p: 2, bgcolor: 'info.50', borderRadius: 1 }}>
        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
          Quick Tips:
        </Typography>
        <ul style={{ margin: '0 0 0 20px', padding: 0, color: 'text.secondary' }}>
          <li>Place this code before the <code style={{ background: 'rgba(0,0,0,0.05)', padding: '2px 4px', borderRadius: 4 }}>&lt;/body&gt;</code> tag</li>
          <li>Customize the button text, color, and position by modifying the config object</li>
          <li>The widget will automatically connect to your knowledge base</li>
        </ul>
      </Box>
    </Box>
  );
}
