'use client';

import { useState, useEffect } from 'react';
import { Box, Typography, TextField, Paper, IconButton, Tooltip } from '@mui/material';
import { ContentCopy as CopyIcon, Check as CheckIcon } from '@mui/icons-material';
import { generateEmbedCode, generateExampleCode, customizationOptions } from '@/lib/widget-config';

const EmbedCodeSnippet: React.FC = () => {
  const [copied, setCopied] = useState(false);
  const [isClient, setIsClient] = useState(false);
  const [widgetUrl, setWidgetUrl] = useState('');

  useEffect(() => {
    setIsClient(true);
    setWidgetUrl(`${window.location.origin}/widget/chatbot-widget-new.js`);
  }, []);

  if (!isClient) {
    return null; // Don't render anything during SSR
  }

  const embedCode = generateEmbedCode(widgetUrl);
  const exampleCode = generateExampleCode(widgetUrl);

  const copyToClipboard = () => {
    navigator.clipboard.writeText(embedCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <Paper elevation={2} sx={{ p: 3, mt: 4, backgroundColor: '#f8f9fa' }}>
      <Typography variant="h6" gutterBottom>
        Add Chatbot to Your Website
      </Typography>
      <Typography variant="body1" paragraph>
        Copy and paste this code into your website to add the CastleRock AI Chatbot:
      </Typography>

      <Box position="relative" sx={{ mb: 2 }}>
        <TextField
          fullWidth
          multiline
          rows={4}
          value={embedCode}
          variant="outlined"
          InputProps={{
            readOnly: true,
            style: {
              fontFamily: 'monospace',
              fontSize: '0.875rem',
              backgroundColor: '#fff',
            },
          }}
        />
        <Tooltip title={copied ? 'Copied!' : 'Copy to clipboard'}>
          <IconButton
            onClick={copyToClipboard}
            sx={{
              position: 'absolute',
              top: 8,
              right: 8,
              backgroundColor: 'rgba(255, 255, 255, 0.8)',
              '&:hover': {
                backgroundColor: 'rgba(255, 255, 255, 1)',
              },
            }}
          >
            {copied ? <CheckIcon color="success" /> : <CopyIcon />}
          </IconButton>
        </Tooltip>
      </Box>

      <Typography variant="body2" color="textSecondary">
        Place this code just before the closing &lt;/body&gt; tag of your website.
      </Typography>

      <Box mt={2}>
        <Typography variant="subtitle2" gutterBottom>
          Customization Options:
        </Typography>
        <Typography variant="body2" component="div" sx={{ pl: 2 }}>
          <ul style={{ margin: '0 0 0 16px', padding: 0 }}>
            {customizationOptions.map((option, index) => (
              <li key={index}><code>{option.name}</code>: {option.description}</li>
            ))}
          </ul>
        </Typography>
      </Box>

      <Box mt={3}>
        <Typography variant="subtitle2" gutterBottom>
          Example with custom options:
        </Typography>
        <Box
          component="pre"
          sx={{
            p: 2,
            bgcolor: '#282c34',
            color: '#abb2bf',
            borderRadius: 1,
            overflowX: 'auto',
            fontSize: '0.8rem',
          }}
        >
          {exampleCode}
        </Box>
      </Box>
    </Paper>
  );
};

export default EmbedCodeSnippet;
