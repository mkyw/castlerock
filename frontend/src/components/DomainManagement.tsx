'use client';

import { useState, useEffect } from 'react';
import { useSession } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import { fetchDomains, addDomain, deleteDomain } from '@/lib/api-client';
import { Box, Typography, Button, TextField, Dialog, DialogTitle, DialogContent, DialogActions, List, ListItem, ListItemText, IconButton, Snackbar, Alert, Paper } from '@mui/material';
import { Add as AddIcon, Delete as DeleteIcon, ContentCopy as CopyIcon } from '@mui/icons-material';

interface DomainLink {
  id: string;
  domain: string;
  api_key: string;
  is_active: boolean;
  created_at: string;
  description?: string;
}

interface DomainManagementProps {
  indexName: string;
}

export default function DomainManagement({ indexName }: DomainManagementProps) {
  const { data: session, status } = useSession();
  const router = useRouter();
  const [domains, setDomains] = useState<DomainLink[]>([]);
  const [loading, setLoading] = useState(true);
  const [openDialog, setOpenDialog] = useState(false);
  const [newDomain, setNewDomain] = useState('');
  const [description, setDescription] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [copied, setCopied] = useState<string | null>(null);

  useEffect(() => {
    if (status === 'authenticated') {
      loadDomains();
    }
  }, [status, indexName]);

  const loadDomains = async () => {
    try {
      setLoading(true);
      const data = await fetchDomains(indexName);
      setDomains(data);
    } catch (err) {
      setError('Failed to load domains');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleAddDomain = async () => {
    if (!newDomain) {
      setError('Please enter a domain');
      return;
    }

    try {
      const data = await addDomain(indexName, newDomain, description || undefined);
      setDomains([...domains, data]);
      setNewDomain('');
      setDescription('');
      setOpenDialog(false);
      setSuccess('Domain added successfully');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to add domain');
    }
  };

  const handleDeleteDomain = async (id: string) => {
    if (!confirm('Are you sure you want to delete this domain?')) return;

    try {
      await deleteDomain(id);
      setDomains(domains.filter(domain => domain.id !== id));
      setSuccess('Domain deleted successfully');
    } catch (err) {
      setError('Failed to delete domain');
      console.error(err);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(text);
    setTimeout(() => setCopied(null), 2000);
  };

  if (status === 'loading' || loading) {
    return (
      <Box className="flex justify-center p-8">
        <Typography>Loading domains...</Typography>
      </Box>
    );
  }

  return (
    <Box className="space-y-4">
      <Box className="flex justify-between items-center">
        <Typography variant="h6">Domain Management</Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => setOpenDialog(true)}
          size="small"
        >
          Add Domain
        </Button>
      </Box>

      <Paper className="p-4 mb-4">
        <Typography variant="body2" className="mb-2">
          Add domains where you want to use this index. The widget will automatically authenticate requests from these domains.
        </Typography>
      </Paper>

      {domains.length === 0 ? (
        <Paper className="p-8 text-center">
          <Typography variant="body1" className="mb-2">No domains added yet</Typography>
          <Typography variant="body2" className="text-gray-500 mb-4">
            Add your first domain to enable domain-based authentication
          </Typography>
          <Button
            variant="outlined"
            startIcon={<AddIcon />}
            onClick={() => setOpenDialog(true)}
          >
            Add Domain
          </Button>
        </Paper>
      ) : (
        <Paper className="overflow-hidden">
          <List>
            {domains.map((domain) => (
              <ListItem
                key={domain.id}
                divider
                secondaryAction={
                  <IconButton
                    edge="end"
                    aria-label="delete"
                    onClick={() => handleDeleteDomain(domain.id)}
                    color="error"
                    size="small"
                  >
                    <DeleteIcon fontSize="small" />
                  </IconButton>
                }
              >
                <ListItemText
                  primary={domain.domain}
                  secondaryTypographyProps={{ component: 'div' }}
                  secondary={
                    <>
                      <Box className="flex items-center mt-1" component="span">
                        <span className="font-mono text-xs bg-gray-100 p-1 rounded mr-2">
                          {domain.api_key}
                        </span>
                        <IconButton
                          size="small"
                          onClick={() => copyToClipboard(domain.api_key)}
                          title="Copy API key"
                          className="text-gray-600 hover:text-gray-900"
                        >
                          <CopyIcon fontSize="small" />
                        </IconButton>
                        {copied === domain.api_key && (
                          <Typography variant="caption" color="success.main" className="ml-1" component="span">
                            Copied!
                          </Typography>
                        )}
                      </Box>
                      {domain.description && (
                        <Typography variant="caption" display="block" className="mt-1" component="span">
                          {domain.description}
                        </Typography>
                      )}
                    </>
                  }
                />
              </ListItem>
            ))}
          </List>
        </Paper>
      )}

      <Dialog open={openDialog} onClose={() => setOpenDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Add New Domain</DialogTitle>
        <DialogContent>
          <Box className="space-y-4 mt-2">
            <TextField
              fullWidth
              label="Domain (e.g., example.com)"
              value={newDomain}
              onChange={(e) => setNewDomain(e.target.value)}
              placeholder="example.com"
              helperText="Enter the domain where you'll host the chat widget"
              size="small"
            />
            <TextField
              fullWidth
              label="Description (optional)"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="e.g., Production Site"
              helperText="A friendly name to identify this domain"
              size="small"
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenDialog(false)}>Cancel</Button>
          <Button onClick={handleAddDomain} variant="contained">
            Add Domain
          </Button>
        </DialogActions>
      </Dialog>

      <Snackbar
        open={!!error}
        autoHideDuration={6000}
        onClose={() => setError('')}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <Alert onClose={() => setError('')} severity="error" sx={{ width: '100%' }}>
          {error}
        </Alert>
      </Snackbar>

      <Snackbar
        open={!!success}
        autoHideDuration={6000}
        onClose={() => setSuccess('')}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <Alert onClose={() => setSuccess('')} severity="success" sx={{ width: '100%' }}>
          {success}
        </Alert>
      </Snackbar>
    </Box>
  );
}
