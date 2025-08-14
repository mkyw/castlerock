"use client";

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useSession, signIn } from 'next-auth/react';
import { forceLogout, handleAuthError } from '@/lib/auth-utils';
import { fetchIndices, createIndex, deleteIndex } from '@/lib/api-client';
import { formatIndexName } from '@/lib/api-utils';
import {
  Button,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  DialogContentText,
  TextField,
  Box,
  CircularProgress,
  IconButton,
  ListItemSecondaryAction,
  Typography,
  Alert
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';

interface Index {
  name: string;
  dimension: number;
  metric: string;
  status: string;
  document_count?: number;
}

export default function IndexList() {
  const [indices, setIndices] = useState<Index[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [indexToDelete, setIndexToDelete] = useState<Index | null>(null);
  const [indexName, setIndexName] = useState('');
  const [creating, setCreating] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const router = useRouter();
  const { data: session, status } = useSession();

  const loadIndices = async (forceRefresh = false) => {
    try {
      setLoading(true);
      setError(null);

      const data = await fetchIndices(forceRefresh);
      setIndices(data);
    } catch (error) {
      console.error('Error fetching indices:', error);
      setError('Failed to load knowledge bases. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (status === 'authenticated') {
      loadIndices();
    } else if (status === 'unauthenticated') {
      router.push('/login');
    }
  }, [status, router]);

  const handleCreateIndex = async () => {
    if (!indexName.trim()) return;

    try {
      setCreating(true);
      setError(null);

      const data = await createIndex(indexName);

      if (data) {
        // Use the returned index name for navigation
        // Use the formatted (short) index name for the dashboard URL
        router.push(`/dashboard/${encodeURIComponent(formatIndexName(data.name))}`);
      }
    } catch (error) {
      console.error('Error creating index:', error);
      setError('Failed to create knowledge base. Please try again.');
    } finally {
      setCreating(false);
      setOpen(false);
      setIndexName('');
      loadIndices(true); // Force refresh after creating
    }
  };

  const handleDeleteClick = (index: Index, event: React.MouseEvent) => {
    event.stopPropagation();
    setIndexToDelete(index);
    setDeleteDialogOpen(true);
  };

  const handleDeleteConfirm = async () => {
    if (!indexToDelete) return;

    try {
      setDeleting(true);
      setError(null);

      await deleteIndex(indexToDelete.name);

      // Refresh the list without showing an alert
      await loadIndices(true); // Force refresh after deleting
      setDeleteDialogOpen(false);
      setIndexToDelete(null);
    } catch (error: any) {
      console.error('Error deleting index:', error);
      setError('Failed to delete knowledge base. Please try again.');

      // If the error is about the index not existing, just refresh the list
      if (error.response?.status === 404 || error.message?.includes('not found')) {
        await loadIndices(true);
      }
    } finally {
      setDeleting(false);
    }
  };

  const handleDeleteCancel = () => {
    setDeleteDialogOpen(false);
    setIndexToDelete(null);
  };

  const handleRelogin = () => {
    forceLogout();
  };

  if (status === 'loading' || loading) {
    return (
      <Box display="flex" justifyContent="center" p={2}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box p={2}>
        <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>
        {error.includes('Authentication') ? (
          <Button variant="contained" color="primary" onClick={handleRelogin}>
            Log in again
          </Button>
        ) : (
          <Button
            variant="outlined"
            onClick={(e) => {
              e.preventDefault();
              loadIndices(true);
            }}
          >
            Try Again
          </Button>
        )}
      </Box>
    );
  }

  return (
    <div>
      {indices.length === 0 ? (
        <Box p={2} textAlign="center">
          <Typography variant="body1" gutterBottom>
            You don't have any knowledge bases yet.
          </Typography>
          <Button
            variant="contained"
            color="primary"
            onClick={() => setOpen(true)}
            sx={{ mt: 2 }}
          >
            Create Your First Knowledge Base
          </Button>
        </Box>
      ) : (
        <>
          <List>
            {indices.map((index, i) => (
              <ListItem
                key={i}
                disablePadding
                secondaryAction={
                  <IconButton
                    edge="end"
                    aria-label="delete"
                    onClick={(e) => handleDeleteClick(index, e)}
                    disabled={deleting}
                  >
                    <DeleteIcon />
                  </IconButton>
                }
              >
                <ListItemButton
                  onClick={() => {
                    // Use the formatted (short) index name for the dashboard URL
                    router.push(`/dashboard/${encodeURIComponent(formatIndexName(index.name))}`);
                  }}
                >
                  <ListItemText
                    primary={formatIndexName(index.name)}
                    secondary={`384d cosine • ${index.document_count || 0} documents`}
                    secondaryTypographyProps={{ component: 'div' }}
                  />
                </ListItemButton>
              </ListItem>
            ))}
          </List>

          <Button
            variant="outlined"
            fullWidth
            onClick={() => setOpen(true)}
            sx={{ mt: 2 }}
          >
            Create New Knowledge Base
          </Button>
        </>
      )}

      <Dialog
        open={deleteDialogOpen}
        onClose={handleDeleteCancel}
        aria-labelledby="delete-dialog-title"
        aria-describedby="delete-dialog-description"
      >
        <DialogTitle id="delete-dialog-title">
          Delete Knowledge Base
        </DialogTitle>
        <DialogContent>
          <DialogContentText id="delete-dialog-description">
            Are you sure you want to delete {indexToDelete ? formatIndexName(indexToDelete.name) : 'this knowledge base'}?
            This action cannot be undone and will permanently delete all data.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleDeleteCancel} disabled={deleting}>
            Cancel
          </Button>
          <Button
            onClick={handleDeleteConfirm}
            color="error"
            disabled={deleting}
            startIcon={deleting ? <CircularProgress size={20} /> : null}
          >
            {deleting ? 'Deleting...' : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog open={open} onClose={() => setOpen(false)}>
        <DialogTitle>Create New Knowledge Base</DialogTitle>
        <DialogContent>
          <DialogContentText sx={{ mb: 2 }}>
            Choose a name for your knowledge base. Please note that knowledge bases cannot be renamed after creation.
          </DialogContentText>
          <TextField
            autoFocus
            margin="dense"
            label="Knowledge Base Name"
            fullWidth
            variant="outlined"
            value={indexName}
            onChange={(e) => setIndexName(e.target.value)}
            disabled={creating}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpen(false)} disabled={creating}>
            Cancel
          </Button>
          <Button
            onClick={handleCreateIndex}
            color="primary"
            disabled={!indexName.trim() || creating}
            startIcon={creating ? <CircularProgress size={20} /> : null}
          >
            {creating ? 'Creating...' : 'Create'}
          </Button>
        </DialogActions>
      </Dialog>
    </div>
  );
}