'use client';

import React, { Component, ErrorInfo, ReactNode } from 'react';
import { Alert, Box, Button, Typography } from '@mui/material';
import { forceLogout } from '@/lib/auth-utils';

interface Props {
    children: ReactNode;
}

interface State {
    hasError: boolean;
    error: Error | null;
}

/**
 * ErrorBoundary component to catch and handle React errors
 * This is particularly useful for catching authentication-related errors
 * that might occur during rendering.
 */
class ErrorBoundary extends Component<Props, State> {
    constructor(props: Props) {
        super(props);
        this.state = {
            hasError: false,
            error: null
        };
    }

    static getDerivedStateFromError(error: Error): State {
        // Update state so the next render will show the fallback UI
        return {
            hasError: true,
            error
        };
    }

    componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
        // Log the error to console
        console.error('ErrorBoundary caught an error:', error, errorInfo);

        // Check if this is an authentication-related error
        if (
            error.message.includes('next-auth') ||
            error.message.includes('authentication') ||
            error.message.includes('CLIENT_FETCH_ERROR') ||
            error.message.includes('JWT')
        ) {
            // Handle auth errors by logging out after a delay
            setTimeout(() => {
                forceLogout();
            }, 3000);
        }
    }

    handleRetry = (): void => {
        // Reset the error state
        this.setState({ hasError: false, error: null });
        // Reload the page to get a fresh state
        window.location.reload();
    };

    handleGoHome = (): void => {
        // Navigate to home page
        window.location.href = '/';
    };

    render(): ReactNode {
        if (this.state.hasError) {
            // Render fallback UI
            return (
                <Box
                    sx={{
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        minHeight: '50vh',
                        p: 3,
                        textAlign: 'center'
                    }}
                >
                    <Alert severity="error" sx={{ mb: 3, width: '100%', maxWidth: 600 }}>
                        Something went wrong
                    </Alert>

                    <Typography variant="h5" gutterBottom>
                        We apologize for the inconvenience
                    </Typography>

                    {this.state.error && (
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                            Error: {this.state.error.message}
                        </Typography>
                    )}

                    <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
                        <Button variant="contained" onClick={this.handleRetry}>
                            Try Again
                        </Button>
                        <Button variant="outlined" onClick={this.handleGoHome}>
                            Go to Home Page
                        </Button>
                    </Box>
                </Box>
            );
        }

        return this.props.children;
    }
}

export default ErrorBoundary; 