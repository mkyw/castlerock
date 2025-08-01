'use client';

import { useState, useEffect, useRef } from 'react';
import { useParams } from 'next/navigation';
import {
    Box,
    Typography,
    TextField,
    Button,
    Paper,
    CircularProgress,
    Alert,
    IconButton,
    Divider,
    Avatar
} from '@mui/material';
import { Send as SendIcon, Close as CloseIcon } from '@mui/icons-material';

export default function AgentChatPage() {
    const params = useParams();
    const indexName = params.indexName as string;
    const connectionId = params.connectionId as string;

    const [messages, setMessages] = useState<any[]>([]);
    const [newMessage, setNewMessage] = useState('');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [socket, setSocket] = useState<WebSocket | null>(null);
    const [connected, setConnected] = useState(false);
    const [reconnectAttempts, setReconnectAttempts] = useState(0);
    const maxReconnectAttempts = 5;

    const messagesEndRef = useRef<HTMLDivElement>(null);

    // Scroll to bottom of messages
    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    // Fetch initial chat history and establish WebSocket connection
    useEffect(() => {
        const fetchChatHistory = async () => {
            try {
                // Fetch chat history from the WebSocket server
                const response = await fetch(`http://localhost:8766/api/chat/history/${indexName}/${connectionId}`);

                if (!response.ok) {
                    throw new Error(`Error fetching chat history: ${response.statusText}`);
                }

                const data = await response.json();
                setMessages(data.messages || []);
                setLoading(false);
            } catch (err) {
                console.error('Error fetching chat history:', err);
                setError(`Failed to fetch chat history: ${err instanceof Error ? err.message : String(err)}`);
                setLoading(false);
            }
        };

        // Connect to WebSocket
        const connectWebSocket = () => {
            if (reconnectAttempts >= maxReconnectAttempts) {
                setError(`Failed to connect after ${maxReconnectAttempts} attempts. Please refresh the page.`);
                return null;
            }

            const ws = new WebSocket(`ws://localhost:8765/agent/${indexName}`);

            ws.onopen = () => {
                console.log('WebSocket connected');
                setConnected(true);
                setReconnectAttempts(0);
                setError(null);

                // Send agent join command
                const joinMessage = {
                    type: 'agent_command',
                    command: 'join',
                    agent_id: 'agent-1', // This would be the actual agent ID in production
                    connection_id: connectionId,
                    index_name: indexName,
                    timestamp: new Date().toISOString()
                };
                ws.send(JSON.stringify(joinMessage));

                // Add system message to show connection status
                setMessages(prev => [...prev, {
                    type: 'system',
                    content: 'Connected to chat as agent',
                    timestamp: new Date().toISOString()
                }]);
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    console.log('Received message:', data);

                    // Handle different message types
                    if (data.type === 'history') {
                        // If we receive a history message, replace our current messages
                        if (data.content && Array.isArray(data.content)) {
                            console.log('Received history with', data.content.length, 'messages');
                            // Only set messages if we actually have content
                            if (data.content.length > 0) {
                                setMessages(data.content);
                                console.log('Updated messages with history:', data.content);
                            }
                        } else {
                            console.error('History message has invalid content:', data.content);
                        }
                    } else if (data.connection_id === connectionId || !data.connection_id) {
                        // For regular messages, add to existing messages
                        console.log('Adding message to chat:', data);
                        setMessages((prevMessages) => {
                            // Avoid duplicates
                            const isDuplicate = prevMessages.some(
                                (msg) =>
                                    msg.timestamp === data.timestamp &&
                                    msg.content === data.content &&
                                    msg.type === data.type
                            );

                            if (isDuplicate) {
                                console.log('Duplicate message, not adding:', data);
                                return prevMessages;
                            }

                            const newMessages = [...prevMessages, data];
                            console.log('Updated messages:', newMessages);
                            return newMessages;
                        });
                    } else {
                        console.log('Ignoring message for different connection:', data.connection_id);
                    }

                    // Ensure scroll happens after state update
                    setTimeout(scrollToBottom, 100);
                } catch (err) {
                    console.error('Error parsing WebSocket message:', err);
                }
            };

            ws.onclose = (event) => {
                console.log(`WebSocket disconnected: ${event.code} ${event.reason}`);
                setConnected(false);

                // Try to reconnect after a delay
                setTimeout(() => {
                    if (socket === ws) {
                        setReconnectAttempts(prev => prev + 1);
                        connectWebSocket();
                    }
                }, 3000);
            };

            ws.onerror = (err) => {
                console.error('WebSocket error:', err);
                setError('WebSocket connection error');
            };

            setSocket(ws);

            return ws;
        };

        // Fetch history and connect to WebSocket
        fetchChatHistory();
        const ws = connectWebSocket();

        // Cleanup on unmount
        return () => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.close();
            }
        };
    }, [indexName, connectionId, reconnectAttempts]);

    // Scroll to bottom when messages change
    useEffect(() => {
        console.log('Messages changed, scrolling to bottom. Count:', messages.length);
        scrollToBottom();
    }, [messages]);

    // Handle sending a new message
    const handleSendMessage = () => {
        if (!newMessage.trim() || !socket || socket.readyState !== WebSocket.OPEN) return;

        const message = {
            type: 'agent_command',
            command: 'message',
            agent_id: 'agent-1', // This would be the actual agent ID in production
            connection_id: connectionId,
            index_name: indexName,
            content: newMessage,
            timestamp: new Date().toISOString()
        };

        socket.send(JSON.stringify(message));
        setNewMessage('');
    };

    // Handle ending the chat
    const handleEndChat = async () => {
        try {
            const response = await fetch(`http://localhost:8766/api/chat/end/${indexName}/${connectionId}`, {
                method: 'GET'
            });

            if (!response.ok) {
                throw new Error(`Error ending chat: ${response.statusText}`);
            }

            // Send end chat message via WebSocket
            if (socket && socket.readyState === WebSocket.OPEN) {
                const endMessage = {
                    type: 'agent_command',
                    command: 'end_chat',
                    agent_id: 'agent-1',
                    connection_id: connectionId,
                    index_name: indexName,
                    timestamp: new Date().toISOString()
                };
                socket.send(JSON.stringify(endMessage));
            }

            // Close the window or navigate back
            window.close();
        } catch (err) {
            console.error('Error ending chat:', err);
            setError(`Failed to end chat: ${err instanceof Error ? err.message : String(err)}`);
        }
    };

    // Get message style based on message type
    const getMessageStyle = (type: string) => {
        switch (type) {
            case 'user':
                return {
                    bgcolor: '#f0f0f0',
                    alignSelf: 'flex-start',
                    borderRadius: '10px 10px 10px 0'
                };
            case 'agent':
                return {
                    bgcolor: '#e3f2fd',
                    alignSelf: 'flex-end',
                    borderRadius: '10px 10px 0 10px'
                };
            case 'assistant':
                return {
                    bgcolor: '#e8f5e9',
                    alignSelf: 'flex-start',
                    borderRadius: '10px 10px 10px 0'
                };
            case 'system':
                return {
                    bgcolor: '#fff3e0',
                    alignSelf: 'center',
                    borderRadius: '10px'
                };
            default:
                return {
                    bgcolor: '#f0f0f0',
                    alignSelf: 'flex-start',
                    borderRadius: '10px'
                };
        }
    };

    // Get avatar for message type
    const getAvatar = (type: string) => {
        switch (type) {
            case 'user':
                return '👤';
            case 'agent':
                return '👨‍💼';
            case 'assistant':
                return '🤖';
            case 'system':
                return '🔔';
            default:
                return '❓';
        }
    };

    if (loading) {
        return (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
                <CircularProgress />
            </Box>
        );
    }

    return (
        <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column', p: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                    Chat: {indexName}/{connectionId.substring(0, 8)}...
                </Typography>
                <Box>
                    {!connected && (
                        <Alert severity="warning" sx={{ mr: 2, display: 'inline-flex' }}>
                            Disconnected {reconnectAttempts > 0 ? `(Reconnecting: ${reconnectAttempts}/${maxReconnectAttempts})` : ''}
                        </Alert>
                    )}
                    <Button
                        variant="outlined"
                        color="error"
                        startIcon={<CloseIcon />}
                        onClick={handleEndChat}
                    >
                        End Chat
                    </Button>
                </Box>
            </Box>

            <Divider sx={{ mb: 2 }} />

            {error && (
                <Alert severity="error" sx={{ mb: 2 }}>
                    {error}
                </Alert>
            )}

            <Paper
                elevation={3}
                sx={{
                    flexGrow: 1,
                    mb: 2,
                    p: 2,
                    overflowY: 'auto',
                    display: 'flex',
                    flexDirection: 'column'
                }}
            >
                {messages.map((message, index) => (
                    <Box
                        key={index}
                        sx={{
                            display: 'flex',
                            mb: 1,
                            ...getMessageStyle(message.type),
                            p: 1.5,
                            maxWidth: '80%'
                        }}
                    >
                        <Avatar sx={{ mr: 1, bgcolor: 'transparent' }}>{getAvatar(message.type)}</Avatar>
                        <Box>
                            <Typography variant="caption" color="text.secondary">
                                {message.type.toUpperCase()} • {new Date(message.timestamp).toLocaleTimeString()}
                            </Typography>
                            <Typography variant="body1">{message.content}</Typography>
                            {message.sources && message.sources.length > 0 && (
                                <Box sx={{ mt: 1 }}>
                                    <Typography variant="caption" color="text.secondary">
                                        Sources:
                                    </Typography>
                                    <ul style={{ margin: 0, paddingLeft: 16 }}>
                                        {message.sources.map((source: string, i: number) => (
                                            <li key={i}>
                                                <Typography variant="caption">{source}</Typography>
                                            </li>
                                        ))}
                                    </ul>
                                </Box>
                            )}
                        </Box>
                    </Box>
                ))}
                <div ref={messagesEndRef} />
            </Paper>

            <Box sx={{ display: 'flex', gap: 1 }}>
                <TextField
                    fullWidth
                    variant="outlined"
                    placeholder="Type your message..."
                    value={newMessage}
                    onChange={(e) => setNewMessage(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                    disabled={!connected}
                />
                <IconButton
                    color="primary"
                    onClick={handleSendMessage}
                    disabled={!connected || !newMessage.trim()}
                >
                    <SendIcon />
                </IconButton>
            </Box>
        </Box>
    );
} 