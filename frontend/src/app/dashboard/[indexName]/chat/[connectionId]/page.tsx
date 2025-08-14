'use client';

import { useState, useEffect, useRef } from 'react';
import { useParams } from 'next/navigation';
import { getChatServiceUrl, getChatWebSocketUrl } from '@/lib/api-config';
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
    // Typing indicator - only show during RAG/LLM processing
    const [isUserTyping, setIsUserTyping] = useState(false);
    const [isRagLlmProcessing, setIsRagLlmProcessing] = useState(false);
    const [isEscalated, setIsEscalated] = useState(false);
    // Track if an agent has taken over the chat
    const [isAgentTakeover, setIsAgentTakeover] = useState(false);
    const maxReconnectAttempts = 5;

    // Ref to track typing indicator timeout
    const typingTimeoutRef = useRef<NodeJS.Timeout | null>(null);

    // Functions to handle typing indicator - only for RAG/LLM processing
    const showTypingIndicator = () => {
        // Only show typing indicator during RAG/LLM processing
        // and never during agent takeover or escalation
        if (!isEscalated && !isAgentTakeover && isRagLlmProcessing) {
            setIsUserTyping(true);
            
            // Clear any existing timeout
            if (typingTimeoutRef.current) {
                clearTimeout(typingTimeoutRef.current);
            }
            
            // Set a timeout to clear the typing indicator after 30 seconds
            // This is a safety measure in case the backend doesn't respond
            typingTimeoutRef.current = setTimeout(() => {
                clearTypingIndicator();
            }, 30000);
        }
    };
    
    const clearTypingIndicator = () => {
        setIsUserTyping(false);
        if (typingTimeoutRef.current) {
            clearTimeout(typingTimeoutRef.current);
            typingTimeoutRef.current = null;
        }
    };

    const messagesEndRef = useRef<HTMLDivElement>(null);
    const unmountingRef = useRef(false);
    const hasInitializedRef = useRef(false);

    // Scroll to bottom of messages
    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    // Fetch initial chat history and establish WebSocket connection
    useEffect(() => {

        const fetchChatHistory = async () => {
            try {
                // Fetch chat history from the WebSocket server
                const response = await fetch(`${getChatServiceUrl()}/api/chat/history/${indexName}/${connectionId}`);

                if (!response.ok) {
                    console.warn(`Chat history not found, will initialize from WebSocket messages`);
                    // Don't throw an error, just continue with empty messages
                    setLoading(false);
                    return;
                }

                const data = await response.json();
                setMessages(data.messages || []);
                setLoading(false);
            } catch (err) {
                console.error('Error fetching chat history:', err);
                // Don't set error, just continue with empty messages
                setLoading(false);
            }
        };

        // Connect to WebSocket - only create a new connection if one doesn't exist
        const connectWebSocket = () => {
            // If we already have a socket that's open or connecting, don't create a new one
            if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
                console.log('WebSocket already connected or connecting, not creating a new one');
                return socket;
            }

            // Check if we've already initialized in this effect run
            if (hasInitializedRef.current) {
                console.log('Already initialized WebSocket in this effect run, skipping');
                return socket;
            }

            // Mark as initialized
            hasInitializedRef.current = true;

            console.log('Creating new WebSocket connection');
            const ws = new WebSocket(`${getChatWebSocketUrl()}/ws/chat/${indexName}`);

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
                    role: 'system',
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
                            }
                        }
                    } else if (data.type === 'typing_indicator') {
                        // Only show typing indicator during RAG/LLM processing
                        console.log('Typing indicator message received');
                        // We don't directly show typing indicator here
                        // It will be shown when RAG/LLM processing is detected
                    } else {
                        // Check message type/role - system messages can come with type='system' or role='system'
                        // Log the raw message data to help debug
                        console.log('Message data for type detection:', { type: data.type, role: data.role });

                        // Handle system messages
                        if (data.type === 'system' || data.role === 'system') {
                            console.log('System message detected');
                            // Always clear typing indicator for system messages
                            clearTypingIndicator();
                            setIsRagLlmProcessing(false);

                            // Check for escalation messages
                            const content = (data.content || '').toLowerCase();
                            if (content.includes('connect you with a human agent') ||
                                content.includes('agent') ||
                                content.includes('escalat')) {
                                console.log('Escalation message detected, setting escalated state');
                                setIsEscalated(true);
                                setIsAgentTakeover(true);
                                // Ensure typing indicator is cleared
                                clearTypingIndicator();
                            }
                        }
                        // Handle AI responses and agent messages
                        else if (data.type === 'assistant' || data.role === 'assistant' ||
                            data.type === 'agent' || data.role === 'agent') {
                            console.log('AI or agent message detected');
                            
                            // End RAG/LLM processing when we get an AI response
                            setIsRagLlmProcessing(false);
                            clearTypingIndicator();

                            // If this is an agent message, set the agent takeover flag
                            if (data.type === 'agent' || data.role === 'agent') {
                                console.log('Agent message detected, setting agent takeover flag');
                                setIsAgentTakeover(true);
                            }
                        }
                        // Handle user messages - start RAG/LLM processing
                        else if (data.type === 'user' || data.role === 'user') {
                            // When a user message is received, start RAG/LLM processing
                            // This indicates the backend is processing the user's query
                            if (!isEscalated && !isAgentTakeover) {
                                console.log('User message detected, starting RAG/LLM processing');
                                setIsRagLlmProcessing(true);
                                showTypingIndicator();
                            }
                        }

                        // For other message types, add to our messages
                        // Check if this message is already in our state to prevent duplicates
                        setMessages(prev => {
                            // More robust duplicate detection:
                            // 1. Exact match (content, timestamp, role/type)
                            // 2. Content match with same role/type (handles different timestamps)
                            // 3. For all message types, be careful about duplication
                            const isDuplicate = prev.some(msg => {
                                // Get role from either role or type property
                                const msgRole = msg.role || msg.type;
                                const dataRole = data.role || data.type;

                                // Exact match check
                                if (msg.content === data.content &&
                                    msg.timestamp === data.timestamp &&
                                    msgRole === dataRole) {
                                    return true;
                                }

                                // Content match with same role check (handles different timestamps)
                                if (msgRole === dataRole && msg.content === data.content) {
                                    // Parse timestamps and check if they're within 1 second of each other
                                    try {
                                        const msgTime = new Date(msg.timestamp).getTime();
                                        const dataTime = new Date(data.timestamp).getTime();
                                        const timeDiff = Math.abs(msgTime - dataTime);

                                        // If timestamps are within 1 second, consider it a duplicate
                                        if (timeDiff < 1000) {
                                            console.log('Detected near-duplicate message with timestamp difference of', timeDiff, 'ms');
                                            return true;
                                        }
                                    } catch (e) {
                                        console.error('Error parsing timestamps for duplicate detection:', e);
                                    }
                                }

                                return false;
                            });

                            // Log duplicate detection
                            if (isDuplicate) {
                                console.log('Duplicate message detected and filtered out:', data);
                            }

                            // Only add the message if it's not a duplicate
                            return isDuplicate ? prev : [...prev, data];
                        });
                    }

                    // Ensure scroll happens after state update
                    setTimeout(scrollToBottom, 100);
                } catch (err) {
                    console.error('Error parsing WebSocket message:', err);
                    // Try to display as plain text if JSON parsing fails
                    setMessages(prev => [...prev, {
                        type: 'system',
                        content: `Received: ${event.data}`,
                        timestamp: new Date().toISOString()
                    }]);
                }
            };

            ws.onclose = (event) => {
                console.log(`WebSocket closed: ${event.code} ${event.reason}`);
                setConnected(false);

                // Only attempt to reconnect if we're not unmounting
                if (!unmountingRef.current) {
                    // Attempt to reconnect without creating a new effect
                    setTimeout(() => {
                        const newAttempts = reconnectAttempts + 1;
                        setReconnectAttempts(newAttempts);

                        if (newAttempts < maxReconnectAttempts) {
                            console.log(`Reconnecting attempt ${newAttempts}/${maxReconnectAttempts}`);
                            connectWebSocket(); // Directly reconnect without changing dependencies
                        } else {
                            setError(`Failed to connect after ${maxReconnectAttempts} attempts. Please refresh the page.`);
                        }
                    }, 1000);
                }
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
            console.log('Component unmounting, cleaning up WebSocket and timeouts');
            unmountingRef.current = true;

            // Typing indicator functionality removed

            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.close();
            }
        };
    }, [indexName, connectionId]); // Remove reconnectAttempts from dependencies

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
            role: 'agent',  // Add role property for consistency
            command: 'message',
            agent_id: 'agent-1', // This would be the actual agent ID in production
            connection_id: connectionId,
            index_name: indexName,
            content: newMessage,
            timestamp: new Date().toISOString()
        };

        // Start RAG/LLM processing when user sends a message
        // Only if not in escalated or agent takeover mode
        if (!isEscalated && !isAgentTakeover) {
            console.log('User sent message, starting RAG/LLM processing');
            setIsRagLlmProcessing(true);
            showTypingIndicator();
        }

        socket.send(JSON.stringify(message));
        setNewMessage('');
    };

    // Handle ending the chat
    const handleEndChat = async () => {
        try {
            // First try to send the end chat message via WebSocket
            // This ensures the backend knows about the agent's intent to end the chat
            if (socket && socket.readyState === WebSocket.OPEN) {
                const endMessage = {
                    type: 'agent_command',
                    role: 'agent',  // Add role for consistency
                    command: 'end_chat',
                    agent_id: 'agent-1',
                    connection_id: connectionId,
                    index_name: indexName,
                    timestamp: new Date().toISOString()
                };
                socket.send(JSON.stringify(endMessage));

                // Give the WebSocket message time to be processed
                await new Promise(resolve => setTimeout(resolve, 500));
            }

            // Then try the API call
            try {
                const response = await fetch(`${getChatServiceUrl()}/api/chat/end/${indexName}/${connectionId}`, {
                    method: 'POST'
                });

                if (!response.ok) {
                    console.warn(`Chat end API returned ${response.status}: ${response.statusText}`);
                    // Continue even if API fails - the WebSocket message may have worked
                }
            } catch (apiErr) {
                console.warn('Chat end API error:', apiErr);
                // Continue even if API fails - the WebSocket message may have worked
            }

            // Navigate back to the dashboard instead of trying to close the window
            // This avoids the "Scripts may close only the windows that were opened by them" error
            window.location.href = `/dashboard/${indexName}`;
        } catch (err) {
            console.error('Error ending chat:', err);
            setError(`Failed to end chat: ${err instanceof Error ? err.message : String(err)}`);
        }
    };

    // Get message style based on message role
    const getMessageStyle = (role: string) => {
        switch (role) {
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

    // Get avatar for message role
    const getAvatar = (role: string) => {
        switch (role) {
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
                            ...getMessageStyle(message.role || message.type),
                            p: 1.5,
                            maxWidth: '80%'
                        }}
                    >
                        <Avatar sx={{ mr: 1, bgcolor: 'transparent' }}>{getAvatar(message.role || message.type)}</Avatar>
                        <Box>
                            <Typography variant="caption" color="text.secondary">
                                {(message.role || message.type) ? (message.role || message.type).toUpperCase() : 'UNKNOWN'} • {new Date(message.timestamp).toLocaleTimeString()}
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

                {/* Typing indicator - only shown during RAG/LLM processing */}
                {isUserTyping && !isEscalated && !isAgentTakeover && isRagLlmProcessing && (
                    <Box
                        sx={{
                            display: 'flex',
                            alignSelf: 'flex-start',
                            mb: 1,
                            p: 1.5,
                            maxWidth: '80%',
                            bgcolor: 'grey.100',
                            borderRadius: 2,
                            borderBottomLeftRadius: 0
                        }}
                    >
                        <Avatar sx={{ mr: 1, bgcolor: 'transparent' }}>{getAvatar('assistant')}</Avatar>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <Typography variant="body1">
                                <span className="typing-dot">.</span>
                                <span className="typing-dot">.</span>
                                <span className="typing-dot">.</span>
                                <style jsx global>{`
                                    @keyframes typingAnimation {
                                        0% { opacity: 0.3; }
                                        50% { opacity: 1; }
                                        100% { opacity: 0.3; }
                                    }
                                    .typing-dot {
                                        animation: typingAnimation 1.4s infinite;
                                        animation-fill-mode: both;
                                        font-size: 20px;
                                        line-height: 10px;
                                        margin-right: 3px;
                                    }
                                    .typing-dot:nth-child(2) {
                                        animation-delay: 0.2s;
                                    }
                                    .typing-dot:nth-child(3) {
                                        animation-delay: 0.4s;
                                    }
                                `}</style>
                            </Typography>
                        </Box>
                    </Box>
                )}

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