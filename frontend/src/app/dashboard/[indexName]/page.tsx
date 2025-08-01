'use client';

import { useSession } from 'next-auth/react';
import { useRouter, useParams } from 'next/navigation';
import { Suspense, useEffect, useMemo, useState } from 'react';
import dynamic from 'next/dynamic';
import { formatIndexName } from '@/lib/api-utils';
import {
  Box,
  Typography,
  Button,
  Paper,
  Container,
  Tabs,
  Tab,
  AppBar,
  Toolbar,
  useTheme,
  useMediaQuery,
  Badge,
  Grid,
  Alert,
  CircularProgress,
  Chip
} from '@mui/material';
import { Code as CodeIcon, Web as WebIcon, Settings as SettingsIcon, Chat as ChatIcon } from '@mui/icons-material';
import EmbedSnippet from '@/components/EmbedSnippet';
import DomainManagement from '@/components/DomainManagement';
import SessionValidator from '@/components/SessionValidator';
import { formatDistanceToNow } from 'date-fns';

// Dynamically import the RAGInterface component with no SSR
const RAGInterface = dynamic(() => import('@/components/RAGInterface'), {
  ssr: false,
  loading: () => (
    <Box className="flex justify-center p-4">
      <Box className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500"></Box>
    </Box>
  )
});

// TabPanel props
type TabPanelProps = {
  children?: React.ReactNode;
  index: number;
  value: number;
};

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function a11yProps(index: number) {
  return {
    id: `tab-${index}`,
    'aria-controls': `tabpanel-${index}`,
  };
}

function KnowledgeBasePage() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const [isClient, setIsClient] = useState(false);
  const [activeTab, setActiveTab] = useState(0);
  const params = useParams();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const [chatStats, setChatStats] = useState({ total: 0, ai_handling: 0, escalation_requested: 0, agent_assigned: 0 });
  const [activeChats, setActiveChats] = useState<any>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Get the indexName from params
  const indexName = useMemo(() => {
    if (!isClient) return '';
    return (params?.indexName as string) || '';
  }, [params?.indexName, isClient]);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  useEffect(() => {
    if (status === 'unauthenticated') {
      router.push('/login');
    }
    setIsClient(true);
  }, [status, router]);

  // Safely decode the index name
  const decodedIndexName = useMemo(() => {
    if (!isClient || !indexName) return '';
    try {
      return decodeURIComponent(indexName);
    } catch (error) {
      console.error('Error decoding index name:', error);
      return indexName; // Return the undecoded version if decoding fails
    }
  }, [indexName, isClient]);

  // Update document title with formatted index name
  useEffect(() => {
    if (decodedIndexName) {
      document.title = `${formatIndexName(decodedIndexName)} | Castlerock`;
    }
  }, [decodedIndexName]);

  // Fetch chat statistics and active chats
  useEffect(() => {
    if (status === 'authenticated' && session?.accessToken && activeTab === 0) {
      const fetchChatData = async () => {
        setLoading(true);
        try {
          // Try to fetch chat statistics from our enhanced WebSocket server
          try {
            const statsResponse = await fetch('http://localhost:8766/api/chat/stats');

            if (statsResponse.ok) {
              const statsData = await statsResponse.json();
              setChatStats({
                total: statsData.total_active || 0,
                ai_handling: statsData.total_active - (statsData.escalation_requested || 0),
                escalation_requested: statsData.escalation_requested || 0,
                agent_assigned: statsData.agent_assigned || 0
              });
            } else {
              console.warn('Chat statistics API not available yet');
              // Set default stats instead of failing
              setChatStats({ total: 0, ai_handling: 0, escalation_requested: 0, agent_assigned: 0 });
            }
          } catch (statsErr) {
            console.warn('Error fetching chat statistics:', statsErr);
            // Set default stats instead of failing
            setChatStats({ total: 0, ai_handling: 0, escalation_requested: 0, agent_assigned: 0 });
          }

          // Try to fetch active chats from our enhanced WebSocket server
          try {
            const chatsResponse = await fetch('http://localhost:8766/api/chat/active');

            if (chatsResponse.ok) {
              const chatsData = await chatsResponse.json();

              // Convert the flat structure to the nested structure expected by the UI
              const formattedChats: Record<string, Record<string, any>> = {};

              chatsData.connections.forEach((chat: any) => {
                const indexName = chat.index_name;
                const connectionId = chat.connection_id;

                if (!formattedChats[indexName]) {
                  formattedChats[indexName] = {};
                }

                formattedChats[indexName][connectionId] = {
                  status: chat.status,
                  connected_at: chat.connected_at,
                  last_activity: chat.last_activity,
                  message_count: chat.message_count,
                  user_agent: chat.user_agent
                };
              });

              setActiveChats(formattedChats);
            } else {
              console.warn('Active chats API not available yet');
              // Set empty chats instead of failing
              setActiveChats({});
            }
          } catch (chatsErr) {
            console.warn('Error fetching active chats:', chatsErr);
            // Set empty chats instead of failing
            setActiveChats({});
          }

          // Clear any previous errors since we're handling them gracefully now
          setError(null);
        } catch (err) {
          console.error('Unexpected error in fetchChatData:', err);
          setError('Some chat data may not be available. The WebSocket server might be starting up.');
        } finally {
          setLoading(false);
        }
      };

      fetchChatData();

      // Set up polling for updates
      const interval = setInterval(fetchChatData, 5000); // Update every 5 seconds

      return () => clearInterval(interval);
    }
  }, [status, session, activeTab]);

  const handleTakeChat = async (indexName: string, connectionId: string) => {
    if (!session?.accessToken) return;

    try {
      // Send command to take over the chat
      const response = await fetch('/api/chat/take-over', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${session.accessToken}`
        },
        body: JSON.stringify({
          index_name: indexName,
          connection_id: connectionId,
          agent_id: session.user?.email || 'unknown'
        })
      });

      if (!response.ok) {
        throw new Error('Failed to take over chat');
      }

      // Refresh chat data
      const chatsResponse = await fetch('/api/chat/active', {
        headers: {
          'Authorization': `Bearer ${session.accessToken}`
        }
      });

      if (!chatsResponse.ok) {
        throw new Error('Failed to fetch active chats');
      }

      const chatsData = await chatsResponse.json();
      setActiveChats(chatsData);

    } catch (err) {
      console.error('Error taking over chat:', err);
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    }
  };

  // Function to get chats by status
  const getChatsByStatus = (status: string) => {
    const result: any[] = [];

    // If we're still loading or there's no data, return empty array
    if (loading || !activeChats) return [];

    // Iterate through all indices and connections
    Object.keys(activeChats).forEach(indexName => {
      Object.keys(activeChats[indexName]).forEach(connectionId => {
        const chat = activeChats[indexName][connectionId];
        if (chat.status === status) {
          result.push({
            ...chat,
            indexName,
            connectionId,
            messageCount: chat.message_count || 0
          });
        }
      });
    });

    // Sort by last activity
    return result.sort((a, b) => {
      return new Date(b.last_activity).getTime() - new Date(a.last_activity).getTime();
    });
  };

  // Get chats for each category
  const agentChats = getChatsByStatus('agent_assigned');
  const escalationChats = getChatsByStatus('escalation_requested');
  const otherChats = getChatsByStatus('active');

  // Helper function to get all chats regardless of status
  const getAllChats = () => {
    const result: any[] = [];

    // If we're still loading or there's no data, return empty array
    if (loading || !activeChats) return [];

    // Iterate through all indices and connections
    Object.keys(activeChats).forEach(indexName => {
      Object.keys(activeChats[indexName]).forEach(connectionId => {
        const chat = activeChats[indexName][connectionId];
        // Include all chats
        result.push({
          ...chat,
          indexName,
          connectionId,
          messageCount: chat.message_count || 0
        });
      });
    });

    // Sort by last activity
    return result.sort((a, b) => {
      return new Date(b.last_activity).getTime() - new Date(a.last_activity).getTime();
    });
  };

  // Get all chats if we don't have any in the specific categories
  const allChats = getAllChats();
  const hasNoSpecificChats = agentChats.length === 0 && escalationChats.length === 0 && otherChats.length === 0;

  if (status === 'loading' || status === 'unauthenticated' || !isClient) {
    return (
      <Box className="flex justify-center items-center min-h-screen">
        <Box className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></Box>
      </Box>
    );
  }

  return (
    <Container maxWidth="xl" className="py-4 md:py-8">
      <SessionValidator />
      <Box className="mb-6">
        <Typography variant="h4" component="h1" gutterBottom>
          {formatIndexName(decodedIndexName)}
        </Typography>

        <AppBar position="static" color="default" elevation={0} className="rounded-md overflow-hidden mb-6">
          <Tabs
            value={activeTab}
            onChange={handleTabChange}
            variant={isMobile ? "scrollable" : "standard"}
            scrollButtons="auto"
            allowScrollButtonsMobile
            aria-label="index management tabs"
            className="bg-gray-50"
            sx={{
              '& .MuiTabs-indicator': {
                backgroundColor: theme.palette.primary.main,
              },
            }}
          >
            <Tab
              icon={<ChatIcon />}
              label={
                <Badge
                  color="error"
                  variant="dot"
                >
                  Live chats
                </Badge>
              }
              {...a11yProps(0)}
            />
            <Tab icon={<CodeIcon />} label="Chat Interface" {...a11yProps(1)} />
            <Tab icon={<WebIcon />} label="Website Integration" {...a11yProps(2)} />
            <Tab icon={<SettingsIcon />} label="Domain Settings" {...a11yProps(3)} />
          </Tabs>
        </AppBar>

        <TabPanel value={activeTab} index={0}>
          <Paper className="p-4 h-full">
            <Typography variant="h6" gutterBottom>
              Live Chat Sessions
            </Typography>

            {loading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                <Box className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500"></Box>
              </Box>
            ) : error ? (
              <Alert severity="info" sx={{ mb: 2 }}>
                {error}
                <Typography variant="body2" sx={{ mt: 1 }}>
                  This feature requires the backend WebSocket service to be running.
                  The data will automatically refresh once the service is available.
                </Typography>
              </Alert>
            ) : chatStats.total === 0 && Object.keys(activeChats).length === 0 ? (
              <Alert severity="info" sx={{ mb: 2 }}>
                No active chat sessions found. This could be because:
                <ul>
                  <li>There are no active chats at the moment</li>
                  <li>The backend WebSocket service is still starting up</li>
                  <li>The WebSocket connection hasn't been established yet</li>
                </ul>
                <Typography variant="body2" sx={{ mt: 1 }}>
                  The data will automatically refresh every 30 seconds.
                </Typography>
              </Alert>
            ) : (
              <>
                {/* Current agent tickets section */}
                <Box sx={{ mb: 4 }}>
                  <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 500, display: 'flex', alignItems: 'center' }}>
                    Current agent tickets
                    <Badge color="primary" badgeContent={agentChats.length} sx={{ ml: 2 }} />
                  </Typography>
                  {agentChats.length > 0 ? (
                    <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: 2 }}>
                      {agentChats.map((chat) => (
                        <Paper
                          key={chat.connectionId}
                          elevation={3}
                          className="p-4"
                          sx={{
                            height: 180,
                            display: 'flex',
                            flexDirection: 'column',
                            justifyContent: 'space-between',
                            cursor: 'pointer',
                            '&:hover': {
                              boxShadow: 6
                            }
                          }}
                          onClick={() => router.push(`/dashboard/${chat.indexName}/chat/${chat.connectionId}`)}
                        >
                          <Box>
                            <Typography variant="subtitle1" fontWeight="bold">
                              Visitor #{chat.connectionId.substring(0, 6)}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              Started: {new Date(chat.connected_at).toLocaleTimeString()}
                            </Typography>
                            <Typography variant="body2" color="primary">
                              Agent: {chat.agent_id || 'Unknown'}
                            </Typography>
                          </Box>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <Typography variant="caption" color="text.secondary">
                              {chat.messageCount} messages
                            </Typography>
                            <Badge color="success" variant="dot" />
                          </Box>
                        </Paper>
                      ))}
                    </Box>
                  ) : (
                    <Typography variant="body2" color="text.secondary">No active agent chats</Typography>
                  )}
                </Box>

                {/* Agent escalation requested section */}
                <Box sx={{ mb: 4 }}>
                  <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 500, display: 'flex', alignItems: 'center' }}>
                    Agent escalation requested
                    <Badge color="error" badgeContent={escalationChats.length} sx={{ ml: 2 }} />
                  </Typography>
                  {escalationChats.length > 0 ? (
                    <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: 2 }}>
                      {escalationChats.map((chat) => {
                        // Calculate waiting time
                        const waitingSince = new Date(chat.escalation_requested_at || chat.last_activity);
                        const now = new Date();
                        const waitingMinutes = Math.floor((now.getTime() - waitingSince.getTime()) / 60000);

                        return (
                          <Paper
                            key={chat.connectionId}
                            elevation={3}
                            className="p-4"
                            sx={{
                              height: 180,
                              display: 'flex',
                              flexDirection: 'column',
                              justifyContent: 'space-between',
                              cursor: 'pointer',
                              borderLeft: '4px solid',
                              borderColor: 'error.main',
                              '&:hover': {
                                boxShadow: 6
                              }
                            }}
                            onClick={() => router.push(`/dashboard/${chat.indexName}/chat/${chat.connectionId}`)}
                          >
                            <Box>
                              <Typography variant="subtitle1" fontWeight="bold">
                                Visitor #{chat.connectionId.substring(0, 6)}
                              </Typography>
                              <Typography variant="body2" color="text.secondary">
                                Started: {new Date(chat.connected_at).toLocaleTimeString()}
                              </Typography>
                              <Typography variant="body2" color="error">
                                Waiting: {waitingMinutes} min
                              </Typography>
                            </Box>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <Typography variant="caption" color="text.secondary">
                                {chat.messageCount} messages
                              </Typography>
                              <Badge color="error" variant="dot" />
                            </Box>
                          </Paper>
                        );
                      })}
                    </Box>
                  ) : (
                    <Typography variant="body2" color="text.secondary">No escalation requests</Typography>
                  )}
                </Box>

                {/* Other chats section */}
                <Box>
                  <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 500, display: 'flex', alignItems: 'center' }}>
                    {hasNoSpecificChats ? "All Active Chats" : "Other"}
                    <Badge color="default" badgeContent={hasNoSpecificChats ? allChats.length : otherChats.length} sx={{ ml: 2 }} />
                  </Typography>
                  {(hasNoSpecificChats ? allChats : otherChats).length > 0 ? (
                    <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: 2 }}>
                      {(hasNoSpecificChats ? allChats : otherChats).map((chat) => (
                        <Paper
                          key={chat.connectionId}
                          elevation={3}
                          className="p-4"
                          sx={{
                            height: 180,
                            display: 'flex',
                            flexDirection: 'column',
                            justifyContent: 'space-between',
                            cursor: 'pointer',
                            opacity: 0.8,
                            '&:hover': {
                              boxShadow: 6,
                              opacity: 1
                            }
                          }}
                          onClick={() => router.push(`/dashboard/${chat.indexName}/chat/${chat.connectionId}`)}
                        >
                          <Box>
                            <Typography variant="subtitle1" fontWeight="bold">
                              Visitor #{chat.connectionId.substring(0, 6)}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              Started: {new Date(chat.connected_at).toLocaleTimeString()}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              AI handling
                            </Typography>
                          </Box>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <Typography variant="caption" color="text.secondary">
                              {chat.messageCount} messages
                            </Typography>
                            <Badge color="info" variant="dot" />
                          </Box>
                        </Paper>
                      ))}
                    </Box>
                  ) : (
                    <Typography variant="body2" color="text.secondary">No AI-handled chats</Typography>
                  )}
                </Box>
              </>
            )}
          </Paper>
        </TabPanel>

        <TabPanel value={activeTab} index={1}>
          <Paper className="p-4 h-full">
            <Suspense fallback={
              <Box className="flex justify-center p-8">
                <Box className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500"></Box>
              </Box>
            }>
              <RAGInterface indexName={decodedIndexName} />
            </Suspense>
          </Paper>
        </TabPanel>

        <TabPanel value={activeTab} index={2}>
          <Paper className="p-4">
            <Typography variant="h6" gutterBottom>
              Add Chatbot to Your Website
            </Typography>
            <Typography variant="body1" paragraph>
              Copy and paste this code into your website's HTML to add the chatbot widget.
              The widget will automatically connect to your "{indexName}" knowledge base.
            </Typography>
            <EmbedSnippet indexName={decodedIndexName} />

            <Box sx={{ mt: 2, p: 2, bgcolor: 'info.50', borderRadius: 1 }}>
              <Typography variant="subtitle1" color="text.primary" gutterBottom>
                Quick Tips:
              </Typography>
              <ul style={{ margin: '0 0 0 20px', padding: 0, color: 'text.primary' }}>
                <li>Place this code before the <code style={{ background: 'rgba(0,0,0,0.05)', padding: '2px 4px', borderRadius: 4 }}>&lt;/body&gt;</code> tag.</li>
                <li>Paste once and forget - your website will automatically get the latest chatbot version.</li>
                <li>The widget will automatically connect to your knowledge base</li>
                <li>The widget inherits your website's colors by default</li>
              </ul>
            </Box>
          </Paper>
        </TabPanel>

        <TabPanel value={activeTab} index={3}>
          <DomainManagement indexName={decodedIndexName} />
        </TabPanel>
      </Box>
    </Container>
  );
}

export default function Page() {
  return (
    <Suspense fallback={
      <Box className="min-h-screen flex items-center justify-center">
        <Box className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></Box>
      </Box>
    }>
      <KnowledgeBasePage />
    </Suspense>
  );
}
