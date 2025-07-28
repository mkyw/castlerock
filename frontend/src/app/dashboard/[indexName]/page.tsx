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
  useMediaQuery
} from '@mui/material';
import { Code as CodeIcon, Web as WebIcon, Settings as SettingsIcon } from '@mui/icons-material';
import EmbedSnippet from '@/components/EmbedSnippet';
import DomainManagement from '@/components/DomainManagement';
import SessionValidator from '@/components/SessionValidator';

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
            <Tab icon={<CodeIcon />} label="Chat Interface" {...a11yProps(0)} />
            <Tab icon={<WebIcon />} label="Website Integration" {...a11yProps(1)} />
            <Tab icon={<SettingsIcon />} label="Domain Settings" {...a11yProps(2)} />
          </Tabs>
        </AppBar>

        <TabPanel value={activeTab} index={0}>
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

        <TabPanel value={activeTab} index={1}>
          <Paper className="p-4">
            <Typography variant="h6" gutterBottom>Website Integration</Typography>
            <Typography variant="body2" color="text.secondary" className="mb-4">
              Add this chat widget to your website by copying the code below:
            </Typography>
            <EmbedSnippet indexName={decodedIndexName} />
          </Paper>
        </TabPanel>

        <TabPanel value={activeTab} index={2}>
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
