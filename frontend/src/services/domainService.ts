import api from '@/lib/api';

interface DomainLink {
  id: string;
  user_id: string;
  index_name: string;
  domain: string;
  description?: string;
  api_key: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

/**
 * Fetches all domain links for the current user
 * @returns Promise with the list of domain links
 */
export const getDomains = async (): Promise<DomainLink[]> => {
  try {
    const response = await api.get('/api/domain-auth/domains');
    return response.data;
  } catch (error) {
    console.error('Error fetching domains:', error);
    throw error;
  }
};

/**
 * Creates a new domain link
 * @param domainData The domain data to create
 * @returns Promise with the created domain link
 */
export const createDomain = async (domainData: {
  index_name: string;
  domain: string;
  description?: string;
}): Promise<DomainLink> => {
  try {
    const response = await api.post('/api/domain-auth/domains', domainData);
    return response.data;
  } catch (error) {
    console.error('Error creating domain:', error);
    throw error;
  }
};

/**
 * Deletes a domain link
 * @param linkId The ID of the domain link to delete
 * @returns Promise that resolves when the domain is deleted
 */
export const deleteDomain = async (linkId: string): Promise<void> => {
  try {
    await api.delete(`/api/domain-auth/domains/${linkId}`);
  } catch (error) {
    console.error('Error deleting domain:', error);
    throw error;
  }
};

/**
 * Gets domain links filtered by index name
 * @param indexName The index name to filter by
 * @returns Promise with the filtered domain links
 */
export const getDomainsByIndex = async (indexName: string): Promise<DomainLink[]> => {
  try {
    const response = await api.get(`/api/domain-auth/domains?index_name=${encodeURIComponent(indexName)}`);
    return response.data;
  } catch (error) {
    console.error(`Error fetching domains for index ${indexName}:`, error);
    throw error;
  }
};
