import { getSession } from 'next-auth/react';
import { handleAuthError } from './auth-utils';

// Cache storage for API responses
interface CacheEntry {
  data: any;
  timestamp: number;
  expiry: number; // Time in milliseconds when this cache entry expires
}

const apiCache: Record<string, CacheEntry> = {};

// Default cache TTL (5 seconds)
const DEFAULT_CACHE_TTL = 5000;

/**
 * Make an authenticated API request with caching support
 * 
 * @param url The API endpoint URL
 * @param options Fetch options
 * @param cacheOptions Caching options
 * @returns The API response data
 */
export async function apiRequest<T = any>(
  url: string,
  options: RequestInit = {},
  cacheOptions: {
    useCache?: boolean,
    ttl?: number,
    forceRefresh?: boolean
  } = {}
): Promise<T> {
  const { useCache = true, ttl = DEFAULT_CACHE_TTL, forceRefresh = false } = cacheOptions;

  // Generate a cache key based on URL and request method
  const cacheKey = `${options.method || 'GET'}:${url}`;

  // Check cache for non-mutation requests if caching is enabled and not forcing refresh
  if (useCache && !forceRefresh && (!options.method || options.method === 'GET')) {
    const cachedResponse = apiCache[cacheKey];
    const now = Date.now();

    if (cachedResponse && now < cachedResponse.expiry) {
      console.log(`Using cached response for ${url}`);
      return cachedResponse.data;
    }
  }

  // Get the session for authentication
  const session = await getSession();
  if (!session?.accessToken) {
    throw new Error('No authentication token available');
  }

  // Set up headers with authentication
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${session.accessToken}`,
    ...options.headers,
  };

  // Make the API request
  const response = await fetch(url, {
    ...options,
    headers,
    credentials: 'include',
  });

  // Handle authentication errors
  const isAuthError = await handleAuthError(response);
  if (isAuthError) {
    throw new Error('Authentication error');
  }

  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }

  // Parse the response if it has content
  let data;
  const contentType = response.headers.get('content-type');
  if (contentType && contentType.includes('application/json')) {
    const text = await response.text();
    data = text ? JSON.parse(text) : null;
  } else {
    data = null;
  }

  // Cache the response for GET requests
  if (useCache && (!options.method || options.method === 'GET') && data !== null) {
    apiCache[cacheKey] = {
      data,
      timestamp: Date.now(),
      expiry: Date.now() + ttl,
    };
  }

  return data;
}

/**
 * Fetch indices with caching
 */
export async function fetchIndices(forceRefresh = false) {
  return apiRequest('/api/rag/indices', {}, { forceRefresh });
}

/**
 * Create a new index
 */
export async function createIndex(displayName: string) {
  const data = await apiRequest('/api/rag/indices', {
    method: 'POST',
    body: JSON.stringify({ display_name: displayName }),
  });

  // Invalidate the indices cache after creating a new index
  delete apiCache['GET:/api/rag/indices'];

  return data;
}

/**
 * Delete an index
 */
export async function deleteIndex(indexName: string) {
  await apiRequest(`/api/rag/indices/${encodeURIComponent(indexName)}`, {
    method: 'DELETE',
  });

  // Invalidate the indices cache after deleting an index
  delete apiCache['GET:/api/rag/indices'];

  return { success: true };
}

/**
 * Fetch domains for an index
 */
export async function fetchDomains(indexName: string) {
  return apiRequest(`/api/domain-auth/domains?index_name=${encodeURIComponent(indexName)}`);
}

/**
 * Add a domain to an index
 */
export async function addDomain(indexName: string, domain: string, description?: string) {
  return apiRequest('/api/domain-auth/domains', {
    method: 'POST',
    body: JSON.stringify({
      index_name: indexName,
      domain,
      description,
    }),
  });
}

/**
 * Delete a domain
 */
export async function deleteDomain(domainId: string) {
  return apiRequest(`/api/domain-auth/domains/${domainId}`, {
    method: 'DELETE',
  });
}
