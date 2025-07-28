import axios from 'axios';

// Create an axios instance with default config
const apiClient = axios.create({
  baseURL: '', // Use relative URLs
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add a request interceptor to handle errors
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

export default apiClient;
