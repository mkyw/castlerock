// API configuration for chat service
export const CHAT_SERVICE_URL = process.env.NEXT_PUBLIC_CHAT_SERVICE_URL || 'http://localhost:5001';

// Helper function to get chat service URL
export function getChatServiceUrl(): string {
  return CHAT_SERVICE_URL;
}

// Helper function to get WebSocket URL for chat
export function getChatWebSocketUrl(): string {
  // Replace http with ws or https with wss
  return CHAT_SERVICE_URL.replace(/^http/, 'ws');
}
