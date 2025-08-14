/**
 * Widget configuration utilities
 * 
 * This file provides the embed code generation for the chatbot widget.
 */

/**
 * Generates the embed code for the widget
 * @param widgetUrl The URL to the widget script
 * @param config Optional configuration for the widget
 * @returns The embed code as a string
 */
export function generateEmbedCode(widgetUrl: string, config?: { ui?: { title?: string } }): string {
    // Generate the simplified embed code with just the script loader
    let embedCode = `<!-- Add this code before the closing </body> tag -->
<script>
  // Load the chatbot widget script
  (function() {
    const script = document.createElement('script');
    script.src = "${widgetUrl}";
    script.async = true;`;
    
    // Add custom configuration if provided
    if (config) {
      embedCode += `
    
    // Add custom configuration
    window.chatbotConfig = ${JSON.stringify(config, null, 2)};
    `;
    }
    
    embedCode += `
    document.body.appendChild(script);
  })();
</script>`;
    
    return embedCode;
}

// C# Chat Service URL
export const CHAT_SERVICE_URL = process.env.NEXT_PUBLIC_CHAT_SERVICE_URL || 'http://localhost:5001';

// WebSocket URL for chat connections
export const BACKEND_WS_URL = process.env.NEXT_PUBLIC_BACKEND_WS_URL || 'ws://localhost:5001/ws/chat';

// Widget customization options
export const customizationOptions = [
    { name: 'theme', description: 'Set the color theme ("light" or "dark")' },
    { name: 'position', description: 'Position of the widget ("bottom-right", "bottom-left", "top-right", "top-left")' },
    { name: 'initialMessage', description: 'Custom initial message to display when chat opens' }
];