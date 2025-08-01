/**
 * Widget configuration utilities
 * 
 * This file centralizes the widget configuration and embed code generation
 * to ensure consistency across different parts of the application.
 */

/**
 * Generates the embed code for the widget
 * @param widgetUrl The URL to the widget script
 * @param customConfig Optional custom configuration to override defaults
 * @returns The embed code as a string
 */
export function generateEmbedCode(widgetUrl: string, customConfig?: Record<string, any>): string {
    // Default configuration
    const defaultConfig = {
        api: {
            baseUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/rag/query',
            timeout: 30000
        },
        ui: {
            buttonText: '💬',
            position: 'bottom-right',
            primaryColor: '#4a90e2',
            textColor: '#000000',
            backgroundColor: '#ffffff',
            title: 'Chat Assistant',
            placeholder: 'Type your message...',
            sendButtonText: 'Send',
            zIndex: 9999
        },
        behavior: {
            autoOpen: false,
            rememberSession: false,
            showTimestamp: true,
            debug: process.env.NODE_ENV === 'development'
        }
    };

    // Merge with custom config if provided
    const config = customConfig ? deepMerge(defaultConfig, customConfig) : defaultConfig;

    // Generate the embed code
    return `<!-- Add this code before the closing </body> tag -->
<script>
  // Configuration for the chatbot widget
  window.chatbotConfig = ${JSON.stringify(config, null, 2)};

  // Load the widget script
  (function() {
    const script = document.createElement('script');
    script.src = "${widgetUrl}";
    script.async = true;
    document.body.appendChild(script);
  })();
</script>`;
}

/**
 * Generates example embed code with custom options
 * @param widgetUrl The URL to the widget script
 * @returns The example embed code as a string
 */
export function generateExampleCode(widgetUrl: string): string {
    const exampleConfig = {
        ui: {
            position: "bottom-right",
            buttonText: "Need help?",
            primaryColor: "#4a90e2",
            textColor: "#ffffff",
            title: "Customer Support",
            backgroundColor: "#ffffff"
        },
        behavior: {
            autoOpen: false,
            rememberSession: true
        },
        api: {
            baseUrl: "https://yourdomain.com/api/rag/query"
        }
    };

    return `<script>
  window.chatbotConfig = ${JSON.stringify(exampleConfig, null, 2)};

  (function() {
    const script = document.createElement('script');
    script.src = "${widgetUrl}";
    script.async = true;
    document.body.appendChild(script);
  })();
</script>`;
}

/**
 * Helper function to deep merge objects
 */
function deepMerge(target: any, source: any): any {
    const output = { ...target };

    if (isObject(target) && isObject(source)) {
        Object.keys(source).forEach(key => {
            if (isObject(source[key])) {
                if (!(key in target)) {
                    Object.assign(output, { [key]: source[key] });
                } else {
                    output[key] = deepMerge(target[key], source[key]);
                }
            } else {
                Object.assign(output, { [key]: source[key] });
            }
        });
    }

    return output;
}

/**
 * Helper function to check if value is an object
 */
function isObject(item: any): boolean {
    return (item && typeof item === 'object' && !Array.isArray(item));
}

/**
 * List of customization options for documentation
 */
export const customizationOptions = [
    { name: 'ui.position', description: "'bottom-right' (default) or 'bottom-left'" },
    { name: 'ui.buttonText', description: 'Customize the button text' },
    { name: 'ui.primaryColor', description: 'Button background color (hex, rgb, or named color)' },
    { name: 'ui.textColor', description: 'Button text color' },
    { name: 'ui.backgroundColor', description: 'Chat window background color' },
    { name: 'ui.title', description: 'Title of the chat window' },
    { name: 'behavior.autoOpen', description: 'true/false to open chat automatically' },
    { name: 'behavior.rememberSession', description: 'true/false to remember chat history' },
    { name: 'api.baseUrl', description: 'API endpoint URL' }
];

// Backend URL for API calls
export const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

// C# Chat Service URL
export const CHAT_SERVICE_URL = process.env.NEXT_PUBLIC_CHAT_SERVICE_URL || 'http://localhost:5000';

// WebSocket URL for chat connections
export const BACKEND_WS_URL = process.env.NEXT_PUBLIC_BACKEND_WS_URL || 'ws://localhost:8765'; 