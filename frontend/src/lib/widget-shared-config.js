/**
 * Shared Widget Configuration
 * 
 * This file contains the default configuration for the chatbot widget.
 * It is used by both the widget script and the embed code generator.
 * 
 * IMPORTANT: This file uses CommonJS module syntax to be compatible with both Node.js and browser environments.
 */

// Default widget version - will be updated by the build script
const VERSION = '2025.7.28-18.46';

// Default configuration
const DEFAULT_CONFIG = {
    // Version number
    version: VERSION,

    // API settings
    api: {
        baseUrl: 'http://localhost:3000/api/chat',
        timeout: 30000,
    },

    // UI settings
    ui: {
        buttonText: '💬',
        title: 'Chat with us',
        placeholder: 'Type your message...',
        sendButtonText: 'Send',
        // Use "inherit" as a special value to inherit from the parent page
        // If the parent page doesn't define these values, fallback colors will be used
        primaryColor: 'inherit:#4a90e2', // Format: "inherit:[fallback-color]"
        textColor: 'inherit:#000000',
        backgroundColor: 'inherit:#ffffff',
        position: 'bottom-right',
        zIndex: 9999,
    },

    // Behavior
    behavior: {
        autoOpen: false,
        rememberSession: true,
        showTimestamp: true,
        debug: false,
    }
};

// Export the configuration
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        VERSION,
        DEFAULT_CONFIG
    };
} 