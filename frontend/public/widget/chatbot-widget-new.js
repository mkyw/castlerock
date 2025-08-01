/**
 * Simple Chatbot Widget
 * A self-contained chat widget that can be embedded on any website
 */

(function () {
  // Version number - updated by build script
  const VERSION = '1.0.0';

  // Default configuration
  const defaultConfig = {
    // API settings
    api: {
      baseUrl: 'http://localhost:8000/api/rag/query',
      timeout: 30000,
    },
    // UI settings
    ui: {
      buttonText: '💬',
      title: 'Castlerock AI',
      placeholder: 'Type your message...',
      sendButtonText: 'Send',
      primaryColor: '#4a90e2',
      textColor: '#333333',
      backgroundColor: '#ffffff',
      position: 'bottom-right',
      zIndex: 9999,
    },
    // Behavior
    behavior: {
      autoOpen: false,
      rememberSession: false,
      showTimestamp: true,
      debug: false,
    }
  };

  // WebSocket URL
  const WS_URL = 'ws://localhost:8765';

  // Main widget class
  class ChatbotWidget {
    constructor(config) {
      this.config = this.mergeConfig(defaultConfig, config || {});
      this.socket = null;
      this.connectionId = null;
      this.messages = [];
      this.isOpen = false;
      this.isConnecting = false;
      this.reconnectAttempts = 0;
      this.maxReconnectAttempts = 5;
      this.reconnectInterval = 3000;
      this.initialize();
    }

    // Merge configurations
    mergeConfig(defaults, overrides) {
      const result = { ...defaults };

      // Deep merge for nested objects
      for (const key in overrides) {
        if (overrides[key] && typeof overrides[key] === 'object' && !Array.isArray(overrides[key])) {
          result[key] = { ...result[key], ...overrides[key] };
        } else {
          result[key] = overrides[key];
        }
      }

      return result;
    }

    // Initialize the widget
    initialize() {
      try {
        // Create elements
        this.createButton();
        this.createChatContainer();
        this.setupEventListeners();

        // Load previous session if enabled
        if (this.config.behavior.rememberSession) {
          this.loadSession();
        }

        // Auto-open if configured
        if (this.config.behavior.autoOpen) {
          this.openChat();
        }

        // Connect to WebSocket when initialized
        this.connectWebSocket();
      } catch (error) {
        console.error('[Chatbot] Initialize error:', error);
      }
    }

    // Create the chat button
    createButton() {
      this.button = document.createElement('button');
      this.button.id = 'chatbot-widget-button';
      this.button.innerHTML = this.config.ui.buttonText;

      // Apply styles
      Object.assign(this.button.style, {
        position: 'fixed',
        [this.config.ui.position.includes('right') ? 'right' : 'left']: '20px',
        [this.config.ui.position.includes('top') ? 'top' : 'bottom']: '20px',
        width: '60px',
        height: '60px',
        borderRadius: '50%',
        backgroundColor: this.config.ui.primaryColor,
        color: 'white',
        border: 'none',
        boxShadow: '0 2px 10px rgba(0, 0, 0, 0.1)',
        cursor: 'pointer',
        fontSize: '24px',
        zIndex: this.config.ui.zIndex,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        transition: 'transform 0.2s ease',
      });

      // Add hover effect
      this.button.addEventListener('mouseenter', () => {
        this.button.style.transform = 'scale(1.05)';
      });

      this.button.addEventListener('mouseleave', () => {
        this.button.style.transform = 'scale(1)';
      });

      // Add to body
      document.body.appendChild(this.button);
    }

    // Create the chat container
    createChatContainer() {
      // Main container
      this.container = document.createElement('div');
      this.container.id = 'chatbot-widget-container';

      // Apply styles
      Object.assign(this.container.style, {
        position: 'fixed',
        [this.config.ui.position.includes('right') ? 'right' : 'left']: '20px',
        [this.config.ui.position.includes('top') ? 'top' : 'bottom']: '20px',
        width: '350px',
        maxWidth: '90vw',
        height: '500px',
        maxHeight: '80vh',
        backgroundColor: this.config.ui.backgroundColor,
        borderRadius: '12px',
        boxShadow: '0 5px 20px rgba(0, 0, 0, 0.15)',
        display: 'none',
        flexDirection: 'column',
        zIndex: this.config.ui.zIndex,
        overflow: 'hidden',
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      });

      // Header
      const header = document.createElement('div');
      header.style.padding = '15px 20px';
      header.style.backgroundColor = this.config.ui.primaryColor;
      header.style.color = 'white';
      header.style.display = 'flex';
      header.style.justifyContent = 'space-between';
      header.style.alignItems = 'center';

      const title = document.createElement('h3');
      title.textContent = this.config.ui.title;
      title.style.margin = '0';
      title.style.fontSize = '16px';
      title.style.fontWeight = '500';

      const closeButton = document.createElement('button');
      closeButton.innerHTML = '&times;';
      closeButton.style.background = 'none';
      closeButton.style.border = 'none';
      closeButton.style.color = 'white';
      closeButton.style.fontSize = '24px';
      closeButton.style.cursor = 'pointer';
      closeButton.style.padding = '0';
      closeButton.style.lineHeight = '1';
      closeButton.style.width = '24px';
      closeButton.style.height = '24px';
      closeButton.style.display = 'flex';
      closeButton.style.alignItems = 'center';
      closeButton.style.justifyContent = 'center';

      closeButton.addEventListener('click', () => this.closeChat());

      header.appendChild(title);
      header.appendChild(closeButton);

      // Messages container
      this.messagesContainer = document.createElement('div');
      this.messagesContainer.style.flex = '1';
      this.messagesContainer.style.padding = '15px';
      this.messagesContainer.style.overflowY = 'auto';
      this.messagesContainer.style.display = 'flex';
      this.messagesContainer.style.flexDirection = 'column';
      this.messagesContainer.style.gap = '10px';

      // Input container
      const inputContainer = document.createElement('div');
      inputContainer.style.padding = '15px';
      inputContainer.style.borderTop = '1px solid #eee';
      inputContainer.style.display = 'flex';
      inputContainer.style.gap = '10px';

      this.input = document.createElement('textarea');
      this.input.placeholder = this.config.ui.placeholder;
      this.input.style.flex = '1';
      this.input.style.padding = '10px 15px';
      this.input.style.border = '1px solid #ddd';
      this.input.style.borderRadius = '20px';
      this.input.style.outline = 'none';
      this.input.style.fontSize = '14px';
      this.input.style.resize = 'vertical';
      this.input.style.overflow = 'auto';
      this.input.style.minHeight = '40px';
      this.input.style.maxHeight = '120px';
      this.input.style.fontFamily = 'inherit';
      this.input.style.lineHeight = '1.4';
      this.input.rows = 1;
      this.input.style.boxSizing = 'border-box';

      const sendButton = document.createElement('button');
      sendButton.textContent = this.config.ui.sendButtonText;
      sendButton.style.padding = '0 20px';
      sendButton.style.backgroundColor = this.config.ui.primaryColor;
      sendButton.style.color = 'white';
      sendButton.style.border = 'none';
      sendButton.style.borderRadius = '20px';
      sendButton.style.cursor = 'pointer';
      sendButton.style.fontSize = '14px';
      sendButton.style.fontWeight = '500';

      // Auto-resize textarea function
      const resizeTextarea = () => {
        this.input.style.height = 'auto';
        this.input.style.height = this.input.scrollHeight + 'px';
      };

      // Add event listeners for sending messages
      const sendMessage = () => {
        const message = this.input.value.trim();
        if (message) {
          this.sendMessage(message);
          this.input.value = '';
          resizeTextarea();
        }
      };

      sendButton.addEventListener('click', sendMessage);
      this.input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          sendMessage();
        }
      });

      this.input.addEventListener('input', resizeTextarea);
      this.input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          sendMessage();
        }
      });

      inputContainer.appendChild(this.input);
      inputContainer.appendChild(sendButton);

      // Assemble the container
      this.container.appendChild(header);
      this.container.appendChild(this.messagesContainer);
      this.container.appendChild(inputContainer);

      // Add to body
      document.body.appendChild(this.container);
    }

    // Set up event listeners
    setupEventListeners() {
      // Toggle chat on button click
      this.button.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        this.toggleChat();
        return false;
      }, true);

      // Listen for messages from parent window
      window.addEventListener('message', (event) => {
        try {
          if (event.data && event.data.type === 'chatbot-auth-token') {
            this.setAuthToken(event.data.token);
          }
        } catch (error) {
          console.error('[Chatbot] Error processing message:', error);
        }
      });
    }

    // Toggle chat visibility
    toggleChat() {
      if (this.isOpen) {
        this.closeChat();
      } else {
        this.openChat();
      }
    }

    // Open chat
    openChat() {
      this.isOpen = true;
      this.container.style.display = 'flex';
      this.button.style.display = 'none';
      this.input.focus();

      // Ensure WebSocket is connected when chat is opened
      if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
        this.connectWebSocket();
      }

      // Save session state
      if (this.config.behavior.rememberSession) {
        this.saveSession();
      }
    }

    // Close chat
    closeChat() {
      this.isOpen = false;
      this.container.style.display = 'none';
      this.button.style.display = 'flex';
    }

    // Add a message to the chat
    addMessage(message) {
      // Handle different message formats
      let processedMessage;

      if (typeof message === 'string') {
        // Simple string message (legacy support)
        processedMessage = {
          text: message,
          isUser: false,
          timestamp: new Date().toISOString(),
        };
      } else if (typeof message === 'object') {
        if (message.type === 'user' || message.type === 'assistant' || message.type === 'system') {
          // JSON message from WebSocket
          processedMessage = {
            text: message.content,
            isUser: message.type === 'user',
            timestamp: message.timestamp || new Date().toISOString(),
            isSystem: message.type === 'system'
          };
        } else {
          // Direct message object
          processedMessage = {
            text: message.text || message.content || '',
            isUser: message.isUser || message.type === 'user',
            timestamp: message.timestamp || new Date().toISOString(),
            isSystem: message.isSystem || message.type === 'system'
          };
        }
      } else {
        console.error('[Chatbot] Invalid message format:', message);
        return;
      }

      this.messages.push(processedMessage);
      this.renderMessage(processedMessage);

      // Save session
      if (this.config.behavior.rememberSession) {
        this.saveSession();
      }
    }

    // Add a system message
    addSystemMessage(text) {
      const message = {
        text,
        isSystem: true,
        timestamp: new Date().toISOString(),
      };
      this.addMessage(message);
    }

    // Render a message in the chat
    renderMessage(message) {
      const messageElement = document.createElement('div');
      messageElement.style.maxWidth = '80%';
      messageElement.style.padding = '10px 15px';
      messageElement.style.borderRadius = '18px';
      messageElement.style.wordBreak = 'break-word';

      if (message.isSystem) {
        // System message styling
        messageElement.style.alignSelf = 'center';
        messageElement.style.backgroundColor = '#f8f9fa';
        messageElement.style.color = '#666';
        messageElement.style.fontSize = '0.9em';
        messageElement.style.border = '1px solid #ddd';
        messageElement.style.margin = '10px 0';
      } else if (message.isUser) {
        messageElement.style.alignSelf = 'flex-end';
        messageElement.style.backgroundColor = this.config.ui.primaryColor;
        messageElement.style.color = 'white';
        messageElement.style.borderBottomRightRadius = '4px';
      } else {
        messageElement.style.alignSelf = 'flex-start';
        messageElement.style.backgroundColor = '#f1f1f1';
        messageElement.style.color = this.config.ui.textColor;
        messageElement.style.borderBottomLeftRadius = '4px';
      }

      // Add timestamp if enabled
      let messageContent = message.text;
      if (this.config.behavior.showTimestamp) {
        const time = new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        messageContent += ` <small style="opacity:0.7;font-size:0.8em;">${time}</small>`;
      }

      messageElement.innerHTML = messageContent;
      this.messagesContainer.appendChild(messageElement);

      // Scroll to bottom
      this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }

    // Helper method to remove typing indicator
    removeTypingIndicator(typingId) {
      const typingElement = document.getElementById(typingId);
      if (typingElement) {
        typingElement.remove();
      }
    }

    // Connect to WebSocket
    connectWebSocket() {
      if (this.isConnecting) return;

      this.isConnecting = true;

      // Close existing socket if any
      if (this.socket) {
        this.socket.close();
      }

      const wsUrl = `${WS_URL}/${this.config.indexName || 'default'}`;
      console.log(`Connecting to WebSocket: ${wsUrl}`);

      try {
        this.socket = new WebSocket(wsUrl);

        this.socket.onopen = () => {
          console.log('WebSocket connected');
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          this.addSystemMessage('Connected to chat server');
        };

        this.socket.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            console.log('Received message:', data);

            // Store connection ID if provided
            if (data.connection_id && !this.connectionId) {
              this.connectionId = data.connection_id;
            }

            // Add message to chat
            this.addMessage(data);

            // Handle escalation requests
            if (data.escalation_requested) {
              this.addSystemMessage('Your request for an agent has been received. Please wait while we connect you.');
            }
          } catch (err) {
            console.error('Error parsing WebSocket message:', err);
            // Try to display as plain text if JSON parsing fails
            this.addSystemMessage('Received: ' + event.data);
          }
        };

        this.socket.onclose = (event) => {
          console.log(`WebSocket closed: ${event.code} ${event.reason}`);
          this.isConnecting = false;

          // Attempt to reconnect
          if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            setTimeout(() => {
              this.addSystemMessage(`Reconnecting (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
              this.connectWebSocket();
            }, this.reconnectInterval);
          } else {
            this.addSystemMessage('Could not reconnect to the chat server. Please refresh the page.');
          }
        };

        this.socket.onerror = (error) => {
          console.error('WebSocket error:', error);
          this.isConnecting = false;
        };
      } catch (err) {
        console.error('Error creating WebSocket:', err);
        this.isConnecting = false;
        this.addSystemMessage('Failed to connect to chat server. Please try again later.');
      }
    }

    // Send message via WebSocket
    sendMessage(message) {
      if (!message.trim()) return;

      // Add user message to chat
      const userMessage = {
        type: 'user',
        content: message,
        timestamp: new Date().toISOString()
      };
      this.addMessage(userMessage);

      // Send message via WebSocket if connected
      if (this.socket && this.socket.readyState === WebSocket.OPEN) {
        this.socket.send(JSON.stringify(userMessage));
      } else {
        // Try to reconnect
        this.addSystemMessage('Not connected to chat server. Attempting to reconnect...');
        this.connectWebSocket();

        // Queue the message to be sent after connection
        setTimeout(() => {
          if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify(userMessage));
          } else {
            this.addSystemMessage('Still not connected. Please try again later.');
          }
        }, 2000);
      }
    }

    // Set authentication token
    setAuthToken(token) {
      this.authToken = token;
      console.log('[Chatbot] Auth token set');

      // Save token to localStorage if rememberSession is enabled
      if (this.config.behavior.rememberSession) {
        try {
          localStorage.setItem('chatbot_auth_token', token);
        } catch (error) {
          console.error('[Chatbot] Error saving auth token:', error);
        }
      }
    }

    // Save session to localStorage
    saveSession() {
      try {
        const sessionData = {
          messages: this.messages,
          isOpen: this.isOpen,
          timestamp: new Date().toISOString(),
          connectionId: this.connectionId
        };

        localStorage.setItem('chatbot_session', JSON.stringify(sessionData));
      } catch (error) {
        console.error('[Chatbot] Error saving session:', error);
      }
    }

    // Load session from localStorage
    loadSession() {
      try {
        // Load auth token if exists
        const token = localStorage.getItem('chatbot_auth_token');
        if (token) {
          this.authToken = token;
        }

        // Load session data
        const sessionData = localStorage.getItem('chatbot_session');
        if (sessionData) {
          const { messages, isOpen, connectionId } = JSON.parse(sessionData);

          // Restore connection ID if available
          if (connectionId) {
            this.connectionId = connectionId;
          }

          // Only restore if messages exist
          if (messages && messages.length > 0) {
            this.messages = messages;

            // Clear existing messages
            this.messagesContainer.innerHTML = '';

            // Render all messages
            messages.forEach(message => this.renderMessage(message));

            // Restore chat state
            if (isOpen) {
              this.openChat();
            }
          }
        }
      } catch (error) {
        console.error('[Chatbot] Error loading session:', error);
      }
    }
  }

  // Auto-initialize if script is included directly
  if (!window.ChatbotWidget) {
    const config = window.chatbotConfig || {};

    // Initialize the widget
    window.ChatbotWidget = new ChatbotWidget(config);
  }

  // Export for module usage
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = ChatbotWidget;
  }
})();
