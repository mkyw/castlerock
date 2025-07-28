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
      buttonText: '💬 Chat',
      title: 'Chat with us',
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
      rememberSession: true,
      showTimestamp: true,
      debug: false,
    }
  };

  // Main widget class
  class ChatbotWidget {
    constructor(config = {}) {
      try {
        // Merge config with defaults
        this.config = this.mergeConfig(defaultConfig, config);

        // State
        this.isOpen = false;
        this.isLoading = false;
        this.messages = [];

        // Initialize
        this.initialize();

        // Add to window for debugging
        if (this.config.behavior.debug) {
          window.ChatbotWidget = this;
        }

        console.log(`[Chatbot] Widget initialized (v${VERSION})`);
      } catch (error) {
        console.error('[Chatbot] Initialization error:', error);
      }
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
        [this.config.ui.position.includes('top') ? 'top' : 'bottom']: '90px',
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

      this.input = document.createElement('input');
      this.input.type = 'text';
      this.input.placeholder = this.config.ui.placeholder;
      this.input.style.flex = '1';
      this.input.style.padding = '10px 15px';
      this.input.style.border = '1px solid #ddd';
      this.input.style.borderRadius = '20px';
      this.input.style.outline = 'none';
      this.input.style.fontSize = '14px';

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

      // Add event listeners for sending messages
      const sendMessage = () => {
        const message = this.input.value.trim();
        if (message) {
          this.addMessage(message, true);
          this.sendMessageToAPI(message);
          this.input.value = '';
        }
      };

      sendButton.addEventListener('click', sendMessage);
      this.input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
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
    addMessage(text, isUser = false) {
      const message = {
        text,
        isUser,
        timestamp: new Date().toISOString(),
      };

      this.messages.push(message);
      this.renderMessage(message);

      // Save session
      if (this.config.behavior.rememberSession) {
        this.saveSession();
      }
    }

    // Render a message in the chat
    renderMessage(message) {
      const messageElement = document.createElement('div');
      messageElement.style.maxWidth = '80%';
      messageElement.style.padding = '10px 15px';
      messageElement.style.borderRadius = '18px';
      messageElement.style.wordBreak = 'break-word';

      if (message.isUser) {
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

    // Send message to API
    async sendMessageToAPI(message) {
      let controller;
      let timeoutId;
      const typingId = 'typing-' + Date.now();

      try {
        this.isLoading = true;

        // Show typing indicator
        this.messagesContainer.insertAdjacentHTML('beforeend', `
          <div id="${typingId}" style="align-self: flex-start; margin: 5px 0;">
            <div class="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        `);

        // Add typing indicator styles if not already present
        if (!document.getElementById('chatbot-typing-styles')) {
          const style = document.createElement('style');
          style.id = 'chatbot-typing-styles';
          style.textContent = `
            @keyframes typing {
              0% { transform: translateY(0); }
              50% { transform: translateY(-5px); }
              100% { transform: translateY(0); }
            }
            .typing-indicator {
              display: flex;
              gap: 5px;
              padding: 10px 15px;
              background: #f1f1f1;
              border-radius: 18px;
              border-bottom-left-radius: 4px;
              width: fit-content;
            }
            .typing-indicator span {
              width: 8px;
              height: 8px;
              background: #999;
              border-radius: 50%;
              display: inline-block;
              animation: typing 1s infinite;
            }
            .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
            .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
          `;
          document.head.appendChild(style);
        }

        // Make API call to the backend
        controller = new AbortController();
        timeoutId = setTimeout(() => controller.abort(), this.config.api.timeout);

        // Get API URL
        const apiUrl = this.config.api.baseUrl;
        if (!apiUrl) {
          throw new Error('API URL is not configured');
        }

        const requestBody = {
          query: message
        };

        console.log('Sending request to:', apiUrl);
        console.log('Request body:', requestBody);

        // Include origin and referer headers for domain authentication
        const headers = {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          'Origin': window.location.origin,
          'Referer': window.location.href
        };

        // Add authorization header if available
        if (this.config.api.domainApiKey) {
          headers['Authorization'] = `Bearer ${this.config.api.domainApiKey}`;
        } else if (this.config.api.userToken) {
          headers['Authorization'] = `Bearer ${this.config.api.userToken}`;
        } else {
          const storedToken = localStorage.getItem('auth_token') || localStorage.getItem('chatbot_auth_token');
          if (storedToken) {
            headers['Authorization'] = `Bearer ${storedToken}`;
          }
        }

        const response = await fetch(apiUrl, {
          method: 'POST',
          headers: headers,
          body: JSON.stringify(requestBody),
          signal: controller.signal,
          credentials: 'include',  // Include cookies if using session-based auth
          mode: 'cors'  // Explicitly set CORS mode
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`HTTP error! status: ${response.status}, body: ${errorText}`);
        }

        const data = await response.json();

        // Check for empty index response
        if (data.error_type === 'empty_index' || data.empty_index === true) {
          this.addMessage("This knowledge base is empty. Please add documents to the index before querying.", false);
          return;
        }

        // Add response to chat
        if (data && data.response) {
          this.addMessage(data.response, false);
        } else if (data && data.answer) {
          this.addMessage(data.answer, false);
        } else {
          throw new Error('Invalid response format from server');
        }

      } catch (error) {
        console.error('[Chatbot] Error sending message:', error);

        let errorMessage = 'An error occurred. Please try again.';
        if (error.name === 'AbortError') {
          errorMessage = 'The request timed out. Please try again.';
        } else if (error.message.includes('401')) {
          errorMessage = 'Authentication failed. Please refresh the page and try again.';
        } else if (error.message.includes('NetworkError') || error.message.includes('Failed to fetch')) {
          errorMessage = 'Unable to connect to the server. Please check your connection.';
        } else if (error.message.includes('CORS')) {
          errorMessage = 'Cross-origin request blocked. Please contact the site administrator.';
        } else if (error.message.includes('list') && error.message.includes('attribute')) {
          errorMessage = 'This knowledge base is empty. Please add documents to the index before querying.';
        }

        this.addMessage(errorMessage, false);

      } finally {
        // Always remove typing indicator and clean up
        this.removeTypingIndicator(typingId);
        if (timeoutId) clearTimeout(timeoutId);
        this.isLoading = false;
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
          const { messages, isOpen } = JSON.parse(sessionData);

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

    // Ensure API URL is set to localhost:3000
    config.api = config.api || {};
    if (!config.api.baseUrl) {
      config.api.baseUrl = 'http://localhost:3000/api/chat';
    }

    // Initialize the widget
    window.ChatbotWidget = new ChatbotWidget(config);
  }

  // Export for module usage
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = ChatbotWidget;
  }
})();
