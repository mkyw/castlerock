/**
 * Widget Authentication Helper
 * This script helps authenticate the chatbot widget with the backend
 */

(function() {
  // Function to get JWT token from cookies or localStorage
  function getAuthToken() {
    // Try to get from localStorage first (used by Next-Auth)
    try {
      const nextAuthSession = localStorage.getItem('next-auth.session-token');
      if (nextAuthSession) {
        return nextAuthSession;
      }
    } catch (e) {
      console.error('Error accessing localStorage:', e);
    }
    
    // Try to get from cookies as fallback
    const cookies = document.cookie.split(';');
    for (let i = 0; i < cookies.length; i++) {
      const cookie = cookies[i].trim();
      // Check for next-auth.session-token
      if (cookie.startsWith('next-auth.session-token=')) {
        return cookie.substring('next-auth.session-token='.length);
      }
      // Also check for JWT token if stored differently
      if (cookie.startsWith('jwt=')) {
        return cookie.substring('jwt='.length);
      }
    }
    
    return null;
  }
  
  // Function to send token to widget iframe
  function sendTokenToWidget() {
    const token = getAuthToken();
    if (!token) {
      console.warn('No authentication token found for widget');
      return;
    }
    
    // Find all chatbot widget iframes
    const iframes = document.querySelectorAll('iframe[id^="chatbot-widget-frame"]');
    if (iframes.length === 0) {
      // If no iframes, the widget might be in the same window
      window.postMessage({
        type: 'chatbot-auth-token',
        token: token
      }, '*');
      return;
    }
    
    // Send token to each iframe
    iframes.forEach(iframe => {
      try {
        iframe.contentWindow.postMessage({
          type: 'chatbot-auth-token',
          token: token
        }, '*');
        console.log('Auth token sent to widget iframe');
      } catch (e) {
        console.error('Error sending token to iframe:', e);
      }
    });
  }
  
  // Send token when page loads
  if (document.readyState === 'complete') {
    sendTokenToWidget();
  } else {
    window.addEventListener('load', sendTokenToWidget);
  }
  
  // Also send token periodically in case widget loads after this script
  setInterval(sendTokenToWidget, 2000);
})();
