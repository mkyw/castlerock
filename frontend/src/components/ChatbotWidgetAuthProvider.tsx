"use client";

import { useSession } from 'next-auth/react';
import { useEffect } from 'react';

/**
 * ChatbotWidgetAuthProvider
 * 
 * This component sends the authentication token to the chatbot widget iframe
 * via postMessage. It should be included in your layout or pages that use the chatbot widget.
 */
const ChatbotWidgetAuthProvider: React.FC = () => {
  const { data: session } = useSession();
  
  useEffect(() => {
    // Function to send token to widget
    const sendTokenToWidget = () => {
      // Check for JWT token in session
      // NextAuth stores the JWT in different places depending on configuration
      // Use type assertion to access potential token locations
      const sessionAny = session as any;
      const token = sessionAny?.accessToken || 
                   sessionAny?.token?.accessToken || 
                   sessionAny?.jwt;
      
      if (!token) {
        console.warn('No JWT token found in session for widget authentication');
        return;
      }
      
      // Find all chatbot widget iframes
      const iframes = document.querySelectorAll('iframe[id^="chatbot-widget-frame"]');
      
      if (iframes.length === 0) {
        // If no iframes found, the widget might be in the same window
        window.postMessage({
          type: 'chatbot-auth-token',
          token: token
        }, '*');
        console.log('Auth token sent to window');
        return;
      }
      
      // Send token to each iframe
      iframes.forEach(iframe => {
        try {
          // Cast iframe to HTMLIFrameElement to access contentWindow
          const iframeElement = iframe as HTMLIFrameElement;
          iframeElement.contentWindow?.postMessage({
            type: 'chatbot-auth-token',
            token: token
          }, '*');
          console.log('Auth token sent to widget iframe');
        } catch (e) {
          console.error('Error sending token to iframe:', e);
        }
      });
    };
    
    // Send token when session changes
    if (session) {
      sendTokenToWidget();
      
      // Also set up an interval to periodically send the token
      // This ensures the token is sent even if the widget loads after this component
      const intervalId = setInterval(sendTokenToWidget, 2000);
      
      return () => clearInterval(intervalId);
    }
  }, [session]);
  
  // This component doesn't render anything
  return null;
};

export default ChatbotWidgetAuthProvider;
