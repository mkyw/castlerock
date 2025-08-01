using System.Text.RegularExpressions;

namespace ChatService.Services
{
    public class ChatService
    {
        private readonly ConnectionManager _connectionManager;
        private readonly ILogger<ChatService> _logger;

        public ChatService(ConnectionManager connectionManager, ILogger<ChatService> logger)
        {
            _connectionManager = connectionManager;
            _logger = logger;
        }

        public async Task SendSystemMessage(string indexName, string connectionId, string content)
        {
            await _connectionManager.SendMessageAsync(indexName, connectionId, "system", content);
        }

        public async Task SendAgentMessage(string indexName, string connectionId, string content, string agentId)
        {
            await _connectionManager.SendMessageAsync(indexName, connectionId, "agent", content, new { agent_id = agentId });
        }

        public async Task ProcessUserMessage(string indexName, string connectionId, string content)
        {
            // Check if the user is requesting an agent
            if (IsEscalationRequest(content))
            {
                _logger.LogInformation($"Escalation requested for {connectionId} in {indexName}");
                
                // Mark the connection as needing an agent
                _connectionManager.RequestEscalation(indexName, connectionId);
                
                // Send a message back to the user
                await _connectionManager.SendMessageAsync(indexName, connectionId, "system", 
                    "I'll connect you with a human agent. Please wait a moment...");
                
                return;
            }
            
            // For now, just echo the message back (this would be replaced with AI processing)
            await _connectionManager.SendMessageAsync(indexName, connectionId, "assistant", 
                $"Echo: {content}");
        }
        
        private bool IsEscalationRequest(string content)
        {
            // Simple check for keywords that might indicate a user wants to talk to a human
            var lowerContent = content.ToLower();
            var escalationKeywords = new[] { "agent", "human", "person", "help", "support", "speak to someone", "talk to someone" };
            
            return escalationKeywords.Any(keyword => lowerContent.Contains(keyword));
        }
    }
} 