using System.Text.RegularExpressions;
using System.Text.Json;
using System.Net.Http;

namespace ChatService.Services
{
    public class ChatService
    {
        private readonly ConnectionManager _connectionManager;
        private readonly ILogger<ChatService> _logger;
        private readonly HttpClient _httpClient;
        private readonly string _backendUrl = "http://localhost:8000";

        public ChatService(ConnectionManager connectionManager, ILogger<ChatService> logger, HttpClient httpClient)
        {
            _connectionManager = connectionManager;
            _logger = logger;
            _httpClient = httpClient;
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
            
            try
            {
                // Forward the message to the Python backend for RAG processing
                var requestData = new
                {
                    query = content,
                    k = 5,
                    index_name = indexName,
                    conversation_history = _connectionManager.GetConversationHistory(indexName, connectionId)
                };
                
                var jsonContent = JsonSerializer.Serialize(requestData);
                var httpContent = new StringContent(jsonContent, System.Text.Encoding.UTF8, "application/json");
                
                // Get the original domain for this connection
                var originalDomain = _connectionManager.GetOriginalDomain(indexName, connectionId);
                
                // Use the original domain if available, otherwise fallback to localhost:3001
                var domain = !string.IsNullOrEmpty(originalDomain) ? originalDomain : "localhost:3001";
                
                _logger.LogInformation($"Sending RAG query to backend: {content} for domain: {domain}");
                
                // Create the request message with headers
                var requestMessage = new HttpRequestMessage(HttpMethod.Post, $"{_backendUrl}/api/rag/query")
                {
                    Content = httpContent
                };
                
                // Add headers to the request message (not content)
                requestMessage.Headers.Add("X-Internal-Service", "chat-service");
                requestMessage.Headers.Add("Origin", $"http://{domain}");
                requestMessage.Headers.Add("Referer", $"http://{domain}/");
                
                var response = await _httpClient.SendAsync(requestMessage);
                
                if (response.IsSuccessStatusCode)
                {
                    var responseContent = await response.Content.ReadAsStringAsync();
                    var ragResponse = JsonSerializer.Deserialize<JsonElement>(responseContent);
                    
                    if (ragResponse.TryGetProperty("answer", out var answerElement))
                    {
                        var aiResponse = answerElement.GetString();
                        if (!string.IsNullOrEmpty(aiResponse))
                        {
                            // SendMessageAsync already adds the message to the conversation history
                            // so we don't need to call AddMessage separately
                            await _connectionManager.SendMessageAsync(indexName, connectionId, "assistant", aiResponse);
                        }
                    }
                    else
                    {
                        _logger.LogError("No 'answer' property found in RAG response");
                        await _connectionManager.SendMessageAsync(indexName, connectionId, "system", 
                            "Sorry, I received an unexpected response format. Please try again.");
                    }
                }
                else
                {
                    _logger.LogError($"Backend RAG query failed: {response.StatusCode}");
                    await _connectionManager.SendMessageAsync(indexName, connectionId, "system", 
                        "Sorry, I'm having trouble processing your request right now. Please try again later.");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError($"Error processing user message: {ex.Message}");
                await _connectionManager.SendMessageAsync(indexName, connectionId, "system", 
                    "Sorry, an error occurred while processing your message. Please try again.");
            }
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