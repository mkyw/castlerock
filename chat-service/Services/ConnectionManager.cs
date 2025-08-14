using System.Collections.Concurrent;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;

namespace ChatService.Services
{
    public class ConnectionManager
    {
        // Main structure to track all active connections
        private readonly ConcurrentDictionary<string, ChatConnection> _activeConnections = new();
        
        // Track connections that need agent attention
        private readonly ConcurrentDictionary<string, bool> _escalationRequests = new();
        
        // Track connections that have been assigned to agents
        private readonly ConcurrentDictionary<string, string> _agentAssigned = new();
        
        // Map agent connection IDs to user connection IDs
        private readonly ConcurrentDictionary<string, string> _agentToUserConnectionMap = new();

        private readonly ILogger<ConnectionManager> _logger;
        
        // Track the last cleanup time to avoid logging too frequently
        private DateTime _lastCleanupLogTime = DateTime.UtcNow;

        public ConnectionManager(ILogger<ConnectionManager> logger)
        {
            _logger = logger;
        }

        public void AddConnection(string indexName, string connectionId, WebSocket socket, string userId = "anonymous", string? originalDomain = null)
        {
            _logger.LogInformation($"Adding connection {connectionId} for index {indexName}");
            
            var connection = new ChatConnection
            {
                Socket = socket,
                IndexName = indexName,
                ConnectionId = connectionId,
                UserId = userId,
                Status = "active",
                ConnectedAt = DateTime.UtcNow,
                Messages = new List<ChatMessage>(),
                OriginalDomain = originalDomain
            };
            
            _activeConnections.TryAdd(connectionId, connection);
            _logger.LogInformation($"Connection {connectionId} added successfully");
        }
        
        public void Disconnect(string indexName, string connectionId)
        {
            _logger.LogInformation($"Disconnecting {connectionId} from index {indexName}");
            
            if (_activeConnections.TryRemove(connectionId, out var connection))
            {
                // Remove from escalation requests if present
                _escalationRequests.TryRemove(connectionId, out _);
                
                // Remove from agent assignments if present
                _agentAssigned.TryRemove(connectionId, out _);
                
                _logger.LogInformation($"Connection {connectionId} removed successfully");
            }
        }
        
        public async Task SendMessageAsync(string indexName, string connectionId, string role, string content, object? metadata = null)
        {
            _logger.LogInformation($"Attempting to send {role} message to connection {connectionId} in index {indexName}");
            
            if (_activeConnections.TryGetValue(connectionId, out var connection))
            {
                _logger.LogInformation($"Found connection {connectionId}, state: {connection.Socket.State}, index: {connection.IndexName}");
                
                if (connection.Socket.State == WebSocketState.Open)
                {
                    var message = new ChatMessage
                    {
                        Role = role,
                        Content = content,
                        Timestamp = DateTime.UtcNow.ToString("o")
                    };
                    
                    // Add message to history
                    connection.Messages.Add(message);
                    
                    // Create response object
                    var responseObj = new Dictionary<string, object>
                    {
                        { "type", role },
                        { "content", content },
                        { "timestamp", message.Timestamp }
                    };
                    
                    // Add any additional metadata
                    if (metadata != null)
                    {
                        _logger.LogInformation($"Adding metadata to message: {JsonSerializer.Serialize(metadata)}");
                        var metadataProps = metadata.GetType().GetProperties();
                        foreach (var prop in metadataProps)
                        {
                            responseObj[prop.Name] = prop.GetValue(metadata) ?? string.Empty;
                        }
                    }
                    
                    var responseJson = JsonSerializer.Serialize(responseObj);
                    _logger.LogInformation($"Sending message payload: {responseJson}");
                    var buffer = Encoding.UTF8.GetBytes(responseJson);
                    
                    try
                    {
                        await connection.Socket.SendAsync(
                            new ArraySegment<byte>(buffer),
                            WebSocketMessageType.Text,
                            true,
                            CancellationToken.None);
                        
                        _logger.LogInformation($"Message successfully sent to {connectionId}: {role}/{content.Substring(0, Math.Min(content.Length, 50))}...");
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError($"Error sending message to {connectionId}: {ex.Message}");
                    }
                }
                else
                {
                    _logger.LogWarning($"Cannot send message to {connectionId}: Connection is in state {connection.Socket.State}");
                }
            }
            else
            {
                _logger.LogWarning($"Cannot send message to {connectionId}: Connection not found");
            }
        }
        
        public bool RequestEscalation(string indexName, string connectionId)
        {
            if (_activeConnections.TryGetValue(connectionId, out var connection))
            {
                connection.Status = "escalation_requested";
                connection.EscalationRequestedAt = DateTime.UtcNow;
                
                _escalationRequests.TryAdd(connectionId, true);
                
                _logger.LogInformation($"Escalation requested for {connectionId}");
                return true;
            }
            
            return false;
        }
        
        public bool AssignAgent(string indexName, string connectionId, string agentId)
        {
            if (_activeConnections.TryGetValue(connectionId, out var connection))
            {
                connection.Status = "agent_assigned";
                connection.AgentId = agentId;
                connection.AgentAssignedAt = DateTime.UtcNow;
                
                // Remove from escalation requests if present
                _escalationRequests.TryRemove(connectionId, out _);
                
                // Add to agent assignments
                _agentAssigned.TryAdd(connectionId, agentId);
                
                _logger.LogInformation($"Agent {agentId} assigned to {connectionId}");
                return true;
            }
            
            return false;
        }
        
        public Dictionary<string, object> GetConnectionsByStatus(string? status = null)
        {
            var result = new Dictionary<string, object>();
            var connections = new List<Dictionary<string, object>>();
            
            foreach (var kvp in _activeConnections)
            {
                var connection = kvp.Value;
                
                // Filter by status if provided
                if (status != null && connection.Status != status)
                {
                    continue;
                }
                
                // Count user messages
                int userMessageCount = connection.Messages.Count(m => m.Role == "user");
                
                // Skip connections with no user messages
                if (userMessageCount == 0)
                {
                    continue;
                }
                
                var connectionInfo = new Dictionary<string, object>
                {
                    { "connection_id", connection.ConnectionId },
                    { "index_name", connection.IndexName },
                    { "status", connection.Status },
                    { "user_id", connection.UserId },
                    { "connected_at", connection.ConnectedAt.ToString("o") },
                    { "message_count", userMessageCount }
                };
                
                if (connection.AgentId != null)
                {
                    connectionInfo["agent_id"] = connection.AgentId;
                    connectionInfo["agent_assigned_at"] = connection.AgentAssignedAt?.ToString("o") ?? "";
                }
                
                if (connection.EscalationRequestedAt.HasValue)
                {
                    connectionInfo["escalation_requested_at"] = connection.EscalationRequestedAt.Value.ToString("o");
                }
                
                connections.Add(connectionInfo);
            }
            
            result["connections"] = connections;
            result["count"] = connections.Count;
            
            return result;
        }
        
        public object GetStats()
        {
            var now = DateTime.UtcNow;
            var lastHour = now.AddHours(-1);
            var lastDay = now.AddDays(-1);
            
            var activeConnections = _activeConnections.Count;
            var escalationRequests = _escalationRequests.Count;
            var agentAssigned = _agentAssigned.Count;
            
            var connectionsLastHour = _activeConnections.Values
                .Count(c => c.ConnectedAt >= lastHour);
            
            var connectionsLastDay = _activeConnections.Values
                .Count(c => c.ConnectedAt >= lastDay);
            
            var stats = new Dictionary<string, object>
            {
                { "total_active", activeConnections },
                { "escalation_requested", escalationRequests },
                { "agent_assigned", agentAssigned },
                { "connections_last_hour", connectionsLastHour },
                { "connections_last_day", connectionsLastDay },
                { "timestamp", now.ToString("o") }
            };
            
            return stats;
        }
        
        public object? GetChatHistory(string indexName, string connectionId)
        {
            if (_activeConnections.TryGetValue(connectionId, out var connection) && 
                connection.IndexName == indexName)
            {
                var messages = connection.Messages.Select(m => {
                    // On agent end, keep user as user (no special labels)
                    // No sender labels needed on user end
                    return new Dictionary<string, string>
                    {
                        { "role", m.Role },
                        { "content", m.Content },
                        { "timestamp", m.Timestamp }
                    };
                }).ToList();
                
                return new Dictionary<string, object>
                {
                    { "connection_id", connectionId },
                    { "index_name", indexName },
                    { "status", connection.Status },
                    { "connected_at", connection.ConnectedAt.ToString("o") },
                    { "messages", messages },
                    { "agent_id", connection.AgentId ?? "" }
                };
            }
            
            return null;
        }
        
        public List<Dictionary<string, string>> GetConversationHistory(string indexName, string connectionId)
        {
            if (_activeConnections.TryGetValue(connectionId, out var connection) && 
                connection.IndexName == indexName)
            {
                return connection.Messages.Select(m => new Dictionary<string, string>
                {
                    { "role", m.Role },
                    { "content", m.Content }
                }).ToList();
            }
            
            return new List<Dictionary<string, string>>();
        }
        
        public string? GetOriginalDomain(string indexName, string connectionId)
        {
            if (_activeConnections.TryGetValue(connectionId, out var connection) && 
                connection.IndexName == indexName)
            {
                return connection.OriginalDomain;
            }
            
            return null;
        }
        
        public bool ChatExists(string indexName, string connectionId)
        {
            return _activeConnections.TryGetValue(connectionId, out var connection) && 
                   connection.IndexName == indexName;
        }
        
        public bool IsAgentAssigned(string indexName, string connectionId)
        {
            // Check if the connection exists and has an agent assigned
            if (_activeConnections.TryGetValue(connectionId, out var connection) && 
                connection.IndexName == indexName)
            {
                return !string.IsNullOrEmpty(connection.AgentId) && connection.AgentAssignedAt.HasValue;
            }
            
            return false;
        }
        
        // Check if a specific agent is assigned to a specific user
        public bool IsAgentAssignedToUser(string indexName, string userConnectionId, string agentId)
        {
            // Check if the connection exists and has this specific agent assigned
            if (_activeConnections.TryGetValue(userConnectionId, out var connection) && 
                connection.IndexName == indexName)
            {
                return connection.AgentId == agentId && connection.AgentAssignedAt.HasValue;
            }
            
            return false;
        }
        
        public void MapAgentToUserConnection(string agentConnectionId, string userConnectionId)
        {
            _logger.LogInformation($"Mapping agent connection {agentConnectionId} to user connection {userConnectionId}");
            
            // First, check if there are any existing agent connections for this user
            var existingAgentConnections = GetAgentConnectionsForUser(userConnectionId);
            
            // If there are existing connections, remove them to enforce single agent connection per chat
            if (existingAgentConnections.Count > 0)
            {
                _logger.LogWarning($"Found {existingAgentConnections.Count} existing agent connections for user {userConnectionId}. Removing them to enforce single agent connection.");
                
                foreach (var existingConnection in existingAgentConnections)
                {
                    _logger.LogInformation($"Removing existing agent connection mapping: {existingConnection} -> {userConnectionId}");
                    _agentToUserConnectionMap.TryRemove(existingConnection, out _);
                }
            }
            
            // Now map the new agent connection
            _agentToUserConnectionMap[agentConnectionId] = userConnectionId;
            _logger.LogInformation($"Successfully mapped agent connection {agentConnectionId} to user {userConnectionId}");
        }
        
        public string GetUserConnectionId(string agentConnectionId)
        {
            if (_agentToUserConnectionMap.TryGetValue(agentConnectionId, out var userConnectionId))
            {
                return userConnectionId;
            }
            
            return string.Empty;
        }
        
        public string GetAssignedAgentId(string indexName, string connectionId)
        {
            _logger.LogInformation($"Getting assigned agent for connection {connectionId} in index {indexName}");
            
            if (_activeConnections.TryGetValue(connectionId, out var connection) && 
                connection.IndexName == indexName)
            {
                return connection.AgentId ?? string.Empty;
            }
            
            return string.Empty;
        }
        
        public List<string> GetAgentConnectionsForUser(string userConnectionId)
        {
            _logger.LogInformation($"Finding agent connections for user {userConnectionId}");
            
            var agentConnections = new List<string>();
            
            // Find all agent connections mapped to this user
            foreach (var kvp in _agentToUserConnectionMap)
            {
                if (kvp.Value == userConnectionId)
                {
                    agentConnections.Add(kvp.Key);
                }
            }
            
            _logger.LogInformation($"Found {agentConnections.Count} agent connections for user {userConnectionId}");
            return agentConnections;
        }
        
        public void AddMessage(string indexName, string connectionId, ChatMessage message)
        {
            if (_activeConnections.TryGetValue(connectionId, out var connection) && 
                connection.IndexName == indexName)
            {
                connection.Messages.Add(message);
                connection.LastActivity = DateTime.UtcNow;
            }
        }
        
        /// <summary>
        /// Cleans up inactive connections that haven't had activity for the specified threshold
        /// </summary>
        /// <param name="inactivityThreshold">The time threshold after which a connection is considered inactive</param>
        /// <returns>The number of connections that were removed</returns>
        public int CleanupInactiveConnections(TimeSpan inactivityThreshold)
        {
            int removedCount = 0;
            var now = DateTime.UtcNow;
            var cutoffTime = now - inactivityThreshold;
            var connectionsToRemove = new List<string>();
            
            // Find inactive connections
            foreach (var kvp in _activeConnections)
            {
                var connectionId = kvp.Key;
                var connection = kvp.Value;
                
                // Check if the connection is inactive
                if (connection.LastActivity < cutoffTime)
                {
                    connectionsToRemove.Add(connectionId);
                }
            }
            
            // Remove inactive connections
            foreach (var connectionId in connectionsToRemove)
            {
                if (_activeConnections.TryRemove(connectionId, out var connection))
                {
                    // Remove from escalation requests if present
                    _escalationRequests.TryRemove(connectionId, out _);
                    
                    // Remove from agent assignments if present
                    _agentAssigned.TryRemove(connectionId, out _);
                    
                    removedCount++;
                }
            }
            
            // Log cleanup results, but not too frequently
            if (removedCount > 0 || (now - _lastCleanupLogTime).TotalMinutes >= 5)
            {
                _logger.LogInformation($"Cleaned up {removedCount} inactive connections. Current active: {_activeConnections.Count}");
                _lastCleanupLogTime = now;
            }
            
            return removedCount;
        }
    }
    
    public class ChatConnection
    {
        public WebSocket Socket { get; set; } = null!;
        public string ConnectionId { get; set; } = string.Empty;
        public string IndexName { get; set; } = string.Empty;
        public string UserId { get; set; } = "anonymous";
        public string Status { get; set; } = "active";
        public List<ChatMessage> Messages { get; set; } = new();
        public DateTime ConnectedAt { get; set; } = DateTime.UtcNow;
        public DateTime LastActivity { get; set; } = DateTime.UtcNow;
        public string? AgentId { get; set; }
        public DateTime? AgentAssignedAt { get; set; }
        public DateTime? EscalationRequestedAt { get; set; }
        public string? OriginalDomain { get; set; }  // Track the original domain
    }
    
    public class ChatMessage
    {
        public string Role { get; set; } = string.Empty;
        public string Content { get; set; } = string.Empty;
        public string Timestamp { get; set; } = DateTime.UtcNow.ToString("o");
    }
} 