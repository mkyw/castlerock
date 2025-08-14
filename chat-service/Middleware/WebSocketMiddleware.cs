using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using ChatService.Services;

namespace ChatService.Middleware
{
    public class WebSocketMiddleware
    {
        private readonly RequestDelegate _next;
        private readonly ConnectionManager _connectionManager;
        private readonly ChatService.Services.ChatService _chatService;
        private readonly ILogger<WebSocketMiddleware> _logger;

        public WebSocketMiddleware(
            RequestDelegate next,
            ConnectionManager connectionManager,
            ChatService.Services.ChatService chatService,
            ILogger<WebSocketMiddleware> logger)
        {
            _next = next;
            _connectionManager = connectionManager;
            _chatService = chatService;
            _logger = logger;
        }

        public async Task InvokeAsync(HttpContext context)
        {
            if (context.WebSockets.IsWebSocketRequest)
            {
                var pathValue = context.Request.Path.Value ?? "/";
                var pathSegments = pathValue.Split('/');
                
                // Check if this is a chat WebSocket request
                if (pathSegments.Length >= 4 && pathSegments[1] == "ws" && pathSegments[2] == "chat")
                {
                    string indexName = pathSegments[3];
                    _logger.LogInformation($"WebSocket connection request for index: {indexName}");
                    
                    // Get the origin header
                    string? origin = context.Request.Headers["Origin"].FirstOrDefault() ?? 
                                   context.Request.Headers["Referer"].FirstOrDefault();
                    
                    // Extract domain from origin for validation and index selection
                    string? originHost = null;
                    bool isTrustedOrigin = false;
                    
                    if (!string.IsNullOrEmpty(origin))
                    {
                        try {
                            var uri = new Uri(origin);
                            originHost = uri.Host;
                            
                            // TODO: Move this to configuration or database lookup
                            // List of trusted origins with their corresponding index names
                            var trustedOrigins = new Dictionary<string, string> {
                                { "localhost", "localhost" },
                                { "127.0.0.1", "localhost" },
                                // Add production domains here, e.g.
                                // { "example.com", "example-com" }
                            };
                            
                            // Check if origin is in whitelist
                            if (trustedOrigins.TryGetValue(originHost, out var mappedIndex))
                            {
                                isTrustedOrigin = true;
                                // Override the index name with the one from the whitelist mapping
                                // This ensures we use the correct index even if the URL specifies something else
                                indexName = mappedIndex;
                            }
                            
                            _logger.LogInformation($"Origin check: {originHost}, trusted: {isTrustedOrigin}, using index: {indexName}");
                        }
                        catch (Exception ex) {
                            _logger.LogError($"Error parsing origin: {ex.Message}");
                        }
                    }
                    
                    // Allow access if authenticated or from trusted origin
                    if (context.User.Identity?.IsAuthenticated == true || isTrustedOrigin)
                    {
                        await HandleWebSocketConnection(context, indexName);
                        return;
                    }
                    else if (context.Request.Query.TryGetValue("token", out var token))
                    {
                        _logger.LogInformation("Token provided in query string, but authentication failed");
                        // Authentication is handled by JwtBearer middleware
                        // If we reach here, the token was invalid
                        context.Response.StatusCode = 401;
                        await context.Response.WriteAsync("Invalid authentication token");
                        return;
                    }
                    else
                    {
                        _logger.LogInformation("No authentication token provided and not from trusted origin");
                        context.Response.StatusCode = 401;
                        await context.Response.WriteAsync("Authentication required");
                        return;
                    }
                }
            }
            
            await _next(context);
        }
        
        private async Task HandleWebSocketConnection(HttpContext context, string indexName)
        {
            // Accept the WebSocket connection
            var webSocket = await context.WebSockets.AcceptWebSocketAsync();
            
            // Generate connection ID
            string connectionId = Guid.NewGuid().ToString();
            string userId = context.User?.Identity?.Name ?? "anonymous";
            
            // Extract original domain from headers
            string? originalDomain = context.Request.Headers["Origin"].FirstOrDefault() ?? 
                                   context.Request.Headers["Referer"].FirstOrDefault();
            
            // Parse domain from URL if needed
            if (!string.IsNullOrEmpty(originalDomain))
            {
                try
                {
                    var uri = new Uri(originalDomain);
                    originalDomain = uri.Host + (uri.Port != 80 && uri.Port != 443 ? $":{uri.Port}" : "");
                }
                catch
                {
                    // If parsing fails, use as-is
                }
            }
            
            _logger.LogInformation($"WebSocket connection established for {indexName}, connection ID: {connectionId}, domain: {originalDomain}");
            
            // Add to connection manager
            _connectionManager.AddConnection(indexName, connectionId, webSocket, userId, originalDomain);
            
            // Not sending system message here as it's already handled by the frontend
            // in chatbot-widget-new.js (this.addSystemMessage('Connected to chat server'))
            
            // Handle the WebSocket connection
            await HandleWebSocketAsync(indexName, connectionId, webSocket);
        }

        private async Task HandleWebSocketAsync(string indexName, string connectionId, WebSocket webSocket)
        {
            var buffer = new byte[1024 * 4];
            WebSocketReceiveResult? result = null;

            try
            {
                while (webSocket.State == WebSocketState.Open)
                {
                    try
                    {
                        // Receive message
                        using var ms = new MemoryStream();
                        do
                        {
                            result = await webSocket.ReceiveAsync(new ArraySegment<byte>(buffer), CancellationToken.None);
                            await ms.WriteAsync(buffer, 0, result.Count);
                        }
                        while (!result.EndOfMessage);

                        // Reset position to read
                        ms.Seek(0, SeekOrigin.Begin);

                        // Check if the connection is closing
                        if (result?.MessageType == WebSocketMessageType.Close)
                        {
                            _logger.LogInformation($"WebSocket closing for {connectionId}");
                            await webSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "Closing", CancellationToken.None);
                            _connectionManager.Disconnect(indexName, connectionId);
                            break;
                        }

                        // Process the message
                        if (result?.MessageType == WebSocketMessageType.Text)
                        {
                            using var reader = new StreamReader(ms, Encoding.UTF8);
                            var message = await reader.ReadToEndAsync();
                            _logger.LogInformation($"Received message from {connectionId}: {message}");

                            try
                            {
                                // Parse the message
                                var jsonMessage = JsonSerializer.Deserialize<JsonElement>(message);
                                
                                // Check if it's an agent command
                                if (jsonMessage.TryGetProperty("type", out var typeElement) && 
                                    typeElement.GetString() == "agent_command")
                                {
                                    if (jsonMessage.TryGetProperty("command", out var commandElement))
                                    {
                                        var command = commandElement.GetString();
                                        
                                        if (command == "take_over" || command == "join")
                                        {
                                            // Agent is taking over or joining the chat
                                            if (jsonMessage.TryGetProperty("agent_id", out var agentIdElement) &&
                                                jsonMessage.TryGetProperty("connection_id", out var userConnectionIdElement))
                                            {
                                                var agentId = agentIdElement.GetString();
                                                var userConnectionId = userConnectionIdElement.GetString();
                                                
                                                if (!string.IsNullOrEmpty(agentId) && !string.IsNullOrEmpty(userConnectionId))
                                                {
                                                    _logger.LogInformation($"Agent {agentId} joining chat for user connection {userConnectionId}");
                                                    
                                                    // Check if this agent is already assigned to this user to prevent duplicate joins
                                                    var isAlreadyAssigned = _connectionManager.IsAgentAssignedToUser(indexName, userConnectionId, agentId);
                                                    
                                                    // Also check if this specific agent connection is already mapped to this user
                                                    var existingUserConnection = _connectionManager.GetUserConnectionId(connectionId);
                                                    var isAlreadyMapped = !string.IsNullOrEmpty(existingUserConnection) && existingUserConnection == userConnectionId;
                                                    
                                                    _logger.LogInformation($"Agent join check: isAlreadyAssigned={isAlreadyAssigned}, isAlreadyMapped={isAlreadyMapped}");
                                                    
                                                    if (!isAlreadyAssigned && !isAlreadyMapped)
                                                    {
                                                        _logger.LogInformation($"Agent {agentId} is not yet assigned to user {userConnectionId}, assigning now");
                                                        
                                                        // Store the agent's WebSocket connection ID mapped to the user's connection ID
                                                        _connectionManager.MapAgentToUserConnection(connectionId, userConnectionId);
                                                        
                                                        // Assign the agent to the user's connection
                                                        _connectionManager.AssignAgent(indexName, userConnectionId, agentId);
                                                        
                                                        // No need to send a system message to the user when an agent joins
                                                        _logger.LogInformation($"Agent {agentId} has joined the chat for user {userConnectionId}");
                                                        
                                                        // Send a system message to the agent indicating they're connected to the user
                                                        await _connectionManager.SendMessageAsync(indexName, connectionId, "system", 
                                                            $"Connected to user {userConnectionId}");
                                                    }
                                                    else
                                                    {
                                                        _logger.LogWarning($"Agent {agentId} is already assigned to user {userConnectionId} or connection is already mapped, ignoring duplicate join request");
                                                        
                                                        // Update the agent connection ID mapping in case the agent reconnected
                                                        _connectionManager.MapAgentToUserConnection(connectionId, userConnectionId);
                                                    }
                                                }
                                            }
                                        }
                                        else if (command == "message")
                                        {
                                            // Agent is sending a message
                                            if (jsonMessage.TryGetProperty("content", out var contentElement) &&
                                                jsonMessage.TryGetProperty("agent_id", out var agentIdElement))
                                            {
                                                var content = contentElement.GetString();
                                                var agentId = agentIdElement.GetString();
                                                
                                                if (!string.IsNullOrEmpty(content) && !string.IsNullOrEmpty(agentId))
                                                {
                                                    // Get the user connection ID this agent is mapped to
                                                    var userConnectionId = _connectionManager.GetUserConnectionId(connectionId);
                                                    
                                                    if (!string.IsNullOrEmpty(userConnectionId))
                                                    {
                                                        // Add the message to the user's chat history
                                                        var chatMessage = new ChatMessage
                                                        {
                                                            Role = "agent",
                                                            Content = content,
                                                            Timestamp = DateTime.UtcNow.ToString("o")
                                                        };
                                                        _connectionManager.AddMessage(indexName, userConnectionId, chatMessage);
                                                        
                                                        // Send the message to the user - use 'agent' type for proper identification
                                                        // This ensures the frontend can distinguish between AI and human agent messages
                                                        await _connectionManager.SendMessageAsync(indexName, userConnectionId, "agent", content, new { agent_id = agentId });
                                                        
                                                        // Also send the message back to the agent so they can see their own messages
                                                        await _connectionManager.SendMessageAsync(indexName, connectionId, "agent", content, new { agent_id = agentId });
                                                    }
                                                    else
                                                    {
                                                        _logger.LogWarning($"User connection not found for agent {agentId}, cannot deliver agent message");
                                                    }
                                                }
                                            }
                                        }
                                        else if (command == "end_chat")
                                        {
                                            // Agent is ending the chat
                                            _logger.LogInformation($"Agent {connectionId} is ending the chat");
                                            
                                            // Get the user connection ID this agent is mapped to
                                            var userConnectionId = _connectionManager.GetUserConnectionId(connectionId);
                                            
                                            if (!string.IsNullOrEmpty(userConnectionId))
                                            {
                                                _logger.LogInformation($"Ending chat for user connection {userConnectionId}");
                                                
                                                // Send a system message to the user that the chat has ended
                                                await _connectionManager.SendMessageAsync(indexName, userConnectionId, "system", 
                                                    "The agent has ended this chat. Thank you for your conversation.");
                                                
                                                // Disconnect the user connection
                                                _connectionManager.Disconnect(indexName, userConnectionId);
                                                
                                                // Also disconnect the agent connection
                                                _connectionManager.Disconnect(indexName, connectionId);
                                                
                                                _logger.LogInformation($"Chat ended successfully for user {userConnectionId} and agent {connectionId}");
                                            }
                                            else
                                            {
                                                _logger.LogWarning($"User connection not found for agent {connectionId}, cannot end chat properly");
                                            }
                                        }
                                    }
                                }
                                else if (jsonMessage.TryGetProperty("type", out var msgTypeElement) && 
                                         msgTypeElement.GetString() == "user" &&
                                         jsonMessage.TryGetProperty("content", out var contentElement))
                                {
                                    // Regular user message
                                    var content = contentElement.GetString();
                                    if (!string.IsNullOrEmpty(content))
                                    {
                                        // Add the message to chat history
                                        var chatMessage = new ChatMessage
                                        {
                                            Role = "user",
                                            Content = content,
                                            Timestamp = DateTime.UtcNow.ToString("o")
                                        };
                                        _connectionManager.AddMessage(indexName, connectionId, chatMessage);
                                        
                                        // Check if an agent is assigned to this connection
                                        if (_connectionManager.IsAgentAssigned(indexName, connectionId))
                                        {
                                            // No need to process through RAG backend
                                            _logger.LogInformation($"Agent assigned to {connectionId}, forwarding message directly");
                                            
                                            // The message is already added to history above
                                            // Get the agent ID assigned to this connection
                                            var agentId = _connectionManager.GetAssignedAgentId(indexName, connectionId);
                                            _logger.LogInformation($"Found assigned agent {agentId} for connection {connectionId}");
                                            
                                            // Find all agent connections that are handling this user
                                            var agentConnections = _connectionManager.GetAgentConnectionsForUser(connectionId);
                                            
                                            if (agentConnections.Any())
                                            {
                                                _logger.LogInformation($"Found {agentConnections.Count} agent connections for user {connectionId}");
                                                
                                                // Check if the current connection is an agent connection to avoid duplication
                                                bool isAgentConnection = agentConnections.Contains(connectionId);
                                                
                                                if (!isAgentConnection)
                                                {
                                                    // Forward the message to only one agent connection to prevent duplication
                                                    // We'll use the first agent connection in the list
                                                    if (agentConnections.Count > 0)
                                                    {
                                                        var primaryAgentConnection = agentConnections[0];
                                                        _logger.LogInformation($"Forwarding user message to primary agent connection {primaryAgentConnection}");
                                                        await _connectionManager.SendMessageAsync(indexName, primaryAgentConnection, "user", content);
                                                        
                                                        // Log other agent connections that we're not forwarding to
                                                        if (agentConnections.Count > 1)
                                                        {
                                                            _logger.LogInformation($"Not forwarding to {agentConnections.Count - 1} other agent connections to prevent duplication");
                                                        }
                                                    }
                                                }
                                                else
                                                {
                                                    _logger.LogInformation($"Message originated from agent connection {connectionId}, not forwarding to avoid duplication");
                                                }
                                            }
                                            else
                                            {
                                                _logger.LogWarning($"No agent connections found for user {connectionId} despite agent {agentId} being assigned");
                                            }
                                        }
                                        else
                                        {
                                            // No agent assigned, process through RAG backend as usual
                                            await _chatService.ProcessUserMessage(indexName, connectionId, content);
                                        }
                                    }
                                }
                            }
                            catch (JsonException ex)
                            {
                                _logger.LogError($"Error parsing message from {connectionId}: {ex.Message}");
                                await _connectionManager.SendMessageAsync(indexName, connectionId, "system", 
                                    "Sorry, I couldn't understand that message. Please try again.");
                            }
                        }
                    }
                    catch (WebSocketException ex) when (ex.WebSocketErrorCode == WebSocketError.ConnectionClosedPrematurely)
                    {
                        _logger.LogInformation($"WebSocket connection closed prematurely for {connectionId}");
                        _connectionManager.Disconnect(indexName, connectionId);
                        break;
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError($"Error handling WebSocket message for {connectionId}: {ex.Message}");
                        try
                        {
                            await _connectionManager.SendMessageAsync(indexName, connectionId, "system", 
                                "Sorry, an error occurred while processing your message.");
                        }
                        catch
                        {
                            // Ignore errors when sending error messages
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError($"WebSocket error for {connectionId}: {ex.Message}");
            }
            finally
            {
                // Ensure connection is removed from manager
                _connectionManager.Disconnect(indexName, connectionId);
                
                // Attempt to close the WebSocket if it's still open
                if (webSocket.State == WebSocketState.Open)
                {
                    try
                    {
                        await webSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "Closing", CancellationToken.None);
                    }
                    catch
                    {
                        // Ignore errors when closing
                    }
                }
            }
        }
    }
} 