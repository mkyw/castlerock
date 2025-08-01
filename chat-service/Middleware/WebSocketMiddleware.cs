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
        private readonly bool _bypassAuthForTesting = true; // Set to false in production

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
                    
                    // For testing, bypass authentication
                    if (_bypassAuthForTesting)
                    {
                        _logger.LogWarning("Authentication bypassed for testing. DO NOT USE IN PRODUCTION!");
                        await HandleWebSocketConnection(context, indexName);
                        return;
                    }
                    
                    // In production, check authentication
                    if (context.User.Identity?.IsAuthenticated == true)
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
                        _logger.LogInformation("No authentication token provided");
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
            
            _logger.LogInformation($"WebSocket connection established for {indexName}, connection ID: {connectionId}");
            
            // Add to connection manager
            _connectionManager.AddConnection(indexName, connectionId, webSocket, userId);
            
            // Send welcome message and connection ID
            await _connectionManager.SendMessageAsync(indexName, connectionId, "system", 
                $"Welcome to the chat for {indexName}");
            await _connectionManager.SendMessageAsync(indexName, connectionId, "system", 
                $"Your connection ID is {connectionId}", new { connection_id = connectionId });
            
            // Handle the WebSocket connection
            await HandleWebSocketAsync(indexName, connectionId, webSocket);
        }

        private async Task HandleWebSocketAsync(string indexName, string connectionId, WebSocket webSocket)
        {
            var buffer = new byte[1024 * 4];
            WebSocketReceiveResult result = null;

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
                        if (result.MessageType == WebSocketMessageType.Close)
                        {
                            _logger.LogInformation($"WebSocket closing for {connectionId}");
                            await webSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "Closing", CancellationToken.None);
                            _connectionManager.Disconnect(indexName, connectionId);
                            break;
                        }

                        // Process the message
                        if (result.MessageType == WebSocketMessageType.Text)
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
                                            if (jsonMessage.TryGetProperty("agent_id", out var agentIdElement))
                                            {
                                                var agentId = agentIdElement.GetString();
                                                if (!string.IsNullOrEmpty(agentId))
                                                {
                                                    _connectionManager.AssignAgent(indexName, connectionId, agentId);
                                                    await _connectionManager.SendMessageAsync(indexName, connectionId, "system", 
                                                        $"Agent {agentId} has joined the chat.");
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
                                                    await _connectionManager.SendMessageAsync(indexName, connectionId, "agent", content);
                                                }
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
                                        // Add the message to the history
                                        var chatMessage = new ChatMessage
                                        {
                                            Role = "user",
                                            Content = content,
                                            Timestamp = DateTime.UtcNow.ToString("o")
                                        };
                                        _connectionManager.AddMessage(indexName, connectionId, chatMessage);
                                        
                                        // Process the message
                                        await _chatService.ProcessUserMessage(indexName, connectionId, content);
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