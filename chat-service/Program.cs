using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.IdentityModel.Tokens;
using System.Text;
using ChatService.Services;
using ChatService.Middleware;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;

var builder = WebApplication.CreateBuilder(args);

// Add services
builder.Services.AddLogging();
builder.Services.AddHttpClient();
builder.Services.AddSingleton<ConnectionManager>();
builder.Services.AddSingleton<ChatService.Services.ChatService>();
builder.Services.AddHostedService<ConnectionCleanupService>();
builder.Services.AddAuthorization();

// Add CORS
builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(policy =>
    {
        policy.AllowAnyOrigin()
              .AllowAnyHeader()
              .AllowAnyMethod();
    });
});

// Configure JWT authentication
var jwtSecret = builder.Configuration["JWT_SECRET"] ?? "your_default_secret_key_here_minimum_16_chars";
var key = Encoding.ASCII.GetBytes(jwtSecret);

builder.Services.AddAuthentication(options =>
{
    options.DefaultAuthenticateScheme = JwtBearerDefaults.AuthenticationScheme;
    options.DefaultChallengeScheme = JwtBearerDefaults.AuthenticationScheme;
})
.AddJwtBearer(options =>
{
    options.RequireHttpsMetadata = false;
    options.SaveToken = true;
    options.TokenValidationParameters = new TokenValidationParameters
    {
        ValidateIssuerSigningKey = true,
        IssuerSigningKey = new SymmetricSecurityKey(key),
        ValidateIssuer = false,
        ValidateAudience = false
    };
    
    // Special handling for WebSockets
    options.Events = new JwtBearerEvents
    {
        OnMessageReceived = context =>
        {
            // For WebSocket connections, the token is passed as a query parameter
            if (context.Request.Path.StartsWithSegments("/ws") && 
                context.Request.Query.TryGetValue("token", out var token))
            {
                context.Token = token;
            }
            return Task.CompletedTask;
        }
    };
});

// WebSockets are handled by middleware

var app = builder.Build();

// Configure the HTTP request pipeline
app.UseWebSockets();
app.UseCors();

// Authentication and authorization enabled
app.UseAuthentication();
app.UseAuthorization();

// Add WebSocket middleware
app.UseMiddleware<WebSocketMiddleware>();

// API endpoints
app.MapGet("/api/chat/stats", (ConnectionManager connectionManager) =>
{
    return Results.Ok(connectionManager.GetStats());
});

app.MapGet("/api/chat/active", (ConnectionManager connectionManager) =>
{
    var connections = connectionManager.GetConnectionsByStatus();
    return Results.Ok(connections);
});

app.MapPost("/api/chat/assign-agent", (dynamic request, ConnectionManager connectionManager) =>
{
    if (string.IsNullOrEmpty(request.ConnectionId) || 
        string.IsNullOrEmpty(request.AgentId) || 
        string.IsNullOrEmpty(request.IndexName))
    {
        return Results.BadRequest("ConnectionId, AgentId, and IndexName are required");
    }
    
    var success = connectionManager.AssignAgent(request.IndexName, request.ConnectionId, request.AgentId);
    
    if (!success)
    {
        return Results.NotFound($"Connection {request.ConnectionId} not found");
    }
    
    return Results.Ok(new { success = true });
});

app.MapGet("/api/chat/history/{indexName}/{connectionId}", (string indexName, string connectionId, ConnectionManager connectionManager) =>
{
    var history = connectionManager.GetChatHistory(indexName, connectionId);
    
    if (history == null)
    {
        return Results.NotFound($"Chat history for {connectionId} not found");
    }
    
    return Results.Ok(history);
});

app.MapPost("/api/chat/end/{indexName}/{connectionId}", (string indexName, string connectionId, ConnectionManager connectionManager, ILogger<Program> logger) =>
{
    logger.LogInformation($"Attempting to end chat for index {indexName}, connection {connectionId}");
    
    // Check if this is an agent connection ID
    string userConnectionId = connectionManager.GetUserConnectionId(connectionId);
    if (!string.IsNullOrEmpty(userConnectionId))
    {
        logger.LogInformation($"Connection {connectionId} is an agent connection for user {userConnectionId}, using user connection ID");
        connectionId = userConnectionId;
    }
    
    if (!connectionManager.ChatExists(indexName, connectionId))
    {
        logger.LogWarning($"Chat {connectionId} not found for index {indexName}");
        return Results.NotFound($"Chat {connectionId} not found");
    }
    
    logger.LogInformation($"Ending chat for connection {connectionId}");
    connectionManager.Disconnect(indexName, connectionId);
    return Results.Ok(new { success = true });
});

Console.WriteLine("Starting ChatService on http://localhost:5001");
app.Run("http://0.0.0.0:5001"); 