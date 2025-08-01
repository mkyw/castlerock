using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.IdentityModel.Tokens;
using System.Text;
using ChatService.Services;
using ChatService.Middleware;
using System.Text.Json;

// Model for agent assignment
public class AssignAgentRequest
{
    public string ConnectionId { get; set; } = string.Empty;
    public string AgentId { get; set; } = string.Empty;
    public string IndexName { get; set; } = string.Empty;
}

var builder = WebApplication.CreateBuilder(args);

// Add services
builder.Services.AddLogging();
builder.Services.AddSingleton<ConnectionManager>();
builder.Services.AddSingleton<ChatService.Services.ChatService>();
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

// Add WebSockets
builder.Services.AddWebSockets(options =>
{
    options.KeepAliveInterval = TimeSpan.FromMinutes(2);
});

var app = builder.Build();

// Configure the HTTP request pipeline
app.UseWebSockets();
app.UseCors();

// Comment out authentication for testing
// app.UseAuthentication();
// app.UseAuthorization();

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

app.MapPost("/api/chat/assign-agent", (AssignAgentRequest request, ConnectionManager connectionManager) =>
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

app.MapPost("/api/chat/end/{indexName}/{connectionId}", (string indexName, string connectionId, ConnectionManager connectionManager) =>
{
    if (!connectionManager.ChatExists(indexName, connectionId))
    {
        return Results.NotFound($"Chat {connectionId} not found");
    }
    
    connectionManager.Disconnect(indexName, connectionId);
    return Results.Ok(new { success = true });
});

Console.WriteLine("Starting ChatService on http://localhost:5000");
app.Run("http://0.0.0.0:5000"); 