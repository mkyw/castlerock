using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace ChatService.Services
{
    public class ConnectionCleanupService : BackgroundService
    {
        private readonly ConnectionManager _connectionManager;
        private readonly ILogger<ConnectionCleanupService> _logger;
        private readonly TimeSpan _cleanupInterval = TimeSpan.FromMinutes(1); // Run cleanup every minute
        private readonly TimeSpan _inactivityThreshold = TimeSpan.FromMinutes(5); // Consider connections inactive after 5 minutes

        public ConnectionCleanupService(
            ConnectionManager connectionManager,
            ILogger<ConnectionCleanupService> logger)
        {
            _connectionManager = connectionManager;
            _logger = logger;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            _logger.LogInformation("Connection cleanup service started");

            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    // Wait for the next cleanup interval
                    await Task.Delay(_cleanupInterval, stoppingToken);
                    
                    // Perform cleanup
                    int removedCount = _connectionManager.CleanupInactiveConnections(_inactivityThreshold);
                    
                    if (removedCount > 0)
                    {
                        _logger.LogInformation($"Cleaned up {removedCount} inactive connections");
                    }
                }
                catch (OperationCanceledException)
                {
                    // Graceful shutdown
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error during connection cleanup");
                }
            }

            _logger.LogInformation("Connection cleanup service stopped");
        }
    }
}
