#!/bin/bash

# Check if .NET SDK is installed
if command -v dotnet &> /dev/null; then
    echo ".NET SDK is installed"
    dotnet --version
else
    echo ".NET SDK is not installed. Please install .NET SDK."
    echo "Visit https://dotnet.microsoft.com/download for installation instructions."
    exit 1
fi

# Create a temporary global.json to use .NET 9.0 SDK
echo '{
  "sdk": {
    "version": "9.0.0",
    "rollForward": "latestFeature"
  }
}' > global.json

echo "Created temporary global.json to use .NET 9.0 SDK"

# Build the project
echo "Building the project..."
dotnet build

# Run the project
echo "Running the project..."
dotnet run --urls http://localhost:5000 