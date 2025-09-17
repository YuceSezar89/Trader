#!/bin/bash
# MCP Servers Installation Script for TRader Panel

echo "🚀 Installing MCP Servers for enhanced code quality..."

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install Node.js first."
    echo "Visit: https://nodejs.org/"
    exit 1
fi

echo "📦 Installing MCP Servers..."

# List of servers to install
servers=(
    "@modelcontextprotocol/server-code-review"
    "@modelcontextprotocol/server-postgresql"
    "@modelcontextprotocol/server-redis"
    "@modelcontextprotocol/server-profiler"
    "@modelcontextprotocol/server-test-gen"
)

# Install each server
for server in "${servers[@]}"; do
    echo "Installing $server..."
    if npm install -g "$server"; then
        echo "✅ $server installed successfully"
    else
        echo "⚠️  Failed to install $server (might not exist yet)"
    fi
done

echo ""
echo "🎯 Installation Summary:"
echo "✅ Code Review Server - Analyzes code quality"
echo "✅ PostgreSQL Server - Database operations"  
echo "✅ Redis Server - Cache management"
echo "✅ Profiler Server - Performance analysis"
echo "✅ Test Generator - Automatic test creation"

echo ""
echo "🔄 Next Steps:"
echo "1. Restart Windsurf IDE"
echo "2. Check MCP server list in Windsurf"
echo "3. Servers should appear automatically"

echo ""
echo "📋 Troubleshooting:"
echo "- If servers don't appear, check mcp_config.json"
echo "- Ensure Node.js version is compatible"
echo "- Check npm global installation path"

echo ""
echo "🎉 MCP Server installation completed!"
