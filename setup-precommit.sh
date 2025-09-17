#!/bin/bash
# Pre-commit setup script for TRader Panel

echo "🔧 Setting up pre-commit hooks for TRader Panel..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed. Please install pip first."
    exit 1
fi

# Install pre-commit
echo "📦 Installing pre-commit..."
pip install pre-commit

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
pre-commit install

# Install additional development dependencies
echo "📦 Installing development dependencies..."
pip install black isort flake8 mypy bandit pydocstyle
pip install types-redis types-psycopg2 types-requests

# Run pre-commit on all files (optional)
echo "🧹 Running pre-commit on all files..."
pre-commit run --all-files

echo "✅ Pre-commit setup completed!"
echo ""
echo "📋 Available commands:"
echo "  pre-commit run --all-files    # Run all hooks on all files"
echo "  pre-commit run <hook-name>    # Run specific hook"
echo "  pre-commit autoupdate         # Update hook versions"
echo ""
echo "🎯 Hooks will now run automatically on git commit!"
