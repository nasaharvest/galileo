#!/bin/bash
# Development environment setup script

set -e

echo "Setting up Galileo development environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Sync dependencies
echo "Syncing dependencies with uv..."
uv sync

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
uv run pre-commit install

echo ""
echo "✅ Setup complete!"
echo ""
echo "Pre-commit hooks are now installed and will run automatically on git commit."
echo "To run checks manually: uv run pre-commit run --all-files"
echo "To run tests with coverage: uv run coverage run -m unittest discover -s tests"
echo ""
