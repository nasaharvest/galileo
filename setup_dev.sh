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

# Install dependencies
echo "Installing dependencies..."
uv pip install --system -r requirements-dev.txt

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

echo ""
echo "✅ Setup complete!"
echo ""
echo "Pre-commit hooks are now installed and will run automatically on git commit."
echo "To run checks manually: pre-commit run --all-files"
echo "To run tests with coverage: python -m coverage run -m unittest discover -s tests"
echo ""
