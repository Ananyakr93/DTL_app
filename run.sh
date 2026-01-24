#!/bin/bash

# AeroClean Dashboard Launcher

echo "=================================="
echo "   AeroClean Dashboard Setup"
echo "=================================="

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "❌ Error: pip is not installed or not in PATH."
    exit 1
fi

echo "📦 Installing necessary dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully."
else
    echo "❌ Error installing dependencies."
    exit 1
fi

echo "=================================="
echo "   Starting Local Server..."
echo "=================================="
echo "🌍 Dashboard will be available at: http://127.0.0.1:5000"

python app.py
