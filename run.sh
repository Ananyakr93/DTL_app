#!/bin/bash
# Run script for Air Quality Prediction Dashboard

echo "🚀 Starting AeroClean Dashboard..."

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Check if dependencies are installed
python -c "import flask" 2>/dev/null || {
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
}

# Run the Flask application
echo "🌐 Starting server at http://localhost:5000"
cd src
python app.py
