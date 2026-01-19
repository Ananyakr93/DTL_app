"""
Vercel Serverless Entry Point for AeroClean Dashboard
This file exposes the Flask app as a Vercel serverless function.
"""
import sys
import os

# Add parent directory to path so we can import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment to indicate serverless deployment
os.environ['VERCEL'] = '1'
os.environ['FLASK_ENV'] = 'production'

from app import app

# Vercel expects the app to be named 'app' or 'handler'
# The Flask app object is already named 'app' in app.py
