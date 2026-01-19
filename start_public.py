"""
Quick script to start Flask with a public ngrok tunnel.
Run this instead of `python app.py` to get a shareable link.
"""
from pyngrok import ngrok
import subprocess
import sys
import time

# Start Flask in background
print("🚀 Starting Flask server...")
flask_process = subprocess.Popen(
    [sys.executable, "app.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

# Wait for Flask to start
time.sleep(5)

# Start ngrok tunnel
print("🔗 Creating public tunnel...")
try:
    public_url = ngrok.connect(5000)
    print("\n" + "="*60)
    print("🎉 YOUR SHAREABLE LINK:")
    print(f"   {public_url}")
    print("="*60)
    print("\nShare this link with anyone to access your dashboard!")
    print("Press Ctrl+C to stop the server.\n")
    
    # Keep running
    while True:
        output = flask_process.stdout.readline()
        if output:
            print(output.strip())
        if flask_process.poll() is not None:
            break
            
except KeyboardInterrupt:
    print("\n🛑 Shutting down...")
    ngrok.kill()
    flask_process.terminate()
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to sign up at ngrok.com and run: ngrok config add-authtoken YOUR_TOKEN")
