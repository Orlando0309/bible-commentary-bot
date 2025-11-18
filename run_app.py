"""
Simple launcher script for the Streamlit app.
Run this script to start the web interface.
"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    app_path = Path(__file__).parent / "src" / "app.py"
    
    if not app_path.exists():
        print(f"Error: App file not found at {app_path}")
        sys.exit(1)
    
    print("Starting Bible Commentary Bot Streamlit app...")
    print(f"App location: {app_path}")
    print("\nThe app will open in your default web browser.")
    print("Press Ctrl+C to stop the server.\n")
    
    try:
        subprocess.run([
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.port=8501",
            "--server.address=localhost"
        ])
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
    except Exception as e:
        print(f"Error starting Streamlit: {e}")
        sys.exit(1)

