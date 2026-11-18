"""
TRACE.AI Unified Platform Launcher
Starts the FastAPI backend & serves the built React web application at http://localhost:8000.
Automatically opens the platform in your default web browser.
"""

import sys
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

def main():
    print("""
================================================================================
   ______ ____   ___   ______ ______     ___     ____
  /_  __// __ \ /   | / ____// ____/    /   |   /  _/
   / /  / /_/ // /| |/ /    / __/      / /| |   / /  
  / /  / _, _// ___ / /___ / /___ _   / ___ |_ / /   
 /_/  /_/ |_|/_/  |_\____//_____/(_) /_/  |_/___/    
                                                     
   Route Intelligence & GPS Map Matching Web Application
   PES University Machine Learning Hackathon (UE24CS352A)
================================================================================
""")
    port = 8000
    url = f"http://localhost:{port}"

    print(f"[+] Starting TRACE.AI Application Engine on {url} ...")
    print(f"[+] Web Interface: {url}")
    print(f"[+] REST API Docs: {url}/docs")
    print(f"[+] Press Ctrl + C to exit.\n")

    # Open browser automatically after a short delay
    def open_browser():
        time.sleep(1.5)
        print(f"[+] Opening {url} in your browser...")
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"[-] Could not open browser automatically: {e}")

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=port, reload=False)

if __name__ == "__main__":
    main()
