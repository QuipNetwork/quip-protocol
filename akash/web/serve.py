#!/usr/bin/env python3
"""
HTTPS server for Akash Web Deployment Interface
Serves the pre-built Vite application from dist/ folder
"""
import http.server
import socketserver
import ssl
import webbrowser
import os
import subprocess
import sys
from pathlib import Path

PORT = 8000
WEB_DIR = Path(__file__).parent
DIST_DIR = WEB_DIR / 'dist'
CERT_FILE = WEB_DIR / 'localhost.pem'


class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with CORS and SPA support"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIST_DIR), **kwargs)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

    def do_GET(self):
        # SPA fallback: serve index.html for non-file routes
        path = self.translate_path(self.path)
        if not os.path.exists(path) and '.' not in os.path.basename(self.path):
            self.path = '/index.html'
        return super().do_GET()


def generate_cert():
    """Generate self-signed certificate"""
    if CERT_FILE.exists():
        return True

    print("Generating self-signed certificate...")
    try:
        subprocess.run([
            'openssl', 'req', '-new', '-x509',
            '-keyout', str(CERT_FILE), '-out', str(CERT_FILE),
            '-days', '365', '-nodes', '-subj', '/CN=localhost'
        ], check=True, capture_output=True)
        print("Certificate generated")
        return True
    except Exception:
        print("OpenSSL not found. Install OpenSSL first.")
        return False


def main():
    # Check dist folder exists
    if not DIST_DIR.exists():
        print(f"dist/ folder not found at {DIST_DIR}")
        print("   Run 'bun run build' first (developer only)")
        sys.exit(1)

    if not generate_cert():
        sys.exit(1)

    with socketserver.TCPServer(("", PORT), CORSHTTPRequestHandler) as httpd:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=str(CERT_FILE))
        httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

        print(f"""
=====================================================================
         Quip Protocol - Akash Web Deployment Interface
=====================================================================

HTTPS Server: https://localhost:{PORT}

To connect Keplr Mobile:
   1. Click "Connect Wallet"
   2. Scan QR code with Keplr Mobile
   3. Approve connection
   4. Sign transactions on your phone

Press Ctrl+C to stop
        """)

        try:
            webbrowser.open(f'https://localhost:{PORT}')
        except Exception:
            pass

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped")


if __name__ == "__main__":
    main()
