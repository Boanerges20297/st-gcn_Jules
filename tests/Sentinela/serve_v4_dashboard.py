import http.server
import socketserver
import webbrowser
import os

PORT = 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def start_server():
    print(f"🚀 Iniciando servidor para o Dashboard Sentinela V4...")
    print(f"📂 Diretorio: {DIRECTORY}")
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        url = f"http://localhost:{PORT}/dashboard_v4.html"
        print(f"🔗 Dashboard disponivel em: {url}")
        print("💡 Pressione Ctrl+C para encerrar o servidor.")
        
        # Abre o navegador automaticamente
        webbrowser.open(url)
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Servidor encerrado.")
            httpd.server_close()

if __name__ == "__main__":
    start_server()
