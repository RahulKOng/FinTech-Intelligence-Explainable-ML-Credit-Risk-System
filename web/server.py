import os
import sys
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(base_dir, 'credit_risk_engine'))
sys.path.insert(0, os.path.join(base_dir, 'src'))

try:
    from src.pipeline import CreditRiskPipeline
except ImportError:
    from pipeline import CreditRiskPipeline

class FinTechAppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.join(base_dir, 'web'), **kwargs)

    def do_POST(self):
        if self.path == '/api/evaluate':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            params = json.loads(body) if body else {}

            dataset_type = params.get('dataset', 'lending_club')
            model_type = params.get('model_type', 'logistic')
            threshold = float(params.get('threshold', 0.20))

            if dataset_type == 'german':
                dataset_path = os.path.join(base_dir, 'data', 'german_credit_sample.csv')
            elif dataset_type == 'give_me_credit':
                dataset_path = os.path.join(base_dir, 'data', 'give_me_some_credit_sample.csv')
            else:
                dataset_path = os.path.join(base_dir, 'data', 'lending_club_sample.csv')

            pipeline = CreditRiskPipeline(model_type=model_type)
            results = pipeline.run(dataset_path, threshold=threshold)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(results).encode('utf-8'))
        else:
            self.send_error(404, "Endpoint not found")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def run_server(port=8080):
    server_address = ('', port)
    httpd = HTTPServer(server_address, FinTechAppHandler)
    print(f"FinTech Explainable ML Web Server running at http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_server(port)
