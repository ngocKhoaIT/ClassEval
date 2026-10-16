"""Local web UI for ClassEval model generation. Uses Python's standard library."""

import argparse
import json
import subprocess
import sys
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "ClassEval_data.json"
INFERENCE_FILE = ROOT / "generation" / "inference.py"
PAGE_FILE = ROOT / "local_demo.html"
COMPARE_PAGE_FILE = ROOT / "local_compare.html"
OUTPUT_DIR = ROOT / "output" / "model_output"
METRIC_FILE = ROOT / "output" / "result" / "pass_at_k_result.json"

# The numeric IDs match generation/inference.py. CodeGeeX2 is a later variant
# of the CodeGeeX model in the original ClassEval paper.
MODELS = [
    {"name": "SantaCoder · 1.1B", "id": 5, "checkpoint": "bigcode/santacoder", "styles": [0, 1, 2, 3]},
    {"name": "DeepSeek-Coder-Instruct · 1.3B", "id": 13, "checkpoint": "deepseek-ai/deepseek-coder-1.3b-instruct", "styles": [0, 1, 2, 3], "max_length": 4096},
    {"name": "PolyCoder · 2.7B", "id": 4, "checkpoint": "", "styles": [0]},
    {"name": "InCoder · 6B", "id": 3, "checkpoint": "", "styles": [0, 1, 2, 3]},
    {"name": "ChatGLM · 6B", "id": 7, "checkpoint": "", "styles": [0, 1, 2, 3]},
    {"name": "Vicuna · 7B", "id": 6, "checkpoint": "", "styles": [0, 1, 2, 3]},
    {"name": "CodeGeeX2 · 6B", "id": 12, "checkpoint": "", "styles": [0, 1, 2, 3]},
    {"name": "WizardCoder · 15B", "id": 1, "checkpoint": "WizardLMTeam/WizardCoder-15B-V1.0", "styles": [0, 1, 2, 3]},
    {"name": "Instruct-StarCoder · 15B", "id": 2, "checkpoint": "", "styles": [0, 1, 2, 3]},
    {"name": "Instruct-CodeGen · 16B", "id": 0, "checkpoint": "", "styles": [0, 1, 2, 3]},
]


def load_tasks():
    with DATA_FILE.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return [{"task_id": item["task_id"], "class_name": item["class_name"],
             "skeleton": item["skeleton"]} for item in data]


def build_command(model_id, checkpoint, task_id, style, output_path, max_length):
    return [sys.executable, str(INFERENCE_FILE),
            "--model", str(model_id), "--checkpoint", checkpoint,
            "--data_path", str(DATA_FILE), "--task_ids", task_id,
            "--generation_strategy", "0", "--prompt_style", str(style),
            "--greedy", "1", "--max_length", str(max_length),
            "--output_path", str(output_path)]


def load_metrics():
    with METRIC_FILE.open(encoding="utf-8") as handle:
        groups = json.load(handle)
    return {"source": "merged", "file": str(METRIC_FILE.relative_to(ROOT)), "groups": groups}


class DemoState:
    def __init__(self):
        self.lock = threading.Lock()
        self.tasks = load_tasks()
        self.task_ids = {task["task_id"] for task in self.tasks}
        self.status = "idle"
        self.logs = ""
        self.output_path = None
        self.error = None
        self.process = None

    def snapshot(self):
        with self.lock:
            return {"status": self.status, "logs": self.logs[-12000:],
                    "output_path": str(self.output_path) if self.output_path else None,
                    "error": self.error}

    def start(self, payload):
        model_id = payload.get("model_id")
        model = next((model for model in MODELS if model["id"] == model_id), None)
        task_id = payload.get("task_id")
        checkpoint = payload.get("checkpoint")
        style = payload.get("style")
        max_length = payload.get("max_length")
        if model is None or task_id not in self.task_ids:
            raise ValueError("Model hoặc bài ClassEval không hợp lệ.")
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            raise ValueError("Hãy nhập ID Hugging Face hoặc đường dẫn checkpoint local.")
        if style not in model["styles"]:
            raise ValueError("Prompt style không phù hợp với model đã chọn.")
        if style == 2 and task_id in ("ClassEval_6", "ClassEval_30"):
            raise ValueError("Bài này đang được dùng làm ví dụ few-shot. Hãy chọn bài khác.")
        if type(max_length) is not int or not 64 <= max_length <= 32768:
            raise ValueError("Max length phải là số nguyên từ 64 đến 32768.")
        checkpoint = checkpoint.strip()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = OUTPUT_DIR / f"local_{model_id}_{task_id}_{stamp}.json"
        command = build_command(model_id, checkpoint, task_id, style, output_path, max_length)
        with self.lock:
            if self.status in ("running", "stopping"):
                raise ValueError("Một model đang chạy. Hãy đợi hoặc bấm Dừng.")
            self.status = "running"
            self.logs = f"Đang tạo {output_path}\n"
            self.output_path = output_path
            self.error = None
        threading.Thread(target=self._worker, args=(command, output_path), daemon=True).start()
        return {"output_path": str(output_path)}

    def _worker(self, command, output_path):
        try:
            process = subprocess.Popen(command, cwd=ROOT / "generation",
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace", bufsize=1)
            with self.lock:
                self.process = process
                if self.status == "stopping":
                    process.terminate()
            for line in process.stdout:
                with self.lock:
                    self.logs = (self.logs + line)[-24000:]
            code = process.wait()
            valid_output = False
            if code == 0 and output_path.is_file():
                try:
                    with output_path.open(encoding="utf-8") as handle:
                        result = json.load(handle)
                    valid_output = (isinstance(result, list) and len(result) == 1
                                    and bool(result[0].get("predict")))
                except (OSError, ValueError, TypeError):
                    pass
            with self.lock:
                self.status = "done" if valid_output else "failed"
                if self.status == "failed":
                    self.error = (f"Lệnh kết thúc với mã {code} nhưng chưa có dự đoán hợp lệ. "
                                  "Xem tab Log để biết chi tiết.")
                self.process = None
        except Exception as exc:
            with self.lock:
                self.status = "failed"
                self.error = str(exc)
                self.process = None

    def cancel(self):
        with self.lock:
            process = self.process
            if self.status != "running":
                return False
            self.status = "stopping"
        if process and process.poll() is None:
            process.terminate()
        return True

    def result(self):
        with self.lock:
            path = self.output_path
        if path is None or not path.is_file():
            return None
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)


class DemoHandler(BaseHTTPRequestHandler):
    state = None

    def log_message(self, _format, *_args):
        pass

    def _send(self, body, content_type="application/json; charset=utf-8", status=200, download=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, status=200):
        self._send(json.dumps(data, ensure_ascii=False).encode("utf-8"), status=status)

    def do_GET(self):
        request = urlparse(self.path)
        path = request.path
        if path == "/":
            self._send(PAGE_FILE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/compare":
            self._send(COMPARE_PAGE_FILE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/tasks":
            self._json(self.state.tasks)
        elif path == "/api/models":
            self._json(MODELS)
        elif path == "/api/metrics":
            self._json(load_metrics())
        elif path == "/api/status":
            self._json(self.state.snapshot())
        elif path in ("/api/result", "/api/download"):
            result = self.state.result()
            if result is None:
                self._json({"error": "Chưa có JSON kết quả."}, 404)
            else:
                body = json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8")
                name = self.state.output_path.name if path == "/api/download" else None
                self._send(body, download=name)
        else:
            self._json({"error": "Không tìm thấy."}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        origin = self.headers.get("Origin")
        valid_origins = {f"http://127.0.0.1:{self.server.server_port}",
                         f"http://localhost:{self.server.server_port}"}
        if origin and origin not in valid_origins:
            self._json({"error": "Origin không hợp lệ."}, 403)
            return
        if path == "/api/cancel":
            self._json({"stopping": self.state.cancel()})
            return
        if path != "/api/run":
            self._json({"error": "Không tìm thấy."}, 404)
            return
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            self._json({"error": "Yêu cầu phải là JSON."}, 415)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 32768:
                raise ValueError("Kích thước yêu cầu không hợp lệ.")
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError("Yêu cầu phải là một đối tượng JSON.")
            self._json(self.state.start(payload), 202)
        except (ValueError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, 400)


def main():
    parser = argparse.ArgumentParser(description="ClassEval local model demo")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    DemoHandler.state = DemoState()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), DemoHandler)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"ClassEval local demo: {url}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        DemoHandler.state.cancel()
        server.server_close()


if __name__ == "__main__":
    main()
