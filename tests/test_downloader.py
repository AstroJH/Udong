"""Tests for the resumable downloader using a local Range-capable HTTP server."""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from udong.data.manga.downloader import download_file, sha256_file


class RangeHandler(BaseHTTPRequestHandler):
    payload = b""

    def do_GET(self):
        body = self.payload
        start = 0
        rng = self.headers.get("Range")
        if rng:
            try:
                start = int(rng.split("=")[1].split("-")[0])
            except (IndexError, ValueError):
                start = 0
        if start > 0:
            body = body[start:]
            self.send_response(206)
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence
        pass


@pytest.fixture
def http_server(tmp_path):
    payload = bytes(range(256)) * 512  # 128 KiB
    RangeHandler.payload = payload
    try:
        server = HTTPServer(("127.0.0.1", 0), RangeHandler)
    except OSError as e:  # sandboxed environments may forbid sockets
        pytest.skip(f"cannot bind a local socket here: {e}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/file.bin"
    yield url, payload, tmp_path
    server.shutdown()


def test_fresh_download(http_server):
    url, payload, tmp_path = http_server
    dest = tmp_path / "out.bin"
    download_file(url, dest, verify_gzip=False)
    assert dest.read_bytes() == payload


def test_resume_download(http_server):
    url, payload, tmp_path = http_server
    dest = tmp_path / "out.bin"
    dest.write_bytes(payload[:1000])
    download_file(url, dest, verify_gzip=False)
    assert dest.read_bytes() == payload


def test_sha256(http_server):
    url, payload, tmp_path = http_server
    dest = tmp_path / "out.bin"
    download_file(url, dest, verify_gzip=False)
    import hashlib

    assert sha256_file(dest) == hashlib.sha256(payload).hexdigest()


class _FakeResponse:
    """Minimal urllib response (context manager) for progress tests."""

    def __init__(self, payload: bytes):
        self._payload = payload
        self._sent = False
        self.status = 200
        self.headers = {"Content-Length": str(len(payload))}

    def read(self, n: int = -1) -> bytes:
        if self._sent:
            return b""
        self._sent = True
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_download_prints_text_prompt(monkeypatch, tmp_path, capsys):
    payload = bytes(range(256)) * 512  # 128 KiB
    monkeypatch.setattr(
        "udong.data.manga.downloader.urllib.request.urlopen",
        lambda req, timeout=None: _FakeResponse(payload),
    )
    dest = tmp_path / "out.bin"
    download_file("https://data.example/file.bin", dest, verify_gzip=False)

    err = capsys.readouterr().err
    assert "[udong] downloading file.bin" in err
    assert str(dest) in err
    assert dest.read_bytes() == payload


def test_download_silent_when_progress_disabled(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        "udong.data.manga.downloader.urllib.request.urlopen",
        lambda req, timeout=None: _FakeResponse(b"abc"),
    )
    download_file("https://data.example/file.bin", tmp_path / "out.bin",
                  verify_gzip=False, progress=False)
    assert capsys.readouterr().err == ""
