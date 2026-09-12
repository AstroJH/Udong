"""Tests for the resumable downloader using a local Range-capable HTTP server."""

import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from udong.data.manga.downloader import download_file


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


class _FakeHttpClient:
    """Stands in for easycat's HttpClient and records the delegated options."""

    def __init__(self, payload: bytes, status: int = 200):
        self.payload = payload
        self.status = status
        self.calls = []

    def download_file(self, url, dest, **kw):
        self.calls.append({"url": url, **kw})
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.payload)
        prog = kw.get("progress")
        if callable(prog):
            prog(len(self.payload), len(self.payload))
        return dest

    def close(self):
        pass


def _patch_client(monkeypatch, payload, status=200):
    fake = _FakeHttpClient(payload, status=status)
    monkeypatch.setattr("udong.data.manga.downloader.http_client", lambda **kw: fake)
    return fake


def test_download_prints_text_prompt(monkeypatch, tmp_path, capsys):
    payload = bytes(range(256)) * 512  # 128 KiB
    _patch_client(monkeypatch, payload)
    dest = tmp_path / "out.bin"
    download_file("https://data.example/file.bin", dest, verify_gzip=False)

    err = capsys.readouterr().err
    assert "[udong] downloading file.bin" in err
    assert str(dest) in err
    assert dest.read_bytes() == payload


def test_download_silent_when_progress_disabled(monkeypatch, tmp_path, capsys):
    fake = _patch_client(monkeypatch, b"abc")
    download_file("https://data.example/file.bin", tmp_path / "out.bin",
                  verify_gzip=False, progress=False)
    assert capsys.readouterr().err == ""
    assert fake.calls[0]["progress"] is False


def test_download_delegates_resume_verification_and_validation(monkeypatch, tmp_path):
    payload = bytes(range(256)) * 512
    fake = _patch_client(monkeypatch, payload)

    dest = tmp_path / "out.fits.gz"
    download_file("https://data.example/out.fits.gz", dest)

    kw = fake.calls[0]
    assert kw["resume"] == "auto"          # resumable .part download
    assert kw["verify_size"] is True
    assert kw["trust_existing"] is False
    assert kw["cleanup_partial"] is False
    assert kw["validate"] == "gzip"

    # non-gzip destinations do not get the gzip validator
    download_file("https://data.example/out.bin", tmp_path / "out.bin",
                  verify_gzip=True, progress=False)
    assert fake.calls[1]["validate"] is None
