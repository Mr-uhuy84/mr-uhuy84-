#!/usr/bin/env python3
"""
Muse Relay — jembatan antara 9Router (di Railway) dan muse-bridge (di VPS).

Alur:
  9Router --POST /v1/chat/completions--> relay (menyimpan job, menunggu)
  VPS poller --GET /relay/poll--> relay (long-poll, mengambil job)
  VPS poller --POST /relay/result--> relay (menjawab)
  relay --> 9Router (respons OpenAI, JSON atau SSE bila stream=true)

Kenapa relay? VPS tidak bisa menerima koneksi inbound (NAT + egress proxy
memblokir semua tunnel). Satu-satunya yang jalan: HTTPS KELUAR dari VPS.
Maka VPS yang proaktif polling ke relay, bukan sebaliknya.

Murni Python stdlib. Deploy: Dockerfile (Railway).

Env:
  PORT               (default 8080; Railway mengisi otomatis)
  RELAY_API_KEY      (wajib) key untuk 9Router -> relay (Bearer)
  RELAY_SECRET       (wajib) secret untuk VPS poller -> relay (X-Relay-Secret)
  RELAY_WAIT_TIMEOUT (default 280) detik menunggu jawaban agen per request

Endpoint:
  GET  /healthz
  GET  /v1/models                      (Bearer RELAY_API_KEY)
  POST /v1/chat/completions            (Bearer RELAY_API_KEY)
  GET  /relay/poll?timeout=55          (X-Relay-Secret)
  POST /relay/result                   (X-Relay-Secret)
"""

import json
import os
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

PORT = int(os.environ.get("PORT", "8080"))
RELAY_API_KEY = os.environ.get("RELAY_API_KEY", "").strip()
RELAY_SECRET = os.environ.get("RELAY_SECRET", "").strip()
WAIT_TIMEOUT = int(os.environ.get("RELAY_WAIT_TIMEOUT", "280"))

MODELS = ["muse-spark", "muse-spark-reasoning"]

_lock = threading.Lock()
_cond = threading.Condition(_lock)
_pending = deque()          # antrean job id
_jobs = {}                  # id -> {"request":..., "event":Event, "result":None|dict, "created":ts}


def _json(status, obj, extra_headers=None):
    body = json.dumps(obj, ensure_ascii=False).encode()
    return status, {"Content-Type": "application/json",
                    "Content-Length": str(len(body))}, body


def _err(status, message, mtype="server_error"):
    return _json(status, {"error": {"message": message, "type": mtype}})


def _check_api_key(handler):
    if not RELAY_API_KEY:
        return True  # belum dikonfigurasi: tolak semua kecuali healthz di bawah
    auth = handler.headers.get("Authorization", "")
    return auth == f"Bearer {RELAY_API_KEY}"


def _check_secret(handler):
    if not RELAY_SECRET:
        return False
    return handler.headers.get("X-Relay-Secret", "") == RELAY_SECRET


def _openai_completion(model, content, tool_calls=None):
    msg = {"role": "assistant", "content": content or ""}
    finish = "stop"
    if tool_calls:
        msg["tool_calls"] = tool_calls
        finish = "tool_calls"
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:12],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _sse_stream(model, content, tool_calls=None):
    """Sintesis SSE dari respons lengkap (untuk stream=true)."""
    cid = "chatcmpl-" + uuid.uuid4().hex[:12]
    created = int(time.time())
    delta = {"role": "assistant", "content": content or ""}
    finish = "stop"
    if tool_calls:
        delta["tool_calls"] = tool_calls
        finish = "tool_calls"
    chunk = {
        "id": cid, "object": "chat.completion.chunk", "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }
    done_chunk = {
        "id": cid, "object": "chat.completion.chunk", "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
    }
    body = (
        "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"
        + "data: " + json.dumps(done_chunk, ensure_ascii=False) + "\n\n"
        + "data: [DONE]\n\n"
    ).encode()
    return 200, {"Content-Type": "text/event-stream",
                 "Cache-Control": "no-cache",
                 "Content-Length": str(len(body))}, body


class Handler(BaseHTTPRequestHandler):
    server_version = "MuseRelay/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, status, headers, body):
        try:
            self.send_response(status)
            for k, v in headers.items():
                self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _read_json(self):
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            n = 0
        raw = self.rfile.read(n) if n > 0 else b""
        try:
            return json.loads(raw.decode()) if raw else {}
        except (ValueError, UnicodeDecodeError):
            return None

    # -- GET ------------------------------------------------------------
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/healthz":
            s, h, b = _json(200, {"status": "ok"})
            return self._send(s, h, b)

        if path == "/v1/models":
            if not _check_api_key(self):
                s, h, b = _err(401, "Invalid relay API key",
                               "authentication_error")
                return self._send(s, h, b)
            data = [{"id": m, "object": "model",
                     "created": int(time.time()), "owned_by": "muse"}
                    for m in MODELS]
            s, h, b = _json(200, {"object": "list", "data": data})
            return self._send(s, h, b)

        if path == "/relay/poll":
            if not _check_secret(self):
                s, h, b = _err(403, "Invalid relay secret", "forbidden")
                return self._send(s, h, b)
            try:
                timeout = min(int(parse_qs(parsed.query)
                                  .get("timeout", ["55"])[0]), 120)
            except ValueError:
                timeout = 55
            deadline = time.time() + timeout
            job = None
            with _cond:
                while time.time() < deadline:
                    # buang job basi (> WAIT_TIMEOUT) agar tidak menumpuk
                    while _pending:
                        jid = _pending[0]
                        j = _jobs.get(jid)
                        if j and time.time() - j["created"] < WAIT_TIMEOUT + 30:
                            break
                        _pending.popleft()
                        _jobs.pop(jid, None)
                    if _pending:
                        jid = _pending.popleft()
                        j = _jobs.pop(jid, None)
                        if j is not None:
                            job = {"id": jid, **j["request"]}
                            j["claimed"] = True
                            _jobs[jid] = j
                        break
                    _cond.wait(timeout=min(5, max(0.5, deadline - time.time())))
            if job is None:
                return self._send(204, {"Content-Length": "0"}, b"")
            s, h, b = _json(200, {"job": job})
            return self._send(s, h, b)

        s, h, b = _err(404, "Not found", "not_found")
        return self._send(s, h, b)

    # -- POST -----------------------------------------------------------
    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/v1/chat/completions":
            if RELAY_API_KEY and not _check_api_key(self):
                s, h, b = _err(401, "Invalid relay API key",
                               "authentication_error")
                return self._send(s, h, b)
            payload = self._read_json()
            if payload is None:
                s, h, b = _err(400, "Body bukan JSON valid", "invalid_request")
                return self._send(s, h, b)
            if not isinstance(payload.get("messages"), list):
                s, h, b = _err(400, "Field 'messages' wajib berupa array",
                               "invalid_request")
                return self._send(s, h, b)

            job_id = uuid.uuid4().hex
            job = {"request": {"model": payload.get("model", "muse-spark"),
                               "messages": payload["messages"],
                               "tools": payload.get("tools"),
                               "tool_choice": payload.get("tool_choice"),
                               "temperature": payload.get("temperature"),
                               "max_tokens": payload.get("max_tokens")},
                   "event": threading.Event(),
                   "result": None,
                   "created": time.time(),
                   "stream": bool(payload.get("stream"))}
            with _cond:
                _jobs[job_id] = job
                _pending.append(job_id)
                _cond.notify_all()

            ok = job["event"].wait(timeout=WAIT_TIMEOUT)
            with _lock:
                result = job["result"]
                _jobs.pop(job_id, None)
                try:
                    _pending.remove(job_id)
                except ValueError:
                    pass
            if not ok or result is None:
                s, h, b = _err(504, "Agen tidak merespons tepat waktu.",
                               "timeout")
                return self._send(s, h, b)
            model = job["request"]["model"]
            if job["stream"]:
                s, h, b = _sse_stream(model, result.get("content"),
                                     result.get("tool_calls"))
            else:
                s, h, b = _json(200, _openai_completion(
                    model, result.get("content"), result.get("tool_calls")))
            return self._send(s, h, b)

        if path == "/relay/result":
            if not _check_secret(self):
                s, h, b = _err(403, "Invalid relay secret", "forbidden")
                return self._send(s, h, b)
            payload = self._read_json()
            if payload is None:
                s, h, b = _err(400, "Body bukan JSON valid", "invalid_request")
                return self._send(s, h, b)
            job_id = str(payload.get("id", ""))
            with _lock:
                job = _jobs.get(job_id)
                if job is None or job.get("result") is not None:
                    s, h, b = _err(404, "Job tidak ditemukan/kedaluwarsa.",
                                   "not_found")
                    return self._send(s, h, b)
                job["result"] = {"content": payload.get("content", ""),
                                 "tool_calls": payload.get("tool_calls")}
                job["event"].set()
            with _cond:
                _cond.notify_all()
            s, h, b = _json(200, {"ok": True})
            return self._send(s, h, b)

        s, h, b = _err(404, "Not found", "not_found")
        return self._send(s, h, b)


def main():
    if not RELAY_API_KEY or not RELAY_SECRET:
        raise SystemExit("RELAY_API_KEY dan RELAY_SECRET wajib diisi.")
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    srv.daemon_threads = True
    print(f"Muse Relay listening on 0.0.0.0:{PORT}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
