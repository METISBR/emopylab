"""Local LLM client for EmoPyLab — auto-detects Apple Silicon and
uses mlx-lm runtime when available, otherwise falls back to any
OpenAI-compatible HTTP endpoint.

This module is the single entry point that every LLM-integrated
algorithm in EmoPyLab (MaACO, LARC-NSGA3, etc.) should use.  It
exposes a uniform ``LocalLLMClient`` with a ``__call__(prompt)``
interface so algorithms can stay backend-agnostic.

Usage
-----
The default endpoint and model target the Qwen2.5-0.5B-Instruct-4bit
variant served by mlx-lm on Apple Silicon:

    from core.llm.local_llm import LocalLLMClient
    client = LocalLLMClient()
    response = client("Return JSON with q=0.6 and xi=0.8")

Run ``python -m mlx_lm server --model mlx-community/Qwen2.5-0.5B-Instruct-4bit``
to launch the local server (default port 8080).

Backend chain (highest priority first)
--------------------------------------
1. **Apple Silicon + mlx-lm server reachable** — HTTP via
   ``http://127.0.0.1:8080/v1`` (the canonical mlx-lm endpoint).
2. **Non-Apple-Silicon + ``llama_cpp`` installed** — in-process GGUF
   runtime via ``Llama.from_pretrained``, no external HTTP server needed.
3. **Any other reachable OpenAI-compatible HTTP server** as a fallback.

Splash screen
-------------
``core.llm.splash.ensure_llm_ready_or_skip`` shows a modal dialog at
application startup that waits for the local LLM to become responsive
(and, on Apple Silicon, attempts to start mlx-lm in the background).
"""

from __future__ import annotations

import importlib.util as _importlib_util
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults — tuned for Apple Silicon via mlx-lm
# ---------------------------------------------------------------------------

# mlx-lm default server port (see: ``python -m mlx_lm server --help``)
DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MLX_PORT = 8080
DEFAULT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
# In-process GGUF model (used by llama-cpp-python fallback on Windows/Linux)
DEFAULT_GGUF_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
DEFAULT_GGUF_FILE_PATTERN = "*q4_k_m.gguf"
DEFAULT_TIMEOUT = 15.0
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 200


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def is_apple_silicon() -> bool:
    """Return True when running on Apple Silicon (macOS, arm64)."""
    return sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def mlx_runtime_available() -> bool:
    """Return True when mlx-lm is importable on this machine."""
    try:
        import mlx_lm  # noqa: F401
        return True
    except Exception:
        return False


def llama_cpp_python_available() -> bool:
    """Return True when llama-cpp-python is importable on this machine.

    Used as the in-process fallback for Windows / Linux / Intel macOS hosts
    where mlx-lm cannot run.
    """
    return _importlib_util.find_spec("llama_cpp") is not None


def auto_backend_status() -> Dict[str, Any]:
    """Report which local LLM backends are usable on this host.

    Backend selection rule (highest priority first):
      1. Explicit env var (EMOPYLAB_LLM_URL / EMOPYLAB_LMSTUDIO_URL) reachable
      2. Running local ports (20128, 8080..8090) reachable -> HTTP
      3. llama-cpp-python in-process
      4. Fallback HTTP
    """
    from .splash import probe_openai_models

    apple = is_apple_silicon()
    mlx_ok = mlx_runtime_available()
    llama_ok = llama_cpp_python_available()

    # Discover actively running local ports (e.g. 20128, 8080..8090)
    discovered_url = DEFAULT_BASE_URL
    env_url = (
        os.environ.get("EMOPYLAB_LLM_URL")
        or os.environ.get("EMOPYLAB_LMSTUDIO_URL")
    )
    if env_url:
        clean_env = env_url.rsplit("/chat/completions", 1)[0].rstrip("/")
        if probe_openai_models(clean_env, timeout=0.5).ready:
            discovered_url = clean_env

    if discovered_url == DEFAULT_BASE_URL:
        for port in (20128, 8080, 8088, 8087, 8086, 8085, 8084, 8083, 8082, 8081):
            url = f"http://127.0.0.1:{port}/v1"
            if probe_openai_models(url, timeout=0.3).ready:
                discovered_url = url
                break

    preferred = "mlx-lm" if apple else ("llama-cpp-python" if llama_ok else "http-fallback")
    auto_start = (
        "python -m mlx_lm server "
        f"--model {DEFAULT_MODEL} --port {DEFAULT_MLX_PORT}"
        if (apple and mlx_ok) else None
    )
    return {
        "apple_silicon": apple,
        "mlx_available": mlx_ok,
        "llama_cpp_available": llama_ok,
        "preferred_backend": preferred,
        "preferred_endpoint": discovered_url,
        "preferred_model": DEFAULT_MODEL,
        "in_process_gguf_repo": DEFAULT_GGUF_REPO,
        "in_process_gguf_filename": DEFAULT_GGUF_FILE_PATTERN,
        "auto_start_command": auto_start,
    }


def terminate_local_server(port: int = DEFAULT_MLX_PORT, timeout_sec: float = 3.0) -> bool:
    """Safely terminate any local server process listening on the given port across platforms.

    Uses psutil to find matching process IDs on Windows, macOS, and Linux,
    falling back to standard platform signals if needed.
    """
    terminated = False
    try:
        import psutil
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr and conn.laddr.port == port and conn.pid:
                try:
                    p = psutil.Process(conn.pid)
                    p.terminate()
                    try:
                        p.wait(timeout=timeout_sec)
                    except psutil.TimeoutExpired:
                        p.kill()
                    terminated = True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
    except Exception:
        # Fallback if psutil fails or permissions prevent full socket scan
        if shutil.which("lsof") is not None:
            try:
                cmd = f"lsof -ti:{int(port)} | xargs kill -9 2>/dev/null || true"
                subprocess.run(cmd, shell=True, timeout=int(timeout_sec))
                terminated = True
            except Exception:
                pass
    return terminated


def ensure_mlx_server_running(base_url: str = DEFAULT_BASE_URL,
                              model: str = DEFAULT_MODEL,
                              port: int = DEFAULT_MLX_PORT,
                              start_if_missing: bool = False) -> bool:
    """Probe the local server.  Optionally launch mlx-lm.server in the
    background if ``start_if_missing`` is True and the endpoint is
    unreachable.

    Returns True when the server is responsive.
    """
    from .splash import probe_openai_models

    if probe_openai_models(base_url, timeout=0.75).ready:
        return True
    if not (start_if_missing and is_apple_silicon() and mlx_runtime_available()):
        return False
    # Try to launch in background using the modern `mlx_lm server` CLI
    cmd = [sys.executable, "-m", "mlx_lm", "server", "--model", model, "--port", str(port)]
    if shutil.which("nohup") is not None:
        subprocess.Popen(
            ["nohup"] + cmd,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    else:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    return True


# ---------------------------------------------------------------------------
# HTTP transport — robust to SSE streaming and non-streaming responses
# ---------------------------------------------------------------------------

def _post_chat_completion(url: str, payload: Dict[str, Any],
                          timeout: float) -> tuple[Optional[str], str, str]:
    """POST a chat completion and return ``(content, status, detail)``.

    The status is intentionally explicit so a scientific caller can
    distinguish transport failure, an empty response, and a response that
    cannot be interpreted as an OpenAI-compatible completion.
    """
    body = json.dumps(payload)
    try:
        proc = subprocess.run(
            ["curl", "-s", "--connect-timeout", "3",
             "--max-time", str(int(timeout)),
             "-H", "Content-Type: application/json",
             "-d", body, url],
            capture_output=True, text=True, timeout=timeout + 5,
        )
    except subprocess.TimeoutExpired as exc:
        return None, "transport_error", f"curl timed out: {exc}"
    except FileNotFoundError as exc:
        return None, "transport_error", f"curl unavailable: {exc}"
    except OSError as exc:
        return None, "transport_error", str(exc)

    if proc.returncode != 0:
        return None, "transport_error", (proc.stderr or f"curl exit {proc.returncode}").strip()
    raw = (proc.stdout or "").strip()
    if not raw:
        return None, "no_response", "empty HTTP response body"
    content = _extract_content(raw)
    if not content:
        return None, "invalid_response", "response did not contain assistant content"
    return content, "success", ""


def _extract_content(raw: str) -> str:
    """Extract the assistant content from either a JSON body or SSE stream."""
    if raw.startswith("data:"):
        parts = []
        for line in raw.split("\n"):
            line = line.strip()
            if not line.startswith("data:"):
                continue
            chunk_str = line[len("data:"):].strip()
            if chunk_str == "[DONE]":
                break
            try:
                chunk = json.loads(chunk_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                p = delta.get("content", "")
                if p:
                    parts.append(p)
            except (json.JSONDecodeError, IndexError, KeyError):
                continue
        content = "".join(parts).strip()
    else:
        try:
            body = json.loads(raw)
            content = body["choices"][0]["message"]["content"].strip()
        except (json.JSONDecodeError, KeyError, IndexError):
            return ""
    # Strip markdown fences
    if content.startswith("```"):
        lines = [l for l in content.split("\n") if not l.startswith("```")]
        content = "\n".join(lines).strip()
    return content


# ---------------------------------------------------------------------------
# Public client
# ---------------------------------------------------------------------------

class LocalLLMClient:
    """Uniform LLM client for EmoPyLab algorithms.

    Backend selection (highest priority first):
      1. **Apple Silicon + mlx-lm server reachable**:
         uses the HTTP OpenAI-compatible endpoint at ``DEFAULT_BASE_URL``
         (``localhost:20128/v1``).  This is the canonical Apple Silicon
         stack (mlx-lm serves ``Qwen2.5-0.5B-Instruct-4bit``).
      2. **Non-Apple-Silicon + ``llama-cpp-python`` installed**:
         uses the in-process GGUF runtime (``Llama.from_pretrained``) so
         no external HTTP server is needed on Windows / Linux / Intel macOS.
      3. **Any other reachable OpenAI-compatible HTTP server** as a final
         fallback (LM Studio, llama.cpp server, vLLM, etc.).

    All algorithms that integrate LLMs (MaACO, LARC-NSGA3, future
    meta-learning agents) should share a single ``LocalLLMClient``
    instance to amortize connection setup.
    """

    def __init__(self,
                 base_url: Optional[str] = None,
                 model: str = DEFAULT_MODEL,
                 timeout: float = DEFAULT_TIMEOUT,
                 temperature: float = DEFAULT_TEMPERATURE,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 auto_start: bool = False,
                 use_inprocess_if_available: bool = True,
                 gguf_repo: str = DEFAULT_GGUF_REPO,
                 gguf_filename: str = DEFAULT_GGUF_FILE_PATTERN,
                 gguf_n_ctx: int = 512,
                 gguf_n_threads: int = 8):
        self.backend_info = auto_backend_status()
        env_url = (
            os.environ.get("EMOPYLAB_LLM_URL")
            or os.environ.get("EMOPYLAB_LMSTUDIO_URL")
        )
        target_url = str(base_url or env_url or self.backend_info.get("preferred_endpoint") or DEFAULT_BASE_URL)
        from .splash import probe_openai_models
        if not probe_openai_models(target_url, timeout=0.2).ready and self.backend_info.get("preferred_endpoint"):
            target_url = str(self.backend_info["preferred_endpoint"])
        if target_url.endswith("/chat/completions"):
            target_url = target_url[:-len("/chat/completions")]
        self.base_url = target_url
        self.model = str(model)
        self.timeout = float(timeout)
        self.temperature = float(max(0.0, min(temperature, 1.0)))
        self.max_tokens = int(max_tokens)
        self._resolved_model: Optional[str] = None
        self.call_count = 0
        self.last_call_status: Dict[str, Any] = {
            "status": "not_called",
            "detail": "",
            "latency_ms": 0.0,
            "model": self.model,
            "backend": "unresolved",
        }
        # GGUF in-process settings (used only on non-Apple-Silicon hosts)
        self.gguf_repo = gguf_repo
        self.gguf_filename = gguf_filename
        self.gguf_n_ctx = int(gguf_n_ctx)
        self.gguf_n_threads = int(gguf_n_threads)
        self._inprocess: Any = None  # type: ignore[var-annotated]
        # Priority 1: Check if an HTTP server (llama-server or mlx-lm) is already reachable on self.base_url
        probe = probe_openai_models(self.base_url, timeout=0.3)
        if not probe.ready and use_inprocess_if_available and not is_apple_silicon() \
                and llama_cpp_python_available():
            try:
                from llama_cpp import Llama  # type: ignore[import-not-found]
                # Priority 1A: Strictly use the pre-downloaded local GGUF model in models/ directory
                local_models_dir = Path(__file__).resolve().parents[2] / "models"
                local_gguf_path = local_models_dir / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
                if local_gguf_path.exists():
                    self._inprocess = Llama(
                        model_path=str(local_gguf_path),
                        n_ctx=self.gguf_n_ctx,
                        n_threads=self.gguf_n_threads,
                        n_gpu_layers=-1,
                        verbose=False,
                    )
                    logger.info("LocalLLMClient: loaded strictly from %s", local_gguf_path)
                else:
                    logger.warning("LocalLLMClient: local model %s not found in models/", local_gguf_path)
            except Exception as exc:
                logger.warning(
                    "Failed to load in-process GGUF model (%s); "
                    "LocalLLMClient will fall back to HTTP.",
                    exc,
                )
                self._inprocess = None
        # Optionally boot the local mlx-lm server (Apple Silicon only)
        if auto_start and is_apple_silicon():
            ensure_mlx_server_running(self.base_url, self.model,
                                       auto_start=True)
        # Resolve a concrete model id from /models when using HTTP.
        # Optimization: when the in-process backend is active, no HTTP
        # resolution is needed (and we have no /models endpoint to query).
        if self._inprocess is None:
            self._resolved_model = self._resolve_model_id()
        else:
            self._resolved_model = f"{self.gguf_repo}::{self.gguf_filename}"

    @property
    def backend_name(self) -> str:
        if self._inprocess is not None:
            return "llama-cpp-python (in-process)"
        if is_apple_silicon():
            return "mlx-lm (HTTP server)"
        return "http-fallback"

    def _resolve_model_id(self) -> str:
        """Return the best available model id from /models, or our default.

        Resolution order:
          1. exact configured model id if available
          2. any id containing 'qwen' (Qwen2.5 family)
          3. any id containing 'vibecoding' (legacy)
          4. first id
          5. configured model as fallback
        """
        import urllib.request
        import urllib.error
        url = self.base_url.rstrip("/") + "/models"
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
            if self.model in ids:
                return self.model
            for needle in ("qwen", "vibecoding"):
                for mid in ids:
                    if needle in mid.lower():
                        return mid
            if ids:
                return ids[0]
        except (urllib.error.URLError, ConnectionError, TimeoutError,
                json.JSONDecodeError, OSError):
            pass
        return self.model

    @property
    def resolved_model(self) -> str:
        return self._resolved_model or self.model

    def _set_last_call_status(self, status: str, detail: str,
                              started_at: float) -> None:
        self.last_call_status = {
            "status": status,
            "detail": detail,
            "latency_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
            "model": self.resolved_model,
            "backend": self.backend_name,
        }

    def __call__(self, prompt: str,
                 system: Optional[str] = None,
                 json_only: bool = False) -> Optional[str]:
        """Dispatch a chat completion and expose a structured call status.

        The return type remains backward-compatible: assistant text on success,
        otherwise ``None``. Callers needing the reason inspect
        ``last_call_status``.
        """
        started_at = time.perf_counter()
        self.call_count += 1
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # ----- In-process branch (Windows / Linux / Intel macOS) -----
        if self._inprocess is not None:
            try:
                result = self._inprocess.create_chat_completion(
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = str(result["choices"][0]["message"]["content"]).strip()
                if content.startswith("```"):
                    lines = [line for line in content.split("\n") if not line.startswith("```")]
                    content = "\n".join(lines).strip()
                if not content:
                    self._set_last_call_status("no_response", "in-process model returned empty content", started_at)
                    return None
                self._set_last_call_status("success", "", started_at)
                return content
            except Exception as exc:
                self._set_last_call_status("inprocess_error", str(exc), started_at)
                return None

        # ----- HTTP server branch (Apple Silicon mlx-lm + universal) -----
        payload = {
            "model": self.resolved_model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        url = self.base_url.rstrip("/") + "/chat/completions"
        content, status, detail = _post_chat_completion(url, payload, self.timeout)
        self._set_last_call_status(status, detail, started_at)
        return content

    def json_call(self, prompt: str, system: Optional[str] = None) -> Optional[Any]:
        """Dispatch a chat completion and parse the response as JSON.

        Parsing status is recorded in ``last_call_status`` so an algorithm can
        distinguish invalid JSON from a transport or empty-response failure.
        """
        content = self(prompt, system=system, json_only=True)
        if not content:
            return None
        try:
            payload = json.loads(content)
            self.last_call_status["status"] = "success"
            return payload
        except json.JSONDecodeError:
            # Try to find the first JSON object/array in explanatory text.
            for opener, closer in (("{", "}"), ("[", "]")):
                start = content.find(opener)
                end = content.rfind(closer)
                if start != -1 and end > start:
                    try:
                        payload = json.loads(content[start:end + 1])
                        self.last_call_status["status"] = "success"
                        self.last_call_status["detail"] = "JSON recovered from explanatory content"
                        return payload
                    except json.JSONDecodeError:
                        continue
            self.last_call_status["status"] = "invalid_json"
            self.last_call_status["detail"] = "assistant content was not valid JSON"
            return None
