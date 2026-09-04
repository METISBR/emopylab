"""Optional local-AI startup flow for EmoPyLab.

The scientific workspace never requires a local LLM. This module provides a
small visual bootstrap surface, lets a user opt into MLX-LM per session,
validates OpenAI-compatible endpoints strictly, and starts model loading in
the background only after that explicit choice.
"""

from __future__ import annotations

import json
import logging
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MLX_PORT = 8080
DEFAULT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
StatusCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class EndpointProbe:
    """Validated result of checking an OpenAI-compatible models endpoint."""

    status: str
    base_url: str
    models: tuple[str, ...] = ()
    detail: str = ""

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "base_url": self.base_url,
            "models": list(self.models),
            "detail": self.detail,
            "ready": self.ready,
        }


@dataclass(frozen=True)
class SplashDecision:
    """The session-level local-AI decision made in the startup splash."""

    use_local_ai: bool
    base_url: str
    port: int
    already_ready: bool = False
    probe_status: str = "not_listening"


@dataclass(frozen=True)
class LLMStartupUpdate:
    """One visible lifecycle transition emitted by local-AI startup."""

    stage: str
    message: str
    state: str = "working"
    elapsed_seconds: int = 0
    progress: int | None = None
    terminal: bool = False
    ready: bool = False
    base_url: str = DEFAULT_BASE_URL

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "message": self.message,
            "state": self.state,
            "elapsed_seconds": self.elapsed_seconds,
            "progress": self.progress,
            "terminal": self.terminal,
            "ready": self.ready,
            "base_url": self.base_url,
        }


def base_url_for_port(port: int) -> str:
    """Return the canonical local OpenAI-compatible URL for a port."""
    return f"http://127.0.0.1:{int(port)}/v1"


def probe_openai_models(base_url: str, timeout: float = 0.75) -> EndpointProbe:
    """Strictly validate ``GET /models`` instead of trusting HTTP 200 alone."""
    normalized = str(base_url).rstrip("/")
    url = f"{normalized}/models"
    try:
        with urllib.request.urlopen(url, timeout=max(0.05, float(timeout))) as response:
            content_type = str(response.headers.get("Content-Type", "")).lower()
            raw = response.read(256 * 1024)
    except urllib.error.HTTPError as exc:
        return EndpointProbe("invalid_response", normalized, detail=f"HTTP {exc.code}")
    except (urllib.error.URLError, ConnectionError, TimeoutError, socket.timeout, OSError) as exc:
        return EndpointProbe("not_listening", normalized, detail=str(exc))

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        status = "wrong_service" if "text/html" in content_type else "invalid_response"
        detail = "Endpoint returned HTML, not an OpenAI-compatible model list." if status == "wrong_service" else "Endpoint did not return JSON."
        return EndpointProbe(status, normalized, detail=detail)

    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        return EndpointProbe("invalid_response", normalized, detail="JSON did not contain a models data list.")

    model_ids = tuple(
        str(entry.get("id", "")).strip()
        for entry in payload["data"]
        if isinstance(entry, dict) and str(entry.get("id", "")).strip()
    )
    if not model_ids:
        return EndpointProbe("invalid_response", normalized, detail="Model list contained no model identifiers.")
    return EndpointProbe("ready", normalized, model_ids)


def wait_for_server(
    base_url: str,
    timeout: float = 30.0,
    poll_interval: float = 0.5,
    request_timeout: float = 0.75,
) -> bool:
    """Poll a strictly validated OpenAI-compatible ``/models`` endpoint."""
    deadline = time.monotonic() + max(0.0, float(timeout))
    while time.monotonic() < deadline:
        result = probe_openai_models(
            base_url,
            timeout=min(max(0.05, request_timeout), max(0.05, deadline - time.monotonic())),
        )
        if result.ready:
            return True
        time.sleep(min(max(0.05, poll_interval), max(0.05, deadline - time.monotonic())))
    return False


def wait_for_port(
    host: str,
    port: int,
    timeout: float = 5.0,
    poll_interval: float = 0.25,
) -> bool:
    """Open a TCP socket repeatedly until it succeeds or ``timeout`` expires."""
    deadline = time.monotonic() + max(0.0, float(timeout))
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, int(port)), timeout=0.5):
                return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            time.sleep(min(max(0.05, poll_interval), max(0.05, deadline - time.monotonic())))
    return False


def find_available_local_port(preferred_port: int = DEFAULT_MLX_PORT, span: int = 32) -> int | None:
    """Return a free loopback TCP port without altering any occupied service."""
    for port in range(int(preferred_port), int(preferred_port) + max(1, int(span))):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    return None


class LocalLLMStartupCoordinator:
    """Best-effort background startup for an explicitly requested local LLM."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        port: int = DEFAULT_MLX_PORT,
        readiness_timeout: float = 120.0,
        status_callback: StatusCallback | None = None,
        auto_start: bool = True,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.model = str(model)
        self.port = int(port)
        self.readiness_timeout = max(1.0, float(readiness_timeout))
        self.auto_start = bool(auto_start)
        self._status_callback = status_callback
        self._cancel_event = threading.Event()
        self._done_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._ready = False
        self._last_update: dict[str, Any] | None = None

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def running(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive())

    def start(self) -> "LocalLLMStartupCoordinator":
        with self._lock:
            if self.running or self._done_event.is_set():
                return self
            self._thread = threading.Thread(target=self._run, name="emopylab-local-llm-startup", daemon=True)
            self._thread.start()
        return self

    def cancel_waiting(self) -> None:
        """Stop waiting; never terminate a user-owned or shared MLX process."""
        self._cancel_event.set()

    def wait(self, timeout: float | None = None) -> bool:
        self._done_event.wait(timeout)
        return self._ready

    def run_blocking(self) -> bool:
        self._run()
        return self._ready

    def _emit(
        self,
        stage: str,
        message: str,
        *,
        started_at: float,
        state: str = "working",
        progress: int | None = None,
        terminal: bool = False,
        ready: bool = False,
    ) -> None:
        update = LLMStartupUpdate(
            stage=stage,
            message=message,
            state=state,
            elapsed_seconds=int(max(0.0, time.monotonic() - started_at)),
            progress=progress,
            terminal=terminal,
            ready=ready,
            base_url=self.base_url,
        ).as_dict()
        self._last_update = update
        if self._status_callback is None:
            print(f"[emopylab-startup] {message}", flush=True)
            return
        try:
            self._status_callback(update)
        except Exception:  # noqa: BLE001
            logger.exception("Local LLM startup status callback failed")

    def _finish(self, stage: str, message: str, *, state: str, started_at: float, ready: bool = False) -> None:
        self._ready = bool(ready)
        self._emit(stage, message, state=state, started_at=started_at, progress=100, terminal=True, ready=ready)
        self._done_event.set()

    def _run(self) -> None:
        started_at = time.monotonic()
        try:
            from .local_llm import auto_backend_status, ensure_mlx_server_running, is_apple_silicon

            status = auto_backend_status()
            self._emit("runtime", "Inspecting local AI runtime…", started_at=started_at, progress=10)
            probe = probe_openai_models(self.base_url, timeout=0.5)
            if probe.ready:
                self._finish("ready", f"Local AI is ready at {self.base_url}.", state="success", started_at=started_at, ready=True)
                return
            if self._cancel_event.is_set():
                self._finish("cancelled", "Local AI loading canceled. EmoPyLab remains fully available.", state="idle", started_at=started_at)
                return
            if not (self.auto_start and is_apple_silicon() and bool(status.get("mlx_available"))):
                self._finish("unavailable", "Local AI is unavailable. EmoPyLab remains fully available.", state="idle", started_at=started_at)
                return
            if probe.status == "wrong_service":
                self._finish("port_conflict", f"{self.base_url} belongs to another app; local AI was not started.", state="warning", started_at=started_at)
                return

            self._emit("starting", "Starting mlx-lm on Apple Silicon…", started_at=started_at, progress=35)
            if not ensure_mlx_server_running(self.base_url, self.model, self.port, start_if_missing=True):
                self._finish("unavailable", "Could not start local AI. EmoPyLab remains fully available.", state="warning", started_at=started_at)
                return

            deadline = time.monotonic() + self.readiness_timeout
            while not self._cancel_event.is_set() and time.monotonic() < deadline:
                elapsed = int(max(0.0, time.monotonic() - started_at))
                self._emit("loading", f"Loading Qwen2.5 locally — {elapsed} s elapsed. The workspace is ready to use.", started_at=started_at)
                if wait_for_server(self.base_url, timeout=min(0.75, max(0.05, deadline - time.monotonic())), poll_interval=0.1, request_timeout=0.35):
                    self._finish("ready", f"Local AI is ready at {self.base_url}.", state="success", started_at=started_at, ready=True)
                    return

            if self._cancel_event.is_set():
                self._finish("cancelled", "Local AI continues independently; EmoPyLab remains fully available.", state="idle", started_at=started_at)
            else:
                self._finish("timeout", "Local AI is still starting. EmoPyLab remains fully available.", state="warning", started_at=started_at)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Local AI startup coordinator failed")
            self._finish("error", f"Local AI startup could not be checked ({exc}). EmoPyLab remains fully available.", state="warning", started_at=started_at)


def _load_splash_tokens() -> dict[str, Any]:
    """Load EmoPyLab visual roles without making the headless path depend on Qt."""
    tokens: dict[str, Any] = {
        "primary": "#B45309",
        "primary_dark": "#92400E",
        "primary_subtle": "#2A1D0B",
        "success": "#10B981",
        "warning": "#FBBF24",
        "danger": "#EF4444",
        "surface": "#111827",
        "surface_variant": "#1F2937",
        "text_primary": "#F8FAFC",
        "text_secondary": "#CBD5E1",
        "text_muted": "#94A3B8",
        "border": "#334155",
        "border_light": "#253247",
        "font_family": "Segoe UI",
    }
    try:
        from styles import AppStyles  # type: ignore
        colors = AppStyles.colors
        typography = AppStyles.typography
        for key in tuple(tokens):
            if hasattr(colors, key):
                tokens[key] = getattr(colors, key)
        tokens["font_family"] = typography.font_family
    except Exception:
        pass
    return tokens


class EmoPyLabStartupSplash:
    """High-performance, asynchronous non-blocking splash screen for EmoPyLab.

    Displays an elegant dark glassmorphic startup dialog with live milestones:
      1. Initializing Tensor Runtime & Hardware Acceleration (CUDA, MLX, SIMD)
      2. Loading Local AI Engine (Qwen2.5-0.5B GGUF / MLX)
      3. Registering 298+ Metaheuristics & Analytical Suites
      4. Scientific Workspace Ready
    """

    def __init__(
        self,
        qt_app: Optional[Any] = None,
        *,
        preferred_port: int = DEFAULT_MLX_PORT,
        base_url: str | None = None,
    ) -> None:
        self.qt_app = qt_app
        self.port = int(preferred_port)
        self.target_url = str(base_url or base_url_for_port(self.port)).rstrip("/")
        self.probe = EndpointProbe("not_listening", self.target_url)
        self._widget: Any = None
        self._progress_bar: Any = None
        self._status_label: Any = None
        self._sub_label: Any = None
        self._step_label: Any = None
        self._llm_coordinator: LocalLLMStartupCoordinator | None = None
        self._create_ui()

    def _create_ui(self) -> None:
        if self.qt_app is None:
            return
        try:
            from PySide6.QtCore import Qt
            from PySide6.QtGui import QColor, QFont, QPixmap
            from PySide6.QtWidgets import (
                QFrame,
                QGraphicsDropShadowEffect,
                QHBoxLayout,
                QLabel,
                QProgressBar,
                QVBoxLayout,
                QWidget,
            )

            tokens = _load_splash_tokens()
            bg_color = "#0f172a"
            card_bg = "#1e293b"
            text_color = "#f8fafc"
            sub_color = "#94a3b8"
            milestone_color = "#cbd5e1"
            accent_color = "#f59e0b"
            accent_cyan = "#38bdf8"
            success_color = "#10b981"
            border_color = "#334155"

            self._widget = QWidget()
            self._widget.setWindowFlags(
                Qt.WindowType.SplashScreen
                | Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
            )
            self._widget.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self._widget.setFixedSize(580, 380)

            root_layout = QVBoxLayout(self._widget)
            root_layout.setContentsMargins(16, 16, 16, 16)
            root_layout.setSpacing(0)

            container = QFrame()
            container.setObjectName("SplashContainer")
            container.setStyleSheet(f"""
                QFrame#SplashContainer {{
                    background-color: {bg_color};
                    border: 1px solid {border_color};
                    border-radius: 12px;
                }}
            """)

            shadow = QGraphicsDropShadowEffect(container)
            shadow.setBlurRadius(28)
            shadow.setColor(QColor(0, 0, 0, 180))
            shadow.setOffset(0, 8)
            container.setGraphicsEffect(shadow)

            main_layout = QVBoxLayout(container)
            main_layout.setContentsMargins(28, 24, 28, 24)
            main_layout.setSpacing(14)

            # Header / Logo area
            header_layout = QHBoxLayout()
            header_layout.setSpacing(18)
            header_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

            logo_path = Path(__file__).resolve().parent.parent.parent / "emopylab.png"
            logo_label = QLabel()
            if logo_path.exists():
                pix = QPixmap(str(logo_path)).scaled(
                    120,
                    120,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                logo_label.setPixmap(pix)
            logo_label.setFixedSize(120, 120)
            logo_label.setStyleSheet("background: transparent; border: none;")
            header_layout.addWidget(logo_label)

            title_vbox = QVBoxLayout()
            title_vbox.setSpacing(4)
            title_vbox.setAlignment(Qt.AlignmentFlag.AlignVCenter)

            title_lbl = QLabel("EmoPyLab")
            title_lbl.setStyleSheet(
                f"font-size: 22px; font-weight: 800; color: {text_color}; letter-spacing: -0.5px; background: transparent; border: none;"
            )

            subtitle_lbl = QLabel("Tensor-Native Evolutionary Multi/Many-Objective Optimization")
            subtitle_lbl.setWordWrap(True)
            subtitle_lbl.setStyleSheet(
                f"font-size: 13px; font-weight: 500; color: {sub_color}; line-height: 1.3; background: transparent; border: none;"
            )

            badge_lbl = QLabel("Scientific Workstation & Offline AI Agent (Qwen2.5-0.5B)")
            badge_lbl.setStyleSheet(
                f"font-size: 11px; font-weight: 600; color: {accent_cyan}; background: transparent; border: none;"
            )

            title_vbox.addWidget(title_lbl)
            title_vbox.addWidget(subtitle_lbl)
            title_vbox.addWidget(badge_lbl)
            header_layout.addLayout(title_vbox, 1)

            main_layout.addLayout(header_layout)

            # Card container for loading status & progress bar
            card = QFrame()
            card.setObjectName("StatusCard")
            card.setStyleSheet(f"""
                QFrame#StatusCard {{
                    background-color: {card_bg};
                    border: 1px solid {border_color};
                    border-radius: 8px;
                }}
            """)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(16, 12, 16, 12)
            card_layout.setSpacing(8)

            status_header = QHBoxLayout()
            self._status_label = QLabel("Initializing Tensor Runtime & Hardware Acceleration...")
            self._status_label.setStyleSheet(
                f"font-size: 12px; font-weight: 600; color: {milestone_color}; background: transparent; border: none;"
            )
            status_header.addWidget(self._status_label, 1)

            self._step_label = QLabel("Phase 1/4")
            self._step_label.setStyleSheet(
                f"font-size: 11px; font-weight: 700; color: {accent_color}; background: transparent; border: none;"
            )
            status_header.addWidget(self._step_label, 0)
            card_layout.addLayout(status_header)

            self._progress_bar = QProgressBar()
            self._progress_bar.setRange(0, 100)
            self._progress_bar.setValue(15)
            self._progress_bar.setFixedHeight(6)
            self._progress_bar.setTextVisible(False)
            self._progress_bar.setStyleSheet(f"""
                QProgressBar {{
                    background-color: #334155;
                    border-radius: 3px;
                    border: none;
                }}
                QProgressBar::chunk {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {accent_color}, stop:1 {accent_cyan});
                    border-radius: 3px;
                }}
            """)
            card_layout.addWidget(self._progress_bar)

            self._sub_label = QLabel("Detecting hardware acceleration (CUDA, MLX, SIMD)...")
            self._sub_label.setStyleSheet(
                f"font-size: 11px; color: {sub_color}; background: transparent; border: none;"
            )
            card_layout.addWidget(self._sub_label)

            main_layout.addWidget(card)

            # Footer note
            footer = QLabel("Offline Local AI Engine active from models/ • Zero cloud telemetry")
            footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
            footer.setStyleSheet(
                f"font-size: 11px; color: {success_color}; font-weight: 500; background: transparent; border: none;"
            )
            main_layout.addWidget(footer)

            root_layout.addWidget(container)

            # Center on screen
            screen = self.qt_app.primaryScreen()
            if screen:
                geo = screen.availableGeometry()
                x = geo.center().x() - 290
                y = geo.center().y() - 190
                self._widget.move(x, y)
        except Exception as exc:
            logger.debug("Could not build graphical splash screen: %s", exc)
            self._widget = None

    def show(self) -> "EmoPyLabStartupSplash":
        if self._widget is not None:
            try:
                self._widget.show()
                if self.qt_app is not None:
                    self.qt_app.processEvents()
            except Exception:
                pass
        return self

    def update_progress(
        self,
        value: int,
        message: str,
        detail: str | None = None,
        step_text: str | None = None,
    ) -> None:
        """Update the visual progress bar and message with non-blocking event pump."""
        if self._widget is not None:
            try:
                if self._progress_bar is not None:
                    self._progress_bar.setValue(int(value))
                if self._status_label is not None:
                    self._status_label.setText(str(message))
                if self._sub_label is not None and detail:
                    self._sub_label.setText(str(detail))
                if self._step_label is not None and step_text:
                    self._step_label.setText(str(step_text))
                if self.qt_app is not None:
                    self.qt_app.processEvents()
            except Exception:
                pass

    def update_step(self, step: int, detail: str | None = None, *, completed_through: int | None = None) -> None:
        milestones = [
            ("Initializing Tensor Runtime & Hardware Acceleration...", "Detecting PyTorch CUDA/MPS, JAX, Apple MLX, and CPU SIMD"),
            ("Loading Local AI Engine (Qwen2.5-0.5B GGUF / MLX)...", "Initializing offline AI model from models/ directory"),
            ("Registering 298+ Metaheuristics & Analytical Suites...", "Registering multi-objective metaheuristics and benchmark suites"),
            ("Workspace Ready.", "Scientific Workspace Initialized"),
        ]
        idx = max(0, min(step - 1, len(milestones) - 1))
        title, default_detail = milestones[idx]
        pct = min(100, int(step * 25))
        self.update_progress(pct, title, detail or default_detail, step_text=f"Phase {step}/4")

    def start_background_llm_probe(self) -> None:
        """Probe or initialize the local LLM in background without blocking GUI."""
        if self._llm_coordinator is not None and self._llm_coordinator.running:
            return
        self._llm_coordinator = LocalLLMStartupCoordinator(
            base_url=self.target_url,
            port=self.port,
            readiness_timeout=120.0,
            auto_start=True,
        )
        self._llm_coordinator.start()

    def choose_local_ai(self) -> SplashDecision:
        return SplashDecision(
            use_local_ai=True,
            base_url=self.target_url,
            port=self.port,
            probe_status=self.probe.status,
        )

    def close(self) -> None:
        if self._widget is not None:
            try:
                self._widget.close()
                self._widget.deleteLater()
                self._widget = None
            except Exception:
                pass

    def complete(self) -> None:
        self.update_progress(100, "Workspace Ready.", "Launching EmoPyLab Scientific Desktop Workstation…", step_text="Ready")
        if self.qt_app is not None:
            try:
                self.qt_app.processEvents()
            except Exception:
                pass
        self.close()


def show_startup_splash(
    qt_app: Optional[Any],
    *,
    preferred_port: int = DEFAULT_MLX_PORT,
    base_url: str | None = None,
) -> SplashDecision:
    """Senior non-blocking splash screen that shows loading milestones and starts local AI."""
    splash = EmoPyLabStartupSplash(qt_app, preferred_port=preferred_port, base_url=base_url)
    splash.show()
    splash.update_progress(
        25,
        "Initializing Tensor Runtime & Hardware Acceleration...",
        "Detecting PyTorch CUDA/MPS, JAX, Apple MLX, and CPU SIMD",
        step_text="Phase 1/4",
    )
    splash.update_progress(
        50,
        "Loading Local AI Engine (Qwen2.5-0.5B GGUF / MLX)...",
        "Initializing offline AI model from models/ directory",
        step_text="Phase 2/4",
    )
    splash.update_progress(
        75,
        "Registering 298+ Metaheuristics & Analytical Suites...",
        "298+ multi-objective metaheuristics and analytical suites loaded",
        step_text="Phase 3/4",
    )
    decision = splash.choose_local_ai()
    splash.update_progress(
        100,
        "Workspace Ready.",
        "Opening EmoPyLab Scientific Desktop Workstation…",
        step_text="Phase 4/4",
    )
    splash.complete()
    return decision

def ensure_llm_ready_or_skip(
    qt_app: Optional[Any] = None,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    port: int = DEFAULT_MLX_PORT,
    timeout: float = 120.0,
    auto_start: bool = True,
) -> bool:
    """Legacy blocking API retained for CLI/headless callers only."""
    _ = qt_app
    coordinator = LocalLLMStartupCoordinator(
        base_url=base_url,
        model=model,
        port=port,
        readiness_timeout=timeout,
        auto_start=auto_start,
    )
    return coordinator.run_blocking()


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MLX_PORT",
    "DEFAULT_MODEL",
    "EndpointProbe",
    "LLMStartupUpdate",
    "EmoPyLabStartupSplash",
    "SplashDecision",
    "base_url_for_port",
    "ensure_llm_ready_or_skip",
    "find_available_local_port",
    "probe_openai_models",
    "show_startup_splash",
    "wait_for_port",
    "wait_for_server",
]
