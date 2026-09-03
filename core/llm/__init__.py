"""LLM-related modules.

Re-exports the local-LLM client (``LocalLLMClient``) used by every
algorithm in EmoPyLab that integrates a language-model meta-controller
(MaACO, LARC-NSGA3, etc.).  The client auto-detects Apple Silicon and
defaults to the mlx-lm runtime serving ``Qwen2.5-0.5B-Instruct-4bit``.
"""

from .local_llm import (
    DEFAULT_BASE_URL,
    DEFAULT_GGUF_FILE_PATTERN,
    DEFAULT_GGUF_REPO,
    DEFAULT_MLX_PORT,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    LocalLLMClient,
    auto_backend_status,
    ensure_mlx_server_running,
    is_apple_silicon,
    llama_cpp_python_available,
    mlx_runtime_available,
)
from .splash import (
    EndpointProbe,
    LLMStartupUpdate,
    LocalLLMStartupCoordinator,
    EmoPyLabStartupSplash,
    SplashDecision,
    base_url_for_port,
    ensure_llm_ready_or_skip,
    find_available_local_port,
    probe_openai_models,
    show_startup_splash,
    wait_for_port,
    wait_for_server,
)

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_GGUF_FILE_PATTERN",
    "DEFAULT_GGUF_REPO",
    "DEFAULT_MLX_PORT",
    "DEFAULT_MODEL",
    "DEFAULT_TIMEOUT",
    "LocalLLMClient",
    "auto_backend_status",
    "EndpointProbe",
    "LLMStartupUpdate",
    "LocalLLMStartupCoordinator",
    "SplashDecision",
    "base_url_for_port",
    "ensure_llm_ready_or_skip",
    "find_available_local_port",
    "probe_openai_models",
    "show_startup_splash",
    "ensure_mlx_server_running",
    "is_apple_silicon",
    "llama_cpp_python_available",
    "mlx_runtime_available",
    "wait_for_port",
    "wait_for_server",
]