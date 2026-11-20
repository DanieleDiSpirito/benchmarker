from ..config import backend_settings


def get_backend(name: str):
    if name == "agy":
        from .agy import AgyBackend
        return AgyBackend(backend_settings("agy"))
    if name == "claude":
        from .claude import ClaudeBackend
        return ClaudeBackend(backend_settings("claude"))
    if name == "codex":
        from .codex import CodexBackend
        return CodexBackend(backend_settings("codex"))
    raise SystemExit(f"backend sconosciuto: {name}")
