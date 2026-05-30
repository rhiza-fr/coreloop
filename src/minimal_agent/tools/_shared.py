import os
from pathlib import Path

from ..registry import ToolInfo, _infer_parameters

_MAX_READ_BYTES = 10 * 1024 * 1024  # 10 MB


def _resolve_safe(requested: str, root: "Path | str") -> str:
    if not isinstance(requested, str):
        raise ValueError(f"path must be a string, got {type(requested).__name__}")

    raw = Path(requested)
    root_path = Path(root)
    if not raw.is_absolute():
        raw = root_path / raw

    try:
        resolved = raw.resolve(strict=False)
    except OSError as exc:
        raise ValueError(f"cannot resolve path {requested!r}: {exc}") from exc

    try:
        resolved.relative_to(root_path)
    except ValueError:
        raise ValueError(
            f"path traversal denied: {requested!r} resolves outside allowed root {root_path}"
        ) from None

    return str(resolved)


def _resolve_safe_strict(requested: str, root: "Path | str") -> str:
    resolved = _resolve_safe(requested, root)
    if not os.path.exists(resolved):
        raise ValueError(f"path does not exist: {requested!r}")
    return resolved


def _fmt_size(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _make_tool_info(fn, **overrides) -> ToolInfo:
    name = overrides.pop("name", fn.__name__)
    desc = overrides.pop("description", (fn.__doc__ or "").strip())
    params = _infer_parameters(fn)
    params.update(overrides.pop("parameters", {}))
    return ToolInfo(name=name, description=desc, parameters=params, fn=fn)
