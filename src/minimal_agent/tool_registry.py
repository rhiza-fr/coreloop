"""Tool registration via the ``@tool`` decorator."""

import inspect
import types
import typing
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, overload


@dataclass
class ToolInfo:
    """Metadata describing a registered tool."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    fn: Callable[..., Coroutine[Any, Any, str]]
    """The async callable that executes this tool."""

    def __call__(self, *args: Any, **kwargs: Any) -> Coroutine[Any, Any, str]:
        """Invoke the wrapped tool function.

        ``@tool`` returns the ``ToolInfo``, so making it callable keeps the
        decorated name usable directly -- ``await my_tool(...)`` delegates to
        the underlying function.
        """
        return self.fn(*args, **kwargs)


# Global registry: tool name -> ToolInfo
_TOOL_REGISTRY: dict[str, ToolInfo] = {}


def _infer_parameters(fn: Callable) -> dict[str, Any]:
    """Build a JSON Schema for *fn* from its type annotations."""
    sig = inspect.signature(fn)
    hints = {k: v for k, v in (getattr(fn, "__annotations__", {}) or {}).items() if k != "return"}

    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name == "return":
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue  # *args, **kwargs not supported yet
        if param.default is inspect.Parameter.empty:
            required.append(name)
        type_hint = hints.get(name)
        js_type = _pytype_to_jsonschema(type_hint) if type_hint else {}
        prop: dict[str, Any] = js_type.copy()
        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        properties[name] = prop

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _pytype_to_jsonschema(tp: type) -> dict[str, Any]:
    """Best-effort Python type -> JSON Schema type mapping."""
    origin = typing.get_origin(tp)
    args = typing.get_args(tp) if origin else ()

    # Union types like int | None (types.UnionType) or Optional[int] (typing.Union)
    if origin is types.UnionType or origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            # Optional[X] -> use the schema for X, allow null via default
            return _pytype_to_jsonschema(non_none[0])
        # Mixed union -- fall through to generic string
        return {"type": "string"}

    if origin is typing.Literal:
        # All Literal values must be the same JSON type for a clean enum
        values = list(args)
        if all(isinstance(v, str) for v in values):
            return {"type": "string", "enum": values}
        if all(isinstance(v, int) for v in values):
            return {"type": "integer", "enum": values}
        return {"enum": values}
    if origin is list:
        item_tp = args[0] if args else str
        return {"type": "array", "items": _pytype_to_jsonschema(item_tp)}
    if origin is dict:
        return {"type": "object"}
    if origin is str or tp is str:
        return {"type": "string"}
    if origin is int or tp is int:
        return {"type": "integer"}
    if origin is float or tp is float:
        return {"type": "number"}
    if origin is bool or tp is bool:
        return {"type": "boolean"}
    warnings.warn(
        f"_pytype_to_jsonschema: unrecognised type {tp!r}, falling back to string. "
        "Consider using str, int, float, bool, list[...], dict, or Optional[...] instead.",
        stacklevel=3,
    )
    return {"type": "string"}


@overload
def tool(
    fn: Callable[..., Coroutine[Any, Any, str]],
    *,
    name: str | None = None,
    description: str | None = None,
    allow_override: bool = False,
) -> ToolInfo:
    """Use as @tool directly on a function."""
    ...


@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    allow_override: bool = False,
) -> Callable[[Callable[..., Coroutine[Any, Any, str]]], ToolInfo]:
    """Use as @tool(...) with keyword arguments."""
    ...


def tool(
    fn: Callable[..., Coroutine[Any, Any, str]] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    allow_override: bool = False,
) -> ToolInfo | Callable[[Callable[..., Coroutine[Any, Any, str]]], ToolInfo]:
    """Decorator to register an async function as a tool.

    Usage::

        @tool
        async def read(path: str) -> str:
            ...

        @tool(name="my_read", description="Read a file")
        async def my_read(path: str) -> str:
            ...

    Raises ``ValueError`` if a tool with the same name is already registered,
    unless ``allow_override=True`` is passed.
    """

    def register(f: Callable[..., Coroutine[Any, Any, str]]) -> ToolInfo:
        """Build and register a ToolInfo for the decorated function."""
        tool_name = name or getattr(f, "__name__", "")
        if tool_name in _TOOL_REGISTRY and not allow_override:
            raise ValueError(
                f"Tool {tool_name!r} is already registered. "
                "Use a different name or pass allow_override=True."
            )
        tool_desc = description or (f.__doc__ and f.__doc__.strip()) or ""
        info = ToolInfo(
            name=tool_name,
            description=tool_desc,
            parameters=_infer_parameters(f),
            fn=f,
        )
        _TOOL_REGISTRY[tool_name] = info
        return info

    if fn is not None:
        return register(fn)
    return register


def clear_registry() -> None:
    """Remove all registered tools. Primarily for use in tests."""
    _TOOL_REGISTRY.clear()


def get_tool(name: str) -> ToolInfo | None:
    """Lookup a tool by name."""
    return _TOOL_REGISTRY.get(name)


def list_tools() -> list[ToolInfo]:
    """Return all registered tools."""
    return list(_TOOL_REGISTRY.values())
