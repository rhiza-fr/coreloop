"""JSON Schema tightening for strict structured output.

Pydantic's ``model_json_schema()`` is not accepted as-is by strict validators
(OpenAI's ``json_schema`` mode in particular). :func:`to_strict_schema` rewrites
the tree into a strict-legal shape.
"""

from typing import Any

from pydantic import BaseModel

__all__ = ["to_strict_schema"]


def _tighten(node: Any) -> None:
    """Recursively rewrite *node* in place into a strict-legal schema."""
    if isinstance(node, list):
        for item in node:
            _tighten(item)
        return
    if not isinstance(node, dict):
        return

    node.pop("title", None)
    node.pop("default", None)

    if isinstance(node.get("properties"), dict):
        node["additionalProperties"] = False
        node["required"] = list(node["properties"])

    for value in node.values():
        _tighten(value)


def to_strict_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Return *schema*'s JSON Schema, tightened for strict structured output.

    Every object with ``properties`` gets ``additionalProperties: false`` and a
    ``required`` list naming **all** its properties; ``title`` and ``default``
    are stripped everywhere. ``$defs`` / ``$ref`` are preserved -- only rewritten
    in place.

    ``X | None`` fields keep working: Pydantic emits ``anyOf: [..., {"type":
    "null"}]``, so the field stays formally required while remaining nullable.
    """
    result = schema.model_json_schema()
    _tighten(result)
    return result
