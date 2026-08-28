"""JSON Schema tightening for strict structured output.

Pydantic's ``model_json_schema()`` is not accepted as-is by strict validators
(OpenAI's ``json_schema`` mode in particular). :func:`to_strict_schema` rewrites
the tree into a strict-legal shape.
"""

from typing import Any

from pydantic import BaseModel

__all__ = ["to_strict_schema"]

# Annotation keywords strict validators reject. Stripped from schema *nodes*
# only -- never from the name maps below, where these appear as field names.
_STRIP_KEYWORDS = ("title", "default")

# Dicts keyed by user-chosen names rather than JSON Schema keywords. Their keys
# are data; only their values are schema nodes to recurse into.
_NAME_MAPS = ("properties", "$defs", "definitions", "patternProperties")


def _tighten(node: Any) -> None:
    """Recursively rewrite *node* in place into a strict-legal schema."""
    if isinstance(node, list):
        for item in node:
            _tighten(item)
        return
    if not isinstance(node, dict):
        return

    for keyword in _STRIP_KEYWORDS:
        node.pop(keyword, None)

    properties = node.get("properties")
    if isinstance(properties, dict):
        node["additionalProperties"] = False
        node["required"] = list(properties)

    for key, value in node.items():
        if key in _NAME_MAPS and isinstance(value, dict):
            # Recurse into the definitions, not over their names. A field called
            # "title" or "default" must survive keyword stripping.
            for subschema in value.values():
                _tighten(subschema)
        else:
            _tighten(value)


def to_strict_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Return *schema*'s JSON Schema, tightened for strict structured output.

    Every object with ``properties`` gets ``additionalProperties: false`` and a
    ``required`` list naming **all** its properties; ``title`` and ``default``
    are stripped from schema nodes. ``$defs`` / ``$ref`` are preserved -- only
    rewritten in place.

    Field names are never treated as keywords: a model with a field called
    ``title`` or ``default`` keeps it.

    ``X | None`` fields keep working: Pydantic emits ``anyOf: [..., {"type":
    "null"}]``, so the field stays formally required while remaining nullable.
    """
    result = schema.model_json_schema()
    _tighten(result)
    return result
