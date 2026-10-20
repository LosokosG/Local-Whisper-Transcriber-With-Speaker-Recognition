"""Recursive redaction for diagnostics that may cross a persistence boundary."""

from __future__ import annotations

import ast
import json
import math
import re
from collections.abc import Mapping
from typing import Any

REDACTED = "[REDACTED]"

_URI_RE = re.compile(
    r"(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*)://(?P<remainder>[^\s\"'<>]+)"
)
_KEYED_VALUE_RE = re.compile(
    r"(?P<prefix>(?<![A-Za-z0-9_-])(?P<key_quote>[\"']?)"
    r"(?P<key>[A-Za-z][A-Za-z0-9_-]*(?:[ \t]+[A-Za-z][A-Za-z0-9_-]*)*)"
    r"(?P=key_quote)\s*[:=]\s*)"
    r"(?:"
    r"(?P<value_quote>[\"'])(?:(?:basic|bearer|bot)\s+)?"
    r"(?P<quoted_secret>(?:\\.|(?!(?P=value_quote)).)*)(?P=value_quote)"
    r"|"
    r"(?:(?:basic|bearer|bot)\s+)?"
    rf"(?P<unquoted_secret>{re.escape(REDACTED)}|[^\s&,;}}\]\)\"']+)"
    r")",
    re.IGNORECASE,
)
_CAMEL_BOUNDARY_RE = re.compile(
    r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
)
_KEY_SEPARATOR_RE = re.compile(r"[^A-Za-z0-9]+")
_SENSITIVE_KEY_TERMINALS = frozenset(
    {
        "authorization",
        "authorizations",
        "credential",
        "credentials",
        "password",
        "passwords",
        "passwd",
        "secret",
        "secrets",
        "token",
        "tokens",
    }
)


def _redact_uri(match: re.Match[str]) -> str:
    scheme = match.group("scheme")
    remainder = match.group("remainder")
    if scheme.lower() == "file":
        return match.group(0)

    boundary_positions = [
        position
        for marker in ("/", "?", "#")
        if (position := remainder.find(marker)) >= 0
    ]
    authority_end = min(boundary_positions, default=len(remainder))
    authority = remainder[:authority_end]
    suffix = remainder[authority_end:]
    if "@" in authority:
        _, host = authority.rsplit("@", 1)
        authority = f"{REDACTED}@{host}"

    if "?" in suffix:
        path, query_and_fragment = suffix.split("?", 1)
        if "#" in query_and_fragment:
            query, fragment = query_and_fragment.split("#", 1)
            fragment = f"#{fragment}"
        else:
            query, fragment = query_and_fragment, ""
        redacted_parameters = []
        for parameter in query.split("&"):
            if not parameter:
                continue
            key = parameter.split("=", 1)[0]
            redacted_parameters.append(f"{key}={REDACTED}" if key else REDACTED)
        suffix = f"{path}?{'&'.join(redacted_parameters)}{fragment}"

    return f"{scheme}://{authority}{suffix}"


def _normalize_key(value: str) -> str:
    separated = _CAMEL_BOUNDARY_RE.sub("_", value)
    return _KEY_SEPARATOR_RE.sub("_", separated).strip("_").lower()


def _is_sensitive_key(value: str) -> bool:
    if value.strip().lower() in _SENSITIVE_KEY_TERMINALS:
        return True
    normalized = _normalize_key(value)
    ordered_parts = tuple(part for part in normalized.split("_") if part)
    if not ordered_parts:
        return False
    return ordered_parts[-1] in _SENSITIVE_KEY_TERMINALS or (
        len(ordered_parts) >= 2
        and ordered_parts[-2] == "api"
        and ordered_parts[-1] in {"key", "keys"}
    )


def _is_uri_key(value: str) -> bool:
    normalized = _normalize_key(value)
    return normalized == "uri" or normalized.endswith("_uri")


def _redact_keyed_value(match: re.Match[str]) -> str:
    quote = match.group("value_quote") or ""
    return f"{match.group('prefix')}{quote}{REDACTED}{quote}"


def _redact_keyed_values(value: str) -> str:
    pieces: list[str] = []
    copied_through = 0
    search_from = 0
    while match := _KEYED_VALUE_RE.search(value, search_from):
        if not _is_sensitive_key(match.group("key")):
            search_from = match.start() + 1
            continue
        pieces.append(value[copied_through : match.start()])
        pieces.append(_redact_keyed_value(match))
        copied_through = match.end()
        search_from = match.end()
    pieces.append(value[copied_through:])
    return "".join(pieces)


def _redact_structured_text(value: str) -> str | None:
    stripped = value.strip()
    if len(stripped) < 2 or (stripped[0], stripped[-1]) not in {
        ("{", "}"),
        ("[", "]"),
        ("(", ")"),
    }:
        return None
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return None
    if not isinstance(parsed, (Mapping, list, tuple)):
        return None
    safe = _redact_value(parsed, set())
    return json.dumps(
        safe,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
    )


def redact_text(value: str) -> str:
    """Remove known credential forms while retaining diagnostic context."""

    if not isinstance(value, str):
        raise TypeError("redact_text expects a string")
    structured = _redact_structured_text(value)
    if structured is not None:
        return structured
    redacted = _URI_RE.sub(_redact_uri, value)
    return _redact_keyed_values(redacted)


def _safe_mapping_key(value: Any) -> str:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, bytes):
        try:
            return redact_text(value.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ValueError("mapping key bytes must be valid UTF-8") from exc
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite numbers are not valid safe JSON mapping keys")
        return json.dumps(value, allow_nan=False)
    raise ValueError(f"unsupported mapping key type: {type(value).__name__}")


def _redact_value(value: Any, active_ids: set[int]) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, bytes):
        return redact_text(value.decode("utf-8", errors="replace"))
    if isinstance(value, BaseException):
        identity = id(value)
        if identity in active_ids:
            return "<recursive exception>"
        active_ids.add(identity)
        try:
            safe_args = tuple(_redact_value(arg, active_ids) for arg in value.args)
        finally:
            active_ids.remove(identity)
        if len(safe_args) == 1:
            message = str(safe_args[0])
        else:
            message = str(safe_args) if safe_args else ""
        return redact_text(f"{type(value).__name__}: {message}")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite numbers are not valid safe JSON values")
        return value
    if value is None or isinstance(value, (bool, int)):
        return value

    identity = id(value)
    if identity in active_ids:
        return "<recursive reference>"

    if isinstance(value, Mapping):
        active_ids.add(identity)
        try:
            result: dict[str, Any] = {}
            for key, item in value.items():
                safe_key = _safe_mapping_key(key)
                if safe_key in result:
                    raise ValueError(
                        f"mapping keys collide after normalization: {safe_key!r}"
                    )
                if (
                    _is_uri_key(safe_key)
                    and isinstance(item, str)
                    and _URI_RE.search(item)
                ):
                    result[safe_key] = redact_text(item)
                elif _is_sensitive_key(safe_key) or _is_uri_key(safe_key):
                    result[safe_key] = REDACTED
                else:
                    result[safe_key] = _redact_value(item, active_ids)
            return result
        finally:
            active_ids.remove(identity)

    if isinstance(value, list):
        active_ids.add(identity)
        try:
            return [_redact_value(item, active_ids) for item in value]
        finally:
            active_ids.remove(identity)
    if isinstance(value, tuple):
        active_ids.add(identity)
        try:
            return tuple(_redact_value(item, active_ids) for item in value)
        finally:
            active_ids.remove(identity)
    if isinstance(value, (set, frozenset)):
        active_ids.add(identity)
        try:
            return [_redact_value(item, active_ids) for item in value]
        finally:
            active_ids.remove(identity)

    return redact_text(str(value))


def redact_value(value: Any) -> Any:
    """Recursively return a credential-free JSON-compatible value."""

    return _redact_value(value, set())


redact = redact_value
