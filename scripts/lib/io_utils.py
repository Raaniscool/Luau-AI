"""Small, dependency-free file and JSON helpers used by every pipeline stage."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


class JsonlError(ValueError):
    """Raised when a JSONL artifact is malformed, with a useful source location."""


def utc_now() -> str:
    """Return an unambiguous, sortable UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_json(path: str | Path) -> Any:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise FileNotFoundError(f"Required file does not exist: {source}") from None
    except json.JSONDecodeError as exc:
        raise JsonlError(f"Invalid JSON in {source}: {exc}") from exc


def write_json_atomic(path: str | Path, value: Any, *, indent: int = 2) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=indent, sort_keys=True) + "\n"
    _atomic_write(destination, payload)


def read_jsonl(path: str | Path, *, allow_blank: bool = True) -> Iterator[dict[str, Any]]:
    """Yield objects from a JSONL file and fail with the precise invalid line."""
    source = Path(path)
    try:
        handle = source.open("r", encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Required JSONL file does not exist: {source}") from None
    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line and allow_blank:
                continue
            if not line:
                raise JsonlError(f"Blank JSONL record at {source}:{line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JsonlError(f"Invalid JSON at {source}:{line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise JsonlError(f"JSONL record at {source}:{line_number} must be an object")
            yield value


def write_jsonl_atomic(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    """Write a complete JSONL artifact atomically and return its record count."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as handle:
        temp_path = Path(handle.name)
        try:
            for record in records:
                handle.write(canonical_json(record))
                handle.write("\n")
                count += 1
        except Exception:
            handle.close()
            temp_path.unlink(missing_ok=True)
            raise
    os.replace(temp_path, destination)
    return count


def _atomic_write(destination: Path, payload: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(payload)
    os.replace(temp_path, destination)


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model response without executing it.

    Models occasionally wrap JSON in Markdown or precede it with a short sentence. This
    function deliberately accepts only a JSON object, not arbitrary Python literals.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    decoder = json.JSONDecoder()
    starts = [match.start() for match in re.finditer(r"\{", cleaned)]
    for start in starts:
        try:
            value, _end = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise JsonlError("The model response did not contain a parseable JSON object")


def text_from_message(record: dict[str, Any], role: str) -> str:
    for message in record.get("messages", []):
        if isinstance(message, dict) and message.get("role") == role:
            content = message.get("content", "")
            return content if isinstance(content, str) else ""
    return ""


def replace_message_content(record: dict[str, Any], role: str, content: str) -> dict[str, Any]:
    """Return a shallowly reconstructed record with one chat message replaced."""
    clone = dict(record)
    messages: list[dict[str, Any]] = []
    replaced = False
    for message in record.get("messages", []):
        copied = dict(message)
        if copied.get("role") == role and not replaced:
            copied["content"] = content
            replaced = True
        messages.append(copied)
    if not replaced:
        raise JsonlError(f"Cannot replace absent {role!r} message")
    clone["messages"] = messages
    return clone


def slugify(value: str, *, max_length: int = 72) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return (normalized[:max_length].rstrip("-") or "artifact")
