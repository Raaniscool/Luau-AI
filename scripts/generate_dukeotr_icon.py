#!/usr/bin/env python3
"""Generate the original, dependency-free DukeOTR Windows icon.

The repository intentionally owns this artwork.  It uses only geometry drawn by this script;
there is no third-party logo, font, or trademarked branding.  The resulting ICO contains several
PNG image sizes accepted by modern Windows shells and by Tk on Windows.

Run ``python scripts/generate_dukeotr_icon.py`` to rewrite the checked-in icon, or pass
``--check`` in CI/tests to verify it has not drifted from its source artwork.
"""

from __future__ import annotations

import argparse
import binascii
import struct
import sys
import zlib
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT = _PROJECT_ROOT / "desktop_app" / "assets" / "dukeotr.ico"
_SIZES = (16, 24, 32, 48, 64, 128, 256)


def _blend(base: tuple[int, int, int, int], top: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Alpha composite ``top`` over ``base`` using integer math."""

    alpha = top[3]
    if alpha <= 0:
        return base
    if alpha >= 255:
        return top
    inverse = 255 - alpha
    return (
        (top[0] * alpha + base[0] * inverse + 127) // 255,
        (top[1] * alpha + base[1] * inverse + 127) // 255,
        (top[2] * alpha + base[2] * inverse + 127) // 255,
        min(255, alpha + (base[3] * inverse + 127) // 255),
    )


def _inside_rounded_rectangle(x: float, y: float, left: float, top: float, right: float, bottom: float, radius: float) -> bool:
    """Return whether a point belongs to a rounded rectangle."""

    if left + radius <= x <= right - radius or top + radius <= y <= bottom - radius:
        return left <= x <= right and top <= y <= bottom
    center_x = left + radius if x < left + radius else right - radius
    center_y = top + radius if y < top + radius else bottom - radius
    return (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2


def _inside_circle(x: float, y: float, center_x: float, center_y: float, radius: float) -> bool:
    return (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2


def _line_distance(x: float, y: float, start: tuple[float, float], end: tuple[float, float]) -> tuple[float, float]:
    """Return distance and interpolation position from a point to a finite segment."""

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return ((x - start[0]) ** 2 + (y - start[1]) ** 2) ** 0.5, 0.0
    position = max(0.0, min(1.0, ((x - start[0]) * dx + (y - start[1]) * dy) / length_squared))
    closest_x = start[0] + position * dx
    closest_y = start[1] + position * dy
    return ((x - closest_x) ** 2 + (y - closest_y) ** 2) ** 0.5, position


def _draw_pixel(x: float, y: float) -> tuple[int, int, int, int]:
    """Draw a single normalized point of the icon on a transparent background."""

    transparent = (0, 0, 0, 0)
    color = transparent

    # Navy rounded tile: intentionally original DukeOTR artwork, not copied branding.
    if _inside_rounded_rectangle(x, y, 8, 8, 248, 248, 50):
        color = (14, 28, 58, 255)

    # Blue inner tile keeps the D mark readable at small Windows icon sizes.
    if _inside_rounded_rectangle(x, y, 20, 20, 236, 236, 39):
        color = _blend(color, (28, 76, 146, 255))

    # A restrained circuit trace makes the mark feel like a local development workspace.
    trace_segments = (
        ((54.0, 65.0), (76.0, 43.0)),
        ((180.0, 43.0), (202.0, 65.0)),
        ((202.0, 191.0), (180.0, 213.0)),
    )
    for start, end in trace_segments:
        distance, position = _line_distance(x, y, start, end)
        if distance <= 5:
            shade = int(220 - 45 * position)
            color = _blend(color, (83, 189, shade, 220))
    for center_x, center_y in ((76, 43), (180, 43), (180, 213)):
        if _inside_circle(x, y, center_x, center_y, 9):
            color = _blend(color, (105, 218, 255, 255))

    # Stylized D, constructed from simple shapes so it remains original and font independent.
    if _inside_rounded_rectangle(x, y, 67, 66, 102, 191, 12):
        color = _blend(color, (244, 250, 255, 255))
    if _inside_circle(x, y, 119, 128, 65) and x >= 89:
        color = _blend(color, (244, 250, 255, 255))
    if _inside_circle(x, y, 119, 128, 38) and x >= 96:
        color = _blend(color, (28, 76, 146, 255))

    # Small warm signal dot distinguishes the icon without adding third-party branding.
    if _inside_circle(x, y, 193, 73, 8):
        color = _blend(color, (255, 194, 80, 255))
    return color


def _render(size: int) -> bytes:
    """Render with 4x supersampling for crisp small icon entries."""

    samples = 4
    pixels = bytearray()
    for y in range(size):
        for x in range(size):
            accumulated = [0, 0, 0, 0]
            for sub_y in range(samples):
                for sub_x in range(samples):
                    point_x = (x + (sub_x + 0.5) / samples) * 256 / size
                    point_y = (y + (sub_y + 0.5) / samples) * 256 / size
                    sample = _draw_pixel(point_x, point_y)
                    for index, channel in enumerate(sample):
                        accumulated[index] += channel
            divisor = samples * samples
            pixels.extend(channel // divisor for channel in accumulated)
    return bytes(pixels)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)


def _encode_png(size: int) -> bytes:
    pixels = _render(size)
    rows = b"".join(b"\x00" + pixels[row * size * 4 : (row + 1) * size * 4] for row in range(size))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(rows, level=9))
        + _png_chunk(b"IEND", b"")
    )


def build_icon() -> bytes:
    """Build a multi-resolution ICO containing PNG payloads."""

    images = [(size, _encode_png(size)) for size in _SIZES]
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries: list[bytes] = []
    payloads: list[bytes] = []
    for size, payload in images:
        dimension = 0 if size == 256 else size
        entries.append(struct.pack("<BBBBHHII", dimension, dimension, 0, 0, 1, 32, len(payload), offset))
        payloads.append(payload)
        offset += len(payload)
    return header + b"".join(entries) + b"".join(payloads)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate or check the original DukeOTR Windows icon.")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT, help="ICO path to write/check")
    parser.add_argument("--check", action="store_true", help="Fail if the checked-in icon differs from generated bytes")
    args = parser.parse_args()

    expected = build_icon()
    output = args.output.resolve()
    if args.check:
        if not output.exists():
            print(f"DukeOTR icon is missing: {output}", file=sys.stderr)
            return 1
        if output.read_bytes() != expected:
            print(f"DukeOTR icon differs from generated artwork: {output}", file=sys.stderr)
            return 1
        print(f"DukeOTR icon matches generated artwork: {output}")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(expected)
    print(f"Wrote DukeOTR icon ({len(expected)} bytes): {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
