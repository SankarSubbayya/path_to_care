"""Camera Capture MCP: ingests a frame snapped from the patient's phone camera.

The actual `getUserMedia` call lives in the browser (frontend-next/src/components/
CameraCapture.tsx). This module is the server-side half: given the captured
bytes (raw bytes, a data URL, or a path), it normalizes to RGB, optionally
saves an audit copy, and returns metadata the orchestrator can route to the
image classifier MCP. Keeping it as a named MCP — rather than inlining bytes
handling in the API route — gives the audit tab a discrete tool invocation to
display ('camera_capture: 1 frame, 1280x720, 184 KB, saved to evidence/...'),
which matches how the other four MCPs are presented.

Usage from the API route:
  result = capture(image_bytes=bytes_from_form, mime="image/jpeg",
                   save_dir="evidence/captures")
  # result.image (PIL.Image) -> classify(); result.meta -> audit trail
"""
from __future__ import annotations

import base64
import io
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Optional


_DATA_URL_RE = re.compile(r"^data:(?P<mime>[^;,]+)(?:;base64)?,(?P<data>.*)$", re.DOTALL)


@dataclass
class CaptureResult:
    image: object              # PIL.Image.RGB, or None if input was empty
    width: int
    height: int
    mime: str
    bytes_in: int
    saved_path: Optional[str]
    source: str                # "data_url" | "raw_bytes" | "path" | "empty"
    case_id: Optional[str]
    wall_seconds: float
    parse_ok: bool

    def meta(self) -> dict:
        m = asdict(self)
        m.pop("image", None)
        return m


def _decode_data_url(data_url: str) -> tuple[bytes, str]:
    m = _DATA_URL_RE.match(data_url.strip())
    if not m:
        raise ValueError("not a data URL")
    mime = m.group("mime") or "image/png"
    payload = m.group("data")
    raw = base64.b64decode(payload)
    return raw, mime


def capture(
    *,
    image_bytes: Optional[bytes] = None,
    data_url: Optional[str] = None,
    image_path: Optional[str] = None,
    mime: Optional[str] = None,
    save_dir: Optional[str] = None,
    case_id: Optional[str] = None,
) -> CaptureResult:
    """Ingest a captured frame and return a CaptureResult.

    Exactly one of image_bytes / data_url / image_path should be supplied.
    If save_dir is provided, persists the frame as PNG with a timestamped
    name so the audit tab can link to it.
    """
    from PIL import Image

    t0 = time.time()
    raw: bytes = b""
    source = "empty"
    used_mime = mime or "image/png"

    if data_url:
        raw, used_mime = _decode_data_url(data_url)
        source = "data_url"
    elif image_bytes:
        raw = image_bytes
        source = "raw_bytes"
    elif image_path:
        with open(image_path, "rb") as f:
            raw = f.read()
        source = "path"
        used_mime = mime or _guess_mime(image_path)

    if not raw:
        return CaptureResult(
            image=None, width=0, height=0, mime=used_mime, bytes_in=0,
            saved_path=None, source="empty", case_id=case_id,
            wall_seconds=round(time.time() - t0, 3), parse_ok=False,
        )

    img = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = img.size

    saved_path: Optional[str] = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        cid = (case_id or "anon").replace("/", "_")[:32]
        saved_path = os.path.join(save_dir, f"capture-{stamp}-{cid}.png")
        img.save(saved_path, format="PNG")

    return CaptureResult(
        image=img, width=w, height=h, mime=used_mime, bytes_in=len(raw),
        saved_path=saved_path, source=source, case_id=case_id,
        wall_seconds=round(time.time() - t0, 3), parse_ok=True,
    )


def _guess_mime(path: str) -> str:
    p = path.lower()
    if p.endswith(".jpg") or p.endswith(".jpeg"):
        return "image/jpeg"
    if p.endswith(".webp"):
        return "image/webp"
    if p.endswith(".png"):
        return "image/png"
    return "application/octet-stream"


# Tool descriptor — mirrors how the orchestrator surfaces the other MCPs in the
# audit trail. The runtime invocation is the `capture` function above; this
# string is what the audit tab displays.
TOOL_SPEC = {
    "name": "camera_capture",
    "description": (
        "Ingest a frame snapped from the patient's phone camera (browser "
        "getUserMedia), normalize to RGB, optionally save an audit copy, "
        "and return metadata for the image_classifier MCP."
    ),
    "inputs": ["image_bytes | data_url | image_path", "mime?", "save_dir?", "case_id?"],
    "outputs": ["image (PIL.RGB)", "width", "height", "mime", "bytes_in", "saved_path", "source"],
}
