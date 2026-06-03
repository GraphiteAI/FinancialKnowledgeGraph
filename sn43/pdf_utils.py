from __future__ import annotations

from functools import lru_cache

_PDF_MAGIC = b"%PDF-"


def looks_like_pdf(data: bytes) -> bool:
    if not data:
        return False
    head = data[:1024].lstrip(b"\x00\r\n\t \xef\xbb\xbf")
    return head.startswith(_PDF_MAGIC)


@lru_cache(maxsize=16)
def _extract_pdf(data: bytes) -> str:
    try:
        import io

        import pdfplumber
    except ImportError:
        return ""
    try:
        parts = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t:
                    parts.append(t)
        return "\n".join(parts)
    except Exception:
        return ""


def pdf_to_text(data: bytes) -> str:
    return _extract_pdf(data)


def bytes_to_text(data: bytes) -> str:
    if looks_like_pdf(data):
        text = pdf_to_text(data)
        if text:
            return text
    return data.decode("utf-8", errors="replace")
