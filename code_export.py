"""
code_export.py

Optional post-response step: if the model's reply contains fenced code
blocks, offer to save each one to a source file with a sensible name
and extension.

Security note: this module is *write-only* by design. It never opens,
reads, or lists any file other than the one it is about to create, and
the only filesystem check it performs is `Path.exists()` (to avoid a
silent overwrite) -- it does not read that file's contents. There is no
path traversal outside the target output directory: filenames are
sanitized before use, and the caller always passes a fixed output dir.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Fenced code block: ```lang\n...code...\n```
_CODE_BLOCK_RE = re.compile(r"```([A-Za-z0-9_+#-]*)\n(.*?)```", re.DOTALL)

# Language hint -> file extension. Extend as needed.
_EXTENSIONS = {
    "python": ".py", "py": ".py",
    "javascript": ".js", "js": ".js",
    "typescript": ".ts", "ts": ".ts",
    "jsx": ".jsx", "tsx": ".tsx",
    "bash": ".sh", "sh": ".sh", "shell": ".sh", "zsh": ".sh",
    "java": ".java",
    "c": ".c",
    "cpp": ".cpp", "c++": ".cpp",
    "csharp": ".cs", "cs": ".cs",
    "go": ".go", "golang": ".go",
    "rust": ".rs", "rs": ".rs",
    "ruby": ".rb", "rb": ".rb",
    "php": ".php",
    "swift": ".swift",
    "kotlin": ".kt",
    "html": ".html",
    "css": ".css",
    "json": ".json",
    "yaml": ".yaml", "yml": ".yaml",
    "sql": ".sql",
    "makefile": "",
    "": ".txt",
}

# Per-language patterns used to pull a logical name out of the code itself.
_NAME_PATTERNS = [
    re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE),          # python
    re.compile(r"^\s*class\s+([A-Za-z_]\w*)", re.MULTILINE),             # python/java/etc
    re.compile(r"^\s*function\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE),     # js
    re.compile(r"^\s*(?:export\s+)?const\s+([A-Za-z_]\w*)\s*=\s*\(", re.MULTILINE),  # js arrow fn
    re.compile(r"^\s*func\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE),         # go/swift
    re.compile(r"^\s*fn\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE),           # rust
    re.compile(r"^\s*public\s+class\s+([A-Za-z_]\w*)", re.MULTILINE),    # java/cs
]

_MAX_STEM_LEN = 40


@dataclass
class CodeBlock:
    language: str
    code: str
    suggested_name: str  # filename including extension, no directory


def extract_code_blocks(reply_text: str, user_msg: str = "") -> List[CodeBlock]:
    """Find fenced code blocks in a reply and attach a suggested filename to each."""
    blocks = []
    for match in _CODE_BLOCK_RE.finditer(reply_text):
        lang_hint = match.group(1).strip().lower()
        code = match.group(2)
        if not code.strip():
            continue
        ext = _EXTENSIONS.get(lang_hint, ".txt")
        stem = _suggest_stem(code, user_msg, lang_hint)
        blocks.append(CodeBlock(language=lang_hint or "text", code=code, suggested_name=f"{stem}{ext}"))
    return blocks


def _slugify(text: str, max_words: int = 5) -> str:
    words = re.findall(r"[A-Za-z0-9]+", text.lower())[:max_words]
    slug = "_".join(words) if words else "generated_code"
    return slug[:_MAX_STEM_LEN].strip("_") or "generated_code"


def _suggest_stem(code: str, user_msg: str, lang_hint: str) -> str:
    """Pick a short, logical name: prefer a def/class/func name found in the
    code, otherwise fall back to a slug of the user's request."""
    for pattern in _NAME_PATTERNS:
        m = pattern.search(code)
        if m:
            name = re.sub(r"(?<!^)(?=[A-Z])", "_", m.group(1)).lower()
            return name[:_MAX_STEM_LEN].strip("_") or _slugify(user_msg)
    return _slugify(user_msg)


def save_code_block(block: CodeBlock, output_dir: Path, filename: Optional[str] = None) -> Path:
    """Write a single code block to disk. Write-only: never reads existing
    file contents. If the chosen path already exists, a numeric suffix is
    appended rather than overwriting it."""
    output_dir.mkdir(parents=True, exist_ok=True)

    name = _sanitize_filename(filename) if filename else block.suggested_name
    target = output_dir / name

    counter = 1
    stem, ext = target.stem, target.suffix
    while target.exists():
        target = output_dir / f"{stem}_{counter}{ext}"
        counter += 1

    target.write_text(block.code, encoding="utf-8")
    return target


def _sanitize_filename(name: str) -> str:
    """Strip anything that could escape the output directory or isn't a
    plain filename character."""
    name = Path(name).name  # drop any directory components
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name or "generated_code.txt"
