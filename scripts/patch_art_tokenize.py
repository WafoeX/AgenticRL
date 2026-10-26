#!/usr/bin/env python3
"""Patch ART 0.5.17's tokenize.py to work with transformers 5.x.

Bug: art/preprocessing/tokenize.py line ~237 assigns
    token_ids = cast(list[int], tokenizer.apply_chat_template(...))
but the `cast` is only a type hint — at runtime with transformers 5.x,
apply_chat_template(..., tools=...) returns a BatchEncoding wrapper,
not a plain list[int]. Then line ~248 calls `.index()` on it, which
fails because BatchEncoding has no `.index()` method.

Fix: insert a normalization block that coerces token_ids to list[int]
right after the assignment.

Run once per pod:
    python3 scripts/patch_art_tokenize.py

Idempotent: detects if already patched and skips.
"""
import sys
from pathlib import Path


# 中文注释：通过已安装的 art 包定位 site-packages 中实际使用的 tokenize.py。
def find_tokenize_py() -> Path:
    """Find ART's tokenize.py in site-packages."""
    import art.preprocessing.tokenize  # type: ignore[import-not-found]
    return Path(art.preprocessing.tokenize.__file__)


# 中文注释：补丁入口负责检查版本、确认标记位置并以幂等方式写入兼容逻辑。
def main() -> int:
    try:
        path = find_tokenize_py()
    except ImportError:
        print("ERROR: art package not importable. Run after installing openpipe-art[backend].")
        return 1

    print(f"Patching: {path}")
    src = path.read_text()

# 中文注释：重复执行脚本不会重复插入补丁，保证部署流程安全。
    # Already patched?
    if "# Patch (transformers 5.x): normalize token_ids" in src:
        print("ALREADY PATCHED — no changes.")
        return 0

    marker = "    )\n    assistant_mask: list[int] = [0] * len(token_ids)"
    if marker not in src:
        print("ERROR: expected marker not found. ART may have been updated.")
        print(f"Looked for: {marker!r}")
        return 1

    # Insert the normalization block between the close-paren and assistant_mask line.
    replacement = (
        "    )\n"
        "    # Patch (transformers 5.x): normalize token_ids to list[int].\n"
        "    # apply_chat_template may return a BatchEncoding wrapper when tools= is\n"
        "    # passed; the cast() above is a type hint and doesn't convert at runtime.\n"
        "    if hasattr(token_ids, \"input_ids\"):\n"
        "        token_ids = token_ids.input_ids\n"
        "    if not isinstance(token_ids, list):\n"
        "        token_ids = list(token_ids)\n"
        "    assistant_mask: list[int] = [0] * len(token_ids)"
    )

    new_src = src.replace(marker, replacement, 1)
    path.write_text(new_src)
    print("PATCH APPLIED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
