#!/usr/bin/env python3
"""
Build script to regenerate HTML examples with fresh base64 encodings.

This ensures the embedded amplifier_webruntime.py is always in sync with source.
"""

import base64
import hashlib
import re
import sys
from pathlib import Path

BUNDLE_ROOT = Path(__file__).parent.parent
SRC_DIR = BUNDLE_ROOT / "src"
EXAMPLES_DIR = BUNDLE_ROOT / "examples"

def compute_hash(content: bytes) -> str:
    """Compute short hash for traceability."""
    return hashlib.sha256(content).hexdigest()[:8]

def encode_file(path: Path) -> tuple[str, str]:
    """Encode file to base64 and return (encoded, hash)."""
    content = path.read_bytes()
    encoded = base64.b64encode(content).decode('ascii')
    return encoded, compute_hash(content)

def update_example(html_path: Path, wheel_b64: str | None = None):
    """Update an HTML example with fresh base64 encodings."""
    
    # Read the HTML
    html = html_path.read_text()
    
    # Encode amplifier_webruntime.py
    browser_py = SRC_DIR / "amplifier_webruntime.py"
    if not browser_py.exists():
        print(f"ERROR: {browser_py} not found")
        sys.exit(1)
    
    browser_b64, browser_hash = encode_file(browser_py)
    
    # Replace the amplifier-browser-py script content
    # Pattern: <script id="amplifier-browser-py" type="text/plain">...content...</script>
    pattern = r'(<script id="amplifier-browser-py" type="text/plain">)\s*.*?\s*(</script>)'
    replacement = f'\\1{browser_b64}\\2'
    
    new_html, count = re.subn(pattern, replacement, html, flags=re.DOTALL)
    
    if count == 0:
        print(f"WARNING: No amplifier-browser-py script found in {html_path}")
        return False
    
    # Write back
    html_path.write_text(new_html)
    print(f"Updated {html_path.name}:")
    print(f"  - amplifier_webruntime.py: hash={browser_hash}")
    
    return True

def main():
    print("Building browser bundle examples...")
    print(f"Source: {SRC_DIR}")
    print(f"Examples: {EXAMPLES_DIR}")
    print()
    
    # Find all HTML examples
    examples = list(EXAMPLES_DIR.glob("*.html"))
    
    if not examples:
        print("No HTML examples found")
        return
    
    for example in examples:
        update_example(example)
    
    print()
    print("Done! Remember to commit the updated examples.")

if __name__ == "__main__":
    main()
