#!/usr/bin/env python3
"""
Regenerate the Jupyter notebook with embedded plots from the marimo source.

Usage:
    python update_notebook.py

This script exports the marimo notebook to Jupyter format with plots embedded,
ensuring proper rendering on GitHub.
"""

import subprocess
import sys
from pathlib import Path


def main():
    print("🔄 Regenerating Jupyter notebook with embedded plots...")

    # Check if marimo source exists
    marimo_file = Path("visualizing_embeddings.py")
    if not marimo_file.exists():
        print(f"❌ Source file not found: {marimo_file}")
        return 1

    # Export with outputs included
    cmd = [
        "uv",
        "run",
        "marimo",
        "export",
        "ipynb",
        str(marimo_file),
        "-o",
        "__marimo__/visualizing_embeddings.ipynb",
        "-f",
        "--include-outputs",
    ]

    try:
        print(f"📝 Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        print("✅ Successfully regenerated notebook with embedded plots!")
        print("📁 Output: __marimo__/visualizing_embeddings.ipynb")
        print("🎯 The notebook should now render plots properly on GitHub")
        return 0

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to regenerate notebook: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return 1
    except FileNotFoundError:
        print("❌ Command not found. Make sure 'uv' and 'marimo' are installed.")
        print("💡 Try running: uv sync --dev")
        return 1


if __name__ == "__main__":
    sys.exit(main())
