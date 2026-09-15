"""Rebuild every generated page on the site: the notes and the blog.

Usage::

    python build.py                # rebuild everything
    python build.py --serve        # rebuild, then preview at http://localhost:8000/
    python build.py --serve 8080   # same, on another port

The preview serves this folder exactly as GitHub Pages does, so root-relative
links such as ``/blog/`` and ``/mlclass/`` resolve. It does not rebuild on its
own: after editing a source file, run the build again and refresh the page.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BUILDS = [
    os.path.join(HERE, "mlclass", "build", "build.py"),
    os.path.join(HERE, "blog", "build.py"),
]


def build() -> int:
    """Run each generator in turn; stop at the first failure.

    Returns
    -------
    int
        Exit status of the first failing generator, or 0.
    """
    for script in BUILDS:
        print(f"== {os.path.relpath(script, HERE)}")
        result = subprocess.run([sys.executable, script], check=False)
        if result.returncode:
            return result.returncode
    return 0


def serve(port: int) -> None:
    """Serve the site folder on localhost until interrupted.

    Parameters
    ----------
    port : int
        TCP port to listen on.
    """
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=HERE)
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        print(
            f"\nServing {HERE}\n  http://localhost:{port}/\n  http://localhost:{port}/blog/\n  http://localhost:{port}/mlclass/\nCtrl-C to stop."
        )
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print()


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(
        description="Rebuild the site, optionally previewing it."
    )
    ap.add_argument(
        "--serve",
        nargs="?",
        const=8000,
        type=int,
        metavar="PORT",
        help="after building, serve on localhost (default port 8000)",
    )
    args = ap.parse_args(argv)
    status = build()
    if status or args.serve is None:
        return status
    serve(args.serve)
    return 0


if __name__ == "__main__":
    sys.exit(main())
