"""Command line entry point for implemented dynamic-alpha stages."""

from __future__ import annotations

import argparse
import sys


def cli() -> None:
    parser = argparse.ArgumentParser(prog="dynamic-alpha")
    parser.add_argument("command", choices=("preflight",),
                        help="implemented stage to run")
    args, remainder = parser.parse_known_args()
    if args.command == "preflight":
        from .preflight import cli as preflight_cli
        sys.argv = [f"{parser.prog} preflight", *remainder]
        preflight_cli()
