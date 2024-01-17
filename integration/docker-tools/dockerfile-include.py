#!/usr/bin/env python3

# This implements a custom "INLCUDE" directive in a Dockerfile
#
# Modify your Dockerfile like this:
#     ...
#     #+ INCLUDEX foo.txt
#     ...
#
# Tool usage:
#    $ dockerfile-include /old/Dockerfile /new/Dockerfile /search/path
# where the file foo.txt can be found in the /search/path dir

from pathlib import Path
import re
import sys


def main() -> int:
    if len(sys.argv) != 4:
        print("Usage: dockerfile-include.py Dockerfile Dockerfile.tmp /search/path")
    old_dockerfile = Path(sys.argv[1])
    if not old_dockerfile.exists():
        print(f"Source Dockerfile not found: {old_dockerfile}")
        return 1

    new_dockerfile = Path(sys.argv[2])

    search_path = Path(sys.argv[3])
    if not search_path.exists():
        print(f"Search path not found: {search_path}")
        return 1
    if not search_path.is_dir():
        print(f"Search path is not a directory: {search_path}")
        return 1

    docker_text = old_dockerfile.read_text()

    lines = []

    pattern = re.compile(r"^#\s+INCLUDEX\s+([A-Za-z0-9_\-.]+)\s*$")
    for line in docker_text.split("\n"):
        m = pattern.findall(line)
        if m:
            path = search_path / m[0]
            if not path.exists():
                print(f"include file not found: {path}")
                return 1
            s = path.read_text()

            lines.append(f"# --- start of included file: {path} ---")
            lines.append(s)
            lines.append(f"# --- end of included file ---")
            print(f"replaced: {path}")
        else:
            lines.append(line)

    new_dockerfile.write_text("\n".join(lines))
    print("done")


status = main()
sys.exit(status)
