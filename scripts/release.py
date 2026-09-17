#!/usr/bin/env python3
"""One-command release: bump the version everywhere + create changelog entry + tag.

Usage:
    python scripts/release.py 0.2.0           # bump to 0.2.0, update all files
    python scripts/release.py 0.2.0 --push    # ...and push with tag (triggers the release workflow)
"""
import os
import re
import subprocess
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INIT = os.path.join(ROOT, "app", "__init__.py")


def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def write(p, c):
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(c)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    new = sys.argv[1]
    if not re.fullmatch(r"\d+\.\d+\.\d+", new):
        print("version must be MAJOR.MINOR.PATCH, e.g. 0.2.0")
        return 1
    push = "--push" in sys.argv

    init = read(INIT)
    m = re.search(r'__version__ = "(\d+\.\d+\.\d+)"', init)
    if not m:
        print("cannot find __version__ in app/__init__.py")
        return 1
    old = m.group(1)
    print(f"version: {old} -> {new}")

    # 1) single source of truth
    write(INIT, init.replace(f'__version__ = "{old}"', f'__version__ = "{new}"'))

    # 2) README badges + version line (both languages)
    for name in ("README.md", "README.de.md"):
        p = os.path.join(ROOT, name)
        c = read(p)
        c = c.replace(f"version-{old}-green.svg", f"version-{new}-green.svg")
        c = c.replace(f"**Version {old}**", f"**Version {new}**")
        c = re.sub(r"(NoSlop v)" + re.escape(old) + r"( — RVQ Artifact Remover)", rf"\g<1>{new}\g<2>", c)
        write(p, c)
        print(f"  updated {name}")

    # 3) web UI title
    p = os.path.join(ROOT, "app", "web", "index.html")
    c = read(p)
    c = re.sub(r"NoSlop v" + re.escape(old) + r"( — RVQ Artifact Remover)", r"NoSlop v" + new + r"\1", c)
    write(p, c)
    print("  updated app/web/index.html")

    # 4) CITATION.cff (version line is regex-based so a skipped bump can't break it)
    p = os.path.join(ROOT, "CITATION.cff")
    if os.path.exists(p):
        c = read(p)
        c = re.sub(r"version: \d+\.\d+\.\d+", f"version: {new}", c, count=1)
        c = re.sub(r"date-released: .*", f"date-released: {date.today().isoformat()}", c)
        write(p, c)
        print("  updated CITATION.cff")

    # 5) CHANGELOG: insert a new section under the header
    p = os.path.join(ROOT, "CHANGELOG.md")
    c = read(p)
    entry = f"## [{new}] - {date.today().isoformat()}\n\n### Added\n- (describe changes here)\n\n"

### Added\n- (describe changes here)\n\n"
    c = c.replace("## [" + old + "]", entry + "## [" + old + "]", 1)
    write(p, c)
    print("  CHANGELOG.md: new section (fill it in!)")

    print("\nDone. Now: edit the CHANGELOG entry, then commit:")
    print(f'  git commit -am "v{new}: <summary>" && git tag v{new}')
    if push:
        print("  git push origin main v" + new)

    if push:
        subprocess.run(["git", "add", "-A"], check=True)
        subprocess.run(["git", "commit", "-m", f"v{new}: version bump (see CHANGELOG)"], check=True)
        subprocess.run(["git", "tag", f"v{new}"], check=True)
        subprocess.run(["git", "push", "origin", "main", f"v{new}"], check=True)
        print("pushed. The release workflow will publish the GitHub release.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
