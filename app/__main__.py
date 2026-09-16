"""python -m app [--version]"""
import sys

from . import __version__


def main():
    if "--version" in sys.argv or "-V" in sys.argv:
        print("NoSlop", __version__)
        return 0
    print("usage: python -m app [--version]")
    return 0


if __name__ == "__main__":
    sys.exit(main())