"""Entry point: python -m src.cli {inventory,cache}"""
import sys

from . import build_inventory, io_road


def main(argv=None):
    cmd = (argv or sys.argv[1:] or ["help"])[0]
    if cmd == "inventory":
        build_inventory.main()
    elif cmd == "cache":
        io_road.cache_all(io_road.load_captures_cfg())
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
