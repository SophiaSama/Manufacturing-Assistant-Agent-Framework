"""Backward-compatibility re-export for cache CLI."""

from enterprise_agent.cache.cli import main

if __name__ == "__main__":
    import sys
    sys.exit(main())
