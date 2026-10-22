"""Cache maintenance CLI for Enterprise Agent."""

from __future__ import annotations

import argparse
import sys

from enterprise_agent.cache.context import EnterpriseCache

_LAYERS = ("emb", "retr", "sql")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enterprise Agent cache maintenance")
    parser.add_argument("--db-path", default="data/cache/cache.db", help="Path to cache sqlite database")
    sub = parser.add_subparsers(dest="command", required=True)

    p_stats = sub.add_parser("stats")
    p_stats.add_argument("--env", choices=["prod", "eval"], default="prod")

    p_clear = sub.add_parser("clear")
    p_clear.add_argument("--env", choices=["prod", "eval"], default="prod")
    p_clear.add_argument("--layer", choices=list(_LAYERS) + ["all"], default="all")

    args = parser.parse_args(argv)

    cache = EnterpriseCache(args.db_path, env=args.env)
    try:
        if args.command == "stats":
            store_stats = cache.stats()
            metrics = cache.snapshot()
            print(f"cache env={args.env} db={args.db_path}")
            print(f"  entries={store_stats['entries']} "
                  f"size_bytes={store_stats['size_bytes']}")
            for layer, m in metrics.items():
                print(
                    f"  {layer}: hits={m['hits']} misses={m['misses']} "
                    f"hit_rate={m['hit_rate']} total_ms={m['total_ms']}"
                )
        elif args.command == "clear":
            if args.layer == "all":
                cache.clear()
                print(f"cleared all entries (env={args.env})")
            else:
                import sqlite3

                deleted = 0
                con = sqlite3.connect(cache.store.db_path)
                try:
                    keys = [r[0] for r in con.execute("SELECT key FROM cache")]
                    for k in keys:
                        parts = k.split(":")
                        if len(parts) >= 3 and parts[2] == args.layer:
                            con.execute("DELETE FROM cache WHERE key = ?", (k,))
                            deleted += 1
                    con.commit()
                finally:
                    con.close()
                print(f"deleted {deleted} entries for layer={args.layer} (env={args.env})")
        return 0
    finally:
        cache.close()


if __name__ == "__main__":
    sys.exit(main())
