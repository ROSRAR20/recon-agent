"""Purge all cached LLM response entries from results/llm_cache/.

Requires explicit --yes confirmation flag.
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_DIR = os.path.join(REPO_ROOT, "results", "llm_cache")


def purge_cache(dry_run: bool = False) -> int:
    """Delete all cache files in results/llm_cache/."""
    if not os.path.isdir(CACHE_DIR):
        print(f"Cache directory does not exist: {CACHE_DIR}")
        return 0

    count = 0
    for root, _, files in os.walk(CACHE_DIR):
        for fname in files:
            if fname.endswith(".json") or fname.endswith(".tmp"):
                filepath = os.path.join(root, fname)
                if not dry_run:
                    try:
                        os.remove(filepath)
                        count += 1
                    except OSError as exc:
                        print(f"Failed to remove {filepath}: {exc}", file=sys.stderr)
                else:
                    count += 1

    action = "Would delete" if dry_run else "Deleted"
    print(f"{action} {count} cache file(s) in {CACHE_DIR}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Purge results/llm_cache entries")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm deletion of all cache files. Required to proceed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be deleted without deleting.",
    )
    args = parser.parse_args()

    if not args.yes and not args.dry_run:
        print(
            "ERROR: Cache purge aborted. The --yes flag is strictly required to delete cache files.\n"
            "Usage: python scripts/purge_cache.py --yes",
            file=sys.stderr,
        )
        sys.exit(1)

    purge_cache(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
