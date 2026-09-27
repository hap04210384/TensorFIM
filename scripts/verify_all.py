# -*- coding: utf-8 -*-
"""Verify every produced result file against the gold references.

Scans for <dataset>-<thr>=Results.txt files next to the datasets and compares
MFI sets + frequencies against data/gold_results/. Exit code 0 = all match.

Usage:  python scripts/verify_all.py
"""
import io, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.path.join(ROOT, "data", "gold_results")
LINE = re.compile(r"^\d+th:\s+\d+\s+\{\s*([^}]*)\}\s+support:\s+\S+\s+frequency:\s+(\d+)")


def load_results(path):
    sets = {}
    with io.open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            m = LINE.match(ln.strip())
            if m:
                sets[frozenset(int(x) for x in m.group(1).split())] = int(m.group(2))
    return sets


def main():
    n_ok = n_bad = 0
    for base in (os.path.join(ROOT, "data"), os.path.join(ROOT, "data", "small")):
        if not os.path.isdir(base):
            continue
        for f in sorted(os.listdir(base)):
            if not f.endswith("=Results.txt"):
                continue
            gold_path = os.path.join(GOLD, f)
            if not os.path.exists(gold_path):
                print("[no gold] %s" % f)
                continue
            mine = load_results(os.path.join(base, f))
            gold = load_results(gold_path)
            if mine == gold:
                n_ok += 1
                print("[identical] %s (%d MFIs)" % (f, len(gold)))
            else:
                n_bad += 1
                print("[MISMATCH] %s (mine=%d gold=%d)" % (f, len(mine), len(gold)))
    print("\n%d identical, %d mismatched" % (n_ok, n_bad))
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
