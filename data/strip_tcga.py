# -*- coding: utf-8 -*-
"""Strip TCGA-BRCA kernel items (items present in EVERY transaction).

The raw TCGA_BRCA transaction set has 1,226 transactions over 60,660 items;
28,977 items appear in all 1,226 transactions ("kernel items"). They carry no
discriminative information for mining (every itemset containing them has the
same support without them), so the paper's benchmark removes them, leaving
31,683 items. We ship the stripped file directly as TCGA_BRCA_stripped.txt.gz;
this script documents the transformation for provenance.

Usage:  python data/strip_tcga.py TCGA_BRCA_transactions.txt TCGA_BRCA_stripped.txt
"""
import sys
from collections import Counter


def main():
    src, dst = sys.argv[1], sys.argv[2]
    freq = Counter()
    rows = []
    with open(src, encoding="utf-8", errors="replace") as f:
        for ln in f:
            items = [int(x) for x in ln.split()]
            rows.append(items)
            freq.update(set(items))
    n = len(rows)
    kernel = {x for x, c in freq.items() if c == n}
    with open(dst, "w", encoding="utf-8") as f:
        for items in rows:
            f.write(" ".join(str(x) for x in items if x not in kernel) + "\n")
    print("transactions=%d items=%d kernel=%d stripped->%s" %
          (n, len(freq), len(kernel), dst))


if __name__ == "__main__":
    main()
