# -*- coding: utf-8 -*-
"""Build the replicated benchmarks used in the paper.

pumsb_x256   = pumsb concatenated 256x   (12,555,776 transactions, ~4.3 GB)
accidents_x4 = accidents concatenated 4x
accidents_x8 = accidents concatenated 8x

Replication preserves item ids, so every itemset's support scales by exactly
the replication factor and mining results at the same relative threshold are
identical in structure (verified in the paper, Table III).

Usage:
  python data/make_replica.py pumsb.dat 256 pumsb_x256.txt
  python data/make_replica.py data/small/accidents.txt 4 data/accidents_x4.txt
  python data/make_replica.py data/small/accidents.txt 8 data/accidents_x8.txt

pumsb.dat is available from the FIMI repository (see DOWNLOAD.md).
"""
import sys


def main():
    src, factor, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    with open(src, "rb") as f:
        blob = f.read()
    if not blob.endswith(b"\n"):
        blob += b"\n"
    with open(dst, "wb") as f:
        for _ in range(factor):
            f.write(blob)
    print("wrote %s (%d x %s)" % (dst, factor, src))


if __name__ == "__main__":
    main()
