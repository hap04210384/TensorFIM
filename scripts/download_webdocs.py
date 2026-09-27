# -*- coding: utf-8 -*-
"""Download the webdocs benchmark.

webdocs (Lucchese, Orlando, Perego, Silvestri — FIMI'04 workshop note) was
distributed through the FIMI repository, whose original mirror is offline.
We therefore mirror the gzipped original byte-for-byte as a release asset of
this repository (release "data-v1"). This script fetches that mirror, verifies
size + MD5, and decompresses to data/webdocs.dat.

  webdocs.dat.gz  512,655,922 bytes  md5 48496bcae8bb47eca1037b6ca71d7400
  webdocs.dat     1,692,082 transactions over 5,267,656 item ids

Usage:  python scripts/download_webdocs.py
"""
import gzip, hashlib, os, shutil, sys, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "https://github.com/hap04210384/TensorFIM/releases/download/data-v1/webdocs.dat.gz"
GZ = os.path.join(ROOT, "data", "webdocs.dat.gz")
DAT = os.path.join(ROOT, "data", "webdocs.dat")
SIZE = 512655922
MD5 = "48496bcae8bb47eca1037b6ca71d7400"


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not os.path.exists(GZ) or md5_of(GZ) != MD5:
        tmp = GZ + ".part"
        print("downloading", URL)
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f, 1 << 24)
        if os.path.getsize(tmp) != SIZE or md5_of(tmp) != MD5:
            os.remove(tmp)
            print("ERROR: downloaded archive does not match the paper copy")
            sys.exit(1)
        os.replace(tmp, GZ)
        print("archive verified:", GZ)
    else:
        print("archive already present and verified:", GZ)
    if not os.path.exists(DAT):
        print("decompressing ->", DAT)
        with gzip.open(GZ, "rb") as fi, open(DAT + ".part", "wb") as fo:
            shutil.copyfileobj(fi, fo, 1 << 24)
        os.replace(DAT + ".part", DAT)
    print("OK: webdocs.dat ready")


if __name__ == "__main__":
    main()
