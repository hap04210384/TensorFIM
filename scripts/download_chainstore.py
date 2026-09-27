# -*- coding: utf-8 -*-
"""Download the chain-store benchmark from the public SPMF dataset library.

Source: https://www.philippe-fournier-viger.com/spmf/publicdatasets/chainstoreFIM.txt
(plain itemset variant of the chain-store retail dataset; 1,112,949 transactions
over 46,086 items, mean length 7.2 — see the SPMF datasets page:
https://www.philippe-fournier-viger.com/spmf/index.php?link=datasets.php)

Writes data/small/chainstore.txt and verifies size + MD5 against the copy used
in the paper (Table III row "chain-store").

Usage:  python scripts/download_chainstore.py
"""
import hashlib, os, sys, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "https://www.philippe-fournier-viger.com/spmf/publicdatasets/chainstoreFIM.txt"
OUT = os.path.join(ROOT, "data", "small", "chainstore.txt")
SIZE = 46650528
MD5 = "564284f9f913e97d8e672100be9c8174"


def main():
    if os.path.exists(OUT):
        h = hashlib.md5(open(OUT, "rb").read()).hexdigest()
        if h == MD5:
            print("already present and verified:", OUT)
            return
        print("existing file failed verification, re-downloading")
    tmp = OUT + ".part"
    print("downloading", URL)
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    data = open(tmp, "rb").read()
    if len(data) != SIZE or hashlib.md5(data).hexdigest() != MD5:
        os.remove(tmp)
        print("ERROR: downloaded file does not match the paper copy "
              "(size %d != %d or MD5 mismatch)" % (len(data), SIZE))
        sys.exit(1)
    os.replace(tmp, OUT)
    print("OK: %d bytes, md5 %s -> %s" % (len(data), MD5, OUT))


if __name__ == "__main__":
    main()
