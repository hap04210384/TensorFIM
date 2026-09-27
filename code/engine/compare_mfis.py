"""Per-set exact comparison of two AnyFIM result files (MFI lists)."""
import re, sys, hashlib

def load(path):
    mfis = []
    for line in open(path, encoding='utf-8', errors='replace'):
        m = re.search(r'\{([^}]*)\}\s+support:\s*\S+\s+frequency:\s*(\d+)', line)
        if m:
            items = tuple(sorted(int(x) for x in m.group(1).split()))
            mfis.append((items, int(m.group(2))))
    return sorted(mfis)

a = load(sys.argv[1])
b = load(sys.argv[2])
ha = hashlib.md5(repr(a).encode()).hexdigest()
hb = hashlib.md5(repr(b).encode()).hexdigest()
print(f"{sys.argv[1]}: {len(a)} MFIs, md5={ha}")
print(f"{sys.argv[2]}: {len(b)} MFIs, md5={hb}")
if a == b:
    print("MATCH: identical MFI sets with identical frequencies")
    sys.exit(0)
sa, sb = set(a), set(b)
print(f"MISMATCH: only-in-first={len(sa-sb)}, only-in-second={len(sb-sa)}")
for x in list(sa - sb)[:5]: print("  only-1st:", x)
for x in list(sb - sa)[:5]: print("  only-2nd:", x)
sys.exit(1)
