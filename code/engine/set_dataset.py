"""Switch dataset/threshold line in the two kernel sources (BOM-safe)."""
import io, re, sys

dataset, thr = sys.argv[1], sys.argv[2]   # e.g. accidents_x4.txt 0.4
pat = re.compile(r'^CString transSetFile = .*double supportThreshold = [\d.]+;', re.M)
new = 'CString transSetFile = _T("..\\\\TransactionSets\\\\%s"); double supportThreshold = %s;' % (dataset, thr)
for f in ['src/kernel_bitmap.cu', 'src/kernel_bmma.cu']:
    s = io.open(f, encoding='utf-8-sig').read()
    s2, n = pat.subn(lambda m: new, s, count=1)   # lambda: no backslash processing in replacement
    assert n == 1, f
    io.open(f, 'w', encoding='utf-8-sig', newline='').write(s2)
    print(f, '->', new)
