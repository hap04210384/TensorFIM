# -*- coding: utf-8 -*-
"""Run the 14 paper configurations on both engines (Table IV / V, Fig. 3/5).

Usage:  python scripts/run_all_configs.py [--reps 5]
Requires: built (or prebuilt) engines in code/engine/run/, datasets prepared
(see data/DOWNLOAD.md). Results land in work/ and are set-verified against
data/gold_results/. Timings are informational; set identity is the gate.
"""
import argparse, io, os, re, statistics, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENG = os.path.join(ROOT, "code", "engine", "run")
GOLD = os.path.join(ROOT, "data", "gold_results")
WORK = os.path.join(ROOT, "work")

CONFIGS = [
    ("mushrooms.txt", 0.3), ("chess.txt", 0.8), ("T10I4D100K.txt", 0.005),
    ("connect.txt", 0.9), ("retail.txt", 0.01),
    ("TCGA_BRCA_stripped.txt", 0.7), ("TCGA_BRCA_stripped.txt", 0.65),
    ("accidents.txt", 0.5), ("accidents_x4.txt", 0.4), ("accidents_x8.txt", 0.4),
    ("kosarak.dat", 0.02), ("pumsb_x256.txt", 0.8), ("webdocs.dat", 0.08),
    ("chainstore.txt", 0.002),
]

COMP = re.compile(r"computation_time\(s\):\s*([\d.]+)")
TOT = re.compile(r"total_time\(s\):\s*([\d.]+)")
LINE = re.compile(r"^\d+th:\s+\d+\s+\{\s*([^}]*)\}\s+support:\s+\S+\s+frequency:\s+(\d+)")


def find_dataset(name):
    for base in (os.path.join(ROOT, "data"), os.path.join(ROOT, "data", "small")):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    return None


def load_results(path):
    sets = {}
    with io.open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            m = LINE.match(ln.strip())
            if m:
                sets[frozenset(int(x) for x in m.group(1).split())] = int(m.group(2))
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    rows, bad = [], 0
    for ds, thr in CONFIGS:
        src = find_dataset(ds)
        gold_path = os.path.join(GOLD, "%s-%.6f=Results.txt" % (ds, thr))
        if not src or not os.path.exists(gold_path):
            print("[skip] %s @%s (dataset or gold missing)" % (ds, thr))
            continue
        gold = load_results(gold_path)
        rp = src + "-%.6f=Results.txt" % thr
        rec = {"dataset": ds, "thr": thr}
        for eng in ("baseline", "bmma"):
            exe = os.path.join(ENG, "CoParaCG_%s.exe" % eng)
            comps = []
            ok = True
            for r in range(args.reps):
                if os.path.exists(rp):
                    os.remove(rp)
                log = os.path.join(WORK, "%s_%.3f_%s_%d.log" % (ds, thr, eng, r))
                with open(log, "wb") as lf:
                    subprocess.run([exe, src, str(thr)], stdout=lf,
                                   stderr=subprocess.STDOUT, timeout=1800)
                if not os.path.exists(rp):
                    ok = False
                    continue
                if load_results(rp) != gold:
                    ok = False
                    print("[DIFF] %s @%s %s rep%d" % (ds, thr, eng, r))
                with io.open(log, encoding="utf-8", errors="replace") as f:
                    c = COMP.findall(f.read())
                if c:
                    comps.append(float(c[-1]))
                time.sleep(2)
            rec[eng] = round(statistics.median(comps), 4) if comps else None
            rec[eng + "_ok"] = ok
            bad += 0 if ok else 1
        if rec.get("baseline") and rec.get("bmma"):
            rec["speedup"] = round(rec["baseline"] / rec["bmma"], 2)
        rows.append(rec)
        print(rec)
    print("\nALL SETS IDENTICAL" if bad == 0 else "\n%d ENGINE-CONFIG FAILURES" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
