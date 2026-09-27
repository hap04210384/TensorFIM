# -*- coding: utf-8 -*-
"""Fig. 6 threshold sweep: accidents, kosarak, connect across thresholds,
both engines. Set identity verified against data/gold_results/.

Usage:  python scripts/run_threshold_sweep.py [--reps 5]
"""
import argparse, io, os, re, statistics, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENG = os.path.join(ROOT, "code", "engine", "run")
GOLD = os.path.join(ROOT, "data", "gold_results")
WORK = os.path.join(ROOT, "work")

CONFIGS = (
    [("accidents.txt", t) for t in (0.30, 0.40, 0.50, 0.60, 0.70)] +
    [("kosarak.dat", t) for t in (0.01, 0.02, 0.03, 0.05)] +
    [("connect.txt", t) for t in (0.80, 0.85, 0.90, 0.95)]
)

COMP = re.compile(r"computation_time\(s\):\s*([\d.]+)")
TOT = re.compile(r"total_time\(s\):\s*([\d.]+)")
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    bad = 0
    for ds, thr in CONFIGS:
        src = os.path.join(ROOT, "data", "small", ds)
        gold_path = os.path.join(GOLD, "%s-%.6f=Results.txt" % (ds, thr))
        if not os.path.exists(src) or not os.path.exists(gold_path):
            print("[skip] %s @%s" % (ds, thr))
            continue
        gold = load_results(gold_path)
        rp = src + "-%.6f=Results.txt" % thr
        line = "%-14s %.2f" % (ds, thr)
        for eng in ("baseline", "bmma"):
            exe = os.path.join(ENG, "CoParaCG_%s.exe" % eng)
            tots = []
            for r in range(args.reps):
                if os.path.exists(rp):
                    os.remove(rp)
                log = os.path.join(WORK, "sweep_%s_%.3f_%s_%d.log" % (ds, thr, eng, r))
                with open(log, "wb") as lf:
                    subprocess.run([exe, src, str(thr)], stdout=lf,
                                   stderr=subprocess.STDOUT, timeout=600)
                if not os.path.exists(rp) or load_results(rp) != gold:
                    bad += 1
                    print("[DIFF] %s @%s %s rep%d" % (ds, thr, eng, r))
                with io.open(log, encoding="utf-8", errors="replace") as f:
                    t = TOT.findall(f.read())
                if t:
                    tots.append(float(t[-1]))
                time.sleep(2)
            line += "   %s=%ss" % (eng, round(statistics.median(tots), 4) if tots else "NA")
        print(line)
    print("\nALL SETS IDENTICAL" if bad == 0 else "\n%d FAILURES" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
