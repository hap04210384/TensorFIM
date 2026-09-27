# -*- coding: utf-8 -*-
"""Table VI ablation: wave-size sweep + split-k on/off (pumsb_x256, accidents_x4).

Usage:  python scripts/run_ablation.py [--reps 5]
Knobs are read by the BMMA engine from environment variables:
  BMMA_WAVE_CAP=N   force wave size to N rows (unset = shipping 96 MB rule)
  BMMA_SPLITK_OFF=1 disable split-k
  BMMA_MASK_WPB=N   mask-build words-per-block quota (default 4096)
Set identity against data/gold_results/ is verified on every run.
"""
import argparse, io, os, re, statistics, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENG = os.path.join(ROOT, "code", "engine", "run", "CoParaCG_bmma.exe")
GOLD = os.path.join(ROOT, "data", "gold_results")
WORK = os.path.join(ROOT, "work")

CELLS = []
for ds, thr in (("pumsb_x256.txt", 0.8), ("accidents_x4.txt", 0.4)):
    for w in (64, 128, 256, 320, 512):
        CELLS.append((ds, thr, "wave%d" % w, {"BMMA_WAVE_CAP": str(w)}))
    CELLS.append((ds, thr, "auto", {}))
    CELLS.append((ds, thr, "auto_splitk_off", {"BMMA_SPLITK_OFF": "1"}))
for wpb in (1024, 16384):
    CELLS.append(("pumsb_x256.txt", 0.8, "maskwpb%d" % wpb, {"BMMA_MASK_WPB": str(wpb)}))

COMP = re.compile(r"computation_time\(s\):\s*([\d.]+)")
LINE = re.compile(r"^\d+th:\s+\d+\s+\{\s*([^}]*)\}\s+support:\s+\S+\s+frequency:\s+(\d+)")


def load_results(path):
    sets = {}
    with io.open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            m = LINE.match(ln.strip())
            if m:
                sets[frozenset(int(x) for x in m.group(1).split())] = int(m.group(2))
    return sets


def find_dataset(name):
    for base in (os.path.join(ROOT, "data"), os.path.join(ROOT, "data", "small")):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    bad = 0
    for ds, thr, label, envx in CELLS:
        src = find_dataset(ds)
        gold_path = os.path.join(GOLD, "%s-%.6f=Results.txt" % (ds, thr))
        if not src or not os.path.exists(gold_path):
            print("[skip] %s %s" % (ds, label))
            continue
        gold = load_results(gold_path)
        rp = src + "-%.6f=Results.txt" % thr
        comps = []
        env = dict(os.environ); env.update(envx)
        for r in range(args.reps):
            if os.path.exists(rp):
                os.remove(rp)
            log = os.path.join(WORK, "ablate_%s_%s_%d.log" % (ds.split(".")[0], label, r))
            with open(log, "wb") as lf:
                subprocess.run([ENG, src, str(thr)], stdout=lf,
                               stderr=subprocess.STDOUT, env=env, timeout=1800)
            if not os.path.exists(rp) or load_results(rp) != gold:
                bad += 1
                print("[DIFF] %s %s rep%d" % (ds, label, r))
            with io.open(log, encoding="utf-8", errors="replace") as f:
                c = COMP.findall(f.read())
            if c:
                comps.append(float(c[-1]))
            time.sleep(2)
        med = round(statistics.median(comps), 4) if comps else None
        print("%-20s %-16s comp median: %s s" % (ds, label, med))
    print("\nALL SETS IDENTICAL" if bad == 0 else "\n%d FAILURES" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
