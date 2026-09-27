"""One ablation run of the BMMA wave engine with env overrides.

usage: python ablate.py <dsfile> <thr6> <tag> <wavecap|0> <splitk_off|0|1> <idx> <ref_results> <outdir>
example: python ablate.py pumsb_x256.txt 0.800000 pumsbx256 320 0 1 run/results_bmma_pumsbx256.txt ../experiments/wavecap_ablation

Runs run/CoParaCG_bmma.exe (cwd=run), saves log + results copy to <outdir>,
parses phase timings, verifies MFI set against <ref_results> (MD5 of sorted list),
appends one row to <outdir>/wavecap_sweep.csv.
"""
import hashlib, os, re, subprocess, sys, shutil, time

dsfile, thr6, tag, wavecap, splitk_off, idx, ref, outdir = sys.argv[1:9]
wpb = int(sys.argv[9]) if len(sys.argv) > 9 else 0
wavecap = int(wavecap); splitk_off = int(splitk_off); idx = int(idx)
root = os.path.dirname(os.path.abspath(__file__))
rundir = os.path.join(root, "run")
os.makedirs(outdir, exist_ok=True)

name = f"wc_{tag}_cap{wavecap or 'auto'}_sk{'off' if splitk_off else 'auto'}" + (f"_wpb{wpb}" if wpb else "") + f"_{idx}"
env = dict(os.environ)
if wavecap > 0: env["BMMA_WAVE_CAP"] = str(wavecap)
else: env.pop("BMMA_WAVE_CAP", None)
if splitk_off: env["BMMA_SPLITK_OFF"] = "1"
else: env.pop("BMMA_SPLITK_OFF", None)
if wpb: env["BMMA_MASK_WPB"] = str(wpb)
else: env.pop("BMMA_MASK_WPB", None)

log_path = os.path.join(outdir, name + ".log")
t0 = time.time()
with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
    subprocess.run([os.path.join(rundir, "CoParaCG_bmma.exe")], cwd=rundir,
                   stdout=lf, stderr=subprocess.STDOUT, env=env)
wall = time.time() - t0

# results file written by the exe next to the dataset
res_src = os.path.join(root, "TransactionSets", f"{dsfile}-{thr6}=Results.txt")
res_dst = os.path.join(outdir, name + "_results.txt")
shutil.copyfile(res_src, res_dst)

def load_mfis(path):
    mfis = []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.search(r"\{([^}]*)\}\s+support:\s*\S+\s+frequency:\s*(\d+)", line)
        if m:
            mfis.append((tuple(sorted(int(x) for x in m.group(1).split())), int(m.group(2))))
    return sorted(mfis)

def md5_of(mfis):
    return hashlib.md5(repr(mfis).encode()).hexdigest()

got = load_mfis(res_dst); want = load_mfis(ref)
md5 = md5_of(got); match = (got == want)

log = open(log_path, encoding="utf-8", errors="replace").read()
def f(pat, default=""):
    m = re.search(pat, log)
    return m.groups() if m else default
prep = f(r"preprocessing_time\(s\):\s*([\d.eE+-]+)", ("",))[0]
comp = f(r"computation_time\(s\):\s*([\d.eE+-]+)", ("",))[0]
mfi  = f(r"MFIsNumber:\s*(\d+)", ("",))[0]
ph   = f(r"BMMA phase ms:\s*mask=([\d.]+)\s*bmma=([\d.]+)\s*memcpy=([\d.]+)\s*scatter=([\d.]+)",
         ("", "", "", ""))
bt   = f(r"BMMA batches:\s*(\d+)\s*\(smem:\s*(\d+)\),\s*rows counted by BMMA:\s*(\d+),\s*fallback waves:\s*(\d+)",
         ("", "", "", ""))

csv = os.path.join(outdir, "wavecap_sweep.csv")
if not os.path.exists(csv):
    open(csv, "w").write("config,wavecap,splitk,idx,prep,comp,mfi,mask_ms,bmma_ms,copy_ms,scatter_ms,batches,rows,md5,match,wall_s\n")
open(csv, "a").write(",".join(map(str, [
    tag, wavecap or "auto", "off" if splitk_off else "auto", idx, prep, comp, mfi,
    *ph, bt[0], bt[2], md5, "MATCH" if match else "MISMATCH", f"{wall:.1f}"])) + "\n")
print(f"{name}: comp={comp}s mfi={mfi} md5={md5[:8]} {'MATCH' if match else 'MISMATCH'} wall={wall:.1f}s")
sys.exit(0 if match else 2)
