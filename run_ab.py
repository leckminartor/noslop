import sys
sys.path.insert(0, "D:/PythonCode/noslop2")
import numpy as np
import soundfile as sf
import time
from app.io import read_audio
from app.dering_ai import dering_ai, metallic_metrics
from app.analysis import analyze_health

files = [("demo_mp364.wav", "mp3 64k"), ("demo_opus32.wav", "opus 32k")]
for f, label in files:
    x, sr, _ = read_audio(f)
    base = metallic_metrics(x, sr)
    print("== %s ==" % label)
    print("  decoded:  comb %.3f @ %.0fHz | stab %.2f | kurt %.0f" % (
        base["comb"], base["comb_f0"], base["stability"], base["kurtosis"]))
    for name, kwargs in (
        ("standard", dict(spike_db=6.0, max_db=7.0, passes=1)),
        ("aggressive", dict(spike_db=5.0, max_db=10.0, passes=2)),
    ):
        t0 = time.time()
        y, osr, rep = dering_ai(x, sr, **kwargs)
        el = time.time() - t0
        m = metallic_metrics(y.astype("float64"), osr)
        a = analyze_health(y.astype("float64"), osr)
        b = analyze_health(x, sr)
        n_notch = sum(r.get("n_notches", 0) for r in rep["passes"])
        n_guard = sum(r.get("n_guarded", 0) for r in rep["passes"])
        print("  %s (%.1fs, %d passes): comb %.3f | stab %.2f | kurt %.0f | "
              "notches %d (guarded %d) | LUFS %.2f (was %.2f) | TP %.1f" % (
                  name, el, n_notch, m["comb"], m["stability"], m["kurtosis"],
                  n_notch, n_guard, a["lufs"], b["lufs"], a["true_peak_dbtp"]))
        out = "ab_%s_%s.wav" % (label.split()[0], name)
        sf.write(out, y, osr, subtype="PCM_24")
        print("     wrote", out)
print("done")