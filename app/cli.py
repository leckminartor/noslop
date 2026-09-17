"""NoSlop CLI: analyze + restore + AI-enhance audio files from the command line.

Usage:
  python -m app.cli analyze <file>
  python -m app.cli restore <file> [--preset gentle|standard|strong] [-o out.wav]
  python -m app.cli enhance <file> [--mode fast|quality|deep] [-o out.wav]

Modes:
  fast    = DSP-only at native rate (seconds per track, no GPU needed)
  quality = + demucs stem separation (GPU) with conservative residual blend
  deep    = + full re-render (residual_alpha=0), strongest de-slop
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(prog="noslop", description="RVQ/codec artifact remover")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a_an = sub.add_parser("analyze", help="analyze codec health of a file")
    a_an.add_argument("file")

    a_re = sub.add_parser("restore", help="restore a file (removes artifacts)")
    a_re.add_argument("file")
    a_re.add_argument("--preset", default="standard", choices=["gentle", "standard", "strong"])
    a_re.add_argument("-o", "--out", default=None)
    a_re.add_argument("--report", default=None, help="write report JSON here")

    a_en = sub.add_parser("enhance", help="AI-enhance (demucs stems + SBR repair, GPU)")
    a_en.add_argument("file")
    a_en.add_argument("--mode", default="quality", choices=["fast", "quality", "deep", "dering", "derverb"])
    a_en.add_argument("-o", "--out", default=None)
    a_en.add_argument("--report", default=None, help="write report JSON here")

    args = ap.parse_args(argv)

    if args.cmd == "enhance":
        from .io import read_audio, write_audio
        from .analysis import analyze_health
        from .enhance import enhance
        x, sr, fmt = read_audio(args.file)
        b = analyze_health(x, sr)
        print(f"file: {args.file} ({b['lufs']} LUFS, TP {b['true_peak_dbtp']} dBTP, "
              f"incoh {b['hf_incoherence']})")
        for f in b["artifact_flags"]:
            print(f"  flag: {f}")
        if args.mode == "fast":
            y, out_sr, rep = enhance(x, sr, use_ai=False)
        elif args.mode == "deep":
            print("enhancing (deep: demucs + full re-render) ...")
            y, out_sr, rep = enhance(x, sr, use_ai=True, sbr=True,
                                     residual_alpha=0.0, sbr_thresh=0.3)
        elif args.mode == "dering":
            from .dering_ai import dering_ai
            print("deringing (metallic/comb removal, demucs-residual-guided) ...")
            y, out_sr, rep = dering_ai(x, sr, spike_db=5.0, max_db=10.0, passes=2)
        elif args.mode == "derverb":
            from .derverb import derverb
            print("derverb (AI-echo tail shortening, anchor-calibrated) ...")
            y, rep = derverb(x, sr, margin_db_s=45.0, passes=2)
            out_sr = rep["out_sr"]
        else:
            print("enhancing (quality: demucs stems + conservative repair) ...")
            y, out_sr, rep = enhance(x, sr, use_ai=True, sbr=True,
                                     residual_alpha=0.25, sbr_thresh=0.35)
        a = analyze_health(y.astype(np.float64), out_sr)
        out = args.out or os.path.splitext(args.file)[0] + "_enhanced.wav"
        write_audio(out, y, out_sr)
        print(f"written: {out}")
        print(f"  after: LUFS {a['lufs']} | TP {a['true_peak_dbtp']} dBTP "
              f"| incoh {a['hf_incoherence']}")
        if args.report:
            import json
            with open(args.report, "w") as f:
                json.dump({"before": b, "after": a, "report": rep}, f,
                          indent=2, default=str)
            print(f"report: {args.report}")
        return 0

    from .io import read_audio, write_audio
    from .analysis import analyze_health
    from .restoration import params_from_preset, restore
    from .loudness import match_loudness

    x, sr, fmt = read_audio(args.file)
    analysis = analyze_health(x, sr)
    print(f"file: {args.file} ({fmt}, {len(x)/sr:.1f}s @ {sr} Hz, {x.shape[1]}ch)")
    print(f"  LUFS {analysis['lufs']:.2f} | TP {analysis['true_peak_dbtp']:.2f} dBTP "
          f"| HF incoherence {analysis['hf_incoherence']:.4f}")
    for f in analysis["artifact_flags"]:
        print(f"  flag: {f}")

    if args.cmd == "analyze":
        return 0

    params = params_from_preset(args.preset)
    print(f"restoring (preset {args.preset}) ...")
    y, rep = restore(x.astype(np.float32), sr, params)
    # loudness was already compensated inside restore(); clamp to -1 dBTP ceiling
    from .loudness import true_peak_dbtp
    tp = true_peak_dbtp(y, sr)
    if tp > -1.0:
        y = y * (10 ** ((-1.0 - tp) / 20.0))
    out = args.out or os.path.splitext(args.file)[0] + "_restored.wav"
    write_audio(out, y, sr)
    # final health check on the output
    y64 = y.astype(np.float64) if y.ndim > 1 else y[:, None]
    final = analyze_health(y64, sr)
    print(f"written: {out}")
    print(f"  after: LUFS {final['lufs']:.2f} | TP {final['true_peak_dbtp']:.2f} dBTP "
          f"| HF incoherence {final['hf_incoherence']:.4f}")
    if args.report:
        import json
        with open(args.report, "w") as f:
            json.dump({"analysis": analysis, "report": rep}, f, indent=2)
        print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())