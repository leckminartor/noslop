"""Combined final polish: metallic de-ringing + AI-echo tail suppression in one pass.

Rationale: the metallic combs (dering) sit disproportionately in the reverb
tails (derverb) - treating them separately leaves comb energy inside the tail
and tail energy modulated by combs. The combined pass runs them in the order
that avoids re-introducing artifacts:

1. dering pass  (Demucs-residual-guided notches): removes spectrum-periodic
   ringing; combs inside tails become broadband and the following stage can
   address them as tail energy.
2. derverb pass (anchor-calibrated causal decay limiter): shortens HF tails
   that linger past the mix's own plausible decay.
3. (optional second round: both again, for strong material.)

Loudness-neutral overall; every sub-pass reports its own metrics.

Available as CLI `enhance --mode polish` and server mode `polish`.
Verified in tests/bench_polish.py.
"""
from __future__ import annotations

import numpy as np

from .loudness import integrated_loudness
from .dering_ai import dering_ai
from .derverb import derverb, tail_metrics


def polish(audio: np.ndarray, sr: int,
           model=None,
           dering_spike_db: float = 5.0,
           dering_max_db: float = 8.0,
           dering_passes: int = 1,
           derverb_margin_db_s: float = 45.0,
           derverb_passes: int = 2,
           rounds: int = 1,
           progress_cb=None) -> tuple[np.ndarray, int, dict]:
    """Combined metallic + tail restoration. Returns (audio, out_sr, report).

    rounds: how often to alternate dering/derverb (1 = one of each; 2 = both
    twice). Each stage is loudness-compensated per pass; the final pass is
    loudness-matched to the input.
    """
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    lufs_in = integrated_loudness(x, sr)

    cur = x.astype(np.float32)
    cur_sr = sr
    report: dict = {"stages": [], "rounds": 0}

    total_stages = max(1, rounds * 2)
    stage_i = 0
    # safety gate: on already-dry material the tail limiter must not fire
    m_in = tail_metrics(x.astype("float32"), cur_sr)
    dry_skip = bool(m_in["tail_peak_ratio_db"] <= -11.5)
    for r_i in range(max(1, int(rounds))):
        # ---- stage A: metallic notches
        stage_i += 1
        if progress_cb:
            progress_cb(0.1 + 0.8 * (stage_i - 1) / total_stages,
                        f"round {r_i + 1}: dering")
        cur, cur_sr, rep_d = dering_ai(cur, cur_sr, model=model,
                                       spike_db=dering_spike_db,
                                       max_db=dering_max_db,
                                       passes=max(1, int(dering_passes)))
        n_notch = sum(r.get("n_notches", 0) for r in rep_d.get("passes", []))
        report["stages"].append({"stage": "dering", "round": r_i + 1,
                                 "n_notches": n_notch,
                                 "loudness_compensation_db": rep_d.get("loudness_compensation_db")})
        if progress_cb:
            progress_cb(0.1 + 0.8 * stage_i / total_stages,
                        f"round {r_i + 1}: derverb")
        # ---- stage B: tail limiter (skipped when the input is already dry)
        if not dry_skip:
            cur, rep_v = derverb(cur, cur_sr, margin_db_s=derverb_margin_db_s,
                                 passes=2)
        else:
            rep_v = {"out_sr": cur_sr, "loudness_compensation_db": None}
        if r_i == 0 and not dry_skip:
            # re-check after the first dering: if tails are now dry enough, stop
            m_before = tail_metrics(cur.astype("float64"), cur_sr)
            dry_skip = bool(m_before["tail_peak_ratio_db"] <= -11.5)
        report["stages"].append({"stage": "derverb", "round": r_i + 1,
                                 "loudness_compensation_db": rep_v.get("loudness_compensation_db")})
        cur_sr = rep_v.get("out_sr", cur_sr)
        report["rounds"] = r_i + 1

    # final loudness match to the input
    lufs_out = integrated_loudness(cur, cur_sr)
    comp = None
    if np.isfinite(lufs_in) and np.isfinite(lufs_out):
        d = lufs_in - lufs_out
        if 0.05 < abs(d) < 12:
            cur = cur * (10 ** (d / 20.0))
            comp = float(d)
    report["loudness_compensation_db"] = comp
    report["out_sr"] = cur_sr
    return cur.astype(np.float32), cur_sr, report


def polish_metrics(sig: np.ndarray, sr: int) -> dict:
    """Combined artifact report for the UI/CLI: tail + metallic + health."""
    from .analysis import analyze_health
    m = tail_metrics(sig, sr)
    h = analyze_health(sig, sr)
    return {**m, **{k: h[k] for k in ("hf_incoherence", "lufs", "true_peak_dbtp", "artifact_flags")}}