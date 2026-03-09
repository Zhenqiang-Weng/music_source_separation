#!/usr/bin/env python3
# compare_ckpt.py
# Compare two PyTorch/Lightning checkpoints: print param shapes first, then diff stats.
# Supports writing ALL prints to a log file (with optional tee to console).

import argparse
import sys
from contextlib import contextmanager
from collections import OrderedDict
import torch

COMMON_PREFIXES = [
    "state_dict.",
    "model.",
    "net.",
    "module.",
    "generator.",
    "student.",
    "teacher.",
    "ema.",
]


@contextmanager
def redirect_stdout(filepath: str, tee: bool = False, encoding: str = "utf-8"):
    """
    Redirect all prints to a file.
    If tee=True, also mirror to original stdout.
    """
    orig = sys.stdout
    f = open(filepath, "w", encoding=encoding)

    if tee:
        class Tee:
            def __init__(self, a, b):
                self.a, self.b = a, b

            def write(self, s):
                self.a.write(s)
                self.b.write(s)

            def flush(self):
                self.a.flush()
                self.b.flush()

        sys.stdout = Tee(orig, f)
    else:
        sys.stdout = f

    try:
        yield
    finally:
        sys.stdout = orig
        f.close()


def load_ckpt(path: str, map_location="cpu"):
    return torch.load(path, map_location=map_location)


def extract_state_dict(obj):
    """
    Try to extract a flat state_dict from many possible checkpoint formats.
    Returns (state_dict, hint_str).
    """
    if isinstance(obj, dict):
        # Common containers
        for k in ["state_dict", "model_state_dict", "model", "net", "weights", "params", "ema_state_dict"]:
            if k in obj and isinstance(obj[k], (dict, OrderedDict)):
                sd = obj[k]
                if len(sd) > 0 and all(isinstance(v, torch.Tensor) for v in sd.values()):
                    return OrderedDict(sd), f"ckpt['{k}']"

        # Directly a state_dict
        if len(obj) > 0 and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return OrderedDict(obj), "ckpt (dict is state_dict)"

        # Find the largest tensor-dict entry (fallback)
        best = None
        best_k = None
        best_n = -1
        for k, v in obj.items():
            if isinstance(v, (dict, OrderedDict)) and len(v) > 0 and all(isinstance(vv, torch.Tensor) for vv in v.values()):
                if len(v) > best_n:
                    best = v
                    best_k = k
                    best_n = len(v)
        if best is not None:
            return OrderedDict(best), f"ckpt['{best_k}'] (largest tensor dict)"

    raise ValueError("Unable to extract a state_dict from this checkpoint structure.")


def normalize_keys(sd: OrderedDict, strip_prefixes=True):
    """
    Optionally strip common prefixes repeatedly, to match keys across wrappers.
    Note: this may cause key collisions; later keys overwrite earlier ones.
    """
    if not strip_prefixes:
        return sd

    new_sd = OrderedDict()
    for k, v in sd.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for p in COMMON_PREFIXES:
                if nk.startswith(p):
                    nk = nk[len(p):]
                    changed = True
        new_sd[nk] = v
    return new_sd


def print_shapes(sd: OrderedDict, title: str, max_print: int = 50):
    print(f"\n==== {title} ====")
    print(f"Total tensors: {len(sd)}")
    shown = 0
    for k, v in sd.items():
        print(f"{k:90s}  shape={tuple(v.shape)}  dtype={v.dtype}")
        shown += 1
        if max_print is not None and shown >= max_print:
            if len(sd) > shown:
                print(f"... (only showing first {shown} / {len(sd)})")
            break


def match_keys(sdA: OrderedDict, sdB: OrderedDict):
    keysA = set(sdA.keys())
    keysB = set(sdB.keys())
    common = sorted(list(keysA & keysB))
    onlyA = sorted(list(keysA - keysB))
    onlyB = sorted(list(keysB - keysA))

    matched = []
    shape_mismatch = []
    for k in common:
        a = sdA[k]
        b = sdB[k]
        if tuple(a.shape) != tuple(b.shape):
            shape_mismatch.append((k, tuple(a.shape), tuple(b.shape)))
        else:
            matched.append((k, a, b))
    return matched, onlyA, onlyB, shape_mismatch


def tensor_diff_stats(a: torch.Tensor, b: torch.Tensor):
    """
    Compute basic diff stats on CPU float32 for stability.
    """
    aa = a.detach().to("cpu")
    bb = b.detach().to("cpu")

    if not torch.is_floating_point(aa):
        aa = aa.float()
    if not torch.is_floating_point(bb):
        bb = bb.float()

    d = (aa - bb).float()
    absd = d.abs()
    max_abs = absd.max().item() if absd.numel() else 0.0
    mean_abs = absd.mean().item() if absd.numel() else 0.0
    l2 = torch.norm(d).item() if d.numel() else 0.0
    return max_abs, mean_abs, l2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--a",
        default="results/results_spk_mel_roformer-with-large-window/melbandroformer_config_by_viperx.ckpt",
        help="Path to checkpoint A",
    )
    ap.add_argument(
        "--b",
        default="results/results_spk_mel_roformer-with-large-window/model_speaker_mel_band_roformer_exportable_ep_0_sdr_0.5771.ckpt",
        help="Path to checkpoint B",
    )
    ap.add_argument("--print_max", type=int, default=100, help="Max number of keys to print shapes for each ckpt")
    ap.add_argument("--no_strip_prefix", action="store_true", help="Do not strip common prefixes when comparing")
    ap.add_argument("--topk", type=int, default=100, help="Show top-K largest differences by max_abs")

    # Write prints to file
    ap.add_argument("--out", type=str, default="./cache/compare_ckpt.log", help="Write all prints to this file")
    ap.add_argument("--tee", action="store_true", help="Also print to console while writing to file")

    args = ap.parse_args()

    with redirect_stdout(args.out, tee=args.tee):
        print("=== Compare Checkpoints ===")
        print(f"A: {args.a}")
        print(f"B: {args.b}")
        print(f"print_max={args.print_max}  topk={args.topk}  strip_prefix={not args.no_strip_prefix}")
        print("")

        ckptA = load_ckpt(args.a)
        ckptB = load_ckpt(args.b)

        sdA_raw, hintA = extract_state_dict(ckptA)
        sdB_raw, hintB = extract_state_dict(ckptB)

        print(f"Loaded A; extracted: {hintA}")
        print(f"Loaded B; extracted: {hintB}")

        # 1) Print shapes first (raw keys)
        print_shapes(sdA_raw, "A (raw keys) shapes", max_print=args.print_max)
        print_shapes(sdB_raw, "B (raw keys) shapes", max_print=args.print_max)

        # 2) Normalize keys and compare
        strip = not args.no_strip_prefix
        sdA = normalize_keys(sdA_raw, strip_prefixes=strip)
        sdB = normalize_keys(sdB_raw, strip_prefixes=strip)

        if strip:
            # detect collisions crudely: if stripping changes counts, collision may have happened
            if len(sdA) != len(sdA_raw):
                print("\n[WARN] A: key collisions may have occurred after prefix stripping (some keys overwritten).")
            if len(sdB) != len(sdB_raw):
                print("\n[WARN] B: key collisions may have occurred after prefix stripping (some keys overwritten).")

        matched, onlyA, onlyB, shape_mismatch = match_keys(sdA, sdB)

        print("\n==== Key set comparison (after normalization) ====")
        print(f"Matched keys (same name): {len(matched)}")
        print(f"Only in A: {len(onlyA)}")
        print(f"Only in B: {len(onlyB)}")
        print(f"Shape mismatches among common keys: {len(shape_mismatch)}")

        if onlyA:
            print("\n-- Only in A (first 50) --")
            for k in onlyA[:50]:
                print(k)
            if len(onlyA) > 50:
                print("...")

        if onlyB:
            print("\n-- Only in B (first 50) --")
            for k in onlyB[:50]:
                print(k)
            if len(onlyB) > 50:
                print("...")

        if shape_mismatch:
            print("\n-- Shape mismatches (first 50) --")
            for k, sa, sb in shape_mismatch[:50]:
                print(f"{k:90s}  A{sa}  vs  B{sb}")
            if len(shape_mismatch) > 50:
                print("...")

        # 3) Diff stats on matched & same-shape
        diffs = []
        for k, a, b in matched:
            max_abs, mean_abs, l2 = tensor_diff_stats(a, b)
            diffs.append((max_abs, mean_abs, l2, k, tuple(a.shape), str(a.dtype), str(b.dtype)))

        diffs.sort(key=lambda x: x[0], reverse=True)

        print("\n==== Weight difference summary (matched & same-shape) ====")
        if len(diffs) == 0:
            print("No comparable tensors found.")
            return

        max_of_max = diffs[0][0]
        mean_of_mean = sum(x[1] for x in diffs) / len(diffs)
        sum_l2 = sum(x[2] for x in diffs)

        print(f"Comparable tensors: {len(diffs)}")
        print(f"Max(max_abs) over tensors: {max_of_max:.6g}")
        print(f"Average(mean_abs) over tensors: {mean_of_mean:.6g}")
        print(f"Sum(L2) over tensors: {sum_l2:.6g}")

        print(f"\n-- Top {args.topk} by max_abs --")
        for i, (max_abs, mean_abs, l2, k, shape, dta, dtb) in enumerate(diffs[:args.topk], 1):
            print(
                f"{i:3d}. {k:90s} shape={shape}  "
                f"max_abs={max_abs:.6g}  mean_abs={mean_abs:.6g}  l2={l2:.6g}  "
                f"dtypeA={dta} dtypeB={dtb}"
            )


if __name__ == "__main__":
    main()
