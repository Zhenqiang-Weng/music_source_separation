"""
Merge two model checkpoint files.
- Replaces matching keys in model A with values from model B (exact key match).
- Additionally tries to replace "semantically corresponding" weights via key mapping rules
  when A/B use different naming (e.g., block index shifts).
- Writes reports into .cache:
    * keys_only_in_A.txt
    * keys_only_in_B.txt
    * replaced_keys.txt
    * mismatch_keys.txt
    * renamed_replaced_keys.txt
    * renamed_candidates_not_used.txt
"""

import os
import re
import torch

# Hardcoded paths
PATH_A = "results/results_spk_mel_roformer-with-large-window/model_speaker_mel_band_roformer_exportable_ep_0_sdr_0.5771.ckpt"
PATH_B = "results/results_spk_mel_roformer-with-large-window/melbandroformer_config_by_viperx.ckpt"
OUTPUT_PATH = "results/results_spk_mel_roformer-with-large-window/output.ckpt"

# Dump paths (project-local .cache)
CACHE_DIR = ".cache"
ONLY_A_TXT = os.path.join(CACHE_DIR, "keys_only_in_A.txt")
ONLY_B_TXT = os.path.join(CACHE_DIR, "keys_only_in_B.txt")
REPLACED_TXT = os.path.join(CACHE_DIR, "replaced_keys.txt")
MISMATCH_TXT = os.path.join(CACHE_DIR, "mismatch_keys.txt")
RENAMED_REPLACED_TXT = os.path.join(CACHE_DIR, "renamed_replaced_keys.txt")
RENAMED_MISS_TXT = os.path.join(CACHE_DIR, "renamed_candidates_not_used.txt")


def _extract_state_dict(checkpoint):
    """
    Normalize different checkpoint formats into a plain state_dict-like dict.
    Supports:
      - plain state_dict dict
      - dict with 'state_dict' field
    """
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")


def _write_lines(path, lines, header=None):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if header:
            f.write(header.rstrip() + "\n")
        for line in lines:
            f.write(line + "\n")


def _is_tensor_like(x) -> bool:
    return hasattr(x, "shape") and hasattr(x, "dtype")


def _shape_tuple(x):
    return tuple(x.shape) if hasattr(x, "shape") else None


def _desc_value(x) -> str:
    if _is_tensor_like(x):
        return f"shape={tuple(x.shape)},dtype={str(x.dtype)}"
    return f"type={type(x).__name__}"


# --- Key mapping rules (A key -> B key candidates) ---
# You can extend these if you discover more systematic renames.
A2B_RULES = [
    # Attention sub-block index shift:
    # A: layers.*.*.layers.0.1.(norm/rotary/to_*)
    # B: layers.*.*.layers.0.0.(norm/rotary/to_*)
    (re.compile(r"\.layers\.0\.1\."), ".layers.0.0."),

    # FFN sub-block index shift:
    # A: layers.*.*.layers.0.3.net.*
    # B: layers.*.*.layers.0.1.net.*
    (re.compile(r"\.layers\.0\.3\.net\."), ".layers.0.1.net."),
]


def map_a_key_to_b_key(a_key: str) -> str:
    b_key = a_key
    for pattern, repl in A2B_RULES:
        b_key = pattern.sub(repl, b_key)
    return b_key


def _replace_if_compatible(
    merged_state_dict: dict,
    a_key: str,
    b_key: str,
    state_dict_a: dict,
    state_dict_b: dict,
):
    """
    Attempt replacement of merged_state_dict[a_key] with state_dict_b[b_key]
    if compatible. Returns (ok: bool, reason: str, a_desc: str, b_desc: str)
    """
    va = state_dict_a[a_key]
    vb = state_dict_b[b_key]

    # tensor-tensor: must match shape
    if _is_tensor_like(va) and _is_tensor_like(vb):
        sa = _shape_tuple(va)
        sb = _shape_tuple(vb)
        if sa == sb:
            merged_state_dict[a_key] = vb
            return True, "shape_match", _desc_value(va), _desc_value(vb)
        return False, "shape_mismatch", _desc_value(va), _desc_value(vb)

    # non-tensor: replace only if same python type
    if type(va) == type(vb):
        merged_state_dict[a_key] = vb
        return True, "type_match", _desc_value(va), _desc_value(vb)

    return False, "type_mismatch", _desc_value(va), _desc_value(vb)


def merge_model_weights(path_a, path_b, output_path, verbose=True):
    if verbose:
        print(f"Loading model A from: {path_a}")
    checkpoint_a = torch.load(path_a, map_location="cpu")

    if verbose:
        print(f"Loading model B from: {path_b}")
    checkpoint_b = torch.load(path_b, map_location="cpu")

    state_dict_a = _extract_state_dict(checkpoint_a)
    state_dict_b = _extract_state_dict(checkpoint_b)

    keys_a = set(state_dict_a.keys())
    keys_b = set(state_dict_b.keys())
    matching_keys = keys_a & keys_b

    keys_only_in_a = keys_a - keys_b
    keys_only_in_b = keys_b - keys_a

    if verbose:
        print(f"\nModel A contains {len(keys_a)} keys")
        print(f"Model B contains {len(keys_b)} keys")
        print(f"Found {len(matching_keys)} exact matching keys")

    merged_state_dict = state_dict_a.copy()

    # Logs
    replaced_logs = []
    mismatch_logs = []
    renamed_replaced_logs = []
    renamed_candidate_not_used = []

    # --- Phase 1: exact key match replacement ---
    replaced_count = 0
    mismatch_count = 0

    for key in sorted(matching_keys):
        ok, reason, a_desc, b_desc = _replace_if_compatible(
            merged_state_dict=merged_state_dict,
            a_key=key,
            b_key=key,
            state_dict_a=state_dict_a,
            state_dict_b=state_dict_b,
        )
        if ok:
            replaced_count += 1
            replaced_logs.append(f"{key}\tA<-B\t{a_desc}")
            if verbose:
                print(f"✓ Replaced: {key} ({a_desc})")
        else:
            mismatch_count += 1
            mismatch_logs.append(f"{key}\tKEEP_A\t{reason}\tA:{a_desc}\tB:{b_desc}")
            if verbose:
                print(f"✗ Not replaced: {key} ({reason}) A:{a_desc} B:{b_desc}")

    # --- Phase 2: mapped key replacement (for non-matching names) ---
    renamed_replaced_count = 0

    # Try mapping only for keys only in A (since matching keys already handled)
    for a_key in sorted(keys_only_in_a):
        b_key = map_a_key_to_b_key(a_key)
        if b_key == a_key:
            continue

        if b_key not in state_dict_b:
            renamed_candidate_not_used.append(f"{a_key}\t->\t{b_key}\tB key not found")
            continue

        ok, reason, a_desc, b_desc = _replace_if_compatible(
            merged_state_dict=merged_state_dict,
            a_key=a_key,
            b_key=b_key,
            state_dict_a=state_dict_a,
            state_dict_b=state_dict_b,
        )
        if ok:
            renamed_replaced_count += 1
            renamed_replaced_logs.append(f"{a_key}\t<=\t{b_key}\t{a_desc}")
            if verbose:
                print(f"✓ Renamed replace: {a_key} <= {b_key} ({a_desc})")
        else:
            renamed_candidate_not_used.append(
                f"{a_key}\t->\t{b_key}\t{reason}\tA:{a_desc}\tB:{b_desc}"
            )
            if verbose:
                print(f"✗ Renamed not used: {a_key} -> {b_key} ({reason})")

    # Recompute "only in" after renamed replacements?
    # Note: we replaced values at A keys; the key sets themselves don't change.
    # These files describe raw checkpoint key differences, not post-merge values.
    # If you want post-merge "still unmatched after mapping", we can compute it,
    # but keeping it simple and transparent here.

    # Save merged checkpoint (state_dict only, consistent with your original behavior)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    if verbose:
        print(f"\nSaving merged checkpoint to: {output_path}")
    torch.save(merged_state_dict, output_path)

    # Write reports
    os.makedirs(CACHE_DIR, exist_ok=True)
    _write_lines(ONLY_A_TXT, sorted(keys_only_in_a), header=f"Keys only in A ({len(keys_only_in_a)}):")
    _write_lines(ONLY_B_TXT, sorted(keys_only_in_b), header=f"Keys only in B ({len(keys_only_in_b)}):")
    _write_lines(REPLACED_TXT, replaced_logs, header=f"Exact-key replaced (A <- B) ({len(replaced_logs)}):")
    _write_lines(MISMATCH_TXT, mismatch_logs, header=f"Exact-key mismatches kept from A ({len(mismatch_logs)}):")
    _write_lines(
        RENAMED_REPLACED_TXT,
        renamed_replaced_logs,
        header=f"Renamed-key replaced (A_key <= B_mapped_key) ({len(renamed_replaced_logs)}):",
    )
    _write_lines(
        RENAMED_MISS_TXT,
        renamed_candidate_not_used,
        header=f"Renamed candidates not used ({len(renamed_candidate_not_used)}):",
    )

    print(f"\n{'='*60}")
    print("Merge Summary:")
    print(f"  Total keys in A: {len(keys_a)}")
    print(f"  Total keys in B: {len(keys_b)}")
    print(f"  Exact matching keys: {len(matching_keys)}")
    print(f"  Exact replaced: {replaced_count}")
    print(f"  Exact mismatches kept A: {mismatch_count}")
    print(f"  Keys only in A (raw): {len(keys_only_in_a)}")
    print(f"  Keys only in B (raw): {len(keys_only_in_b)}")
    print(f"  Renamed replacements via mapping: {renamed_replaced_count}")
    print(f"{'='*60}\n")

    if verbose:
        print("Saved reports to:")
        print(f"  - {ONLY_A_TXT}")
        print(f"  - {ONLY_B_TXT}")
        print(f"  - {REPLACED_TXT}")
        print(f"  - {MISMATCH_TXT}")
        print(f"  - {RENAMED_REPLACED_TXT}")
        print(f"  - {RENAMED_MISS_TXT}")
        print("✓ Merge completed successfully!")

    return {
        "total_keys_a": len(keys_a),
        "total_keys_b": len(keys_b),
        "matching_keys": len(matching_keys),
        "exact_replaced": replaced_count,
        "exact_mismatch": mismatch_count,
        "only_in_a": len(keys_only_in_a),
        "only_in_b": len(keys_only_in_b),
        "renamed_replaced": renamed_replaced_count,
        "only_a_txt": ONLY_A_TXT,
        "only_b_txt": ONLY_B_TXT,
        "replaced_txt": REPLACED_TXT,
        "mismatch_txt": MISMATCH_TXT,
        "renamed_replaced_txt": RENAMED_REPLACED_TXT,
        "renamed_miss_txt": RENAMED_MISS_TXT,
    }


def main():
    if not os.path.exists(PATH_A):
        raise FileNotFoundError(f"Model A not found: {PATH_A}")
    if not os.path.exists(PATH_B):
        raise FileNotFoundError(f"Model B not found: {PATH_B}")

    merge_model_weights(
        path_a=PATH_A,
        path_b=PATH_B,
        output_path=OUTPUT_PATH,
        verbose=True
    )


if __name__ == "__main__":
    main()