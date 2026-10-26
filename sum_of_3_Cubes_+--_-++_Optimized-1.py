"""Optimized Python version of sum_of_3_Cubes_+--_-++.py.

Preserves the original sign-transition algorithm and the same seven output
columns. Double-click to launch; the console stays open when finished or if
an error occurs.  Searches 1 <= n <= 999, both +-- and -++.

Requires Python 3.9+. Optionally uses Numba for much faster computation;
without Numba it automatically runs a standard-library-only implementation.
Works on Windows and Linux; keeps output/checkpoint separate from the original.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import traceback
import time
from pathlib import Path

# Optional JIT accelerator. The search remains Python code; when installed,
# Numba compiles only the innermost arithmetic loop into native instructions.
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception as numba_import_error:
    NUMBA_AVAILABLE = False
    NUMBA_IMPORT_ERROR = str(numba_import_error)

# Conservative signed-64-bit JIT bound for this FINITE-DIFFERENCE formulation.
# For s <= 50,000,000, the sign-transition coordinate stays below 4*s+2
# and the finite differences are O(s^2), not O(s^3); the intermediate
# values are comfortably below 2^63-1. Calculating raw cubes *inside*
# the JIT loop would NOT have this guarantee. Initial raw-cube arithmetic
# runs in Python's unbounded integers.
# Beyond this cutoff automatically use exact arbitrary-precision pure Python.
NUMBA_MAX_SECTION = 50_000_000

# =============================
# ===== CONFIG (EDIT HERE) ====
# =============================
CONTRIBUTOR = "Cubes71"
S_START = 29_950_001
S_END = 30_000_000
CORES = 10
BLOCK_SIZE = 50
PROGRESS_INTERVAL = 10_000   # Progress report every N completed sections
CHECKPOINT_EVERY = 1_000    # Save checkpoint every N completed blocks
USE_SL_SKIP = False
SKIP_FIRST_SL = 500_000

# IMPORTANT: The new files must remain SEPARATE from your running Python job.
OUTPUT_PATH = f"sum_of_3_Cubes_optimized_{S_START}_to_{S_END}_output.txt"
CHECKPOINT_FILE = f"sum_of_3_Cubes_optimized_{S_START}_to_{S_END}_checkpoint.json"

# When launched by double-click, do not close the console automatically.
# This also works after an error, and after --self-test finishes.
PAUSE_ON_EXIT = True
# =============================

HEADER = "Contributor\tX\tY\tZ\tn\ts\tsl\n"
VERSION = 1


def cube_difference(s, z, d):
    """F(z,d)=(z+s)^3-(z+d)^3-z^3; used once per section."""
    return (z + s) ** 3 - (z + d) ** 3 - z ** 3


def first_nonnegative_z(s, d):
    """Exact integer search for the LAST z >= 0 with F(z,d) >= 0.

    For 0 <= d < s, F has exactly one positive real zero, so this
    bisection/correction approach does not skip over another root.
    """
    if d == 0:
        # Rapid near-root prediction.  All decisions are corrected using
        # exact (unbounded) Python integers, not floating-point equality.
        z = int(s / (2 ** (1.0 / 3.0) - 1.0))
        while cube_difference(s, z, d) < 0:
            z -= 1
        while cube_difference(s, z + 1, d) >= 0:
            z += 1
        return z

    # General case when the user elects to skip initial section lines.
    lo, hi = 0, 4 * s + d
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if cube_difference(s, mid, d) >= 0:
            lo = mid
        else:
            hi = mid - 1
    return lo


def search_one_section_pure(s, skip_first_sl=0, contributor=CONTRIBUTOR):
    """Return the same candidates as the original process_one_s(s).

    z is the unsigned smallest cube base, d=sl-1.  At fixed d:
      F(z,d) = (z+s)^3-(z+d)^3-z^3
      D = F(z+1,d)-F(z,d)
      E = D(z+1,d)-D(z,d).
    D,E can be updated exactly by additions; the third finite difference
    is -6.  Each new section line changes d to d+1.
    """
    if skip_first_sl >= s:
        return []

    d = skip_first_sl
    z = first_nonnegative_z(s, d)
    n = cube_difference(s, z, d)
    a, b = z + s, z + d
    delta = 3 * (a * a - b * b - z * z) + 3 * (a - b - z) - 1
    second = 6 * (s - d - z) - 6
    hits = []
    add = hits.append

    for sl in range(d + 1, s + 1):
        # These corrections reproduce the original sign-transition logic.
        while n > 0:
            n += delta
            delta += second
            second -= 6
            z += 1
        while n < 0:
            z -= 1
            second += 6
            delta -= second
            n -= delta

        if 0 < n < 1000:
            add(f"{contributor}\t{z+s}\t{-z-d}\t{-z}\t{n}\t{s}\t{sl}\n")

        n_negative = n + delta
        if -1000 < n_negative < 0:
            add(f"{contributor}\t{-z-s-1}\t{z+d+1}\t{z+1}\t{-n_negative}\t{s}\t{sl}\n")

        # Move z -> z+1, then d -> d+1, exactly as the original loop.
        n = n_negative
        delta += second
        second -= 6
        z += 1
        q = z + d
        n -= 3 * q * q + 3 * q + 1
        delta -= 6 * (q + 1)
        second -= 6
        d += 1

    return hits


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _numba_transition_loop(s, z, d, n, delta, second):
        """Fast exact integer finite-difference loop. Return numeric rows.

        The input values and all intermediate residues/differences fit signed
        int64 within the conservative NUMBA_MAX_SECTION limit; the initial
        cube difference is computed using Python's arbitrary precision.
        """
        hits = []
        for sl in range(d + 1, s + 1):
            while n > 0:
                n += delta
                delta += second
                second -= 6
                z += 1
            while n < 0:
                z -= 1
                second += 6
                delta -= second
                n -= delta
            if 0 < n < 1000:
                hits.append((z + s, -z - d, -z, n, s, sl))
            n_negative = n + delta
            if -1000 < n_negative < 0:
                hits.append((-z-s-1, z+d+1, z+1, -n_negative, s, sl))
            n = n_negative
            delta += second
            second -= 6
            z += 1
            q = z + d
            n -= 3 * q * q + 3 * q + 1
            delta -= 6 * (q + 1)
            second -= 6
            d += 1
        return hits


def search_one_section(s, skip_first_sl=0, contributor=CONTRIBUTOR):
    """Auto-select optional Numba acceleration or pure-Python fallback."""
    if skip_first_sl >= s:
        return []
    if not NUMBA_AVAILABLE or s > NUMBA_MAX_SECTION:
        return search_one_section_pure(s, skip_first_sl, contributor)
    d = skip_first_sl
    z = first_nonnegative_z(s, d)
    n = cube_difference(s, z, d)
    a, b = z + s, z + d
    delta = 3 * (a*a - b*b - z*z) + 3 * (a-b-z) - 1
    second = 6 * (s-d-z) - 6
    numeric_rows = _numba_transition_loop(s, z, d, n, delta, second)
    return [f"{contributor}\t{X}\t{Y}\t{Z}\t{N}\t{sec}\t{sl}\n"
            for X, Y, Z, N, sec, sl in numeric_rows]


def process_block(job):
    lo, hi, skip_first_sl, contributor = job
    hits = []
    for s in range(lo, hi + 1):
        hits.extend(search_one_section(s, skip_first_sl, contributor))
    return hi + 1, hits


def original_reference(s, skip_first_sl=0, contributor=CONTRIBUTOR):
    """Independent transcript of original Python steps, for self-tests."""
    x, y = skip_first_sl, 2 * skip_first_sl
    hits = []
    for sl in range(skip_first_sl + 1, s + 1):
        n = (5*s+x)**3 - (4*s+y)**3 - (4*s+x)**3
        while n > 0:
            x += 1; y += 1
            n = (5*s+x)**3 - (4*s+y)**3 - (4*s+x)**3
        while n < 0:
            x -= 1; y -= 1
            n = (5*s+x)**3 - (4*s+y)**3 - (4*s+x)**3
        if 0 < n < 1000:
            hits.append(f"{contributor}\t{5*s+x}\t{-4*s-y}\t{-4*s-x}\t{n}\t{s}\t{sl}\n")
        x += 1; y += 1
        n = (5*s+x)**3 - (4*s+y)**3 - (4*s+x)**3
        if -1000 < n < 0:
            hits.append(f"{contributor}\t{-5*s-x}\t{4*s+y}\t{4*s+x}\t{-n}\t{s}\t{sl}\n")
        y += 1
    return hits


def self_test():
    t0 = time.perf_counter()
    for s in range(1, 501):
        for skip in (0, s // 2, s - 1):
            expected = original_reference(s, skip)
            for method in (search_one_section_pure, search_one_section):
                got = method(s, skip)
                if got != expected:
                    raise AssertionError(f"Mismatch with {method.__name__} "
                                         f"at s={s}, skip={skip}: {got[:4]} != {expected[:4]}")
    for s in (1000, 2000, 5000, 9999, 20000):
        for skip in (0, s // 3):
            expected = original_reference(s, skip)
            if (search_one_section(s, skip) != expected or
                search_one_section_pure(s, skip) != expected):
                raise AssertionError(f"Mismatch at s={s}, skip={skip}")
    target = "Cubes71\t11338054\t-11091439\t-4531539\t126\t6806515\t6559901\n"
    assert target in search_one_section(6_806_515, 6_559_899, "Cubes71")
    print(f"PASS: original-reference comparison, pure and accelerated Python, "
          f"skipped lines, and large n=126 test ({time.perf_counter()-t0:.2f}s)")


def checkpoint_config(args):
    return {
        "version": VERSION,
        "start": args.start,
        "end": args.end,
        "block_size": args.block_size,
        "skip_first_sl": args.skip_first_sl,
        "contributor": args.contributor,
        "output_path": str(Path(args.out).resolve()),
    }


def save_checkpoint(path, settings, next_s, output_bytes):
    """Atomic checkpoint. The corresponding output data must be fsynced first."""
    payload = dict(settings, next_s=next_s, output_bytes=output_bytes)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def init_output(args, settings):
    """Returns (next_s, writable output handle) with safe truncation on resume.

    If an interrupted run wrote results beyond its last checkpoint, truncate
    those *uncommitted* rows; they will be recomputed, avoiding duplicates.
    """
    cp_path = args.checkpoint
    out_path = args.out
    if os.path.exists(cp_path):
        with open(cp_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        for key, val in settings.items():
            if state.get(key) != val:
                raise ValueError(f"Checkpoint mismatch: {key}. Keep the same settings "
                                 "or choose a NEW output/checkpoint file pair.")
        next_s = state["next_s"]
        output_bytes = state["output_bytes"]
        if not (args.start <= next_s <= args.end + 1):
            raise ValueError("Invalid saved next_s")
        if not os.path.exists(out_path) or os.path.getsize(out_path) < output_bytes:
            raise ValueError("Output file missing or shorter than checkpoint; cannot safely resume")
        f = open(out_path, "r+b")
        f.truncate(output_bytes)
        f.seek(output_bytes)
        print(f"Resuming from section {next_s:,}; uncommitted output, if any, was rolled back")
        return next_s, f

    if os.path.exists(out_path):
        raise FileExistsError("Existing output without matching checkpoint. Rename or back up "
                              "it, or set new OUTPUT_PATH and CHECKPOINT_FILE values.")
    f = open(out_path, "w+b")
    f.write(HEADER.encode("utf-8"))
    f.flush()
    os.fsync(f.fileno())
    save_checkpoint(cp_path, settings, args.start, f.tell())
    return args.start, f


def jobs(start, end, block_size, skip_first_sl, contributor):
    for lo in range(start, end + 1, block_size):
        yield (lo, min(end, lo + block_size - 1), skip_first_sl, contributor)


def run(args):
    if args.start < 1 or args.end < args.start:
        raise ValueError("Require 1 <= start <= end")
    if args.cores < 1 or args.block_size < 1 or args.checkpoint_every < 1:
        raise ValueError("Cores, block size, and checkpoint interval must be positive")
    if args.skip_first_sl < 0 or args.progress_interval < 0:
        raise ValueError("Skip and progress interval must be nonnegative")
    if Path(args.out).resolve() == Path(args.checkpoint).resolve():
        raise ValueError("Output and checkpoint must be different files")

    settings = checkpoint_config(args)
    next_s, output_file = init_output(args, settings)
    if next_s > args.end:
        output_file.close()
        print("Already complete. No additional work necessary.")
        return

    engine = ("Numba-accelerated Python" if NUMBA_AVAILABLE else "pure Python")
    print(f"Searching sections {next_s:,} to {args.end:,} on {args.cores} cores")
    print(f"Engine: {engine} (Numba active for sections <= {NUMBA_MAX_SECTION:,})")
    print(f"All n=1..999; +-- and -++; skipping first {args.skip_first_sl:,} lines")
    print(f"Block size={args.block_size}, checkpoint every {args.checkpoint_every} blocks")
    t0 = time.perf_counter()
    blocks_done = 0
    hits = 0
    processed = 0
    committed_next = next_s
    last_print = processed

    try:
        # Ordered imap ensures we commit only a *contiguous prefix* of sections,
        # regardless of the order in which the worker processes finish.
        with mp.Pool(processes=args.cores) as pool:
            iterator = pool.imap(
                process_block,
                jobs(next_s, args.end, args.block_size,
                     args.skip_first_sl, args.contributor),
                chunksize=1,
            )
            for finished_next, block_hits in iterator:
                if block_hits:
                    output_file.write("".join(block_hits).encode("utf-8"))
                    hits += len(block_hits)
                blocks_done += 1
                processed = finished_next - next_s
                committed_next = finished_next
                if blocks_done % args.checkpoint_every == 0:
                    output_file.flush()
                    os.fsync(output_file.fileno())
                    save_checkpoint(args.checkpoint, settings, committed_next,
                                    output_file.tell())
                if (args.progress_interval > 0 and
                    processed - last_print >= args.progress_interval):
                    elapsed = time.perf_counter() - t0
                    speed = processed / elapsed if elapsed else 0
                    print(f"s={finished_next-1:,} | sections={processed:,} "
                          f"| hits={hits:,} | elapsed={elapsed:.1f}s "
                          f"| {speed:.2f} sections/s", flush=True)
                    last_print = processed

        # Only declare final success after ALL sections were returned normally.
        output_file.flush()
        os.fsync(output_file.fileno())
        save_checkpoint(args.checkpoint, settings, args.end + 1, output_file.tell())
        print(f"COMPLETE: {processed:,} sections, {hits:,} new rows, "
              f"{time.perf_counter()-t0:.1f}s. Output: {args.out}")
    except BaseException:
        # Save the latest fully committed prefix even if Ctrl+C was pressed.
        output_file.flush()
        os.fsync(output_file.fileno())
        save_checkpoint(args.checkpoint, settings, committed_next, output_file.tell())
        print(f"Interrupted: safe checkpoint at section {committed_next:,}")
        raise
    finally:
        output_file.close()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=int, default=S_START)
    p.add_argument("--end", type=int, default=S_END)
    p.add_argument("--cores", type=int, default=CORES)
    p.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    p.add_argument("--progress-interval", type=int, default=PROGRESS_INTERVAL)
    p.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    p.add_argument("--skip-first-sl", type=int,
                   default=SKIP_FIRST_SL if USE_SL_SKIP else 0)
    p.add_argument("--contributor", type=str, default=CONTRIBUTOR)
    p.add_argument("--out", type=str, default=OUTPUT_PATH)
    p.add_argument("--checkpoint", type=str, default=CHECKPOINT_FILE)
    p.add_argument("--self-test", action="store_true", help="Check against original algorithm")
    p.add_argument("--no-pause", action="store_true", help="Exit without pressing Enter (for automated/command-line use)")
    return p.parse_args()


def main():
    """Double-click-friendly entry point: always show success or failure."""
    # Running from File Explorer can give an unexpected working directory.
    # Keep output/checkpoint files alongside THIS Python script.
    script_folder = Path(__file__).resolve().parent
    os.chdir(script_folder)
    print("=" * 67, flush=True)
    print("SUM OF THREE CUBES - OPTIMIZED PYTHON SEARCH", flush=True)
    print(f"Folder: {script_folder}", flush=True)
    print("=" * 67, flush=True)
    options = parse_args()
    if not NUMBA_AVAILABLE and 'NUMBA_IMPORT_ERROR' in globals():
        print(f"Numba unavailable ({NUMBA_IMPORT_ERROR}); using pure Python.", flush=True)
    if options.self_test:
        self_test()
    else:
        run(options)


if __name__ == "__main__":
    mp.freeze_support()  # Required for multiprocessing on Windows.
    exit_code = 0
    try:
        main()
    except KeyboardInterrupt:
        exit_code = 130
        print("\nStopped by user (Ctrl+C).", flush=True)
    except Exception:
        exit_code = 1
        print("\nERROR: The search was not completed successfully.", flush=True)
        print("Full error details:", flush=True)
        traceback.print_exc()
        print("Check the output and checkpoint files before restarting.", flush=True)
    finally:
        if PAUSE_ON_EXIT and "--no-pause" not in sys.argv:
            print("\n" + "=" * 67, flush=True)
            print("The window will stay open so you can read the results.", flush=True)
            try:
                input("Press ENTER when you are ready to close this window... ")
            except (EOFError, KeyboardInterrupt):
                # No interactive console, e.g., scheduled/background execution.
                pass
    sys.exit(exit_code)
