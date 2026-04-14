# dynamic_blocks_sum_three_cubes_writer.py
# Dynamic block scheduling + threaded writer (no file locks in workers).
# Workers push result lines to a multiprocessing.Queue; a single writer
# thread in the main process drains/batches and writes to disk.

import multiprocessing as mp
from multiprocessing import Manager
from queue import Empty as QueueEmpty
import threading
import time
import os

# ============================
# ====== CONFIG (edit) =======
# ============================
M_START = 5_310_001
M_END   = 5_350_000

CORES = 5                    # auto-caps to available CPUs
BLOCK_SIZE = 5_000          # number of m-values per task block
PROGRESS_INTERVAL = 50    # per worker: print every this many m’s processed
OUTPUT_PATH = "output.txt"

# Writer batching / queue behavior
WRITER_BATCH_SIZE = 512      # number of lines per batch write
WRITER_FLUSH_SEC  = 0.5      # flush whatever is collected at least this often
QUEUE_MAXSIZE     = 50_000   # back-pressure cap; tune for your RAM/throughput
# ============================

# ---------- your math (unchanged) ----------
def calculate_n(m, x, y):
    return (5 * m + x) ** 3 - (4 * m + y) ** 3 - (4 * m + x) ** 3
 
def to_XYZ(m, x, y):
    X = (5 * m + x)
    Y = -(4 * m + y)
    Z = -(4 * m + x)
    return X, Y, Z

def process_one_m(m):
    """
    Run your original (x,y) walk for a single m and yield lines "X\tY\tZ\tn".
    The zero always occurs at ridge m+1, so we stop after m ridges
    instead of scanning until n==0 -- eliminates one full ridge per m.
    """
    x = 0
    y = 0
    for _ in range(m):
        n = calculate_n(m, x, y)

        if n > 0:
            while n > 0:
                x += 1
                y += 1
                n = calculate_n(m, x, y)

        if n < 0:
            while n < 0:
                x -= 1
                y -= 1
                n = calculate_n(m, x, y)

        if 0 < n < 1000:
            X, Y, Z = to_XYZ(m, x, y)
            yield f"{X}\t{Y}\t{Z}\t{n}"

        y += 1

# ---------- threaded writer ----------
SENTINEL = None

def writer_thread_fn(out_q, output_path: str,
                     batch_size: int, flush_sec: float):

    """
    Drain out_q, batch lines, and write to disk. Stop when SENTINEL is received
    and the queue is empty.
    """
    buf = []
    last_flush = time.time()

    # write header once
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# X\tY\tZ\tn\n")

    f = open(output_path, "a", encoding="utf-8")
    try:
        while True:
            # Time-based flush to avoid long delays if queue is quiet
            timeout = max(0.0, flush_sec - (time.time() - last_flush))
            try:
                item = out_q.get(timeout=timeout)
            except Exception:
                item = ...  # timeout sentinel for periodic flush

            if item is SENTINEL:
                # Drain any remaining items quickly
                while True:
                    try:
                        nxt = out_q.get_nowait()
                        if nxt is SENTINEL:
                            continue
                        buf.append(nxt)
                    except QueueEmpty:
                        break
                if buf:
                    f.write("\n".join(buf) + "\n"); f.flush(); buf.clear()
                break

            if item is not ...:
                buf.append(item)

            # Flush on size or time
            now = time.time()
            if len(buf) >= batch_size or (now - last_flush) >= flush_sec:
                if buf:
                    f.write("\n".join(buf) + "\n"); f.flush()
                    buf.clear()
                last_flush = now
    finally:
        f.close()

# ---------- worker: dynamic blocks, push to queue ----------
def worker(proc_idx, task_q, out_q, progress_interval):
    hits = 0
    processed_m = 0
    t0 = time.time()
    _process_one_m = process_one_m

    while True:
        try:
            ms, me = task_q.get_nowait()
        except QueueEmpty:
            break

        for m in range(ms, me + 1):
            processed_m += 1
            if progress_interval and (processed_m % progress_interval == 0):
                dt = time.time() - t0
                print(f"[P{proc_idx}] m={m} (+{progress_interval}), hits={hits}, time={dt:.1f}s", flush=True)

            for line in _process_one_m(m):
                # Back-pressure: blocks when queue is full
                out_q.put(line)
                hits += 1

    print(f"[P{proc_idx}] done. hits={hits}", flush=True)

# ---------- block builder ----------
def build_blocks(m_start, m_end, block_size):
    a = m_start
    while a <= m_end:
        b = min(a + block_size - 1, m_end)
        yield (a, b)
        a = b + 1

def main():
    print(f"Scanning m in [{M_START}, {M_END}] with BLOCK_SIZE={BLOCK_SIZE}")
    max_procs = max(1, min(CORES, mp.cpu_count() or 1))
    print(f"Using {max_procs} worker(s). Output -> {os.path.abspath(OUTPUT_PATH)}")

    with Manager() as mgr:
        # Dynamic task queue of m-blocks
        task_q = mgr.Queue()
        for blk in build_blocks(M_START, M_END, BLOCK_SIZE):
            task_q.put(blk)

        # Multiprocessing queue for results; read by a thread in main proc
        out_q = mp.Queue(maxsize=QUEUE_MAXSIZE)

        # Start writer thread
        wt = threading.Thread(
            target=writer_thread_fn,
            args=(out_q, OUTPUT_PATH, WRITER_BATCH_SIZE, WRITER_FLUSH_SEC),
            daemon=True
        )
        wt.start()

        # Launch workers
        procs = []
        for i in range(1, max_procs + 1):
            p = mp.Process(target=worker,
                           args=(i, task_q, out_q, PROGRESS_INTERVAL),
                           daemon=False)
            p.start()
            procs.append(p)

        # Wait for workers, then tell writer to stop
        for p in procs:
            p.join()

        out_q.put(SENTINEL)   # signal writer to finish
        wt.join()

    print("All done.")
    input("\nPress ENTER to close...")

if __name__ == "__main__":
    try:
        mp.freeze_support()
        main()
    except Exception as e:
        import traceback
        print("\nERROR:", e)
        traceback.print_exc()
        input("\nPress ENTER to close...")
