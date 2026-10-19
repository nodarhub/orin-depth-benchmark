"""Per-frame GPU time from an Nsight Systems sqlite export.

GPU work (kernels + memcpys) is merged into busy intervals; frames are separated
by idle gaps longer than --gap_ms. For each frame we report:
  busy_ms: union of GPU activity (true GPU compute cost per frame)
  span_ms: first-to-last GPU activity (lower bound on GPU latency)
Optionally splits busy time by kernel-name regex categories.
"""
import argparse
import json
import re
import sqlite3

import numpy as np


def load_intervals(db):
    con = sqlite3.connect(db)
    rows = []
    q = ("SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
         "JOIN StringIds s ON k.shortName = s.id")
    rows += [(a, b, n) for a, b, n in con.execute(q)]
    try:
        rows += [(a, b, "memcpy") for a, b in
                 con.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY")]
    except sqlite3.OperationalError:
        pass
    rows.sort()
    return rows


def union_len(iv):
    total, cur_s, cur_e = 0, None, None
    for s, e in iv:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("db")
    p.add_argument("--gap_ms", type=float, default=20)
    p.add_argument("--skip", type=int, default=3, help="drop first/last N frames")
    p.add_argument("--categories", default="{}", help='JSON {"name": "regex"}')
    a = p.parse_args()
    cats = {k: re.compile(v) for k, v in json.loads(a.categories).items()}
    rows = load_intervals(a.db)
    frames, cur, last_end = [], [], None
    for r in rows:
        if last_end is not None and r[0] - last_end > a.gap_ms * 1e6:
            frames.append(cur)
            cur = []
        cur.append(r)
        last_end = r[1] if last_end is None else max(last_end, r[1])
    frames.append(cur)
    frames = frames[a.skip:len(frames) - a.skip]
    busy = np.array([union_len([(s, e) for s, e, _ in f]) for f in frames]) / 1e6
    span = np.array([max(e for _, e, _ in f) - f[0][0] for f in frames]) / 1e6
    out = {"frames": len(frames), "busy_ms_median": float(np.median(busy)),
           "busy_ms_p90": float(np.percentile(busy, 90)), "span_ms_median": float(np.median(span))}
    for name, rx in cats.items():
        cb = [union_len([(s, e) for s, e, n in f if rx.search(n)]) / 1e6 for f in frames]
        out[f"busy_ms_{name}"] = float(np.median(cb))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
