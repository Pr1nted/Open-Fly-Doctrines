"""Pack Open Fly's connectome into something a mod can hold in 64 MiB.

    python3 tools/pack_connectome.py --fly <Open-Fly checkout> --out data/brain.odfly

WHY THIS EXISTS

A Gearbox mod gets 1024 memory pages -- 64 MiB -- and connectome.bin is 91 MB.
That is the whole problem, and it is a memory problem rather than a download
one: the simulator needs random access to every row, so compressing the file
without compressing what sits in memory buys nothing.

WHAT WAS TRIED FIRST, AND WHY IT FAILED

Pruning to the neurons on a path from a sense to a descending neuron. Measured:
97.7% of neurons and 99.9% of synapses survive that cut. The fly brain is
densely connected and there is no fat to trim -- the pruned model would have
been 91.0 MB and no longer the same model.

WHAT WORKS INSTEAD, AND COSTS NOTHING

The CSR rows are sorted by target (the exporter lexsorts on (pre, post)), so
within a row the indices ascend and their DELTAS are small. Delta plus varint
is LOSSLESS -- every index and every synapse count comes back exactly -- and it
halves the thing:

    indices  uint32 -> varint    60.4 MB -> 27.1 MB
    weights  int16  -> varint    30.2 MB -> 15.1 MB
                                 ------------------
                                 91.0 MB -> 42.8 MB

With ~3.5 MB of simulator state that is 46.3 MB, which fits with room to spare.
No neuron is dropped, no synapse is dropped, and no timestep is coarsened: the
model the mod runs is the model Open Fly runs.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

MAGIC = b"ODFLY1\0\0"
CHANNELS = ("sugar", "bitter", "water", "jon")


def varint(v: int, out: bytearray) -> None:
    """LEB128, unsigned. Small deltas cost one byte, which is the whole point."""
    while True:
        b = v & 0x7F
        v >>= 7
        out.append(b | (0x80 if v else 0))
        if not v:
            return


def zigzag(v: int) -> int:
    """Signed to unsigned without making small negatives expensive."""
    return (v << 1) ^ (v >> 63) if v < 0 else (v << 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fly", required=True, help="an Open-Fly checkout")
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", action="store_true",
                    help="decode it again and assert it is identical")
    args = ap.parse_args()

    con_path = os.path.join(args.fly, "web", "data", "connectome.bin")
    meta_path = os.path.join(args.fly, "web", "data", "brain.json")
    for p in (con_path, meta_path):
        if not os.path.exists(p):
            print(f"missing {p}", file=sys.stderr)
            return 1

    raw = np.fromfile(con_path, dtype=np.uint8)
    if raw[:4].tobytes() != b"SFC1":
        print("connectome.bin has the wrong header", file=sys.stderr)
        return 1
    n, nsyn = struct.unpack("<II", raw[4:12].tobytes())
    off = 12
    indptr = raw[off:off + (n + 1) * 4].view("<u4").astype(np.int64); off += (n + 1) * 4
    post = raw[off:off + nsyn * 4].view("<u4").astype(np.int64); off += nsyn * 4
    cnt = raw[off:off + nsyn * 2].view("<i2").astype(np.int64)

    meta = json.load(open(meta_path))
    assert meta["n"] == n, (meta["n"], n)

    # ---- the varint blob, one row at a time ------------------------------
    blob = bytearray()
    row_off = np.zeros(n + 1, np.uint32)
    for i in range(n):
        row_off[i] = len(blob)
        s, e = indptr[i], indptr[i + 1]
        prev = 0
        for k in range(s, e):
            # First edge of a row is absolute; the rest are deltas from the
            # previous target, which is why the row must stay sorted.
            varint(int(post[k]) - prev, blob)
            varint(zigzag(int(cnt[k])), blob)
            prev = int(post[k])
    row_off[n] = len(blob)

    # ---- the header ------------------------------------------------------
    #
    # Everything the simulator needs to start, so the mod reads ONE file and
    # never has to parse JSON: no allocator, no parser, no failure mode that
    # only shows up on somebody else's machine.
    p = meta["params"]
    head = bytearray()
    head += MAGIC
    head += struct.pack("<II", n, nsyn)
    head += struct.pack("<10d", p["v_0"], p["v_rst"], p["v_th"], p["t_mbr"],
                        p["tau"], p["t_rfc"], p["t_dly"], p["w_syn"],
                        float(p["f_poi"]), p["dt"])
    for ch in CHANNELS:
        idx = meta["sensory"][ch]
        head += struct.pack("<I", len(idx))
        head += np.asarray(idx, "<u4").tobytes()
    dn = sorted(set(int(i) for i in meta["dn"]))
    head += struct.pack("<I", len(dn))
    head += np.asarray(dn, "<u4").tobytes()
    head += struct.pack("<I", int(meta["partition_seed"]))
    head += struct.pack("<Q", len(blob))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(head)
        f.write(row_off.astype("<u4").tobytes())
        f.write(bytes(blob))

    size = os.path.getsize(args.out)
    print(f"  neurons        {n:,}")
    print(f"  synapses       {nsyn:,}")
    print(f"  sensory        {sum(len(meta['sensory'][c]) for c in CHANNELS)}")
    print(f"  descending     {len(dn)}")
    print(f"  header         {len(head):,} B")
    print(f"  row offsets    {(n + 1) * 4 / 1e6:.1f} MB")
    print(f"  varint blob    {len(blob) / 1e6:.1f} MB")
    print(f"  TOTAL          {size / 1e6:.1f} MB   (connectome.bin is "
          f"{os.path.getsize(con_path) / 1e6:.0f} MB)")

    if args.verify:
        ok = verify(args.out, indptr, post, cnt, n, nsyn)
        print("  lossless       " + ("yes, every index and weight" if ok else "NO"))
        return 0 if ok else 1
    return 0


def verify(path, indptr, post, cnt, n, nsyn) -> bool:
    """Decode it back and compare against what went in.

    Not a checksum: a checksum would tell us the bytes round-tripped, which is
    not the claim. The claim is that the DECODER reconstructs the same graph,
    so this runs the decoder."""
    data = open(path, "rb").read()
    o = len(MAGIC) + 8 + 80
    for _ in range(4):
        c = struct.unpack_from("<I", data, o)[0]; o += 4 + c * 4
    c = struct.unpack_from("<I", data, o)[0]; o += 4 + c * 4
    o += 4 + 8
    row_off = np.frombuffer(data, "<u4", count=n + 1, offset=o); o += (n + 1) * 4
    blob = memoryview(data)[o:]

    seen = 0
    for i in range(n):
        k, end = int(row_off[i]), int(row_off[i + 1])
        prev, e = 0, int(indptr[i])
        while k < end:
            d = 0; shift = 0
            while True:
                b = blob[k]; k += 1
                d |= (b & 0x7F) << shift; shift += 7
                if not (b & 0x80): break
            w = 0; shift = 0
            while True:
                b = blob[k]; k += 1
                w |= (b & 0x7F) << shift; shift += 7
                if not (b & 0x80): break
            w = (w >> 1) ^ -(w & 1)
            prev += d
            if prev != int(post[e]) or w != int(cnt[e]):
                print(f"  MISMATCH at neuron {i}, edge {e}: "
                      f"{prev}/{w} vs {int(post[e])}/{int(cnt[e])}", file=sys.stderr)
                return False
            e += 1; seen += 1
    return seen == nsyn


if __name__ == "__main__":
    raise SystemExit(main())
