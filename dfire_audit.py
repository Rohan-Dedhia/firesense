"""
FireSense - audit the D-Fire image dataset before any training touches it.

    python dfire_audit.py --data ~/datasets/d-fire/extracted

Answers the questions that decide whether a test number can be trusted:

  * how many images per split and per image-level class
    (fire only, smoke only, both, neither)
  * does any filename appear in both train and test
  * does any image appear in both, byte-identical or visually near-identical
  * do filenames look like consecutive video frames - if so, a random
    train/validation split would put near-copies on both sides, which is the
    same leak CLAUDE.md rule 3 forbids for sensor sessions

It also writes the frame groups used by train_vision.py for its validation
split (groups.json next to the data), so the grouping is decided once, from
measurement, and reused.

Nothing here modifies the dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SPLITS = ("train", "test")
CLASSES = ("neither", "fire only", "smoke only", "both")

# D-Fire YOLO class ids, as shipped by the authors.
SMOKE_ID, FIRE_ID = 0, 1

# Two 64-bit difference hashes this close are treated as "the same picture".
# 64-bit dHash: 0 = identical thumbnails; a few bits = re-encode, slight crop or
# the next frame of a static camera. Unrelated photos sit around 32.
NEAR_DUP_BITS = 4
# Consecutive file numbers this similar are taken as frames of one video.
# Looser than NEAR_DUP_BITS on purpose: neighbouring frames of a moving
# scene differ more than a re-encode does, and grouping too much only costs
# a little validation variance, while grouping too little leaks.
FRAME_BITS = 10

NAME_RE = re.compile(r"^([A-Za-z]+)(\d+)\.\w+$")


def image_label(label_path: Path) -> tuple[int, int]:
    """(fire, smoke) for one image, from its YOLO box file. Empty = neither."""
    fire = smoke = 0
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            cid = int(float(parts[0]))
            if cid == FIRE_ID:
                fire = 1
            elif cid == SMOKE_ID:
                smoke = 1
    return fire, smoke


def class_name(fire: int, smoke: int) -> str:
    return CLASSES[fire + 2 * smoke]


def _hash_one(path: str) -> tuple[str, str, int, tuple[int, int]]:
    """sha256 of the bytes, a 64-bit dHash of the picture, and its size."""
    from PIL import Image

    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    with Image.open(path) as im:
        size = im.size
        g = im.convert("L").resize((9, 8), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    dh = int("".join("1" if b else "0" for b in bits), 2)
    return path, sha, dh, size


def hamming(a: np.ndarray, b: int) -> np.ndarray:
    """Bit distance between every hash in `a` (uint64) and one hash `b`."""
    x = np.bitwise_xor(a, np.uint64(b))
    # popcount, byte by byte - plain numpy, no extra dependency
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(Path.home() / "datasets/d-fire/extracted"))
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    root = Path(a.data).expanduser()

    items = []  # (split, name, path, fire, smoke)
    for split in SPLITS:
        for img in sorted((root / split / "images").glob("*.jpg")):
            f, s = image_label(root / split / "labels" / (img.stem + ".txt"))
            items.append((split, img.name, str(img), f, s))

    # -- counts ---------------------------------------------------------------
    print("\n  images per split and class")
    counts = {sp: Counter() for sp in SPLITS}
    for sp, _, _, f, s in items:
        counts[sp][class_name(f, s)] += 1
    print(f"  {'split':6s} " + " ".join(f"{c:>11s}" for c in CLASSES) + f" {'total':>7s}")
    for sp in SPLITS:
        tot = sum(counts[sp].values())
        print(f"  {sp:6s} " + " ".join(f"{counts[sp][c]:11d}" for c in CLASSES)
              + f" {tot:7d}")

    # -- name overlap ---------------------------------------------------------
    names = {sp: {n for s, n, *_ in items if s == sp} for sp in SPLITS}
    name_overlap = sorted(names["train"] & names["test"])
    print(f"\n  filenames in both train and test: {len(name_overlap)}")

    # -- prefixes and number ranges --------------------------------------------
    ranges = {}
    print("\n  filename prefixes (number range per split)")
    for sp in SPLITS:
        by = defaultdict(list)
        for s, n, *_ in items:
            if s == sp and (m := NAME_RE.match(n)):
                by[m.group(1)].append(int(m.group(2)))
        for pre, nums in sorted(by.items()):
            ranges[f"{sp}/{pre}"] = (min(nums), max(nums), len(nums))
            print(f"  {sp:6s} {pre:14s} {len(nums):6d}  {min(nums):6d} .. {max(nums):6d}")

    # -- hashes -----------------------------------------------------------------
    print(f"\n  hashing {len(items)} images...")
    with ProcessPoolExecutor(a.workers) as ex:
        hashed = list(ex.map(_hash_one, [it[2] for it in items], chunksize=64))
    info = {path: (sha, dh, size) for path, sha, dh, size in hashed}

    sizes = Counter(info[it[2]][2] for it in items)
    print("  most common image sizes:", ", ".join(
        f"{w}x{h} ({c})" for (w, h), c in sizes.most_common(5)))

    sha_split = defaultdict(set)
    for sp, _, path, *_ in items:
        sha_split[info[path][0]].add(sp)
    exact_cross = sum(1 for v in sha_split.values() if len(v) == 2)
    sha_count = Counter(info[it[2]][0] for it in items)
    exact_within = sum(c - 1 for c in sha_count.values() if c > 1)
    print(f"  byte-identical images across train/test : {exact_cross}")
    print(f"  byte-identical extra copies (any split) : {exact_within}")

    tr = [it for it in items if it[0] == "train"]
    te = [it for it in items if it[0] == "test"]
    tr_h = np.array([info[it[2]][1] for it in tr], dtype=np.uint64)
    near_cross = []
    for it in te:
        d = hamming(tr_h, info[it[2]][1])
        j = int(d.argmin())
        if d[j] <= NEAR_DUP_BITS:
            near_cross.append((it[1], tr[j][1], int(d[j])))
    print(f"  test images with a near-duplicate in train (dHash <= {NEAR_DUP_BITS} bits): "
          f"{len(near_cross)} of {len(te)}")
    near_by_prefix = Counter(NAME_RE.match(t).group(1) for t, _, _ in near_cross)
    for pre, c in sorted(near_by_prefix.items()):
        n_pre = sum(1 for it in te if it[1].startswith(pre))
        print(f"    {pre:14s} {c:5d} of {n_pre}")

    # -- video-frame structure inside train -----------------------------------
    # For each prefix, how similar are images whose numbers are consecutive?
    # If neighbours look alike far more than random pairs do, the numbering
    # follows video order and neighbours must stay on the same side of a split.
    by_name = {it[1]: it for it in tr}
    rng = np.random.default_rng(0)
    print("\n  consecutive-number similarity in train (median dHash bits)")
    frame_stats = {}
    for pre in sorted({NAME_RE.match(it[1]).group(1) for it in tr}):
        nums = sorted(int(NAME_RE.match(it[1]).group(2)) for it in tr
                      if it[1].startswith(pre))
        width = len(NAME_RE.match(next(it[1] for it in tr if it[1].startswith(pre))).group(2))
        nb = []
        for n0, n1 in zip(nums, nums[1:]):
            if n1 == n0 + 1:
                h0 = info[by_name[f"{pre}{n0:0{width}d}.jpg"][2]][1]
                h1 = info[by_name[f"{pre}{n1:0{width}d}.jpg"][2]][1]
                nb.append(int(hamming(np.array([h0], dtype=np.uint64), h1)[0]))
        idx = [i for i, it in enumerate(tr) if it[1].startswith(pre)]
        rnd = [int(hamming(np.array([tr_h[i]], dtype=np.uint64), int(tr_h[j]))[0])
               for i, j in rng.choice(idx, size=(min(2000, len(idx)), 2))]
        close = sum(1 for d in nb if d <= FRAME_BITS)
        frame_stats[pre] = {"neighbour_median": float(np.median(nb)) if nb else None,
                            "random_median": float(np.median(rnd)),
                            "neighbours_within_frame_bits": close,
                            "neighbour_pairs": len(nb)}
        print(f"  {pre:14s} neighbours {np.median(nb) if nb else float('nan'):5.1f}   "
              f"random pairs {np.median(rnd):5.1f}   "
              f"{close}/{len(nb)} neighbour pairs within {FRAME_BITS} bits")

    # -- groups for the validation split -----------------------------------------
    # Union-find over train: link consecutive numbers that look alike
    # (FRAME_BITS) and any two images that are near-duplicates (NEAR_DUP_BITS),
    # wherever they sit in the numbering.
    parent = list(range(len(tr)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    pos = {it[1]: i for i, it in enumerate(tr)}
    for i, it in enumerate(tr):
        m = NAME_RE.match(it[1])
        nxt = f"{m.group(1)}{int(m.group(2)) + 1:0{len(m.group(2))}d}.jpg"
        if nxt in pos:
            j = pos[nxt]
            if int(hamming(np.array([tr_h[i]], dtype=np.uint64), int(tr_h[j]))[0]) <= FRAME_BITS:
                union(i, j)
    for i in range(len(tr)):
        d = hamming(tr_h[i + 1:], int(tr_h[i])) if i + 1 < len(tr) else np.array([])
        for k in np.nonzero(d <= NEAR_DUP_BITS)[0]:
            union(i, i + 1 + int(k))

    groups: dict[str, int] = {}
    roots: dict[int, int] = {}
    for i, it in enumerate(tr):
        r = find(i)
        groups[it[1]] = roots.setdefault(r, len(roots))
    sizes_g = Counter(groups.values())
    multi = [s for s in sizes_g.values() if s > 1]
    print(f"\n  train groups: {len(sizes_g)} "
          f"({len(multi)} with more than one image, largest {max(sizes_g.values())}, "
          f"{sum(multi)} images in multi-image groups)")

    out = root / "groups.json"
    out.write_text(json.dumps({"rule": f"consecutive numbers within {FRAME_BITS} dHash bits, "
                               f"or any pair within {NEAR_DUP_BITS} bits",
                               "groups": groups}, indent=0))
    report = {
        "counts": {sp: dict(counts[sp]) for sp in SPLITS},
        "filename_overlap": len(name_overlap),
        "ranges": ranges,
        "image_sizes_top5": [[f"{w}x{h}", c] for (w, h), c in sizes.most_common(5)],
        "exact_cross_split_duplicates": exact_cross,
        "exact_duplicate_extra_copies": exact_within,
        "near_dup_test_in_train": len(near_cross),
        "near_dup_by_prefix": dict(near_by_prefix),
        "near_dup_examples": near_cross[:20],
        # Test images with NO near-duplicate in train: the stricter subset
        # train_vision.py reports alongside the official test set.
        "test_without_near_dup": sorted(
            {it[1] for it in te} - {t for t, _, _ in near_cross}),
        "frame_stats": frame_stats,
        "train_groups": len(sizes_g),
        "train_multi_image_groups": len(multi),
        "train_images_in_multi_groups": sum(multi),
        "largest_group": max(sizes_g.values()),
    }
    (root / "audit.json").write_text(json.dumps(report, indent=2))
    print(f"\n  wrote {out} and {root / 'audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
