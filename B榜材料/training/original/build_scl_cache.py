# -*- coding: utf-8 -*-
"""Build the official 18-channel cache used by Official-MultiSpec-V3.

The first 16 channels are copied from the audited official cache.  Channels
17-18 are derived only from the official SCL raster for the same optical
image: normalized SCL class and cloud/shadow indicator.  No labels from the
validation or test split are used.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import tifffile


ROOT = Path(os.environ.get("XINGHE_WORK_ROOT", "."))


def resize_nearest(a: np.ndarray, h: int = 256, w: int = 256) -> np.ndarray:
    iy = np.minimum((np.arange(h) * a.shape[0] // h), a.shape[0] - 1)
    ix = np.minimum((np.arange(w) * a.shape[1] // w), a.shape[1] - 1)
    return a[iy[:, None], ix[None, :]]


def build_split(manifest, split: str, out_root: Path, limit: int = 0) -> tuple[int, int]:
    pairs = manifest[split]
    old_root = ROOT / "code_cache" / ("cache_" + split)
    out_dir = out_root / ("cache_" + split)
    out_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    done = missing = 0
    for pair in pairs:
        key = pair["key"]
        if key in seen:
            continue
        seen.add(key)
        if limit and done >= limit:
            break
        src = old_root / (key + ".npz")
        if not src.is_file():
            raise FileNotFoundError(src)
        opt = Path(pair["opt"])
        scl = opt.parent.parent / "SCL" / opt.name
        if not scl.is_file():
            missing += 1
            print("SCL_MISSING", split, key, scl, flush=True)
            continue
        with np.load(src) as z:
            x = z["x"].astype(np.float32)
            payload = {k: z[k] for k in z.files if k != "x"}
        if x.shape != (16, 256, 256):
            raise ValueError(f"unexpected cache shape {src}: {x.shape}")
        raw = np.asarray(tifffile.imread(scl))
        if raw.ndim != 2:
            raise ValueError(f"unexpected SCL shape {scl}: {raw.shape}")
        raw = resize_nearest(np.nan_to_num(raw, nan=0.0)).astype(np.float32)
        scl_norm = np.clip(raw / 9.0, 0.0, 1.0)
        cloud_shadow = np.isin(raw.astype(np.uint8), [3, 8, 9]).astype(np.float32)
        x18 = np.concatenate([x, scl_norm[None], cloud_shadow[None]], axis=0).astype(np.float16)
        dst = out_dir / (key + ".npz")
        tmp = dst.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, x=x18, **payload)
        os.replace(tmp, dst)
        done += 1
        if done % 250 == 0:
            print("CACHE", split, done, "of", len(pairs), flush=True)
    return done, missing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "code_cache_scl"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "area_qa_patch_v2" / "manifest.json").read_text())
    result = {}
    for split in ("train", "val"):
        result[split] = build_split(manifest, split, out_root, args.limit)
    meta = {
        "source_cache": str(ROOT / "code_cache"),
        "channels": [
            "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A",
            "B11", "B12", "B01", "B09", "VV", "VH", "opt_valid", "sar_valid",
            "scl_norm", "scl_cloud_shadow",
        ],
        "scl_classes": {"cloud_shadow": [3, 8, 9]},
        "result": result,
    }
    (out_root / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print("CACHE_DONE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
