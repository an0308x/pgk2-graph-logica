#!/usr/bin/env python3
"""Embed the PGK2 high-information DEL pool with frozen SELFormer.

Embeddings are chemical-neighborhood features only.  They are not labels and
do not include ASMS validation/test molecules.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import selfies as sf
import torch
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool",
        type=Path,
        default=ROOT / "artifacts/pgk2_selfies_neighbor_pool/selfies_neighbor_pool.parquet",
    )
    parser.add_argument("--hf-cache", type=Path, default=ROOT / "models/hf-cache")
    parser.add_argument("--model", default="HUBioDataLab/SELFormer")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None, help="Bounded smoke run only.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts/pgk2_selfies_embeddings"
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_length < 1:
        raise ValueError("batch size and max length must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    pool = pl.read_parquet(args.pool)
    if args.limit is not None:
        pool = pool.head(args.limit)
    smiles = pool.get_column("SMILES").to_list()
    selfies: list[str] = []
    keep: list[int] = []
    for index, value in enumerate(smiles):
        try:
            selfies.append(sf.encoder(value))
            keep.append(index)
        except sf.EncoderError:
            continue
    if not selfies:
        raise RuntimeError("No valid SELFIES encodings")
    pool = pool[keep]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=str(args.hf_cache), local_files_only=True
    )
    model = AutoModel.from_pretrained(
        args.model, cache_dir=str(args.hf_cache), local_files_only=True
    ).to(device).eval()
    parts: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(selfies), args.batch_size):
            batch = selfies[offset : offset + args.batch_size]
            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            special = tokens.pop("special_tokens_mask")
            attention = tokens["attention_mask"].bool()
            valid = attention & ~special.bool()
            outputs = model(**{key: value.to(device) for key, value in tokens.items()}).last_hidden_state
            weights = valid.to(device).unsqueeze(-1)
            pooled = (outputs * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            parts.append(pooled.cpu().numpy().astype(np.float32, copy=False))
    embeddings = np.concatenate(parts, axis=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "embeddings.npy", embeddings)
    pool.write_parquet(args.output_dir / "metadata.parquet", compression="zstd")
    report = {
        "model": args.model,
        "frozen": True,
        "pool_rows": pool.height,
        "embedding_shape": list(embeddings.shape),
        "device": device,
        "normalization": "L2-normalized mean of non-special SELFormer token embeddings",
        "metadata": str(args.output_dir / "metadata.parquet"),
        "embeddings": str(args.output_dir / "embeddings.npy"),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
