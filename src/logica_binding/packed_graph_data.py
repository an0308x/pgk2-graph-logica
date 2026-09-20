"""Packed graph I/O without importing evaluation or scientific-metric scripts."""
from pathlib import Path
import numpy as np
import torch
from .full_graph_model import PackedLigandBatch


def load_archive(path: Path, need_graph: bool, need_fp: bool) -> dict[str, np.ndarray]:
    keys = ["fingerprints"] if need_fp else []
    if need_graph:
        keys += ["node_features", "edge_index", "edge_features", "node_offsets", "edge_offsets"]
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in keys}


def build_packed_batch(archive, indices, need_graph=True, need_fp=False):
    fingerprints = None
    if need_fp:
        bits = np.unpackbits(archive["fingerprints"][indices], axis=1, bitorder="little")
        fingerprints = torch.from_numpy(bits.astype(np.float32))
    if not need_graph:
        return PackedLigandBatch(torch.empty((0, 20)), torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, 7)), torch.zeros(len(indices)+1, dtype=torch.long), fingerprints)
    if len(indices) == 0:
        raise ValueError("Cannot build an empty ligand batch")
    nodes, edges, features, offsets = [], [], [], [0]
    for index in indices:
        ns, ne = archive["node_offsets"][index:index+2]
        es, ee = archive["edge_offsets"][index:index+2]
        if ne <= ns:
            raise ValueError("Empty or failed molecular graph")
        nodes.append(archive["node_features"][ns:ne])
        edges.append(archive["edge_index"][:, es:ee] + offsets[-1])
        features.append(archive["edge_features"][es:ee])
        offsets.append(offsets[-1] + int(ne-ns))
    return PackedLigandBatch(torch.from_numpy(np.concatenate(nodes)),
        torch.from_numpy(np.concatenate(edges, axis=1)).long(),
        torch.from_numpy(np.concatenate(features)), torch.tensor(offsets, dtype=torch.long), fingerprints)
