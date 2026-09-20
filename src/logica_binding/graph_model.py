"""Pocket-restricted graph conditioning for LogiCA.

The scalable path uses fixed protein-pocket structures and deterministic RDKit
conformers for each ligand. Every ligand atom may exchange learned messages
with the 26 direct inhibitor-pocket residues, while the broader 49-residue
shell is protein context only. These candidate interaction edges deliberately
carry no intermolecular distance: ligand and protein coordinates are in
independent frames. Pose-derived distance edges remain available as an
optional diagnostic when a true complex prediction exists.
"""

from __future__ import annotations

import json
import hashlib
import shlex
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch import nn

from .model import LogiCAModel, PreparedPairs, _mean_log_probability


AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
ELEMENTS = ("C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B")
POCKET_FEATURE_DIM = len(AA_ORDER) + 4
POCKET_EDGE_DIM = 6
LIGAND_FEATURE_DIM = len(ELEMENTS) + 10
LIGAND_EDGE_DIM = 7
CONTACT_EDGE_DIM = 5


@dataclass(frozen=True)
class PocketGraph:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    coordinates: torch.Tensor
    residue_indices: torch.Tensor
    direct_mask: torch.Tensor
    template: str

    def to(self, device: torch.device | str) -> "PocketGraph":
        return replace(
            self,
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_features=self.edge_features.to(device),
            coordinates=self.coordinates.to(device),
            residue_indices=self.residue_indices.to(device),
            direct_mask=self.direct_mask.to(device),
        )


@dataclass(frozen=True)
class LigandGraph:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    coordinates: torch.Tensor | None
    smiles: str

    def to(self, device: torch.device | str) -> "LigandGraph":
        return replace(
            self,
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_features=self.edge_features.to(device),
            coordinates=None if self.coordinates is None else self.coordinates.to(device),
        )


@dataclass(frozen=True)
class ContactGraph:
    """Bipartite edges whose first row is pocket and second is ligand index."""

    edge_index: torch.Tensor
    edge_features: torch.Tensor

    def to(self, device: torch.device | str) -> "ContactGraph":
        return replace(
            self,
            edge_index=self.edge_index.to(device),
            edge_features=self.edge_features.to(device),
        )


def _parse_pdb_residues(pdb_path: Path, chain: str) -> dict[int, dict[str, object]]:
    residues: dict[int, dict[str, object]] = {}
    for line in pdb_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM") or line[21].strip() != chain:
            continue
        altloc = line[16].strip()
        if altloc not in ("", "A"):
            continue
        try:
            residue_number = int(line[22:26])
            coord = np.asarray(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=np.float32,
            )
            bfactor = float(line[60:66])
        except ValueError:
            continue
        record = residues.setdefault(
            residue_number,
            {"name": line[17:20].strip(), "atoms": [], "ca": None, "bfactors": []},
        )
        record["atoms"].append(coord)
        record["bfactors"].append(bfactor)
        if line[12:16].strip() == "CA":
            record["ca"] = coord
    return residues


def _pocket_graph_from_residues(
    residues: dict[int, dict[str, object]],
    pocket_json: Path,
    template: str,
    spatial_cutoff: float = 10.0,
) -> PocketGraph:
    if spatial_cutoff <= 0:
        raise ValueError("spatial_cutoff must be positive")
    definition = json.loads(pocket_json.read_text(encoding="utf-8"))
    context = tuple(definition["protein_context_shell"]["residue_numbers_1_based"])
    direct = set(definition["direct_ligand_edges"]["residue_numbers_1_based"])
    auxiliary = set(definition["auxiliary_3pg_subsite"]["residue_numbers_1_based"])
    missing = [number for number in context if number not in residues]
    if missing:
        raise ValueError(f"Template {template} is missing pocket residues {missing}")

    coordinates: list[np.ndarray] = []
    features: list[list[float]] = []
    for number in context:
        record = residues[number]
        atom_coordinates = np.stack(record["atoms"])
        coordinate = record["ca"]
        if coordinate is None:
            coordinate = atom_coordinates.mean(axis=0)
        coordinates.append(np.asarray(coordinate, dtype=np.float32))
        one_hot = [float(THREE_TO_ONE.get(str(record["name"]), "") == aa) for aa in AA_ORDER]
        features.append(
            one_hot
            + [
                number / 417.0,
                float(number in direct),
                float(number in auxiliary),
                float(np.mean(record["bfactors"]) / 100.0),
            ]
        )
    xyz = np.stack(coordinates)
    sources: list[int] = []
    destinations: list[int] = []
    edge_features: list[list[float]] = []
    for source in range(len(context)):
        for destination in range(len(context)):
            if source == destination:
                continue
            distance = float(np.linalg.norm(xyz[source] - xyz[destination]))
            sequence_neighbor = abs(context[source] - context[destination]) == 1
            if distance > spatial_cutoff and not sequence_neighbor:
                continue
            sources.append(source)
            destinations.append(destination)
            edge_features.append(
                [
                    distance / spatial_cutoff,
                    float(np.exp(-distance / 2.0)),
                    float(np.exp(-((distance - 4.0) / 2.0) ** 2)),
                    float(np.exp(-((distance - 7.0) / 2.0) ** 2)),
                    float(sequence_neighbor),
                    float(context[source] in direct and context[destination] in direct),
                ]
            )
    return PocketGraph(
        node_features=torch.tensor(features, dtype=torch.float32),
        edge_index=torch.tensor([sources, destinations], dtype=torch.long),
        edge_features=torch.tensor(edge_features, dtype=torch.float32),
        coordinates=torch.tensor(xyz, dtype=torch.float32),
        residue_indices=torch.tensor([number - 1 for number in context], dtype=torch.long),
        direct_mask=torch.tensor([number in direct for number in context], dtype=torch.bool),
        template=template,
    )


def build_pocket_graph(
    pdb_path: Path,
    pocket_json: Path,
    chain: str = "A",
    spatial_cutoff: float = 10.0,
) -> PocketGraph:
    """Build a residue graph for the 49-residue inhibitor context shell."""
    return _pocket_graph_from_residues(
        _parse_pdb_residues(pdb_path, chain),
        pocket_json,
        template=pdb_path.stem,
        spatial_cutoff=spatial_cutoff,
    )


def _parse_boltz_cif(
    cif_path: Path,
    protein_chain: str = "A",
    ligand_chain: str = "B",
) -> tuple[dict[int, dict[str, object]], np.ndarray, list[str]]:
    """Read protein residues and ordered ligand heavy atoms from a Boltz CIF."""
    lines = cif_path.read_text(encoding="utf-8").splitlines()
    try:
        header = lines.index("_atom_site.group_PDB")
    except ValueError as exc:
        raise ValueError(f"No atom_site loop in {cif_path}") from exc
    columns: list[str] = []
    index = header
    while index < len(lines) and lines[index].startswith("_atom_site."):
        columns.append(lines[index].removeprefix("_atom_site."))
        index += 1
    column = {name: position for position, name in enumerate(columns)}
    required = {
        "group_PDB", "type_symbol", "label_atom_id", "label_comp_id",
        "label_seq_id", "label_asym_id", "Cartn_x", "Cartn_y", "Cartn_z",
        "B_iso_or_equiv",
    }
    if not required.issubset(column):
        raise ValueError(f"Unexpected atom_site columns in {cif_path}")
    residues: dict[int, dict[str, object]] = {}
    ligand_coordinates: list[list[float]] = []
    ligand_elements: list[str] = []
    for line in lines[index:]:
        if not line or line == "#" or line.startswith("loop_") or line.startswith("_"):
            break
        fields = shlex.split(line)
        if len(fields) != len(columns):
            continue
        raw_element = fields[column["type_symbol"]]
        element = raw_element[:1].upper() + raw_element[1:].lower()
        if element == "H":
            continue
        coordinate = np.asarray(
            [float(fields[column[key]]) for key in ("Cartn_x", "Cartn_y", "Cartn_z")],
            dtype=np.float32,
        )
        chain = fields[column["label_asym_id"]]
        if chain == protein_chain:
            residue_number = int(fields[column["label_seq_id"]])
            record = residues.setdefault(
                residue_number,
                {
                    "name": fields[column["label_comp_id"]],
                    "atoms": [],
                    "ca": None,
                    "bfactors": [],
                },
            )
            record["atoms"].append(coordinate)
            record["bfactors"].append(float(fields[column["B_iso_or_equiv"]]))
            if fields[column["label_atom_id"]] == "CA":
                record["ca"] = coordinate
        elif chain == ligand_chain:
            ligand_coordinates.append(coordinate.tolist())
            ligand_elements.append(element)
    if not ligand_coordinates:
        raise ValueError(f"No ligand heavy atoms in chain {ligand_chain} of {cif_path}")
    return residues, np.asarray(ligand_coordinates, dtype=np.float32), ligand_elements


def build_boltz_complex_graphs(
    cif_path: Path,
    smiles: str,
    pocket_json: Path,
    contact_cutoff: float = 6.0,
    spatial_cutoff: float = 10.0,
) -> tuple[PocketGraph, LigandGraph, ContactGraph]:
    """Build aligned pocket, ligand, and contact graphs from a Boltz complex.

    Boltz writes ligand atoms in the input molecular atom order.  We verify the
    full heavy-element sequence against RDKit before attaching coordinates, so
    a changed writer or mismatched SMILES fails instead of corrupting contacts.
    """
    residues, coordinates, cif_elements = _parse_boltz_cif(cif_path)
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse {smiles!r}")
    smiles_elements = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    if smiles_elements != cif_elements:
        raise ValueError(
            "Boltz ligand atom order does not match the supplied SMILES: "
            f"SMILES={smiles_elements}, CIF={cif_elements}"
        )
    pocket = _pocket_graph_from_residues(
        residues,
        pocket_json,
        template=cif_path.stem,
        spatial_cutoff=spatial_cutoff,
    )
    ligand = smiles_to_ligand_graph(smiles, coordinates=coordinates)
    return pocket, ligand, build_contact_graph(pocket, ligand, cutoff=contact_cutoff)


def smiles_to_ligand_graph(
    smiles: str,
    coordinates: np.ndarray | torch.Tensor | None = None,
    spatial_cutoff: float | None = None,
) -> LigandGraph:
    """Convert SMILES to an atom graph with optional internal 3-D radius edges."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse {smiles!r}")
    atom_features: list[list[float]] = []
    for atom in molecule.GetAtoms():
        symbol = atom.GetSymbol()
        atom_features.append(
            [float(symbol == value) for value in ELEMENTS]
            + [
                float(symbol not in ELEMENTS),
                atom.GetDegree() / 4.0,
                float(np.clip(atom.GetFormalCharge(), -3, 3) / 3.0),
                float(atom.GetIsAromatic()),
                float(atom.IsInRing()),
                float(atom.GetHybridization() == Chem.HybridizationType.SP),
                float(atom.GetHybridization() == Chem.HybridizationType.SP2),
                float(atom.GetHybridization() == Chem.HybridizationType.SP3),
                float(atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW),
                float(atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CCW),
            ]
        )
    xyz: torch.Tensor | None = None
    if coordinates is not None:
        xyz = torch.as_tensor(coordinates, dtype=torch.float32)
        if xyz.shape != (molecule.GetNumAtoms(), 3):
            raise ValueError(
                f"coordinates must have shape ({molecule.GetNumAtoms()}, 3), got {tuple(xyz.shape)}"
            )
    sources: list[int] = []
    destinations: list[int] = []
    bond_features: list[list[float]] = []
    bond_types = (
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
        Chem.BondType.AROMATIC,
    )
    for bond in molecule.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        length = 0.0 if xyz is None else float(torch.linalg.vector_norm(xyz[begin] - xyz[end]))
        feature = [float(bond.GetBondType() == kind) for kind in bond_types] + [
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
            length / 3.0,
        ]
        sources.extend((begin, end))
        destinations.extend((end, begin))
        bond_features.extend((feature, feature))
    if spatial_cutoff is not None:
        if xyz is None:
            raise ValueError("spatial_cutoff requires ligand coordinates")
        if spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")
        bonded = {
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in molecule.GetBonds()
        }
        bonded |= {(destination, source) for source, destination in bonded}
        for source in range(molecule.GetNumAtoms()):
            for destination in range(molecule.GetNumAtoms()):
                if source == destination or (source, destination) in bonded:
                    continue
                length = float(torch.linalg.vector_norm(xyz[source] - xyz[destination]))
                if length <= spatial_cutoff:
                    sources.append(source)
                    destinations.append(destination)
                    # Zero bond flags identify a non-covalent intramolecular
                    # radius edge; its final component is the internal distance.
                    bond_features.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, length / 3.0])
    return LigandGraph(
        node_features=torch.tensor(atom_features, dtype=torch.float32),
        edge_index=torch.tensor([sources, destinations], dtype=torch.long),
        edge_features=torch.tensor(bond_features, dtype=torch.float32).reshape(-1, LIGAND_EDGE_DIM),
        coordinates=xyz,
        smiles=smiles,
    )


def smiles_to_rdkit_3d_graph_with_quality(
    smiles: str,
    seed: int = 2026,
    spatial_cutoff: float = 4.5,
    optimization_max_iters: int = 200,
    warn_on_fallback: bool = True,
) -> tuple[LigandGraph, str]:
    """Generate a deterministic ligand graph and report its geometry route.

    Coordinates describe internal ligand geometry only. They are never used as
    distances to PGK2 unless an independently aligned complex pose is supplied.
    Rare molecules that cannot be embedded after the normal and random-coordinate
    ETKDG attempts use deterministic 2-D coordinates instead of aborting a panel.
    """
    if optimization_max_iters < 0:
        raise ValueError("optimization_max_iters must be non-negative")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse {smiles!r}")
    embedded = Chem.AddHs(molecule)
    digest = hashlib.blake2b(f"{seed}:{smiles}".encode(), digest_size=4).digest()
    random_seed = int.from_bytes(digest, "little") % 2_147_483_647 or 1
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = random_seed
    status = AllChem.EmbedMolecule(embedded, parameters)
    geometry_quality = "etkdg"
    if status != 0:
        parameters.useRandomCoords = True
        status = AllChem.EmbedMolecule(embedded, parameters)
        geometry_quality = "etkdg_random_coords"
    embedded_3d = status == 0
    if not embedded_3d:
        if warn_on_fallback:
            warnings.warn(
                f"RDKit ETKDG could not embed {smiles!r}; using deterministic 2-D coordinates",
                RuntimeWarning,
                stacklevel=2,
            )
        AllChem.Compute2DCoords(embedded)
        geometry_quality = "2d_fallback"
    elif optimization_max_iters == 0:
        geometry_quality += "_unoptimized"
    elif AllChem.MMFFHasAllMoleculeParams(embedded):
        convergence = AllChem.MMFFOptimizeMolecule(
            embedded, confId=0, maxIters=optimization_max_iters
        )
        geometry_quality += "_mmff_converged" if convergence == 0 else "_mmff_limited"
    elif AllChem.UFFHasAllMoleculeParams(embedded):
        convergence = AllChem.UFFOptimizeMolecule(
            embedded, confId=0, maxIters=optimization_max_iters
        )
        geometry_quality += "_uff_converged" if convergence == 0 else "_uff_limited"
    else:
        geometry_quality += "_no_forcefield"
    heavy = Chem.RemoveHs(embedded)
    conformer = heavy.GetConformer()
    coordinates = np.asarray(
        [list(conformer.GetAtomPosition(index)) for index in range(heavy.GetNumAtoms())],
        dtype=np.float32,
    )
    return (
        smiles_to_ligand_graph(
            smiles,
            coordinates=coordinates,
            spatial_cutoff=spatial_cutoff,
        ),
        geometry_quality,
    )


def smiles_to_rdkit_3d_graph(
    smiles: str,
    seed: int = 2026,
    spatial_cutoff: float = 4.5,
) -> LigandGraph:
    """Generate the historical deterministic RDKit graph used by Graph LogiCA."""
    graph, _ = smiles_to_rdkit_3d_graph_with_quality(
        smiles,
        seed=seed,
        spatial_cutoff=spatial_cutoff,
        optimization_max_iters=200,
        warn_on_fallback=True,
    )
    return graph


def build_candidate_interaction_graph(
    pocket: PocketGraph,
    ligand: LigandGraph,
) -> ContactGraph:
    """Connect every ligand atom to every allowed direct-pocket residue.

    The cross edges encode *possible* interactions. Their first four features
    are zero because no protein--ligand distance is known; the final feature
    marks the edge as a learned candidate interaction rather than a pose edge.
    """
    pocket_indices = torch.nonzero(pocket.direct_mask, as_tuple=False).flatten()
    device = pocket.direct_mask.device
    ligand_indices = torch.arange(
        ligand.node_features.shape[0], dtype=torch.long, device=device
    )
    pairs = torch.cartesian_prod(pocket_indices, ligand_indices)
    if pairs.ndim == 1:
        pairs = pairs.reshape(1, 2)
    features = torch.zeros(
        (pairs.shape[0], CONTACT_EDGE_DIM), dtype=torch.float32, device=device
    )
    features[:, -1] = 1.0
    return ContactGraph(edge_index=pairs.T.contiguous(), edge_features=features)


def build_contact_graph(
    pocket: PocketGraph,
    ligand: LigandGraph,
    cutoff: float = 6.0,
) -> ContactGraph:
    """Construct allowed pose contacts to the direct inhibitor-pocket nodes."""
    if ligand.coordinates is None:
        raise ValueError(
            "Protein-ligand contact edges require ligand coordinates in the protein "
            "template frame; a SMILES graph alone is insufficient."
        )
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    distances = torch.cdist(pocket.coordinates, ligand.coordinates)
    allowed = (distances <= cutoff) & pocket.direct_mask[:, None]
    pocket_index, ligand_index = torch.nonzero(allowed, as_tuple=True)
    edge_distance = distances[pocket_index, ligand_index]
    feature = torch.stack(
        [
            edge_distance / cutoff,
            torch.exp(-edge_distance / 2.0),
            torch.exp(-((edge_distance - 3.0) / 1.5) ** 2),
            torch.ones_like(edge_distance),
            torch.zeros_like(edge_distance),
        ],
        dim=-1,
    )
    return ContactGraph(
        edge_index=torch.stack([pocket_index, ligand_index], dim=0),
        edge_features=feature,
    )


class EdgeMessageLayer(nn.Module):
    def __init__(self, width: int, edge_dim: int) -> None:
        super().__init__()
        self.message = nn.Sequential(nn.Linear(width + edge_dim, width), nn.GELU())
        self.update = nn.Sequential(nn.Linear(width * 2, width), nn.GELU())
        self.norm = nn.LayerNorm(width)

    def forward(
        self, nodes: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor
    ) -> torch.Tensor:
        if edge_index.shape[1] == 0:
            return nodes
        source, destination = edge_index
        message = self.message(torch.cat([nodes[source], edge_features], dim=-1))
        aggregate = torch.zeros_like(nodes)
        aggregate.index_add_(0, destination, message)
        degree = torch.bincount(destination, minlength=nodes.shape[0]).to(nodes.dtype).clamp_min(1)
        aggregate = aggregate / degree[:, None]
        return self.norm(nodes + self.update(torch.cat([nodes, aggregate], dim=-1)))


class GraphEncoder(nn.Module):
    def __init__(self, input_dim: int, edge_dim: int, width: int, layers: int = 2) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, width)
        self.layers = nn.ModuleList([EdgeMessageLayer(width, edge_dim) for _ in range(layers)])

    def forward(
        self, nodes: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.input(nodes)
        for layer in self.layers:
            hidden = layer(hidden, edge_index, edge_features)
        return hidden


class GraphLogiCAConditioner(nn.Module):
    """Inject local residue/atom messages into LogiCA token streams.

    Protein graph updates can only touch the ESM tokens mapped from the pocket
    residue indices. Ligand token queries attend to atom nodes rather than a
    globally pooled molecular vector. Cross edges may be learned candidate
    interactions or optional pose-derived contacts.
    """

    def __init__(
        self,
        protein_width: int,
        drug_width: int,
        graph_width: int = 128,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.pocket_encoder = GraphEncoder(POCKET_FEATURE_DIM, POCKET_EDGE_DIM, graph_width)
        self.ligand_encoder = GraphEncoder(LIGAND_FEATURE_DIM, LIGAND_EDGE_DIM, graph_width)
        # The structure graph is local, but each residue receives its contextual
        # embedding from ESM run over the complete 417-residue PGK2 sequence.
        self.protein_context_in = nn.Linear(protein_width, graph_width, bias=False)
        self.protein_context_norm = nn.LayerNorm(graph_width)
        self.ligand_to_pocket = nn.Sequential(
            nn.Linear(graph_width * 2 + CONTACT_EDGE_DIM, graph_width), nn.GELU()
        )
        self.pocket_to_ligand = nn.Sequential(
            nn.Linear(graph_width * 2 + CONTACT_EDGE_DIM, graph_width), nn.GELU()
        )
        self.pocket_norm = nn.LayerNorm(graph_width)
        self.ligand_norm = nn.LayerNorm(graph_width)
        self.protein_out = nn.Linear(graph_width, protein_width, bias=False)
        self.drug_query = nn.Linear(drug_width, graph_width, bias=False)
        self.drug_attention = nn.MultiheadAttention(graph_width, heads, batch_first=True)
        self.drug_out = nn.Linear(graph_width, drug_width, bias=False)
        # Less saturated than the released LogiCA residual gates: this branch is
        # new and must receive useful gradients from a small PGK2 pair set.
        self.protein_graph_gate = nn.Parameter(torch.tensor([-2.0]))
        self.drug_graph_gate = nn.Parameter(torch.tensor([-2.0]))

    def _contact_update(
        self,
        pocket_hidden: torch.Tensor,
        ligand_hidden: torch.Tensor,
        contacts: ContactGraph,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if contacts.edge_index.shape[1] == 0:
            return pocket_hidden, ligand_hidden
        pocket_index, ligand_index = contacts.edge_index
        to_pocket = self.ligand_to_pocket(
            torch.cat(
                [
                    ligand_hidden[ligand_index],
                    pocket_hidden[pocket_index],
                    contacts.edge_features,
                ],
                dim=-1,
            )
        )
        to_ligand = self.pocket_to_ligand(
            torch.cat(
                [
                    pocket_hidden[pocket_index],
                    ligand_hidden[ligand_index],
                    contacts.edge_features,
                ],
                dim=-1,
            )
        )
        pocket_aggregate = torch.zeros_like(pocket_hidden)
        ligand_aggregate = torch.zeros_like(ligand_hidden)
        pocket_aggregate.index_add_(0, pocket_index, to_pocket)
        ligand_aggregate.index_add_(0, ligand_index, to_ligand)
        return (
            self.pocket_norm(pocket_hidden + pocket_aggregate),
            self.ligand_norm(ligand_hidden + ligand_aggregate),
        )

    def forward(
        self,
        protein_hidden: torch.Tensor,
        drug_hidden: torch.Tensor,
        pocket_graph: PocketGraph,
        ligand_graphs: Sequence[LigandGraph],
        contact_graphs: Sequence[ContactGraph] | None,
        require_contacts: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = protein_hidden.shape[0]
        if len(ligand_graphs) != batch_size:
            raise ValueError("One ligand graph is required per batch row")
        if require_contacts and contact_graphs is None:
            raise ValueError("Graph LogiCA requires pocket-ligand interaction graphs")
        if contact_graphs is not None and len(contact_graphs) != batch_size:
            raise ValueError("One contact graph is required per batch row")

        pocket_base = self.pocket_encoder(
            pocket_graph.node_features, pocket_graph.edge_index, pocket_graph.edge_features
        )
        protein_result = protein_hidden.clone()
        drug_result = drug_hidden.clone()
        token_positions = pocket_graph.residue_indices + 1  # ESM leading <cls>
        if int(token_positions.max()) >= protein_hidden.shape[1]:
            raise ValueError("Pocket residue token lies outside the protein hidden sequence")
        for row, ligand_graph in enumerate(ligand_graphs):
            ligand_nodes = self.ligand_encoder(
                ligand_graph.node_features,
                ligand_graph.edge_index,
                ligand_graph.edge_features,
            )
            pocket_nodes = self.protein_context_norm(
                pocket_base
                + self.protein_context_in(protein_hidden[row, token_positions])
            )
            if contact_graphs is not None:
                pocket_nodes, ligand_nodes = self._contact_update(
                    pocket_nodes, ligand_nodes, contact_graphs[row]
                )
            protein_result[row, token_positions] = (
                protein_result[row, token_positions]
                + torch.sigmoid(self.protein_graph_gate) * self.protein_out(pocket_nodes)
            )
            attended, _ = self.drug_attention(
                self.drug_query(drug_hidden[row : row + 1]),
                ligand_nodes[None, :, :],
                ligand_nodes[None, :, :],
                need_weights=False,
            )
            drug_result[row] = (
                drug_result[row]
                + torch.sigmoid(self.drug_graph_gate) * self.drug_out(attended[0])
            )
        return protein_result, drug_result


class GraphLogiCAModel(nn.Module):
    """Compose the official LogiCA backbone with pocket graph conditioning."""

    def __init__(self, base: LogiCAModel, graph_width: int = 128, graph_conditioning: bool = True) -> None:
        super().__init__()
        self.base = base
        self.graph_conditioning = graph_conditioning
        self.conditioner = GraphLogiCAConditioner(
            protein_width=int(base.esm.config.hidden_size),
            drug_width=int(base.drug_encoder.config.hidden_size),
            graph_width=graph_width,
        )

    def forward_from_hidden_with_graph(
        self,
        protein_hidden: torch.Tensor,
        protein_attention_mask: torch.Tensor,
        drug_hidden: torch.Tensor,
        drug_attention_mask: torch.Tensor,
        pocket_graph: PocketGraph,
        ligand_graphs: Sequence[LigandGraph],
        contact_graphs: Sequence[ContactGraph] | None,
        require_contacts: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.graph_conditioning:
            protein_conditioned, drug_conditioned = self.conditioner(
                protein_hidden,
                drug_hidden,
                pocket_graph,
                ligand_graphs,
                contact_graphs,
                require_contacts=require_contacts,
            )
        else:
            protein_conditioned, drug_conditioned = protein_hidden, drug_hidden
        interface = torch.zeros_like(protein_attention_mask, dtype=torch.bool)
        positions = pocket_graph.residue_indices[pocket_graph.direct_mask] + 1
        interface[:, positions] = True
        return self.base.forward_from_hidden(
            protein_conditioned,
            protein_attention_mask,
            drug_conditioned,
            drug_attention_mask,
            protein_interface_mask=interface,
        )


def score_graph_prepared(
    model: GraphLogiCAModel,
    prepared: PreparedPairs,
    indices: torch.Tensor,
    pocket_graphs: Sequence[PocketGraph],
    ligand_graphs: Sequence[LigandGraph],
) -> torch.Tensor:
    """Score selected pairs using an ensemble of fixed PGK2 pocket graphs."""
    if not pocket_graphs:
        raise ValueError("At least one pocket graph is required")
    device = next(model.parameters()).device
    cpu_indices = indices.to(dtype=torch.long, device="cpu")
    batch_size = int(cpu_indices.numel())
    protein_hidden = prepared.protein_hidden.expand(batch_size, -1, -1).to(device)
    protein_attention = prepared.protein_attention_mask.expand(batch_size, -1).to(device)
    protein_original = prepared.protein_original_ids.expand(batch_size, -1).to(device)
    protein_selected = prepared.protein_selected.expand(batch_size, -1).to(device)
    drug_hidden = prepared.drug_hidden[cpu_indices].to(device)
    drug_attention = prepared.drug_attention_mask[cpu_indices].to(device)
    drug_original = prepared.drug_original_ids[cpu_indices].to(device)
    drug_selected = prepared.drug_selected[cpu_indices].to(device)
    selected_ligands = [ligand_graphs[int(index)].to(device) for index in cpu_indices]

    template_scores: list[torch.Tensor] = []
    for pocket_graph in pocket_graphs:
        pocket = pocket_graph.to(device)
        interactions = [
            build_candidate_interaction_graph(pocket, ligand) for ligand in selected_ligands
        ]
        protein_logits, drug_logits, _, _ = model.forward_from_hidden_with_graph(
            protein_hidden,
            protein_attention,
            drug_hidden,
            drug_attention,
            pocket,
            selected_ligands,
            interactions,
            require_contacts=True,
        )
        protein_score = _mean_log_probability(
            protein_logits, protein_original, protein_selected
        )
        drug_score = _mean_log_probability(drug_logits, drug_original, drug_selected)
        alpha = model.base.pair_alpha()
        template_scores.append(alpha * protein_score + (1 - alpha) * drug_score)
    return torch.stack(template_scores, dim=0).mean(dim=0)
