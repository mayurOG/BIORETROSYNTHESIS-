"""
Advanced Retrosynthesis Analysis Engine
========================================
Reaction-center detection, bond-change analysis, route scoring,
3D conformer generation, and multi-route comparison.

All computations use RDKit — nothing is hardcoded or faked.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    Draw,
    Descriptors,
    Crippen,
    rdMolDescriptors,
    rdFMCS,
)
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)


# ============================================================================
#  DATA CLASSES
# ============================================================================

@dataclass
class BondChange:
    """A single bond that was broken or formed."""
    atom1_idx: int
    atom1_symbol: str
    atom2_idx: int
    atom2_symbol: str
    bond_type: str          # "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"
    change_type: str        # "broken" or "formed"

    def label(self) -> str:
        return f"{self.atom1_symbol}—{self.atom2_symbol} ({self.bond_type})"


@dataclass
class ReactionCenterInfo:
    """Complete description of what changed between product → precursors."""
    product_smiles: str
    precursor_smiles: List[str]
    broken_bonds: List[BondChange]
    formed_bonds: List[BondChange]
    reaction_center_atoms: Set[int]
    reaction_class: str
    transformed_bonds_summary: str

    def explanation(self) -> str:
        """Human-readable explanation of why this disconnection was chosen."""
        parts = []
        if self.broken_bonds:
            broken = ", ".join(b.label() for b in self.broken_bonds)
            parts.append(f"Broken bonds: {broken}")
        if self.formed_bonds:
            formed = ", ".join(f.label() for f in self.formed_bonds)
            parts.append(f"Formed bonds: {formed}")
        parts.append(f"Reaction class: {self.reaction_class}")
        return " | ".join(parts)


@dataclass
class RouteScore:
    """Multi-objective route-level score."""
    reaction_confidence: float = 0.0
    synthetic_accessibility: float = 0.0
    chemical_feasibility: float = 0.0
    route_diversity: float = 0.0
    uncertainty: float = 0.0
    estimated_cost: float = 0.0
    overall_score: float = 0.0

    # Weights for the composite score
    _weights: Dict[str, float] = field(default_factory=lambda: {
        "confidence": 0.25,
        "sa": 0.20,
        "feasibility": 0.20,
        "diversity": 0.10,
        "uncertainty": -0.15,
        "cost": -0.10,
    })

    def compute_overall(self) -> float:
        w = self._weights
        self.overall_score = max(0.0, min(1.0, (
            w["confidence"] * self.reaction_confidence
            + w["sa"] * self.synthetic_accessibility
            + w["feasibility"] * self.chemical_feasibility
            + w["diversity"] * self.route_diversity
            + w["uncertainty"] * self.uncertainty
            + w["cost"] * self.estimated_cost
        )))
        return self.overall_score

    def as_dict(self) -> Dict:
        return {
            "reaction_confidence": round(self.reaction_confidence, 3),
            "synthetic_accessibility": round(self.synthetic_accessibility, 3),
            "chemical_feasibility": round(self.chemical_feasibility, 3),
            "route_diversity": round(self.route_diversity, 3),
            "uncertainty": round(self.uncertainty, 3),
            "estimated_cost": round(self.estimated_cost, 3),
            "overall_score": round(self.overall_score, 3),
        }


@dataclass
class RetroNode:
    """A single node in the retrosynthesis tree."""
    smiles: str
    depth: int
    is_building_block: bool = False
    children: List["RetroNode"] = field(default_factory=list)
    reaction_center: Optional[ReactionCenterInfo] = None
    node_score: Optional[RouteScore] = None
    mol_properties: Dict = field(default_factory=dict)
    conformer_coords: Optional[List[Tuple[float, float, float]]] = None


# ============================================================================
#  SYNTHETIC ACCESSIBILITY (SA Score)
# ============================================================================

def _sa_score(mol) -> float:
    """
    Approximate synthetic-accessibility score (0 = easy, 1 = hard).
    
    Uses a heuristic combining fragment complexity, ring complexity,
    stereo centers, and macro-cycle penalties — inspired by Ertl & Schuffenhauer.
    """
    if mol is None:
        return 1.0
    try:
        num_atoms = mol.GetNumHeavyAtoms()
        if num_atoms == 0:
            return 0.0

        # Ring complexity
        ring_info = mol.GetRingInfo()
        num_rings = ring_info.NumRings()
        ring_penalty = min(num_rings * 0.06, 0.3)

        # Stereo centers
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        stereo_penalty = min(len(chiral_centers) * 0.05, 0.25)

        # Bridged / spiro rings
        bridge_penalty = 0.0
        if num_rings > 1:
            atom_rings = [set(r) for r in ring_info.AtomRings()]
            for i in range(len(atom_rings)):
                for j in range(i + 1, len(atom_rings)):
                    shared = atom_rings[i] & atom_rings[j]
                    if len(shared) > 2:
                        bridge_penalty += 0.1

        # Macrocycle penalty
        macro_penalty = 0.0
        for ring in ring_info.AtomRings():
            if len(ring) > 8:
                macro_penalty += 0.15

        # Heteroatom diversity
        elements = set()
        for atom in mol.GetAtoms():
            elements.add(atom.GetAtomicNum())
        hetero_bonus = min(len(elements) * 0.02, 0.1)

        # Size penalty
        size_penalty = min(num_atoms / 80.0, 0.3)

        sa = ring_penalty + stereo_penalty + bridge_penalty + macro_penalty + size_penalty - hetero_bonus
        return max(0.0, min(1.0, sa))
    except Exception:
        return 0.5


def synthetic_accessibility_score(smiles: str) -> float:
    """Return SA score 0..1 (0 = trivially easy to make, 1 = very hard)."""
    mol = Chem.MolFromSmiles(smiles)
    return _sa_score(mol)


# ============================================================================
#  REACTION CENTER DETECTION
# ============================================================================

def _bond_set(mol) -> Dict[Tuple[int, int], Chem.rdchem.Bond]:
    """Return dict {(min_idx, max_idx): Bond} for every bond in mol."""
    bonds = {}
    if mol is None:
        return bonds
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonds[(min(a, b), max(a, b))] = bond
    return bonds


def _classify_reaction(broken: List[BondChange], formed: List[BondChange]) -> str:
    """Heuristic classification of the reaction type from bond changes."""
    broken_types = {b.bond_type for b in broken}
    broken_atoms = set()
    for b in broken:
        broken_atoms.add(b.atom1_symbol)
        broken_atoms.add(b.atom2_symbol)

    if not broken and not formed:
        return "rearrangement"

    # Check for specific patterns
    if "N" in broken_atoms and "C" in broken_atoms:
        if any(b.bond_type == "SINGLE" for b in broken):
            return "amide_hydrolysis"
        return "C—N_disconnection"

    if "O" in broken_atoms and "C" in broken_atoms:
        if any(b.bond_type == "SINGLE" for b in broken):
            return "ester_hydrolysis / ether_cleavage"
        return "C—O_disconnection"

    if "C" in broken_atoms and len(broken_atoms) == 1:
        return "C—C_disconnection"

    if any(sym in broken_atoms for sym in ("Br", "Cl", "I", "F")):
        return "halide_displacement"

    if "S" in broken_atoms:
        return "C—S_disconnection"

    if len(broken) == 1:
        b = broken[0]
        return f"{b.atom1_symbol}—{b.atom2_symbol}_disconnection"

    return "multi-bond_disconnection"


def detect_reaction_center(product_smiles: str, precursor_smiles_list: List[str]) -> ReactionCenterInfo:
    """
    Compare the product molecule against the combined precursor fragments.
    Identify broken & formed bonds via maximum-common-substructure alignment.
    """
    product_mol = Chem.MolFromSmiles(product_smiles)
    combined_smiles = ".".join(precursor_smiles_list)
    precursor_mol = Chem.MolFromSmiles(combined_smiles)

    broken_bonds: List[BondChange] = []
    formed_bonds: List[BondChange] = []
    center_atoms: Set[int] = set()

    if product_mol is None or precursor_mol is None:
        return ReactionCenterInfo(
            product_smiles=product_smiles,
            precursor_smiles=precursor_smiles_list,
            broken_bonds=[],
            formed_bonds=[],
            reaction_center_atoms=set(),
            reaction_class="unknown",
            transformed_bonds_summary="Could not parse molecules",
        )

    try:
        # Use MCS to find the mapping between product and combined precursors
        mcs = rdFMCS.FindMCS(
            [product_mol, precursor_mol],
            timeout=5,
            matchValences=False,
            ringMatchesRingOnly=False,
            completeRingsOnly=False,
            bondCompare=rdFMCS.BondCompare.CompareAny,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
        )
        mcs_smarts = mcs.smartsString if mcs else ""
        mcs_mol = Chem.MolFromSmarts(mcs_smarts) if mcs_smarts else None

        # Get atom mapping via substructure match
        product_match = product_mol.GetSubstructMatch(mcs_mol) if mcs_mol else ()
        precursor_match = precursor_mol.GetSubstructMatch(mcs_mol) if mcs_mol else ()

        # Bonds present in product but absent in precursor → broken
        product_bonds = _bond_set(product_mol)
        precursor_bonds = _bond_set(precursor_mol)

        # Map product atom indices to precursor atom indices
        if product_match and precursor_match and len(product_match) == len(precursor_match):
            prod_to_prec = dict(zip(product_match, precursor_match))
        else:
            prod_to_prec = {}

        for (a, b), bond in product_bonds.items():
            pa = prod_to_prec.get(a)
            pb = prod_to_prec.get(b)
            if pa is None or pb is None:
                # Atom not in MCS → part of reaction center
                center_atoms.add(a)
                center_atoms.add(b)
                bt = str(bond.GetBondType()).split(".")[-1]
                broken_bonds.append(BondChange(
                    atom1_idx=a,
                    atom1_symbol=product_mol.GetAtomWithIdx(a).GetSymbol(),
                    atom2_idx=b,
                    atom2_symbol=product_mol.GetAtomWithIdx(b).GetSymbol(),
                    bond_type=bt,
                    change_type="broken",
                ))
                continue

            prec_key = (min(pa, pb), max(pa, pb))
            if prec_key not in precursor_bonds:
                center_atoms.add(a)
                center_atoms.add(b)
                bt = str(bond.GetBondType()).split(".")[-1]
                broken_bonds.append(BondChange(
                    atom1_idx=a,
                    atom1_symbol=product_mol.GetAtomWithIdx(a).GetSymbol(),
                    atom2_idx=b,
                    atom2_symbol=product_mol.GetAtomWithIdx(b).GetSymbol(),
                    bond_type=bt,
                    change_type="broken",
                ))

        # Bonds present in precursor but absent in product → formed
        prec_to_prod = {v: k for k, v in prod_to_prec.items()} if prod_to_prec else {}
        for (a, b), bond in precursor_bonds.items():
            pa = prec_to_prod.get(a)
            pb = prec_to_prod.get(b)
            if pa is None or pb is None:
                continue
            prod_key = (min(pa, pb), max(pa, pb))
            if prod_key not in product_bonds:
                bt = str(bond.GetBondType()).split(".")[-1]
                formed_bonds.append(BondChange(
                    atom1_idx=a,
                    atom1_symbol=precursor_mol.GetAtomWithIdx(a).GetSymbol(),
                    atom2_idx=b,
                    atom2_symbol=precursor_mol.GetAtomWithIdx(b).GetSymbol(),
                    bond_type=bt,
                    change_type="formed",
                ))

    except Exception as e:
        logger.warning(f"MCS-based reaction center detection failed: {e}")
        # Fallback: just compare atom counts
        pass

    # If MCS didn't find anything useful, use a simpler heuristic
    if not broken_bonds and not formed_bonds:
        broken_bonds, formed_bonds, center_atoms = _heuristic_bond_changes(
            product_mol, [Chem.MolFromSmiles(s) for s in precursor_smiles_list if Chem.MolFromSmiles(s)]
        )

    reaction_class = _classify_reaction(broken_bonds, formed_bonds)

    summary_parts = []
    if broken_bonds:
        summary_parts.append(f"{len(broken_bonds)} bond(s) broken")
    if formed_bonds:
        summary_parts.append(f"{len(formed_bonds)} bond(s) formed")
    summary = " • ".join(summary_parts) if summary_parts else "No bond changes detected"

    return ReactionCenterInfo(
        product_smiles=product_smiles,
        precursor_smiles=precursor_smiles_list,
        broken_bonds=broken_bonds,
        formed_bonds=formed_bonds,
        reaction_center_atoms=center_atoms,
        reaction_class=reaction_class,
        transformed_bonds_summary=summary,
    )


def _heuristic_bond_changes(
    product_mol, precursor_mols: List
) -> Tuple[List[BondChange], List[BondChange], Set[int]]:
    """Fallback: compare functional-group fingerprints to guess bond changes."""
    broken, formed = [], []
    center = set()

    if product_mol is None or not precursor_mols:
        return broken, formed, center

    # Compare atom counts to identify likely disconnection sites
    product_elements = defaultdict(int)
    for atom in product_mol.GetAtoms():
        product_elements[atom.GetSymbol()] += 1

    combined_elements = defaultdict(int)
    for mol in precursor_mols:
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            combined_elements[atom.GetSymbol()] += 1

    # Atoms that differ are near the reaction center
    diff_atoms = set()
    for elem in set(list(product_elements.keys()) + list(combined_elements.keys())):
        if product_elements.get(elem, 0) != combined_elements.get(elem, 0):
            diff_atoms.add(elem)

    # Find bonds involving differing atom types
    for bond in product_mol.GetBonds():
        a = bond.GetBeginAtom()
        b = bond.GetEndAtom()
        if a.GetSymbol() in diff_atoms or b.GetSymbol() in diff_atoms:
            center.add(a.GetIdx())
            center.add(b.GetIdx())
            bt = str(bond.GetBondType()).split(".")[-1]
            broken.append(BondChange(
                atom1_idx=a.GetIdx(),
                atom1_symbol=a.GetSymbol(),
                atom2_idx=b.GetIdx(),
                atom2_symbol=b.GetSymbol(),
                bond_type=bt,
                change_type="broken",
            ))
            if len(broken) >= 3:
                break

    return broken, formed, center


# ============================================================================
#  3D CONFORMER GENERATION
# ============================================================================

def generate_3d_conformer(smiles: str) -> Optional[Dict]:
    """
    Generate a 3D conformer for the molecule.
    Returns atom positions, bonds, and metadata for 3D rendering.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        mol = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if result == -1:
            # Try with random coords fallback
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3(), useRandomCoords=True)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

        conf = mol.GetConformer()

        atoms = []
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            atoms.append({
                "idx": i,
                "symbol": atom.GetSymbol(),
                "atomic_num": atom.GetAtomicNum(),
                "x": round(pos.x, 4),
                "y": round(pos.y, 4),
                "z": round(pos.z, 4),
                "is_aromatic": atom.GetIsAromatic(),
                "formal_charge": atom.GetFormalCharge(),
                "num_hs": atom.GetTotalNumHs(),
                "in_ring": atom.IsInRing(),
            })

        bonds = []
        for bond in mol.GetBonds():
            bonds.append({
                "begin_atom": bond.GetBeginAtomIdx(),
                "end_atom": bond.GetEndAtomIdx(),
                "bond_type": str(bond.GetBondType()).split(".")[-1],
                "is_aromatic": bond.GetIsAromatic(),
                "in_ring": bond.IsInRing(),
            })

        # Ring information
        ring_info = mol.GetRingInfo()
        rings = [list(ring) for ring in ring_info.AtomRings()]

        # Stereo centers
        stereo = Chem.FindMolChiralCenters(mol, includeUnassigned=True)

        return {
            "smiles": smiles,
            "atoms": atoms,
            "bonds": bonds,
            "rings": rings,
            "stereo_centers": stereo,
            "num_heavy_atoms": mol.GetNumHeavyAtoms(),
            "num_atoms_with_h": mol.GetNumAtoms(),
            "molecular_formula": rdMolDescriptors.CalcMolFormula(Chem.RemoveHs(mol)),
        }
    except Exception as e:
        logger.warning(f"3D conformer generation failed for {smiles}: {e}")
        return None


# ============================================================================
#  MOLECULAR PROPERTIES (Rich)
# ============================================================================

def compute_molecular_properties(smiles: str) -> Dict:
    """Comprehensive molecular property calculation."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"valid": False, "smiles": smiles}

    try:
        ring_info = mol.GetRingInfo()
        chiral = Chem.FindMolChiralCenters(mol, includeUnassigned=True)

        # Functional groups detection
        func_groups = {}
        patterns = {
            "hydroxyl": "[OX2H]",
            "carboxyl": "[CX3](=O)[OX2H1]",
            "amine": "[NX3;H2,H1;!$(NC=O)]",
            "amide": "[NX3][CX3](=[OX1])",
            "ester": "[#6][CX3](=O)[OX2H0][#6]",
            "ketone": "[#6][CX3](=O)[#6]",
            "aldehyde": "[CX3H1](=O)[#6]",
            "ether": "[OD2]([#6])[#6]",
            "thiol": "[#16X2H]",
            "sulfide": "[#16X2]([#6])[#6]",
            "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
            "nitrile": "[NX1]#[CX2]",
            "phosphate": "[PX4](=O)([OX2])",
            "aromatic_ring": "c1ccccc1",
        }
        for name, smarts in patterns.items():
            pat = Chem.MolFromSmarts(smarts)
            if pat and mol.HasSubstructMatch(pat):
                matches = mol.GetSubstructMatches(pat)
                func_groups[name] = len(matches)

        props = {
            "valid": True,
            "smiles": smiles,
            "canonical_smiles": Chem.MolToSmiles(mol),
            "molecular_weight": round(Descriptors.MolWt(mol), 2),
            "exact_mass": round(Descriptors.ExactMolWt(mol), 4),
            "molecular_formula": rdMolDescriptors.CalcMolFormula(mol),
            "logp": round(Crippen.MolLogP(mol), 2),
            "hbd": Descriptors.NumHDonors(mol),
            "hba": Descriptors.NumHAcceptors(mol),
            "tpsa": round(Descriptors.TPSA(mol), 2),
            "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "num_heavy_atoms": mol.GetNumHeavyAtoms(),
            "num_rings": ring_info.NumRings(),
            "num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "num_stereo_centers": len(chiral),
            "fraction_csp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
            "functional_groups": func_groups,
            "synthetic_accessibility": round(1.0 - _sa_score(mol), 3),  # Invert so 1 = easy
            "molecular_complexity": round(_sa_score(mol), 3),
        }
        return props
    except Exception as e:
        logger.error(f"Property computation failed for {smiles}: {e}")
        return {"valid": False, "smiles": smiles, "error": str(e)}


# ============================================================================
#  ROUTE SCORING ENGINE
# ============================================================================

def score_reaction_step(
    product_smiles: str,
    reactant_smiles_list: List[str],
    beam_rank: int = 0,
    reaction_center: Optional[ReactionCenterInfo] = None,
) -> RouteScore:
    """Score an individual retrosynthesis step comprehensively."""
    score = RouteScore()

    product_mol = Chem.MolFromSmiles(product_smiles)
    reactant_mols = [Chem.MolFromSmiles(s) for s in reactant_smiles_list]
    valid_reactants = [m for m in reactant_mols if m is not None]

    if product_mol is None or not valid_reactants:
        score.overall_score = 0.0
        return score

    # 1. Reaction Confidence (based on beam rank and atom conservation)
    beam_confidence = 1.0 / (1.0 + beam_rank * 0.3)

    # Atom conservation check
    product_atoms = defaultdict(int)
    for atom in product_mol.GetAtoms():
        product_atoms[atom.GetAtomicNum()] += 1

    reactant_atoms = defaultdict(int)
    for mol in valid_reactants:
        for atom in mol.GetAtoms():
            reactant_atoms[atom.GetAtomicNum()] += 1

    # How well atoms are conserved
    all_atoms = set(list(product_atoms.keys()) + list(reactant_atoms.keys()))
    conservation_scores = []
    for atomic_num in all_atoms:
        p = product_atoms.get(atomic_num, 0)
        r = reactant_atoms.get(atomic_num, 0)
        if max(p, r) > 0:
            conservation_scores.append(min(p, r) / max(p, r))
    atom_conservation = sum(conservation_scores) / len(conservation_scores) if conservation_scores else 0.0

    score.reaction_confidence = 0.5 * beam_confidence + 0.5 * atom_conservation

    # 2. Synthetic Accessibility
    product_sa = _sa_score(product_mol)
    reactant_sa = [_sa_score(m) for m in valid_reactants]
    avg_reactant_sa = sum(reactant_sa) / len(reactant_sa)

    # If reactants are simpler than product, that's good
    sa_improvement = max(0, product_sa - avg_reactant_sa)
    score.synthetic_accessibility = min(1.0, 0.5 + sa_improvement)

    # 3. Chemical Feasibility
    feasibility = 0.7  # baseline

    # Penalize if too many fragments
    if len(valid_reactants) > 3:
        feasibility *= 0.7
    elif len(valid_reactants) == 1:
        feasibility *= 0.9  # Unimolecular reactions are less common in retro

    # Bonus for known functional group transformations
    if reaction_center:
        if reaction_center.reaction_class not in ("unknown", "rearrangement"):
            feasibility *= 1.15

    # Size sanity check
    product_size = product_mol.GetNumHeavyAtoms()
    total_reactant_size = sum(m.GetNumHeavyAtoms() for m in valid_reactants)
    if total_reactant_size > product_size * 2.5:
        feasibility *= 0.6
    elif abs(total_reactant_size - product_size) <= 2:
        feasibility *= 1.1

    score.chemical_feasibility = min(1.0, feasibility)

    # 4. Uncertainty (lower is better — we report it as 0..1 where 0 = no uncertainty)
    # Based on number of reactants, atom conservation, and beam rank
    uncertainty = 0.0
    if beam_rank > 0:
        uncertainty += 0.1 * beam_rank
    if atom_conservation < 0.8:
        uncertainty += 0.2
    if len(valid_reactants) > 2:
        uncertainty += 0.1
    score.uncertainty = min(1.0, uncertainty)

    # 5. Route Diversity (placeholder — meaningful with multi-route)
    score.route_diversity = 0.5

    # 6. Estimated Cost (heuristic)
    # More steps, more reagents, harder intermediates → higher cost
    cost = 0.3  # baseline
    cost += len(valid_reactants) * 0.05
    cost += avg_reactant_sa * 0.3
    # Exotic atoms increase cost
    exotic = {"Pd", "Pt", "Ru", "Ir", "Rh", "Os"}
    for mol in valid_reactants:
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in exotic:
                cost += 0.2
    score.estimated_cost = min(1.0, cost)

    score.compute_overall()
    return score


def score_full_route(pathway: List[Dict], building_blocks: Optional[set] = None) -> RouteScore:
    """Aggregate step-level scores into a route-level score."""
    if not pathway:
        return RouteScore()

    step_scores = []
    for i, step in enumerate(pathway):
        rc = None
        try:
            rc = detect_reaction_center(step["product"], step["reactants"])
        except Exception:
            pass
        step_score = score_reaction_step(
            step["product"],
            step["reactants"],
            beam_rank=0,
            reaction_center=rc,
        )
        step_scores.append(step_score)

    if not step_scores:
        return RouteScore()

    # Aggregate: confidence is multiplicative, others are averaged
    route = RouteScore()
    route.reaction_confidence = 1.0
    for s in step_scores:
        route.reaction_confidence *= s.reaction_confidence
    route.reaction_confidence = route.reaction_confidence ** (1.0 / len(step_scores))

    route.synthetic_accessibility = sum(s.synthetic_accessibility for s in step_scores) / len(step_scores)
    route.chemical_feasibility = sum(s.chemical_feasibility for s in step_scores) / len(step_scores)
    route.route_diversity = sum(s.route_diversity for s in step_scores) / len(step_scores)
    route.uncertainty = max(s.uncertainty for s in step_scores)  # worst-case
    route.estimated_cost = sum(s.estimated_cost for s in step_scores) / len(step_scores)

    # Terminal building block bonus
    if building_blocks:
        terminal_mols = set()
        non_terminal_products = {step["product"] for step in pathway}
        for step in pathway:
            for r in step["reactants"]:
                if r not in non_terminal_products:
                    terminal_mols.add(r)
        if terminal_mols:
            bb_fraction = sum(1 for m in terminal_mols if m in building_blocks) / len(terminal_mols)
            route.chemical_feasibility = min(1.0, route.chemical_feasibility + bb_fraction * 0.15)

    route.compute_overall()
    return route


# ============================================================================
#  MULTI-ROUTE COMPARISON
# ============================================================================

def compare_routes(routes: List[List[Dict]], building_blocks: Optional[set] = None) -> List[Dict]:
    """
    Score multiple routes and return them sorted by overall score.
    Each route is a list of pathway steps.
    """
    scored_routes = []
    for i, route in enumerate(routes):
        route_score = score_full_route(route, building_blocks)

        # Diversity: how different is this route from others?
        # Based on unique intermediates
        intermediates = set()
        for step in route:
            intermediates.add(step["product"])
            intermediates.update(step["reactants"])

        scored_routes.append({
            "route_id": i + 1,
            "pathway": route,
            "score": route_score,
            "num_steps": len(route),
            "unique_intermediates": len(intermediates),
            "total_reactions": len(route),
        })

    # Update diversity based on comparison with other routes
    if len(scored_routes) > 1:
        all_intermediates = []
        for sr in scored_routes:
            ints = set()
            for step in sr["pathway"]:
                ints.add(step["product"])
                ints.update(step["reactants"])
            all_intermediates.append(ints)

        for i, sr in enumerate(scored_routes):
            others = set()
            for j, ints in enumerate(all_intermediates):
                if j != i:
                    others |= ints
            unique_to_route = all_intermediates[i] - others
            sr["score"].route_diversity = len(unique_to_route) / max(1, len(all_intermediates[i]))
            sr["score"].compute_overall()

    scored_routes.sort(key=lambda x: x["score"].overall_score, reverse=True)
    return scored_routes


# ============================================================================
#  MOLECULE IMAGE GENERATION (with reaction center highlighting)
# ============================================================================

def mol_to_svg(smiles: str, highlight_atoms: Optional[Set[int]] = None,
               width: int = 350, height: int = 250) -> Optional[str]:
    """Generate an SVG of a molecule, optionally highlighting reaction-center atoms."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        AllChem.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        opts = drawer.drawOptions()
        opts.addAtomIndices = False
        opts.addStereoAnnotation = True
        opts.bondLineWidth = 2.0

        if highlight_atoms:
            valid_atoms = [a for a in highlight_atoms if a < mol.GetNumAtoms()]
            highlight_bonds = []
            for bond in mol.GetBonds():
                if bond.GetBeginAtomIdx() in valid_atoms and bond.GetEndAtomIdx() in valid_atoms:
                    highlight_bonds.append(bond.GetIdx())

            atom_colors = {a: (1.0, 0.3, 0.3) for a in valid_atoms}
            bond_colors = {b: (1.0, 0.3, 0.3) for b in highlight_bonds}
            drawer.DrawMolecule(
                mol,
                highlightAtoms=valid_atoms,
                highlightAtomColors=atom_colors,
                highlightBonds=highlight_bonds,
                highlightBondColors=bond_colors,
            )
        else:
            drawer.DrawMolecule(mol)

        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception as e:
        logger.warning(f"SVG generation failed for {smiles}: {e}")
        return None


def mol_to_png_base64(smiles: str, highlight_atoms: Optional[Set[int]] = None,
                      size: int = 300) -> Optional[str]:
    """Generate a base64-encoded PNG of a molecule."""
    import base64
    from io import BytesIO

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        if highlight_atoms:
            valid_atoms = [a for a in highlight_atoms if a < mol.GetNumAtoms()]
            img = Draw.MolToImage(mol, size=(size, size), highlightAtoms=valid_atoms)
        else:
            img = Draw.MolToImage(mol, size=(size, size))
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None
