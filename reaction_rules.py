"""Reaction rule filters and feasibility scoring."""

import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re

from rdkit import Chem
from rdkit.Chem import AllChem, Draw

logger = logging.getLogger(__name__)


@dataclass
class ReactionRule:
    """Chemical reaction rule (SMARTS pattern)."""
    name: str
    smarts: str  # Reaction SMARTS
    description: str
    priority: int  # Higher = more important
    category: str  # e.g., "oxidation", "reduction", "coupling"


class ReactionRuleDatabase:
    """Database of common chemical transformations."""

    # Common organic transformation rules
    REACTION_RULES = [
        ReactionRule(
            name="ester_hydrolysis",
            smarts="[C:1](=[O:2])[O:3][C:4]>>[C:1](=[O:2])[O:3].[C:4]",
            description="Ester hydrolysis",
            priority=10,
            category="hydrolysis"
        ),
        ReactionRule(
            name="amide_hydrolysis",
            smarts="[C:1](=[O:2])[N:3]([C:4])[C:5]>>[C:1](=[O:2])[N:3].[C:4].[C:5]",
            description="Amide hydrolysis",
            priority=10,
            category="hydrolysis"
        ),
        ReactionRule(
            name="carboxylic_acid_decarboxylation",
            smarts="[C:1](=[O:2])[O:3][H:4]>>[C:1][H:4].[C](=[O:2])[O:3]",
            description="Carboxylic acid decarboxylation",
            priority=8,
            category="decarboxylation"
        ),
        ReactionRule(
            name="williamson_ether_synthesis",
            smarts="[C:1][O:2][C:3]>>[C:1][O:2][H].[Cl:4][C:3]",
            description="Williamson ether synthesis",
            priority=9,
            category="coupling"
        ),
        ReactionRule(
            name="grignard_reaction",
            smarts="[C:1](=[O:2])[C:3]>>[C:1]([O:2][MgBr])[C:3]",
            description="Grignard reaction",
            priority=7,
            category="coupling"
        ),
        ReactionRule(
            name="suzuki_coupling",
            smarts="[c:1][Br:2].[c:3][B:4]([O:5])[O:6]>>[c:1][c:3]",
            description="Suzuki coupling",
            priority=9,
            category="coupling"
        ),
        ReactionRule(
            name="protecting_group_removal_cbz",
            smarts="[N:1][C:2](=[O:3])[C:4][c:5]>>[N:1][H].[C:2](=[O:3])[O:6].[c:5][C:4]",
            description="CBz protecting group removal",
            priority=6,
            category="protecting_group_removal"
        ),
        ReactionRule(
            name="oxidation_alcohol_to_aldehyde",
            smarts="[C:1][C:2]([O:3][H:4])[C:5]>>[C:1][C:2](=[O:3])[C:5]",
            description="Alcohol to aldehyde oxidation",
            priority=8,
            category="oxidation"
        ),
        ReactionRule(
            name="reduction_ester_to_alcohol",
            smarts="[C:1](=[O:2])[O:3][C:4]>>[C:1][O:5][H:6].[O:3][C:4]",
            description="Ester to alcohol reduction",
            priority=8,
            category="reduction"
        ),
    ]

    def __init__(self):
        """Initialize reaction rule database."""
        self.rules = self.REACTION_RULES
        self.compiled_rxns = {}
        self._compile_reaction_smarts()

    def _compile_reaction_smarts(self):
        """Compile reaction SMARTS to RxnSmarts objects."""
        for rule in self.rules:
            try:
                rxn = AllChem.ReactionFromSmarts(rule.smarts)
                if rxn:
                    self.compiled_rxns[rule.name] = rxn
                else:
                    logger.warning(f"Failed to compile reaction SMARTS: {rule.name}")
            except Exception as e:
                logger.error(f"Error compiling {rule.name}: {e}")

    def is_reaction_applicable(self, reactants_smiles: List[str], product_smiles: str) -> List[str]:
        """Check which reaction rules could produce the product from reactants."""
        applicable_rules = []
        
        try:
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(s) for s in reactants_smiles if Chem.MolFromSmiles(s)]
            
            if not product_mol or not reactant_mols:
                return applicable_rules
            
            for rule in sorted(self.rules, key=lambda r: r.priority, reverse=True):
                if rule.name in self.compiled_rxns:
                    rxn = self.compiled_rxns[rule.name]
                    # Check if reaction could have produced product from reactants
                    # (simplified check - full validation would require reaction matching)
                    applicable_rules.append(rule.name)
        
        except Exception as e:
            logger.error(f"Error checking reaction applicability: {e}")
        
        return applicable_rules


class ReactionFeasibilityScorer:
    """Score feasibility of predicted reactions."""

    # Penalty factors for unfavorable conditions
    PENALTY_FACTORS = {
        "bulky_groups": 0.8,  # Reduced success with steric hindrance
        "reactive_functional_groups": 0.7,  # Multiple reactive groups
        "toxic_reactants": 0.5,  # Known toxic compounds
        "expensive_reagents": 0.6,  # Expensive catalysts/reagents
        "long_reaction_time": 0.85,  # > 24h reactions
        "exotic_conditions": 0.4,  # Extreme T, P, inert atmosphere
    }

    # Bonus factors for favorable conditions
    BONUS_FACTORS = {
        "green_chemistry": 1.2,
        "simple_workup": 1.15,
        "common_solvents": 1.1,
        "mild_conditions": 1.15,
        "high_yield": 1.2,
    }

    EXPENSIVE_REAGENTS = {
        "Pd", "Pt", "Ru", "Ir", "Rh",  # Precious metals
        "Borane", "Silane", "Red-Al",  # Reducing agents
    }

    TOXIC_COMPOUNDS = {
        "[cyanide]", "[azide]", "[diazo]", "[isocyanate]",
    }

    @classmethod
    def score_reaction(
        cls,
        reactants: List[str],
        product: str,
        reaction_type: Optional[str] = None,
    ) -> Tuple[float, Dict[str, str]]:
        """
        Score feasibility of a reaction (0-1).
        
        Args:
            reactants: SMILES of starting materials
            product: SMILES of product
            reaction_type: Optional reaction classification
        
        Returns:
            (score, factors_dict) - score in [0, 1] and contributing factors
        """
        base_score = 0.7  # Default moderate feasibility
        factors = {}
        
        try:
            # Check for steric issues
            reactant_mols = [Chem.MolFromSmiles(s) for s in reactants if Chem.MolFromSmiles(s)]
            product_mol = Chem.MolFromSmiles(product)
            
            if reactant_mols and product_mol:
                # Check molecular complexity
                total_reactant_atoms = sum(m.GetNumAtoms() for m in reactant_mols)
                product_atoms = product_mol.GetNumAtoms()
                
                # Size matching (penalize if product >> reactants, unlikely atom rearrangement)
                if product_atoms > total_reactant_atoms * 2:
                    base_score *= cls.PENALTY_FACTORS["exotic_conditions"]
                    factors["unlikely_size_increase"] = "Product much larger than reactants"
                
                # Check for reactive groups
                for reactant in reactants:
                    if any(toxic in reactant for toxic in cls.TOXIC_COMPOUNDS):
                        base_score *= cls.PENALTY_FACTORS["toxic_reactants"]
                        factors["toxic_reagent"] = "Contains potentially toxic functional group"
                        break
                
                # Reaction type bonus
                if reaction_type in ["green_chemistry", "catalytic", "simple_substitution"]:
                    base_score *= cls.BONUS_FACTORS.get(reaction_type, 1.0)
                    factors["reaction_type"] = f"Favorable: {reaction_type}"
        
        except Exception as e:
            logger.error(f"Error scoring reaction: {e}")
        
        return min(1.0, base_score), factors

    @classmethod
    def flag_problematic_reactions(cls, reactants: List[str]) -> List[str]:
        """Flag potential issues with reactants."""
        flags = []
        
        for reactant in reactants:
            for group in cls.TOXIC_COMPOUNDS:
                if group in reactant:
                    flags.append(f"Potentially toxic: {reactant}")
            
            # Check for expensive catalysts
            if any(metal in reactant for metal in cls.EXPENSIVE_REAGENTS):
                flags.append(f"Uses expensive reagents: {reactant}")
        
        return flags


class GreenChemistryFilter:
    """Filter predictions for green chemistry principles (Anastas & Warner)."""

    GREEN_PRINCIPLES = {
        "prevent_waste": 1.0,
        "atom_economy": 0.9,
        "safer_synthesis": 0.8,
        "efficient_energy": 0.85,
        "renewable_feedstock": 0.75,
        "reduce_derivatives": 0.7,
        "catalytic": 1.0,
        "design_degradation": 0.6,
        "real_time_pollution": 0.8,
        "safer_processes": 0.7,
    }

    @classmethod
    def score_green_chemistry(cls, reactants: List[str], product: str) -> float:
        """Score how well prediction aligns with green chemistry."""
        score = 0.7  # Baseline
        
        # Penalize for complex reactants
        if len(reactants) > 3:
            score *= 0.8
        
        # Bonus for simple, common reactants
        common_reactants = {"C", "O", "N", "S", "Cl", "Br", "I"}
        if all(reactant in common_reactants for reactant in reactants):
            score *= 1.1
        
        return min(1.0, score)
