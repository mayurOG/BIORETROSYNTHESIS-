"""Advanced chemistry utilities for multi-format input, validation, and structure conversion."""

import logging
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import re

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Descriptors3D
from rdkit.Chem import rdMolDescriptors
import pubchempy as pcp
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class ChemicalStructure:
    """Unified chemical structure representation."""
    smiles: str
    inchi: Optional[str] = None
    inchi_key: Optional[str] = None
    molecular_weight: Optional[float] = None
    molecular_formula: Optional[str] = None
    exact_mass: Optional[float] = None
    logp: Optional[float] = None
    hbd: Optional[int] = None  # H-bond donors
    hba: Optional[int] = None  # H-bond acceptors
    rotatable_bonds: Optional[int] = None
    aromatic_rings: Optional[int] = None
    source: str = "input"  # "input", "pubchem", "conversion"
    validity: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "smiles": self.smiles,
            "inchi": self.inchi,
            "inchi_key": self.inchi_key,
            "molecular_weight": round(self.molecular_weight, 2) if self.molecular_weight else None,
            "molecular_formula": self.molecular_formula,
            "exact_mass": round(self.exact_mass, 4) if self.exact_mass else None,
            "logp": round(self.logp, 2) if self.logp else None,
            "hbd": self.hbd,
            "hba": self.hba,
            "rotatable_bonds": self.rotatable_bonds,
            "aromatic_rings": self.aromatic_rings,
            "source": self.source,
            "validity": self.validity,
            "error": self.error_message,
        }


class ChemistryValidator:
    """Comprehensive chemical structure validation."""

    # Lipinski's Rule of Five thresholds
    LIPINSKI_LIMITS = {
        "mw": 500,
        "logp": 5,
        "hbd": 5,
        "hba": 10,
    }

    # Common chemical patterns (SMART patterns)
    COMMON_PATTERNS = {
        "aromatic": "[a]",
        "nitrogen": "[N,n]",
        "sulfur": "[S,s]",
        "halogen": "[F,Cl,Br,I]",
        "amide": "[N][C](=O)",
        "carboxylic_acid": "[C](=O)[O]",
        "ester": "[C](=O)[O][C]",
        "ether": "[O][C]",
        "thiol": "[S][H]",
        "amine": "[N][H]",
        "hydroxyl": "[O][H]",
    }

    @staticmethod
    def validate_smiles(smiles: str) -> Tuple[bool, Optional[str]]:
        """Validate SMILES string."""
        if not smiles or not isinstance(smiles, str):
            return False, "Invalid input: SMILES must be a non-empty string"
        
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return False, f"RDKit cannot parse SMILES: {smiles}"
        
        # Check for common issues
        try:
            Chem.SanitizeMol(mol)
            Chem.AllChem.EmbedMolecule(mol, randomSeed=42)
        except Exception as e:
            return False, f"Sanitization failed: {str(e)}"
        
        return True, None

    @staticmethod
    def validate_valence(mol) -> Tuple[bool, Optional[str]]:
        """Check valence correctness."""
        if mol is None:
            return False, "Invalid molecule"
        
        errors = Chem.DetectChemistryProblems(mol)
        if errors:
            return False, f"Valence errors: {[str(e) for e in errors]}"
        
        return True, None

    @staticmethod
    def check_lipinski_compliance(mol) -> Dict[str, bool]:
        """Check Lipinski's Rule of Five."""
        if mol is None:
            return {"compliant": False, "violations": ["Invalid molecule"]}
        
        violations = []
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        
        if mw > ChemistryValidator.LIPINSKI_LIMITS["mw"]:
            violations.append(f"MW ({mw:.1f}) > 500")
        if logp > ChemistryValidator.LIPINSKI_LIMITS["logp"]:
            violations.append(f"LogP ({logp:.1f}) > 5")
        if hbd > ChemistryValidator.LIPINSKI_LIMITS["hbd"]:
            violations.append(f"HBD ({hbd}) > 5")
        if hba > ChemistryValidator.LIPINSKI_LIMITS["hba"]:
            violations.append(f"HBA ({hba}) > 10")
        
        return {
            "compliant": len(violations) <= 1,
            "violations": violations,
            "metrics": {"mw": mw, "logp": logp, "hbd": hbd, "hba": hba}
        }

    @staticmethod
    def detect_functional_groups(mol) -> Dict[str, List[Tuple]]:
        """Detect functional groups in molecule."""
        if mol is None:
            return {}
        
        groups = {}
        for name, smart in ChemistryValidator.COMMON_PATTERNS.items():
            pattern = Chem.MolFromSmarts(smart)
            matches = mol.GetSubstructMatches(pattern)
            if matches:
                groups[name] = matches
        
        return groups


class InputConverter:
    """Convert various chemical input formats to SMILES."""

    @staticmethod
    @lru_cache(maxsize=1024)
    def inchi_to_smiles(inchi: str) -> Optional[str]:
        """Convert InChI to SMILES."""
        try:
            mol = Chem.MolFromInchi(inchi)
            if mol:
                return Chem.MolToSmiles(mol)
        except Exception as e:
            logger.warning(f"InChI conversion failed: {e}")
        return None

    @staticmethod
    @lru_cache(maxsize=1024)
    def inchi_key_to_smiles(inchi_key: str) -> Optional[str]:
        """Convert InChI Key to SMILES via PubChem lookup."""
        try:
            results = pcp.get_compounds(inchi_key, "inchikey")
            if results:
                return results[0].canonical_smiles
        except Exception as e:
            logger.warning(f"InChI Key lookup failed: {e}")
        return None

    @staticmethod
    @lru_cache(maxsize=1024)
    def name_to_smiles(name: str) -> Optional[str]:
        """Convert chemical name or CAS number to SMILES via PubChem."""
        try:
            # Try as compound name
            results = pcp.get_compounds(name, "name")
            if results:
                return results[0].canonical_smiles
            
            # Try as CAS number
            results = pcp.get_compounds(name, "cid")
            if results:
                return results[0].canonical_smiles
        except Exception as e:
            logger.warning(f"Name lookup failed: {e}")
        return None

    @staticmethod
    @lru_cache(maxsize=1024)
    def formula_to_smiles(formula: str) -> Optional[str]:
        """Convert molecular formula to SMILES via PubChem."""
        try:
            results = pcp.get_compounds(formula, "formula")
            if results:
                # Return most common isotopologue
                return max(results, key=lambda x: x.cid).canonical_smiles
        except Exception as e:
            logger.warning(f"Formula lookup failed: {e}")
        return None

    @staticmethod
    def detect_input_format(user_input: str) -> str:
        """Auto-detect input format."""
        user_input = user_input.strip()
        
        # InChI
        if user_input.startswith("InChI="):
            return "inchi"
        
        # InChI Key (format: XXXX-XXXX-X)
        if re.match(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$", user_input):
            return "inchi_key"
        
        # CAS number (format: XXXXX-XX-X or XXXXXX-XX-X, etc.)
        if re.match(r"^\d{1,7}-\d{2}-\d$", user_input):
            return "cas_number"
        
        # Molecular formula (elements + numbers, e.g., C6H12O6)
        if re.match(r"^[A-Z][a-z]?(\d+)?([A-Z][a-z]?(\d+)?)*$", user_input) and not any(c in user_input for c in "()=[]"):
            return "formula"
        
        # SMILES (contains special characters typical of SMILES)
        if any(c in user_input for c in "()=[]\\/@#+-"):
            return "smiles"
        
        # Assume chemical name
        return "name"


class StructureProcessor:
    """Main processor for converting to and validating ChemicalStructure."""

    def __init__(self):
        self.validator = ChemistryValidator()
        self.converter = InputConverter()

    def process_input(self, user_input: str) -> ChemicalStructure:
        """Process user input and return validated ChemicalStructure."""
        user_input = user_input.strip()
        
        # Detect format and convert to SMILES
        input_format = self.converter.detect_input_format(user_input)
        logger.info(f"Detected input format: {input_format}")
        
        smiles = None
        source = input_format
        
        if input_format == "smiles":
            smiles = user_input
        elif input_format == "inchi":
            smiles = self.converter.inchi_to_smiles(user_input)
            source = "inchi_conversion"
        elif input_format == "inchi_key":
            smiles = self.converter.inchi_key_to_smiles(user_input)
            source = "pubchem"
        elif input_format in ["name", "cas_number"]:
            smiles = self.converter.name_to_smiles(user_input)
            source = "pubchem"
        elif input_format == "formula":
            smiles = self.converter.formula_to_smiles(user_input)
            source = "pubchem"
        
        if not smiles:
            return ChemicalStructure(
                smiles="",
                source=source,
                validity=False,
                error_message=f"Could not convert {input_format} input to SMILES"
            )
        
        # Validate SMILES
        valid, error = self.validator.validate_smiles(smiles)
        if not valid:
            return ChemicalStructure(
                smiles=smiles,
                source=source,
                validity=False,
                error_message=error
            )
        
        # Parse molecule and compute properties
        mol = Chem.MolFromSmiles(smiles)
        
        # Compute descriptors
        try:
            inchi = Chem.MolToInchi(mol)
            inchi_key = Chem.MolToInchiKey(mol)
            molecular_formula = rdMolDescriptors.CalcMolFormula(mol)
            molecular_weight = Descriptors.MolWt(mol)
            exact_mass = Descriptors.ExactMolWt(mol)
            logp = Crippen.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
            aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        except Exception as e:
            logger.error(f"Descriptor calculation failed: {e}")
            inchi = inchi_key = molecular_formula = None
            molecular_weight = exact_mass = logp = hbd = hba = rotatable_bonds = aromatic_rings = None
        
        return ChemicalStructure(
            smiles=smiles,
            inchi=inchi,
            inchi_key=inchi_key,
            molecular_weight=molecular_weight,
            molecular_formula=molecular_formula,
            exact_mass=exact_mass,
            logp=logp,
            hbd=hbd,
            hba=hba,
            rotatable_bonds=rotatable_bonds,
            aromatic_rings=aromatic_rings,
            source=source,
            validity=True,
        )

    def get_detailed_analysis(self, smiles: str) -> Dict:
        """Get comprehensive analysis of a molecule."""
        valid, _ = self.validator.validate_smiles(smiles)
        if not valid:
            return {"valid": False, "error": "Invalid SMILES"}
        
        mol = Chem.MolFromSmiles(smiles)
        
        return {
            "valid": True,
            "valence_check": self.validator.validate_valence(mol),
            "lipinski": self.validator.check_lipinski_compliance(mol),
            "functional_groups": self.validator.detect_functional_groups(mol),
        }
