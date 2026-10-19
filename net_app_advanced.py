# Copyright (C) 2026 Mayur Nhavalde
#
# This file is part of BioRetrosynthesis.
#
# BioRetrosynthesis is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BioRetrosynthesis is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with BioRetrosynthesis. If not, see <https://www.gnu.org/licenses/>.

"""
BioRetrosynthesis — Advanced Retrosynthesis Visualization & Planning
====================================================================
Interactive retrosynthesis search tree, 3D molecular conformers,
reaction-center analysis, multi-objective route scoring, and
multi-route comparison — all computed from real chemistry via RDKit.

Run:  streamlit run net_app_advanced.py
"""

import os
import streamlit as st
import pandas as pd
import json
import time
import base64
import logging
from pathlib import Path
from io import BytesIO
from typing import List, Dict, Optional, Set
from collections import defaultdict

# RDKit
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors, Crippen, rdMolDescriptors
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# Plotly for interactive visualizations
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx

# Our modules
from retrosynthesis_engine import (
    detect_reaction_center,
    generate_3d_conformer,
    compute_molecular_properties,
    score_reaction_step,
    score_full_route,
    compare_routes,
    mol_to_svg,
    mol_to_png_base64,
    synthetic_accessibility_score,
    ReactionCenterInfo,
    RouteScore,
    BondChange,
)

logger = logging.getLogger(__name__)

# ============================================================================
#  PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="🔬 BioRetrosynthesis — Advanced Visualization",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
#  GLOBAL CSS
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global styling */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Main header */
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        letter-spacing: -0.02em;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #cbd5e1 !important;
        margin-top: 4px;
        margin-bottom: 24px;
    }

    /* All Section Dividers and Headings in BRIGHT WHITE */
    .section-header {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        margin: 32px 0 16px 0 !important;
        padding-bottom: 8px !important;
        border-bottom: 3px solid #818cf8 !important;
        display: inline-block !important;
        letter-spacing: -0.01em !important;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.5) !important;
    }

    h1, h2, h3, h4, h5, h6,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* Sidebar Headings & Text in High Contrast White/Light */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p strong,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
    }
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #94a3b8 !important;
    }

    /* General Markdown text in light slate */
    .stMarkdown p, .stMarkdown strong {
        color: #f1f5f9;
    }

    /* Image container constraint to PREVENT ANY OVERLAP */
    [data-testid="stImage"] {
        max-width: 100% !important;
        overflow: hidden !important;
    }
    [data-testid="stImage"] img {
        max-width: 100% !important;
        height: auto !important;
        border-radius: 12px !important;
        background: #ffffff !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
        padding: 8px !important;
        box-sizing: border-box !important;
        display: block !important;
        margin: 0 auto !important;
    }
    [data-testid="stImage"] [data-testid="stCaptionContainer"] {
        color: #e2e8f0 !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        text-align: center !important;
        margin-top: 6px !important;
    }

    /* Metric cards: Prevent ellipsis truncation, allow full text display */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.65) !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 12px !important;
        padding: 10px 14px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25) !important;
        min-width: 0 !important;
        transition: transform 0.15s ease, border-color 0.15s ease !important;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(129, 140, 248, 0.6) !important;
        transform: translateY(-2px) !important;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }
    [data-testid="stMetricLabel"] * {
        color: #94a3b8 !important;
        font-size: 0.8rem !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.15rem !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        white-space: normal !important;
        word-break: break-word !important;
        overflow: visible !important;
        text-overflow: clip !important;
        line-height: 1.25 !important;
    }
    [data-testid="stMetricValue"] * {
        white-space: normal !important;
        word-break: break-word !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }

    /* Cards */
    .score-card {
        background: rgba(30, 41, 59, 0.7);
        padding: 20px 24px;
        border-radius: 16px;
        margin: 8px 0;
        border: 1px solid rgba(148, 163, 184, 0.2);
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .score-card h4 {
        margin: 0 0 8px 0;
        color: #ffffff !important;
        font-weight: 600;
    }
    .score-value {
        font-size: 2rem;
        font-weight: 700;
        color: #818cf8;
    }

    /* Bond change badges */
    .bond-broken {
        display: inline-block;
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 2px 4px;
    }
    .bond-formed {
        display: inline-block;
        background: linear-gradient(135deg, #2ed573, #009432);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 2px 4px;
    }
    .reaction-class {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.88rem;
        font-weight: 600;
        margin: 4px 0;
    }

    /* Tree node styling */
    .tree-node-target {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 10px 20px;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
    }
    .tree-node-intermediate {
        background: linear-gradient(135deg, #ffecd2, #fcb69f);
        padding: 10px 20px;
        border-radius: 12px;
        text-align: center;
    }
    .tree-node-building-block {
        background: linear-gradient(135deg, #a8edea, #fed6e3);
        padding: 10px 20px;
        border-radius: 12px;
        text-align: center;
    }

    /* Score bar */
    .score-bar-container {
        background: #334155;
        border-radius: 10px;
        height: 12px;
        overflow: hidden;
        margin: 4px 0;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.6s ease;
    }

    /* Tooltip chip styling */
    .info-chip {
        display: inline-block;
        background: rgba(99, 102, 241, 0.25) !important;
        color: #c7d2fe !important;
        border: 1px solid rgba(165, 180, 252, 0.35) !important;
        padding: 4px 12px;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 3px;
    }

    /* Route comparison card styles */
    .route-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 14px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .route-card:hover {
        transform: translateY(-2px);
        border-color: rgba(129, 140, 248, 0.6);
    }
    .badge-recommended {
        display: inline-block;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white !important;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        margin-bottom: 8px;
    }
    .badge-green {
        display: inline-block;
        background: linear-gradient(135deg, #10b981, #059669);
        color: white !important;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        margin-bottom: 8px;
    }
    .badge-alt {
        display: inline-block;
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white !important;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        margin-bottom: 8px;
    }
    .exec-brief-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
        border-left: 4px solid #818cf8;
        border-radius: 12px;
        padding: 18px 22px;
        margin: 16px 0;
        color: #f1f5f9;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================================
#  HELPER: SMILES → IMAGE
# ============================================================================

def smiles_to_image_bytes(smiles: str, size: int = 300, highlight_atoms=None) -> Optional[bytes]:
    """Generate PNG bytes for a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        AllChem.Compute2DCoords(mol)
        if highlight_atoms:
            valid = [a for a in highlight_atoms if a < mol.GetNumAtoms()]
            img = Draw.MolToImage(mol, size=(size, size), highlightAtoms=valid)
        else:
            img = Draw.MolToImage(mol, size=(size, size))
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def smiles_to_b64(smiles: str, size: int = 250, highlight_atoms=None) -> Optional[str]:
    data = smiles_to_image_bytes(smiles, size, highlight_atoms)
    if data:
        return f"data:image/png;base64,{base64.b64encode(data).decode()}"
    return None


def to_subscript(text: str) -> str:
    """Convert digits in a molecular formula to unicode subscripts."""
    if not text:
        return ""
    subscripts = {
        '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
        '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉'
    }
    return ''.join(subscripts.get(c, c) for c in str(text))


def render_molecule_image(smiles: str, caption: str = "", size: int = 280, highlight_atoms=None, use_container_width: bool = True):
    """Render a molecule in Streamlit without overflowing containers."""
    data = smiles_to_image_bytes(smiles, size, highlight_atoms)
    if data:
        st.image(data, caption=caption, use_container_width=use_container_width)
    else:
        st.code(smiles, language=None)
        st.caption(f"{caption} (structure not renderable)")


# ============================================================================
#  INPUT PROCESSING (using real RDKit + PubChem)
# ============================================================================

def resolve_to_smiles(user_input: str) -> tuple:
    """
    Resolve any input format to a canonical SMILES string.
    Returns (smiles, source_format, error_message).
    """
    inp = user_input.strip()
    if not inp:
        return None, "empty", "No input provided"

    import re

    # 1. Try as SMILES first
    mol = Chem.MolFromSmiles(inp)
    if mol is not None:
        return Chem.MolToSmiles(mol), "SMILES", None

    # 2. Try as InChI
    if inp.startswith("InChI="):
        mol = Chem.MolFromInchi(inp)
        if mol:
            return Chem.MolToSmiles(mol), "InChI", None
        return None, "InChI", "Could not parse InChI string"

    # 3. Try as CAS number
    if re.match(r"^\d{1,7}-\d{2}-\d$", inp):
        try:
            import pubchempy as pcp
            results = pcp.get_compounds(inp, "name")
            if results:
                return results[0].canonical_smiles, "CAS", None
        except Exception:
            pass
        return None, "CAS", f"Could not resolve CAS number: {inp}"

    # 4. Try as molecular formula
    if re.match(r"^[A-Z][a-z]?(\d+)?([A-Z][a-z]?(\d+)?)*$", inp) and not any(c in inp for c in "()=[]"):
        try:
            import pubchempy as pcp
            results = pcp.get_compounds(inp, "formula")
            if results:
                return results[0].canonical_smiles, "Formula", None
        except Exception:
            pass

    # 5. Try as chemical name via PubChem
    try:
        import pubchempy as pcp
        results = pcp.get_compounds(inp, "name")
        if results:
            return results[0].canonical_smiles, "Name", None
    except Exception:
        pass

    return None, "unknown", f"Could not resolve input: {inp}"


# ============================================================================
#  RETROSYNTHESIS PREDICTION (uses real model if available, else smart demo)
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent


@st.cache_resource(show_spinner="Loading the retrosynthesis model...")
def _load_real_engine():
    from net_model_utils import load_model
    from tbr_opt import RetroEngine, BuildingBlockIndex

    bundle = load_model(str(PROJECT_ROOT / "final_model"))
    blocks = BuildingBlockIndex.from_csv(str(PROJECT_ROOT / "bio_building_block.csv"))
    return (
        RetroEngine(
            bundle["model"], bundle["tokenizer"], blocks,
            num_beams=3, constrained=True, batch_size=16,
        ),
        blocks,
    )


def _try_load_model():
    """Attempt to load the ReactionT5v2 model."""
    try:
        engine, blocks = _load_real_engine()
        return engine, blocks, True
    except Exception as e:
        logger.warning("Model not loaded: %s", e)
        return None, None, False


def _get_building_blocks() -> set:
    """Load building blocks as a set of SMILES."""
    try:
        import csv
        with open(PROJECT_ROOT / "bio_building_block.csv", newline="") as f:
            return {row[0].strip() for row in csv.reader(f) if row and row[0].strip()}
    except Exception:
        return set()


def predict_retrosynthesis(smiles: str, max_depth: int = 5, num_beams: int = 3) -> List[Dict]:
    """
    Run retrosynthesis prediction.
    Uses the real model if available, otherwise uses RDKit-based retrosynthetic rules.
    """
    # Try real model first
    try:
        engine, blocks, loaded = _try_load_model()
        if loaded:
            pathway, stats = engine.predict(smiles, max_depth=max_depth)
            return pathway
    except Exception:
        pass

    # Fallback: RDKit reaction-rule based retrosynthesis
    return _rdkit_retrosynthesis(smiles, max_depth)


def _rdkit_retrosynthesis(
    smiles: str,
    max_depth: int = 5,
    skip_root_template_idx: Optional[int] = None,
    stop_on_building_blocks: bool = True,
    building_blocks: Optional[set] = None,
) -> List[Dict]:
    """
    RDKit-based retrosynthesis using SMARTS reaction templates.
    Supports strategy variations: skipping specific root templates and conditional early termination on building blocks.
    """
    RETRO_TEMPLATES = [
        # ----- Highest priority: heteroatom-containing functional groups -----
        ("[C:1](=[O:2])[O:3][#6:4]", "[C:1](=[O:2])O.[#6:4]O", "ester_hydrolysis"),
        ("[C:1](=[O:2])[NH:3][#6:4]", "[C:1](=[O:2])O.[NH2:3][#6:4]", "amide_hydrolysis"),
        ("[#6:1][O:2][C:3](=[O:4])[N:5]", "[#6:1][O:2].[C:3](=[O:4])=[N:5]", "carbamate_disconnection"),
        ("[#6:1][S:2](=[O:3])(=[O:4])[NH:5][#6:6]",
         "[#6:1][S:2](=[O:3])(=[O:4])Cl.[NH2:5][#6:6]", "sulfonamide_disconnection"),

        # ----- Medium priority: C–heteroatom bonds -----
        ("[#6:1][O:2][#6:3]", "[#6:1][O:2].[#6:3]Br", "ether_disconnection"),
        ("[#6:1][NH:2][#6:3]", "[#6:1]Br.[NH2:2][#6:3]", "CN_disconnection"),
        ("[#6:1][S:2][#6:3]", "[#6:1][S:2].[#6:3]Br", "thioether_disconnection"),

        # ----- Coupling / named reactions -----
        ("[c:1]-[c:2]", "[c:1]Br.[c:2]B(O)O", "suzuki_disconnection"),
        ("[c:1][C:2](=[O:3])[#6:4]", "[cH:1].[C:2](=[O:3])([#6:4])Cl", "friedel_crafts_retro"),
        ("[#6:1][CH:2]([OH:3])[CH2:4][C:5]=[O:6]",
         "[#6:1][CH:2]=[O:3].[CH3:4][C:5]=[O:6]", "aldol_retro"),

        # ----- N-alkyl / heterocycle disconnections -----
        ("[n:1][CH3:2]", "[nH:1].[CH3:2]Br", "N_demethylation"),
        ("[N:1]([#6:2])[CH3:3]", "[NH:1][#6:2].[CH3:3]Br", "N_dealkylation"),
        ("[c:1][N:2]([#6:3])[#6:4]", "[c:1]Br.[NH:2]([#6:3])[#6:4]", "buchwald_hartwig_retro"),

        # ----- Lowest priority: generic C–C -----
        ("[C:1]-[C:2]", "[C:1]Br.[C:2]", "CC_disconnection"),
    ]

    if building_blocks is None:
        building_blocks = _get_building_blocks()
    pathway = []
    frontier = [smiles]
    expanded = set()

    for depth in range(1, max_depth + 1):
        if not frontier:
            break
        next_frontier = []
        for mol_smiles in frontier:
            canon = Chem.MolToSmiles(Chem.MolFromSmiles(mol_smiles)) if Chem.MolFromSmiles(mol_smiles) else mol_smiles
            if canon in expanded:
                continue
            expanded.add(canon)

            if stop_on_building_blocks and canon in building_blocks and depth > 1:
                continue

            mol = Chem.MolFromSmiles(mol_smiles)
            if mol is None:
                continue

            found_reaction = False
            for t_idx, (reactant_smarts, product_smarts, rxn_name) in enumerate(RETRO_TEMPLATES):
                if depth == 1 and skip_root_template_idx is not None and t_idx == skip_root_template_idx:
                    continue
                pattern = Chem.MolFromSmarts(reactant_smarts)
                if pattern is None:
                    continue
                if mol.HasSubstructMatch(pattern):
                    try:
                        rxn_str = f"{reactant_smarts}>>{product_smarts}"
                        rxn = AllChem.ReactionFromSmarts(rxn_str)
                        if rxn is None:
                            continue
                        products = rxn.RunReactants((mol,))
                        if products:
                            for product_set in products:
                                reactant_smiles = []
                                valid = True
                                for p in product_set:
                                    try:
                                        Chem.SanitizeMol(p)
                                        smi = Chem.MolToSmiles(p)
                                        if smi and Chem.MolFromSmiles(smi) is not None:
                                            reactant_smiles.append(smi)
                                        else:
                                            valid = False
                                            break
                                    except Exception:
                                        valid = False
                                        break

                                if valid and reactant_smiles and len(reactant_smiles) >= 1:
                                    if any(r == mol_smiles for r in reactant_smiles):
                                        continue
                                    pathway.append({
                                        "product": mol_smiles,
                                        "reactants": reactant_smiles,
                                        "depth": depth,
                                        "reaction_type": rxn_name,
                                    })
                                    next_frontier.extend(reactant_smiles)
                                    found_reaction = True
                                    break
                            if found_reaction:
                                break
                    except Exception:
                        continue

            if not found_reaction and mol.GetNumHeavyAtoms() > 3:
                reactants = _simple_disconnection(mol)
                if reactants:
                    pathway.append({
                        "product": mol_smiles,
                        "reactants": reactants,
                        "depth": depth,
                        "reaction_type": "bond_disconnection",
                    })
                    next_frontier.extend(reactants)

        frontier = next_frontier

    return pathway


def predict_multiple_routes(smiles: str, max_depth: int = 4, num_beams: int = 3) -> List[Dict]:
    """
    Generate multiple diverse, chemically valid retrosynthetic pathways.
    Strategy 1: Optimal Convergent (Terminates when bio-derived building blocks are reached)
    Strategy 2: Stepwise Deep Route (Explores deeper fragmentation for alternative feedstocks)
    Strategy 3: Alternative Disconnection (Dissects secondary functional group/bond first)
    """
    building_blocks = _get_building_blocks()
    routes = []
    seen = set()

    def pway_sig(p):
        return tuple(sorted((s["product"], tuple(sorted(s["reactants"]))) for s in p))

    # Route 1: Optimal Convergent
    r1 = _rdkit_retrosynthesis(smiles, max_depth=max_depth, stop_on_building_blocks=True, building_blocks=building_blocks)
    if r1:
        seen.add(pway_sig(r1))
        routes.append({
            "route_id": 1,
            "strategy_name": "Optimal Convergent Pathway",
            "strategy_desc": "High-priority functional cleavage, terminating immediately when registered bio-renewable precursors are reached.",
            "pathway": r1,
        })

    # Route 2: Deep Stepwise Catalytic Route
    r2 = _rdkit_retrosynthesis(smiles, max_depth=max(2, max_depth), stop_on_building_blocks=False, building_blocks=building_blocks)
    if r2 and pway_sig(r2) not in seen:
        seen.add(pway_sig(r2))
        routes.append({
            "route_id": len(routes) + 1,
            "strategy_name": "Stepwise Deep Route",
            "strategy_desc": "Deeper chemical fragmentation down to fundamental precursors, maximizing synthetic flexibility.",
            "pathway": r2,
        })

    # Route 3: Alternative Disconnection Strategy (skips primary transform on root)
    for skip_i in range(5):
        r3 = _rdkit_retrosynthesis(smiles, max_depth=max_depth, skip_root_template_idx=skip_i, stop_on_building_blocks=True, building_blocks=building_blocks)
        if r3 and pway_sig(r3) not in seen:
            seen.add(pway_sig(r3))
            routes.append({
                "route_id": len(routes) + 1,
                "strategy_name": "Alternative Disconnection Route",
                "strategy_desc": "Cleaves secondary functional groups first, establishing an orthogonal synthetic sequence.",
                "pathway": r3,
            })
            break

    # If only 1 route, generate a single-stage direct route
    if len(routes) < 2 and max_depth > 1:
        r_direct = _rdkit_retrosynthesis(smiles, max_depth=1, stop_on_building_blocks=False)
        if r_direct and pway_sig(r_direct) not in seen:
            routes.append({
                "route_id": len(routes) + 1,
                "strategy_name": "Direct Single-Stage Route",
                "strategy_desc": "Single-step retrosynthetic disconnection to immediate precursors.",
                "pathway": r_direct,
            })

    return routes


def score_and_rank_routes(diverse_routes: List[Dict], building_blocks: set) -> List[Dict]:
    """Score, enrich, and rank candidate retrosynthesis pathways."""
    raw_pathways = [r["pathway"] for r in diverse_routes]
    scored = compare_routes(raw_pathways, building_blocks)

    enriched = []
    for item in scored:
        idx = item["route_id"] - 1
        meta = diverse_routes[idx] if idx < len(diverse_routes) else {}
        pathway = item["pathway"]

        # Terminal precursors (reactants not produced by any step)
        non_terminal = {s["product"] for s in pathway}
        terminal = set()
        for s in pathway:
            for r in s["reactants"]:
                if r not in non_terminal:
                    terminal.add(r)

        bio_hits = sum(1 for r in terminal if r in building_blocks or (
            Chem.MolFromSmiles(r) and Chem.MolToSmiles(Chem.MolFromSmiles(r)) in building_blocks
        ))
        bio_pct = (bio_hits / max(1, len(terminal))) * 100.0

        # Cumulative yield estimate
        cum_yield = 1.0
        for s in pathway:
            f = item["score"].chemical_feasibility
            cum_yield *= (0.75 + 0.20 * f)
        cum_yield_pct = cum_yield * 100.0

        enriched.append({
            "route_id": item["route_id"],
            "strategy_name": meta.get("strategy_name", f"Pathway #{item['route_id']}"),
            "strategy_desc": meta.get("strategy_desc", "Retrosynthetic transformation sequence."),
            "pathway": pathway,
            "score": item["score"],
            "num_steps": len(pathway),
            "max_depth": max((s["depth"] for s in pathway), default=1),
            "terminal_reactants": terminal,
            "bio_bb_count": bio_hits,
            "bio_coverage_pct": bio_pct,
            "cumulative_yield_pct": cum_yield_pct,
            "overall_score": item["score"].overall_score,
            "feasibility": item["score"].chemical_feasibility,
            "confidence": item["score"].reaction_confidence,
            "sa_score": item["score"].synthetic_accessibility,
        })

    if enriched:
        enriched.sort(key=lambda x: x["overall_score"], reverse=True)
        enriched[0]["award_badge"] = "🏆 Recommended (Optimal Composite)"
        enriched[0]["badge_class"] = "badge-recommended"

        # Tag greenest
        greenest = max(enriched, key=lambda x: x["bio_coverage_pct"])
        if "award_badge" not in greenest:
            greenest["award_badge"] = "🌿 Green Bio-Preferred"
            greenest["badge_class"] = "badge-green"

        # Tag shortest
        shortest = min(enriched, key=lambda x: x["num_steps"])
        if "award_badge" not in shortest:
            shortest["award_badge"] = "⚡ Most Convergent (Fewest Steps)"
            shortest["badge_class"] = "badge-alt"

        for r in enriched:
            if "award_badge" not in r:
                r["award_badge"] = "🧪 Alternative Pathway"
                r["badge_class"] = "badge-alt"

    return enriched


def _simple_disconnection(mol) -> Optional[List[str]]:
    """
    Simple bond disconnection: break the most 'central' non-ring single bond.
    Returns list of fragment SMILES or None.
    """
    if mol is None:
        return None

    try:
        # Find non-ring single bonds
        candidates = []
        for bond in mol.GetBonds():
            if (bond.GetBondType() == Chem.rdchem.BondType.SINGLE
                    and not bond.IsInRing()
                    and not bond.GetBeginAtom().GetSymbol() == "H"
                    and not bond.GetEndAtom().GetSymbol() == "H"):
                # Prefer bonds connecting larger fragments
                candidates.append(bond.GetIdx())

        if not candidates:
            return None

        # Pick the most central bond (closest to middle of molecule)
        num_atoms = mol.GetNumAtoms()
        best_bond = None
        best_balance = float('inf')

        for bond_idx in candidates:
            bond = mol.GetBondWithIdx(bond_idx)
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()

            # Fragment balance: how evenly the bond splits the molecule
            em = Chem.RWMol(mol)
            em.RemoveBond(a1, a2)
            frags = Chem.GetMolFrags(em, asMols=True, sanitizeFrags=False)
            if len(frags) >= 2:
                sizes = [f.GetNumHeavyAtoms() for f in frags]
                balance = abs(sizes[0] - sizes[1]) if len(sizes) == 2 else sum(sizes)
                if balance < best_balance:
                    best_balance = balance
                    best_bond = bond_idx

        if best_bond is None:
            return None

        bond = mol.GetBondWithIdx(best_bond)
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()

        em = Chem.RWMol(mol)
        em.RemoveBond(a1, a2)
        frags = Chem.GetMolFrags(em, asMols=True, sanitizeFrags=False)

        result = []
        for frag in frags:
            try:
                Chem.SanitizeMol(frag)
                smi = Chem.MolToSmiles(frag)
                if smi and Chem.MolFromSmiles(smi) is not None:
                    result.append(smi)
            except Exception:
                pass

        return result if len(result) >= 2 else None

    except Exception:
        return None


# ============================================================================
#  VISUALIZATION: Interactive Retrosynthesis Tree (Plotly)
# ============================================================================

def build_tree_graph(pathway: List[Dict], building_blocks: set) -> go.Figure:
    """
    Build an interactive retrosynthesis tree using Plotly.
    Nodes are colored by type (target / intermediate / building block).
    Edges show reaction info on hover.
    """
    if not pathway:
        return go.Figure()

    G = nx.DiGraph()
    node_types = {}
    node_scores = {}

    target = pathway[0]["product"] if pathway else ""
    node_types[target] = "target"

    for step in pathway:
        product = step["product"]
        if product not in node_types:
            node_types[product] = "intermediate"

        # Score this step
        try:
            step_score = score_reaction_step(product, step["reactants"])
        except Exception:
            step_score = RouteScore()
        node_scores[product] = step_score

        for reactant in step["reactants"]:
            if reactant not in node_types:
                canon = Chem.MolToSmiles(Chem.MolFromSmiles(reactant)) if Chem.MolFromSmiles(reactant) else reactant
                if canon in building_blocks or reactant in building_blocks:
                    node_types[reactant] = "building_block"
                else:
                    node_types[reactant] = "intermediate"
            G.add_edge(product, reactant)

    # Compute layout using Graphviz-like hierarchical layout
    try:
        pos = _hierarchical_layout(G, target)
    except Exception:
        pos = nx.spring_layout(G, k=2.0, iterations=100, seed=42)

    # Create Plotly traces
    edge_x, edge_y = [], []
    edge_hover = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2.5, color='#94a3b8'),
        hoverinfo='none',
        mode='lines',
        showlegend=False,
    )

    # Arrow annotations
    annotations = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        annotations.append(dict(
            ax=x0, ay=y0, x=x1, y=y1,
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True,
            arrowhead=3,
            arrowsize=1.2,
            arrowwidth=2,
            arrowcolor='#94a3b8',
            opacity=0.7,
        ))

    # Separate node traces by type for legend
    type_configs = {
        "target": {"color": "#667eea", "symbol": "diamond", "size": 28, "name": "Target Molecule"},
        "intermediate": {"color": "#f59e0b", "symbol": "circle", "size": 22, "name": "Intermediate"},
        "building_block": {"color": "#10b981", "symbol": "square", "size": 20, "name": "Building Block"},
    }

    node_traces = []
    for ntype, config in type_configs.items():
        nodes = [n for n, t in node_types.items() if t == ntype]
        if not nodes:
            continue
        nx_list = [pos[n][0] for n in nodes]
        ny_list = [pos[n][1] for n in nodes]

        hover_texts = []
        display_labels = []
        for n in nodes:
            label = n if len(n) <= 25 else n[:22] + "..."
            display_labels.append(label)
            props = compute_molecular_properties(n)
            hover = f"<b>{n}</b><br>"
            if props.get("valid"):
                hover += f"MW: {props.get('molecular_weight', '—')}<br>"
                hover += f"Formula: {props.get('molecular_formula', '—')}<br>"
                hover += f"LogP: {props.get('logp', '—')}<br>"
                hover += f"SA: {props.get('synthetic_accessibility', '—')}<br>"
            if n in node_scores:
                s = node_scores[n]
                hover += f"<br><b>Step Score:</b> {s.overall_score:.2f}<br>"
                hover += f"Confidence: {s.reaction_confidence:.2f}<br>"
                hover += f"Feasibility: {s.chemical_feasibility:.2f}"
            hover_texts.append(hover)

        trace = go.Scatter(
            x=nx_list, y=ny_list,
            mode='markers+text',
            marker=dict(
                size=config["size"],
                color=config["color"],
                symbol=config["symbol"],
                line=dict(width=2, color='white'),
                opacity=0.92,
            ),
            text=display_labels,
            textposition="bottom center",
            textfont=dict(size=9, color='#374151'),
            hovertext=hover_texts,
            hoverinfo='text',
            name=config["name"],
        )
        node_traces.append(trace)

    fig = go.Figure(data=[edge_trace] + node_traces)
    fig.update_layout(
        title=dict(
            text="Interactive Retrosynthesis Search Tree",
            font=dict(size=20, color='#ffffff'),
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(size=12),
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(248,250,252,1)',
        paper_bgcolor='rgba(248,250,252,1)',
        height=600,
        margin=dict(l=20, r=20, t=80, b=20),
        annotations=annotations,
        hovermode='closest',
    )
    return fig


def _hierarchical_layout(G: nx.DiGraph, root: str) -> Dict:
    """Compute a top-down hierarchical layout."""
    if root not in G:
        return nx.spring_layout(G, seed=42)

    levels = {}
    visited = set()
    queue = [(root, 0)]
    visited.add(root)

    while queue:
        node, level = queue.pop(0)
        levels[node] = level
        for child in G.successors(node):
            if child not in visited:
                visited.add(child)
                queue.append((child, level + 1))

    # For nodes not reachable from root
    for node in G.nodes():
        if node not in levels:
            levels[node] = max(levels.values()) + 1 if levels else 0

    # Group by level
    level_groups = defaultdict(list)
    for node, level in levels.items():
        level_groups[level].append(node)

    max_width = max(len(v) for v in level_groups.values()) if level_groups else 1
    pos = {}
    for level, nodes in level_groups.items():
        width = len(nodes)
        for i, node in enumerate(nodes):
            x = (i - (width - 1) / 2.0) * (max_width / max(width, 1)) * 1.5
            y = -level * 2.0
            pos[node] = (x, y)

    return pos


# ============================================================================
#  VISUALIZATION: 3D Molecular Conformer (Plotly)
# ============================================================================

# Atom color mapping (CPK-like)
ATOM_COLORS = {
    "H": "#FFFFFF", "C": "#404040", "N": "#3050F8", "O": "#FF0D0D",
    "F": "#90E050", "Cl": "#1FF01F", "Br": "#A62929", "I": "#940094",
    "S": "#FFFF30", "P": "#FF8000", "B": "#FFB5B5", "Si": "#DAA520",
    "Na": "#AB5CF2", "K": "#8F40D4", "Ca": "#3DFF00", "Fe": "#E06633",
    "Mg": "#8AFF00", "Zn": "#7D80B0", "Cu": "#C88033",
}

ATOM_RADII = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "Cl": 1.02, "Br": 1.20, "I": 1.39, "S": 1.05, "P": 1.07,
}


def build_3d_molecule_view(smiles: str, highlight_atoms: Optional[Set[int]] = None,
                            title: str = "3D Molecular Structure") -> Optional[go.Figure]:
    """Build a 3D Plotly visualization of a molecule's conformer."""
    conf_data = generate_3d_conformer(smiles)
    if conf_data is None:
        return None

    atoms = conf_data["atoms"]
    bonds = conf_data["bonds"]

    # Filter to non-hydrogen for cleaner view (but keep coords)
    heavy_atoms = [a for a in atoms if a["symbol"] != "H"]
    heavy_idx_set = {a["idx"] for a in heavy_atoms}

    # Atom trace
    x = [a["x"] for a in heavy_atoms]
    y = [a["y"] for a in heavy_atoms]
    z = [a["z"] for a in heavy_atoms]
    colors = []
    sizes = []
    hover_texts = []

    for a in heavy_atoms:
        is_highlighted = highlight_atoms and a["idx"] in highlight_atoms
        color = "#FF4444" if is_highlighted else ATOM_COLORS.get(a["symbol"], "#808080")
        colors.append(color)
        radius = ATOM_RADII.get(a["symbol"], 0.8) * 12
        sizes.append(radius if not is_highlighted else radius * 1.4)
        hover = f"<b>{a['symbol']}</b> (atom {a['idx']})<br>"
        hover += f"Position: ({a['x']:.2f}, {a['y']:.2f}, {a['z']:.2f})<br>"
        if a["is_aromatic"]:
            hover += "Aromatic<br>"
        if a["in_ring"]:
            hover += "In ring<br>"
        if a["formal_charge"] != 0:
            hover += f"Charge: {a['formal_charge']:+d}<br>"
        hover_texts.append(hover)

    atom_trace = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers+text',
        marker=dict(
            size=sizes,
            color=colors,
            opacity=0.95,
            line=dict(width=1, color='#1e293b'),
        ),
        text=[a["symbol"] for a in heavy_atoms],
        textfont=dict(size=10, color='white'),
        textposition="middle center",
        hovertext=hover_texts,
        hoverinfo='text',
        name='Atoms',
    )

    # Bond traces
    bond_traces = []
    for bond in bonds:
        if bond["begin_atom"] not in heavy_idx_set or bond["end_atom"] not in heavy_idx_set:
            continue
        a1 = next(a for a in atoms if a["idx"] == bond["begin_atom"])
        a2 = next(a for a in atoms if a["idx"] == bond["end_atom"])

        color = "#94a3b8"
        width = 3
        if bond["is_aromatic"]:
            color = "#667eea"
            width = 4
        elif bond["bond_type"] == "DOUBLE":
            width = 5
            color = "#475569"
        elif bond["bond_type"] == "TRIPLE":
            width = 7
            color = "#1e293b"

        if highlight_atoms:
            if bond["begin_atom"] in highlight_atoms and bond["end_atom"] in highlight_atoms:
                color = "#FF4444"
                width = width + 2

        bond_traces.append(go.Scatter3d(
            x=[a1["x"], a2["x"]], y=[a1["y"], a2["y"]], z=[a1["z"], a2["z"]],
            mode='lines',
            line=dict(width=width, color=color),
            hoverinfo='none',
            showlegend=False,
        ))

    fig = go.Figure(data=bond_traces + [atom_trace])

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#ffffff')),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor='rgba(248,250,252,1)',
            camera=dict(
                up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5),
            ),
        ),
        paper_bgcolor='rgba(248,250,252,1)',
        height=500,
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
    )
    return fig


# ============================================================================
#  VISUALIZATION: Route Scoring Dashboard
# ============================================================================

def render_route_score_dashboard(route_score: RouteScore, route_id: int = 1):
    """Render a comprehensive route score dashboard."""
    scores = route_score.as_dict()

    st.markdown(f"""
    <div class="score-card">
        <h4>Route #{route_id} — Overall Score</h4>
        <span class="score-value">{scores['overall_score']:.2f}</span>
    </div>
    """, unsafe_allow_html=True)

    # Score breakdown
    metrics = [
        ("Reaction Confidence", scores["reaction_confidence"], "#667eea"),
        ("Synthetic Accessibility", scores["synthetic_accessibility"], "#10b981"),
        ("Chemical Feasibility", scores["chemical_feasibility"], "#3b82f6"),
        ("Route Diversity", scores["route_diversity"], "#8b5cf6"),
        ("Uncertainty", scores["uncertainty"], "#ef4444"),
        ("Estimated Cost", scores["estimated_cost"], "#f59e0b"),
    ]

    cols = st.columns(3)
    for i, (name, value, color) in enumerate(metrics):
        with cols[i % 3]:
            pct = int(value * 100)
            st.markdown(f"""
            <div style="margin: 8px 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.85rem; font-weight: 500; color: #374151;">{name}</span>
                    <span style="font-size: 0.85rem; font-weight: 700; color: {color};">{value:.2f}</span>
                </div>
                <div class="score-bar-container">
                    <div class="score-bar-fill" style="width: {pct}%; background: {color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================================
#  VISUALIZATION: Reaction Center
# ============================================================================

def render_reaction_center(step: Dict, idx: int):
    """Render the reaction-center analysis for a single step."""
    product = step["product"]
    reactants = step["reactants"]

    rc = detect_reaction_center(product, reactants)

    st.markdown(f'<span class="reaction-class">{rc.reaction_class}</span>', unsafe_allow_html=True)
    st.caption(rc.transformed_bonds_summary)

    # Bond changes visualization
    col1, col2 = st.columns(2)
    with col1:
        if rc.broken_bonds:
            st.markdown("**Broken Bonds:**")
            for b in rc.broken_bonds:
                st.markdown(
                    f'<span class="bond-broken">✂ {b.label()}</span>',
                    unsafe_allow_html=True
                )
        else:
            st.info("No specific broken bonds identified")

    with col2:
        if rc.formed_bonds:
            st.markdown("**Formed Bonds:**")
            for f in rc.formed_bonds:
                st.markdown(
                    f'<span class="bond-formed">🔗 {f.label()}</span>',
                    unsafe_allow_html=True
                )
        else:
            st.caption("No new bonds formed in precursors")

    # Product with highlighted reaction center
    col_prod, col_arrow, col_react = st.columns([2, 1, 3])

    with col_prod:
        st.markdown("**Product**")
        render_molecule_image(
            product, caption=product[:30],
            size=250,
            highlight_atoms=rc.reaction_center_atoms if rc.reaction_center_atoms else None
        )

    with col_arrow:
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("<div style='text-align:center; font-size:2rem; color:#667eea; padding-top:40px;'>⟶</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center;'><span class='info-chip'>{rc.reaction_class}</span></div>", unsafe_allow_html=True)

    with col_react:
        st.markdown("**Precursors**")
        react_cols = st.columns(min(len(reactants), 3))
        for j, reactant in enumerate(reactants):
            with react_cols[j % len(react_cols)]:
                render_molecule_image(reactant, caption=reactant[:20], size=200)

    # Explanation
    with st.expander(f"🔍 Why this disconnection? (Step {idx})"):
        st.write(rc.explanation())
        if rc.reaction_center_atoms:
            st.write(f"**Reaction center atoms (product indices):** {sorted(rc.reaction_center_atoms)}")
        st.write(f"**Number of broken bonds:** {len(rc.broken_bonds)}")
        st.write(f"**Number of formed bonds:** {len(rc.formed_bonds)}")


# ============================================================================
#  VISUALIZATION: Multi-Route Comparison
# ============================================================================

def render_route_comparison(scored_routes: List[Dict], target_smiles: str = "", building_blocks: Optional[set] = None):
    """
    Render a comprehensive, state-of-the-art Multi-Route Analysis & Strategic Decision Platform.
    Includes:
    - Executive AI Strategic Synthesis Brief
    - Side-by-Side Route KPI Scorecards & Badges
    - Multi-Objective Polar Radar Plot (Dark Theme)
    - Grouped Analytical Metric Bar Charts (Yield, Steps, Bio-feedstock, Feasibility)
    - Interactive Side-by-Side Step-by-Step Pathway Inspector with 2D structures
    - Complete Analytical Decision Matrix (Table)
    - CSV & JSON Export
    """
    if not scored_routes:
        st.info("No comparative routes available.")
        return

    if building_blocks is None:
        building_blocks = _get_building_blocks()

    # ----------------------------------------------------------------
    #  1. EXECUTIVE AI SYNTHESIS STRATEGIC BRIEF
    # ----------------------------------------------------------------
    best_route = scored_routes[0]
    greenest_route = max(scored_routes, key=lambda r: r.get("bio_coverage_pct", 0.0))
    shortest_route = min(scored_routes, key=lambda r: r.get("num_steps", 999))

    brief_html = f"""
    <div class="exec-brief-box">
        <div style="font-size:1.15rem; font-weight:700; color:#ffffff; margin-bottom:6px;">
            💡 Executive Synthesis Strategic Brief
        </div>
        <div style="font-size:0.92rem; color:#cbd5e1; line-height:1.6;">
            • <b>Primary Recommendation:</b> <b>Route #{best_route['route_id']} ({best_route['strategy_name']})</b> achieves the top multi-objective score of <b>{best_route['overall_score']:.2f}</b>, combining high feasibility (<b>{best_route['feasibility']:.2f}</b>) and an estimated cumulative yield of <b>{best_route['cumulative_yield_pct']:.1f}%</b> across <b>{best_route['num_steps']} reaction step(s)</b>.<br>
            • <b>Bio-Renewability & Green Footprint:</b> <b>Route #{greenest_route['route_id']}</b> maximizes sustainability with <b>{greenest_route['bio_coverage_pct']:.0f}%</b> of starting precursors sourced directly from biological/metabolic feedstocks ({greenest_route['bio_bb_count']} bio-building block(s)).<br>
            • <b>Directness & Scalability:</b> <b>Route #{shortest_route['route_id']}</b> delivers the fastest laboratory conversion with the fewest synthetic transformations (<b>{shortest_route['num_steps']} step(s)</b>).
        </div>
    </div>
    """
    st.markdown(brief_html, unsafe_allow_html=True)

    # ----------------------------------------------------------------
    #  2. SIDE-BY-SIDE ROUTE KPI SCORECARDS
    # ----------------------------------------------------------------
    st.markdown("#### 📊 Candidate Route Scorecards")
    route_cols = st.columns(min(len(scored_routes), 3))

    for idx, r_data in enumerate(scored_routes[:3]):
        with route_cols[idx]:
            badge_html = f'<span class="{r_data.get("badge_class", "badge-recommended")}">{r_data.get("award_badge", "Candidate Route")}</span>'
            st.markdown(badge_html, unsafe_allow_html=True)
            st.markdown(f"**Route #{r_data['route_id']}: {r_data['strategy_name']}**")
            st.caption(r_data.get("strategy_desc", ""))

            # Score Progress Bar
            ov_score = float(r_data.get("overall_score", 0.0))
            st.progress(min(1.0, max(0.0, ov_score)), text=f"Overall Composite Score: {ov_score:.2f}")

            # Sub-metrics grid
            m1, m2 = st.columns(2)
            m1.metric("Steps", f"{r_data['num_steps']}", help="Total number of retrosynthetic steps")
            m2.metric("Est. Yield", f"{r_data.get('cumulative_yield_pct', 0.0):.1f}%", help="Compounded estimated reaction yield")

            m3, m4 = st.columns(2)
            m3.metric("Bio-Feedstock", f"{r_data.get('bio_coverage_pct', 0.0):.0f}%", help="Percentage of starting materials in bio-building block registry")
            m4.metric("Feasibility", f"{r_data.get('feasibility', 0.0):.2f}", help="Average reaction feasibility score (0-1)")

            # Starting Precursors Chips
            st.markdown("<div style='font-size:0.8rem; font-weight:600; color:#94a3b8; margin-top:8px;'>STARTING PRECURSORS:</div>", unsafe_allow_html=True)
            term_smi = list(r_data.get("terminal_reactants", []))
            chips_html = " ".join(
                f'<span class="info-chip" title="{smi}">'
                f'{"🌿 " if (smi in building_blocks or (Chem.MolFromSmiles(smi) and Chem.MolToSmiles(Chem.MolFromSmiles(smi)) in building_blocks)) else "🧪 "}'
                f'{smi[:15]}...</span>'
                for smi in term_smi[:4]
            )
            st.markdown(chips_html if chips_html else "<span style='color:#64748b;'>None</span>", unsafe_allow_html=True)

    st.markdown("---")

    # ----------------------------------------------------------------
    #  3. MULTI-OBJECTIVE RADAR & TRADE-OFF VISUALIZATIONS
    # ----------------------------------------------------------------
    col_radar, col_bars = st.columns([1, 1])

    with col_radar:
        st.markdown("#### 🎯 Multi-Objective Trade-Off (Spider / Radar)")
        categories = ["Feasibility", "Bio-Feedstock", "Est. Yield", "Step Efficiency", "1 - Complexity", "Confidence"]
        fig_radar = go.Figure()
        palette = ["#818cf8", "#10b981", "#f59e0b", "#ec4899", "#38bdf8"]

        for i, rd in enumerate(scored_routes[:5]):
            s = rd["score"]
            step_eff = 1.0 / (rd["num_steps"] + 0.5)
            bio_norm = rd.get("bio_coverage_pct", 0.0) / 100.0
            yield_norm = rd.get("cumulative_yield_pct", 0.0) / 100.0
            sa_norm = max(0.0, 1.0 - (s.synthetic_accessibility / 5.0))

            vals = [
                s.chemical_feasibility,
                bio_norm,
                yield_norm,
                step_eff,
                sa_norm,
                s.reaction_confidence,
            ]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name=f"Route #{rd['route_id']} ({rd['strategy_name'][:14]})",
                line=dict(color=palette[i % len(palette)], width=2),
                opacity=0.65,
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], color='#94a3b8', gridcolor='#334155'),
                angularaxis=dict(color='#ffffff', gridcolor='#334155'),
                bgcolor='rgba(15,23,42,0.8)',
            ),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(color='#ffffff', size=11)),
            height=430,
            paper_bgcolor='#0e1117',
            margin=dict(l=35, r=35, t=30, b=50),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_bars:
        st.markdown("#### 📈 Key Metrics Direct Comparison")
        route_names = [f"Route #{rd['route_id']}" for rd in scored_routes]
        fig_bar = go.Figure()

        # Step Count
        fig_bar.add_trace(go.Bar(
            name='Reaction Steps',
            x=route_names,
            y=[rd['num_steps'] for rd in scored_routes],
            marker_color='#818cf8',
        ))
        # Cumulative Yield (%)
        fig_bar.add_trace(go.Bar(
            name='Est. Yield (%)',
            x=route_names,
            y=[rd.get('cumulative_yield_pct', 0.0) for rd in scored_routes],
            marker_color='#10b981',
        ))
        # Bio-Feedstock (%)
        fig_bar.add_trace(go.Bar(
            name='Bio-Feedstock (%)',
            x=route_names,
            y=[rd.get('bio_coverage_pct', 0.0) for rd in scored_routes],
            marker_color='#f59e0b',
        ))
        # Feasibility x 100
        fig_bar.add_trace(go.Bar(
            name='Feasibility Score (x100)',
            x=route_names,
            y=[rd.get('feasibility', 0.0) * 100.0 for rd in scored_routes],
            marker_color='#38bdf8',
        ))

        fig_bar.update_layout(
            barmode='group',
            height=430,
            paper_bgcolor='#0e1117',
            plot_bgcolor='rgba(15,23,42,0.8)',
            font=dict(color='#ffffff'),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(color='#ffffff', size=11)),
            yaxis=dict(gridcolor='#334155', title="Value / Percentage"),
            xaxis=dict(gridcolor='#334155'),
            margin=dict(l=30, r=20, t=30, b=50),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ----------------------------------------------------------------
    #  4. INTERACTIVE STEP-BY-STEP PATHWAY INSPECTOR
    # ----------------------------------------------------------------
    st.markdown("#### 🔍 Interactive Step-by-Step Pathway Inspector")
    route_labels = [
        f"Route #{r['route_id']}: {r['strategy_name']} ({r.get('award_badge', '').split()[0]})"
        for r in scored_routes
    ]
    selected_idx = st.selectbox(
        "Choose Route to inspect chemical transformations:",
        options=range(len(scored_routes)),
        format_func=lambda i: route_labels[i],
        key="inspect_route_select",
    )

    chosen_route = scored_routes[selected_idx]
    pathway_steps = chosen_route["pathway"]

    st.markdown(f"**Inspecting {chosen_route['strategy_name']}** — *{chosen_route.get('strategy_desc', '')}*")

    for step_num, step in enumerate(pathway_steps, 1):
        with st.container():
            st.markdown(f"""
            <div style="background:rgba(30,41,59,0.5); border:1px solid rgba(148,163,184,0.15); border-radius:12px; padding:14px 18px; margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:700; color:#ffffff; font-size:1.05rem;">Step {step_num} of {len(pathway_steps)}: <span style="color:#818cf8;">{step.get('reaction_type', 'transformation')}</span></span>
                    <span class="info-chip">Depth: {step.get('depth', 1)}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            c_prod, c_arrow, c_react = st.columns([1.2, 0.6, 2.2])
            with c_prod:
                st.caption("Product")
                render_molecule_image(step["product"], caption=step["product"][:22], size=220, use_container_width=True)

            with c_arrow:
                st.markdown("<div style='text-align:center; font-size:2rem; color:#818cf8; padding-top:40px;'>⟶</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align:center;'><span class='reaction-class'>{step.get('reaction_type', 'reaction')}</span></div>", unsafe_allow_html=True)

            with c_react:
                st.caption(f"Precursors ({len(step['reactants'])})")
                r_cols = st.columns(min(len(step["reactants"]), 3))
                for r_i, reactant in enumerate(step["reactants"]):
                    with r_cols[r_i % len(r_cols)]:
                        render_molecule_image(reactant, caption=reactant[:18], size=180, use_container_width=True)
                        is_bb = reactant in building_blocks or (
                            Chem.MolFromSmiles(reactant) and Chem.MolToSmiles(Chem.MolFromSmiles(reactant)) in building_blocks
                        )
                        if is_bb:
                            st.markdown("<span style='color:#10b981; font-size:0.78rem; font-weight:600;'>🌿 Bio-Building Block</span>", unsafe_allow_html=True)
                        else:
                            st.markdown("<span style='color:#94a3b8; font-size:0.78rem;'>🧪 Intermediate</span>", unsafe_allow_html=True)

    # ----------------------------------------------------------------
    #  5. COMPREHENSIVE ROUTE DECISION MATRIX (Table)
    # ----------------------------------------------------------------
    st.markdown("#### 📋 Route Comparison Decision Matrix")
    matrix_rows = []
    for rd in scored_routes:
        s = rd["score"]
        matrix_rows.append({
            "Route": f"Route #{rd['route_id']}",
            "Strategy": rd["strategy_name"],
            "Rank Badge": rd.get("award_badge", ""),
            "Overall Score": f"{rd.get('overall_score', s.overall_score):.3f}",
            "Steps": rd["num_steps"],
            "Est. Cumulative Yield": f"{rd.get('cumulative_yield_pct', 0.0):.1f}%",
            "Bio-Feedstock %": f"{rd.get('bio_coverage_pct', 0.0):.0f}%",
            "Chemical Feasibility": f"{s.chemical_feasibility:.3f}",
            "Reaction Confidence": f"{s.reaction_confidence:.3f}",
            "SA Complexity": f"{s.synthetic_accessibility:.3f}",
            "Starting Precursors Count": len(rd.get("terminal_reactants", [])),
        })

    matrix_df = pd.DataFrame(matrix_rows)
    st.dataframe(matrix_df, use_container_width=True, hide_index=True)

    # ----------------------------------------------------------------
    #  6. DOWNLOAD & EXPORT SUITE
    # ----------------------------------------------------------------
    c_csv, c_json = st.columns([1, 1])
    with c_csv:
        csv_data = matrix_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Route Comparison (CSV)",
            data=csv_data,
            file_name="multi_route_comparison_matrix.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c_json:
        json_export = json.dumps([
            {
                "route_id": rd["route_id"],
                "strategy": rd["strategy_name"],
                "badge": rd.get("award_badge", ""),
                "overall_score": rd.get("overall_score", 0.0),
                "steps": rd["num_steps"],
                "cumulative_yield_pct": rd.get("cumulative_yield_pct", 0.0),
                "bio_coverage_pct": rd.get("bio_coverage_pct", 0.0),
                "terminal_precursors": list(rd.get("terminal_reactants", [])),
                "pathway": rd["pathway"],
            }
            for rd in scored_routes
        ], indent=2).encode('utf-8')
        st.download_button(
            "📥 Download Full Multi-Route Diagnostics (JSON)",
            data=json_export,
            file_name="multi_route_diagnostics.json",
            mime="application/json",
            use_container_width=True,
        )


# ============================================================================
#  MAIN APPLICATION
# ============================================================================

# Header
st.markdown('<div class="main-title">🔬 BioRetrosynthesis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Advanced Retrosynthesis Visualization & Planning • Interactive Search Tree • 3D Molecular Conformers • Reaction-Center Analysis • Multi-Route Scoring</div>', unsafe_allow_html=True)

# ============================================================================
#  SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    st.markdown("**🧪 Input**")
    st.caption("Supports: SMILES, InChI, Chemical Names, CAS #, Molecular Formulas")

    st.markdown("**🔧 Search Parameters**")
    max_depth = st.slider("Max recursion depth", 1, 8, 4, help="Maximum tree depth")
    num_beams = st.slider("Beam width", 1, 10, 3, help="Predictions per level")

    st.markdown("**📊 Visualization**")
    show_3d = st.checkbox("3D Molecular Conformers", value=True)
    show_reaction_centers = st.checkbox("Reaction-Center Analysis", value=True)
    show_route_scoring = st.checkbox("Multi-Objective Route Scoring", value=True)
    show_properties = st.checkbox("Detailed Molecular Properties", value=True)

    st.markdown("---")
    st.markdown("**🎨 Theme**")
    dark_mode = st.checkbox("Dark mode for 3D viewer", value=False)

    st.markdown("---")
    st.caption("**BioRetrosynthesis**")
    st.caption("Developed by Mayur Nhavalde")

# ============================================================================
#  MAIN INPUT
# ============================================================================

col_input, col_btn = st.columns([4, 1])

with col_input:
    user_input = st.text_input(
        "Enter target molecule",
        placeholder="SMILES: CC(=O)Oc1ccccc1C(=O)O  |  Name: aspirin  |  CAS: 50-78-2  |  Formula: C9H8O4",
        key="molecule_input",
        label_visibility="collapsed",
    )

with col_btn:
    predict_btn = st.button("🚀 Predict Retrosynthesis", type="primary", use_container_width=True)

# Quick examples
st.markdown("**Quick examples:**")
example_cols = st.columns(6)
examples = [
    ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("Ibuprofen", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"),
    ("Caffeine", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"),
    ("Glucose", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"),
    ("Ethanol", "CCO"),
    ("Acetic acid", "CC(=O)O"),
]
for i, (name, smi) in enumerate(examples):
    with example_cols[i]:
        if st.button(name, key=f"ex_{i}", use_container_width=True):
            st.session_state["molecule_input"] = smi
            st.rerun()


# ============================================================================
#  PREDICTION & ANALYSIS
# ============================================================================

if predict_btn or (user_input and st.session_state.get("_last_input") != user_input):
    if user_input and user_input.strip():
        st.session_state["_last_input"] = user_input

        # Step 1: Resolve input
        with st.spinner("🔄 Resolving molecular input..."):
            smiles, source_format, error = resolve_to_smiles(user_input.strip())

        if error or not smiles:
            st.error(f"❌ {error}")
            st.stop()

        st.success(f"✅ Resolved as **{source_format}** → `{smiles}`")

        # ----------------------------------------------------------------
        #  INPUT MOLECULE ANALYSIS
        # ----------------------------------------------------------------
        st.markdown('<div class="section-header">📊 Target Molecule Analysis</div>', unsafe_allow_html=True)

        props = compute_molecular_properties(smiles)

        col_img, col_props = st.columns([1, 2.6])
        with col_img:
            render_molecule_image(smiles, caption="Target Molecule", size=300, use_container_width=True)

        with col_props:
            if props.get("valid"):
                formula_sub = to_subscript(props.get("molecular_formula", ""))
                p1, p2, p3, p4 = st.columns(4)
                p1.metric(
                    "MW",
                    f"{props['molecular_weight']:.1f} Da",
                    help=f"Exact Molecular Weight: {props['molecular_weight']} Da (g/mol)"
                )
                p2.metric(
                    "LogP",
                    f"{props['logp']:.2f}",
                    help=f"Calculated Octanol-Water Partition Coefficient (LogP): {props['logp']}"
                )
                p3.metric(
                    "HBD / HBA",
                    f"{props['hbd']} / {props['hba']}",
                    help=f"Hydrogen Bond Donors: {props['hbd']} | Hydrogen Bond Acceptors: {props['hba']}"
                )
                p4.metric(
                    "Formula",
                    formula_sub if formula_sub else props.get("molecular_formula", ""),
                    help=f"Full Molecular Formula: {props['molecular_formula']}"
                )

                p5, p6, p7, p8 = st.columns(4)
                p5.metric(
                    "TPSA",
                    f"{props['tpsa']:.1f} Å²",
                    help=f"Topological Polar Surface Area: {props['tpsa']} Å²"
                )
                p6.metric(
                    "Rot. Bonds",
                    str(props["rotatable_bonds"]),
                    help=f"Number of Rotatable Bonds: {props['rotatable_bonds']}"
                )
                p7.metric(
                    "Rings",
                    f"{props['num_rings']} ({props['num_aromatic_rings']} ar)",
                    help=f"Total Rings: {props['num_rings']} (Aromatic: {props['num_aromatic_rings']})"
                )
                p8.metric(
                    "SA Score",
                    f"{props['synthetic_accessibility']:.2f}",
                    help=f"Synthetic Accessibility Score: {props['synthetic_accessibility']:.2f} (0=easy, 1=complex)"
                )

                if props.get("functional_groups"):
                    st.markdown("**Functional Groups:**")
                    fg_html = " ".join(
                        f'<span class="info-chip">{name} ×{count}</span>'
                        for name, count in props["functional_groups"].items()
                    )
                    st.markdown(fg_html, unsafe_allow_html=True)

                # Lipinski check
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    violations = []
                    if props["molecular_weight"] > 500:
                        violations.append(f"MW={props['molecular_weight']}")
                    if props["logp"] > 5:
                        violations.append(f"LogP={props['logp']}")
                    if props["hbd"] > 5:
                        violations.append(f"HBD={props['hbd']}")
                    if props["hba"] > 10:
                        violations.append(f"HBA={props['hba']}")
                    if not violations:
                        st.success("✅ Lipinski's Rule of Five: Compliant")
                    else:
                        st.warning(f"⚠️ Lipinski violations: {', '.join(violations)}")

                with st.expander("🔍 Click to view Full Molecule Information & Exact Specs", expanded=False):
                    m_col1, m_col2 = st.columns(2)
                    with m_col1:
                        st.markdown(f"**Full Formula:** `{props.get('molecular_formula', '')}`")
                        st.markdown(f"**Exact Molecular Weight:** `{props.get('molecular_weight', '')} Da`")
                        st.markdown(f"**LogP (Lipophilicity):** `{props.get('logp', '')}`")
                        st.markdown(f"**TPSA:** `{props.get('tpsa', '')} Å²`")
                    with m_col2:
                        st.markdown(f"**H-Bond Donors (HBD):** `{props.get('hbd', '')}`")
                        st.markdown(f"**H-Bond Acceptors (HBA):** `{props.get('hba', '')}`")
                        st.markdown(f"**Rings:** `{props.get('num_rings', '')} total ({props.get('num_aromatic_rings', '')} aromatic)`")
                        st.markdown(f"**SA Score:** `{props.get('synthetic_accessibility', 0.0):.4f}`")
                    st.markdown(f"**Canonical SMILES:** `{smiles}`")

        # ----------------------------------------------------------------
        #  3D CONFORMER (Target)
        # ----------------------------------------------------------------
        if show_3d:
            st.markdown('<div class="section-header">🧬 3D Molecular Structure</div>', unsafe_allow_html=True)
            fig_3d = build_3d_molecule_view(smiles, title=f"3D Conformer — {smiles[:30]}")
            if fig_3d:
                if dark_mode:
                    fig_3d.update_layout(
                        scene=dict(bgcolor='#1e293b'),
                        paper_bgcolor='#0f172a',
                        title=dict(font=dict(color='white')),
                    )
                st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.warning("Could not generate 3D conformer for this molecule.")

        # ----------------------------------------------------------------
        #  RETROSYNTHESIS PREDICTION
        # ----------------------------------------------------------------
        st.markdown('<div class="section-header">🌳 Retrosynthesis Prediction</div>', unsafe_allow_html=True)

        with st.spinner("⏳ Running retrosynthesis analysis (this may take a moment)..."):
            start_time = time.time()
            pathway = predict_retrosynthesis(smiles, max_depth=max_depth, num_beams=num_beams)
            elapsed = time.time() - start_time

        if not pathway:
            st.warning("⚠️ No retrosynthetic disconnections found for this molecule. Try a more complex target.")
            st.stop()

        st.success(f"✅ Found **{len(pathway)} retrosynthetic step(s)** in {elapsed:.2f}s")

        # Stats
        stat_cols = st.columns(4)
        all_reactants = set()
        for step in pathway:
            all_reactants.update(step["reactants"])
        building_blocks = _get_building_blocks()
        bb_count = sum(1 for r in all_reactants if r in building_blocks or (
            Chem.MolFromSmiles(r) and Chem.MolToSmiles(Chem.MolFromSmiles(r)) in building_blocks
        ))

        stat_cols[0].metric("Steps", len(pathway))
        stat_cols[1].metric("Unique Intermediates", len(all_reactants))
        stat_cols[2].metric("Building Blocks Hit", bb_count)
        stat_cols[3].metric("Max Depth", max(s["depth"] for s in pathway))

        # ----------------------------------------------------------------
        #  INTERACTIVE RETROSYNTHESIS TREE
        # ----------------------------------------------------------------
        st.markdown('<div class="section-header">🗺️ Interactive Retrosynthesis Search Tree</div>', unsafe_allow_html=True)
        st.caption("Hover over nodes for molecular properties • Diamonds = target • Circles = intermediates • Squares = building blocks")

        tree_fig = build_tree_graph(pathway, building_blocks)
        st.plotly_chart(tree_fig, use_container_width=True)

        # ----------------------------------------------------------------
        #  ROUTE SCORING
        # ----------------------------------------------------------------
        if show_route_scoring:
            st.markdown('<div class="section-header">📊 Multi-Objective Route Scoring</div>', unsafe_allow_html=True)

            route_score = score_full_route(pathway, building_blocks)
            render_route_score_dashboard(route_score, route_id=1)

            # Route score formula
            with st.expander("📐 Route Score Formula"):
                st.latex(r"S_{route} = w_1 \cdot C + w_2 \cdot SA + w_3 \cdot F + w_4 \cdot D - w_5 \cdot U - w_6 \cdot Cost")
                st.markdown("""
                Where:
                - **C** = Reaction confidence (atom conservation × beam rank)
                - **SA** = Synthetic accessibility (product vs. precursor complexity)
                - **F** = Chemical feasibility (reaction type, size balance, fragment count)
                - **D** = Route diversity (unique intermediates vs. other routes)
                - **U** = Uncertainty (beam rank, atom conservation, fragment count)
                - **Cost** = Estimated synthesis cost (reagent complexity, exotic atoms)
                """)

        # ----------------------------------------------------------------
        #  STEP-BY-STEP REACTION ANALYSIS
        # ----------------------------------------------------------------
        st.markdown('<div class="section-header">🧪 Step-by-Step Reaction Analysis</div>', unsafe_allow_html=True)

        for i, step in enumerate(pathway, 1):
            with st.expander(
                f"Step {i} (Depth {step['depth']}): {step['product'][:40]}... → {len(step['reactants'])} precursor(s)",
                expanded=(i == 1)
            ):
                # Step score
                step_score = score_reaction_step(step["product"], step["reactants"], beam_rank=0)
                score_cols = st.columns(6)
                score_cols[0].metric("Confidence", f"{step_score.reaction_confidence:.2f}")
                score_cols[1].metric("SA", f"{step_score.synthetic_accessibility:.2f}")
                score_cols[2].metric("Feasibility", f"{step_score.chemical_feasibility:.2f}")
                score_cols[3].metric("Uncertainty", f"{step_score.uncertainty:.2f}")
                score_cols[4].metric("Cost", f"{step_score.estimated_cost:.2f}")
                score_cols[5].metric("Step Score", f"{step_score.overall_score:.2f}")

                st.markdown("---")

                # Reaction center analysis
                if show_reaction_centers:
                    render_reaction_center(step, i)

                st.markdown("---")

                # Precursor properties
                if show_properties:
                    st.markdown("**Precursor Properties:**")
                    for j, reactant in enumerate(step["reactants"]):
                        r_props = compute_molecular_properties(reactant)
                        if r_props.get("valid"):
                            r_cols = st.columns([1, 1, 1, 1, 1])
                            r_cols[0].markdown(f"**{reactant[:20]}...**" if len(reactant) > 20 else f"**{reactant}**")
                            r_cols[1].caption(f"MW: {r_props['molecular_weight']}")
                            r_cols[2].caption(f"SA: {r_props['synthetic_accessibility']:.2f}")
                            r_cols[3].caption(f"LogP: {r_props['logp']}")
                            bb_status = "✅ Building Block" if reactant in building_blocks else "⬜ Not in BB database"
                            r_cols[4].caption(bb_status)

                # 3D comparison: product vs precursors
                if show_3d:
                    st.markdown("**3D Structure Comparison:**")
                    view_cols = st.columns(1 + min(len(step["reactants"]), 2))
                    with view_cols[0]:
                        fig_prod = build_3d_molecule_view(
                            step["product"], title="Product"
                        )
                        if fig_prod:
                            fig_prod.update_layout(height=350)
                            if dark_mode:
                                fig_prod.update_layout(
                                    scene=dict(bgcolor='#1e293b'),
                                    paper_bgcolor='#0f172a',
                                    title=dict(font=dict(color='white')),
                                )
                            st.plotly_chart(fig_prod, use_container_width=True)

                    for j, reactant in enumerate(step["reactants"][:2]):
                        with view_cols[j + 1]:
                            fig_r = build_3d_molecule_view(
                                reactant, title=f"Precursor {j+1}"
                            )
                            if fig_r:
                                fig_r.update_layout(height=350)
                                if dark_mode:
                                    fig_r.update_layout(
                                        scene=dict(bgcolor='#1e293b'),
                                        paper_bgcolor='#0f172a',
                                        title=dict(font=dict(color='white')),
                                    )
                                st.plotly_chart(fig_r, use_container_width=True)

        # ----------------------------------------------------------------
        #  MULTI-ROUTE COMPARISON (Advanced Multi-Objective Suite)
        # ----------------------------------------------------------------
        st.markdown('<div class="section-header">🔄 Multi-Route Strategic Comparison Suite</div>', unsafe_allow_html=True)

        with st.spinner("🔄 Generating diverse candidate pathways and running multi-objective ranking..."):
            candidate_routes = predict_multiple_routes(smiles, max_depth=max_depth, num_beams=num_beams)
            if not candidate_routes:
                candidate_routes = [{
                    "route_id": 1,
                    "strategy_name": "Primary Predicted Pathway",
                    "strategy_desc": "Standard retrosynthetic tree search.",
                    "pathway": pathway,
                }]
            scored_routes = score_and_rank_routes(candidate_routes, building_blocks)

        render_route_comparison(scored_routes, smiles, building_blocks)

        # ----------------------------------------------------------------
        #  FULL DIAGNOSTICS
        # ----------------------------------------------------------------
        with st.expander("📋 Full Pathway JSON & Diagnostics"):
            pathway_df = pd.DataFrame([
                {
                    "Step": i,
                    "Product": step["product"],
                    "Reactants": " + ".join(step["reactants"]),
                    "Depth": step["depth"],
                    "Reaction Type": step.get("reaction_type", "model_prediction"),
                }
                for i, step in enumerate(pathway, 1)
            ])
            st.dataframe(pathway_df, use_container_width=True, hide_index=True)

            st.json({
                "target": smiles,
                "source_format": source_format,
                "num_steps": len(pathway),
                "elapsed_seconds": round(elapsed, 3),
                "route_score": route_score.as_dict() if show_route_scoring else None,
                "pathway": pathway,
            })

        # ----------------------------------------------------------------
        #  EXPORT
        # ----------------------------------------------------------------
        st.markdown("---")
        export_cols = st.columns(3)
        with export_cols[0]:
            json_str = json.dumps({
                "target": smiles,
                "pathway": pathway,
                "route_score": route_score.as_dict() if show_route_scoring else None,
            }, indent=2)
            st.download_button(
                "📥 Download Results (JSON)",
                data=json_str,
                file_name=f"retrosynthesis_{smiles[:10]}.json",
                mime="application/json",
                use_container_width=True,
            )
        with export_cols[1]:
            csv_str = pathway_df.to_csv(index=False)
            st.download_button(
                "📊 Download Pathway (CSV)",
                data=csv_str,
                file_name=f"pathway_{smiles[:10]}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    else:
        if predict_btn:
            st.warning("Please enter a molecule (SMILES, chemical name, formula, CAS number, or InChI)")


# ============================================================================
#  FOOTER
# ============================================================================

st.markdown("---")

st.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <div style="font-size: 0.9rem; color: #94a3b8;">
        <b>BioRetrosynthesis</b> — Advanced Retrosynthesis Visualization & Planning<br>
        Model: QLoRA-fine-tuned ReactionT5v2 • Data: KEGG/MetaCyc/USPTO-NPL • Chemistry: RDKit<br>
        Developed and Maintained by <b>Mayur Nhavalde</b>
    </div>
</div>
""", unsafe_allow_html=True)
