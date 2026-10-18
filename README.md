# 🧬 BioRetrosynthesis

> **Advanced Retrosynthesis Visualization, Planning & Prediction Platform**  
> Interactive Search Tree • 3D Molecular Conformers • Reaction-Center Analysis • Multi-Objective Route Scoring

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)]
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![RDKit](https://img.shields.io/badge/chemistry-RDKit-green.svg)](https://www.rdkit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

DEPLOYED ON STREAMLIT - https://bioretrosynthesis-advanced-cmwrxjryhb9snf2noiq2sx.streamlit.app/

---
## 🌟 Key Features

- **🔬 Multi-Format Molecular Input:** Enter SMILES, Chemical Names (e.g. *Aspirin*, *Caffeine*), Molecular Formulas, or CAS numbers.
- **🗺️ Interactive Retrosynthesis Tree:** Dynamic, collapsible search tree mapping target molecules to bio-compatible building blocks.
- **🧬 3D Molecular Conformers:** Real-time 3D conformer generation and interactive rotation.
- **⚡ Reaction-Center Detection:** Automated identification of transformed, broken, and formed chemical bonds for each synthetic step.
- **📊 Multi-Objective Route Scoring:** Comprehensive scoring based on synthetic accessibility (SA score), route length, feasibility, and biological building-block availability.
- **📈 Detailed Molecular Diagnostics:** Complete unclipped properties (MW, LogP, TPSA, HBD/HBA, Ring counts, and Lipinski's Rule of 5).

---

## 🚀 Quick Start (Local)

### 1. Clone the repository
```bash
git clone https://github.com/mayurOG/BIORETROSYNTHESIS-.git
cd BIORETROSYNTHESIS-
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Streamlit
```bash
streamlit run streamlit_app.py
```
Open **http://localhost:8501** in your browser.

---
---

## 📁 Repository Structure

```
BIORETROSYNTHESIS-/
├── .streamlit/
│   └── config.toml           # Streamlit theme & UI styling
├── final_model/              # Retrosynthesis model weights & tokenizer
├── tbr_opt/                  # TBR search algorithms and index
├── bio_building_block.csv    # Curated biological building blocks
├── chemistry_utils.py        # Chemical structure processing & validation
├── reaction_rules.py         # Reaction templates & feasibility rules
├── retrosynthesis_engine.py  # RDKit analysis engine & conformer generation
├── net_app_advanced.py       # Core Streamlit interactive application
├── streamlit_app.py          # Streamlit Community Cloud entrypoint
├── requirements.txt          # Python dependencies
├── packages.txt              # Linux apt packages for headless graphics
└── README.md                 # Project documentation
```

---

## 👨‍💻 Author

**Mayur Nhavalde**  
- Email: [mnhavalde1@gmail.com](mailto:mnhavalde1@gmail.com)  
- GitHub: [@mayurOG](https://github.com/mayurOG)
