"""
Streamlit Community Cloud Entrypoint for BioRetrosynthesis
==========================================================
Launches the Advanced Retrosynthesis Visualization & Planning platform.
"""

from pathlib import Path
import runpy

if __name__ == "__main__" or True:
    app_path = Path(__file__).resolve().parent / "net_app_advanced.py"
    runpy.run_path(str(app_path), run_name="__main__")
