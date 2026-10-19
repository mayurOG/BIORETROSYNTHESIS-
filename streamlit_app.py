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
Streamlit Community Cloud Entrypoint for BioRetrosynthesis
==========================================================
Launches the Advanced Retrosynthesis Visualization & Planning platform.
"""

from pathlib import Path
import runpy

if __name__ == "__main__" or True:
    app_path = Path(__file__).resolve().parent / "net_app_advanced.py"
    runpy.run_path(str(app_path), run_name="__main__")
