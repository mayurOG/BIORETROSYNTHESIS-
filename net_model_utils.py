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

from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftConfig, PeftModel
from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO
import base64

def load_model(model_path: str):
    model_dir = Path(model_path).resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        "sagawa/ReactionT5v2-retrosynthesis-USPTO_50k",
        return_tensors="pt",
    )
    peft_config = PeftConfig.from_pretrained(str(model_dir))
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch.float32,
    )
    model = PeftModel.from_pretrained(base_model, str(model_dir))
    model.eval()
    return {"model": model, "tokenizer": tokenizer}

def is_building_block(smiles: str, blocks_df) -> bool:
    return smiles in set(blocks_df.iloc[:, 0].tolist())

def predict_single_step(model_bundle, product_smiles: str) -> str:
    model = model_bundle["model"]
    tokenizer = model_bundle["tokenizer"]
    inputs = tokenizer(product_smiles, return_tensors='pt')
    outputs = model.generate(
        **inputs,
        num_beams=5,
        num_return_sequences=1,
        max_length=128,
        early_stopping=True
    )
    reactants = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reactants.replace(" ", "").rstrip(".")

def predict_multistep(model_bundle, target_smiles: str, blocks_df, max_depth: int = 5):
    model = model_bundle["model"]
    tokenizer = model_bundle["tokenizer"]
    results = []

    def recurse(smiles: str, depth: int):
        if depth >= max_depth or is_building_block(smiles, blocks_df):
            return
        inputs = tokenizer(smiles, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            num_beams=5,
            num_return_sequences=1,
            max_length=128,
            early_stopping=True,
        )
        reactants = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(" ", "")
        results.append({
            "product": smiles,
            "reactants": reactants.split('.'),
            "depth": depth
        })
        for r in reactants.split('.'):
            recurse(r, depth + 1)

    recurse(target_smiles, 1)
    return results

def smiles_to_image_base64(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol, size=(200, 200))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"
    else:
        return None
