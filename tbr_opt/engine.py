"""Batched retrosynthesis engine with memoized, canonicalized caching."""

import time
from collections import OrderedDict
from dataclasses import dataclass, field

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def canonicalize_smiles(smiles):
    """RDKit canonical form, so trivially different inputs share a cache entry.

    Falls back to the whitespace-stripped input for strings RDKit cannot parse, which
    keeps the cache usable on invalid model output instead of missing on every lookup.
    """
    if not smiles:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles.strip()
    return Chem.MolToSmiles(mol)


class BuildingBlockIndex:
    """O(1) building-block lookup over both raw and canonical SMILES.

    Prefer checking `.raw` / `.canon` directly when you already have a canonical key:
    `__contains__` canonicalizes on a miss, and an RDKit round-trip costs far more than
    the set rebuild this class exists to avoid.
    """

    def __init__(self, smiles_list):
        self._raw = frozenset(s.strip() for s in smiles_list if isinstance(s, str) and s.strip())
        self._canon = frozenset(c for c in (canonicalize_smiles(s) for s in self._raw) if c)

    @property
    def raw(self):
        return self._raw

    @property
    def canon(self):
        return self._canon

    def __contains__(self, smiles):
        return smiles in self._raw or canonicalize_smiles(smiles) in self._canon

    def __len__(self):
        return len(self._raw)

    @classmethod
    def from_csv(cls, path):
        import csv

        with open(path, newline="") as fh:
            return cls(row[0] for row in csv.reader(fh) if row)

    @classmethod
    def from_dataframe(cls, df):
        return cls(df.iloc[:, 0].tolist())


class LRUCache:
    def __init__(self, maxsize=4096):
        self.maxsize = maxsize
        self._data = OrderedDict()

    def get(self, key):
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        return None

    def put(self, key, value):
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def clear(self):
        self._data.clear()

    def __len__(self):
        return len(self._data)


@dataclass
class RetroStats:
    nodes_expanded: int = 0
    generate_calls: int = 0
    molecules_scored: int = 0
    cache_hits: int = 0
    invalid_reactants: int = 0
    duplicates_pruned: int = 0
    wall_seconds: float = 0.0
    extra: dict = field(default_factory=dict)

    def as_dict(self):
        return {
            "nodes_expanded": self.nodes_expanded,
            "generate_calls": self.generate_calls,
            "molecules_scored": self.molecules_scored,
            "avg_batch_size": round(self.molecules_scored / self.generate_calls, 2)
            if self.generate_calls
            else 0.0,
            "cache_hits": self.cache_hits,
            "invalid_reactants": self.invalid_reactants,
            "duplicates_pruned": self.duplicates_pruned,
            "wall_seconds": round(self.wall_seconds, 3),
        }


def _split_valid(reactants_str, stats):
    """Split a '.'-joined string and drop fragments RDKit cannot parse.

    Without this, invalid SMILES reach the UI, where MolToImage returns None and
    st.image(None) raises.
    """
    valid = []
    for part in reactants_str.split("."):
        if not part:
            continue
        if Chem.MolFromSmiles(part) is not None:
            valid.append(part)
        else:
            stats.invalid_reactants += 1
    return valid


class RetroEngine:
    """Level-order retrosynthesis search.

    The original implementation recursed depth-first and called model.generate() once
    per molecule at batch size 1. This gathers every molecule at the same depth and
    scores them in a single batched call, which is where the throughput win comes from.

    Two distinct dedup layers: `expanded` collapses repeats inside one search, while the
    canonical-keyed LRU cache persists results across predict() calls, so a repeated
    query - or a Streamlit rerun - costs nothing.
    """

    def __init__(
        self,
        model,
        tokenizer,
        building_blocks,
        num_beams=3,
        max_length=128,
        constrained=True,
        cache_size=4096,
        batch_size=16,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.blocks = building_blocks
        self.num_beams = num_beams
        self.max_length = max_length
        self.constrained = constrained
        self.batch_size = batch_size
        self.cache = LRUCache(cache_size)
        self._device = next(model.parameters()).device
        self._constraint = None

    @property
    def device(self):
        return self._device

    def _is_block(self, mol, key):
        """Membership test that reuses an already-canonicalized key.

        Going through `mol in self.blocks` would canonicalize a second time per node.
        """
        raw = getattr(self.blocks, "raw", None)
        if raw is not None:
            return mol in raw or key in self.blocks.canon
        return mol in self.blocks

    def _get_constraint(self):
        if self._constraint is None:
            from .grammar import SmilesGrammarConstraint

            self._constraint = SmilesGrammarConstraint(self.tokenizer)
        return self._constraint

    def generate_batch(self, smiles_list, stats=None):
        """Score a list of product SMILES in as few generate() calls as possible."""
        import torch

        if not smiles_list:
            return []

        outputs = []
        for start in range(0, len(smiles_list), self.batch_size):
            chunk = smiles_list[start : start + self.batch_size]
            enc = self.tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True, max_length=256
            ).to(self._device)

            kwargs = dict(
                num_beams=self.num_beams,
                num_return_sequences=1,
                max_length=self.max_length,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            if self.constrained:
                kwargs["logits_processor"] = [self._get_constraint()]

            with torch.no_grad():
                seqs = self.model.generate(**enc, **kwargs)

            decoded = self.tokenizer.batch_decode(seqs, skip_special_tokens=True)
            outputs.extend(d.replace(" ", "").rstrip(".") for d in decoded)

            if stats is not None:
                stats.generate_calls += 1
                stats.molecules_scored += len(chunk)

        return outputs

    def predict(self, target_smiles, max_depth=5, use_cache=True):
        """Returns (pathway, stats). Each pathway entry matches the legacy dict shape."""
        started = time.perf_counter()
        stats = RetroStats()
        pathway = []

        expanded = set()
        frontier = [target_smiles.strip()]

        for depth in range(1, max_depth + 1):
            if not frontier:
                break

            pending, pending_keys, next_frontier = [], set(), []

            for mol in frontier:
                key = canonicalize_smiles(mol)
                if not key:
                    continue
                if key in expanded:
                    stats.duplicates_pruned += 1
                    continue
                expanded.add(key)

                if self._is_block(mol, key):
                    continue

                cached = self.cache.get(key) if use_cache else None
                if cached is not None:
                    stats.cache_hits += 1
                    stats.nodes_expanded += 1
                    pathway.append(
                        {"product": mol, "reactants": list(cached), "depth": depth}
                    )
                    next_frontier.extend(cached)
                    continue

                if key in pending_keys:
                    continue
                pending_keys.add(key)
                pending.append(mol)

            if pending:
                for mol, reactants in zip(pending, self.generate_batch(pending, stats)):
                    stats.nodes_expanded += 1
                    parts = _split_valid(reactants, stats)
                    self.cache.put(canonicalize_smiles(mol), parts)
                    pathway.append({"product": mol, "reactants": parts, "depth": depth})
                    next_frontier.extend(parts)

            frontier = next_frontier

        stats.wall_seconds = time.perf_counter() - started
        return pathway, stats
