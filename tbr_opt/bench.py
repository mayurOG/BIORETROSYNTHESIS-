"""Before/after benchmark for the two optimizations.

Sections 1-3 are model-free and deterministic. Pass --model to add validity and latency
measured on a real seq2seq model.

    python -m tbr_opt.bench
    python -m tbr_opt.bench --model hf-internal-testing/tiny-random-T5ForConditionalGeneration
    python -m tbr_opt.bench --model <base> --adapter final_model

Read section 2 first. End-to-end latency is dominated by model.generate(), so the
numbers that matter are how many generate() calls happen and how many molecules get
scored at all. Sections 1 and 3 are real but second-order, and are reported that way.
"""

import argparse
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "..", "bio_building_block.csv")

# A layered DAG with deliberately shared intermediates, so the baseline re-expands
# nodes the batched engine scores once. Chemistry is illustrative; every string is a
# real parseable SMILES so the validity filter does not distort the shape.
LEVELS = [
    ["CC(=O)Nc1ccc(O)cc1"],
    ["CC(=O)O", "Nc1ccc(O)cc1", "CC(=O)Cl"],
    ["CCO", "O=C=O", "c1ccc(O)cc1", "NCC", "CC", "CCN"],
    ["C", "CC", "O", "N", "S", "Cl"],
]

NET = {
    "CC(=O)Nc1ccc(O)cc1": "CC(=O)O.Nc1ccc(O)cc1.CC(=O)Cl",
    "CC(=O)O": "CCO.O=C=O",
    "Nc1ccc(O)cc1": "c1ccc(O)cc1.NCC",
    "CC(=O)Cl": "CC.CC(=O)O",
    "CCO": "CC.O",
    "O=C=O": "O.C",
    "c1ccc(O)cc1": "C.O",
    "NCC": "C.N",
    "CC": "C.C",
    "CCN": "CC.N",
}

BLOCKS = ["C", "CC", "O", "N", "S", "Cl"]


def _load_blocks():
    import csv

    with open(CSV_PATH, newline="") as fh:
        return [r[0].strip() for r in csv.reader(fh) if r and r[0].strip()]


def bench_building_block_index(lookups=20000):
    """net_model_utils.is_building_block rebuilds a 387-entry set on every call.

    Reported in three parts so the cost of canonical matching is not smuggled into the
    "speedup": the engine reuses the canonical key it already computes for its cache, so
    the hot path is two frozenset lookups, not an RDKit round-trip.
    """
    from .engine import BuildingBlockIndex, canonicalize_smiles

    rows = _load_blocks()
    rng = random.Random(0)
    probes = [rng.choice(rows) for _ in range(lookups // 2)]
    probes += ["CC(=O)O", "OC(C)=O", "not_a_molecule"] * (lookups // 6)

    idx = BuildingBlockIndex(rows)
    keys = [canonicalize_smiles(p) for p in probes]

    t0 = time.perf_counter()
    legacy_hits = sum(p in set(rows) for p in probes)
    legacy_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    hot_hits = sum(p in idx.raw or k in idx.canon for p, k in zip(probes, keys))
    hot_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for p in probes[:2000]:
        canonicalize_smiles(p)
    canon_s = (time.perf_counter() - t0) / 2000

    assert hot_hits == legacy_hits, "index disagreed with the legacy lookup"
    return {
        "lookups": len(probes),
        "legacy_seconds (rebuild each call)": legacy_s,
        "optimized_seconds (hot path)": hot_s,
        "hot_path_speedup": legacy_s / hot_s if hot_s else float("inf"),
        "canonicalize_seconds_per_molecule": canon_s,
        "matches_agree": hot_hits == legacy_hits,
    }


class _Enc(dict):
    """tokenizer(...) returns a mapping in HF, so ** unpacking has to work."""

    def to(self, device):
        return self


class StubTokenizer:
    """Passes strings through unchanged so the search shape is measured, not the model."""

    pad_token_id = 0

    def __call__(self, texts, **kwargs):
        return _Enc(input_ids=list(texts))

    def batch_decode(self, ids, skip_special_tokens=True):
        return list(ids)


class StubModel:
    def __init__(self, mapping, default="C.C", per_call=0.0):
        self.mapping = mapping
        self.default = default
        self.per_call = per_call
        self.calls = []
        self._p = None

    def parameters(self):
        if self._p is None:
            import torch

            self._p = torch.zeros(1)
        return iter([self._p])

    def generate(self, input_ids=None, **kwargs):
        if self.per_call:
            time.sleep(self.per_call)
        self.calls.append(list(input_ids))
        return [self.mapping.get(t, self.default) for t in input_ids]


def legacy_predict(target, mapping, blocks, max_depth=6):
    """Reproduces net_model_utils.predict_multistep: depth-first, batch size 1, no dedup."""
    calls, lookups, results = 0, 0, []

    def is_block(s):
        nonlocal lookups
        lookups += 1
        return s in set(blocks)

    def recurse(smiles, depth):
        nonlocal calls
        if depth >= max_depth or is_block(smiles):
            return
        calls += 1
        reactants = mapping.get(smiles, "C.C").split(".")
        results.append({"product": smiles, "reactants": reactants, "depth": depth})
        for r in reactants:
            recurse(r, depth + 1)

    recurse(target, 1)
    return results, calls, lookups


def bench_search_shape(per_call=0.0, max_depth=6):
    """Same network, same answers - measures how the work is scheduled."""
    from .engine import BuildingBlockIndex, RetroEngine

    target = LEVELS[0][0]

    t0 = time.perf_counter()
    legacy_results, legacy_calls, legacy_lookups = legacy_predict(
        target, NET, BLOCKS, max_depth
    )
    legacy_s = time.perf_counter() - t0

    model = StubModel(NET, per_call=per_call)
    engine = RetroEngine(
        model, StubTokenizer(), BuildingBlockIndex(BLOCKS),
        constrained=False, batch_size=8,
    )
    t0 = time.perf_counter()
    pathway, stats = engine.predict(target, max_depth=max_depth)
    new_s = time.perf_counter() - t0

    legacy_edges = {(r["product"], x) for r in legacy_results for x in r["reactants"]}
    new_edges = {(r["product"], x) for r in pathway for x in r["reactants"]}

    return {
        "legacy_generate_calls": legacy_calls,
        "optimized_generate_calls": stats.generate_calls,
        "generate_call_reduction": legacy_calls / stats.generate_calls
        if stats.generate_calls else float("inf"),
        "legacy_nodes_scored": len(legacy_results),
        "optimized_nodes_scored": stats.nodes_expanded,
        "node_reduction": len(legacy_results) / stats.nodes_expanded
        if stats.nodes_expanded else float("inf"),
        "duplicates_pruned": stats.duplicates_pruned,
        "avg_batch_size": stats.as_dict()["avg_batch_size"],
        "legacy_bb_lookups": legacy_lookups,
        "pathways_identical": legacy_edges == new_edges,
        "legacy_seconds": legacy_s,
        "optimized_seconds": new_s,
    }


def bench_repeat_queries(rounds=5):
    from .engine import BuildingBlockIndex, RetroEngine

    model = StubModel(NET)
    engine = RetroEngine(
        model, StubTokenizer(), BuildingBlockIndex(BLOCKS), constrained=False
    )
    cold = None
    warm = []
    for i in range(rounds):
        before = len(model.calls)
        engine.predict(LEVELS[0][0], max_depth=6)
        delta = len(model.calls) - before
        if i == 0:
            cold = delta
        else:
            warm.append(delta)
    return {
        "cold_generate_calls": cold,
        "warm_generate_calls": warm,
        "cached_entries": len(engine.cache),
        "all_warm_queries_free": all(w == 0 for w in warm),
    }


def bench_model(model_id, adapter=None, num_beams=3, max_length=64):
    """Validity and latency on a real model. Needs network access for the weights."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from .grammar import INITIAL, SmilesGrammarConstraint, advance, is_terminal

    def syntactically_ok(s):
        st = advance(INITIAL, s.replace(" ", ""))
        return st is not None and is_terminal(st)

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch.float32)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    prompts = [
        "CC(=O)O",
        "OC(=O)CCCCC(=O)O",
        "N[C@@H](Cc1ccccc1)C(=O)O",
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "C1CCNCC1",
        "OC(=O)[C@H](N)CC(=O)O",
        "CCO",
        "c1ccc2ccccc2c1",
    ]
    device = next(model.parameters()).device
    constraint = SmilesGrammarConstraint(tok)

    def run(constrained, beams):
        t0 = time.perf_counter()
        outs = []
        for p in prompts:
            enc = tok([p], return_tensors="pt").to(device)
            kwargs = dict(
                num_beams=beams, num_return_sequences=1, max_length=max_length,
                early_stopping=True, pad_token_id=tok.pad_token_id,
            )
            if constrained:
                kwargs["logits_processor"] = [constraint]
            with torch.no_grad():
                seqs = model.generate(**enc, **kwargs)
            outs.append(
                tok.batch_decode(seqs, skip_special_tokens=True)[0]
                .replace(" ", "").rstrip(".")
            )
        elapsed = time.perf_counter() - t0
        non_empty = [o for o in outs if o]
        return {
            "syntactic_validity": sum(syntactically_ok(o) for o in non_empty) / len(non_empty)
            if non_empty else 0.0,
            "seconds_total": elapsed,
            "seconds_per_query": elapsed / len(prompts),
            "non_empty": len(non_empty),
            "samples": non_empty[:4],
        }

    return {
        "beams_baseline": 5,
        "beams_constrained": num_beams,
        "unconstrained": run(False, 5),
        "constrained": run(True, num_beams),
        "distinct_masks_built": constraint.final_builds,
        "distinct_base_masks": constraint.base_builds,
    }


def _print(title, rows):
    print(f"\n{title}\n{'-' * len(title)}")
    _print_nested(rows, 2)


def _print_nested(rows, indent):
    pad = " " * indent
    width = max((len(str(k)) for k in rows), default=0)
    for k, v in rows.items():
        if isinstance(v, dict):
            print(f"{pad}{k}:")
            _print_nested(v, indent + 4)
        else:
            print(f"{pad}{k:<{width}}  {_fmt(v)}")


def _fmt(v):
    if isinstance(v, float):
        if v == 0:
            return "0.0"
        if abs(v) < 0.001:
            return f"{v:.3e}"
        return f"{v:,.3f}" if v < 1000 else f"{v:,.1f}"
    return v


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="HF model id for the validity/latency benchmark")
    parser.add_argument("--adapter", help="PEFT adapter dir (e.g. final_model)")
    parser.add_argument("--beams", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--per-call", type=float, default=0.0,
                        help="simulated per-generate-call overhead in seconds")
    args = parser.parse_args(argv)

    print(f"python {sys.version.split()[0]}  •  building blocks: {os.path.basename(CSV_PATH)}")

    shape = bench_search_shape(per_call=args.per_call)
    _print("1. Search shape — the number that matters (DFS batch=1 vs level-order batched)",
           shape)
    _print("2. Repeated queries — cold vs warm canonical cache", bench_repeat_queries())
    _print("3. Building-block lookup — micro-optimization, second-order",
           bench_building_block_index())

    if args.model:
        _print(f"4. Model benchmark: {args.model}",
               bench_model(args.model, args.adapter, args.beams, args.max_length))
    else:
        print("\n4. Model benchmark skipped (pass --model to run it)")

    print("\nInterpretation")
    print("-" * 14)
    print("  Section 1 counts are scheduling facts, not GPU measurements. Wall-clock")
    print("  gain scales with your per-call overhead; re-run with --per-call set to a")
    print("  measured value for your hardware. Section 3's microseconds are negligible")
    print("  beside a single generate() call and are included for completeness.")
    print("  Syntactic validity from the grammar constraint does not imply chemical")
    print("  validity; engine._split_valid applies the RDKit filter on top.")
    if args.model:
        print("  CAVEAT on section 4: an unconstrained baseline of 0.0 means the model is")
        print("  randomly weighted, so this is a mechanism check - it proves the constraint")
        print("  forces valid syntax out of noise. A real before/after validity figure needs")
        print("  the fine-tuned weights: --adapter final_model. Compare distinct_masks_built")
        print("  against total decode steps to see the cache holding, and read the")
        print("  seconds_per_query delta as the honest cost of the constraint.")
    return 0 if shape["pathways_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
