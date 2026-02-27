"""
run_scaling.py — Scaling experiment across LLM families and parameter counts.

Evaluates LA-TQD with three model families (Qwen, Mistral, Llama)
at three parameter scales (<10B, ~30B, 70B+).

Usage:
    python scripts/run_scaling.py --dry-run         # show config only
    python scripts/run_scaling.py --rounds 30 --load-in-4bit
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_DIR, N_ROUNDS, N_RUNS, RESULTS_DIR

DEFAULT_MODEL_GROUPS = {
    "Qwen": {
        "<10B": "Qwen/Qwen2.5-7B-Instruct",
        "~30B": "Qwen/Qwen2.5-32B-Instruct",
        "70B+": "Qwen/Qwen2.5-72B-Instruct",
    },
    "Mistral": {
        "<10B": "mistralai/Mistral-7B-Instruct-v0.3",
        "~30B": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "70B+": "mistralai/Mixtral-8x22B-Instruct-v0.1",
    },
    "Llama": {
        "<10B": "meta-llama/Llama-3.1-8B-Instruct",
        "~30B": "meta-llama/Llama-3.1-70B-Instruct",
        "70B+": "meta-llama/Llama-3.1-70B-Instruct",
    },
}


def main():
    parser = argparse.ArgumentParser(description="LA-TQD Model Scaling Experiment")
    parser.add_argument("--models-config", default=None)
    parser.add_argument("--rounds", type=int, default=N_ROUNDS)
    parser.add_argument("--runs", type=int, default=N_RUNS)
    parser.add_argument("--no-esm", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=os.path.join(RESULTS_DIR, "scaling"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    groups = DEFAULT_MODEL_GROUPS
    if args.models_config and os.path.exists(args.models_config):
        with open(args.models_config) as f:
            groups = json.load(f)

    if args.dry_run:
        print(json.dumps(groups, indent=2))
        return

    with open(os.path.join(DATA_DIR, "seed_sequences.json")) as f:
        data = json.load(f)
    seeds = data if isinstance(data, list) else data["sequences"]

    esm_model, esm_tokenizer = None, None
    if not args.no_esm:
        from src.esm_scorer import load_esm2
        esm_model, esm_tokenizer = load_esm2(device=args.device)

    all_results = {}
    for family, sizes in groups.items():
        all_results[family] = {}
        for size_label, model_name in sizes.items():
            print(f"\n[Scaling] {family} {size_label}: {model_name}")
            from src.llm_mutator import LLMMutator
            from src.search import run_experiment
            mutator = LLMMutator(model_name, device=args.device, load_in_4bit=args.load_in_4bit)
            results = run_experiment(
                mutator=mutator, seeds=seeds, n_runs=args.runs, n_rounds=args.rounds,
                esm_model=esm_model, esm_tokenizer=esm_tokenizer, device=args.device,
                save_dir=args.output_dir, experiment_name=f"{family}_{size_label}",
            )
            all_results[family][size_label] = {
                "model": model_name,
                "coverage_mean": results["coverage_mean"],
                "coverage_std":  results["coverage_std"],
            }
            del mutator
            try:
                import torch; torch.cuda.empty_cache()
            except: pass

    out = os.path.join(args.output_dir, "scaling_results.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 65)
    print(f"{'Family':<12} {'<10B':>18} {'~30B':>18} {'70B+':>18}")
    print("-" * 65)
    for family, sizes in all_results.items():
        vals = [f"{sizes[s]['coverage_mean']:.2f}+/-{sizes[s]['coverage_std']:.2f}"
                if s in sizes else "N/A" for s in ["<10B", "~30B", "70B+"]]
        print(f"{family:<12} " + "  ".join(f"{v:>18}" for v in vals))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
