"""
cli.py

Command-line REPL for the plasticity engine.

Usage:
    python -m plasticity.cli
    python -m plasticity.cli --model mlx-community/Qwen2.5-3B-Instruct-4bit
    python -m plasticity.cli --db ~/.llm-plasticity/work.db
"""

from __future__ import annotations

import argparse
import sys

from . import config
from .engine import PlasticityEngine
from .code_export import extract_code_blocks, save_code_block

HELP_TEXT = """\
Commands:
  /graph            Show memory graph stats (node/edge counts, most-used concepts)
  /show <concept>   Show a concept's current neighbors and edge strengths
  /forget <concept> Remove a concept and its connections from memory
  /reset            Wipe the entire memory graph (irreversible)
  /help             Show this message
  /exit, /quit      Leave
"""


def main():
    parser = argparse.ArgumentParser(description="Command-line LLM with an evolving associative memory.")
    parser.add_argument("--model", default=config.DEFAULT_MODEL, help="MLX model repo id")
    parser.add_argument("--db", default=str(config.DEFAULT_DB_PATH), help="Path to the memory graph database")
    parser.add_argument("--max-tokens", type=int, default=config.MAX_NEW_TOKENS)
    args = parser.parse_args()

    try:
        engine = PlasticityEngine(model_name=args.model, db_path=args.db, max_new_tokens=args.max_tokens)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Ready. Model: {args.model}")
    print(f"Memory:  {args.db}")
    print("Type /help for commands, /exit to quit.\n")

    try:
        while True:
            try:
                user_input = input("you> ").strip()
            except EOFError:
                break
            if not user_input:
                continue

            if user_input in ("/exit", "/quit"):
                break
            if user_input == "/help":
                print(HELP_TEXT)
                continue
            if user_input == "/graph":
                _print_graph_stats(engine)
                continue
            if user_input.startswith("/show "):
                _print_neighbors(engine, user_input[len("/show "):].strip())
                continue
            if user_input.startswith("/forget "):
                label = user_input[len("/forget "):].strip()
                removed = engine.graph.forget(label)
                print(f"{'forgot' if removed else 'no such concept:'} '{label}'")
                continue
            if user_input == "/reset":
                confirm = input("This wipes the entire memory graph. Type 'yes' to confirm: ")
                if confirm.strip().lower() == "yes":
                    engine.graph.reset()
                    print("Memory graph cleared.")
                continue

            turn = engine.chat(user_input)
            if turn.activated_memory:
                labels = ", ".join(f"{n.label}({n.activation:.2f})" for n in turn.activated_memory[:5])
                print(f"  [memory: {labels}]", file=sys.stderr)
            print(f"llm> {turn.assistant_msg}\n")

            _offer_code_save(turn.assistant_msg, user_input)

    finally:
        engine.close()
        print("Session closed. Memory graph saved.")


def _offer_code_save(assistant_msg: str, user_msg: str):
    """If the reply contains fenced code blocks, ask (per block) whether to
    save it to a source file. Write-only: this never reads any existing
    file, it only ever creates new ones under config.CODE_EXPORT_DIR."""
    blocks = extract_code_blocks(assistant_msg, user_msg)
    if not blocks:
        return

    for i, block in enumerate(blocks, start=1):
        label = f" ({i}/{len(blocks)})" if len(blocks) > 1 else ""
        answer = input(
            f"Save this {block.language or 'text'} code block{label} as "
            f"'{block.suggested_name}'? [y/N/name]: "
        ).strip()

        if not answer or answer.lower() in ("n", "no"):
            continue

        custom_name = None if answer.lower() in ("y", "yes") else answer
        path = save_code_block(block, config.CODE_EXPORT_DIR, filename=custom_name)
        print(f"  saved -> {path}")


def _print_graph_stats(engine: PlasticityEngine):
    stats = engine.graph.stats()
    print(f"nodes: {stats['nodes']}   edges: {stats['edges']}")
    if stats["top_nodes"]:
        print("most-activated concepts:")
        for label, count in stats["top_nodes"]:
            print(f"  {label}  (seen {count}x)")


def _print_neighbors(engine: PlasticityEngine, label: str):
    neighbors = engine.graph.neighbors(label)
    if not neighbors:
        print(f"no connections found for '{label}' (or it doesn't exist yet)")
        return
    print(f"'{label}' connects to:")
    for other, weight in neighbors:
        print(f"  {other}  (weight {weight:.2f})")


if __name__ == "__main__":
    main()
