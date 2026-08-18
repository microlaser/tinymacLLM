"""
engine.py

Wires together:
  1. MLX-LM inference (the frozen base model)
  2. MemoryGraph retrieval (spreading activation -> context)
  3. MemoryGraph writing (concept extraction -> reinforcement)

This is the "plasticity loop": every turn reads the graph's current state
before generating, and writes to the graph after generating, so the next
turn's behavior is shaped by everything that came before -- without ever
touching the model's weights.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Optional

from . import config
from .memory_graph import ActivatedNode, MemoryGraph
from .concept_extractor import extract_concepts

SYSTEM_PROMPT = (
    "You are a helpful assistant with an evolving associative memory. "
    "When relevant prior context is provided below, use it naturally -- "
    "don't mention that it came from a 'memory system' unless asked."
)


@dataclass
class Turn:
    user_msg: str
    assistant_msg: str
    activated_memory: List[ActivatedNode]


class PlasticityEngine:
    def __init__(
        self,
        model_name: str = config.DEFAULT_MODEL,
        db_path=config.DEFAULT_DB_PATH,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        temperature: float = config.TEMPERATURE,
        verbose_load: bool = True,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.graph = MemoryGraph(
            db_path,
            decay_half_life_seconds=config.DECAY_HALF_LIFE_SECONDS,
            reinforcement_gain=config.REINFORCEMENT_GAIN,
            max_edge_weight=config.MAX_EDGE_WEIGHT,
        )

        self.history: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        if verbose_load:
            print(f"Loading {model_name} ... (first run downloads the weights, then it's cached)",
                  file=sys.stderr)
        self._load_model()

    # ------------------------------------------------------------------ #
    def _load_model(self):
        try:
            from mlx_lm import load
        except ImportError as e:
            raise RuntimeError(
                "mlx-lm is not installed. This tool requires an Apple Silicon Mac.\n"
                "Install with: pip install mlx-lm"
            ) from e
        self.model, self.tokenizer = load(self.model_name)

    def _raw_generate(self, prompt: str, max_tokens: int) -> str:
        """Low-level single-shot generation, used both for chat replies and
        for the concept-extraction calls. No memory side effects here."""
        from mlx_lm import generate as mlx_generate

        text = mlx_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        return text

    def _apply_chat_template(self, messages: List[dict]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        # fallback: plain concatenation if the tokenizer has no chat template
        parts = [f"{m['role'].upper()}: {m['content']}" for m in messages]
        parts.append("ASSISTANT:")
        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    def _retrieve_memory(self, user_msg: str) -> List[ActivatedNode]:
        query_terms = user_msg.split()
        return self.graph.spreading_activation(
            query_terms,
            depth=config.SPREADING_DEPTH,
            decay_per_hop=config.SPREADING_DECAY_PER_HOP,
            activation_floor=config.ACTIVATION_FLOOR,
            top_k=config.TOP_K_MEMORY_NODES,
        )

    def _format_memory_block(self, nodes: List[ActivatedNode]) -> Optional[str]:
        if not nodes:
            return None
        lines = ["[Associative memory activated by this prompt, strongest first:]"]
        for n in nodes:
            note = f" -- {n.notes[-1]}" if n.notes else ""
            lines.append(f"- {n.label} (activation {n.activation:.2f}){note}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    def chat(self, user_msg: str) -> Turn:
        activated = self._retrieve_memory(user_msg)
        memory_block = self._format_memory_block(activated)

        turn_messages = list(self.history)
        if memory_block:
            turn_messages.append({"role": "system", "content": memory_block})
        turn_messages.append({"role": "user", "content": user_msg})

        prompt = self._apply_chat_template(turn_messages)
        assistant_msg = self._raw_generate(prompt, self.max_new_tokens).strip()

        # commit to conversational history (not the memory graph -- that's separate)
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": assistant_msg})

        # the plasticity write: extract concepts, reinforce the graph
        concepts = extract_concepts(
            user_msg, assistant_msg,
            generate_fn=lambda p, max_tokens: self._raw_generate(p, max_tokens),
        )
        snippet = user_msg[:120]
        self.graph.record_exchange(concepts, note=snippet)

        return Turn(user_msg=user_msg, assistant_msg=assistant_msg, activated_memory=activated)

    def close(self):
        self.graph.close()
