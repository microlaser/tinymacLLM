"""
memory_graph.py

An associative memory graph that emulates synaptic plasticity:
- Concepts become nodes.
- Concepts that co-occur in an exchange get an edge between them.
- Edges strengthen (Hebbian-style) every time their concepts co-occur again.
- Edges decay exponentially the longer they go unused (computed lazily,
  no background thread needed).
- Queries seed the graph and activation spreads outward through weighted
  edges (spreading activation), surfacing the most relevant associated
  memory for the current prompt.

Storage is a single SQLite file, so it's durable across CLI sessions
and cheap enough to keep entirely in RAM-cache on an 8GB machine.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    label           TEXT PRIMARY KEY,
    first_seen      REAL NOT NULL,
    last_seen       REAL NOT NULL,
    activation_count INTEGER NOT NULL DEFAULT 0,
    notes           TEXT NOT NULL DEFAULT '[]'  -- JSON list of short snippets
);

CREATE TABLE IF NOT EXISTS edges (
    node_a          TEXT NOT NULL,
    node_b          TEXT NOT NULL,
    weight          REAL NOT NULL DEFAULT 0.0,
    last_reinforced REAL NOT NULL,
    PRIMARY KEY (node_a, node_b),
    FOREIGN KEY (node_a) REFERENCES nodes(label),
    FOREIGN KEY (node_b) REFERENCES nodes(label)
);

CREATE INDEX IF NOT EXISTS idx_edges_a ON edges(node_a);
CREATE INDEX IF NOT EXISTS idx_edges_b ON edges(node_b);
"""


# Query terms shorter than this are ignored as seeds -- a 1-2 letter word is
# a substring of nearly every label and would otherwise trigger spurious
# full-strength (1.0) activation on unrelated concepts.
MIN_SEED_TERM_LEN = 3

# Query terms need to be at least this long before they're allowed to match
# as a substring of a single label word (e.g. 'gpu' -> 'gpus'). Below this,
# only an exact whole-word match counts.
MIN_SUBSTRING_TERM_LEN = 4

_QUERY_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "if", "then", "so", "for", "to", "of", "in", "on",
    "at", "by", "with", "as", "it", "its", "this", "that", "i", "you", "we",
    "they", "he", "she", "do", "does", "did", "can", "could", "will",
    "would", "should", "have", "has", "had", "not", "no", "yes", "what",
    "how", "why", "when", "who", "there", "here", "just", "also", "any",
}


def _canon(label: str) -> str:
    """Canonicalize a concept label so 'GPU', 'gpu ', 'Gpu' collapse to one node."""
    return " ".join(label.strip().lower().split())


def _edge_key(a: str, b: str) -> Tuple[str, str]:
    """Edges are undirected; store with a stable ordering."""
    a, b = _canon(a), _canon(b)
    return (a, b) if a <= b else (b, a)


@dataclass
class ActivatedNode:
    label: str
    activation: float
    notes: List[str] = field(default_factory=list)


class MemoryGraph:
    def __init__(
        self,
        db_path: str | Path,
        decay_half_life_seconds: float = 60 * 60 * 24 * 3,  # 3 days
        reinforcement_gain: float = 1.0,
        max_edge_weight: float = 10.0,
    ):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

        self.decay_half_life_seconds = decay_half_life_seconds
        self.reinforcement_gain = reinforcement_gain
        self.max_edge_weight = max_edge_weight

    # ------------------------------------------------------------------ #
    # Decay
    # ------------------------------------------------------------------ #
    def _decayed_weight(self, raw_weight: float, last_reinforced: float, now: float) -> float:
        """Exponential decay: weight halves every decay_half_life_seconds of disuse."""
        elapsed = max(0.0, now - last_reinforced)
        if self.decay_half_life_seconds <= 0:
            return raw_weight
        decay_factor = 0.5 ** (elapsed / self.decay_half_life_seconds)
        return raw_weight * decay_factor

    # ------------------------------------------------------------------ #
    # Writing: reinforce concepts + their co-occurrence
    # ------------------------------------------------------------------ #
    def touch_node(self, label: str, note: Optional[str] = None, max_notes: int = 3) -> None:
        label = _canon(label)
        if not label:
            return
        now = time.time()
        cur = self.conn.execute("SELECT notes, activation_count FROM nodes WHERE label = ?", (label,))
        row = cur.fetchone()
        if row is None:
            notes = [note] if note else []
            self.conn.execute(
                "INSERT INTO nodes (label, first_seen, last_seen, activation_count, notes) "
                "VALUES (?, ?, ?, 1, ?)",
                (label, now, now, json.dumps(notes)),
            )
        else:
            notes = json.loads(row[0])
            if note:
                notes.append(note)
                notes = notes[-max_notes:]
            self.conn.execute(
                "UPDATE nodes SET last_seen = ?, activation_count = activation_count + 1, notes = ? "
                "WHERE label = ?",
                (now, json.dumps(notes), label),
            )

    def reinforce_edge(self, a: str, b: str) -> None:
        a, b = _edge_key(a, b)
        if a == b:
            return
        now = time.time()
        cur = self.conn.execute(
            "SELECT weight, last_reinforced FROM edges WHERE node_a = ? AND node_b = ?", (a, b)
        )
        row = cur.fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO edges (node_a, node_b, weight, last_reinforced) VALUES (?, ?, ?, ?)",
                (a, b, self.reinforcement_gain, now),
            )
        else:
            current = self._decayed_weight(row[0], row[1], now)
            new_weight = min(self.max_edge_weight, current + self.reinforcement_gain)
            self.conn.execute(
                "UPDATE edges SET weight = ?, last_reinforced = ? WHERE node_a = ? AND node_b = ?",
                (new_weight, now, a, b),
            )

    def record_exchange(self, concepts: List[str], note: Optional[str] = None) -> None:
        """
        Call this once per turn with the concepts extracted from that exchange.
        Every concept gets touched (reinforced as a node); every pair that
        co-occurred gets its edge strengthened. This is the Hebbian step.
        """
        concepts = [_canon(c) for c in concepts if _canon(c)]
        concepts = list(dict.fromkeys(concepts))  # de-dup, keep order
        for c in concepts:
            self.touch_node(c, note=note)
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                self.reinforce_edge(concepts[i], concepts[j])
        self.conn.commit()

    def forget(self, label: str) -> bool:
        label = _canon(label)
        cur = self.conn.execute("DELETE FROM nodes WHERE label = ?", (label,))
        self.conn.execute("DELETE FROM edges WHERE node_a = ? OR node_b = ?", (label, label))
        self.conn.commit()
        return cur.rowcount > 0

    def reset(self) -> None:
        self.conn.execute("DELETE FROM nodes")
        self.conn.execute("DELETE FROM edges")
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Reading: neighbors, stats
    # ------------------------------------------------------------------ #
    def neighbors(self, label: str) -> List[Tuple[str, float]]:
        label = _canon(label)
        now = time.time()
        rows = self.conn.execute(
            "SELECT node_a, node_b, weight, last_reinforced FROM edges WHERE node_a = ? OR node_b = ?",
            (label, label),
        ).fetchall()
        out = []
        for a, b, w, last in rows:
            other = b if a == label else a
            out.append((other, self._decayed_weight(w, last, now)))
        out.sort(key=lambda x: -x[1])
        return out

    def find_matching_nodes(self, query_terms: List[str]) -> List[str]:
        """Cheap substring/fuzzy seed matching -- no embedding model required.

        Two guards keep this from over-triggering:
          - short/common words (stopwords, or anything under MIN_SEED_TERM_LEN)
            are skipped entirely, since a bare 'a' or 'an' is a substring of
            almost every label and would seed the whole graph at activation 1.0.
          - matching against a multi-word label is done per-word (or as a whole
            phrase), not as a raw substring of the joined string, so a short
            query term can't accidentally match inside an unrelated word
            (e.g. 'an' inside 'grand', 'tequila bar', etc.).
        """
        all_labels = [r[0] for r in self.conn.execute("SELECT label FROM nodes").fetchall()]
        seeds = set()
        for term in query_terms:
            t = _canon(term)
            if not t or len(t) < MIN_SEED_TERM_LEN or t in _QUERY_STOPWORDS:
                continue
            for label in all_labels:
                label_words = label.split()
                if t == label:
                    seeds.add(label)
                    continue
                # whole-word match against any word in a multi-word label
                if t in label_words:
                    seeds.add(label)
                    continue
                # longer terms may still match as a substring of a single word
                # (e.g. 'gpu' matching 'gpus'), but never across word boundaries
                if len(t) >= MIN_SUBSTRING_TERM_LEN and any(
                    t in w or w in t for w in label_words if len(w) >= MIN_SEED_TERM_LEN
                ):
                    seeds.add(label)
        return list(seeds)

    def spreading_activation(
        self,
        query_terms: List[str],
        depth: int = 2,
        decay_per_hop: float = 0.55,
        activation_floor: float = 0.05,
        top_k: int = 8,
    ) -> List[ActivatedNode]:
        """
        Seed nodes get activation 1.0; activation spreads to neighbors scaled
        by (edge weight, normalized) * decay_per_hop, breadth-first, up to `depth`
        hops. This is the retrieval mechanism: what's relevant right now is
        whatever the graph's current wiring says is strongly associated with
        the query, not just exact keyword hits.
        """
        seeds = self.find_matching_nodes(query_terms)
        if not seeds:
            return []

        activation: Dict[str, float] = {s: 1.0 for s in seeds}
        frontier = list(seeds)

        for _ in range(depth):
            next_frontier: Dict[str, float] = {}
            for node in frontier:
                base_activation = activation.get(node, 0.0)
                if base_activation < activation_floor:
                    continue
                neighs = self.neighbors(node)
                if not neighs:
                    continue
                max_w = max(w for _, w in neighs) or 1.0
                for other, w in neighs:
                    spread = base_activation * (w / max_w) * decay_per_hop
                    if spread < activation_floor:
                        continue
                    total = activation.get(other, 0.0) + spread
                    activation[other] = min(1.0, total)
                    next_frontier[other] = activation[other]
            frontier = list(next_frontier.keys())
            if not frontier:
                break

        ranked = sorted(activation.items(), key=lambda x: -x[1])[:top_k]
        results = []
        for label, score in ranked:
            row = self.conn.execute("SELECT notes FROM nodes WHERE label = ?", (label,)).fetchone()
            notes = json.loads(row[0]) if row else []
            results.append(ActivatedNode(label=label, activation=score, notes=notes))
        return results

    def stats(self) -> Dict[str, object]:
        n_nodes = self.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        n_edges = self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        top = self.conn.execute(
            "SELECT label, activation_count FROM nodes ORDER BY activation_count DESC LIMIT 10"
        ).fetchall()
        return {"nodes": n_nodes, "edges": n_edges, "top_nodes": top}

    def close(self) -> None:
        self.conn.close()
