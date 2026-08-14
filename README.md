# llm-plasticity

A command-line chat tool for macOS that pairs a small quantized open-weight
model with an evolving associative memory graph. As you use it, concepts you
talk about become nodes; concepts that come up together get connected; those
connections strengthen with repeated use and fade if left idle (Hebbian-style
reinforcement with time-based decay). Future prompts automatically pull in
whatever's currently well-connected to them via spreading activation, so the
tool's behavior shifts based on your usage history — without ever
retraining or fine-tuning the model itself.

**Status: early / not yet run on real hardware.** The memory graph
(`plasticity/memory_graph.py`) is unit-tested in plain Python — reinforcement,
decay, and spreading activation all behave as intended (see `tests/`). The
MLX-dependent half (model loading and generation in `plasticity/engine.py`)
has **not** been run end-to-end on an actual Mac yet, since it was built in
an environment without Apple hardware. Treat it as a working design, not a
verified one, until someone runs it on real Apple Silicon and reports back.
Issues and PRs on that front are very welcome.

## What this is (and isn't)

This is **not** a new model architecture and it's **not** trained from
scratch. The base model is an existing, openly released, pretrained
instruct model (Llama 3.2 or Qwen2.5, quantized to 4-bit via
[`mlx-community`](https://huggingface.co/mlx-community) on Hugging Face) —
swap in whichever `mlx-community` checkpoint fits your RAM. Nothing here
was distilled or scraped from any other AI product; the weights are exactly
what their respective publishers released.

What *is* new here is the layer wrapped around that model: a small
associative memory graph that reinforces and decays with use, described
below. It's closer to "online associative memory formation" than to
classical unsupervised learning (no clustering, no embedding training) —
it's Hebbian graph reinforcement, and that's the more precise name for it.

## Why this design, and what it isn't

A transformer's weights are frozen at inference time -- there's no
literal synaptic rewiring happening inside the model as you chat with it.
This tool doesn't pretend otherwise. What it does is put a real, working
analogue of plasticity *around* the model: a graph that reinforces with
use and decays with disuse (Hebbian-style), and a retrieval mechanism
(spreading activation) that lets that graph shape what the model sees
before it answers. The model's weights never change. The system's
behavior does.

If you later want actual weight-level plasticity (small LoRA updates
applied over time), that's a separate, riskier layer that can be added
on top of this one -- see "Extending to real weight updates" below.

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4). MLX is Metal-accelerated and
  doesn't run on Intel Macs or Linux.
- ~8GB RAM target. The default model (Llama 3.2 3B, 4-bit) is roughly
  2GB on disk and should use a similar order of magnitude resident,
  per its published quantized size — **this hasn't been measured on
  real hardware yet**, so treat "runs in 8GB" as the design target,
  not a confirmed benchmark. The SQLite-backed memory graph itself is
  negligible (single-digit MB even after thousands of exchanges).
- Python 3.10+.

## Setup

```bash
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
python -m plasticity.cli
```

First run downloads the model weights (~2GB) from Hugging Face and
caches them locally in `~/.cache/huggingface`. After that it's fully
offline.

## Usage

```
you> tell me about the sewer gas situation at the old place
llm> ...
  [memory: sewer gas(1.00), pwd case(0.55), attorney xavier(0.30)]
```

The `[memory: ...]` line (printed to stderr, so it won't pollute piped
output) shows what the graph activated for that prompt and how strongly
-- this is the plasticity in action.

Commands inside the REPL:

| Command             | Effect                                              |
|---------------------|------------------------------------------------------|
| `/graph`             | Show node/edge counts and most-activated concepts   |
| `/show <concept>`    | Show a concept's current neighbors and edge weights |
| `/forget <concept>`  | Remove a concept and its connections                |
| `/reset`             | Wipe the entire memory graph (asks for confirmation)|
| `/help`              | List commands                                        |
| `/exit`, `/quit`     | Quit (graph is already saved -- SQLite, no explicit save step needed) |

Flags:

```bash
python -m plasticity.cli --model mlx-community/Qwen2.5-3B-Instruct-4bit
python -m plasticity.cli --db ~/.llm-plasticity/work.db   # separate memory per project
```

## Swapping the base model

Any MLX-converted instruct model on Hugging Face under `mlx-community/`
works. For 8GB RAM, stay in the 1B-4B parameter range, 4-bit quantized:

- `mlx-community/Llama-3.2-3B-Instruct-4bit` (default) -- good general balance
- `mlx-community/Llama-3.2-1B-Instruct-4bit` -- fastest, most headroom, less capable
- `mlx-community/Qwen2.5-3B-Instruct-4bit` -- strong for size, good at structured output (helps concept extraction)
- `mlx-community/Phi-3.5-mini-instruct-4bit` -- ~3.8B, still fits, strong reasoning for its size

## How the plasticity loop works

1. **Retrieval (read):** your prompt's words seed matching nodes in the
   graph. Activation spreads outward through weighted edges (2 hops by
   default), attenuating at each hop. The most-activated nodes get
   folded into the model's context as a short memory block before it
   generates a reply.
2. **Generation:** the base model (unchanged) produces a response,
   informed by that memory block plus normal conversation history.
3. **Consolidation (write):** the model is asked, in a second small
   call, to name the 3-8 key concepts from that exchange. Those become
   (or reinforce existing) nodes, and every pair that co-occurred in
   the exchange gets its edge strengthened -- the Hebbian step.
4. **Decay:** edge weights aren't decremented on a timer. Instead,
   decay is computed lazily, whenever a weight is read, as a function
   of elapsed time since it was last reinforced (exponential half-life,
   3 days by default). Connections you keep using stay strong;
   connections you stop using fade on their own.

All of this lives in `plasticity/memory_graph.py`, independent of the
model -- it's tested with plain Python (`tests/test_memory_graph.py`,
no MLX/model dependency) so you can inspect and tune the decay/spread
math without needing a Mac.

## Tuning

Edit `plasticity/config.py`:

- `DECAY_HALF_LIFE_SECONDS` -- how fast unused connections fade. Lower
  = shorter memory, more responsive to recent topics. Higher = longer
  memory, more resistant to drift.
- `SPREADING_DEPTH` / `SPREADING_DECAY_PER_HOP` -- how far and how
  strongly activation travels. Depth 1 = only directly-mentioned
  concepts' immediate neighbors; depth 3 = looser, wider associations.
- `TOP_K_MEMORY_NODES` -- cap on how much activated memory gets folded
  into each prompt (keeps context short on a small model).

## Extending to real weight updates

The associative graph never touches the model's weights, which is why
it's safe to run indefinitely without risk of degradation. If you
later want actual plasticity at the weight level, MLX supports LoRA
fine-tuning (`mlx_lm.lora`) on Apple Silicon. A reasonable next step
would be a background process that periodically trains a small LoRA
adapter on a rolling window of (prompt, response) pairs the graph has
flagged as high-activation/high-recurrence -- i.e., use the graph as a
curator for *what's worth actually learning*, rather than training on
every exchange indiscriminately. That's meaningfully harder to get
right (catastrophic forgetting, needing held-out eval to catch drift)
and is deliberately not included in v1.

## Project layout

```
llm-plasticity/
  plasticity/
    memory_graph.py       # the plasticity engine: nodes, edges, decay, spreading activation
    concept_extractor.py  # LLM-driven concept extraction (with regex fallback)
    engine.py              # ties MLX inference to the memory graph
    cli.py                  # REPL
    config.py               # tuning knobs
  tests/
    test_memory_graph.py  # pure-Python tests, no MLX required
  setup.sh
  requirements.txt
```
