from pathlib import Path

# Default model: 4-bit quantized, ~1.8GB on disk, comfortably fits an 8GB Mac
# alongside the OS and the graph store. Swap for a bigger one if you have
# more headroom (see README for options).
DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"

DEFAULT_DATA_DIR = Path.home() / ".llm-plasticity"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "memory.db"

# Plasticity tuning
DECAY_HALF_LIFE_SECONDS = 60 * 60 * 24 * 3   # unused connections halve in strength every 3 days
REINFORCEMENT_GAIN = 1.0                      # how much a co-occurrence strengthens an edge
MAX_EDGE_WEIGHT = 10.0                        # cap so a few topics can't dominate forever
SPREADING_DEPTH = 2                           # how many hops activation travels outward
SPREADING_DECAY_PER_HOP = 0.55                # activation attenuation per hop
ACTIVATION_FLOOR = 0.05                       # stop spreading below this activation level
TOP_K_MEMORY_NODES = 8                        # max nodes folded into context per turn

# Generation
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7

# Code export: where saved source files go when you accept the save prompt.
# Write-only -- the tool never reads from this directory, only writes to it.
CODE_EXPORT_DIR = DEFAULT_DATA_DIR / "generated_code"
