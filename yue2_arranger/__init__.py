"""YuE2 score tools: make planner scores longer, more varied and instrumental; measure how repetitive a render is."""
from .score import ScoreError, parse, describe, stats, arrange, make_instrumental, transpose_text, semitones_to_key  # noqa: F401
from .auto import DEFAULT_TERMS, auto_arrange, select_scores, style_variants  # noqa: F401

__version__ = "0.1.0"
