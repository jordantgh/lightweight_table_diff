"""table_diff — cell-level table diffing for Polars."""
from .core import batch_diff_tbls, diff_tbls
from .runner import run_comparison, run_config

__all__ = ["diff_tbls", "batch_diff_tbls", "run_comparison", "run_config"]