"""
SPSS .sav adapter.  Loads via pyreadstat into an in-memory Polars frame.
Normalisation (null alignment, trailing-zero stripping, etc.)
is left to the normaliser layer.
"""
import polars as pl


def load_sav(source_def: dict) -> pl.LazyFrame:
    try:
        import pyreadstat
    except ImportError:
        raise ImportError(
            "pyreadstat is required for .sav files: pip install pyreadstat"
        ) from None

    pdf, _meta = pyreadstat.read_sav(
        str(source_def["path"]),
        apply_value_formats=source_def.get("apply_value_formats", False),
        formats_as_category=False,
        user_missing=True,
    )
    return pl.from_pandas(pdf).lazy()