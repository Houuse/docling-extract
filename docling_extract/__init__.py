"""Extraction, chunking and embedding for PDF corpora.

The command-line tools at the repository root are the usual entry points:

    extract.py   one PDF  -> DoclingDocument JSON
    batch.py     many PDFs, resumable
    embed.py     cached JSON -> .npy embedding caches

This package holds the parts a consumer imports rather than runs:

    chunking     DoclingDocument -> chunks and facts
    embedding    strings -> vectors, cached by content hash

Both are deliberately free of storage concerns. What you do with the chunks —
which database, which schema — is the caller's business.
"""

from docling_extract import chunking, embedding

__all__ = ["chunking", "embedding"]
