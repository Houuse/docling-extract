"""Embed chunk and fact strings, caching to disk by content.

Embedding is the expensive half of ingestion and the half a GPU actually
helps with — unlike conversion, measured at 2% GPU utilisation. Keeping it
here lets it run on whatever machine has the hardware, with the results
carried to wherever the database lives.

The cache is keyed by an md5 of the exact strings passed in, so a failure
anywhere downstream — a bad insert, a dropped connection — costs no
re-encoding, and a machine that has already embedded a document contributes
that work to any other machine it copies the file to.
"""

import sys
from pathlib import Path

EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBED_DIM = 768
EMBED_MAX_TOKENS = 8192
DOC_PREFIX = "search_document: "  # ADR 0003: asymmetric, mandatory, silent if wrong
QUERY_PREFIX = "search_query: "  # the other half of the pair; never mix them

def embed(texts: list[str], stem: str, cache: Path, batch: int = 8):
    """Embed, caching to disk keyed by the text content.

    Embedding is the expensive step (minutes on CPU). A failure anywhere after
    it — a bad insert, a dropped connection — must not cost it again.
    """
    import hashlib

    import numpy as np

    key = hashlib.md5(
        ("\x00".join(texts)).encode(), usedforsecurity=False
    ).hexdigest()[:16]
    cached = Path(cache) / f"{stem}.emb.{key}.npy"
    if cached.exists():
        vecs = np.load(cached)
        print(f"reusing cached embeddings {cached.name}  {vecs.shape}")
        return vecs

    vecs = _encode(texts, batch)
    np.save(cached, vecs)
    print(f"cached embeddings → {cached.name}")
    return vecs


def _encode(texts: list[str], batch: int):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
    # SentenceTransformer carries its own sequence cap, independent of the
    # tokenizer's model_max_length. If it is lower than the chunker's budget,
    # long chunks are truncated here with no error at all.
    if model.max_seq_length < EMBED_MAX_TOKENS:
        print(
            f"raising max_seq_length {model.max_seq_length} -> {EMBED_MAX_TOKENS}"
        )
        model.max_seq_length = EMBED_MAX_TOKENS

    # Sort by length so long chunks batch with long chunks. Mixed batches pad
    # every short chunk up to the longest one, wasting time and memory.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    vecs_sorted = model.encode(
        [DOC_PREFIX + texts[i] for i in order],
        batch_size=batch,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    import numpy as np

    vecs = np.empty_like(vecs_sorted)
    vecs[order] = vecs_sorted
    if vecs.shape[1] != EMBED_DIM:
        sys.exit(f"expected {EMBED_DIM} dims, model gave {vecs.shape[1]}")
