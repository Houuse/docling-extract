"""Embed converted documents, writing the .npy caches a loader can reuse.

Conversion is the half a GPU turns out not to help with — measured at 2%
utilisation, see the README. Embedding is the half it should: a few hundred
thousand batched transformer forward passes. This runs only that half, needs
no database, and leaves cache files behind.

    python embed.py --all --out ./out

Copy the resulting `*.emb.*.npy` next to the JSON on whatever machine owns the
database. A loader that calls `docling_extract.embedding.embed` with the same
strings finds them cached and skips encoding entirely.

Resumable at document granularity: a document whose .npy already exists is
skipped, so rerunning after an interruption is free.
"""

import argparse
import sys
import time
from pathlib import Path

from docling_extract import chunking
from docling_extract import embedding as emb


def build_texts(out: Path, stem: str):
    """The chunk and fact strings for one document.

    Fact strings are built from the *scrubbed* heading trail, and scrubbing
    happens before embedding so that a vector and the text a consumer stores
    come from the same characters.
    """
    from docling_core.types.doc.document import DoclingDocument

    path = out / f"{stem}.json"
    if not path.exists():
        return None, f"no cached document at {path}"

    doc = DoclingDocument.load_from_json(path)
    meta = chunking.doc_meta(stem, doc)
    sections = chunking.item_sections(doc)

    chunks = chunking.prose_chunks(doc, sections) + chunking.table_chunks(doc, sections)
    facts = [f for i in range(len(doc.tables)) for f in chunking.table_facts(doc, i)]

    for c in chunks:
        c.text = chunking.scrub(c.text)
        c.item_section = chunking.scrub(c.item_section) or None
        c.heading_trail = [chunking.scrub(h) for h in c.heading_trail]
    for f in facts:
        f.row_label = chunking.scrub(f.row_label)
        f.column_label = chunking.scrub(f.column_label)
        f.value_raw = chunking.scrub(f.value_raw)
    chunks = [c for c in chunks if c.text.strip()]

    trail_of = {
        c.table_ordinal: (c.heading_trail[-1] if c.heading_trail else None)
        for c in chunks
        if c.kind == "table"
    }
    fact_texts = [
        chunking.fact_text(meta.company, trail_of.get(f.table_ordinal), f)
        for f in facts
    ]
    return (meta, [c.text for c in chunks], fact_texts), None


def install_encoder(device: str | None, batch: int) -> str:
    """Replace embedding._encode with one that loads the model once.

    The library builds a SentenceTransformer per call, which is right for one
    document and wasteful for several hundred. Vectors are unaffected: same
    model, same prefix, same normalisation, same sequence cap.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        sys.exit(
            "--device cuda but torch reports no CUDA. Recent Python versions "
            "have no CUDA wheel and the resolver installs a CPU-only build; "
            "setup.sh checks for this."
        )

    model = SentenceTransformer(emb.EMBED_MODEL, trust_remote_code=True, device=device)
    # SentenceTransformer's own cap is independent of the tokenizer's and is
    # lower than the chunker's budget, so long chunks would be truncated here
    # with no error at all.
    if model.max_seq_length < emb.EMBED_MAX_TOKENS:
        print(f"raising max_seq_length {model.max_seq_length} -> {emb.EMBED_MAX_TOKENS}")
        model.max_seq_length = emb.EMBED_MAX_TOKENS

    def _encode(texts: list[str], _batch: int):
        import numpy as np

        # Sort by length so long chunks batch with long chunks; mixed batches
        # pad every short chunk up to the longest one.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        vecs_sorted = model.encode(
            [emb.DOC_PREFIX + texts[i] for i in order],
            batch_size=batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vecs = np.empty_like(vecs_sorted)
        vecs[order] = vecs_sorted
        if vecs.shape[1] != emb.EMBED_DIM:
            sys.exit(f"expected {emb.EMBED_DIM} dims, model gave {vecs.shape[1]}")
        return vecs

    emb._encode = _encode
    return device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="*", help="document stems, e.g. 3M_2018_10K.accurate")
    ap.add_argument("--all", action="store_true", help="every *.json in --out")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("./out"),
        help="directory holding <stem>.json; also where .npy files are written",
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=64,
        help="encode batch size; 8 suits a CPU, a GPU wants far more",
    )
    ap.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    args = ap.parse_args()

    out = args.out.resolve()
    if not out.is_dir():
        sys.exit(f"no such directory: {out}")

    if args.all:
        stems = sorted(
            p.name[: -len(".json")]
            for p in out.glob("*.json")
            if not p.name.endswith(".meta.json")
        )
    else:
        stems = args.stems
    if not stems:
        sys.exit("nothing to do: pass stems or --all")

    device = install_encoder(args.device, args.batch)
    print(f"device {device}  batch {args.batch}  out {out}")
    print(f"{len(stems)} document(s)\n")

    t0 = time.perf_counter()
    done = 0
    failures: list[tuple[str, str]] = []

    for i, stem in enumerate(stems, 1):
        try:
            built, err = build_texts(out, stem)
            if err:
                failures.append((stem, err))
                print(f"[{i}/{len(stems)}] SKIP {stem}: {err}")
                continue
            meta, chunk_texts, fact_texts = built

            if chunk_texts:
                emb.embed(chunk_texts, stem, out, args.batch)
            if fact_texts:
                emb.embed(fact_texts, f"{meta.doc_name}.facts", out, args.batch)

            done += 1
            eta = (time.perf_counter() - t0) / done * (len(stems) - done) / 60
            print(
                f"[{i}/{len(stems)}] {stem}  "
                f"{len(chunk_texts)} chunks  {len(fact_texts)} facts  "
                f"(~{eta:.0f} min left)"
            )
        except Exception as exc:  # one bad document must not end the run
            failures.append((stem, f"{type(exc).__name__}: {exc}"))
            print(f"[{i}/{len(stems)}] FAIL {stem}: {type(exc).__name__}: {exc}")

    mins = (time.perf_counter() - t0) / 60
    print(f"\n=== done: {done}/{len(stems)} in {mins:.0f} min, {len(failures)} failed")
    for stem, detail in failures:
        print(f"    {stem}: {detail}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
