"""Convert one PDF to a structured JSON document with Docling.

    python extract.py report.pdf
    python extract.py report.pdf --device cuda
    python extract.py report.pdf --out ./out --mode fast

Converts in page batches, so that:
  * progress is real (pages actually finished), not a clock-based guess
  * peak memory is bounded by the batch, not the document
  * a crash costs one batch — rerun and it resumes from the parts on disk

Output per document, in --out:
    <name>.<mode>.json        the full DoclingDocument
    <name>.<mode>.md          markdown, for reading
    <name>.<mode>.meta.json   timings and counts
    parts/<name>.<mode>/      per-batch files, the resume points
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

MODE_DEFAULT = "accurate"


def page_count(pdf: Path) -> int:
    import pypdfium2

    doc = pypdfium2.PdfDocument(str(pdf))
    try:
        return len(doc)
    finally:
        doc.close()


def build_converter(mode: str, threads: int, device: str, ocr: bool):
    from docling.datamodel.accelerator_options import (
        AcceleratorDevice,
        AcceleratorOptions,
    )
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption

    # OCR is off by default. Born-digital PDFs already carry their text layer,
    # and running OCR over one risks replacing correct embedded characters with
    # recognised ones. Turn it on only for scanned documents.
    opts = PdfPipelineOptions(do_table_structure=True, do_ocr=ocr)
    opts.table_structure_options.mode = (
        TableFormerMode.ACCURATE if mode == "accurate" else TableFormerMode.FAST
    )
    # Map the predicted table structure back onto the PDF's real text cells, so
    # extracted digits are the ones in the file rather than the model's reading
    # of an image. Important for anything numeric.
    opts.table_structure_options.do_cell_matching = True

    opts.accelerator_options = AcceleratorOptions(
        num_threads=threads, device=AcceleratorDevice(device)
    )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


class Heartbeat:
    """Proof of life during a blocking convert() call.

    Reports elapsed time only. Within a batch there is no way to know how far
    along we are, and an invented percentage invites trust it has not earned.
    Real progress is reported between batches, in pages actually finished.
    """

    def __init__(self, every: float = 15.0) -> None:
        import threading

        self._every = every
        self._stop = threading.Event()
        self._threading = threading

    def __enter__(self) -> "Heartbeat":
        started = time.perf_counter()

        def run() -> None:
            while not self._stop.wait(self._every):
                mins, secs = divmod(int(time.perf_counter() - started), 60)
                print(f"      ... still working  {mins}m{secs:02d}s", flush=True)

        self._threading.Thread(target=run, daemon=True).start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()


def batches(total: int, size: int) -> list[tuple[int, int]]:
    return [(s, min(s + size - 1, total)) for s in range(1, total + 1, size)]


def convert(
    pdf: Path,
    out: Path,
    mode: str = MODE_DEFAULT,
    threads: int = 8,
    device: str = "auto",
    ocr: bool = False,
    batch: int = 12,
    resume: bool = True,
    quiet: bool = False,
) -> dict:
    from docling_core.types.doc import ImageRefMode
    from docling_core.types.doc.document import DoclingDocument

    stem = f"{pdf.stem}.{mode}" + (".ocr" if ocr else "")
    out.mkdir(parents=True, exist_ok=True)
    parts_dir = out / "parts" / stem
    parts_dir.mkdir(parents=True, exist_ok=True)

    total = page_count(pdf)
    plan = batches(total, batch)

    def say(msg: str) -> None:
        if not quiet:
            print(msg, flush=True)

    say(f"{pdf.name}  {total} pages  mode={mode}  device={device}  ocr={ocr}")
    say(f"{len(plan)} batches of {batch} pages, {threads} threads")

    converter = build_converter(mode, threads, device, ocr)

    done_pages = 0
    t_start = time.perf_counter()

    for i, (lo, hi) in enumerate(plan, 1):
        part = parts_dir / f"p{lo:04d}-{hi:04d}.json"
        span = hi - lo + 1
        head = f"[{i}/{len(plan)}] pages {lo}-{hi}"

        if part.exists() and resume:
            done_pages += span
            say(f"{head}  cached, skipping")
            continue

        t0 = time.perf_counter()
        with Heartbeat():
            result = converter.convert(str(pdf), page_range=(lo, hi))
        took = time.perf_counter() - t0

        result.document.save_as_json(part, image_mode=ImageRefMode.PLACEHOLDER)

        done_pages += span
        remaining = (total - done_pages) * (time.perf_counter() - t_start) / done_pages
        say(
            f"{head}  {took:5.1f}s  ({took / span:.1f}s/page)   "
            f"{done_pages}/{total} pages, {100 * done_pages / total:.0f}%   "
            f"~{remaining / 60:.0f} min left"
        )

        del result
        gc.collect()

    say("merging parts...")
    docs = [DoclingDocument.load_from_json(p) for p in sorted(parts_dir.glob("p*.json"))]
    merged = DoclingDocument.concatenate(docs)

    json_path = out / f"{stem}.json"
    merged.save_as_json(json_path, image_mode=ImageRefMode.PLACEHOLDER)
    (out / f"{stem}.md").write_text(merged.export_to_markdown(), encoding="utf-8")

    elapsed = time.perf_counter() - t_start
    pages_seen = sorted(merged.pages)
    meta = {
        "pdf": str(pdf),
        "mode": mode,
        "device": device,
        "ocr": ocr,
        "pdf_pages": total,
        "batch_size": batch,
        "batches": len(plan),
        "seconds": round(elapsed, 1),
        "seconds_per_page": round(elapsed / total, 2),
        "merged_pages": len(merged.pages),
        "page_range_seen": [pages_seen[0], pages_seen[-1]] if pages_seen else None,
        "tables": len(merged.tables),
        "texts": len(merged.texts),
    }
    (out / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2))

    if not quiet:
        print()
        for k, v in meta.items():
            print(f"  {k:18} {v}")

    # Page numbering surviving a batched convert is the one thing that could
    # silently corrupt every page reference downstream. Say so loudly.
    if pages_seen and (pages_seen[0], pages_seen[-1]) != (1, total):
        print(
            f"\n  !! merged pages run {pages_seen[0]}..{pages_seen[-1]}, "
            f"expected 1..{total} — page numbers need remapping before use",
            file=sys.stderr,
        )

    say(f"\nwrote {json_path}")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="PDF -> structured JSON via Docling")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--mode", choices=["accurate", "fast"], default=MODE_DEFAULT)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps", "xpu"],
                    default="auto")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=12, help="pages per batch")
    ap.add_argument("--ocr", action="store_true", help="scanned PDFs only")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    if not args.pdf.exists():
        sys.exit(f"no such file: {args.pdf}")

    convert(
        args.pdf,
        args.out,
        mode=args.mode,
        threads=args.threads,
        device=args.device,
        ocr=args.ocr,
        batch=args.batch,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
