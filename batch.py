"""Convert a directory of PDFs. Resumable, one document at a time.

    python batch.py ./pdfs
    python batch.py ./pdfs --device cuda --out ./out
    python batch.py ./pdfs --limit 10
    python batch.py ./pdfs --dry-run

Each document runs in its own subprocess. That costs a few seconds of model
loading per document and buys isolation: a PDF that crashes Docling or gets
OOM-killed takes down one document, not the run.

Sequential on purpose. Docling peaks at several GB per document, so parallel
documents risk the OOM killer. Raise --workers only if you have the RAM.
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable


def log(path: Path, msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run_one(pdf: Path, out: Path, args) -> tuple[Path, bool, str]:
    cmd = [
        PY, str(HERE / "extract.py"), str(pdf),
        "--out", str(out),
        "--mode", args.mode,
        "--device", args.device,
        "--threads", str(args.threads),
        "--batch", str(args.batch),
    ]
    if args.ocr:
        cmd.append("--ocr")

    proc = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603
    if proc.returncode == 0:
        tail = [ln for ln in (proc.stdout or "").splitlines() if "seconds_per_page" in ln]
        return pdf, True, (tail[-1].strip() if tail else "ok")

    if proc.returncode < 0:
        # Killed by a signal: no traceback, typically the OOM killer.
        return pdf, False, f"killed by signal {-proc.returncode} (likely OOM)"
    lines = (proc.stderr or proc.stdout or "").strip().splitlines()
    return pdf, False, lines[-1] if lines else f"exit {proc.returncode}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert a directory of PDFs")
    ap.add_argument("pdf_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--mode", choices=["accurate", "fast"], default="accurate")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps", "xpu"],
                    default="auto")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--workers", type=int, default=1,
                    help="documents in parallel; needs several GB of RAM each")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.pdf_dir.is_dir():
        sys.exit(f"not a directory: {args.pdf_dir}")

    pdfs = sorted(args.pdf_dir.glob("*.pdf"))
    suffix = f".{args.mode}" + (".ocr" if args.ocr else "")
    todo = [p for p in pdfs if not (args.out / f"{p.stem}{suffix}.json").exists()]
    if args.limit:
        todo = todo[: args.limit]

    print(f"pdfs found   {len(pdfs)}")
    print(f"already done {len(pdfs) - len([p for p in pdfs if not (args.out / f'{p.stem}{suffix}.json').exists()])}")
    print(f"to process   {len(todo)}")
    print(f"device       {args.device}   workers {args.workers}")

    if args.dry_run or not todo:
        for p in todo[:20]:
            print(f"  {p.name}")
        if len(todo) > 20:
            print(f"  ... {len(todo) - 20} more")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    logfile = args.out / "batch.log"
    log(logfile, f"=== start: {len(todo)} documents, device={args.device}")

    t0 = time.perf_counter()
    failures: list[tuple[str, str]] = []
    done = 0

    def report(res: tuple[Path, bool, str]) -> None:
        nonlocal done
        pdf, ok, detail = res
        if ok:
            done += 1
            elapsed = (time.perf_counter() - t0) / 60
            eta = elapsed / done * (len(todo) - done)
            log(logfile, f"[{done}/{len(todo)}] {pdf.name}  {detail}  (~{eta:.0f} min left)")
        else:
            failures.append((pdf.name, detail))
            log(logfile, f"FAIL {pdf.name}: {detail}")

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for res in pool.map(lambda p: run_one(p, args.out, args), todo):
                report(res)
    else:
        for p in todo:
            report(run_one(p, args.out, args))

    mins = (time.perf_counter() - t0) / 60
    log(logfile, f"=== done: {done}/{len(todo)} in {mins:.0f} min, {len(failures)} failed")
    for name, detail in failures:
        log(logfile, f"    {name}: {detail}")
    if failures:
        print("\nRe-run the same command to retry failures; finished documents are skipped.")
        sys.exit(1)


if __name__ == "__main__":
    main()
