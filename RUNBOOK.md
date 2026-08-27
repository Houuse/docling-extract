# Runbook: extract the corpus on a GPU machine and share it

Copy-paste, top to bottom. Each step says what to expect so you can tell a
problem from normal slowness.

## Windows note — read this first

Commands below use `.venv/bin/python`, which is the Linux and macOS layout.
**On Windows the interpreter is `.venv/Scripts/python`.** Substitute it
everywhere, or set a variable once in Git Bash and use `$PY`:

```bash
PY=.venv/Scripts/python      # Windows (Git Bash / PowerShell)
PY=.venv/bin/python          # Linux, macOS, WSL2
```

Everything else works the same in Git Bash, forward slashes included. CUDA
torch has native Windows wheels, so WSL2 is not required — though WSL2 also
works if you prefer it, with the Linux paths.

The two Windows-specific substitutions further down:

| Step | Linux | Windows |
|---|---|---|
| interpreter | `.venv/bin/python` | `.venv/Scripts/python` |
| compression | `zstd` | `tar -czf` (gzip, built in) |

## 1. Check the machine

```bash
nvidia-smi
python --version      # try python3 if that is not found
```

Expect a GPU table with a driver and CUDA version, and Python 3.10 or newer.
No GPU listed means `--device cuda` will not work — stop and fix the driver
first.

## 2. Get the code and the PDFs

```bash
cd ~
git clone https://github.com/Houuse/docling-extract
git clone https://github.com/patronus-ai/financebench
cd docling-extract
```

`financebench/pdfs/` should hold 368 PDFs:

```bash
ls ../financebench/pdfs/*.pdf | wc -l
```

## 3. Install

Install CUDA torch **first**. If you install docling first, pip resolves the
CPU-only wheel and the GPU sits idle.

```bash
pip install uv
uv venv --python 3.12
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt
```

Check torch can actually see the GPU:

```bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Must print `True` and your GPU's name. If it prints `False`, the CUDA wheel did
not take — redo this step before going further.

## 4. Convert one document as a test

```bash
.venv/bin/python extract.py ../financebench/pdfs/3M_2018_10K.pdf --out ./out --device cuda
```

Expect:

- roughly 160 pages in 14 batches
- `seconds_per_page` well under 1.8 (that was the CPU baseline on 8 cores)
- `merged_pages 160` and `page_range_seen [1, 160]`
- **no** `!!` warning about page numbers

While it runs, in another terminal:

```bash
nvidia-smi
```

A python process should be holding VRAM. If nothing is, it is running on CPU
regardless of the flag — stop and go back to step 3.

## 5. Convert everything

```bash
.venv/bin/python batch.py ../financebench/pdfs --out ./out --device cuda
```

368 documents. Resumable: re-run the same command after any interruption and
finished documents are skipped. Progress and failures also go to
`out/batch.log`.

If the machine has plenty of RAM and VRAM, try more at once:

```bash
.venv/bin/python batch.py ../financebench/pdfs --out ./out --device cuda --workers 2
```

Watch memory on the first few documents before raising it further. Docling
peaks at several GB per document, and the OOM killer removes processes without
a traceback.

## 6. Check the result

```bash
ls out/*.accurate.json | wc -l          # expect 368
grep -c FAIL out/batch.log || true      # expect 0
tail -20 out/batch.log
```

Any failures are listed at the end of the log. Re-running the batch retries
only those.

## 7. Package it

`parts/` is per-batch resume files, redundant once the merged JSON exists, and
about as large again. Exclude it. JSON compresses roughly 20:1.

Linux or macOS, with zstd:

```bash
tar --exclude=parts -cf - out/ | zstd -10 -o corpus-368.tar.zst
ls -lh corpus-368.tar.zst
```

Windows — `tar` is built in and does gzip, so no install needed:

```bash
tar --exclude=parts -czf corpus-368.tar.gz out/
ls -lh corpus-368.tar.gz
```

Expect around 150 MB with zstd, around 200 MB with gzip. The difference is not
worth installing anything for.

## 8. Publish it

FinanceBench is permissively licensed and the underlying SEC filings are public
records, so publishing the extractions is fine. Keep the attribution in the
release notes.

Use whichever archive you produced in step 7 — `.tar.zst` or `.tar.gz`.

```bash
gh auth login
gh release create corpus-v1 corpus-368.tar.zst \
  --repo Houuse/docling-extract \
  --title "Extracted corpus (368 filings)" \
  --notes "Docling extractions of the FinanceBench corpus (patronus-ai/financebench).
docling 2.122.0, accurate mode, do_cell_matching, OCR off."
```

GitHub Releases take up to 2 GB per file and do not consume LFS quota, so no
LFS setup is needed.

## 9. On the receiving machine

```bash
gh release download corpus-v1 --repo Houuse/docling-extract

# zstd archive
zstd -d corpus-368.tar.zst -c | tar -xf -

# gzip archive (Windows-produced)
tar -xzf corpus-368.tar.gz
```

Read a document back:

```python
from docling_core.types.doc.document import DoclingDocument
doc = DoclingDocument.load_from_json("out/3M_2018_10K.accurate.json")
print(len(doc.pages), len(doc.tables))
doc.tables[40].export_to_dataframe(doc=doc)
```

## Note on mixing machines

Do not merge output produced by different docling versions into one corpus.
Half the documents extracted by one model version and half by another puts an
invisible confound into every downstream measurement. Let one machine produce
the whole set, and record the version — it is in each `*.meta.json`.
