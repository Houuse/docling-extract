# docling-extract

Turns PDFs into structured JSON documents — sections, tables as row/column
grids, and every element carrying its page number and bounding box. This is the
slow, CPU-or-GPU-intensive step of a document pipeline, split out so it can run
on whatever machine has the hardware and the results shared afterwards.

Nothing here needs a database or an embedding model.

## Quick start

One script does everything — installs, verifies the GPU is genuinely in use,
and converts. Works in Git Bash on Windows and on Linux/macOS.

```bash
./setup.sh --pdfs ../financebench/pdfs
```

Useful variants:

```bash
./setup.sh --setup-only              # install and verify, convert nothing
./setup.sh --limit 3                 # convert 3 documents as a test
./setup.sh --cpu                     # no GPU
./setup.sh --python 3.11             # if 3.12 has no CUDA torch wheel
```

It pins the virtualenv to **Python 3.12** regardless of the system Python,
because torch's CUDA wheels lag new Python releases — on 3.14 there is no CUDA
build and you silently get a CPU-only one. It also refuses to continue if
`torch.cuda.is_available()` is false, rather than letting you discover it hours
into a run.

First run downloads model weights (about 1 GB) to `~/.cache`.

## Use the tools directly

The interpreter path differs by platform. Set it once:

```bash
PY=.venv/bin/python          # Linux, macOS, WSL2
PY=.venv/Scripts/python      # Windows (Git Bash)
```

```bash
$PY extract.py report.pdf --out ./out          # one document
$PY batch.py ./pdfs --out ./out                # a directory
$PY batch.py ./pdfs --out ./out --device cuda  # on a GPU
$PY batch.py ./pdfs --dry-run                  # show the plan, do nothing
```

Both are resumable: rerun the same command after an interruption and finished
work is skipped.

## Share the output

`parts/` holds per-batch resume files, redundant once the merged JSON exists and
about as large again. Exclude it. JSON compresses roughly 20:1.

```bash
tar --exclude=parts -czf corpus.tar.gz out/
```

GitHub Releases accept up to 2 GB per file and do not consume LFS quota, so a
release is a simpler home for this than git or LFS:

```bash
gh release create corpus-v1 corpus.tar.gz --notes "docling 2.122.0, accurate mode"
gh release download corpus-v1              # on the receiving machine
tar -xzf corpus.tar.gz
```

**Do not mix output from different docling versions into one corpus.** Half the
documents extracted by one model version and half by another puts an invisible
confound into everything downstream. Each `*.meta.json` records the version it
was produced with.

## Output

Per document, in `--out`:

| File | Contents |
|---|---|
| `<name>.accurate.json` | the full DoclingDocument — this is the artefact |
| `<name>.accurate.md` | markdown, for reading |
| `<name>.accurate.meta.json` | timings, page count, table count |
| `parts/<name>.accurate/` | per-batch files, the resume points |

The JSON is large: a 160-page filing produces 10–30 MB. `parts/` is roughly the
same again and can be deleted once the merged JSON exists.

Read it back with:

```python
from docling_core.types.doc.document import DoclingDocument
doc = DoclingDocument.load_from_json("out/report.accurate.json")
doc.tables[0].export_to_dataframe(doc=doc)
```

## Configuration that matters

**`--mode accurate` (default)** uses the larger TableFormer model for table
structure. `fast` is the alternative; the quality difference on a given corpus
is worth measuring rather than assuming.

**`do_cell_matching=True`** (always on) maps the predicted table structure back
onto the PDF's real text cells, so extracted digits are the characters in the
file rather than a model's reading of an image. Do not turn this off for
anything numeric.

**OCR is off by default.** Born-digital PDFs already carry a text layer, and
OCR over one risks replacing correct characters with recognised ones. Use
`--ocr` only for scanned documents.

**`--batch 12`** converts twelve pages at a time. Batching bounds peak memory,
makes progress real rather than estimated, and makes a crash cost one batch
instead of a whole document. Rerunning resumes from `parts/`.

**`--workers 1`** by default. Docling peaks at several GB per document; raise
it only if you have the RAM, or the OOM killer will take processes silently.

## Known behaviour worth knowing

**Tables do not merge across page breaks.** Docling assembles tables strictly
per page and gives each a single-page provenance entry. A financial statement
spanning two pages arrives as two tables, and the second carries no header row.
Detect continuations by looking for a table with no `column_header` cells whose
column count matches a table on the previous page.

**Heading classification is inconsistent.** Identically formatted lines can come
back as `section_header` or as plain `text`. If you rely on headings, also
accept short lines ending in a colon.

**Page labels are not page indexes.** The printed page number in a document may
differ from its physical position. Locate pages by content, not by a stated
number.

**A batched convert preserves absolute page numbers** — verified: a 160-page
document merges back to pages 1..160. `extract.py` checks this after every
merge and warns loudly on stderr if it ever stops holding.

## Performance

Measured on 8 CPU cores, no GPU, `accurate` mode, OCR off:

| | |
|---|---|
| Prose-heavy pages | ~1.3 s/page |
| Table-heavy pages | ~4 s/page |
| A 160-page 10-K | ~5 min |

Cost is dominated by the layout model and TableFormer, not by reading the PDF.
A CUDA GPU should give roughly 3–8x end to end — less than the model speedup
alone, because PDF parsing, page rasterisation and orchestration stay on the
CPU.
