# docling-extract

Turns PDFs into structured JSON documents — sections, tables as row/column
grids, and every element carrying its page number and bounding box. This is the
slow, CPU-or-GPU-intensive step of a document pipeline, split out so it can run
on whatever machine has the hardware and the results shared afterwards.

Nothing here needs a database or an embedding model.

## Install

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

For an NVIDIA GPU, install a CUDA build of torch first, or `--device cuda`
will fail on the CPU-only wheel:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt
```

First run downloads model weights (about 1 GB) to `~/.cache`.

## Use

```bash
# one document
.venv/bin/python extract.py report.pdf --out ./out

# a directory
.venv/bin/python batch.py ./pdfs --out ./out

# on a GPU
.venv/bin/python batch.py ./pdfs --out ./out --device cuda

# see the plan without doing anything
.venv/bin/python batch.py ./pdfs --dry-run
```

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
