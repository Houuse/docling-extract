"""Turn a converted DoclingDocument into the chunks and facts a RAG store needs.

Extraction alone gives you a document; this gives you the retrievable units.
Prose is split by structure rather than character count, each table is kept
whole *and* decomposed into one record per numeric cell, and every unit carries
the heading trail and page range needed to cite it.

The design decisions behind this are recorded as ADRs 0001, 0002 and 0005 in
the rag-lab repository: store every table twice, chunk prose with
HybridChunker, and make each fact its own retrievable unit.

Nothing here touches a database or a network. The strings these functions
return are hashed to key the embedding cache, so **any change to them
invalidates every cached embedding and every vector already stored**. Treat
the output as a wire format.
"""

import re
from dataclasses import dataclass, field

# Chunk budgets must be measured in the tokens of the model that will embed
# them, so the tokenizer identity belongs to the embedding module and is
# imported rather than duplicated. embedding does not import chunking, so
# there is no cycle.
from docling_extract.embedding import EMBED_MAX_TOKENS, EMBED_MODEL

ITEM_RE = re.compile(r"^\s*item\s+(\d+[A-Z]?)\b[.:]?\s*(.*)", re.I)
# Filings write this many ways: "(Millions)", "(Dollars in millions, except per
# share amount)", "(In thousands)". Match the word, not a bracketed exact form.
SCALE_RE = re.compile(r"\b(millions|thousands|billions)\b", re.I)
PERIOD_RE = re.compile(
    r"(?:year|years|quarter|quarters|period|periods)\s+ended\s+"
    r"([A-Z][a-z]+)\s+(\d{1,2})?",
    re.I,
)
MONTHS = {
    m: i
    for i, m in enumerate(
        "january february march april may june july august september "
        "october november december".split(),
        1,
    )
}

# --------------------------------------------------------------------------
# 1. document metadata
# --------------------------------------------------------------------------

DOC_TYPE_RE = re.compile(r"(10K|10Q|8K|EARNINGS)", re.I)


@dataclass
class DocMeta:
    doc_name: str
    company: str
    doc_type: str
    fiscal_year: int | None
    period_end: str | None = None
    pdf_pages: int | None = None
    source_path: str = ""


def doc_meta(stem: str, doc) -> DocMeta:
    """Parse FinanceBench's naming convention: COMPANY_YEAR[Qn]_TYPE[_extra]."""
    name = stem.split(".")[0]
    parts = name.split("_")
    year = next((int(m.group()) for p in parts if (m := re.fullmatch(r"(19|20)\d{2}", p))), None)
    if year is None:
        m = re.search(r"(19|20)\d{2}", name)
        year = int(m.group()) if m else None
    tm = DOC_TYPE_RE.search(name)
    doc_type = tm.group(1).lower() if tm else "unknown"
    company_parts = []
    for p in parts:
        if re.match(r"^(19|20)\d{2}", p) or DOC_TYPE_RE.fullmatch(p):
            break
        company_parts.append(p)
    return DocMeta(
        doc_name=name,
        company=" ".join(company_parts) or name,
        doc_type=doc_type,
        fiscal_year=year,
        pdf_pages=len(doc.pages) or None,
    )


def period_end(doc, fiscal_year: int | None) -> str | None:
    """Find the statements' own 'Years ended <Month> <day>' line.

    The period identity lives here, not in a table's column header: a column
    headed 2016 is calendar year-end for 3M and May 2016 for Nike.
    """
    if fiscal_year is None:
        return None
    for tx in doc.texts:
        m = PERIOD_RE.search(tx.text or "")
        if not m:
            continue
        month = MONTHS.get(m.group(1).lower())
        if not month:
            continue
        day = int(m.group(2)) if m.group(2) else None
        if day is None:
            # No day given; fall back to the month's last day.
            import calendar

            day = calendar.monthrange(fiscal_year, month)[1]
        return f"{fiscal_year:04d}-{month:02d}-{day:02d}"
    return None


# --------------------------------------------------------------------------
# 2. Item sections — derived, not a Docling field
# --------------------------------------------------------------------------


def item_sections(doc) -> dict[int, str]:
    """page_no -> the Item section in force on that page.

    Docling has no notion of SEC Item structure, so scan text items in reading
    order for 'Item 7'-style headings and carry the last one seen forward.
    """
    current: str | None = None
    out: dict[int, str] = {}
    for tx in doc.texts:
        pg = tx.prov[0].page_no if tx.prov else None
        text = (tx.text or "").strip()
        m = ITEM_RE.match(text)
        # Require it to look like a heading, not a cross-reference buried in a
        # sentence ("as described in Item 7 below").
        if m and len(text) < 120:
            current = f"Item {m.group(1).upper()}"
        if pg is not None and current and pg not in out:
            out[pg] = current
    return out


# --------------------------------------------------------------------------
# 3. facts — one row per numeric table cell
# --------------------------------------------------------------------------

NUMERIC_RE = re.compile(r"^\(?\s*-?\$?\s*([\d,]+(?:\.\d+)?)\s*\)?%?$")


def scrub(text: str | None) -> str:
    """Remove characters PostgreSQL text columns cannot store.

    Some PDFs yield NUL bytes in their text layer, which psycopg rejects
    outright. Applied before embedding as well as before insert, so the vector
    and the stored string are computed from identical text.
    """
    if not text:
        return ""
    return text.replace("\x00", "")

# Row labels that are really header/units rows, not line items.
HEADER_LABEL_RE = re.compile(
    r"^\(?\s*(millions|thousands|billions|dollars|in\s|except\s|amounts\s)", re.I
)


@dataclass
class Fact:
    table_ordinal: int
    row_label: str
    column_label: str
    value_raw: str
    value: float
    parenthesized: bool
    scale: str | None
    unit: str | None
    page: int


def _cell_grid(tbl) -> dict[tuple[int, int], str]:
    return {
        (c.start_row_offset_idx, c.start_col_offset_idx): (c.text or "").strip()
        for c in tbl.data.table_cells
    }


def _header_map(tbl, grid: dict[tuple[int, int], str]) -> dict[int, str]:
    """col index -> column label, from the cells Docling flagged as headers."""
    rows = [c.start_row_offset_idx for c in tbl.data.table_cells if c.column_header]
    hdr_row = min(rows) if rows else 0
    labels: dict[int, str] = {}
    for col in range(tbl.data.num_cols):
        txt = grid.get((hdr_row, col), "")
        if txt:
            labels[col] = txt
    return labels


def table_facts(doc, idx: int) -> list[Fact]:
    tbl = doc.tables[idx]
    page = tbl.prov[0].page_no if tbl.prov else 0
    grid = _cell_grid(tbl)
    headers = _header_map(tbl, grid)

    # Table-level scale: usually the corner cell ("(Millions)", "(Dollars in
    # millions, except per share amount)"), sometimes the header row.
    corner = grid.get((0, 0), "")
    m = SCALE_RE.search(corner) or SCALE_RE.search(" ".join(headers.values()))
    table_scale = m.group(1).lower() if m else None

    # Fallback: text on the page immediately around the table. Restricted to
    # short items so a sentence like "millions of customers" in body prose
    # cannot supply a scale.
    if table_scale is None:
        for tx in doc.texts:
            if tx.prov and tx.prov[0].page_no == page and len(tx.text or "") < 120:
                pm = SCALE_RE.search(tx.text or "")
                if pm:
                    table_scale = pm.group(1).lower()
                    break

    facts: list[Fact] = []
    for (row, col), text in sorted(grid.items()):
        if col == 0 or row == 0 or not text:
            continue  # col 0 is the row label; row 0 is the header
        nm = NUMERIC_RE.match(text)
        if not nm:
            continue  # blanks, "$", footnote markers, prose cells
        label = grid.get((row, 0), "").strip()
        column_label = headers.get(col, "")
        if not label or not column_label:
            continue
        # A row whose label is a units marker is a second header row, not data.
        # Tables with multi-row headers otherwise leak "(Millions) · Capital
        # Expenditures · 2018" as a fact whose value is the year.
        if HEADER_LABEL_RE.match(label):
            continue
        # Row-level scale wins: segment tables put it in the row label
        # ("Sales (millions)") rather than once for the whole table.
        rm = SCALE_RE.search(label)
        scale = rm.group(1).lower() if rm else table_scale

        facts.append(
            Fact(
                table_ordinal=idx,
                row_label=label,
                column_label=column_label,
                value_raw=text,
                value=float(nm.group(1).replace(",", "")),
                # Accounting notation: (1,577) is an outflow. Recorded, not
                # resolved — the schema's generated `signed` column applies it.
                parenthesized=text.strip().startswith("("),
                scale=scale,
                unit="percent" if text.rstrip().endswith("%") else None,
                page=page,
            )
        )
    return facts


# --------------------------------------------------------------------------
# 4. chunks
# --------------------------------------------------------------------------


@dataclass
class Chunk:
    kind: str
    text: str
    item_section: str | None
    heading_trail: list[str] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    table_ordinal: int | None = None


def prose_chunks(doc, sections: dict[int, str]) -> list[Chunk]:
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import (
        HuggingFaceTokenizer,
    )
    from transformers import AutoTokenizer

    # ADR 0003: the token budget must come from the embedding model's own
    # tokenizer. HybridChunker needs Docling's wrapper — handed a raw HF
    # tokenizer it does not take max_tokens from it, and oversized chunks are
    # then silently truncated at embed time with only a warning.
    hf = AutoTokenizer.from_pretrained(EMBED_MODEL, trust_remote_code=True)

    # HybridChunker asks the tokenizer to count a *candidate* merged chunk and
    # merges only if the count fits the budget. Counting an over-budget
    # candidate makes transformers print "Token indices sequence length is
    # longer than the specified maximum sequence length", once per candidate,
    # about text that is then rejected and never emitted. The warning is about
    # sequences passed to the model; these are only measured.
    #
    # It is pure noise, and worse, it is noise shaped exactly like the real
    # hazard — so it trains you to ignore the thing you must not ignore. The
    # budget that actually governs chunking is max_tokens on the wrapper
    # below, unaffected by this. `oversized()` is the honest check.
    hf.model_max_length = int(1e12)

    tok = HuggingFaceTokenizer(tokenizer=hf, max_tokens=EMBED_MAX_TOKENS)
    chunker = HybridChunker(tokenizer=tok)

    out: list[Chunk] = []
    for ch in chunker.chunk(dl_doc=doc):
        pages = [
            p.page_no
            for it in ch.meta.doc_items
            for p in (it.prov or [])
        ]
        if not (ch.text or "").strip():
            continue
        lo = min(pages) if pages else None
        out.append(
            Chunk(
                kind="prose",
                text=ch.text,
                item_section=sections.get(lo) if lo else None,
                heading_trail=list(ch.meta.headings or []),
                page_start=lo,
                page_end=max(pages) if pages else None,
            )
        )
    return out


TRAIL_DEPTH = 3


def table_heading_trails(doc) -> dict[int, list[str]]:
    """table index -> the section headings printed above it, in reading order.

    A segment table's identity is not in its cells: page 33's table says
    "Sales (millions)" and nothing about "Industrial Business", which is a
    heading above it. Without this, five segments' figures are
    indistinguishable.

    Docling reports every section_header at level 1, so there is no hierarchy
    to walk. The last few headers in reading order are used instead: the
    nearest one alone is often useless ("Year 2018 results:").
    """
    ordinal = {id(t): i for i, t in enumerate(doc.tables)}
    recent: list[str] = []
    trails: dict[int, list[str]] = {}
    for item, _ in doc.iterate_items():
        label = str(getattr(item, "label", ""))
        text = (getattr(item, "text", "") or "").strip()
        # Docling's heading classification is inconsistent: "Consumer Business
        # (14.6% of consolidated sales):" comes back as `text` while the four
        # identically-formatted sibling segments come back as `section_header`.
        # Treat a short colon-terminated line as a heading too.
        looks_like_heading = (
            "section_header" in label
            or "title" in label
            or ("text" in label and text.endswith(":") and len(text) < 100)
        )
        if looks_like_heading and text:
            recent.append(text)
            del recent[:-TRAIL_DEPTH]
        elif "table" in label and id(item) in ordinal:
            trails[ordinal[id(item)]] = list(recent)
    return trails


def table_chunks(doc, sections: dict[int, str]) -> list[Chunk]:
    trails = table_heading_trails(doc)
    out: list[Chunk] = []
    for i, tbl in enumerate(doc.tables):
        page = tbl.prov[0].page_no if tbl.prov else None
        md = tbl.export_to_markdown(doc=doc)
        if not md.strip():
            continue
        out.append(
            Chunk(
                kind="table",
                text=md,
                item_section=sections.get(page) if page else None,
                heading_trail=trails.get(i, []),
                page_start=page,
                page_end=page,
                table_ordinal=i,
            )
        )
    return out


# --------------------------------------------------------------------------
# 6. write
# --------------------------------------------------------------------------


def fact_text(company: str, context: str | None, f: Fact) -> str:
    """One fact as a short retrievable string (ADR 0005).

    Everything distinguishing this fact from its neighbours goes in: company,
    which statement it came from, the line item, the period, the value. Inside
    a whole-table chunk the line item is a fortieth of the text; here it is
    most of it, which is what moved capex from rank 10 to rank 1.
    """
    parts = [company]
    if context:
        parts.append(context.rstrip(":"))
    parts += [
        f.row_label,
        f.column_label,
        f"{f.value_raw} {f.scale or ''}".strip(),
    ]
    return " · ".join(p for p in parts if p)


def oversized(texts: list[str]) -> list[tuple[int, int]]:
    """Which of these exceed the embedding window, as (index, token count).

    The check the noisy tokenizer warning does not perform. Prose is budgeted
    by HybridChunker, but `table_chunks` keeps every table whole (ADR 0001) and
    consults no tokenizer, so an unusually large table can produce a chunk
    above the window. SentenceTransformer then truncates it at encode time
    with no error, storing a vector for text that was cut — one of the silent
    failures this pipeline is built to avoid.

    Measured across 366 filings the count was zero, longest chunk 8,173
    tokens. That is a property of this corpus, not a guarantee.
    """
    from transformers import AutoTokenizer

    hf = AutoTokenizer.from_pretrained(EMBED_MODEL, trust_remote_code=True)
    hf.model_max_length = int(1e12)  # counting only; see prose_chunks
    out = []
    for i, t in enumerate(texts):
        n = len(hf(t, add_special_tokens=False, verbose=False)["input_ids"])
        if n > EMBED_MAX_TOKENS:
            out.append((i, n))
    return out
