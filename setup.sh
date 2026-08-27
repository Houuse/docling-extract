#!/usr/bin/env bash
# One-shot setup and run. Works in Git Bash on Windows, and on Linux/macOS.
#
#   ./setup.sh                      set up, verify, convert everything
#   ./setup.sh --setup-only         set up and verify, convert nothing
#   ./setup.sh --pdfs ../fb/pdfs    where the PDFs are
#   ./setup.sh --cpu                force CPU
#   ./setup.sh --gpu                require a GPU, fail if unusable
#   ./setup.sh --limit 3            convert only 3 documents (a real test)
#
# Every step that can fail silently is verified before the next one runs.

set -euo pipefail

PDFS="../financebench/pdfs"
OUT="./out"
PYVER="3.12"          # torch CUDA wheels lag new Python releases; 3.14 has none
DEVICE="auto"         # auto | cuda | cpu
WORKERS=1
LIMIT=""
SETUP_ONLY=0
FORCED_GPU=0

while [ $# -gt 0 ]; do
  case "$1" in
    --pdfs)        PDFS="$2"; shift 2 ;;
    --out)         OUT="$2"; shift 2 ;;
    --python)      PYVER="$2"; shift 2 ;;
    --workers)     WORKERS="$2"; shift 2 ;;
    --limit)       LIMIT="$2"; shift 2 ;;
    --cpu)         DEVICE="cpu"; shift ;;
    --gpu)         DEVICE="cuda"; FORCED_GPU=1; shift ;;
    --setup-only)  SETUP_ONLY=1; shift ;;
    -h|--help)     sed -n '2,11p' "$0"; exit 0 ;;
    *)             echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")"

say()  { printf '\n=== %s\n' "$*"; }
die()  { printf '\nFAILED: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. uv
say "1/6  uv"
if ! command -v uv >/dev/null 2>&1; then
  python -m pip install --quiet uv 2>/dev/null \
    || python3 -m pip install --quiet uv \
    || die "could not install uv. Install Python 3.12+ and pip, then rerun."
fi
uv --version

# ------------------------------------------------------- 2. venv, pinned
# uv downloads the requested interpreter if the system does not have it, which
# is the point: the system Python may be too new for torch's CUDA wheels.
say "2/6  virtualenv on Python $PYVER"

# Windows puts the interpreter in Scripts/, everything else in bin/.
find_py() {
  if   [ -x .venv/Scripts/python.exe ]; then echo .venv/Scripts/python.exe
  elif [ -x .venv/Scripts/python ];     then echo .venv/Scripts/python
  elif [ -x .venv/bin/python ];         then echo .venv/bin/python
  fi
}

# Idempotent: this script is meant to be rerun after any interruption, and
# `uv venv` refuses to touch an existing directory. Reuse a venv that is
# already on the right Python; replace one that is not.
PY="$(find_py)"
if [ -n "$PY" ] && [ "$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)" = "$PYVER" ]; then
  echo "reusing existing .venv"
else
  if [ -d .venv ]; then
    echo "existing .venv is not on Python $PYVER — replacing it"
    uv venv --python "$PYVER" --clear --quiet
  else
    uv venv --python "$PYVER" --quiet
  fi
  PY="$(find_py)"
fi

[ -n "$PY" ] || die "no interpreter in .venv — check the uv output above"
echo "interpreter: $PY"
"$PY" --version

ACTUAL="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
[ "$ACTUAL" = "$PYVER" ] || die "venv is on Python $ACTUAL, wanted $PYVER"

# ------------------------------------------------------------ 3. docling
say "3/6  docling"
uv pip install --python "$PY" --quiet -r requirements.txt
"$PY" -c 'import docling, importlib.metadata as m; print("docling", m.version("docling"))'

# --------------------------------------------------------------- 4. torch
# Must come AFTER docling: resolving docling's own torch dependency replaces a
# CUDA wheel with the CPU-only one, and the failure only appears at runtime as
# "Torch not compiled with CUDA enabled".
# In auto mode, decide from what is actually present rather than asking the
# user to know. --gpu forces CUDA and fails loudly if it is unusable; --cpu
# skips all of this.
if [ "$DEVICE" = "auto" ]; then
  say "4/6  detecting GPU"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "no nvidia-smi -> using CPU"
    DEVICE="cpu"
  elif ! nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi present but failing (driver problem?) -> using CPU"
    DEVICE="cpu"
  else
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true
    DEVICE="cuda"
  fi
fi

if [ "$DEVICE" = "cuda" ]; then
  say "4/6  CUDA torch"
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found. Install the NVIDIA driver, or rerun with --cpu."

  CUDA_VER="$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\.[0-9]*\).*/\1/p' | head -1)"
  [ -n "$CUDA_VER" ] || die "could not read a CUDA version from nvidia-smi"
  echo "driver supports CUDA $CUDA_VER"
  echo -n "torch as installed: "
  "$PY" -c 'import torch; print(torch.__version__, "| cuda:", torch.cuda.is_available())'

  # The default PyPI wheel is usually already CUDA-enabled, so do not touch it
  # if it works — forcing a specific index can downgrade a good install. Only
  # intervene when CUDA is actually unavailable.
  if "$PY" -c 'import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    echo "already CUDA-capable, leaving it alone"
  else
    MAJOR="${CUDA_VER%%.*}"; MINOR="${CUDA_VER#*.}"
    if   [ "$MAJOR" -ge 13 ]; then IDX=cu128
    elif [ "$MAJOR" -eq 12 ] && [ "$MINOR" -ge 8 ]; then IDX=cu128
    elif [ "$MAJOR" -eq 12 ] && [ "$MINOR" -ge 4 ]; then IDX=cu124
    elif [ "$MAJOR" -eq 12 ]; then IDX=cu121
    else IDX=cu118
    fi
    echo "CUDA unavailable; reinstalling torch from index $IDX"
    uv pip install --python "$PY" --force-reinstall --quiet \
       --index-url "https://download.pytorch.org/whl/$IDX" torch torchvision \
       || die "no $IDX torch wheel for Python $PYVER. Try --python 3.11, or --cpu."
  fi

  say "5/6  verify GPU"
  if "$PY" - <<'EOF'
import sys, torch
v, ok = torch.__version__, torch.cuda.is_available()
print("torch:", v)
print("cuda available:", ok)
if ok:
    print("device:", torch.cuda.get_device_name(0))
else:
    print("torch cannot see the GPU.", file=sys.stderr)
    if v.endswith("+cpu"):
        print("A CPU-only wheel is installed: this Python version has no CUDA"
              " build. Try --python 3.11.", file=sys.stderr)
    else:
        print(f"torch {v} is built for a newer CUDA than the driver supports."
              " Updating the NVIDIA driver would fix it.", file=sys.stderr)
sys.exit(0 if ok else 1)
EOF
  then
    :
  elif [ "$FORCED_GPU" = "1" ]; then
    die "GPU was requested with --gpu but is not usable (see above)"
  else
    # Auto mode: carry on rather than stopping, but say so unmistakably —
    # a silent CPU fallback is how you lose a day to a nine-hour run.
    printf '\n!! GPU unusable — falling back to CPU. This will be many times slower.\n'
    printf '!! Ctrl-C now if you would rather fix the GPU first.\n\n'
    DEVICE="cpu"
  fi
else
  say "4/6  torch (CPU)"
  "$PY" -c 'import torch; print("torch:", torch.__version__)'
  say "5/6  verify — skipped, running on CPU"
fi

# ---------------------------------------------------------------- 6. run
if [ "$SETUP_ONLY" = "1" ]; then
  say "setup complete"
  echo "convert with:  $PY batch.py \"$PDFS\" --out \"$OUT\" --device $DEVICE"
  exit 0
fi

[ -d "$PDFS" ] || die "no such directory: $PDFS  (pass --pdfs <dir>)"
COUNT="$(find "$PDFS" -maxdepth 1 -name '*.pdf' | wc -l | tr -d ' ')"
[ "$COUNT" -gt 0 ] || die "no PDFs in $PDFS"

say "6/6  converting $COUNT PDFs from $PDFS"
ARGS=(batch.py "$PDFS" --out "$OUT" --device "$DEVICE" --workers "$WORKERS")
# Written as if/then, not `[ -n .. ] && ..` — under `set -e` that form exits the
# script when LIMIT is empty, which is the common case.
if [ -n "$LIMIT" ]; then
  ARGS+=(--limit "$LIMIT")
fi

# Resumable: rerun this script after any interruption and finished documents
# are skipped.
"$PY" "${ARGS[@]}"

say "done"
echo "output:  $OUT"
echo "log:     $OUT/batch.log"
echo
echo "package it:"
echo "  tar --exclude=parts -czf corpus.tar.gz $OUT/"
