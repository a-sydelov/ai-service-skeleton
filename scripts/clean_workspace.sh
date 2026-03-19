#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DRY_RUN=0
VERBOSE=0

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--dry-run] [--verbose]

Removes local project garbage safely:
  - Python cache dirs/files
  - pytest/mypy/ruff caches
  - local .cache contents
  - coverage/build artifacts
  - editor/OS temp files

Does NOT remove:
  - models/
  - .env
  - Docker images/volumes/containers
USAGE
}

log() {
  printf '%s\n' "$*"
}

remove_path() {
  local path="$1"

  if [[ ! -e "$path" ]]; then
    return 0
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "[dry-run] remove $path"
    return 0
  fi

  rm -rf -- "$path"
  log "removed $path"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --verbose)
      VERBOSE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      log "Unknown argument: $1"
      usage
      exit 2
      ;;
  esac
done

log "Cleaning workspace: $ROOT_DIR"

# Directories
while IFS= read -r -d '' path; do
  remove_path "$path"
done < <(
  find . \( \
    -type d \( \
      -name '__pycache__' -o \
      -name '.pytest_cache' -o \
      -name '.mypy_cache' -o \
      -name '.ruff_cache' -o \
      -name '.hypothesis' -o \
      -name '.tox' -o \
      -name '.nox' -o \
      -name 'htmlcov' -o \
      -name 'build' -o \
      -name 'dist' -o \
      -name '.eggs' \
    \) \
  \) -print0
)

# Files
while IFS= read -r -d '' path; do
  remove_path "$path"
done < <(
  find . \( \
    -type f \( \
      -name '*.pyc' -o \
      -name '*.pyo' -o \
      -name '*.pyd' -o \
      -name '*.tmp' -o \
      -name '*.bak' -o \
      -name '.coverage' -o \
      -name 'coverage.xml' -o \
      -name '.DS_Store' \
    \) -o \
    -type f -path './*.egg-info/*' \
  \) -print0
)

# Egg-info directories
while IFS= read -r -d '' path; do
  remove_path "$path"
done < <(find . -type d -name '*.egg-info' -print0)

# Local .cache contents only, keep the directory itself if it exists.
if [[ -d .cache ]]; then
  while IFS= read -r -d '' path; do
    remove_path "$path"
  done < <(find .cache -mindepth 1 -maxdepth 1 -print0)
fi

if [[ "$VERBOSE" -eq 1 ]]; then
  log "Remaining top-level entries:"
  find . -maxdepth 1 | sort
fi

log "Done."
