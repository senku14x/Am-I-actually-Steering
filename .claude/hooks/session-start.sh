#!/bin/bash
# SessionStart hook for Claude Code on the web.
# Builds the CPU analysis + dev environment (the package's "[dev]" extra) so that
# `pytest` and `ruff` work in a freshly cloned, ephemeral web container.
#
# GPU/annotate extras (torch/transformers/openai) are intentionally NOT installed
# here — those run on a dedicated GPU/API host, not in CI/web sessions (see SPEC.md).
set -euo pipefail

# Only run in remote (Claude Code on the web) environments; no-op locally.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

# Create a project virtualenv once; reuse it on cached/resumed containers (idempotent).
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate

# Install pinned CPU deps + dev tools. `-e` keeps it editable; pip skips work when
# everything is already satisfied, so re-runs are cheap. Noise -> stderr to keep the
# session-context stdout clean.
python -m pip install --upgrade pip 1>&2
pip install -e ".[dev]" 1>&2

# Persist the venv for the rest of the session so `pytest`/`ruff`/`python` resolve to it.
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  {
    echo "export VIRTUAL_ENV=\"$CLAUDE_PROJECT_DIR/.venv\""
    echo "export PATH=\"$CLAUDE_PROJECT_DIR/.venv/bin:\$PATH\""
  } >> "$CLAUDE_ENV_FILE"
fi

echo "session-start: strat-geom [dev] environment ready (.venv)."
