#!/bin/bash
# Always launches the dashboard with THIS project's .venv, regardless of
# whether the venv is activated in the current shell or a global Python/
# streamlit install exists on PATH.
cd "$(dirname "$0")"
exec .venv/bin/streamlit run app.py "$@"
