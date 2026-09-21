#!/usr/bin/env bash
# Script to launch interactive JupyterLab notebook environment in browser
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"

echo "=== Launching JupyterLab for NLP Prosit 1 (Ewe Language Modeling) ==="
echo "Project Directory: $DIR"
echo "Active Virtualenv: $DIR/.venv"
echo "Opening browser..."

"$DIR/.venv/bin/jupyter" lab --notebook-dir="$DIR/notebooks"
