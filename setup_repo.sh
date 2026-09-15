#!/bin/bash
# Run this once to scaffold the dissertation repo structure locally.
mkdir -p data/raw data/processed notebooks src/ingestion src/features src/models src/eval src/dashboard models tests
touch app.py requirements.txt README.md
echo "Repo structure created:"
find . -type d | sort
