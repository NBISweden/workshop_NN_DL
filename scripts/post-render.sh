#!/bin/bash
# post-render.sh

for nb in $(find _site -name "*.ipynb"); do
  echo "Fixing cell metadata in $nb..."
  python scripts/fix-ipynb.py "$nb"
done
