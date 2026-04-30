#!/bin/bash

for md in $(find . -name "*.md"); do
    echo "Removing cell_style and slideshow tags from $md..."
    sed -i 's/```python.*/```python/g' $md
done
