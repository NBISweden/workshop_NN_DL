#!/bin/bash

find ./*/* -not -path '*/.*' -name '*md' | xargs -I {} quarto convert {}
find ./*/* -not -path '*/.*' -name '*md' | xargs touch
