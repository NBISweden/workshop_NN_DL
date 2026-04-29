#!/bin/bash
QUARTO_META=${QUARTO_META:-""}
exec quarto render "$@" $QUARTO_META
