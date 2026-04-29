#!/bin/bash
QUARTO_META=${QUARTO_META:-"-M eval:false"}
exec quarto render "$@" $QUARTO_META
