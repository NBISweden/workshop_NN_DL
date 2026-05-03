LABS = $(wildcard *.qmd)
NOTEBOOKS = $(LABS:.qmd=.ipynb)

.PHONY: all clean

all: $(NOTEBOOKS)

%.ipynb: %.qmd
	quarto render $< --to ipynb --output $@ --output-dir .

clean:
	rm -f $(NOTEBOOKS)
