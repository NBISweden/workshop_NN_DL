LABS = $(wildcard *.qmd)
NOTEBOOKS = $(LABS:.qmd=.ipynb)

.PHONY: all clean

all: $(NOTEBOOKS)

%.ipynb: %.qmd
	quarto render $< --to ipynb --output $@

clean:
	rm -f $(NOTEBOOKS)
