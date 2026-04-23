LECTURES = CNN_lecture_1_classification CNN_lecture_2_segmentation

HTML = $(LECTURES:=.html)

.PHONY: all clean $(LECTURES)

all: $(HTML)

# Per-lecture phony shortcuts: `make CNN_lecture_1_classification`
$(LECTURES): %: %.html

%.html: %.qmd
	rm -rf $*_files/
	quarto render $< --to nbis-quarto-revealjs

clean:
	rm -f $(HTML)
	rm -rf $(LECTURES:=_files)
