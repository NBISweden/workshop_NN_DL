LECTURE  = lecture
FORMAT  ?= mp4      # override: make FORMAT=gif

VIZ_PY = $(wildcard viz/*.py)

# Stamp files track successful renders independently of Manim's output path,
# which may vary based on quality settings configured inside the Python files.
STAMPS    = .stamps
DP_STAMP  = $(STAMPS)/dot_product.done
SPL_STAMP = $(STAMPS)/supervised_learning_labeling.done
RL_STAMP  = $(STAMPS)/representation_learning.done
TL_STAMP  = $(STAMPS)/transfer_learning.done

.PHONY: all slides media clean dp rl tl

all: slides

slides: $(LECTURE).html

$(LECTURE).html: $(LECTURE).qmd $(DP_STAMP) $(SPL_STAMP) $(RL_STAMP) $(TL_STAMP) $(VIZ_PY)
	rm -rf $(LECTURE)_files/
	quarto render $(LECTURE).qmd --to revealjs

# Render all Manim scenes
media: $(DP_STAMP) $(SPL_STAMP) $(RL_STAMP) $(TL_STAMP)

# Individual scene shortcuts
dp: $(DP_STAMP)
spl: $(SPL_STAMP)
rl: $(RL_STAMP)
tl: $(TL_STAMP)

$(STAMPS):
	mkdir -p $(STAMPS)

$(DP_STAMP): viz/dot_product.py | $(STAMPS)
	python -m manim --format $(FORMAT) viz/dot_product.py DotProductScene
	touch $@

$(SPL_STAMP): viz/supervised_learning_labeling.py | $(STAMPS)
	manim render --format $(FORMAT) viz/supervised_learning_labeling.py SupervisedLearningLabelingScene
	touch $@

$(RL_STAMP): viz/representation_learning.py | $(STAMPS)
	manim render --format $(FORMAT) viz/representation_learning.py RepresentationLearningScene
	touch $@

$(TL_STAMP): viz/transfer_learning.py | $(STAMPS)
	manim render --format $(FORMAT) viz/transfer_learning.py TransferLearningScene
	touch $@

clean:
	rm -f $(LECTURE).html
	rm -rf $(LECTURE)_files/ media $(STAMPS)
