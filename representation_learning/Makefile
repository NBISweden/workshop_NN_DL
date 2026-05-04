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
AE_STAMP  = $(STAMPS)/autoencoder.done
SSP_STAMP = $(STAMPS)/selfsupervised_principle.done
PCA_STAMP = $(STAMPS)/pca.done
SMX_STAMP = $(STAMPS)/softmax_viz.done
TRI_STAMP = $(STAMPS)/triplet_viz.done
RC_STAMP  = $(STAMPS)/representation_collapse.done
TC_STAMP  = $(STAMPS)/triplet_concept.done

.PHONY: all slides media clean dp rl tl ae ssp pca softmax triplet rc triplet-concept

all: slides

slides: $(LECTURE).html

$(LECTURE).html: $(LECTURE).qmd $(DP_STAMP) $(SPL_STAMP) $(RL_STAMP) $(TL_STAMP) $(AE_STAMP) $(SSP_STAMP) $(VIZ_PY)
	rm -rf $(LECTURE)_files/
	quarto render $(LECTURE).qmd --to revealjs

# Render all Manim scenes
media: $(DP_STAMP) $(SPL_STAMP) $(RL_STAMP) $(TL_STAMP) $(AE_STAMP) $(SSP_STAMP) $(PCA_STAMP)

# Individual scene shortcuts
dp: $(DP_STAMP)
spl: $(SPL_STAMP)
rl: $(RL_STAMP)
tl: $(TL_STAMP)
ae: $(AE_STAMP)
ssp: $(SSP_STAMP)
pca: $(PCA_STAMP)
softmax: $(SMX_STAMP)
triplet: $(TRI_STAMP)
rc: $(RC_STAMP)
triplet-concept: $(TC_STAMP)

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

$(AE_STAMP): viz/autoencoder.py | $(STAMPS)
	manim render --format $(FORMAT) viz/autoencoder.py AutoencoderScene
	touch $@

$(SSP_STAMP): viz/selfsupervised_principle.py | $(STAMPS)
	manim render --format $(FORMAT) viz/selfsupervised_principle.py SelfsupervisedPrincipleScene
	touch $@

$(PCA_STAMP): viz/pca.py | $(STAMPS)
	manim render --format $(FORMAT) viz/pca.py PCAScene
	touch $@

$(SMX_STAMP): viz/softmax_viz.py media/softmax_cache.npz | $(STAMPS)
	python -m manim --format $(FORMAT) viz/softmax_viz.py SoftmaxScene
	touch $@

media/softmax_cache.npz: viz/softmax_train.py
	pixi run -e local-amd python viz/softmax_train.py

$(TRI_STAMP): viz/triplet_viz.py media/triplet_cache.npz | $(STAMPS)
	python -m manim --format $(FORMAT) viz/triplet_viz.py TripletScene
	touch $@

media/triplet_cache.npz: viz/triplet_train.py
	pixi run -e local-amd python viz/triplet_train.py

$(RC_STAMP): viz/representation_collapse.py | $(STAMPS)
	python -m manim --format $(FORMAT) viz/representation_collapse.py RepresentationCollapseScene
	touch $@

$(TC_STAMP): viz/triplet_concept.py | $(STAMPS)
	python -m manim --format $(FORMAT) viz/triplet_concept.py TripletConceptScene
	touch $@

clean:
	rm -f $(LECTURE).html
	rm -rf $(LECTURE)_files/ media $(STAMPS)
