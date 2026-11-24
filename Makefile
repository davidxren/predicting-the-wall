# Run the pipeline end to end: make all. Each target only needs the one before it.

PYTHON ?= python

.PHONY: all install data clean features figures model experiments test

all: figures model experiments

install:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) src/download.py

clean: data
	$(PYTHON) src/validate.py

features: clean
	$(PYTHON) src/features.py

figures: features
	$(PYTHON) src/plots.py


model: features
	$(PYTHON) src/model.py

experiments: features
	$(PYTHON) src/experiments.py
	
test:
	$(PYTHON) -m pytest -q