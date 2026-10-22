# Run the pipeline end to end: make all. Each target only needs the one before it.

PYTHON ?= python

.PHONY: all install data clean features figures test

all: figures

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

test:
	$(PYTHON) -m pytest -q