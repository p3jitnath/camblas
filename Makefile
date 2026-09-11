PYTHON ?= python3

.PHONY: all reference grace test format format-check help
all: reference
reference:
	$(PYTHON) scripts/build.py --target reference --cc "$(CC)"
grace:
	$(PYTHON) scripts/build.py --target grace --cc "$(CC)"
test:
	$(PYTHON) scripts/build.py --target reference --cc "$(CC)" --test
	$(PYTHON) -m unittest discover -s tests -p test_tools.py
format:
	$(PYTHON) scripts/style.py --fix
format-check:
	$(PYTHON) scripts/style.py
help:
	@echo 'make [reference|grace|test] CC=gcc-14'
	@echo 'Framework adapters: python3 scripts/build.py --help'
	@echo 'Framework source builds: python3 scripts/frameworks.py --help'
	@echo 'Three-round comparisons: python3 bench/compare.py --help'
	@echo 'Source style: make [format|format-check] PYTHON=.frameworks/style-env/bin/python'
